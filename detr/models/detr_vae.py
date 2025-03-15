# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from collections import deque
import einops
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone, build_film_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .agent.networks.rgb_modules import BaseEncoder, ResnetEncoder
from .agent.networks.policy_head import DeterministicHead, GMMHead, BeTHead, VQBeTHead, DiffusionHead
from .agent.networks.gpt import GPT, GPTConfig
from .agent.networks.mlp import MLP
from einops import rearrange, repeat, reduce, pack, unpack
from torchvision import transforms as T
import pickle
import detr.models.agent.utils as utils

import numpy as np
import os

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class LinearTo2D(nn.Module):
    def __init__(self, in_features, out_shape):
        super(LinearTo2D, self).__init__()
        self.out_shape = out_shape
        self.linear = nn.Linear(in_features, out_shape[0] * out_shape[1])

    def forward(self, x):
        x = self.linear(x)
        return x.view(x.size(0), x.size(1), self.out_shape[0], self.out_shape[1])

class FoAM(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, image_embedding_dim, is_multi_task,
                 use_film: bool = False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        # 动作预测头
        self.action_head = nn.Linear(hidden_dim, state_dim)  # 假设动作维度是14
        # 图片嵌入预测头
        self.image_embedding_head = LinearTo2D(hidden_dim, (image_embedding_dim[0], image_embedding_dim[1]))
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.multi_task = is_multi_task
        self.use_film = use_film

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if self.multi_task:
            self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for proprio and latent and text embeddings
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        ## if we condition on task emebdding for multi-task
        self.proj_text_emb = nn.Linear(384, hidden_dim) # project text embedding to 512 dim


    def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_emb=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # goal_image_embedding = self.encode_goal_image(goal_image)

        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            # encoder_input不是图像+qpos+cls吗？怎么是action_embed 了 这里在计算latent variable z
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) # False: not a padding
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            goal_image_embedding = []
            # 提取camera的视角特征
            for cam_id in range(image.shape[1]):
                if self.use_film:
                    if cam_id == image.shape[1] - 1:
                        features, pos = self.backbones[cam_id](image[:, cam_id])  # HARDCODED
                    else:
                        features, pos = self.backbones[cam_id](image[:, cam_id], task_emb=task_emb) # HARDCODED
                    # features, pos = self.backbones[0](image[:, cam_id], task_emb=task_emb) # HARDCODED
                else:
                    features, pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                # 此时输出的 goal_image_embedding 融合了文字信息
                if cam_id == image.shape[1] - 1:
                    goal_image_embedding.append(self.input_proj(features))
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            # src 目前是图片数据
            src = torch.cat(all_cam_features, axis=3)
            goal_image_embedding = torch.cat(goal_image_embedding, axis=3)
            goal_image_embedding = goal_image_embedding.flatten(2).permute(2, 0, 1)
            pos = torch.cat(all_cam_pos, axis=3)
            if self.multi_task:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=task_emb)[0]
            else:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=None)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
            # goal_image_embedding = encoder_output[300:600, :]
        # 动作预测
        a_hat = self.action_head(hs)
        # 图片嵌入预测
        image_embedding_hat = self.image_embedding_head(hs)
        # 获取每个预测的最后一组图片嵌入 (1, batch_size,time_step_length, 300, 512) 保留最后一个步长的预测，使其变成(1, batch_size, 300, 512)
        # last_image_embedding = image_embedding_hat[:, -1, :, :]
        last_idx = ((~is_pad).sum(dim=1)).squeeze() - cls_joint_is_pad.size(1)
        # last_image_embedding = image_embedding_hat[:, last_idx -1, :, :]
        # 调整维度为 (num_queries, batch_size, 512)
        # 使用 last_idx 索引 image_embedding_hat
        last_image_embedding_hat = image_embedding_hat[torch.arange(image_embedding_hat.size(0)), last_idx - 1]
        # if torch.equal(image_embedding_hat[1, 84], last_image_embedding[1]):
        #     print('True')
        # 调整维度为 (num_queries, batch_size, 512)
        last_image_embedding_hat = last_image_embedding_hat.permute(1, 0, 2)  # (300, batch_size, 512)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar], last_image_embedding_hat, goal_image_embedding

class FoAM_Infer(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, is_multi_task,
                 use_film: bool = False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        # 动作预测头
        self.action_head = nn.Linear(hidden_dim, state_dim)  # 假设动作维度是14
        # 图片嵌入预测头
        # self.image_embedding_head = LinearTo2D(hidden_dim, (image_embedding_dim[0], image_embedding_dim[1]))
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.multi_task = is_multi_task
        self.use_film = use_film

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if self.multi_task:
            self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for proprio and latent and text embeddings
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        ## if we condition on task emebdding for multi-task
        self.proj_text_emb = nn.Linear(384, hidden_dim) # project text embedding to 512 dim


    def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_emb=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # goal_image_embedding = self.encode_goal_image(goal_image)
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            # encoder_input不是图像+qpos+cls吗？怎么是action_embed 了 这里在计算latent variable z
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) # False: not a padding
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            # 提取camera的视角特征
            for cam_id in range(image.shape[1]):
                if self.use_film:
                    if cam_id == image.shape[1] - 1:
                        features, pos = self.backbones[cam_id](image[:, cam_id])  # HARDCODED
                    else:
                        features, pos = self.backbones[cam_id](image[:, cam_id], task_emb=task_emb) # HARDCODED
                    # features, pos = self.backbones[0](image[:, cam_id], task_emb=task_emb) # HARDCODED
                else:
                    features, pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                # 此时输出的 goal_image_embedding 融合了文字信息
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            # src 目前是图片数据
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            if self.multi_task:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=task_emb)[0]
            else:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=None)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
            # goal_image_embedding = encoder_output[300:600, :]
        # 动作预测
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]

class FoAM_wo_MPH(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, is_multi_task,
                 use_film: bool = False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        # 动作预测头
        self.action_head = nn.Linear(hidden_dim, state_dim)  # 假设动作维度是14
        # 图片嵌入预测头
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.multi_task = is_multi_task
        self.use_film = use_film

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if self.multi_task:
            self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for proprio and latent and text embeddings
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        ## if we condition on task emebdding for multi-task
        self.proj_text_emb = nn.Linear(384, hidden_dim) # project text embedding to 512 dim


    def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_emb=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # goal_image_embedding = self.encode_goal_image(goal_image)

        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            # encoder_input不是图像+qpos+cls吗？怎么是action_embed 了 这里在计算latent variable z
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) # False: not a padding
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            goal_image_embedding = []
            # 提取camera的视角特征
            for cam_id in range(image.shape[1]):
                if self.use_film:
                    if cam_id == image.shape[1] - 1:
                        features, pos = self.backbones[cam_id](image[:, cam_id])  # HARDCODED
                    else:
                        features, pos = self.backbones[cam_id](image[:, cam_id], task_emb=task_emb) # HARDCODED
                    # features, pos = self.backbones[0](image[:, cam_id], task_emb=task_emb) # HARDCODED
                else:
                    features, pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
                # 此时输出的 goal_image_embedding 融合了文字信息
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            # src 目前是图片数据
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            if self.multi_task:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=task_emb)[0]
            else:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=None)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        # 动作预测
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]

class G_img_ACT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(image.shape[1]):
                features, pos = self.backbones[cam_id](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

class MT_ACT(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, is_multi_task,
                 use_film: bool = False):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        # 动作预测头
        self.action_head = nn.Linear(hidden_dim, state_dim)  # 假设动作维度是14
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.multi_task = is_multi_task
        self.use_film = use_film
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        # self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        # self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.encoder_proj = nn.Linear(14, hidden_dim) # project state to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        # self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        if self.multi_task:
            self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for proprio and latent and text embeddings
        else:
            self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        ## if we condition on task emebdding for multi-task
        self.proj_text_emb = nn.Linear(384, hidden_dim) # project text embedding to 512 dim

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_emb=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # goal_image_embedding = self.encode_goal_image(goal_image)

        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_proj(actions) # (bs, seq, hidden_dim)
            # qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            # qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            # encoder_input不是图像+qpos+cls吗？怎么是action_embed 了 这里在计算latent variable z
            encoder_input = torch.cat([cls_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # cls_joint_is_pad = torch.full((bs, 1), False).to(qpos.device) # False: not a padding
            cls_is_pad = torch.full((bs, 1), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
            ## for multi-task embedding
            task_emb = self.proj_text_emb(task_emb) ## project task emb to 512 dim

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            # 提取camera的视角特征
            for cam_id, cam_name in enumerate(self.camera_names):
                if cam_name != 'goal_images':
                    if self.use_film:
                        features, pos = self.backbones[0](image[:, cam_id], task_emb=task_emb)  # HARDCODED
                    else:
                        features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                    features = features[0] # take the last layer feature
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
                else:
                    continue
                # 此时输出的 goal_image_embedding 融合了文字信息
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            # src 目前是图片数据
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            if self.multi_task:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=task_emb)
                hs = hs[0]
            else:
                hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_emb=None)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            if self.multi_task:
                hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight,task_emb=task_emb)[0]
            else:
                hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight,task_emb=None)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

class BAKU(nn.Module):
    def __init__(self, a_hat_shape0, policy_type, policy_head, img_encoder, language_projector, proprio_projector, state_dim, num_queries, camera_names, is_multi_task,
                 use_film: bool = False):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.img_encoder = img_encoder
        self.a_hat_shape0 = a_hat_shape0
        self.language_projector = language_projector
        self.proprio_projector = proprio_projector
        # 动作预测头
        self.multi_task = is_multi_task
        self.use_film = use_film
        self.history_len = 1
        self.language_dim = 384
        self.lang_repr_dim = 512
        self.repr_dim = 512
        self.hidden_dim = 256
        self.temporal_agg = True
        self._policy_type = policy_type
        self._policy_head = policy_head
        self._repr_dim = 512
        self._act_dim = state_dim * self.num_queries  # TODO hardcode  state_dim * cs 根据chunksize修改参数 self.num_queries
        self._num_feat_per_step = 2 # TODO hardcode  _num_feat_per_step与相机数量有关，当视角数为1的时候，_num_feat_per_step=2.为2的时候_num_feat_per_step=3,以此类推

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, self.repr_dim))

        # GPT model
        if policy_type == "gpt":
            self._policy = GPT(
                GPTConfig(
                    block_size=65,
                    input_dim=self.repr_dim,
                    output_dim=self.hidden_dim,
                    n_layer=8,
                    n_head=4,
                    n_embd=self.hidden_dim,
                    dropout=0.1,
                )
            )
        elif policy_type == "mlp":
            self._policy = nn.Sequential(
                nn.Linear(self.repr_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            )

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                self.hidden_dim, self._act_dim, hidden_size=self.hidden_dim, num_layers=2
            )
        elif policy_head == "gmm":
            self._action_head = GMMHead(
                self.hidden_dim, self._act_dim, hidden_size=self.hidden_dim, num_layers=2
            )
        elif policy_head == "bet":
            self._action_head = BeTHead(
                self.hidden_dim,
                self._act_dim,
                hidden_size=self.hidden_dim,
                num_layers=2,
            )
        elif policy_head == "vqbet":
            self._action_head = VQBeTHead(
                self.hidden_dim,
                self._act_dim,
                hidden_size=self.hidden_dim,
            )
        elif policy_head == "diffusion":
            self._action_head = DiffusionHead(
                input_size=self.hidden_dim,
                output_size=self._act_dim,
                obs_horizon=10,  # 3 (dmc - diffusion)
                pred_horizon=10,  # 3 (dmc - diffusion)
                hidden_size=self.hidden_dim,
                num_layers=2,
            )
        self.apply(utils.weight_init)

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, task_emb=None, cluster_centers=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # goal_image_embedding = self.encode_goal_image(goal_image)
        shape = image.shape
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if True:
            features = []
            lang_features = (
                task_emb.float()[:, None].repeat(1, self.history_len, 1)
            )
            lang_features = self.language_projector(lang_features)
            lang_features = einops.rearrange(lang_features, "b t d -> (b t) d")
            lang = lang_features
            image = einops.rearrange(image, "b t c h w -> (b t) c h w")
            pixel = self.img_encoder(image, lang=lang)
            pixel = einops.rearrange(pixel, "(b t) d -> b t d", t=shape[1])
            features.append(pixel)

            # proprioception features
            proprio = einops.rearrange(qpos, "(b t) d -> b t d", t=shape[1])
            proprio = self.proprio_projector(proprio)
            features.append(proprio)

            # concatenate
            # print('batchsize:', pixel.shape[0])
            features = torch.cat(features, dim=-1).view(
                pixel.shape[0], -1, 512
            )
            # prompt
            prompt_features = []
            lang_features = einops.rearrange(
                lang_features, "(b t) d -> b t d", t=shape[1]
            )
            prompt_features.append(lang_features[:, -1:])
            num_prompt_feats = len(prompt_features) if len(prompt_features) > 0 else 0
            # num_prompt_feats = 1
            if num_prompt_feats > 0:
                prompt_features = torch.cat(prompt_features, dim=-1).view(
                    pixel.shape[0], -1, self.repr_dim
                )
            # prepend prompt features
            features = torch.cat([prompt_features, features], dim=1)
            # rearrange action
            # actions = einops.rearrange(
            #     actions, "(b t) d x -> b t d x", t=shape[1]
            # )
            if self.temporal_agg and is_training:
                actions = einops.rearrange(actions, "b t1 t2 d -> b t1 (t2 d)")

            # stddev = utils.schedule(self.stddev_schedule, step)

            stddev = 0.1
            # kwargs = {}
            # _, actor_loss = self.actor(features, num_prompt_feats, stddev, actions, **kwargs)
            B, T, D = features.shape
            if self._policy_type == "gpt":
                # insert action token at each self._num_feat_per_step interval
                prompt = features[:, :num_prompt_feats]
                obs = features[:, num_prompt_feats:]
                obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
                action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
                obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
                obs = torch.cat([prompt, obs], dim=1)
                # get action features
                features = self._policy(obs)
                features = features[:, num_prompt_feats:]
                num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
                features = features[:, num_feat_per_step - 1:: num_feat_per_step]

            # action head
            pred_action = self._action_head(
                features,
                stddev,
                **{"cluster_centers": cluster_centers, "action_seq": actions},
            )
            if actions is None:
                a_hat = pred_action.loc
                a_hat = einops.rearrange(a_hat, "b t1 (t2 d) -> b t1 t2 d", t2=self.a_hat_shape0, d=14)
                return a_hat
            else:
                is_pad = is_pad[:, :self.num_queries]
                loss = self._action_head.loss_fn(
                    is_pad,
                    pred_action,
                    actions,
                    reduction="mean",
                    **{"cluster_centers": cluster_centers},
                )
                # 先检查 loss 是否是一个元组
                if isinstance(loss, tuple):
                    # 如果是元组，取元组的第一个元素
                    loss_value = loss[0]
                else:
                    # 如果不是元组，直接使用 loss
                    loss_value = loss

                # 返回 pred_action 和 loss_value
                return pred_action, loss_value

class RT1(nn.Module):
    def __init__(
        self,
        stats_path,
        obs_shape,
        action_shape,
        hidden_dim,
        stddev_schedule,
        stddev_clip,
        use_tb,
        augment,
        pixel_keys,
        norm,
        history,
        history_len,
        eval_history_len,
        use_language,
        film,
    ):
        super().__init__()
        self.stats_path = stats_path
        self.qpos_shape = action_shape
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.use_tb = use_tb
        self.augment = augment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = norm
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1
        self.use_language = use_language
        self.language_proj_type = "mlp"  # mlp or identity
        self.film = film
        # language
        self.language_fusion = "none" if not self.use_language else "film"
        self.language_dim = 384
        self.lang_repr_dim = 512

        # actor parameters
        self._act_dim = action_shape
        # keys
        self.pixel_keys = pixel_keys

        # number of inputs per time step

        num_feat_per_step = len(self.pixel_keys) * 8
        num_feat_per_step += 1

        # observation params
        # proprio_shape = obs_shape[self.proprio_key]
        # obs_shape = obs_shape[self.pixel_keys[0]]

        # Track model size
        model_size = 0

        # encoder
        self.encoder = ResnetEncoder(
            obs_shape,
            512,
            language_dim=self.lang_repr_dim,
            language_fusion=self.language_fusion,
        )
        model_size += sum(
            p.numel() for p in self.encoder.parameters() if p.requires_grad
        )
        self.repr_dim = 512

        # token learner
        self.token_learner = TokenLearner(dim=128)
        self.image_projector = MLP(128, hidden_channels=[512])

        # language encoder        # projector
        if self.language_proj_type == "mlp":
            self.language_projector = MLP(
                self.language_dim,
                hidden_channels=[self.lang_repr_dim, self.lang_repr_dim],
            )
        else:
            self.language_projector = nn.Identity()
        self.language_projector.apply(utils.weight_init)
        model_size += sum(
            p.numel()
            for p in self.language_projector.parameters()
            if p.requires_grad
        )

        # projector for proprioceptive features
        self.proprio_projector = MLP(
            self.qpos_shape, hidden_channels=[self.repr_dim, self.repr_dim]
        )
        self.proprio_projector.apply(utils.weight_init)
        model_size += sum(
            p.numel()
            for p in self.proprio_projector.parameters()
            if p.requires_grad
        )

        self.actor = RT1Actor(self.stats_path,
            self.repr_dim,
            self._act_dim,
            hidden_dim,
            num_feat_per_step,
        )
        # model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        # print(f"Total number of parameters in the model: {model_size}")
        # data augmentation
        if self.augment:
            self.test_aug = T.Compose([T.ToPILImage(), T.ToTensor()])
        # self.train()
        self.buffer_reset()


    def forward(self, qpos, image, is_pad, actions=None, task_emb=None):

        pass

    def buffer_reset(self):
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[key] = deque(maxlen=self.eval_history_len)
        self.proprio_buffer = deque(maxlen=self.eval_history_len)

    def clear_buffers(self):
        del self.observation_buffer
        if self.obs_type == "pixels" and self.use_proprio:
            del self.proprio_buffer

    def discretize(self, actions, preprocess):
        print("Discretizing actions ...")

        # organize actions into shape (N, A)
        reshaped_actions = []
        for action in actions:
            action = preprocess["actions"](action)
            reshaped_actions.extend(action)
        reshaped_actions = np.array(reshaped_actions)

        self.actor._action_head.discretize(reshaped_actions, self.device)

        print("Discretization complete.")

    def reinit_optimizers(self):
        params = list(self.encoder.parameters())
        self.encoder_opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        if self.use_proprio:
            self.proprio_opt = torch.optim.AdamW(
                self.proprio_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        if self.use_language:
            self.language_opt = torch.optim.AdamW(
                self.language_projector.parameters(), lr=self.lr, weight_decay=1e-4
            )
        params = list(self.actor.parameters())
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=self.lr, weight_decay=1e-4
        )

class TokenLearner(nn.Module):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(self, *, dim, ff_mult=2, num_output_tokens=8):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.Sequential(
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, groups=num_output_tokens),
        )

    def forward(self, x):
        x, ps = pack_one(x, "* c h w")
        x = repeat(x, "b c h w -> b (g c) h w", g=self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, "b g h w -> b 1 g h w")
        x = rearrange(x, "b (g c) h w -> b c g h w", g=self.num_output_tokens)

        x = reduce(x * attn, "b c g h w -> b c g", "mean")
        x = unpack_one(x, ps, "* c n")
        return x

class RT1Actor(nn.Module):
    def __init__(
        self,
        stats_path,
        repr_dim,
        act_dim,
        hidden_dim,
        num_feat_per_step=1,
    ):
        super().__init__()

        self._repr_dim = repr_dim
        self._act_dim = act_dim
        self._num_feat_per_step = num_feat_per_step

        self._action_token = nn.Parameter(torch.randn(1, 1, 1, repr_dim))

        # GPT model
        self._policy = GPT(
            GPTConfig(
                block_size=205,  # 110 libero, 205 xarm
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=8,
                n_head=4,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )

        self._action_head = RT1Head(stats_path,
            hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
        )

        self.apply(utils.weight_init)

    def forward(self, obs, num_prompt_feats, stddev, action=None):
        B, T, D = obs.shape
        prompt = obs[:, :num_prompt_feats]
        obs = obs[:, num_prompt_feats:]
        obs = obs.view(B, -1, self._num_feat_per_step, obs.shape[-1])
        action_token = self._action_token.repeat(B, obs.shape[1], 1, 1)
        obs = torch.cat([obs, action_token], dim=-2).view(B, -1, D)
        obs = torch.cat([prompt, obs], dim=1)

        # get action features
        features = self._policy(obs)
        features = features[:, num_prompt_feats:]
        num_feat_per_step = self._num_feat_per_step + 1  # +1 for action token
        features = features[:, num_feat_per_step - 1 :: num_feat_per_step]

        # action head
        pred_action = self._action_head(
            features,
            stddev,
            **{"action_seq": action},
        )

        if action is None:
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                action,
                reduction="mean",
            )
            return pred_action, loss[0] if isinstance(loss, tuple) else loss

class RT1Head(nn.Module):
    def __init__(
        self,
        # network_kwargs
        stats_path,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=2,
        nbins=256,
    ):
        super().__init__()
        self.output_size = output_size
        self.nbins = nbins

        if num_layers > 0:
            sizes = [input_size] + [hidden_size] * num_layers
            layers = []
            for i in range(num_layers):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            layers += [nn.Linear(sizes[-2], sizes[-1])]
            self.share = nn.Sequential(*layers)
        else:
            self.share = nn.Identity()

        # Bin head
        self.bin_head = nn.Sequential(nn.Linear(hidden_size, output_size * nbins))

        # loss
        self.criterion = nn.CrossEntropyLoss()

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        # initialize action max and min for discretization
        self.action_max, self.action_min = stats["action_max"], stats["action_min"]
        self.action_max = self.action_max.to(device="cuda")
        self.action_min = self.action_min.to(device="cuda")

    def find_closest_cluster(self, actions, cluster_centers) -> torch.Tensor:
        N, T, _ = actions.shape
        actions = einops.rearrange(actions, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (actions[:, None, :] - cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # N K A -> N K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N )
        closest_cluster_center = closest_cluster_center.view(N, T)
        return closest_cluster_center

    def forward(self, x, stddev=None, cluster_centers=None, **kwargs):
        feat = self.share(x)

        # Bin head
        bin_logits = self.bin_head(feat)

        # discretize each action dim
        bin_logits = einops.rearrange(bin_logits, "N T (A K) -> N T A K", K=self.nbins)
        # bin_logits = torch.softmax(bin_logits, dim=-1)

        return self.discrete_to_continuous(bin_logits), bin_logits

    def discretize(self, actions, device):
        actions = torch.tensor(actions)
        self.action_max = torch.max(actions, dim=0)[0].to(device)
        self.action_min = torch.min(actions, dim=0)[0].to(device)

    def discrete_to_continuous(self, action_logits):
        action_logits = torch.argmax(action_logits, dim=-1)
        action_logits = action_logits.float()
        action_logits = (action_logits / (self.nbins - 1)) * (
            self.action_max - self.action_min
        ) + self.action_min
        return action_logits

    def continuous_to_discrete(self, actions):
        actions = (actions - self.action_min) / (self.action_max - self.action_min)
        actions = actions * (self.nbins - 1)
        actions = actions.round()
        return actions

    def loss_fn(self, action, gt_actions, reduction="mean", cluster_centers=None):
        _, action_logits = action

        gt_actions = self.continuous_to_discrete(gt_actions)
        # rearrage for cross entropy loss
        gt_actions = einops.rearrange(gt_actions, "N T A -> (N T) A").long()
        action_logits = einops.rearrange(action_logits, "N T A K -> (N T) K A")

        # loss
        loss = self.criterion(action_logits, gt_actions)

        return {
            "actor_loss": loss,
        }

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]

def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder

def build_foam(args):
    state_dim = 14 # TODO hardcode
    image_embedding_dim = (300, 512)  # 预测图像嵌入维度为300*512
    backbones = []
    use_film = True
    is_train = not args.eval
    if use_film:
        for cam_id, cam_name in enumerate(args.camera_names):
            backbone = build_film_backbone(args)
            print(f'Building {cam_id} build_film_backbone for {cam_name}')
            backbones.append(backbone)
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)
    if args.use_goal_img:
        backbone = build_backbone(args)
        # 冻结use_goal_img的backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbones.append(backbone)
        print(f'Building build_backbone for goal_img')
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    if is_train:
        model = FoAM(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            image_embedding_dim=image_embedding_dim,
            is_multi_task=args.multi_task,
            use_film=use_film
        )
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))
    else:
        model = FoAM_Infer(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
            # image_embedding_dim=image_embedding_dim,
            is_multi_task=args.multi_task,
            use_film=use_film
        )
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model

def build_foam_wo_mph(args):
    state_dim = 14 # TODO hardcode
    backbones = []
    use_film = True
    if use_film:
        for cam_id, cam_name in enumerate(args.camera_names):
            backbone = build_film_backbone(args)
            print(f'Building {cam_id} build_film_backbone for {cam_name}')
            backbones.append(backbone)
    else:
        for _ in args.camera_names:
            backbone = build_backbone(args)
            backbones.append(backbone)
    if args.use_goal_img:
        backbone = build_backbone(args)
        # 冻结use_goal_img的backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbones.append(backbone)
        print(f'Building build_backbone for goal_img')
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = FoAM_wo_MPH(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        is_multi_task=args.multi_task,
        use_film=use_film
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_mt_act(args):
    state_dim = 14 # TODO hardcode
    backbones = []
    use_film = True
    if use_film:
        backbone = build_film_backbone(args)
        backbones.append(backbone)
    else:
        # 提取图片编码 好像是512维
        backbone = build_backbone(args)
        backbones.append(backbone)
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = MT_ACT(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        is_multi_task=args.multi_task,
        use_film=use_film
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    return model

def build_g_img_act(args):
    state_dim = 14 # TODO hardcode
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
    if args.use_goal_img:
        backbone = build_backbone(args)
        # 冻结use_goal_img的backbone
        for param in backbone.parameters():
            param.requires_grad = False
        backbones.append(backbone)
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    model = G_img_ACT(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    return model

def build_baku(args):
    state_dim = 14 # TODO hardcode
    use_film = True
    chunk_size = args.chunk_size
    language_dim = 384
    ImgShape = [3, 480, 640] # TODO hardcode
    ImgEncoder = ResnetEncoder(
        ImgShape,
        512,
        language_dim=512,
        language_fusion='film',
    )
    language_projector = MLP(language_dim, hidden_channels=[512, 512], )
    language_projector.apply(utils.weight_init)

    # 14 是自由度
    proprio_projector = MLP(14, hidden_channels=[512, 512]) # TODO hardcode
    proprio_projector.apply(utils.weight_init)

    policy_type = "gpt"

    policy_head = 'deterministic'

    model = BAKU(
        chunk_size,
        policy_type,
        policy_head,
        ImgEncoder,
        language_projector,
        proprio_projector,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        is_multi_task=args.multi_task,
        use_film=use_film
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))
    return model

def build_rt1(args):
    state_dim = 14 # TODO hardcode
    backbones = []
    use_film = True
    language_dim = 384
    lang_repr_dim = 512
    repr_dim = 512
    stats_path = os.path.join(args.ckpt_dir, f'dataset_stats.pkl')
    num_feat_per_step = len(args.camera_names) * 8
    num_feat_per_step += 1 # add one qpos feature
    img_shape = [3, 480, 640] # TODO hardcode
    stddev_clip = 0.3
    hidden_dim = 512
    use_language = True
    stddev_clip = 0.3
    stddev_schedule = 0.1
    augment = True
    history = True
    history_len = args.history_len
    eval_history_len = args.history_len - 1
    use_tb = True
    norm = False
    # 图像编码器
    # ImgEncoder = ResnetEncoder(
    #     img_shape,
    #     512,
    #     language_dim=lang_repr_dim,
    #     language_fusion='film',
    # )
    # # token_learner
    # token_learner = TokenLearner(dim=128) # 传入RT1初始化
    # # 图片投影
    # image_projector = MLP(128, hidden_channels=[512]) # 传入RT1初始化
    #
    # language_projector = MLP(language_dim, hidden_channels=[lang_repr_dim, lang_repr_dim], )
    #
    #
    # # 14 是自由度
    # proprio_projector = MLP(14, hidden_channels=[repr_dim, repr_dim]) # TODO hardcode

    model = RT1(
        stats_path,
        img_shape,
        state_dim,
        hidden_dim,
        stddev_schedule,
        stddev_clip,
        use_tb,
        augment,
        args.camera_names,
        norm,
        history,
        history_len,
        eval_history_len,
        use_language,
        use_film,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


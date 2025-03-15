import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
from detr.main import build_MT_ACT_model_and_optimizer, build_FoAM_model_and_optimizer, build_G_img_ACT_model_and_optimizer, build_baku_model_and_optimizer, build_FoAM_wo_MPH_model_and_optimizer, build_rt1_model_and_optimizer
import IPython
e = IPython.embed

class FoAM(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_FoAM_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.huber_weight = args_override['huber_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if task_emb is not None:
                a_hat, is_pad_hat, (mu, logvar), last_image_embedding_hat, goal_image_embedding = self.model(qpos, image, env_state, actions, is_pad, task_emb)
            else:
                a_hat, is_pad_hat, (mu, logvar), last_image_embedding_hat, goal_image_embedding = self.model(qpos, image, env_state, actions, is_pad)

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            huber_loss_mean = huber_loss(goal_image_embedding, last_image_embedding_hat)
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['huber_loss'] = huber_loss_mean
            # loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['huber_loss'] * 1
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['huber_loss'] * self.huber_weight

            return loss_dict
        else: # inference time
            if task_emb is not None:
                a_hat, _, (_, _) = self.model(qpos, image, env_state, task_emb=task_emb) # no action, sample from prior
            else:
                a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class FoAM_wo_MPH(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_FoAM_wo_MPH_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']

    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if task_emb is not None:
                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, task_emb)
            else:
                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict
        else: # inference time
            if task_emb is not None:
                a_hat, _, (_, _) = self.model(qpos, image, env_state, task_emb=task_emb) # no action, sample from prior
            else:
                a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class MT_ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_MT_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if task_emb is not None:
                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad, task_emb)
            else:
                a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            if task_emb is not None:
                a_hat, _, (_, _) = self.model(qpos, image, env_state, task_emb=task_emb) # no action, sample from prior
            else:
                a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class G_img_ACT(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_G_img_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class BAKU(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_baku_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if task_emb is not None:
                _, actor_loss = self.model(qpos, image, env_state, actions, is_pad, task_emb)
            else:
                pass
            actor_loss['loss'] = actor_loss['actor_loss']
            return actor_loss
        else: # inference time
            if task_emb is not None:
                a_hat = self.model(qpos, image, env_state, actions, is_pad, task_emb) # no action, sample from prior
            else:
                pass
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class RT1(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_rt1_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None, task_emb=None):
        pass

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

# 定义Huber损失函数
def huber_loss(intermediate_embedding, goal_embedding, delta_threshold=1.0):
    # 交换维度，使得batch_size在第一个维度
    intermediate_embedding = intermediate_embedding.permute(1, 0, 2)  # (2, 300, 512)
    goal_embedding = goal_embedding.permute(1, 0, 2)  # (2, 300, 512)

    # 计算 intermediate_embedding 和 goal_embedding 之间的差值
    delta = intermediate_embedding - goal_embedding
    # 计算差值的绝对值
    abs_delta = torch.abs(delta)
    # 计算平方误差和绝对误差之间的最小值
    quadratic = torch.min(abs_delta, torch.tensor(delta_threshold).to(abs_delta.device))
    # 计算大于阈值部分的误差
    linear = abs_delta - quadratic
    # 计算 Huber 损失
    loss = 0.5 * quadratic**2 + delta_threshold * linear

    # 对每个样本的损失求平均
    batch_loss = loss.mean(dim=[1, 2])  # 对嵌入的维度求平均

    return batch_loss.mean()  # 对批次求平均


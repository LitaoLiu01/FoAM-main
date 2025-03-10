import torch
import numpy as np
import pickle
import argparse
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import json
import os
import einops
from utils import load_data # data functions
from utils import test_OpenCabinet_sample, normalize_qpos_data, denormalize_action_data, post_process_rt1_action_data, normalize_qpos_data_rt1 # robot functions
from constants import CAMERA_NAMES, DT, GetTextEmbeddings
from random_initialize_envs import initialize_envs
from utils import test_sample # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, judge_function # helper functions
from policy import MT_ACTPolicy, DREAM, G_img_ACT, BAKU, DREAM_wo_MPH, RT1
# from detr.models.agent.baku import BCAgent as BAKU
from visualize_episodes import save_videos
from sim_env import BOX_POSE, ENV_POSE
from GetMultiGoalCondition4Inference import get_goal_img_4_task, get_infer_task_emb, get_infer_redundant_task_emb



import IPython
e = IPython.embed

def main(args):
    is_eval = args['eval']
    is_use_redundant_task_emb = args['use_redundant_task_emb']
    if is_eval:
        task_name = args['task_name']
        if args['use_redundant_task_emb']:
            TEXT_EMBEDDINGS = GetTextEmbeddings(is_use_redundant_task_emb)
            task_emb = get_infer_redundant_task_emb(task_name, TEXT_EMBEDDINGS)
        else:
            TEXT_EMBEDDINGS = GetTextEmbeddings(is_use_redundant_task_emb)
            task_emb = get_infer_task_emb(task_name, TEXT_EMBEDDINGS)
        task_emb = np.asarray(task_emb)
        task_emb = torch.from_numpy(task_emb).float().cuda()
        task_emb = task_emb.unsqueeze(0)

    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    dataset_dir = args['dataset_dir']
    dataset_dir = dataset_dir[0]
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    ambiguity_env_name = args['ambiguity_env_name']

    # get task parameters
    is_sim = ambiguity_env_name[:3] == 'Sim'
    if is_sim:
        from constants import SIM_AMBIGUITY_NAMES, SIM_TASK_CONFIGS
        task_config = SIM_AMBIGUITY_NAMES[ambiguity_env_name]
    else:
        print("have no the ambiguity environment")
    # dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'DREAM_wo_MPH' or policy_class == 'MT-ACT' or policy_class == 'G_img_ACT' or policy_class == 'DREAM':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'chunk_size': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'huber_weight': args['huber_weight'],
                         }
    elif policy_class == 'BAKU':
        policy_config = {'lr': 1e-4,
                         'num_queries': args['chunk_size'],
                         'chunk_size': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'camera_names': camera_names,
                         'huber_weight': args['huber_weight'],
                         'history_len': 1

                         }
    elif policy_class == 'RT-1':
        policy_config = {'lr': 1e-4,
                         'num_queries': args['chunk_size'],
                         'chunk_size': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'camera_names': camera_names,
                         'huber_weight': args['huber_weight'],
                         'history_len': 6
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'sample_full_episode': args['sample_full_episode'],
        'use_goal_img': args['use_goal_img'],
        'use_redundant_task_emb': args['use_redundant_task_emb']

    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, task_emb, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, config)
    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_policy(policy_class, policy_config):
    if policy_class == 'DREAM':
        policy = DREAM(policy_config)
    elif policy_class == 'DREAM_wo_MPH':
        policy = DREAM_wo_MPH(policy_config)
    elif policy_class == 'G_img_ACT':
        policy = G_img_ACT(policy_config)
    elif policy_class == 'MT-ACT':
        policy = MT_ACTPolicy(policy_config)
    elif policy_class == 'BAKU':
        policy = BAKU(policy_config)
    elif policy_class == 'RT-1':
        policy = RT1(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'DREAM':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'DREAM_wo_MPH':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'G_img_ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'MT-ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'BAKU':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'RT-1':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def get_imagewithgoalcondition(ts, camera_names, goal_image):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    goal_image = rearrange(goal_image, 'h w c -> c h w')
    curr_images.append(goal_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        if cam_name == 'goal_images':
            continue
        else:
            curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, task_emb, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'head'
    config['using_goal_image_inference'] = False
    config['goal_image_camera'] = 'head'
    config['check_success'] = True
    config['training_on_multi_gpu'] = True

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    checkpoint_path = f'{ckpt_dir}/policy_best.ckpt'
    state_dict = torch.load(checkpoint_path)
    if config['training_on_multi_gpu']:
        updated_state_dict = {}

        for k, v in state_dict.items():
            # 如果键以 'model.module.' 开头，去掉这个前缀
            if k.startswith('model.module.'):
                # 替换 'model.module.' 为 'model.'
                new_key = k.replace('model.module.', 'model.')
                updated_state_dict[new_key] = v
            else:
                updated_state_dict[k] = v
        # 将原始字典更新为修改后的字典
        state_dict = updated_state_dict

    if policy_class == 'DREAM':
        params_to_remove = [
            "model.image_embedding_head.linear.weight",
            "model.image_embedding_head.linear.bias"
        ]
        # 确认加载的对象是一个字典
        if not isinstance(state_dict, dict):
            raise ValueError(f"Expected state_dict to be a dict, but got {type(state_dict)}")
        # 删除指定的参数
        for param in params_to_remove:
            if param in state_dict:
                del state_dict[param]
        loading_status = policy.load_state_dict(state_dict)
        # loading_status = policy.load_state_dict(torch.load(ckpt_path))
    else:
        loading_status = policy.load_state_dict(state_dict)

    print(loading_status)
    policy.cuda()
    policy.eval()
    stats_normalization = None
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_class != 'BAKU' and policy_class != 'RT-1':
        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    elif policy_class == 'BAKU':
        stats_normalization = {
            "actions": {
                "min": 0,
                "max": 4,
            },
            "proprioceptive": {
                "min": 0,
                "max": 4,
            },
        }
        pre_process = None
        post_process = None

    from sim_env import make_sim_env_goal_condition

    env = make_sim_env_goal_condition(task_name, config)
    env_max_reward = env.task.max_reward
    # (task_name, config, init_obj_poses)
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    episode_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        goal_image, hdf5_name = get_goal_img_4_task(task_name, config)
        ts = env.reset()
        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        if onscreen_render:
            image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
            plt_img.set_data(image)
            plt.pause(DT)
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)
                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                if 'BAKU' not in policy_class and 'RT-1' not in policy_class:
                    qpos = pre_process(qpos_numpy)
                elif 'BAKU' in policy_class:
                    qpos = normalize_qpos_data(qpos_numpy, stats_normalization)
                elif 'RT-1' in policy_class:
                    qpos = normalize_qpos_data_rt1(qpos_numpy, stats_normalization)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                if config['use_goal_img']:
                    curr_image = get_imagewithgoalcondition(ts, camera_names, goal_image)
                else:
                    curr_image = get_image(ts, camera_names)
                ### query policy
                ### query policy
                if config['policy_class'] == 'MT-ACT' or config['policy_class'] == 'G_img_ACT' or config['policy_class'] == 'DREAM_wo_MPH':
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, task_emb=task_emb)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # exp_weights = torch.flip(exp_weights, [0])
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                elif config['policy_class'] == 'DREAM':
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, task_emb=task_emb)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1) # bool value
                        # 只取前100个生成的轨迹动作，如果少于100个则取当前已有的所有轨迹
                        # 假设为100
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        max_len = min(100, actions_for_curr_step.shape[0])
                        actions_for_curr_step = actions_for_curr_step[-max_len:]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # 最后加权求和
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                elif config['policy_class'] == 'BAKU':
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, task_emb=task_emb)
                        stats_normalization = {
                            "actions": {
                                "min": 0,
                                "max": 4,
                            },
                            "proprioceptive": {
                                "min": 0,
                                "max": 4,
                            },
                        }
                        all_actions = einops.rearrange(all_actions, "b t1 t2 d -> (b t1) t2 d")
                        all_actions = denormalize_action_data(all_actions, stats_normalization)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # exp_weights = torch.flip(exp_weights, [0])
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = raw_action
                elif config['policy_class'] == 'RT-1':
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image, task_emb=task_emb)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        # exp_weights = torch.flip(exp_weights, [0])
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        # raw_action = all_actions[:, t % query_frequency]
                        raw_action = all_actions.squeeze(0).cpu().numpy()

                        stats_normalization = {
                            "actions": {
                                "min": 0,
                                "max": 4,
                            },
                            "proprioceptive": {
                                "min": 0,
                                "max": 4,
                            },
                        }
                        action = post_process_rt1_action_data(raw_action, stats_normalization)
                        print(action)
                    action = action[-1]
                else:
                    raise NotImplementedError
                ### post-process actions
                target_qpos = action
                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        # if real_robot:
        #     move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
        #     pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)

        episode_reward = judge_function(task_name, rewards, episode_return, episode_highest_reward)
        episode_rewards.append(episode_reward)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n, {episode_reward=}, {env_max_reward=}, Success: {episode_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}_{task_name}.mp4'))

    success_rate = np.mean(np.array(episode_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    # for r in range(env_max_reward+1):
    #     more_or_equal_r = (np.array(highest_rewards) >= r).sum()
    #     more_or_equal_r_rate = more_or_equal_r / num_rollouts
    #     summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = f'result_{task_name}' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(episode_rewards))

    return success_rate, avg_return

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, task_emb = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    task_emb = task_emb.cuda()
    return policy(qpos_data, image_data, action_data, is_pad, task_emb) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info




def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--dataset_dir', nargs='+', help='dataset_dir', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--huber_weight', action='store', type=float, help='Huber_loss Weight', default=0.8)
    parser.add_argument('--run_name', '--run_name', action='store', type=str, help='run name for logs', required=True)
    parser.add_argument('--ambiguity_env_name', action='store', type=str, help='ambiguity_env_name', required=True)
    parser.add_argument('--sample_full_episode', action='store', type=bool, help='sample_full_episode', default=False)
    parser.add_argument('--use_goal_img', action='store_true')
    parser.add_argument('--use_redundant_task_emb', action='store_true')
    parser.add_argument('--real_temporal_agg_range', action='store', type=int, help='real_temporal_agg_range', default=100)


    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # add this for multi_task embedding condition
    parser.add_argument('--multi_task', action='store_true')

    main(vars(parser.parse_args()))

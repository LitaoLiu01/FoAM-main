### env utils
import numpy as np
import torch
import os
import h5py
import random
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
from constants import TEXT_EMBEDDINGS, CAMERA_NAMES
from pathlib import Path

e = IPython.embed

class EpisodicDatasetDream(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, norm_stats, num_episodes, config):
        super(EpisodicDatasetDream).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.norm_stats = norm_stats
        self.num_episodes = num_episodes
        self.is_sim = None
        self.h5s = []
        self.trials = []
        self.task_emb_per_trial = []
        self.verbose = True
        self.h5s = {}
        self.config = config
        lens = []

        path = Path(dataset_dir)
        files = []
        subfolders = list_subfolders(path)
        for subfolder in subfolders:
            subfolder_path = os.path.join(path, subfolder)
            sub_subfolders = list_subfolders(subfolder_path)
            for sub_subfolder in sub_subfolders:
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
                OneTaskFiles = list_all_hdf5_files_with_path(sub_subfolder_path)
                files.extend(OneTaskFiles)
        files = list(files)
        files = sorted(files)
        self.trial_names = files
        self._history_len = 1
        self.aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
        self.stats_normalization = {
            "actions": {
                "min": 0,
                "max": 3.14,
            },
            "proprioceptive": {
                "min": 0,
                "max": 3.14,
            },
        }
        # files = sorted(glob.glob(os.path.join(dataset_dir + "*/*/", '*.h5')))
        for filename in files:
            if config['use_redundant_task_emb']:
                task_emb = get_train_redundant_task_emb(filename)
            else:
                task_emb = get_train_task_emb(filename)

            self.task_emb_per_trial.append(task_emb)
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, idx):
        sample_full_episode = False  # hardcode
        config = self.config
        if config['sample_full_episode']:
            sample_full_episode = True
            print('#################################################################################################################################################')
            print('Open sample_full_episode!')
        # episode_ids是打乱之后得到的<0.8 * 总数>的打乱后的数，idx是每轮可变的索引数。则trial_idx是全部数据中的索引数
        trial_idx = self.episode_ids[idx]
        # trial_names 是全部数据按照sorted函数排列后的数据文件名字的索引
        trial_name = self.trial_names[trial_idx]
        task_emb = self.task_emb_per_trial[trial_idx]
        camera_names = CAMERA_NAMES
        dataset_path = os.path.join(self.dataset_dir, f'{trial_name}')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                if config['policy_class'] != 'RT-1' and config['policy_class'] != 'BAKU':
                    start_ts = np.random.choice(episode_len)
                else:
                    start_ts = np.random.choice(episode_len - config['policy_config']['history_len'])
            # get observation at start_ts only
            image_dict = dict()
            for cam_name in camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                # 提取头部视角的goal image
                if config['use_goal_img'] and cam_name == 'head':
                    goal_img = root[f'/observations/images/{cam_name}'][-1]
            # new axis for different cameras
            if config['use_goal_img']:
                image_dict['goal_img'] = goal_img
            all_cam_images = []
            for cam_name in image_dict:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            # get all actions after and including start_ts
            if config['policy_class'] != 'BAKU' and config['policy_class'] != 'RT-1':
                qpos = root['/observations/qpos'][start_ts]
                if is_sim:
                    action = root['/action'][start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = root['/action'][max(0, start_ts):]  # hack, to make timesteps more aligned
                    action_len = episode_len - max(0, start_ts)  # hack, to make timesteps more aligned
                self.is_sim = is_sim
                padded_action = np.zeros(original_action_shape, dtype=np.float32)
                padded_action[:action_len] = action
                is_pad = np.zeros(episode_len)
                is_pad[action_len:] = 1
                # construct observations
                image_data = torch.from_numpy(all_cam_images)
                qpos_data = torch.from_numpy(qpos).float()
                action_data = torch.from_numpy(padded_action).float()
                is_pad = torch.from_numpy(is_pad).bool()
                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)
                # normalize image and change dtype to float
                image_data = image_data / 255.0
                task_emb = torch.from_numpy(np.asarray(task_emb)).float()

                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
                qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

            elif config['policy_class'] == 'BAKU':
                qpos = root['/observations/qpos'][start_ts]
                qpos_data = torch.from_numpy(qpos).float()
                action = root['/action']
                # arrange sampled action to be of shape (history_len, num_queries, action_dim)
                action_data = np.zeros(
                    (self._history_len, config['policy_config']['chunk_size'], 14) # TODO HardCode actions.shape[-1] self._history_len = 1
                )
                num_actions = (
                        self._history_len + config['policy_config']['chunk_size'] - 1
                )  # -1 since its num_queries including the last action of the history
                act = np.zeros((num_actions, 14))  # (10, 7)
                if start_ts + num_actions > episode_len:
                    start_ts = episode_len - num_actions
                act[: min(episode_len, start_ts + num_actions) - start_ts] = action[start_ts: start_ts + num_actions]
                # sample_idx 随机值，边界处理，当sample_idx + num_actions>len(actions)，就会取少于10的数作为随机采样值
                action_data = np.lib.stride_tricks.sliding_window_view(
                    act, (config['policy_config']['chunk_size'], 14)
                )
                action_data = action_data[:, 0]
                action_data = torch.from_numpy(action_data).float()
                # construct observations
                image_data = torch.from_numpy(all_cam_images)
                # action_data = (action_data - self.stats["actions"]["min"]) / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5)
                # 对第 0-5 列和第 7-12 列进行归一化处理
                # 对 action_data 进行归一化处理
                # 对 action_data 进行归一化处理
                action_data[:, :, 0:6] = (action_data[:, :, 0:6] - self.stats_normalization["actions"]["min"]) / (
                        self.stats_normalization["actions"]["max"] - self.stats_normalization["actions"]["min"] + 1e-5)

                action_data[:, :, 7:13] = (action_data[:, :, 7:13] - self.stats_normalization["actions"]["min"]) / (
                        self.stats_normalization["actions"]["max"] - self.stats_normalization["actions"]["min"] + 1e-5)

                action_data[:, :, 6] = action_data[:, :, 6] / 255.0
                action_data[:, :, 13] = action_data[:, :, 13] / 255.0

                # 对 qpos_data 进行同样的归一化处理
                qpos_data[0:6] = (qpos_data[0:6] - self.stats_normalization["proprioceptive"]["min"]) / (
                            self.stats_normalization["proprioceptive"]["max"] - self.stats_normalization["proprioceptive"]["min"] + 1e-5)
                qpos_data[7:13] = (qpos_data[7:13] - self.stats_normalization["proprioceptive"]["min"]) / (
                            self.stats_normalization["proprioceptive"]["max"] - self.stats_normalization["proprioceptive"]["min"] + 1e-5)
                qpos_data[6] = qpos_data[6] / 255.0
                qpos_data[13] = qpos_data[13] / 255.0

                # channel last
                image_data = torch.einsum('k h w c -> k c h w', image_data)
                # normalize image and change dtype to float
                image_data = image_data / 255.0
                task_emb = torch.from_numpy(np.asarray(task_emb)).float()
                is_pad = np.zeros(episode_len)
                is_pad = torch.from_numpy(is_pad).bool()
            elif config['policy_class'] == 'RT-1':
                image_dict = dict()
                for cam_name in camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts : start_ts + config['policy_config']['history_len']]
                all_cam_images = []

                for cam_name in image_dict:
                    all_cam_images.append(image_dict[cam_name])
                all_cam_images = all_cam_images[0]
                all_cam_images = torch.stack([self.aug(all_cam_images[i]) for i in range(len(all_cam_images))])
                image_data = all_cam_images
                qpos = root['/observations/qpos'][start_ts : start_ts + config['policy_config']['history_len']]
                qpos_data = torch.from_numpy(qpos).float()


                action = root['/action']
                action_data = action[start_ts : start_ts + config['policy_config']['history_len']]
                action_data = torch.from_numpy(action_data).float()
                # 对 action_data 进行归一化处理
                action_data[:, 0:6] = (action_data[:, 0:6] - self.stats_normalization["actions"]["min"]) / (
                        self.stats_normalization["actions"]["max"] - self.stats_normalization["actions"]["min"] + 1e-5)

                action_data[:, 7:13] = (action_data[:, 7:13] - self.stats_normalization["actions"]["min"]) / (
                        self.stats_normalization["actions"]["max"] - self.stats_normalization["actions"]["min"] + 1e-5)

                action_data[:, 6] = action_data[:, 6] / 255.0
                action_data[:, 6] = torch.where(action_data[:, 6] < 0.4, torch.tensor(-1.0), torch.tensor(1.0))
                action_data[:, 13] = action_data[:, 13] / 255.0
                action_data[:, 13] = torch.where(action_data[:, 13] < 0.4, torch.tensor(-1.0), torch.tensor(1.0))

                # 对 qpos_data 进行同样的归一化处理
                qpos_data[:, 0:6] = (qpos_data[:, 0:6] - self.stats_normalization["proprioceptive"]["min"]) / (
                            self.stats_normalization["proprioceptive"]["max"] - self.stats_normalization["proprioceptive"]["min"] + 1e-5)
                qpos_data[:, 7:13] = (qpos_data[:, 7:13] - self.stats_normalization["proprioceptive"]["min"]) / (
                            self.stats_normalization["proprioceptive"]["max"] - self.stats_normalization["proprioceptive"]["min"] + 1e-5)
                qpos_data[:, 6] = qpos_data[:, 6] / 255.0
                qpos_data[:, 6] = torch.where(qpos_data[:, 6] < 0.4, torch.tensor(-1.0), torch.tensor(1.0))
                qpos_data[:, 13] = qpos_data[:, 13] / 255.0
                qpos_data[:, 13] = torch.where(qpos_data[:, 13] < 0.4, torch.tensor(-1.0), torch.tensor(1.0))


                # channel last
                # image_data = torch.einsum('k h w c -> k c h w', image_data)
                # normalize image and change dtype to float
                image_data = image_data / 255.0
                task_emb = torch.from_numpy(np.asarray(task_emb)).float()
                is_pad = np.zeros(episode_len)
                is_pad = torch.from_numpy(is_pad).bool()


        return image_data, qpos_data, action_data, is_pad, task_emb

def get_norm_stats_dream(dataset_dir, num_episodes):
    path = Path(dataset_dir)
    files = []
    subfolders = list_subfolders(path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        sub_subfolders = list_subfolders(subfolder_path)
        for sub_subfolder in sub_subfolders:
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
            Hdf5Files = list_all_hdf5_files(sub_subfolder_path)
            files.extend(Hdf5Files)

    files = list(files)
    files = sorted(files)

    if len(files) == num_episodes:
        print('Data check pass!')
    else:
        print('Data check nonpass!')
        exit()
    all_qpos_data = []
    all_action_data = []

    subfolders = list_subfolders(path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(path, subfolder)
        sub_subfolders = list_subfolders(subfolder_path)
        for sub_subfolder in sub_subfolders:
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
            Hdf5Files = list_all_hdf5_files(sub_subfolder_path)
            for Hdf5FileName in Hdf5Files:
                Hdf5File_path = os.path.join(sub_subfolder_path, Hdf5FileName)
                with h5py.File(Hdf5File_path, 'r') as root:
                    qpos = root['/observations/qpos'][()]
                    qvel = root['/observations/qvel'][()]
                    action = root['/action'][()]
                all_qpos_data.append(torch.from_numpy(qpos))
                all_action_data.append(torch.from_numpy(action))

    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, 10)  # clipping
    action_max = all_action_data.view(-1, 14).max(dim=0, keepdim=True).values
    action_max = action_max.squeeze()
    action_min = all_action_data.view(-1, 14).min(dim=0, keepdim=True).values
    action_min = action_min.squeeze()
    action_max[6] = 1
    action_max[13] = 1
    action_min[6] = -1
    action_min[13] = -1

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, 10)  # clipping
    qpos_max = all_qpos_data.view(-1, 14).max(dim=0, keepdim=True).values
    qpos_max = qpos_max.squeeze()
    qpos_min = all_qpos_data.view(-1, 14).min(dim=0, keepdim=True).values
    qpos_min = qpos_min.squeeze()
    qpos_max[6] = 1
    qpos_max[13] = 1
    qpos_min[6] = -1
    qpos_min[13] = -1



    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos, "action_max": action_max, "action_min": action_min, "qpos_max": qpos_max, "qpos_min": qpos_min}

    print('get stats successfully!')
    return stats



def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, config):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats_dream(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDatasetDream(train_indices, dataset_dir, norm_stats, num_episodes, config)
    val_dataset = EpisodicDatasetDream(val_indices, dataset_dir, norm_stats, num_episodes, config)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True,
                                  num_workers=8, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=8,
                                prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def sample_box_pose():
    x_range = [0.4, 0.45]
    y_range = [-0.40, -0.3]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_cabinet_pose():
    x_range = [0.73, 0.83]
    # x_range = [0.73, 0.83]
    y_range = [-0.45, -0.35]
    # y_range = [-0.45, -0.35]
    z_range = [0.32, 0.32]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])


def sample_mark_pose():
    # 定义九个不同的坐标，每个坐标对应一个从1到9的编号
    x_range = [0.4, 0.45]
    y_range = [-0.26, -0.18]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    selected_box_coordinate = np.random.uniform(ranges[:, 0], ranges[:, 1])

    mark_quat = np.array([1, 0, 0, 0])
    return np.concatenate([selected_box_coordinate, mark_quat])



def test_sample():

    x_range = [0.4, 0.45]
    y_range = [-0.4, -0.28]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])

    # 定义九个不同的坐标，每个坐标对应一个从1到9的编号
    x_range = [0.4, 0.45]
    y_range = [-0.26, -0.18]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    selected_box_coordinate = np.random.uniform(ranges[:, 0], ranges[:, 1])
    mark_quat = np.array([1, 0, 0, 0])

    return np.concatenate([cube_position, cube_quat, selected_box_coordinate, mark_quat])

def test_OpenCabinet_sample():
    x_range = [0.73, 0.83]
    # x_range = [0.73, 0.83]
    y_range = [-0.45, -0.35]
    # y_range = [-0.45, -0.35]
    z_range = [0.32, 0.32]
    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])


### helper functions
def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def list_subfolders(directory):
    subfolders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    return subfolders

def list_all_hdf5_files(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(file)
    return hdf5_files

def list_all_hdf5_files_with_path(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                full_path = os.path.join(root, file)
                hdf5_files.append(full_path)
    return hdf5_files


def normalize_qpos_data(qpos_data, stats_normalization):
    """
    对 qpos_data 进行归一化处理。

    参数:
    qpos_data: 需要归一化的数据。
    stats_normalization: 包含 "proprioceptive" 数据的最小值和最大值。

    返回:
    归一化后的 qpos_data。
    """
    # 对 qpos_data 的前6个和第7到12个元素进行归一化处理
    qpos_data[0:6] = (qpos_data[0:6] - stats_normalization["proprioceptive"]["min"]) / (
            stats_normalization["proprioceptive"]["max"] - stats_normalization["proprioceptive"]["min"] + 1e-5)
    qpos_data[7:13] = (qpos_data[7:13] - stats_normalization["proprioceptive"]["min"]) / (
            stats_normalization["proprioceptive"]["max"] - stats_normalization["proprioceptive"]["min"] + 1e-5)

    # 对第7和第13个元素进行255的归一化
    qpos_data[6] = qpos_data[6] / 255.0
    # qpos_data[6] = torch.where(qpos_data[6] < 0.4, torch.tensor(-1.0), torch.tensor(1.0))

    qpos_data[13] = qpos_data[13] / 255.0
    # qpos_data[13] = torch.where(qpos_data[13] < 0.4, torch.tensor(-1.0), torch.tensor(1.0))


    return qpos_data

def normalize_qpos_data_rt1(qpos_data, stats_normalization):
    # 对 qpos_data 的前6个和第7到12个元素进行归一化处理
    qpos_data[0:6] = (qpos_data[0:6] - stats_normalization["proprioceptive"]["min"]) / (
            stats_normalization["proprioceptive"]["max"] - stats_normalization["proprioceptive"]["min"] + 1e-5)
    qpos_data[7:13] = (qpos_data[7:13] - stats_normalization["proprioceptive"]["min"]) / (
            stats_normalization["proprioceptive"]["max"] - stats_normalization["proprioceptive"]["min"] + 1e-5)

    # 对第7和第13个元素进行255的归一化
    qpos_data[6] = qpos_data[6] / 255.0
    qpos_data[6] = np.where(qpos_data[6] < 0.4, -1.0, 1.0)

    qpos_data[13] = qpos_data[13] / 255.0
    qpos_data[13] = np.where(qpos_data[13] < 0.4, -1.0, 1.0)


    return qpos_data

def denormalize_action_data(a_hat, stats_normalization):
    # 对 a_hat 的前6个和第7到12个元素进行逆归一化处理
    a_hat[:, :, 0:6] = a_hat[:, :, 0:6] * (
                stats_normalization["actions"]["max"] - stats_normalization["actions"]["min"] + 1e-5) + \
                       stats_normalization["actions"]["min"]
    a_hat[:, :, 7:13] = a_hat[:, :, 7:13] * (
                stats_normalization["actions"]["max"] - stats_normalization["actions"]["min"] + 1e-5) + \
                        stats_normalization["actions"]["min"]

    # 对第7和第13个元素进行逆归一化
    a_hat[:, :, 6] = a_hat[:, :, 6] * 255.0
    a_hat[:, :, 13] = a_hat[:, :, 13] * 255.0

    return a_hat

def post_process_rt1_action_data(a_hat, stats_normalization):
    # 对 a_hat 的前6个和第7到12个元素进行逆归一化处理
    a_hat[:, 0:6] = a_hat[:, 0:6] * (
                stats_normalization["actions"]["max"] - stats_normalization["actions"]["min"] + 1e-5) + \
                       stats_normalization["actions"]["min"]
    a_hat[:, 7:13] = a_hat[:, 7:13] * (
                stats_normalization["actions"]["max"] - stats_normalization["actions"]["min"] + 1e-5) + \
                        stats_normalization["actions"]["min"]

    # 对第7和第13个元素进行逆归一化
    a_hat[:, 6] = np.where(a_hat[:, 6] == -1.0, 0.0, 255.0)
    a_hat[:, 13] = np.where(a_hat[:, 13] == -1.0, 0.0, 255.0)

    return a_hat

def judge_function(task_name, rewards, episode_return, episode_highest_reward):
    if 'sim_pick' in task_name and 'block' in task_name: # 过滤出pick block任务
        episode_reward = rewards[-1]
    elif 'sim_close_cabinet' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_left_arm_put' in task_name and 'to_cabinet' in task_name:
        episode_reward = np.max(rewards)
    elif 'sim_right_arm_put' in task_name and 'to_cabinet' in task_name:
        episode_reward = np.max(rewards)
    elif 'sim_dual_arm_put' in task_name and 'to_cabinet' in task_name:
        episode_reward = np.max(rewards)
    elif 'sim_open_cabinet' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_put' in task_name and 'basket' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_put' in task_name and 'locker_top_layer' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_put' in task_name and 'pan' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_put' in task_name and 'locker_middle_layer' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_left_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_right_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        episode_reward = rewards[-1]
    elif 'sim_transfer' in task_name and 'block' in task_name:
        episode_reward = rewards[-1]
    else:
        raise NotImplementedError

    return episode_reward

def get_train_task_emb(filename):
    # for 20 tasks hardcoded, modify as needed
    if 'PickLeftBlueBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[0]
    elif 'PickLeftGreenBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[1]
    elif 'PickLeftRedBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[2]
    elif 'PickLeftYellowBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[3]
    elif 'PickRightBlueBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[4]
    elif 'PickRightGreenBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[5]
    elif 'PickRightRedBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[6]
    elif 'PickRightYellowBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[7]
    elif 'CloseBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[8]
    elif 'CloseMiddleDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[9]
    elif 'CloseTopDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[10]
    elif 'LeftArmPutStuff2Drawer/PutAppleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'LeftArmPutStuff2Drawer/PutBananaToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'LeftArmPutStuff2Drawer/PutBlueBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'LeftArmPutStuff2Drawer/PutGreenBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[14]
    elif 'RightArmPutStuff2Drawer/PutAppleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'RightArmPutStuff2Drawer/PutBananaToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'RightArmPutStuff2Drawer/PutBlueBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'RightArmPutStuff2Drawer/PutGreenBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[14]
    elif 'DualArmPutStuff2Drawer/PutAppleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'DualArmPutStuff2Drawer/PutBananaToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'DualArmPutStuff2Drawer/PutBlueBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'DualArmPutStuff2Drawer/PutGreenBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[14]
    elif 'OpenBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[15]
    elif 'OpenMiddleDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[16]
    elif 'OpenTopDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[17]
    elif 'PutBlackStaplerToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[18]
    elif 'PutCameraToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[19]
    elif 'PutGreenStaplerToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[20]
    elif 'PutHammerToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[21]
    elif 'PutBlackStaplerToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[22]
    elif 'PutCameraToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[23]
    elif 'PutGreenStaplerToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[24]
    elif 'PutHammerToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[25]
    elif 'PutAppleToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[26]
    elif 'PutDuckToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[27]
    elif 'PutPigToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[28]
    elif 'PutTeapotToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[29]
    elif 'PutBlackStaplerToLockerMiddleLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[30]
    elif 'PutBlueBlockToLockerMiddleLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[31]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[32]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[33]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[34]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[35]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[36]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[37]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[38]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[39]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[40]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[41]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[42]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[43]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[44]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[45]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[46]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[47]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[32]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[33]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[34]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[35]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[36]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[37]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[38]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[39]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[40]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[41]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[42]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[43]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[44]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[45]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[46]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[47]
    elif 'TransferLeftBlueBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[48]
    elif 'TransferLeftGreenBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[49]
    elif 'TransferLeftRedBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[50]
    elif 'TransferLeftYellowBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[51]
    elif 'TransferRightBlueBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[52]
    elif 'TransferRightGreenBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[53]
    elif 'TransferRightRedBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[54]
    elif 'TransferRightYellowBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[55]
    else:
        'SINGLE TASK embedding wont be used'
        print('There is no the corrspondding task embedding')
        exit()

    return task_emb


def get_train_redundant_task_emb(filename):
    if 'PickLeftBlueBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[0]
    elif 'PickLeftGreenBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[1]
    elif 'PickLeftRedBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[2]
    elif 'PickLeftYellowBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[3]
    elif 'PickRightBlueBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[4]
    elif 'PickRightGreenBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[5]
    elif 'PickRightRedBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[6]
    elif 'PickRightYellowBlock' in filename:
        task_emb = TEXT_EMBEDDINGS[7]
    elif 'CloseBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[8]
    elif 'CloseMiddleDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[9]
    elif 'CloseTopDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[10]
    elif 'LeftArmPutStuff2Drawer/PutAppleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'LeftArmPutStuff2Drawer/PutBananaToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'LeftArmPutStuff2Drawer/PutBlueBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'LeftArmPutStuff2Drawer/PutGreenBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[14]
    elif 'RightArmPutStuff2Drawer/PutAppleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[15]
    elif 'RightArmPutStuff2Drawer/PutBananaToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[16]
    elif 'RightArmPutStuff2Drawer/PutBlueBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[17]
    elif 'RightArmPutStuff2Drawer/PutGreenBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[18]
    elif 'DualArmPutStuff2Drawer/PutAppleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[19]
    elif 'DualArmPutStuff2Drawer/PutBananaToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[20]
    elif 'DualArmPutStuff2Drawer/PutBlueBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[21]
    elif 'DualArmPutStuff2Drawer/PutGreenBottleToBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[22]
    elif 'OpenBottomDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[23]
    elif 'OpenMiddleDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[24]
    elif 'OpenTopDrawer' in filename:
        task_emb = TEXT_EMBEDDINGS[25]
    elif 'PutBlackStaplerToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[26]
    elif 'PutCameraToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[27]
    elif 'PutGreenStaplerToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[28]
    elif 'PutHammerToBasket' in filename:
        task_emb = TEXT_EMBEDDINGS[29]
    elif 'PutBlackStaplerToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[30]
    elif 'PutCameraToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[31]
    elif 'PutGreenStaplerToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[32]
    elif 'PutHammerToLockerTopLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[33]
    elif 'PutAppleToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[34]
    elif 'PutDuckToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[35]
    elif 'PutPigToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[36]
    elif 'PutTeapotToPan' in filename:
        task_emb = TEXT_EMBEDDINGS[37]
    elif 'PutBlackStaplerToLockerMiddleLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[38]
    elif 'PutBlueBlockToLockerMiddleLayer' in filename:
        task_emb = TEXT_EMBEDDINGS[39]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[40]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[41]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[42]
    elif 'LeftArmBlockCoverSmallerMarker/BlueBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[43]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[44]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[45]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[46]
    elif 'LeftArmBlockCoverSmallerMarker/GreenBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[47]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[48]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[49]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[50]
    elif 'LeftArmBlockCoverSmallerMarker/RedBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[51]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[52]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[53]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[54]
    elif 'LeftArmBlockCoverSmallerMarker/YellowBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[55]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[56]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[57]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[58]
    elif 'RightArmBlockCoverSmallerMarker/BlueBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[59]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[60]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[61]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[62]
    elif 'RightArmBlockCoverSmallerMarker/GreenBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[63]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[64]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[65]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[66]
    elif 'RightArmBlockCoverSmallerMarker/RedBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[67]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverBottomLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[68]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverBottomRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[69]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverUpperLeftMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[70]
    elif 'RightArmBlockCoverSmallerMarker/YellowBoxCoverUpperRightMarker' in filename:
        task_emb = TEXT_EMBEDDINGS[71]
    elif 'TransferLeftBlueBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[72]
    elif 'TransferLeftGreenBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[73]
    elif 'TransferLeftRedBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[74]
    elif 'TransferLeftYellowBlockToRight' in filename:
        task_emb = TEXT_EMBEDDINGS[75]
    elif 'TransferRightBlueBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[76]
    elif 'TransferRightGreenBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[77]
    elif 'TransferRightRedBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[78]
    elif 'TransferRightYellowBlockToLeft' in filename:
        task_emb = TEXT_EMBEDDINGS[79]
    else:
        'SINGLE TASK embedding wont be used'
        print('There is no the corrspondding task embedding')
        exit()

    return task_emb

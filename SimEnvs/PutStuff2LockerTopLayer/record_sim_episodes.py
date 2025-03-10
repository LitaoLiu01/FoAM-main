import time
import os
import numpy as np
import argparse
import matplotlib
# matplotlib.use('TkAgg')  # 设置后端为TkAgg或其他适合的后端
import matplotlib.pyplot as plt
import h5py

from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, Random_POSE
from scripted_policy import PickAndTransferPolicy
import IPython
e = IPython.embed


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """
    config = dict()
    task_name = args['task_name']
    if task_name == "sim_put_hammer_to_locker_top_layer":
        dataset_dir = 'data/PutHammerToLockerTopLayer/'
    elif task_name == "sim_put_camera_to_locker_top_layer":
        dataset_dir = 'data/PutCameraToLockerTopLayer/'
    elif task_name == "sim_put_green_stapler_to_locker_top_layer":
        dataset_dir = 'data/PutGreenStaplerToLockerTopLayer/'
    elif task_name == "sim_put_black_stapler_to_locker_top_layer":
        dataset_dir = 'data/PutBlackStaplerToLockerTopLayer/'
    else:
        NotImplementedError
    num_episodes = 50
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'angle_left'
    config['task_name'] = task_name
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    # if task_name[:23] == 'sim_open_cabinet_drawer':
    if 'sim_' in task_name:
        policy_cls = PickAndTransferPolicy
    else:
        raise NotImplementedError

    success = []
    try_count = 0
    while True:
        if np.sum(success) >= num_episodes:
            break
        # print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name, config)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(config, inject_noise)
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()
        rewards = [ts.reward for ts in episode[1:]]
        episode_return = np.sum(rewards)
        # episode_max_reward = np.max(rewards)
        episode_reward = rewards[-1]
        if episode_reward == env.task.max_reward:
            print(f"{try_count=} Successful, {episode_return=}")
        else:
            print(f"{try_count=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = ctrl[0]
            right_ctrl = ctrl[1]
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0
        print('subtask',subtask_info)
        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name, config)
        Random_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)
        rewards = [ts.reward for ts in episode_replay[1:]]
        episode_return = np.sum(rewards)
        # episode_max_reward = np.max(rewards)
        episode_reward = rewards[-1]
        if episode_reward == env.task.max_reward:
            success.append(1)
            print(f"{try_count=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{try_count=} Failed")

        plt.close()
        try_count = try_count + 1
        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """
        if episode_reward == env.task.max_reward:
            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            joint_traj = joint_traj[:-1]
            episode_replay = episode_replay[:-1]

            # len(joint_traj) i.e. actions: max_timesteps
            # len(episode_replay) i.e. time steps: max_timesteps + 1
            max_timesteps = len(joint_traj)
            while joint_traj:
                action = joint_traj.pop(0)
                ts = episode_replay.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            # HDF5
            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'{task_name}_{np.sum(success)-1}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
        else:
            print(f'Failed demo, retry next demo\n')

        print(f'Saved to {dataset_dir}')
        print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    # parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))


import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import matplotlib
# matplotlib.use('TkAgg')
from constants import SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env, make_ee_sim_env_goal_condition
from sim_env import make_sim_env, BOX_POSE, make_sim_env_goal_condition, ENV_POSE
from scripted_policy import PickBlockPolicy, CloseCabinetDrawerPolicy, LeftArmPutStuff2DrawerPolicy, \
    RightArmPutStuff2DrawerPolicy, DualArmPutStuff2DrawerPolicy, OpenCabinetPolicy, PutStuff2BasketPolicy, \
PutStuff2LockerTopLayerPolicy, PutStuff2PanPolicy, PutStuff2LockerMiddleLayerPolicy, LeftArmBlockCoverSmallerMarkerPolicy, \
RightArmBlockCoverSmallerMarkerPolicy, TransferBlockPolicy
from constants import TEXT_EMBEDDINGS
# from imitate_episodes import GetTextEmbeddings
import IPython
e = IPython.embed
from GetGoalImg import save_goal_image_to_hdf5, get_goal_image
from random_initialize_envs import initialize_envs


def PickBlock_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head'] # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
    # if True:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
        # if True:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f" Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    ENV_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    # episode_max_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f" Successful, {episode_return=}")
    else:
        print(f"Failed")
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

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

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'


def CloseCabinetDrawer_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_max_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    # episode_max_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

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

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def LeftArmPutStuff2Drawer_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_max_reward = np.max(rewards)
    episode_reward = np.max(rewards)
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    # episode_max_reward = np.max(rewards)
    episode_reward = np.max(rewards)
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

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

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def RightArmPutStuff2Drawer_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_max_reward = np.max(rewards)
    episode_reward = np.max(rewards)
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = np.max(rewards)
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

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

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def DualArmPutStuff2Drawer_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    episode_reward = np.max(rewards)
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = np.max(rewards)
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def OpenCabinet_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def PutStuff2Basket_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def PutStuff2LockerTopLayer_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def PutStuff2Pan_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def PutStuff2LockerMiddleLayer_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def LeftArmBlockCoverSmallerMarker_GetGoalImg4Inference(config, init_obj_poses):
    # config['get_sim_goal_image'] = True
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'


def RightArmBlockCoverSmallerMarker_GetGoalImg4Inference(config, init_obj_poses):
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def TransferBlock_GetGoalImg4Inference(config, init_obj_poses):
    camera_names = ['head']  # TODO Hardcode
    render_cam_name = 'head'
    # setup the environment
    inject_noise = False
    task_name = config['task_name']
    env = make_ee_sim_env_goal_condition(task_name, config, init_obj_poses)
    ts = env.reset()
    episode = [ts]
    policy_cls = get_policy_cls(task_name)
    policy = policy_cls(config, inject_noise)
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for step in range(config['episode_len']):
        action = policy(ts)
        ts = env.step(action)
        episode.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    plt.close()

    plt.close()
    rewards = [ts.reward for ts in episode[1:]]
    episode_return = np.sum(rewards)
    # episode_reward = np.max(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    joint_traj = [ts.observation['qpos'] for ts in episode]
    # replace gripper pose with gripper control
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = ctrl[0]
        right_ctrl = ctrl[1]
        joint[6] = left_ctrl
        joint[6 + 7] = right_ctrl

    subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
    print('subtask', subtask_info)
    # clear unused variables
    del env
    del episode
    del policy

    # setup the environment
    print('Replaying joint commands')
    env = make_sim_env_goal_condition(task_name, config)
    BOX_POSE[0] = subtask_info  # make sure the sim_env has the same object configurations as ee_sim_env
    ts = env.reset()

    episode_replay = [ts]
    # setup plotting
    if config['onscreen_render']:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])
        plt.ion()
    for t in range(len(joint_traj)):  # note: this will increase episode length by 1
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)
        if config['onscreen_render']:
            plt_img.set_data(ts.observation['images'][render_cam_name])
            plt.pause(0.02)
    episode_reward = 0
    rewards = [ts.reward for ts in episode_replay[1:]]
    episode_return = np.sum(rewards)
    episode_reward = rewards[-1]
    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")
    else:
        print(f"Failed")

    if episode_reward == env.task.max_reward:
        print(f"Successful, {episode_return=}")

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        t0 = time.time()
        save_dir = f'tempt/{task_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = os.path.join(save_dir, f'episode_new.hdf5')
        import h5py
        with h5py.File(file_name, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        max_steps = config['episode_len']
        import h5py
        goal_image = get_goal_image(file_name, max_steps, config['goal_image_camera'])
        print(f'get goal_image successful !')
        goal_images = np.tile(goal_image, (max_steps, 1, 1, 1))
        save_goal_image_to_hdf5(file_name, goal_images)
        print(f'save goal_image to hdf5 file successful !')
        plt.close()
        return goal_image, file_name
    else:
        print(f'Failed! the goal-image/dominstartion is bad, use another init env-pose !')
        return 'continue', 'continue'

def get_policy_cls(task_name):
    if 'sim_pick' in task_name and 'block' in task_name:
        policy_cls = PickBlockPolicy
    elif 'sim_close_cabinet' in task_name:
        policy_cls = CloseCabinetDrawerPolicy
    elif 'sim_left_arm_put' in task_name and 'to_cabinet' in task_name:
        policy_cls = LeftArmPutStuff2DrawerPolicy
    elif 'sim_right_arm_put' in task_name and 'to_cabinet' in task_name:
        policy_cls = RightArmPutStuff2DrawerPolicy
    elif 'sim_dual_arm_put' in task_name and 'to_cabinet' in task_name:
        policy_cls = DualArmPutStuff2DrawerPolicy
    elif 'sim_open_cabinet' in task_name:
        policy_cls = OpenCabinetPolicy
    elif 'sim_put' in task_name and 'basket' in task_name:
        policy_cls = PutStuff2BasketPolicy
    elif 'sim_put' in task_name and 'locker_top_layer' in task_name:
        policy_cls = PutStuff2LockerTopLayerPolicy
    elif 'sim_put' in task_name and 'pan' in task_name:
        policy_cls = PutStuff2PanPolicy
    elif 'sim_put' in task_name and 'locker_middle_layer' in task_name:
        policy_cls = PutStuff2LockerMiddleLayerPolicy
    elif 'sim_left_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        policy_cls = LeftArmBlockCoverSmallerMarkerPolicy
    elif 'sim_right_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        policy_cls = RightArmBlockCoverSmallerMarkerPolicy
    elif 'sim_transfer' in task_name and 'block' in task_name:
        policy_cls = TransferBlockPolicy
    else:
        NotImplementedError

    return policy_cls

def get_goal_img_4_task(task_name, config):
    while True:
        # elif 'sim_insertion' in task_name:
        initial_pose_of_env = initialize_envs(task_name)
        ENV_POSE[0] = initial_pose_of_env
        if config['using_goal_image_inference'] or config['check_success']:
            if 'sim_pick' in task_name and 'block' in task_name:
                goal_image, hdf5_name = PickBlock_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_close_cabinet' in task_name:
                goal_image, hdf5_name = CloseCabinetDrawer_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_left_arm_put' in task_name and 'to_cabinet' in task_name:
                goal_image, hdf5_name = LeftArmPutStuff2Drawer_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_right_arm_put' in task_name and 'to_cabinet' in task_name:
                goal_image, hdf5_name = RightArmPutStuff2Drawer_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_dual_arm_put' in task_name and 'to_cabinet' in task_name:
                goal_image, hdf5_name = DualArmPutStuff2Drawer_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_open_cabinet' in task_name:
                goal_image, hdf5_name = OpenCabinet_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_put' in task_name and 'basket' in task_name:
                goal_image, hdf5_name = PutStuff2Basket_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_put' in task_name and 'locker_top_layer' in task_name:
                goal_image, hdf5_name = PutStuff2LockerTopLayer_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_put' in task_name and 'pan' in task_name:
                goal_image, hdf5_name = PutStuff2Pan_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_put' in task_name and 'locker_middle_layer' in task_name:
                goal_image, hdf5_name = PutStuff2LockerMiddleLayer_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_left_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
                goal_image, hdf5_name = LeftArmBlockCoverSmallerMarker_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_right_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
                goal_image, hdf5_name = RightArmBlockCoverSmallerMarker_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            elif 'sim_transfer' in task_name and 'block' in task_name:
                goal_image, hdf5_name = TransferBlock_GetGoalImg4Inference(config, initial_pose_of_env)
                if str(goal_image) != 'continue':
                    break
            else:
                NotImplementedError

    return goal_image, hdf5_name

def get_infer_task_emb(task_name, TEXT_EMBEDDINGS):
    if 'sim_pick_left_blue_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[0]
    elif 'sim_pick_left_green_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[1]
    elif 'sim_pick_left_red_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[2]
    elif 'sim_pick_left_yellow_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[3]
    elif 'sim_pick_right_blue_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[4]
    elif 'sim_pick_right_green_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[5]
    elif 'sim_pick_right_red_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[6]
    elif 'sim_pick_right_yellow_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[7]

    elif 'sim_close_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[8]
    elif 'sim_close_cabinet_middle_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[9]
    elif 'sim_close_cabinet_top_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[10]

    elif 'sim_left_arm_put_apple_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'sim_left_arm_put_banana_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'sim_left_arm_put_blue_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'sim_left_arm_put_green_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[14]

    elif 'sim_right_arm_put_apple_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'sim_right_arm_put_banana_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'sim_right_arm_put_blue_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'sim_right_arm_put_green_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[14]

    elif 'sim_dual_arm_put_apple_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'sim_dual_arm_put_banana_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'sim_dual_arm_put_blue_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'sim_dual_arm_put_green_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[14]

    elif 'sim_open_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[15]
    elif 'sim_open_cabinet_middle_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[16]
    elif 'sim_open_cabinet_top_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[17]

    elif 'sim_put_black_stapler_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[18]
    elif 'sim_put_camera_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[19]
    elif 'sim_put_green_stapler_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[20]
    elif 'sim_put_hammer_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[21]

    elif 'sim_put_black_stapler_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[22]
    elif 'sim_put_camera_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[23]
    elif 'sim_put_green_stapler_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[24]
    elif 'sim_put_hammer_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[25]

    elif 'sim_put_apple_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[26]
    elif 'sim_put_duck_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[27]
    elif 'sim_put_pig_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[28]
    elif 'sim_put_teapot_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[29]

    elif 'sim_put_black_stapler_to_locker_middle_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[30]
    elif 'sim_put_blue_block_to_locker_middle_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[31]

    elif 'sim_left_arm_blue_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[32]
    elif 'sim_left_arm_blue_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[33]
    elif 'sim_left_arm_blue_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[34]
    elif 'sim_left_arm_blue_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[35]
    elif 'sim_left_arm_green_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[36]
    elif 'sim_left_arm_green_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[37]
    elif 'sim_left_arm_green_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[38]
    elif 'sim_left_arm_green_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[39]
    elif 'sim_left_arm_red_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[40]
    elif 'sim_left_arm_red_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[41]
    elif 'sim_left_arm_red_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[42]
    elif 'sim_left_arm_red_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[43]
    elif 'sim_left_arm_yellow_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[44]
    elif 'sim_left_arm_yellow_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[45]
    elif 'sim_left_arm_yellow_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[46]
    elif 'sim_left_arm_yellow_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[47]

    elif 'sim_right_arm_blue_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[32]
    elif 'sim_right_arm_blue_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[33]
    elif 'sim_right_arm_blue_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[34]
    elif 'sim_right_arm_blue_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[35]
    elif 'sim_right_arm_green_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[36]
    elif 'sim_right_arm_green_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[37]
    elif 'sim_right_arm_green_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[38]
    elif 'sim_right_arm_green_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[39]
    elif 'sim_right_arm_red_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[40]
    elif 'sim_right_arm_red_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[41]
    elif 'sim_right_arm_red_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[42]
    elif 'sim_right_arm_red_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[43]
    elif 'sim_right_arm_yellow_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[44]
    elif 'sim_right_arm_yellow_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[45]
    elif 'sim_right_arm_yellow_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[46]
    elif 'sim_right_arm_yellow_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[47]

    elif 'sim_transfer_left_blue_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[48]
    elif 'sim_transfer_left_green_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[49]
    elif 'sim_transfer_left_red_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[50]
    elif 'sim_transfer_left_yellow_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[51]
    elif 'sim_transfer_right_blue_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[52]
    elif 'sim_transfer_right_green_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[53]
    elif 'sim_transfer_right_red_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[54]
    elif 'sim_transfer_right_yellow_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[55]
    else:
        print('There is no the TASK embedding, please change the task_name to correct!')
        exit()

    return task_emb

def get_infer_redundant_task_emb(task_name, TEXT_EMBEDDINGS):
    if 'sim_pick_left_blue_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[0]
    elif 'sim_pick_left_green_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[1]
    elif 'sim_pick_left_red_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[2]
    elif 'sim_pick_left_yellow_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[3]
    elif 'sim_pick_right_blue_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[4]
    elif 'sim_pick_right_green_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[5]
    elif 'sim_pick_right_red_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[6]
    elif 'sim_pick_right_yellow_block' in task_name:
        task_emb = TEXT_EMBEDDINGS[7]

    elif 'sim_close_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[8]
    elif 'sim_close_cabinet_middle_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[9]
    elif 'sim_close_cabinet_top_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[10]

    elif 'sim_left_arm_put_apple_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[11]
    elif 'sim_left_arm_put_banana_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[12]
    elif 'sim_left_arm_put_blue_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[13]
    elif 'sim_left_arm_put_green_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[14]

    elif 'sim_right_arm_put_apple_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[15]
    elif 'sim_right_arm_put_banana_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[16]
    elif 'sim_right_arm_put_blue_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[17]
    elif 'sim_right_arm_put_green_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[18]

    elif 'sim_dual_arm_put_apple_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[19]
    elif 'sim_dual_arm_put_banana_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[20]
    elif 'sim_dual_arm_put_blue_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[21]
    elif 'sim_dual_arm_put_green_bottle_to_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[22]

    elif 'sim_open_cabinet_bottom_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[23]
    elif 'sim_open_cabinet_middle_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[24]
    elif 'sim_open_cabinet_top_drawer' in task_name:
        task_emb = TEXT_EMBEDDINGS[25]

    elif 'sim_put_black_stapler_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[26]
    elif 'sim_put_camera_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[27]
    elif 'sim_put_green_stapler_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[28]
    elif 'sim_put_hammer_to_basket' in task_name:
        task_emb = TEXT_EMBEDDINGS[29]

    elif 'sim_put_black_stapler_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[30]
    elif 'sim_put_camera_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[31]
    elif 'sim_put_green_stapler_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[32]
    elif 'sim_put_hammer_to_locker_top_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[33]

    elif 'sim_put_apple_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[34]
    elif 'sim_put_duck_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[35]
    elif 'sim_put_pig_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[36]
    elif 'sim_put_teapot_to_pan' in task_name:
        task_emb = TEXT_EMBEDDINGS[37]

    elif 'sim_put_black_stapler_to_locker_middle_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[38]
    elif 'sim_put_blue_block_to_locker_middle_layer' in task_name:
        task_emb = TEXT_EMBEDDINGS[39]

    elif 'sim_left_arm_blue_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[40]
    elif 'sim_left_arm_blue_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[41]
    elif 'sim_left_arm_blue_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[42]
    elif 'sim_left_arm_blue_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[43]
    elif 'sim_left_arm_green_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[44]
    elif 'sim_left_arm_green_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[45]
    elif 'sim_left_arm_green_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[46]
    elif 'sim_left_arm_green_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[47]
    elif 'sim_left_arm_red_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[48]
    elif 'sim_left_arm_red_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[49]
    elif 'sim_left_arm_red_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[50]
    elif 'sim_left_arm_red_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[51]
    elif 'sim_left_arm_yellow_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[52]
    elif 'sim_left_arm_yellow_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[53]
    elif 'sim_left_arm_yellow_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[54]
    elif 'sim_left_arm_yellow_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[55]

    elif 'sim_right_arm_blue_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[56]
    elif 'sim_right_arm_blue_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[57]
    elif 'sim_right_arm_blue_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[58]
    elif 'sim_right_arm_blue_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[59]
    elif 'sim_right_arm_green_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[60]
    elif 'sim_right_arm_green_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[61]
    elif 'sim_right_arm_green_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[62]
    elif 'sim_right_arm_green_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[63]
    elif 'sim_right_arm_red_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[64]
    elif 'sim_right_arm_red_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[65]
    elif 'sim_right_arm_red_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[66]
    elif 'sim_right_arm_red_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[67]
    elif 'sim_right_arm_yellow_box_cover_bottom_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[68]
    elif 'sim_right_arm_yellow_box_cover_bottom_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[69]
    elif 'sim_right_arm_yellow_box_cover_upper_left_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[70]
    elif 'sim_right_arm_yellow_box_cover_upper_right_marker' in task_name:
        task_emb = TEXT_EMBEDDINGS[71]

    elif 'sim_transfer_left_blue_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[72]
    elif 'sim_transfer_left_green_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[73]
    elif 'sim_transfer_left_red_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[74]
    elif 'sim_transfer_left_yellow_block_to_right' in task_name:
        task_emb = TEXT_EMBEDDINGS[75]
    elif 'sim_transfer_right_blue_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[76]
    elif 'sim_transfer_right_green_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[77]
    elif 'sim_transfer_right_red_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[78]
    elif 'sim_transfer_right_yellow_block_to_left' in task_name:
        task_emb = TEXT_EMBEDDINGS[79]
    else:
        print('There is no the TASK embedding, please change the task_name to correct!')
        exit()

    return task_emb


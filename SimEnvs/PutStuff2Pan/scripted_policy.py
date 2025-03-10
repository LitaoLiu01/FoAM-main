import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from ee_sim_env import make_ee_sim_env
from constants import SIM_TASK_CONFIGS

import IPython
e = IPython.embed

class BasePolicy:
    def __init__(self, config, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None
        self.config = config

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)
        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]
        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]
        # print(next_left_waypoint)
        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)
        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)
        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])
        self.step_count += 1
        return np.concatenate([action_left, action_right])

class PickAndTransferPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        PanPose = AllStuffPose[:7]
        Duck = AllStuffPose[7:14]
        Apple = AllStuffPose[14:21]
        Pig = AllStuffPose[21:28]
        Teapot = AllStuffPose[28:35]
        config = self.config
        if config['task_name'] == 'sim_put_duck_to_pan':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Duck[:3] + np.array([0., -0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 120, "xyz": Duck[:3] + np.array([0., -0.02, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 160, "xyz": Duck[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat.elements,
                 "gripper": 0},
                {"t": 200, "xyz": Duck[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 250, "xyz": Duck[:3] + np.array([0., 0.04, 0.3]), "quat": gripper_pick_quat.elements,
                 "gripper": 200},
                {"t": 350, "xyz": PanPose[:3] + np.array([0.02, 0.00, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 400, "xyz": PanPose[:3] + np.array([0.02, 0.00, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        elif config['task_name'] == 'sim_put_apple_to_pan':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 40}, # sleep
                {"t": 80, "xyz": Apple[:3] + np.array([0.00, -0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 40},
                {"t": 120, "xyz": Apple[:3] + np.array([0.00, -0.02, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 40},
                {"t": 160, "xyz": Apple[:3] + np.array([0.00, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 0},
                {"t": 200, "xyz": Apple[:3] + np.array([0.00, -0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 250, "xyz": Apple[:3] + np.array([0.00, 0.04, 0.3]), "quat": gripper_pick_quat.elements,
                 "gripper": 200},
                {"t": 350, "xyz": PanPose[:3] + np.array([0.01, 0.01, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 400, "xyz": PanPose[:3] + np.array([0.01, 0.01, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        elif config['task_name'] == 'sim_put_pig_to_pan':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Pig[:3] + np.array([0.01, -0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 120, "xyz": Pig[:3] + np.array([0.01, -0.01, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 160, "xyz": Pig[:3] + np.array([0.01, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 200, "xyz": Pig[:3] + np.array([0.01, -0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 250, "xyz": Pig[:3] + np.array([0.01, -0.01, 0.3]), "quat": gripper_pick_quat.elements,
                 "gripper": 200},
                {"t": 350, "xyz": PanPose[:3] + np.array([0.01, 0.01, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 400, "xyz": PanPose[:3] + np.array([0.01, 0.01, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        elif config['task_name'] == 'sim_put_teapot_to_pan':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Teapot[:3] + np.array([-0.005, -0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 120, "xyz": Teapot[:3] + np.array([-0.005, -0.01, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 160, "xyz": Teapot[:3] + np.array([-0.005, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 200, "xyz": Teapot[:3] + np.array([-0.005, -0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 250, "xyz": Teapot[:3] + np.array([-0.005, 0.04, 0.3]), "quat": gripper_pick_quat.elements,
                 "gripper": 200},
                {"t": 350, "xyz": PanPose[:3] + np.array([0.0, 0.01, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 200},
                {"t": 400, "xyz": PanPose[:3] + np.array([0.0, 0.01, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        else:
            NotImplementedError

def test_policy(task_name):
    config = {}
    config['task_name'] = task_name
    show_viwer = 'head'
    # example rolling out pick_and_transfer policy
    onscreen_render = True
    inject_noise = False

    # setup the environment
    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    if 'sim_' in task_name:
        env = make_ee_sim_env(task_name, config)
    # elif 'sim_insertion' in task_name:
    #     env = make_ee_sim_env('sim_insertion')
    else:
        raise NotImplementedError

    for episode_idx in range(1):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            ax = plt.subplot()
            # plt_img = ax.imshow(ts.observation['images']['Head'])
            plt_img = ax.imshow(ts.observation['images'][show_viwer])
            plt.ion()

        policy = PickAndTransferPolicy(config, inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                # plt_img.set_data(ts.observation['images']['Head'])
                plt_img.set_data(ts.observation['images'][show_viwer])
                plt.pause(0.02)
        plt.close()
        rewards = [ts.reward for ts in episode[1:]]
        episode_reward = rewards[-1]
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if config['task_name'] == "sim_open_cabinet_drawer_bottom":
            if episode_return >= 450:
                print(f"{episode_idx=} Successful, {episode_return=}")
            else:
                print(f"{episode_idx=}, {episode_return=}, Failed")
        else:
            if episode_reward == env.task.max_reward:
                print(f"{episode_idx=} Successful, {episode_return=}")
            else:
                print(f"{episode_idx=}, {episode_return=}, Failed")


if __name__ == '__main__':
    test_task_name = 'sim_put_duck_to_pan'
    test_policy(test_task_name)

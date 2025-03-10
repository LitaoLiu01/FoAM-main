import numpy as np
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


class PickBlockPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllBoxPose = np.array(ts_first.observation['env_state'])
        RightRedBox = AllBoxPose[:7]
        RightYellowBox = AllBoxPose[7:14]
        RightBlueBox = AllBoxPose[14:21]
        RightGreenBox = AllBoxPose[21:28]
        LeftRedBox = AllBoxPose[28:35]
        LeftYellowBox = AllBoxPose[35:42]
        LeftBlueBox = AllBoxPose[42:49]
        LeftGreenBox = AllBoxPose[49:56]
        config = self.config
        if config['task_name'] == 'sim_pick_right_red_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 100, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 200, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 250, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 350, "xyz": RightRedBox[:3] + np.array([0, 0.14, 0.5]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            ]
        elif config['task_name'] == 'sim_pick_right_yellow_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 100, "xyz": RightYellowBox[:3] + np.array([-0.01, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 200, "xyz": RightYellowBox[:3] + np.array([-0.01, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 250, "xyz": RightYellowBox[:3] + np.array([-0.01, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 350, "xyz": RightYellowBox[:3] + np.array([-0.01, -0.14, 0.5]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            ]
        elif config['task_name'] == 'sim_pick_right_blue_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 100, "xyz": RightBlueBox[:3] + np.array([0, 0.04, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 200, "xyz": RightBlueBox[:3] + np.array([0, 0.04, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 250, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 350, "xyz": RightBlueBox[:3] + np.array([0.02, -0.06, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            ]
        elif config['task_name'] == 'sim_pick_right_green_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 100, "xyz": RightGreenBox[:3] + np.array([-0.01, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 200, "xyz": RightGreenBox[:3] + np.array([-0.01, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 250, "xyz": RightGreenBox[:3] + np.array([-0.01, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 350, "xyz": RightGreenBox[:3] + np.array([-0.01, 0.14, 0.5]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            ]
        elif config['task_name'] == 'sim_pick_left_red_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                {"t": 100, "xyz": LeftRedBox[:3] + np.array([-0.01, 0.01, 0.25]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 200, "xyz": LeftRedBox[:3] + np.array([-0.01, 0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 250, "xyz": LeftRedBox[:3] + np.array([-0.01, 0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LeftRedBox[:3] + np.array([-0.01, 0.14, 0.5]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},
            ]
        elif config['task_name'] == 'sim_pick_left_yellow_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                {"t": 100, "xyz": LeftYellowBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 200, "xyz": LeftYellowBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 250, "xyz": LeftYellowBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LeftYellowBox[:3] + np.array([0, -0.14, 0.5]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},
            ]
        elif config['task_name'] == 'sim_pick_left_blue_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                {"t": 100, "xyz": LeftBlueBox[:3] + np.array([0, -0.025, 0.25]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 200, "xyz": LeftBlueBox[:3] + np.array([0, -0.025, 0.15]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 250, "xyz": LeftBlueBox[:3] + np.array([0, -0.025, 0.12]), "quat": gripper_pick_quat.elements,
                 "gripper": 80},
                {"t": 300, "xyz": LeftBlueBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LeftBlueBox[:3] + np.array([0.01, 0.04, 0.4]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},
            ]
        elif config['task_name'] == 'sim_pick_left_green_block':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                {"t": 100, "xyz": LeftGreenBox[:3] + np.array([-0.01, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                 "gripper": 40},
                {"t": 200, "xyz": LeftGreenBox[:3] + np.array([-0.01, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 40},
                {"t": 250, "xyz": LeftGreenBox[:3] + np.array([-0.01, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LeftGreenBox[:3] + np.array([0, -0.14, 0.5]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},
            ]
        else:
            NotImplementedError

class CloseCabinetDrawerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        cabinet_info = np.array(ts_first.observation['env_state'])
        cabinet_xyz = cabinet_info[:3]
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        config = self.config

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
        ]

        if config['task_name'] == 'sim_close_cabinet_bottom_drawer':
        # Trajectory_0: open the bottom drawer, divided into several range
            # when x = 0.73
            if cabinet_xyz[0] <= 0.83 and cabinet_xyz[1] >= -0.45:
                print(1)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                # z-axis是夹爪对着的方向
                # gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[1, 0, 0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255}, # sleep
                    # approach the cabinet
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.65, -0.04, -0.07]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    # change the pose
                    {"t": 210, "xyz": cabinet_xyz + np.array([-0.55, -0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # attach the cabinet
                    {"t": 260, "xyz": cabinet_xyz + np.array([-0.45, -0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # close the gripper
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.365, -0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.365, -0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": cabinet_xyz + np.array([-0.365, -0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.52, -0.1, 0.26]), "quat": init_mocap_pose_right[3:],
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
            elif cabinet_xyz[0] > 0.83 and cabinet_xyz[1] >= -0.45:
                print(2)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                # z-axis是夹爪对着的方向
                # gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[1, 0, 0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255}, # sleep
                    # approach the cabinet
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.65, 0.04, -0.07]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    # change the pose
                    {"t": 210, "xyz": cabinet_xyz + np.array([-0.55, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # attach the cabinet
                    {"t": 260, "xyz": cabinet_xyz + np.array([-0.45, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # close the gripper
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.365, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.365, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": cabinet_xyz + np.array([-0.365, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.52, -0.1, 0.07]), "quat": init_mocap_pose_right[3:],
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
            elif cabinet_xyz[0] <= 0.83 and cabinet_xyz[1] < -0.45:
                print(3)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                # z-axis是夹爪对着的方向
                # gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[1, 0, 0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255}, # sleep
                    # approach the cabinet
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.65, 0.04, -0.07]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    # change the pose
                    {"t": 210, "xyz": cabinet_xyz + np.array([-0.55, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # attach the cabinet
                    {"t": 260, "xyz": cabinet_xyz + np.array([-0.45, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # close the gripper
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.36, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.36, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 360, "xyz": cabinet_xyz + np.array([-0.36, 0.04, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.52, -0.1, 0.26]), "quat": init_mocap_pose_right[3:],
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
            elif cabinet_xyz[0] > 0.83 and cabinet_xyz[1] < -0.45:
                print(4)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                # z-axis是夹爪对着的方向
                # gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[1, 0, 0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255}, # sleep
                    # approach the cabinet
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.65, 0.1, -0.07]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    # change the pose
                    {"t": 210, "xyz": cabinet_xyz + np.array([-0.55, 0.1, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # attach the cabinet
                    {"t": 260, "xyz": cabinet_xyz + np.array([-0.45, 0.1, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # close the gripper
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.365, 0.1, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.365, 0.1, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": cabinet_xyz + np.array([-0.365, 0.1, -0.07]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.52, -0.1, 0.26]), "quat": init_mocap_pose_right[3:],
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
            else:
                print('else')

        elif config['task_name'] == 'sim_close_cabinet_middle_drawer':
            # Trajectory_1: open the middle drawer
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
            # gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[1, 0, 0], degrees=90)

            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255}, # sleep
                # approach the cabinet
                {"t": 150, "xyz": cabinet_xyz + np.array([-0.65, 0.0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 255},
                # change the pose
                {"t": 210, "xyz": cabinet_xyz + np.array([-0.55, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # attach the cabinet
                {"t": 260, "xyz": cabinet_xyz + np.array([-0.45, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # close the gripper
                {"t": 300, "xyz": cabinet_xyz + np.array([-0.39, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 320, "xyz": cabinet_xyz + np.array([-0.37, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 350, "xyz": cabinet_xyz + np.array([-0.37, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # pull the drawer
                {"t": 400, "xyz": cabinet_xyz + np.array([-0.52, -0.1, 0.26]), "quat": init_mocap_pose_right[3:],
                 "gripper": 255},
                # pull the drawer
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
            ]

        elif config['task_name'] == 'sim_close_cabinet_top_drawer':
            # Trajectory_2: open the top drawer
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255}, # sleep
                # approach the cabinet
                {"t": 150, "xyz": cabinet_xyz + np.array([-0.60, 0.0, 0.26]), "quat": gripper_pick_quat.elements, "gripper": 255},
                # change the pose
                {"t": 225, "xyz": cabinet_xyz + np.array([-0.53, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # attach the cabinet
                {"t": 260, "xyz": cabinet_xyz + np.array([-0.45, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # close the gripper
                {"t": 300, "xyz": cabinet_xyz + np.array([-0.39, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 320, "xyz": cabinet_xyz + np.array([-0.37, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                {"t": 350, "xyz": cabinet_xyz + np.array([-0.37, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # pull the drawer
                {"t": 400, "xyz": cabinet_xyz + np.array([-0.52, -0.1, 0.26]), "quat": init_mocap_pose_right[3:],
                 "gripper": 255},
                # pull the drawer
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
            ]
        else:
            print('There is no correct task_name, get trajectory fail! ')

class LeftArmPutStuff2DrawerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        CabinetPose = AllStuffPose[:7]
        BottomDrawer = AllStuffPose[7:8]
        MiddleDrawer = AllStuffPose[8:9]
        TopDrawer = AllStuffPose[9:10]
        Apple = AllStuffPose[10:17]
        Banana = AllStuffPose[17:24]
        Cup = AllStuffPose[24:31]
        Bottle = AllStuffPose[31:38]
        config = self.config
        if CabinetPose[0] >= 0.82:
            if config['task_name'] == 'sim_left_arm_put_apple_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 80},
                    {"t": 150, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 80},
                    {"t": 200, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 220},
                    {"t": 250, "xyz": Apple[:3] + np.array([-0.05, -0.01, 0.2]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 220},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 220},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_left_arm_put_banana_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Banana[:3] + np.array([-0.015, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": Banana[:3] + np.array([-0.015, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Banana[:3] + np.array([-0.015, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": Banana[:3] + np.array([-0.05, -0.02, 0.2]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_left_arm_put_green_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": Cup[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": Cup[:3] + np.array([-0.05, 0.1, 0.2]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_left_arm_put_blue_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep

                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.25]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 80},
                    {"t": 150, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 80},
                    {"t": 200, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": Bottle[:3] + np.array([-0.05, 0.1, 0.3]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_2.elements, "gripper": 255},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, 0.05, 0.2]),
                     "quat": gripper_pick_quat_2.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError
        if CabinetPose[0] < 0.82:
            if config['task_name'] == 'sim_left_arm_put_apple_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 80},
                    {"t": 150, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 80},
                    {"t": 200, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 220},
                    {"t": 250, "xyz": Apple[:3] + np.array([-0.05, -0.01, 0.2]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 220},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.38, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 220},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.38, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 0},
                    {"t": 425, "xyz": Apple[:3] + np.array([0, 0, CabinetPose[2] + 0.2]),
                     "quat": init_mocap_pose_right[3:], "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_left_arm_put_banana_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Banana[:3] + np.array([-0.005, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": Banana[:3] + np.array([-0.005, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Banana[:3] + np.array([-0.005, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": Banana[:3] + np.array([-0.05, -0.02, 0.2]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.38, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.38, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 0},
                    {"t": 425, "xyz": Banana[:3] + np.array([0, 0, CabinetPose[2] + 0.2]),
                     "quat": init_mocap_pose_right[3:], "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_left_arm_put_green_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": Cup[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": Cup[:3] + np.array([-0.05, 0.1, 0.2]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.05, 0.2]),
                     "quat": gripper_pick_quat_1.elements, "gripper": 0},
                    {"t": 425, "xyz": Cup[:3] + np.array([0, 0, CabinetPose[2] + 0.2]),
                     "quat": init_mocap_pose_right[3:], "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_left_arm_put_blue_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)

                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.25]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 80},
                    {"t": 150, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 80},
                    {"t": 200, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": Bottle[:3] + np.array([-0.05, -0.03, 0.3]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.05, 0.2]),
                     "quat": gripper_pick_quat_2.elements, "gripper": 255},
                    {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.05, 0.2]),
                     "quat": gripper_pick_quat_2.elements, "gripper": 0},
                    {"t": 425, "xyz": Bottle[:3] + np.array([0, 0, CabinetPose[2] + 0.2]),
                     "quat": init_mocap_pose_right[3:], "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

class RightArmPutStuff2DrawerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        CabinetPose = AllStuffPose[:7]
        BottomDrawer = AllStuffPose[7:8]
        MiddleDrawer = AllStuffPose[8:9]
        TopDrawer = AllStuffPose[9:10]
        Apple = AllStuffPose[10:17]
        Banana = AllStuffPose[17:24]
        Cup = AllStuffPose[24:31]
        Bottle = AllStuffPose[31:38]
        config = self.config
        if CabinetPose[0] >= 0.82:
            if config['task_name'] == 'sim_right_arm_put_apple_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Apple[:3] + np.array([-0.015, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 150, "xyz": Apple[:3] + np.array([-0.015, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 200, "xyz": Apple[:3] + np.array([-0.015, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 220},
                {"t": 250, "xyz": Apple[:3] + np.array([-0.05, 0.01, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 220},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 220},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_right_arm_put_banana_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 80, "xyz": Banana[:3] + np.array([-0.015, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 150, "xyz": Banana[:3] + np.array([-0.015, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 200, "xyz": Banana[:3] + np.array([-0.015, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 250, "xyz": Banana[:3] + np.array([-0.05, 0.02, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_right_arm_put_green_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 80, "xyz": Cup[:3] + np.array([-0.0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 150, "xyz": Cup[:3] + np.array([-0.015, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 200, "xyz": Cup[:3] + np.array([-0.0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 250, "xyz": Cup[:3] + np.array([-0.05, -0.1, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_right_arm_put_blue_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Bottle[:3] + np.array([0.01, 0.05, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 150, "xyz": Bottle[:3] + np.array([0.01, 0.05, 0.1]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 200, "xyz": Bottle[:3] + np.array([0.01, 0.05, 0.1]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 250, "xyz": Bottle[:3] + np.array([-0.05, -0.1, 0.3]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.4, -0.05, 0.2]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError
        if CabinetPose[0] < 0.82:
            if config['task_name'] == 'sim_right_arm_put_apple_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Apple[:3] + np.array([-0.015, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 150, "xyz": Apple[:3] + np.array([-0.015, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 80},
                {"t": 200, "xyz": Apple[:3] + np.array([-0.015, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 220},
                {"t": 250, "xyz": Apple[:3] + np.array([-0.05, 0.01, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 220},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 220},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 0},
                {"t": 425, "xyz": Apple[:3] + np.array([0, 0, CabinetPose[2] + 0.2]), "quat": init_mocap_pose_right[3:], "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_right_arm_put_banana_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 80, "xyz": Banana[:3] + np.array([-0.015, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 150, "xyz": Banana[:3] + np.array([-0.015, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 200, "xyz": Banana[:3] + np.array([-0.015, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 250, "xyz": Banana[:3] + np.array([-0.05, 0.02, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 0},
                {"t": 425, "xyz": Banana[:3] + np.array([0, 0, CabinetPose[2] + 0.2]), "quat": init_mocap_pose_right[3:], "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_right_arm_put_green_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                {"t": 80, "xyz": Cup[:3] + np.array([-0.0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 150, "xyz": Cup[:3] + np.array([-0.015, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                {"t": 200, "xyz": Cup[:3] + np.array([-0.0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                {"t": 250, "xyz": Cup[:3] + np.array([-0.05, -0.1, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_1.elements, "gripper": 0},
                {"t": 425, "xyz": Cup[:3] + np.array([0, 0, CabinetPose[2] + 0.2]), "quat": init_mocap_pose_right[3:], "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif config['task_name'] == 'sim_right_arm_put_blue_bottle_to_cabinet_bottom_drawer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Bottle[:3] + np.array([0.01, 0.05, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 150, "xyz": Bottle[:3] + np.array([0.01, 0.05, 0.1]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 200, "xyz": Bottle[:3] + np.array([0.01, 0.05, 0.1]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 250, "xyz": Bottle[:3] + np.array([-0.05, -0.1, 0.3]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, -0.05, 0.2]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                {"t": 425, "xyz": Bottle[:3] + np.array([0, 0, CabinetPose[2] + 0.2]), "quat": init_mocap_pose_right[3:], "gripper": 255},
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

class DualArmPutStuff2DrawerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        CabinetPose = AllStuffPose[:7]
        cabinet_xyz = CabinetPose[:3]
        BottomDrawer = AllStuffPose[7:8]
        MiddleDrawer = AllStuffPose[8:9]
        TopDrawer = AllStuffPose[9:10]
        Apple = AllStuffPose[10:17]
        Banana = AllStuffPose[17:24]
        Cup = AllStuffPose[24:31]
        Bottle = AllStuffPose[31:38]
        config = self.config
        # get right arm trajectory, pick the stuff
        if 'sim_' in config['task_name']:
            if cabinet_xyz[0] <= 0.75 and cabinet_xyz[1] >= -0.4:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    # approach the cabinet
                    {"t": 100, "xyz": cabinet_xyz + np.array([-0.47, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # change the pose
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.47, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 200, "xyz": cabinet_xyz + np.array([-0.335, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 220, "xyz": cabinet_xyz + np.array([-0.335, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.50, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                ]
            elif 0.75 < cabinet_xyz[0] <= 0.78 and cabinet_xyz[1] >= -0.4:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    # approach the cabinet
                    {"t": 100, "xyz": cabinet_xyz + np.array([-0.47, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # change the pose
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.47, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 200, "xyz": cabinet_xyz + np.array([-0.34, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 220, "xyz": cabinet_xyz + np.array([-0.34, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.50, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": cabinet_xyz + np.array([-0.60, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                ]
            elif 0.78 <= cabinet_xyz[0] <= 0.80 and cabinet_xyz[1] >= -0.4:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    # approach the cabinet
                    {"t": 100, "xyz": cabinet_xyz + np.array([-0.47, -0.02, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # change the pose
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.47, -0.02, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 200, "xyz": cabinet_xyz + np.array([-0.34, -0.02, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 220, "xyz": cabinet_xyz + np.array([-0.34, -0.02, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.50, -0.02, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 350, "xyz": cabinet_xyz + np.array([-0.58, 0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.58, 0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                ]
            elif 0.80 < cabinet_xyz[0] and cabinet_xyz[1] >= -0.4:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    # approach the cabinet
                    {"t": 100, "xyz": cabinet_xyz + np.array([-0.47, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # change the pose
                    {"t": 150, "xyz": cabinet_xyz + np.array([-0.47, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 200, "xyz": cabinet_xyz + np.array([-0.34, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 220, "xyz": cabinet_xyz + np.array([-0.34, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.50, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 350, "xyz": cabinet_xyz + np.array([-0.58, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.58, -0.01, -0.04]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                ]
            else:
                NotImplementedError

        # get left arm trajectory, open the drawer
        if 'sim_' in config['task_name']:
            if CabinetPose[0] >= 0.82:
                if config['task_name'] == 'sim_dual_arm_put_apple_to_cabinet_bottom_drawer':
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    print('Apple', Apple[:3])
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 60, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                         "gripper": 80},
                        {"t": 120, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 80},
                        {"t": 160, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 220},
                        {"t": 200, "xyz": Apple[:3] + np.array([-0.015, 0.04, 0.3]),
                         "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.385, 0.275, 0.1]) + np.array([-0.05, -0.01, 0.2]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 220},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 220},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                elif config['task_name'] == 'sim_dual_arm_put_banana_to_cabinet_bottom_drawer':
                    print('trajectory_1')
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 70, "xyz": Banana[:3] + np.array([-0.008, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 110, "xyz": Banana[:3] + np.array([-0.008, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 160, "xyz": Banana[:3] + np.array([-0.008, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 200, "xyz": Banana[:3] + np.array([-0.008, 0.04, 0.3]),
                         "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.385, 0.275, 0.1]) + np.array([-0.05, -0.02, 0.2]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 255},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 255},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                elif config['task_name'] == 'sim_dual_arm_put_green_bottle_to_cabinet_bottom_drawer':
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 80, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 150, "xyz": Cup[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 200, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.385, 0.275, 0.1]) + np.array([-0.05, 0.1, 0.2]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 255},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 255},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                elif config['task_name'] == 'sim_dual_arm_put_blue_bottle_to_cabinet_bottom_drawer':
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
                    gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    # gripper_pick_quat_3 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=90)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 80, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.25]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 80},
                        {"t": 150, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 80},
                        {"t": 200, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.385, 0.275, 0.1]) + np.array([-0.05, 0.1, 0.3]), "quat": gripper_pick_quat_2.elements,
                         "gripper": 255},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_2.elements, "gripper": 255},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.06, 0.2]),
                         "quat": gripper_pick_quat_2.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                else:
                    NotImplementedError
            elif 0.78 <= CabinetPose[0] < 0.82:
                if config['task_name'] == 'sim_dual_arm_put_apple_to_cabinet_bottom_drawer':
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    gripper_pick_quat_1 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=40)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 60, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                         "gripper": 80},
                        {"t": 120, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 80},
                        {"t": 160, "xyz": Apple[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 220},
                        {"t": 200, "xyz": Apple[:3] + np.array([-0.015, 0.04, 0.3]),
                         "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.38, 0.27, 0.1]) + np.array([-0.05, -0.01, 0.2]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 220},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.09, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 220},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.09, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                elif config['task_name'] == 'sim_dual_arm_put_banana_to_cabinet_bottom_drawer':
                    print('trajectory_2')
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    gripper_pick_quat_1 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=40)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 70, "xyz": Banana[:3] + np.array([-0.008, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 110, "xyz": Banana[:3] + np.array([-0.008, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 160, "xyz": Banana[:3] + np.array([-0.008, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 200, "xyz": Banana[:3] + np.array([-0.008, 0.04, 0.3]),
                         "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.38, 0.27, 0.1]) + np.array([-0.05, -0.01, 0.2]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 220},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.375, 0.09, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 220},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.375, 0.09, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                elif config['task_name'] == 'sim_dual_arm_put_green_bottle_to_cabinet_bottom_drawer':
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    gripper_pick_quat_1 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=40)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 80, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 150, "xyz": Cup[:3] + np.array([-0.015, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 0},
                        {"t": 200, "xyz": Cup[:3] + np.array([-0.0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.38, 0.27, 0.1]) + np.array([-0.05, -0.01, 0.2]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 220},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.37, 0.09, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 220},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.37, 0.09, 0.2]),
                         "quat": gripper_pick_quat_1.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                elif config['task_name'] == 'sim_dual_arm_put_blue_bottle_to_cabinet_bottom_drawer':
                    gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                    gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                    gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
                    gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                    gripper_pick_quat_2 = gripper_pick_quat_2 * Quaternion(axis=[1.0, .0, .0], degrees=40)
                    self.left_trajectory = [
                        {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                        {"t": 80, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.25]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 80},
                        {"t": 150, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 80},
                        {"t": 200, "xyz": Bottle[:3] + np.array([0.01, -0.03, 0.1]), "quat": gripper_pick_quat_1.elements,
                         "gripper": 255},
                        {"t": 250, "xyz": np.array([0.38, 0.27, 0.1]) + np.array([-0.05, 0.1, 0.3]), "quat": gripper_pick_quat_2.elements,
                         "gripper": 255},
                        {"t": 350, "xyz": CabinetPose[:3] + np.array([-0.38, 0.11, 0.2]),
                         "quat": gripper_pick_quat_2.elements, "gripper": 255},
                        {"t": 400, "xyz": CabinetPose[:3] + np.array([-0.38, 0.11, 0.2]),
                         "quat": gripper_pick_quat_2.elements, "gripper": 0},
                        {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    ]
                else:
                    NotImplementedError
            else:
                NotImplementedError

class OpenCabinetPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        config = self.config
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        cabinet_info = np.array(ts_first.observation['env_state'])
        cabinet_xyz = cabinet_info[:3]
        print('cabinet_xyz', cabinet_xyz)
        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        config = self.config
        # box_xyz = [0.44976987, -0.36175279,  0.05]
        # mark_xyz = [0.42193599, -0.22175123,  0.05]
        # 绕y轴旋转90度的四元数
        # rotation_y = Quaternion(axis=[0, 1, 0], degrees=90)
        # meet_left_quat = meet_left_quat * Quaternion(axis=[.0, 1.0, .0], degrees=90)
        # meet_left_quat2 = meet_left_quat * Quaternion(axis=[1.0, 0, .0], degrees=90)
        # meet_left_quat = (meet_left_quat * rotation_x * rotation_y) / (2**0.5)

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
            {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
        ]
        if config['task_name'] == 'sim_open_cabinet_bottom_drawer':
        # Trajectory_0: open the bottom drawer, divided into several range
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
            # when x = 0.73
            if cabinet_xyz[0] <= 0.75 and cabinet_xyz[1] >= -0.4:
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, -0.04, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.335, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.335, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.04, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.04, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]
            if 0.75 < cabinet_xyz[0] <= 0.78 and cabinet_xyz[1] >= -0.4:
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, -0.03, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.34, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.34, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.03, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.03, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]

            if 0.78 < cabinet_xyz[0] <= 0.80 and cabinet_xyz[1] >= -0.4:
                print('trajectory 1')
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, -0.02, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, -0.02, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.34, -0.02, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.34, -0.02, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.02, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.02, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]
            if 0.80 < cabinet_xyz[0] <= 0.83 and cabinet_xyz[1] >= -0.4:
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, 0, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, 0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.34, 0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.34, 0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, 0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, 0, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]
            if cabinet_xyz[0] <= 0.75 and cabinet_xyz[1] < -0.4:
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.495, -0.025, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.495, -0.025, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.335, -0.025, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.335, -0.025, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.025, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.025, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]
            if 0.75 < cabinet_xyz[0] <= 0.78 and cabinet_xyz[1] < -0.4:
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, -0.01, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, -0.01, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.34, -0.01, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.34, -0.01, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.02, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.02, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]

            if 0.78 < cabinet_xyz[0] <= 0.83 and cabinet_xyz[1] < -0.4:
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    # approach the cabinet
                    {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, -0.0, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    # change the pose
                    {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, -0.0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # attach the cabinet
                    {"t": 300, "xyz": cabinet_xyz + np.array([-0.34, -0.0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    # close the gripper
                    {"t": 320, "xyz": cabinet_xyz + np.array([-0.34, -0.0, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.015, -0.05]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    # pull the drawer
                    {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.02, -0.05]), "quat": gripper_pick_quat.elements, "gripper": 255},
                ]


        if config['task_name'] == 'sim_open_cabinet_middle_drawer':
            # Trajectory_1: open the middle drawer
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                # approach the cabinet
                {"t": 125, "xyz": cabinet_xyz + np.array([-0.47, 0.0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 0},
                # change the pose
                {"t": 250, "xyz": cabinet_xyz + np.array([-0.47, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 0},
                # attach the cabinet
                {"t": 300, "xyz": cabinet_xyz + np.array([-0.39, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 0},
                # close the gripper
                {"t": 320, "xyz": cabinet_xyz + np.array([-0.39, 0.0, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # pull the drawer
                {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, -0.02, 0.08]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # pull the drawer
                {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, -0.02, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 255},
            ]

        if config['task_name'] == 'sim_open_cabinet_top_drawer':
            # Trajectory_2: open the top drawer
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 1, 0], degrees=90)
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0, 0, 1], degrees=90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                # approach the cabinet
                {"t": 125, "xyz": cabinet_xyz + np.array([-0.45, 0.0, 0.26]), "quat": init_mocap_pose_right[3:], "gripper": 0},
                # change the pose
                {"t": 250, "xyz": cabinet_xyz + np.array([-0.45, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 0},
                # attach the cabinet
                {"t": 300, "xyz": cabinet_xyz + np.array([-0.39, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 0},
                # close the gripper
                {"t": 320, "xyz": cabinet_xyz + np.array([-0.39, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # pull the drawer
                {"t": 400, "xyz": cabinet_xyz + np.array([-0.50, 0.0, 0.26]), "quat": gripper_pick_quat.elements,
                 "gripper": 255},
                # pull the drawer
                {"t": 450, "xyz": cabinet_xyz + np.array([-0.60, 0.0, 0.26]), "quat": gripper_pick_quat.elements, "gripper": 255},
            ]

class PutStuff2BasketPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        LockerPose = AllStuffPose[:7]
        Hammer = AllStuffPose[7:14]
        Camera = AllStuffPose[14:21]
        Toothpaste = AllStuffPose[21:28]
        Stapler = AllStuffPose[28:35]
        config = self.config
        if config['task_name'] == 'sim_put_hammer_to_basket':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
            gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Hammer[:3] + np.array([0., -0.02, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 120, "xyz": Hammer[:3] + np.array([0., -0.02, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 160, "xyz": Hammer[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat_1.elements,
                 "gripper": 0},
                {"t": 200, "xyz": Hammer[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 250, "xyz": Camera[:3] + np.array([0., 0.0, 0.4]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LockerPose[:3] + np.array([-0.15, -0, 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 400, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                {"t": 420, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.45]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        elif config['task_name'] == 'sim_put_camera_to_basket':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            gripper_pick_quat_1 = gripper_pick_quat
            gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Camera[:3] + np.array([0.0, -0.02, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 120, "xyz": Camera[:3] + np.array([0.0, -0.02, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 160, "xyz": Camera[:3] + np.array([0.0, -0.02, 0.09]), "quat": gripper_pick_quat_1.elements,
                 "gripper": 0},
                {"t": 200, "xyz": Camera[:3] + np.array([0.0, -0.02, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 250, "xyz": Camera[:3] + np.array([0.0, 0.04, 0.4]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LockerPose[:3] + np.array([-0.15, -0, 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 400, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                {"t": 420, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.45]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        elif config['task_name'] == 'sim_put_green_stapler_to_basket':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            gripper_pick_quat_1 = gripper_pick_quat
            gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Toothpaste[:3] + np.array([0.0, -0.02, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 120, "xyz": Toothpaste[:3] + np.array([0.0, -0.02, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 160, "xyz": Toothpaste[:3] + np.array([0.0, -0.02, 0.09]), "quat": gripper_pick_quat_1.elements,
                 "gripper": 0},
                {"t": 200, "xyz": Toothpaste[:3] + np.array([0.0, -0.02, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 250, "xyz": Camera[:3] + np.array([0.0, 0.04, 0.4]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LockerPose[:3] + np.array([-0.15, -0, 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 400, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                {"t": 420, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.45]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        elif config['task_name'] == 'sim_put_black_stapler_to_basket':
            gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
            gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
            gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
            gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
            self.right_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
            ]
            self.left_trajectory = [
                {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                {"t": 80, "xyz": Stapler[:3] + np.array([0.01, -0.04, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 120, "xyz": Stapler[:3] + np.array([0.01, -0.04, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                {"t": 160, "xyz": Stapler[:3] + np.array([0.01, -0.04, 0.09]), "quat": gripper_pick_quat_1.elements,
                 "gripper": 0},
                {"t": 200, "xyz": Stapler[:3] + np.array([0.01, -0.04, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                {"t": 250, "xyz": Camera[:3] + np.array([0.0, 0.04, 0.4]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 255},
                {"t": 350, "xyz": LockerPose[:3] + np.array([-0.15, -0, 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                {"t": 400, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.25]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                {"t": 420, "xyz": LockerPose[:3] + np.array([-0.15, -0., 0.45]), "quat": gripper_pick_quat_2.elements,
                 "gripper": 0},
                {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
            ]
        else:
            NotImplementedError

class PutStuff2LockerTopLayerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        LockerPose = AllStuffPose[:7]
        Hammer = AllStuffPose[7:14]
        Camera = AllStuffPose[14:21]
        Toothpaste = AllStuffPose[21:28]
        Stapler = AllStuffPose[28:35]
        config = self.config
        if 'top' in config['task_name']:
            if config['task_name'] == 'sim_put_hammer_to_locker_top_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                    {"t": 80, "xyz": Hammer[:3] + np.array([0., -0.02, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                    {"t": 120, "xyz": Hammer[:3] + np.array([0., -0.02, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                    {"t": 160, "xyz": Hammer[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Hammer[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 250, "xyz": Camera[:3] + np.array([0., 0.0, 0.4]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                    {"t": 400, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                    {"t": 420, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.6]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            elif config['task_name'] == 'sim_put_camera_to_locker_top_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                    {"t": 80, "xyz": Camera[:3] + np.array([0., -0.0, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                    {"t": 120, "xyz": Camera[:3] + np.array([0., -0.0, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                    {"t": 160, "xyz": Camera[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Camera[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 250, "xyz": Camera[:3] + np.array([0., 0.04, 0.4]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                    {"t": 400, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                    {"t": 420, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.6]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            elif config['task_name'] == 'sim_put_green_stapler_to_locker_top_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[.0, .0, 1.0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 80}, # sleep
                    {"t": 80, "xyz": Toothpaste[:3] + np.array([0.01, -0.05, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                    {"t": 120, "xyz": Toothpaste[:3] + np.array([0.01, -0.05, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 80},
                    {"t": 160, "xyz": Toothpaste[:3] + np.array([0.01, -0.05, 0.09]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Toothpaste[:3] + np.array([0.01, -0.05, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 250, "xyz": Toothpaste[:3] + np.array([0.01, 0.04, 0.4]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                    {"t": 400, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                    {"t": 420, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.6]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            elif config['task_name'] == 'sim_put_black_stapler_to_locker_top_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat
                gripper_pick_quat_2 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 100}, # sleep
                    {"t": 80, "xyz": Stapler[:3] + np.array([0., -0.0, 0.25]), "quat": gripper_pick_quat_1.elements, "gripper": 100},
                    {"t": 120, "xyz": Stapler[:3] + np.array([0., -0.0, 0.20]), "quat": gripper_pick_quat_1.elements, "gripper": 100},
                    {"t": 160, "xyz": Stapler[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Stapler[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat_1.elements, "gripper": 255},
                    {"t": 250, "xyz": Camera[:3] + np.array([0., 0, 0.4]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 350, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 255},
                    {"t": 400, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.4]), "quat": gripper_pick_quat_2.elements, "gripper": 0},
                    {"t": 420, "xyz": LockerPose[:3] + np.array([-0.1, -0., 0.6]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            else:
                NotImplementedError
        else:
            NotImplementedError

class PutStuff2PanPolicy(BasePolicy):
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

class PutStuff2LockerMiddleLayerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllStuffPose = np.array(ts_first.observation['env_state'])
        Hammer = AllStuffPose[:7]
        Camera = AllStuffPose[7:14]
        Toothpaste = AllStuffPose[14:21]
        Stapler = AllStuffPose[21:28]
        BlueBlock = AllStuffPose[28:35]
        Locker = np.array([0.55, 0, 0.18, 0, 0.9239, 0.3827, 0])
        config = self.config
        if 'middle' in config['task_name']:
            if config['task_name'] == 'sim_put_black_stapler_to_locker_middle_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat_1 * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_3 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                print('stapler', Stapler[:3])
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": Stapler[:3] + np.array([0., -0.0, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": Stapler[:3] + np.array([0., -0.0, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 160, "xyz": Stapler[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Stapler[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 230, "xyz": Stapler[:3] + np.array([0.1, -0.0, 0.25]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},

                    {"t": 260, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.05]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    # 完成姿态变换                    # np.array([0.395, 0, 0.26])locker中层的固定坐标
                    # 进入locker中间层
                    {"t": 350, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    # np.array([0.395, 0, 0.26])
                    {"t": 375, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    {"t": 400, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 0},
                    {"t": 420, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            elif config['task_name'] == 'sim_put_blue_block_to_locker_middle_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat_1 * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_3 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 40}, # sleep
                    {"t": 80, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 40},
                    {"t": 120, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 40},
                    {"t": 160, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 230, "xyz": BlueBlock[:3] + np.array([0.1, -0.0, 0.25]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 260, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.05]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    # 完成姿态变换                    # np.array([0.395, 0, 0.26])locker中层的固定坐标
                    # 进入locker中间层
                    {"t": 350, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    # np.array([0.395, 0, 0.26])
                    {"t": 375, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    {"t": 400, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 0},
                    {"t": 420, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            else:
                NotImplementedError

        elif 'bottom' in config['task_name']:
            if config['task_name'] == 'sim_put_black_stapler_to_locker_bottom_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat_1 * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                gripper_pick_quat_3 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 40}, # sleep
                    {"t": 80, "xyz": Stapler[:3] + np.array([0., -0.0, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 40},
                    {"t": 120, "xyz": Stapler[:3] + np.array([0., -0.0, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 40},
                    {"t": 160, "xyz": Stapler[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": Stapler[:3] + np.array([0., -0.0, 0.09]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 230, "xyz": Stapler[:3] + np.array([0.1, -0.0, 0.25]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 260, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.03]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.03]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.03]),
                     "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    # 完成姿态变换                    # np.array([0.395, 0, 0.26])locker中层的固定坐标
                    # 进入locker中间层
                    # {"t": 350, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.0]), "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 255},
                    # {"t": 450, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.0]),
                    #  "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 255},
                    # {"t": 500, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.0]),
                    #  "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 255},
                    # {"t": 600, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.0]),
                    #  "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 255},
                    # # np.array([0.395, 0, 0.26])
                    # {"t": 375, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, -0.08]), "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 255},
                    # {"t": 400, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, -0.08]), "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 0},
                    # {"t": 420, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, -0.08]), "quat": gripper_pick_quat_3.elements,
                    #  "gripper": 0},
                    # {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            elif config['task_name'] == 'sim_put_blue_block_to_locker_bottom_layer':
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                gripper_pick_quat_1 = gripper_pick_quat * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_2 = gripper_pick_quat_1 * Quaternion(axis=[0, 1.0, .0], degrees=-90)
                gripper_pick_quat_3 = gripper_pick_quat_1 * Quaternion(axis=[1.0, .0, .0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # sleep
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 40}, # sleep
                    {"t": 80, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 40},
                    {"t": 120, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.20]), "quat": gripper_pick_quat.elements, "gripper": 40},
                    {"t": 160, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 200, "xyz": BlueBlock[:3] + np.array([0., -0.02, 0.09]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 230, "xyz": BlueBlock[:3] + np.array([0.1, -0.0, 0.25]), "quat": gripper_pick_quat_1.elements,
                     "gripper": 255},
                    {"t": 260, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.05]), "quat": gripper_pick_quat_2.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.40, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    # 完成姿态变换                    # np.array([0.395, 0, 0.26])locker中层的固定坐标
                    # 进入locker中间层
                    {"t": 350, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    # np.array([0.395, 0, 0.26])
                    {"t": 375, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 255},
                    {"t": 400, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.15, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 0},
                    {"t": 420, "xyz": np.array([0.395, 0, 0.26]) + np.array([0.05, 0.30, 0.05]), "quat": gripper_pick_quat_3.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},
                ]
            else:
                NotImplementedError


        else:
            NotImplementedError

class LeftArmBlockCoverSmallerMarkerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllBoxPose = np.array(ts_first.observation['env_state'])
        RightRedBox = AllBoxPose[:7]
        RightYellowBox = AllBoxPose[7:14]
        RightBlueBox = AllBoxPose[14:21]
        RightGreenBox = AllBoxPose[21:28]
        Marker_1 = AllBoxPose[28:35]
        Marker_2 = AllBoxPose[35:42]
        Marker_3 = AllBoxPose[42:49]
        Marker_4 = AllBoxPose[49:56]
        config = self.config
        if 'upper_left_marker' in config['task_name']:
            if 'sim_left_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0.02, 0.02, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, -0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, 0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, 0.06, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError
        elif 'upper_right_marker' in config['task_name']:
            if 'sim_left_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0.02, -0.02, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.01, -0.005, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.01, -0.005, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.01, -0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.01, -0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.005, -0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.005, -0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, 0.0, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.005, -0.005, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.005, -0.005, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.005, -0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.005, -0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError
        elif 'bottom_right_marker' in config['task_name']:
            if 'sim_left_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0, 0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([0.005, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([0.005, 0.0, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([0.005, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([0.005, 0.0, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([-0.01, 0.01, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([-0.01, 0.01, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([-0.01, 0.01, 0.15]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([-0.005, 0.01, 0.15]),
                     "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, -0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([-0.01, 0.02, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([-0.01, 0.02, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([-0.01, 0.02, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([-0.01, 0.02, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, -0.02, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([0.005, 0.005, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([0.005, 0.005, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([0.005, 0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([0.005, 0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, 0.005, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError
        elif 'bottom_left_marker' in config['task_name']:
            if 'sim_left_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, -0.01, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0, 0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([0.0025, 0.0025, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([0.0025, 0.0025, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([0.0025, 0.0025, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([0.0025, 0.0025, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, -0.03, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([-0., 0.03, 0.3]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([-0., 0.03, 0.2]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([-0., 0.03, 0.15]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([-0., 0.03, 0.15]),
                     "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, 0.015, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, -0.02, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([0.0, 0.03, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([0.0, 0.03, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([0.0, 0.03, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([-0., 0.03, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, 0.02, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_left_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, -0.02, 0.1]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([0.005, 0.005, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([0.005, 0.005, 0.2]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([0.005, 0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([0.005, 0.005, 0.15]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, 0.005, 0.3]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError
        else:
            NotImplementedError

class RightArmBlockCoverSmallerMarkerPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllBoxPose = np.array(ts_first.observation['env_state'])
        RightRedBox = AllBoxPose[:7]
        RightYellowBox = AllBoxPose[7:14]
        RightBlueBox = AllBoxPose[14:21]
        RightGreenBox = AllBoxPose[21:28]
        Marker_1 = AllBoxPose[28:35]
        Marker_2 = AllBoxPose[35:42]
        Marker_3 = AllBoxPose[42:49]
        Marker_4 = AllBoxPose[49:56]
        config = self.config

        if 'upper_left_marker' in config['task_name']:
            if 'sim_right_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0.02, -0.02, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, 0.0, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, 0.0, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_1[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_1[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

        elif 'upper_right_marker' in config['task_name']:
            if 'sim_right_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0.02, -0.02, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.02, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.02, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.02, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.02, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, 0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, 0.0, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_2[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_2[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

        elif 'bottom_right_marker' in config['task_name']:
            if 'sim_right_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0, -0.02, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([-0.005, -0.02, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([-0.005, -0.02, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([-0.005, -0.02, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([-0.005, -0.02, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, -0.02, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([0.0, -0.02, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([0.0, -0.02, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([0.0, -0.02, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([0.0, -0.02, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, -0.02, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, 0.0, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_3[:3] + np.array([0.01, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_3[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

        elif 'bottom_left_marker' in config['task_name']:
            if 'sim_right_arm_red_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([0.005, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([0.005, 0.0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([0.005, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([0.005, 0.0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, 0.0, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_yellow_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, 0.03, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([-0.005, -0.015, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([-0.005, -0.015, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([-0.005, -0.015, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([-0.005, -0.015, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, -0.015, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_blue_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([0.0, -0.02, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([0.0, -0.02, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([0.0, -0.02, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([0.0, -0.02, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, -0.02, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'sim_right_arm_green_box' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0, 0.0, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 240, "xyz": Marker_4[:3] + np.array([0.01, -0.005, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 290, "xyz": Marker_4[:3] + np.array([0.01, -0.005, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 340, "xyz": Marker_4[:3] + np.array([0.01, -0.005, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 365, "xyz": Marker_4[:3] + np.array([0.01, -0.005, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 385, "xyz": Marker_4[:3] + np.array([0.10, -0.005, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

        else:
            NotImplementedError

class TransferBlockPolicy(BasePolicy):
    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']
        AllBoxPose = np.array(ts_first.observation['env_state'])
        RightRedBox = AllBoxPose[:7]
        RightYellowBox = AllBoxPose[7:14]
        RightBlueBox = AllBoxPose[14:21]
        RightGreenBox = AllBoxPose[21:28]
        LeftRedBox = AllBoxPose[28:35]
        LeftYellowBox = AllBoxPose[35:42]
        LeftBlueBox = AllBoxPose[42:49]
        LeftGreenBox = AllBoxPose[49:56]
        meet_xyz = np.array([0.4, 0, 0.4])
        config = self.config
        if 'to_left' in config['task_name']:
            if 'red_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                right_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightRedBox[:3] + np.array([0, 0.01, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightRedBox[:3] + np.array([0.02, -0.02, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_transition.elements, "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, -0.3, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                left_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                left_arm_quat_meet = left_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},  # sleep
                ]
            elif 'yellow_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                right_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightYellowBox[:3] + np.array([0, 0.01, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightYellowBox[:3] + np.array([0, 0.01, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightYellowBox[:3] + np.array([0, 0.01, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightYellowBox[:3] + np.array([0.02, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_transition.elements, "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, -0.3, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                left_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                left_arm_quat_meet = left_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},  # sleep
                ]
            elif 'blue_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                right_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightBlueBox[:3] + np.array([0, 0.04, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightBlueBox[:3] + np.array([0, 0.04, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightBlueBox[:3] + np.array([0, 0.02, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightBlueBox[:3] + np.array([0.02, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_transition.elements, "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, -0.3, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                left_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                left_arm_quat_meet = left_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},  # sleep
                ]
            elif 'green_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                right_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 80, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.25]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 120, "xyz": RightGreenBox[:3] + np.array([0, 0.02, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0},
                    {"t": 150, "xyz": RightGreenBox[:3] + np.array([0, 0.01, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 190, "xyz": RightGreenBox[:3] + np.array([0.02, -0.04, 0.4]), "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_transition.elements, "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.09]), "quat": right_arm_quat_meet.elements, "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, -0.3, 0.09]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},
                ]
                left_arm_quat_meet = right_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                left_arm_quat_meet = left_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 255},  # sleep
                ]
            else:
                NotImplementedError

        elif 'to_right' in config['task_name']:
            if 'red_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                left_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                left_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                right_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, -0.3, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": LeftRedBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": LeftRedBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": LeftRedBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": LeftRedBox[:3] + np.array([0.02, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_transition.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'yellow_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                left_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                left_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                right_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, -0.3, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": LeftYellowBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": LeftYellowBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": LeftYellowBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": LeftYellowBox[:3] + np.array([0.02, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_transition.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'blue_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                left_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                left_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                right_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, -0.3, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": LeftBlueBox[:3] + np.array([0, -0.04, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": LeftBlueBox[:3] + np.array([0, -0.04, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": LeftBlueBox[:3] + np.array([0, -0.02, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": LeftBlueBox[:3] + np.array([0.02, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_transition.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            elif 'green_block' in config['task_name']:
                gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
                gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=180)
                left_arm_transition = gripper_pick_quat * Quaternion(axis=[.0, 1.0, .0], degrees=-90)
                left_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=90)
                right_arm_quat_meet = left_arm_transition * Quaternion(axis=[1.0, .0, .0], degrees=-90)
                right_arm_quat_meet = right_arm_quat_meet * Quaternion(axis=[.0, .0, 1.0], degrees=90)
                self.right_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
                    {"t": 220, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 250, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 300, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 320, "xyz": meet_xyz + np.array([0, -0.15, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 360, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 390, "xyz": meet_xyz + np.array([0, -0.13, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 410, "xyz": meet_xyz + np.array([0, -0.3, 0.08]), "quat": right_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 450, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 255},
                ]
                self.left_trajectory = [
                    {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                    {"t": 80, "xyz": LeftGreenBox[:3] + np.array([0, -0.01, 0.25]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 120, "xyz": LeftGreenBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 0},
                    {"t": 150, "xyz": LeftGreenBox[:3] + np.array([0, -0.01, 0.12]), "quat": gripper_pick_quat.elements,
                     "gripper": 255},
                    {"t": 190, "xyz": LeftGreenBox[:3] + np.array([0.02, 0.04, 0.4]),
                     "quat": gripper_pick_quat.elements, "gripper": 255},
                    {"t": 220, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_transition.elements,
                     "gripper": 255},
                    {"t": 250, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 280, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 320, "xyz": meet_xyz + np.array([0, 0.15, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 360, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 255},
                    {"t": 390, "xyz": meet_xyz + np.array([0, 0.13, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 405, "xyz": meet_xyz + np.array([0, 0.3, 0.08]), "quat": left_arm_quat_meet.elements,
                     "gripper": 0},
                    {"t": 450, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # sleep
                ]
            else:
                NotImplementedError

        else:
            NotImplementedError



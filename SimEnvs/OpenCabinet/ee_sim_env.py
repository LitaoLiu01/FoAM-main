import numpy as np
import os
import collections

from constants import XML_DIR, DT
from utils import sample_box_pose, sample_cabinet_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython

e = IPython.embed


def make_ee_sim_env(task_name, config):
    """
    动作空间：[left_mocap_pos(7),   #左手末端位置与姿态（四元数）
             left_gripper_positions,     #左手手爪开度(0~1)
             right_mocap_pos(7),   #右手末端位置与姿态（四元数）
             right_gripper_positions,    #右手手爪开度(0~1)   
    ]

    观测空间：{"qpos": Concat[ left_arm_qpos (6),          # 左臂关节角度
                             left_gripper_position (1),  # 左手手爪开度(0~1)
                             right_arm_qpos (6),         # 右臂关节角度
                             right_gripper_qpos (1)]     # 右手手爪开度(0~1)
             "qvel": Concat[ left_arm_qvel (6),          # 左臂关节速度
                             left_gripper_velocity (1),  # 左手手爪速度
                             right_arm_qvel (6),         # 右臂关节速度
                             right_gripper_qvel (1)]     # 右手手爪速度
             "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_' in task_name:
        xml_path = os.path.join(XML_DIR, f'sim_ee_env.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferTask(config, random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env


class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]
        # mocap是一个动作捕捉，通过mocap控制机器人关节末端，数据结构为左手位置、左手四元数、右手位置、右手四元数
        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # left_pos = physics.named.data.xpos['left_link6']
        # left_quat = physics.named.data.xquat['left_link6']
        # print("now the pos and quat is", left_pos, left_quat)s
        # set gripper
        g_left_ctrl = action_left[7]
        g_right_ctrl = action_right[7]

        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = 0

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        left_start_pos = physics.named.data.qpos['left_link6']
        left_start_quat = physics.named.data.xquat['left_link6']
        print("left_start_pos is", left_start_pos, "left_start_quat is", left_start_quat)

        right_start_pos = physics.named.data.xpos['right_link6']
        right_start_quat = physics.named.data.xquat['right_link6']
        print("right_start_pos is", right_start_pos, "right_start_quat is", right_start_quat)

        np.copyto(physics.data.mocap_pos[0], [0.077, 0.72455, 0.56105])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], [0.077, -0.72334915, 0.68105])
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([0, 0])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        # # 这里的数组要跟自己的数据匹配
        # left_joint1 = physics.model.name2id('left_link1', 'joint')
        # left_joint2 = physics.model.name2id('left_link2', 'joint')
        # left_joint3 = physics.model.name2id('left_link3', 'joint')
        # left_joint4 = physics.model.name2id('left_link4', 'joint')
        # left_joint5 = physics.model.name2id('left_link5', 'joint')
        # left_joint6 = physics.model.name2id('left_link6', 'joint')
        # left_arm_qpos = np.concatenate(physics.data.qpos[left_joint1],
        #                                physics.data.qpos[left_joint2],
        #                                physics.data.qpos[left_joint3],
        #                                physics.data.qpos[left_joint4],
        #                                physics.data.qpos[left_joint5],
        #                                physics.data.qpos[left_joint6])
        # right_joint1 = physics.model.name2id('right_link1', 'joint')
        # right_joint2 = physics.model.name2id('right_link2', 'joint')
        # right_joint3 = physics.model.name2id('right_link3', 'joint')
        # right_joint4 = physics.model.name2id('right_link4', 'joint')
        # right_joint5 = physics.model.name2id('right_link5', 'joint')
        # right_joint6 = physics.model.name2id('right_link6', 'joint')
        # right_arm_qpos = np.concatenate(physics.data.qpos[right_joint1],
        #                                 physics.data.qpos[right_joint2],
        #                                 physics.data.qpos[right_joint3],
        #                                 physics.data.qpos[right_joint4],
        #                                 physics.data.qpos[right_joint5],
        #                                 physics.data.qpos[right_joint6])
        # left_actuator_index = physics.model.name2id('left-hand_fingers_actuator', 'actuator')
        # left_gripper_qpos = physics.data.ctrl[left_actuator_index]
        # right_actuator_index = physics.model.name2id('right-hand_fingers_actuator', 'actuator')
        # right_gripper_qpos = physics.data.ctrl[right_actuator_index]
        # 获取左臂关节位置
        left_arm_qpos = np.array([physics.data.qpos[physics.model.name2id(f'left_link{i}', 'joint')]
                                  for i in range(1, 7)])
        # 获取右臂关节位置
        right_arm_qpos = np.array([physics.data.qpos[physics.model.name2id(f'right_link{i}', 'joint')]
                                   for i in range(1, 7)])
        # 获取左右手执行器的控制值
        left_gripper_qpos = np.array(
            [physics.data.ctrl[physics.model.name2id('left-hand_fingers_actuator', 'actuator')]])
        right_gripper_qpos = np.array(
            [physics.data.ctrl[physics.model.name2id('right-hand_fingers_actuator', 'actuator')]])
        # 合并所有位置和控制值
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        # 获取左臂关节速度
        left_arm_qvel = np.array([physics.data.qvel[physics.model.name2id(f'left_link{i}', 'joint')]
                                  for i in range(1, 7)])
        # 获取右臂关节速度
        right_arm_qvel = np.array([physics.data.qvel[physics.model.name2id(f'right_link{i}', 'joint')]
                                   for i in range(1, 7)])

        return np.concatenate([left_arm_qvel, right_arm_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)  # 我们少一个手爪速度
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['head'] = physics.render(height=480, width=640, camera_id='head')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferTask(BimanualViperXEETask):
    def __init__(self, config, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.config = config


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # initialize the task ---- open cabinet's drawer
        cabinet_pose = sample_cabinet_pose()
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        physics.data.qpos[cabinet_body_id:cabinet_body_id + 7] = cabinet_pose
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        # box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        # env_state = physics.data.qpos.copy()[box_start_idx: box_start_idx + 7]
        # print('now the box is located at',env_state)
        # return env_state
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 7]
        # print('now the box is located at', env_state)
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        if config['task_name'] == 'sim_open_cabinet_top_drawer':
            gripper_touch_handle = ("right-hand_right_pad1", "top_handle") in all_contact_pairs
        if config['task_name'] == 'sim_open_cabinet_middle_drawer':
            gripper_touch_handle = ("right-hand_right_pad1", "middle_handle") in all_contact_pairs
        if config['task_name'] == 'sim_open_cabinet_bottom_drawer':
            gripper_touch_handle = ("right-hand_right_pad1", "bottom_handle") in all_contact_pairs

        reward = 0
        if gripper_touch_handle:
            reward = 4
        print('The reward of task is',reward)
        return reward


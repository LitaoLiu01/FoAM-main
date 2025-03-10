import numpy as np
import os
import collections

from constants import XML_DIR, DT
from utils import sample_box_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from constants import DT, XML_DIR, START_ARM_POSE

import IPython

e = IPython.embed

Random_POSE = [None] # to be changed from outside

def make_sim_env(task_name, config):
    """
    动作空间：[left_arm_pos(6),   #左手关节位置
             left_gripper_positions,     #左手手爪开度(0~1)
             right_arm_pos(6),   #右手关节位置
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
        xml_path = os.path.join(XML_DIR, f'sim_env.xml')
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
        # left_arm_action = action[:6]
        # right_arm_action = action[7:7 + 6]
        # left_gripper_action = action[6]
        # right_gripper_action = action[7 + 6]
        # 获取关节角度和手爪开度
        env_action = action[:14]
        # env_action = np.concatenate([left_arm_action, left_gripper_action, right_arm_action, right_gripper_action])
        super().before_step(env_action, physics)
        # 这里是有问题的，记得改成自己的 np.copyto(physics.data.ctrl, np.array([left_arm_action, left_gripper_action,
        # right_arm_action, right_gripper_action]))
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
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
        # qvel_raw = physics.data.qvel.copy()
        # left_qvel_raw = qvel_raw[:7]
        # right_qvel_raw = qvel_raw[7:14]
        # left_arm_qvel = left_qvel_raw[:6]
        # right_arm_qvel = right_qvel_raw[:6]
        # left_gripper_qvel = left_qvel_raw[6]
        # right_gripper_qvel = right_qvel_raw[6]
        # return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

        # 获取左臂关节速度
        left_arm_qvel = np.array([physics.data.qvel[physics.model.name2id(f'left_link{i}', 'joint')]
                                  for i in range(1, 7)])

        # 获取右臂关节速度
        right_arm_qvel = np.array([physics.data.qvel[physics.model.name2id(f'right_link{i}', 'joint')]
                                   for i in range(1, 7)])

        left_gripper_qvel = np.array(
            [physics.data.ctrl[physics.model.name2id('left-hand_fingers_actuator', 'actuator')]])
        right_gripper_qvel = np.array(
            [physics.data.ctrl[physics.model.name2id('right-hand_fingers_actuator', 'actuator')]])
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

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
        obs['images']['front'] = physics.render(height=480, width=640, camera_id='front')
        obs['images']['angle_left'] = physics.render(height=480, width=640, camera_id='angle_left')


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
        # TODO Notice: this function does not randomize the env configuration. Instead, set Random_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:14] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert Random_POSE[0] is not None
            print(Random_POSE[0])
            cabinet_body_id = physics.model.name2id('locker_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 35] = Random_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('locker_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 35]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        all_contact_pairs = []
        reward = 0
        IsInHand = False
        IsInDrawer = False
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        # print(all_contact_pairs)
        if 'hammer' in config['task_name']:
            IsInHand = [("left-hand_right_pad2", f"hammer_visual") in all_contact_pairs]
            IsInDrawer = [("locker_visual", f"hammer_contact{i}") in all_contact_pairs for i in range(2)]
        elif 'camera' in config['task_name']:
            IsInHand = [("left-hand_right_pad2", f"camera_contact{i}") in all_contact_pairs for i in range(2)]
            IsInDrawer = [("locker_visual", f"camera_contact{i}") in all_contact_pairs for i in range(2)]
        elif 'green_stapler' in config['task_name']:
            IsInHand = [(f"green_stapler_visual", "left-hand_right_pad2") in all_contact_pairs]
            IsInDrawer = [("locker_visual", f"green_stapler_contact{i}") in all_contact_pairs for i in range(5)]
        elif 'black_stapler' in config['task_name']:
            IsInHand = [(f"black_stapler_visual", "left-hand_right_pad2") in all_contact_pairs]
            IsInDrawer = [("locker_visual", f"black_stapler_contact{i}") in all_contact_pairs for i in range(5)]
        else:
            NotImplementedError

        if any(IsInHand):
            reward = 2
        if any(IsInDrawer):
            reward = 4
        # print('The reward of task is', reward)
        return reward





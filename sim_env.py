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

BOX_POSE = [None] # to be changed from outside
ENV_POSE = [None]

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
    xml_path, task = get_xml_and_task_cls_sim_env(task_name, config)
    physics = mujoco.Physics.from_xml_path(xml_path)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)

    return env

def make_sim_env_goal_condition(task_name, config):
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

    xml_path, task = get_xml_and_task_cls_sim_env(task_name, config)
    physics = mujoco.Physics.from_xml_path(xml_path)
    env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)

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
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['head'] = physics.render(height=480, width=640, camera_id='head')

        return obs

    def get_reward(self, physics):
        raise NotImplementedError

class PickBlockTask(BimanualViperXEETask):
    def __init__(self, config, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.config = config

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:14] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('right_red_box_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 56] = ENV_POSE[0]

            print(f"{ENV_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('right_red_box_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 56]
        # print('now the box and mark is located at', env_state)
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        all_contact_pairs = []
        touch_right_gripper = 0
        reward = 0
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        if 'right_red' in config['task_name']:
            touch_right_gripper = ("right-hand_right_pad1", "right_red_box") in all_contact_pairs
        elif 'right_yellow' in config['task_name']:
            touch_right_gripper = ("right-hand_right_pad1", "right_yellow_box") in all_contact_pairs
        elif 'right_blue' in config['task_name']:
            touch_right_gripper = ("right-hand_right_pad1", "right_blue_box") in all_contact_pairs
        elif 'right_green' in config['task_name']:
            touch_right_gripper = ("right-hand_right_pad1", "right_green_box") in all_contact_pairs
        elif 'left_red' in config['task_name']:
            touch_right_gripper = ("left-hand_right_pad1", "left_red_box") in all_contact_pairs
        elif 'left_yellow' in config['task_name']:
            touch_right_gripper = ("left-hand_right_pad1", "left_yellow_box") in all_contact_pairs
        elif 'left_blue' in config['task_name']:
            touch_right_gripper = ("left-hand_right_pad1", "left_blue_box") in all_contact_pairs
        elif 'left_green' in config['task_name']:
            touch_right_gripper = ("left-hand_right_pad1", "left_green_box") in all_contact_pairs
        else:
            NotImplementedError

        if touch_right_gripper:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class CloseCabinetTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 10] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 10]
        # print('now the box is located at', env_state)
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        Issucess = False
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        cabinet_state = physics.data.qpos[cabinet_body_id:cabinet_body_id + 10]
        if 'bottom' in config['task_name']:
            Issucess = cabinet_state[-3] <= 0.015
        if 'middle' in config['task_name']:
            Issucess = cabinet_state[-2] <= 0.015
        if 'top' in config['task_name']:
            Issucess = cabinet_state[-1] <= 0.015

        reward = 0
        if Issucess:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class LeftArmPutStuff2DrawerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 38] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 38]
        # print('now the box is located at', env_state)
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
        if 'bottom' in config['task_name'] and 'apple' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "apple_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "apple_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'banana' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "banana_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "banana_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'green_bottle' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "green_bottle_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "green_bottle_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'blue_bottle' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "blue_bottle_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "blue_bottle_visual") in all_contact_pairs
        else:
            NotImplementedError

        if IsInHand:
            reward = 2
        if IsInDrawer:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class RightArmPutStuff2DrawerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 38] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 38]
        # print('now the box is located at', env_state)
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
        if 'bottom' in config['task_name'] and 'apple' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "apple_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "apple_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'banana' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "banana_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "banana_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'green_bottle' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "green_bottle_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "green_bottle_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'blue_bottle' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "blue_bottle_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "blue_bottle_visual") in all_contact_pairs
        else:
            NotImplementedError

        if IsInHand:
            reward = 2
        if IsInDrawer:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class DualArmPutStuff2DrawerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 38] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 38]
        # print('now the box is located at', env_state)
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
        if 'bottom' in config['task_name'] and 'apple' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "apple_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "apple_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'banana' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "banana_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "banana_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'green_bottle' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "green_bottle_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "green_bottle_visual") in all_contact_pairs
        if 'bottom' in config['task_name'] and 'blue_bottle' in config['task_name']:
            IsInHand = ("right-hand_right_pad1", "blue_bottle_visual") in all_contact_pairs
            IsInDrawer = ("drawer_bottom-27", "blue_bottle_visual") in all_contact_pairs
        else:
            NotImplementedError

        if IsInHand:
            reward = 2
        if IsInDrawer:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class OpenCabinetTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('cabinet_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 7] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
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

        joint_info = {}
        for i_joint in range(physics.model.njnt):
            # 获取 joint 名字
            joint_name = physics.model.id2name(i_joint, 'joint')
            # 获取 joint 的位置索引
            joint_qpos_id = physics.model.jnt_qposadr[i_joint]
            # 从 qpos 中获取 joint 的位置数值
            joint_value = physics.data.qpos[joint_qpos_id]
            # 记录 joint 名字和对应的值
            joint_info[joint_name] = joint_value

        reward = 0
        if config['task_name'] == 'sim_open_cabinet_top_drawer':
            value_joint = joint_info['joint_2']
            if value_joint >= 0.13:
                reward = 4
        if config['task_name'] == 'sim_open_cabinet_middle_drawer':
            value_joint = joint_info['joint_1']
            if value_joint >= 0.115:
                reward = 4
        if config['task_name'] == 'sim_open_cabinet_bottom_drawer':
            value_joint = joint_info['joint_0']
            if value_joint >= 0.115:
                reward = 4

        return reward

class PutStuff2BasketTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('basket_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 35] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('basket_joint', 'joint')
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
            IsInHand = [(f"left-hand_right_pad{i}", f"hammer_visual") in all_contact_pairs for i in range(3)]
            IsInDrawer = [(f"mug_contact{i}", 'hammer_visual') in all_contact_pairs for i in range(14)]
        elif 'camera' in config['task_name']:
            IsInHand = [(f"left-hand_right_pad{i}", "camera_visual") in all_contact_pairs for i in range(3)]
            IsInDrawer = [(f"mug_contact{i}", "camera_visual") in all_contact_pairs for i in range(14)]
        elif 'green_stapler' in config['task_name']:
            IsInHand = [(f"green_stapler_visual", "left-hand_right_pad2") in all_contact_pairs]
            IsInDrawer = [("green_stapler_visual", f"mug_contact{i}") in all_contact_pairs for i in range(14)]
        elif 'black_stapler' in config['task_name']:
            IsInHand = [(f"black_stapler_visual", "left-hand_right_pad2") in all_contact_pairs]
            IsInDrawer = [("black_stapler_visual", f"mug_contact{i}") in all_contact_pairs for i in range(14)]
        else:
            NotImplementedError

        if any(IsInHand):
            reward = 2
        if any(IsInDrawer):
            reward = 4
        # print('The reward of task is', reward)
        return reward

class PutStuff2LockerTopLayerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('locker_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 35] = ENV_POSE[0]
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

class PutStuff2PanTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('pan_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 35] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('pan_joint', 'joint')
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
        if 'duck' in config['task_name']:
            IsInHand = [("left-hand_right_pad2", f"duck_contact{i}") in all_contact_pairs for i in range(3)]
            IsInDrawer = [("fryingpan_visual", f"duck_contact{i}") in all_contact_pairs for i in range(3)]
        elif 'apple' in config['task_name']:
            IsInHand = [("left-hand_right_pad2", f"apple_contact{i}") in all_contact_pairs for i in range(2)]
            IsInDrawer = [("fryingpan_visual", f"apple_contact{i}") in all_contact_pairs for i in range(2)]
        elif 'pig' in config['task_name']:
            IsInHand = [("left-hand_right_pad2", f"pig_contact{i}") in all_contact_pairs for i in range(6)]
            IsInDrawer = [("fryingpan_visual", f"pig_contact{i}") in all_contact_pairs for i in range(6)]
        elif 'teapot' in config['task_name']:
            IsInHand = [("left-hand_right_pad2", f"teapot_contact{i}") in all_contact_pairs for i in range(41)]
            IsInDrawer = [("fryingpan_visual", f"teapot_contact{i}") in all_contact_pairs for i in range(41)]
        else:
            NotImplementedError

        if any(IsInHand):
            reward = 2
        if any(IsInDrawer):
            reward = 4
        # print('The reward of task is', reward)
        return reward

class PutStuff2LockerMiddleLayerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('hammer_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 35] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('hammer_joint', 'joint')
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
        if 'middle' in config['task_name']:
            if 'black_stapler' in config['task_name']:
                IsInHand = [(f"black_stapler_visual", "left-hand_right_pad2") in all_contact_pairs]
                IsInDrawer = [("locker_middle", f"black_stapler_contact{i}") in all_contact_pairs for i in range(5)]
            elif 'blue_block' in config['task_name']:
                IsInHand = [(f"blue_block", "left-hand_right_pad2") in all_contact_pairs]
                IsInDrawer = [("locker_middle", f"blue_block") in all_contact_pairs]
            else:
                NotImplementedError
        elif 'bottom' in config['task_name']:
            if 'black_stapler' in config['task_name']:
                IsInHand = [(f"black_stapler_visual", "left-hand_right_pad2") in all_contact_pairs]
                IsInDrawer = [("locker_bottom", f"black_stapler_contact{i}") in all_contact_pairs for i in range(5)]
            elif 'blue_block' in config['task_name']:
                IsInHand = [(f"blue_block", "left-hand_right_pad2") in all_contact_pairs]
                IsInDrawer = [("locker_bottom", f"blue_block") in all_contact_pairs]
            else:
                NotImplementedError
        else:
            NotImplementedError

        if any(IsInHand):
            reward = 2
        if any(IsInDrawer):
            reward = 4
        # print('The reward of task is', reward)
        # body_id = physics.model.name2id('locker_base', 'body')
        # print('locker_base', physics.data.xpos[body_id])
        return reward

class LeftArmBlockCoverSmallerMarkerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('left_red_box_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 56] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('left_red_box_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 56]
        # print('now the box is located at', env_state)
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        all_contact_pairs = []
        touch_right_gripper = False
        touch_table = False
        reward = 0
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        if 'upper_left_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_1', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_red_box") in all_contact_pairs
                touch_table = ("left_red_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_yellow_box") in all_contact_pairs
                touch_table = ("left_yellow_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_blue_box") in all_contact_pairs
                touch_table = ("right_blue_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_green_box") in all_contact_pairs
                touch_table = ("left_green_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        elif 'upper_right_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_2', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_red_box") in all_contact_pairs
                touch_table = ("left_red_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_yellow_box") in all_contact_pairs
                touch_table = ("left_yellow_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_blue_box") in all_contact_pairs
                touch_table = ("left_blue_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_green_box") in all_contact_pairs
                touch_table = ("left_green_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        elif 'bottom_right_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_3', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_red_box") in all_contact_pairs
                touch_table = ("left_red_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_yellow_box") in all_contact_pairs
                touch_table = ("left_yellow_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_blue_box") in all_contact_pairs
                touch_table = ("left_blue_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_green_box") in all_contact_pairs
                touch_table = ("left_green_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        elif 'bottom_left_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_4', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_red_box") in all_contact_pairs
                touch_table = ("left_red_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_yellow_box") in all_contact_pairs
                touch_table = ("left_yellow_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_blue_box") in all_contact_pairs
                touch_table = ("left_blue_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "left_green_box") in all_contact_pairs
                touch_table = ("left_green_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('left_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        else:
            NotImplementedError

        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if x_dis <= 0.02 and y_dis <=0.02 and touch_right_gripper:
            reward = 3
        if x_dis <= 0.02 and y_dis <=0.02 and not touch_right_gripper:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class RightArmBlockCoverSmallMarkerTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('right_red_box_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 56] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('right_red_box_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 56]
        # print('now the box is located at', env_state)
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        all_contact_pairs = []
        touch_right_gripper = False
        touch_table = False
        reward = 0
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        if 'upper_left_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_1', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_red_box") in all_contact_pairs
                touch_table = ("right_red_box", "table") in all_contact_pairs
                marker_touch = ('marker_1', "right_red_box") in all_contact_pairs
                box_id = physics.model.name2id('right_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_yellow_box") in all_contact_pairs
                touch_table = ("right_yellow_box", "table") in all_contact_pairs
                marker_touch = ('marker_1', "right_yellow_box") in all_contact_pairs
                box_id = physics.model.name2id('right_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_blue_box") in all_contact_pairs
                touch_table = ("right_blue_box", "table") in all_contact_pairs
                marker_touch = ('marker_1', "right_blue_box") in all_contact_pairs
                box_id = physics.model.name2id('right_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_green_box") in all_contact_pairs
                touch_table = ("right_green_box", "table") in all_contact_pairs
                marker_touch = ('marker_1', "right_green_box") in all_contact_pairs
                box_id = physics.model.name2id('right_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        elif 'upper_right_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_2', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_red_box") in all_contact_pairs
                touch_table = ("right_red_box", "table") in all_contact_pairs
                marker_touch = ('marker_2', "right_red_box") in all_contact_pairs
                box_id = physics.model.name2id('right_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_yellow_box") in all_contact_pairs
                touch_table = ("right_yellow_box", "table") in all_contact_pairs
                marker_touch = ('marker_2', "right_yellow_box") in all_contact_pairs
                box_id = physics.model.name2id('right_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_blue_box") in all_contact_pairs
                touch_table = ("right_blue_box", "table") in all_contact_pairs
                marker_touch = ('marker_2', "right_blue_box") in all_contact_pairs
                box_id = physics.model.name2id('right_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_green_box") in all_contact_pairs
                touch_table = ("right_green_box", "table") in all_contact_pairs
                marker_touch = ('marker_2', "right_green_box") in all_contact_pairs
                box_id = physics.model.name2id('right_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        elif 'bottom_right_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_3', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_red_box") in all_contact_pairs
                touch_table = ("right_red_box", "table") in all_contact_pairs
                marker_touch = ('marker_3', "right_red_box") in all_contact_pairs
                box_id = physics.model.name2id('right_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_yellow_box") in all_contact_pairs
                touch_table = ("right_yellow_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('right_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_blue_box") in all_contact_pairs
                touch_table = ("right_blue_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('right_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_green_box") in all_contact_pairs
                touch_table = ("right_green_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('right_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        elif 'bottom_left_marker' in config['task_name']:
            marker_1_id = physics.model.name2id('marker_4', 'body')
            marker_pos = physics.data.xpos[marker_1_id]
            if 'red_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_red_box") in all_contact_pairs
                touch_table = ("right_red_box", "table") in all_contact_pairs
                marker_touch = ('marker_4', "right_red_box") in all_contact_pairs
                box_id = physics.model.name2id('right_red_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'yellow_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_yellow_box") in all_contact_pairs
                touch_table = ("right_yellow_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('right_yellow_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'blue_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_blue_box") in all_contact_pairs
                touch_table = ("right_blue_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('right_blue_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            elif 'green_box' in config['task_name']:
                touch_right_gripper = ("right-hand_right_pad1", "right_green_box") in all_contact_pairs
                touch_table = ("right_green_box", "table") in all_contact_pairs
                box_id = physics.model.name2id('right_green_box', 'body')
                box_pos = physics.data.xpos[box_id]
                x_dis = abs(box_pos[0] - marker_pos[0])
                y_dis = abs(box_pos[1] - marker_pos[1])
            else:
                NotImplementedError
        else:
            NotImplementedError

        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if x_dis <= 0.02 and y_dis <=0.02 and touch_right_gripper:
            reward = 3
        if x_dis <= 0.02 and y_dis <=0.02 and not touch_right_gripper:
            reward = 4
        # print('The reward of task is', reward)
        return reward

class TransferBlockTask(BimanualViperXEETask):
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
            assert ENV_POSE[0] is not None
            print(ENV_POSE[0])
            cabinet_body_id = physics.model.name2id('right_red_box_joint', 'joint')
            physics.named.data.qpos[cabinet_body_id: cabinet_body_id + 56] = ENV_POSE[0]
            # print(f"{Random_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        cabinet_body_id = physics.model.name2id('right_red_box_joint', 'joint')
        env_state = physics.data.qpos.copy()[cabinet_body_id: cabinet_body_id + 56]
        # print('now the box is located at', env_state)
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        config = self.config
        all_contact_pairs = []
        picked_by_pick_arm = False
        transfer_to_receive_arm = False
        reward = 0
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        if 'to_left' in config['task_name']:
            if 'red_block' in config['task_name']:
                picked_by_pick_arm = ("right-hand_right_pad1", "right_red_box") in all_contact_pairs
                transfer_to_receive_arm = ("left-hand_right_pad1", "right_red_box") in all_contact_pairs
            elif 'yellow_block' in config['task_name']:
                picked_by_pick_arm = ("right-hand_right_pad1", "right_yellow_box") in all_contact_pairs
                transfer_to_receive_arm = ("left-hand_right_pad1", "right_yellow_box") in all_contact_pairs
            elif 'blue_block' in config['task_name']:
                picked_by_pick_arm = ("right-hand_right_pad1", "right_blue_box") in all_contact_pairs
                transfer_to_receive_arm = ("left-hand_right_pad1", "right_blue_box") in all_contact_pairs
            elif 'green_block' in config['task_name']:
                picked_by_pick_arm = ("right-hand_right_pad1", "right_green_box") in all_contact_pairs
                transfer_to_receive_arm = ("left-hand_right_pad1", "right_green_box") in all_contact_pairs
            else:
                NotImplementedError

        elif 'to_right' in config['task_name']:
            if 'red_block' in config['task_name']:
                picked_by_pick_arm = ("left-hand_right_pad1", "left_red_box") in all_contact_pairs
                transfer_to_receive_arm = ("right-hand_right_pad1", "left_red_box") in all_contact_pairs
            elif 'yellow_block' in config['task_name']:
                picked_by_pick_arm = ("left-hand_right_pad1", "left_yellow_box") in all_contact_pairs
                transfer_to_receive_arm = ("right-hand_right_pad1", "left_yellow_box") in all_contact_pairs
            elif 'blue_block' in config['task_name']:
                picked_by_pick_arm = ("left-hand_right_pad1", "left_blue_box") in all_contact_pairs
                transfer_to_receive_arm = ("right-hand_right_pad1", "left_blue_box") in all_contact_pairs
            elif 'green_block' in config['task_name']:
                picked_by_pick_arm = ("left-hand_right_pad1", "left_green_box") in all_contact_pairs
                transfer_to_receive_arm = ("right-hand_right_pad1", "left_green_box") in all_contact_pairs
            else:
                NotImplementedError

        else:
            NotImplementedError

        if picked_by_pick_arm:
            reward = 2
        elif transfer_to_receive_arm:
            reward = 4
        # print('The reward of task is', reward)
        return reward

def get_xml_and_task_cls_sim_env(task_name, config):
    if 'sim_pick' in task_name and 'block' in task_name: # 过滤出pick block任务
        xml_path = os.path.join(XML_DIR, 'PickBlock/model/sim_env.xml')
        task = PickBlockTask(config, random=None)
    elif 'sim_close_cabinet' in task_name:
        xml_path = os.path.join(XML_DIR, 'CloseCabinet/model/sim_env.xml')
        task = CloseCabinetTask(config, random=None)
    elif 'sim_left_arm_put' in task_name and 'to_cabinet' in task_name:
        xml_path = os.path.join(XML_DIR, 'LeftArmPutStuff2Drawer/model/sim_env.xml')
        task = LeftArmPutStuff2DrawerTask(config, random=None)
    elif 'sim_right_arm_put' in task_name and 'to_cabinet' in task_name:
        xml_path = os.path.join(XML_DIR, 'RightArmPutStuff2Drawer/model/sim_env.xml')
        task = RightArmPutStuff2DrawerTask(config, random=None)
    elif 'sim_dual_arm_put' in task_name and 'to_cabinet' in task_name:
        xml_path = os.path.join(XML_DIR, 'DualArmPutStuff2BottomDrawer/model/sim_env.xml')
        task = DualArmPutStuff2DrawerTask(config, random=False)
    elif 'sim_open_cabinet' in task_name:
        xml_path = os.path.join(XML_DIR, 'OpenCabinet/model/sim_env.xml')
        task = OpenCabinetTask(config, random=False)
    elif 'sim_put' in task_name and 'basket' in task_name:
        xml_path = os.path.join(XML_DIR, 'PutStuff2Basket/model/sim_env.xml')
        task = PutStuff2BasketTask(config, random=False)
    elif 'sim_put' in task_name and 'locker_top_layer' in task_name:
        xml_path = os.path.join(XML_DIR, 'PutStuff2LockerTopLayer/model/sim_env.xml')
        task = PutStuff2LockerTopLayerTask(config, random=False)
    elif 'sim_put' in task_name and 'pan' in task_name:
        xml_path = os.path.join(XML_DIR, 'PutStuff2Pan/model/sim_env.xml')
        task = PutStuff2PanTask(config, random=False)
    elif 'sim_put' in task_name and 'locker_middle_layer' in task_name:
        xml_path = os.path.join(XML_DIR, 'PutStuff2LockerMiddleLayer/model/sim_env.xml')
        task = PutStuff2LockerMiddleLayerTask(config, random=False)
    elif 'sim_left_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        xml_path = os.path.join(XML_DIR, 'LeftArmBlockCoverSmallerMarker/model/sim_env.xml')
        task = LeftArmBlockCoverSmallerMarkerTask(config, random=False)
    elif 'sim_right_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        xml_path = os.path.join(XML_DIR, 'RightArmBlockCoverSmallerMarker/model/sim_env.xml')
        task = RightArmBlockCoverSmallMarkerTask(config, random=False)
    elif 'sim_transfer' in task_name and 'block' in task_name:
        xml_path = os.path.join(XML_DIR, 'TransferBlock/model/sim_env.xml')
        task = TransferBlockTask(config, random=False)
    else:
        raise NotImplementedError

    return xml_path, task

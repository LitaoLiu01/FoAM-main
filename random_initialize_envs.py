### env utils
import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

e = IPython.embed

def initialize_envs(task_name):
    if 'sim_pick' in task_name and 'block' in task_name:
        initial_pose_of_env = PickBlockTaskInitEnv(task_name)  # used in sim reset
    elif 'sim_close_cabinet' in task_name:
        initial_pose_of_env = CloseCabinetDrawerTaskInitEnv(task_name)
    elif 'sim_left_arm_put' in task_name and 'to_cabinet' in task_name:
        initial_pose_of_env = LeftArmPutStuff2DrawerTaskInitEnv(task_name)
    elif 'sim_right_arm_put' in task_name and 'to_cabinet' in task_name:
        initial_pose_of_env = RightArmPutStuff2DrawerTaskInitEnv(task_name)
    elif 'sim_dual_arm_put' in task_name and 'to_cabinet' in task_name:
        initial_pose_of_env = DualArmPutStuff2DrawerTaskInitEnv(task_name)
    elif 'sim_open_cabinet' in task_name:
        initial_pose_of_env = OpenCabinetTaskInitEnv(task_name)
    elif 'sim_put' in task_name and 'basket' in task_name:
        initial_pose_of_env = PutStuff2BasketTaskInitEnv(task_name)
    elif 'sim_put' in task_name and 'locker_top_layer' in task_name:
        initial_pose_of_env = PutStuff2LockerTopLayerTaskInitEnv(task_name)
    elif 'sim_put' in task_name and 'pan' in task_name:
        initial_pose_of_env = PutStuff2PanTaskInitEnv(task_name)
    elif 'sim_put' in task_name and 'locker_middle_layer' in task_name:
        initial_pose_of_env = PutStuff2LockerMiddleLayerTaskInitEnv(task_name)
    elif 'sim_left_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        initial_pose_of_env = LeftArmBlockCoverSmallerMarkerTaskInitEnv(task_name)
    elif 'sim_right_arm' in task_name and 'cover' in task_name and 'marker' in task_name:
        initial_pose_of_env = RightArmBlockCoverSmallerMarkerTaskInitEnv(task_name)
    elif 'sim_transfer' in task_name and 'block' in task_name:
        initial_pose_of_env = TransferBlockTaskInitEnv(task_name)
    else:
        NotImplementedError

    return initial_pose_of_env

# Pcik Block Task
def PickBlockTaskInitEnv(task_name):
    RightBoxCenterPosition = PickBlock_SampleRightBlockCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    RightRedBox = RightBoxCenterPosition.copy()
    RightRedBox[0] += 0.05
    RightRedBox[1] += 0.05
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightYellowBox = RightBoxCenterPosition.copy()
    RightYellowBox[0] -= 0.05
    RightYellowBox[1] -= 0.05
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightBlueBox = RightBoxCenterPosition.copy()
    RightBlueBox[0] += 0.05
    RightBlueBox[1] -= 0.05
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightGreenBox = RightBoxCenterPosition.copy()
    RightGreenBox[0] -= 0.05
    RightGreenBox[1] += 0.05

    LeftBoxCenterPosition = PickBlock_LeftRightBlockCenterPosition()
    LeftRedBox = LeftBoxCenterPosition.copy()
    LeftRedBox[0] -= 0.05
    LeftRedBox[1] += 0.05
    LeftYellowBox = LeftBoxCenterPosition.copy()
    LeftYellowBox[0] += 0.05
    LeftYellowBox[1] -= 0.05
    LeftBlueBox = LeftBoxCenterPosition.copy()
    LeftBlueBox[0] += 0.05
    LeftBlueBox[1] += 0.05
    LeftGreenBox = LeftBoxCenterPosition.copy()
    LeftGreenBox[0] -= 0.05
    LeftGreenBox[1] -= 0.05

    AllBoxPose = np.concatenate((RightRedBox, RightYellowBox, RightBlueBox, RightGreenBox, LeftRedBox, LeftYellowBox,
                                 LeftBlueBox, LeftGreenBox))

    return AllBoxPose

def PickBlock_SampleRightBlockCenterPosition():
    x_range = [0.4, 0.47]
    y_range = [-0.33, -0.23]
    z_range = [0.08, 0.08]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def PickBlock_LeftRightBlockCenterPosition():
    x_range = [0.4, 0.47]
    y_range = [0.23, 0.33]
    z_range = [0.08, 0.08]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# Initialize Close Cabinet Drawer Task
def CloseCabinetDrawerTaskInitEnv(task_name):
    x_range = [0.78, 0.88]
    # x_range = [0.78, 0.88]
    y_range = [-0.53, -0.33]
    # y_range = [-0.53, -0.33]
    z_range = [0.32, 0.32]
    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    drawer_joints = None
    cabinet_quat = np.array([1, 0, 0, 0])
    if 'top' in task_name:
        drawer_joints = np.array([0, 0, 0.15])
    elif 'middle' in task_name:
        drawer_joints = np.array([0, 0.15, 0])
    elif 'bottom' in task_name:
        drawer_joints = np.array([0.12, 0, 0])

    return np.concatenate([cabinet_position, cabinet_quat, drawer_joints])

# Initialize Left Arm Put Stuff 2 Drawer
def LeftArmPutStuff2DrawerTaskInitEnv(task_name):
    CabinetPosition = LeftArmPutStuff2Drawer_SapleCabinetPose()
    drawer_joints = np.array([0, 0, 0])
    if 'bottom_drawer' in task_name:
        drawer_joints = drawer_joints + np.array([0.2, 0, 0])
    elif 'middle_drawer' in task_name:
        drawer_joints = drawer_joints + np.array([0, 0.15, 0])
    else:
        NotImplementedError
    CabinetState = np.concatenate((CabinetPosition, drawer_joints))
    RightStuffCenterPosition = LeftArmPutStuff2Drawer_SampleRightBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    Banana = RightStuffCenterPosition.copy()
    Banana[0] += 0.05
    Banana[1] += 0.05
    Banana[2] = 0.08
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Bottle = RightStuffCenterPosition.copy()
    Bottle[0] -= 0.08
    Bottle[1] -= 0.08
    Bottle[2] = 0.09
    Bottle[-4:] = np.array([0.5, -0.5, 0.5, 0.5])
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Cup = RightStuffCenterPosition.copy()
    Cup[0] += 0.05
    Cup[1] -= 0.05
    Cup[2] = 0.09
    Cup[-4:] = np.array([0.5, 0.5, 0.5, 0.5])
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Apple = RightStuffCenterPosition.copy()
    Apple[0] -= 0.05
    Apple[1] += 0.05
    Apple[2] = 0.1

    AllStuffPose = np.concatenate((CabinetState, Apple, Banana, Cup, Bottle))

    return AllStuffPose

def LeftArmPutStuff2Drawer_SampleRightBoxCenterPosition():
    x_range = [0.4, 0.45]
    y_range = [0.2, 0.3]
    z_range = [0.08, 0.08]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def LeftArmPutStuff2Drawer_SapleCabinetPose():
    x_range = [0.78, 0.88]
    # x_range = [0.78, 0.88]
    y_range = [-0.25, -0.20]
    # y_range = [0.2, 0.25]
    z_range = [0.32, 0.32]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])

# Initialize Right Arm Put Stuff 2 Drawer
def RightArmPutStuff2DrawerTaskInitEnv(task_name):
    CabinetPosition = RightArmPutStuff2Drawer_SapleCabinetPose()
    drawer_joints = np.array([0, 0, 0])
    if 'bottom_drawer' in task_name:
        drawer_joints = drawer_joints + np.array([0.2, 0, 0])
    elif 'middle_drawer' in task_name:
        drawer_joints = drawer_joints + np.array([0, 0.15, 0])
    else:
        NotImplementedError
    CabinetState = np.concatenate((CabinetPosition, drawer_joints))
    RightStuffCenterPosition = RightArmPutStuff2Drawer_SampleRightBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    Banana = RightStuffCenterPosition.copy()
    Banana[0] += 0.05
    Banana[1] += 0.05
    Banana[2] = 0.08
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Bottle = RightStuffCenterPosition.copy()
    Bottle[0] -= 0.08
    Bottle[1] -= 0.08
    Bottle[2] = 0.09
    Bottle[-4:] = np.array([0.5, 0.5, -0.5, 0.5])
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Cup = RightStuffCenterPosition.copy()
    Cup[0] += 0.05
    Cup[1] -= 0.05
    Cup[2] = 0.09
    Cup[-4:] = np.array([0.5, 0.5, 0.5, 0.5])
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Apple = RightStuffCenterPosition.copy()
    Apple[0] -= 0.05
    Apple[1] += 0.05
    Apple[2] = 0.1

    AllStuffPose = np.concatenate((CabinetState, Apple, Banana, Cup, Bottle))

    return AllStuffPose

def RightArmPutStuff2Drawer_SampleRightBoxCenterPosition():
    x_range = [0.4, 0.45]
    y_range = [-0.3, -0.2]
    z_range = [0.08, 0.08]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def RightArmPutStuff2Drawer_SapleCabinetPose():
    x_range = [0.78, 0.88]
    # x_range = [0.78, 0.88]
    y_range = [0.20, 0.25]
    # y_range = [0.2, 0.25]
    z_range = [0.32, 0.32]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])

# Initialize dual Arm Put Stuff 2 Drawer

def DualArmPutStuff2DrawerTaskInitEnv(task_name):
    CabinetPosition = DualArmPutStuff2Drawer_SapleCabinetPose()
    drawer_joints = np.array([0, 0, 0])
    if 'bottom_drawer' in task_name:
        drawer_joints = drawer_joints + np.array([0, 0, 0])
    elif 'middle_drawer' in task_name:
        drawer_joints = drawer_joints + np.array([0, 0., 0])
    else:
        NotImplementedError
    CabinetState = np.concatenate((CabinetPosition, drawer_joints))
    RightStuffCenterPosition = DualArmPutStuff2Drawer_SampleRightBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    Banana = RightStuffCenterPosition.copy()
    Banana[0] += 0.05
    Banana[1] += 0.05
    Banana[2] = 0.08
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Bottle = RightStuffCenterPosition.copy()
    Bottle[0] -= 0.08
    Bottle[1] -= 0.08
    Bottle[2] = 0.09
    Bottle[-4:] = np.array([0.5, -0.5, 0.5, 0.5])
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Cup = RightStuffCenterPosition.copy()
    Cup[0] += 0.05
    Cup[1] -= 0.05
    Cup[2] = 0.09
    Cup[-4:] = np.array([0.5, 0.5, 0.5, 0.5])
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Apple = RightStuffCenterPosition.copy()
    Apple[0] -= 0.05
    Apple[1] += 0.05
    Apple[2] = 0.1

    AllStuffPose = np.concatenate((CabinetState, Apple, Banana, Cup, Bottle))

    return AllStuffPose

def DualArmPutStuff2Drawer_SapleCabinetPose():
    x_range = [0.78, 0.84]
    # x_range = [0.78, 0.82, 0.84]
    y_range = [-0.25, -0.2]
    #     y_range = [-0.25, -0.2]
    z_range = [0.32, 0.32]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])

def DualArmPutStuff2Drawer_SampleRightBoxCenterPosition():
    x_range = [0.4, 0.47]
    y_range = [0.2, 0.3]
    z_range = [0.08, 0.08]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# Initialize Open Cabinet drawer
def OpenCabinetTaskInitEnv(task_name):
    x_range = [0.73, 0.83]
    # x_range = [0.73, 0.83]
    y_range = [-0.45, -0.35]
    # y_range = [-0.45, -0.35]
    z_range = [0.32, 0.32]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cabinet_quat = np.array([1, 0, 0, 0])

    return np.concatenate([cabinet_position, cabinet_quat])

# Initialize PutStuff2Basket
def PutStuff2BasketTaskInitEnv(task_name):
    BasketPose = PutStuff2Basket_SampleBasketPose()
    LeftStuffCenterPosition = PutStuff2Basket_SampleLeftBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    Hammer = LeftStuffCenterPosition.copy()
    Hammer[0] -= 0.05
    Hammer[1] -= 0.05
    Hammer[2] = 0.13
    Hammer[-4:] = [0, -0.7071, 0.7071, 0]
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Camera = LeftStuffCenterPosition.copy()
    Camera[0] += 0.05
    Camera[1] += 0.07
    Camera[2] = 0.14
    Camera[-4:] = [0, 0, 0, 1]
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Toothpaste = LeftStuffCenterPosition.copy()
    Toothpaste[0] -= 0.07
    Toothpaste[1] += 0.07
    Toothpaste[2] = 0.13
    Toothpaste[-4:] = [1, 0, 0, 0]

    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Stapler = LeftStuffCenterPosition.copy()
    Stapler[0] += 0.06
    Stapler[1] -= 0.05
    Stapler[2] = 0.14
    Stapler[-4:] = [0.5, 0.5, 0.5, 0.5]

    AllStuffPose = np.concatenate((BasketPose, Hammer, Camera, Toothpaste, Stapler))

    return AllStuffPose

def PutStuff2Basket_SampleBasketPose():
    x_range = [0.55, 0.65]
    # x_range = [0.55, 0.65]
    y_range = [-0.05, 0.05]
    # y_range = [-0.05, 0.05]
    z_range = [0.18, 0.18]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    LockerQuat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, LockerQuat])

def PutStuff2Basket_SampleLeftBoxCenterPosition():
    x_range = [0.4, 0.5]
    y_range = [0.17, 0.27]
    z_range = [0.13, 0.13]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# Initialize PutStuff2LockerTopLayer
def PutStuff2LockerTopLayerTaskInitEnv(task_name):
    LockerPose = PutStuff2LockerTopLayer_SampleLockerPose()
    LeftStuffCenterPosition = PutStuff2LockerTopLayer_SampleLeftBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    Hammer = LeftStuffCenterPosition.copy()
    Hammer[0] -= 0.05
    Hammer[1] -= 0.05
    Hammer[2] = 0.13
    Hammer[-4:] = [0, -0.7071, 0.7071, 0]
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Camera = LeftStuffCenterPosition.copy()
    Camera[0] += 0.05
    Camera[1] -= 0.08
    Camera[2] = 0.14
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Toothpaste = LeftStuffCenterPosition.copy()
    Toothpaste[0] += 0.07
    Toothpaste[1] += 0.07
    Toothpaste[2] = 0.13
    Toothpaste[-4:] = [0.5, 0.5, 0.5, 0.5]

    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Stapler = LeftStuffCenterPosition.copy()
    Stapler[0] -= 0.09
    Stapler[1] += 0.09
    Stapler[2] = 0.14

    AllStuffPose = np.concatenate((LockerPose, Hammer, Camera, Toothpaste, Stapler))

    return AllStuffPose

def PutStuff2LockerTopLayer_SampleLockerPose():
    x_range = [0.55, 0.65]
    # x_range = [0.4, 0.5]
    y_range = [-0.15, -0.05]
    # y_range = [-0.15, -0.05]
    z_range = [0.18, 0.18]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    LockerQuat = np.array([0, 0.9239, 0.3827, 0])
    return np.concatenate([cabinet_position, LockerQuat])

def PutStuff2LockerTopLayer_SampleLeftBoxCenterPosition():
    x_range = [0.4, 0.5]
    y_range = [0.17, 0.27]
    z_range = [0.13, 0.13]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


# Initialize PutStuff2PanTaskInitEnv
def PutStuff2PanTaskInitEnv(task_name):
    PanPose = PutStuff2Pan_SamplePanPose()
    LeftStuffCenterPosition = PutStuff2Pan_SampleLeftBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    Pig = LeftStuffCenterPosition.copy()
    Pig[0] += 0.05
    Pig[1] += 0.05
    Pig[2] = 0.08
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Duck = LeftStuffCenterPosition.copy()
    Duck[0] -= 0.07
    Duck[1] -= 0.07
    Duck[2] = 0.085
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Apple = LeftStuffCenterPosition.copy()
    Apple[0] += 0.05
    Apple[1] -= 0.05
    Apple[2] = 0.085
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Teapot = LeftStuffCenterPosition.copy()
    Teapot[0] -= 0.05
    Teapot[1] += 0.05
    Teapot[2] = 0.08

    AllStuffPose = np.concatenate((PanPose, Duck, Apple, Pig, Teapot))

    return AllStuffPose

def PutStuff2Pan_SamplePanPose():
    x_range = [0.4, 0.5]
    # x_range = [0.78, 0.88]
    y_range = [-0.05, 0.05]
    # y_range = [0.2, 0.25]
    z_range = [0.075, 0.075]

    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    PanQuat = np.array([0, 0, 0, 1])
    return np.concatenate([cabinet_position, PanQuat])

def PutStuff2Pan_SampleLeftBoxCenterPosition():
    x_range = [0.4, 0.5]
    y_range = [0.15, 0.25]
    z_range = [0.08, 0.08]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# Initialize PutStuff2LockerMiddleLayerTaskInitEnv
def PutStuff2LockerMiddleLayerTaskInitEnv(task_name):
    LeftStuffCenterPosition = PutStuff2LockerMiddleLayer_SampleLeftBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变

    Hammer = LeftStuffCenterPosition.copy()
    Hammer[0] -= 0.05
    Hammer[1] -= 0.05
    Hammer[2] = 0.13
    Hammer[-4:] = [1, 0, 0, 0]
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Camera = LeftStuffCenterPosition.copy()
    Camera[0] += 0.05
    Camera[1] -= 0.08
    Camera[2] = 0.14
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Toothpaste = LeftStuffCenterPosition.copy()
    Toothpaste[0] += 0.07
    Toothpaste[1] += 0.07
    Toothpaste[2] = 0.13
    Toothpaste[-4:] = [0.5, 0.5, 0.5, 0.5]

    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    Stapler = LeftStuffCenterPosition.copy()
    Stapler[0] -= 0.07
    Stapler[1] += 0.03
    Stapler[2] = 0.14

    BlueBlock = LeftStuffCenterPosition.copy()
    BlueBlock[0] = Stapler[0] + 0.025
    BlueBlock[1] = Stapler[1] + 0.09
    BlueBlock[2] = 0.14
    BlueBlock[-4:] = [0.5, 0.5, 0.5, 0.5]

    AllStuffPose = np.concatenate((Hammer, Camera, Toothpaste, Stapler, BlueBlock))

    return AllStuffPose

def PutStuff2LockerMiddleLayer_SampleLeftBoxCenterPosition():
    x_range = [0.4, 0.5]
    # x_range = [0.4, 0.5]
    y_range = [0.23, 0.30]
    # y_range = [0.23, 0.30]
    z_range = [0.13, 0.13]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# Initialize LeftArmBlockCoverSmallerMarkerTaskInitEnv
def LeftArmBlockCoverSmallerMarkerTaskInitEnv(task_name):
    RightBoxCenterPosition = LeftArmBlockCoverSmallerMarker_SampleRightBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    MarkerPose = LeftArmBlockCoverSmallerMarker_SampleMarkerCenter()

    RightRedBox = RightBoxCenterPosition.copy()
    RightRedBox[0] -= 0.05
    RightRedBox[1] += 0.05
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightYellowBox = RightBoxCenterPosition.copy()
    RightYellowBox[0] += 0.05
    RightYellowBox[1] += 0.05
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightBlueBox = RightBoxCenterPosition.copy()
    RightBlueBox[0] += 0.05
    RightBlueBox[1] -= 0.05
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightGreenBox = RightBoxCenterPosition.copy()
    RightGreenBox[0] -= 0.05
    RightGreenBox[1] -= 0.05

    # marker_1
    Marker_1 = MarkerPose.copy()
    Marker_1[0] += 0.0375
    Marker_1[1] += 0.0375
    # marker_2
    Marker_2 = MarkerPose.copy()
    Marker_2[0] += 0.0375
    Marker_2[1] -= 0.0375
    # marker_3
    Marker_3 = MarkerPose.copy()
    Marker_3[0] -= 0.0375
    Marker_3[1] -= 0.0375
    # marker_4
    Marker_4 = MarkerPose.copy()
    Marker_4[0] -= 0.0375
    Marker_4[1] += 0.0375

    AllBoxPose = np.concatenate(
        (RightRedBox, RightYellowBox, RightBlueBox, RightGreenBox, Marker_1, Marker_2, Marker_3, Marker_4))

    return AllBoxPose

def LeftArmBlockCoverSmallerMarker_SampleMarkerCenter():
    x_range = [0.42, 0.47]
    # 小于0.42需要修改np.array([0.01, 0.0, 0.3])为np.array([0.0, 0.0, 0.3])
    # x_range = [0.42, 0.47]
    y_range = [0.01, 0.11]
    #     y_range = [0.01, 0.11]
    z_range = [0.06, 0.06]
    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])

def LeftArmBlockCoverSmallerMarker_SampleRightBoxCenterPosition():
    x_range = [0.40, 0.45]
    # x_range = [0.40, 0.45]
    y_range = [0.23, 0.33]
    # y_range = [0.23, 0.33]
    z_range = [0.08, 0.08]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

# Initialize RightArmBlockCoverSmallerMarkerTaskInitEnv
def RightArmBlockCoverSmallerMarkerTaskInitEnv(task_name):
    # initialize the task ---- open cabinet's drawer
    RightBoxCenterPosition = RightArmBlockCoverSmallerMarker_SampleRightBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    MarkerPose = RightArmBlockCoverSmallerMarker_SampleMarkerCenter()

    RightRedBox = RightBoxCenterPosition.copy()
    RightRedBox[0] -= 0.05
    RightRedBox[1] -= 0.05
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightYellowBox = RightBoxCenterPosition.copy()
    RightYellowBox[0] += 0.05
    RightYellowBox[1] -= 0.05
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightBlueBox = RightBoxCenterPosition.copy()
    RightBlueBox[0] += 0.05
    RightBlueBox[1] += 0.05
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightGreenBox = RightBoxCenterPosition.copy()
    RightGreenBox[0] -= 0.05
    RightGreenBox[1] += 0.05

    # marker_1
    Marker_1 = MarkerPose.copy()
    Marker_1[0] += 0.0375
    Marker_1[1] += 0.0375
    # marker_2
    Marker_2 = MarkerPose.copy()
    Marker_2[0] += 0.0375
    Marker_2[1] -= 0.0375
    # marker_3
    Marker_3 = MarkerPose.copy()
    Marker_3[0] -= 0.0375
    Marker_3[1] -= 0.0375
    # marker_4
    Marker_4 = MarkerPose.copy()
    Marker_4[0] -= 0.0375
    Marker_4[1] += 0.0375

    AllBoxPose = np.concatenate(
        (RightRedBox, RightYellowBox, RightBlueBox, RightGreenBox, Marker_1, Marker_2, Marker_3, Marker_4))

    return AllBoxPose

def RightArmBlockCoverSmallerMarker_SampleRightBoxCenterPosition():
    x_range = [0.40, 0.45]
    # x_range = [0.40, 0.45]
    y_range = [-0.33, -0.23]
    # y_range = [-0.33, -0.23]
    z_range = [0.08, 0.08]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def RightArmBlockCoverSmallerMarker_SampleMarkerCenter():
    x_range = [0.42, 0.47]
    # 小于0.42需要修改np.array([0.01, 0.0, 0.3])为np.array([0.0, 0.0, 0.3])
    # x_range = [0.42, 0.47]
    y_range = [-0.11, -0.01]
    # y_range = [-0.11, -0.01]
    z_range = [0.06, 0.06]
    ranges = np.vstack([x_range, y_range, z_range])
    cabinet_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cabinet_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cabinet_position, cabinet_quat])


# Initialize TransferBlockTaskInitEnv
def TransferBlockTaskInitEnv(task_name):
    RightBoxCenterPosition = TransferBlock_SampleRightBoxCenterPosition()
    # RightRedBox 等于 BoxCenterPosition 的前两个数加 0.05，其他数保持不变
    RightRedBox = RightBoxCenterPosition.copy()
    RightRedBox[0] += 0.05
    RightRedBox[1] += 0.05
    # RightYellowBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightYellowBox = RightBoxCenterPosition.copy()
    RightYellowBox[0] -= 0.05
    RightYellowBox[1] -= 0.05
    # RightBlueBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightBlueBox = RightBoxCenterPosition.copy()
    RightBlueBox[0] += 0.05
    RightBlueBox[1] -= 0.05
    # RightGreenBox 第一个数加 0.05，第二个数减去 0.05，其他数保持不变
    RightGreenBox = RightBoxCenterPosition.copy()
    RightGreenBox[0] -= 0.05
    RightGreenBox[1] += 0.05

    LeftBoxCenterPosition = TransferBlock_SampleLeftBoxCenterPosition()
    LeftRedBox = LeftBoxCenterPosition.copy()
    LeftRedBox[0] -= 0.05
    LeftRedBox[1] += 0.05
    LeftYellowBox = LeftBoxCenterPosition.copy()
    LeftYellowBox[0] += 0.05
    LeftYellowBox[1] -= 0.05
    LeftBlueBox = LeftBoxCenterPosition.copy()
    LeftBlueBox[0] += 0.05
    LeftBlueBox[1] += 0.05
    LeftGreenBox = LeftBoxCenterPosition.copy()
    LeftGreenBox[0] -= 0.05
    LeftGreenBox[1] -= 0.05

    AllBoxPose = np.concatenate((RightRedBox, RightYellowBox, RightBlueBox, RightGreenBox, LeftRedBox, LeftYellowBox,
                                 LeftBlueBox, LeftGreenBox))

    return AllBoxPose

def TransferBlock_SampleRightBoxCenterPosition():
    x_range = [0.40, 0.47]
    # x_range = [0.40, 0.47]
    y_range = [-0.33, -0.23]
    # y_range = [-0.33, -0.23]
    z_range = [0.08, 0.08]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def TransferBlock_SampleLeftBoxCenterPosition():
    x_range = [0.40, 0.47]
    # x_range = [0.40, 0.45]
    y_range = [0.23, 0.33]
    # y_range = [0.23, 0.33]
    z_range = [0.08, 0.08]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

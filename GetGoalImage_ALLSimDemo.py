import os
import h5py
import numpy as np
import argparse

# 使用的时候需要根据数据修改最大步长 self.max_steps = 450 和数据目录
# self.directory = '/data/litao.liu/CASIA_Intern/ALL_DATA/data_FunctionTest'  # 替换为你的文件夹路径

class Args:
    def __init__(self):
        self.max_steps = 450
        self.goal_image_camera = 'head'
        # self.goal_image_frame = 660
        self.directory = '/data/litao.liu/CASIA_Intern/ALL_DATA/'  # 替换为你的文件夹路径

def main():
    args = Args()
    directory = args.directory
    count = 0
    # 使用示例
    subfolders = list_subfolders(directory)
    print(subfolders)
    for subfolder in subfolders:
        subfolder_path = os.path.join(directory, subfolder)
        sub_subfolders = list_subfolders(subfolder_path)
        print(sub_subfolders)
        for sub_subfolder in sub_subfolders:
            print(sub_subfolder)
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
            Hdf5Files = list_all_hdf5_files(sub_subfolder_path)
            print(f'Length of {sub_subfolder_path} folders HDF5Files: ', len(Hdf5Files))
            for Hdf5FileName in Hdf5Files:
                Hdf5File_path = os.path.join(sub_subfolder_path, Hdf5FileName)
                goal_image_camera = args.goal_image_camera
                goal_image_frame = args.max_steps  # it can be changed to self.goal_image_frame = 660
                goal_image = get_goal_image(Hdf5File_path, goal_image_frame, goal_image_camera)
                goal_images = np.tile(goal_image, (args.max_steps, 1, 1, 1))
                save_goal_image_to_hdf5(Hdf5File_path, goal_images)
                count += 1
                print(f'Finish the goal images record in {Hdf5FileName}, count={count}')

def list_subfolders(directory):
    subfolders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subfolders.append(item)
    return subfolders

def save_goal_image_to_hdf5(file_name, goal_images):
    with h5py.File(file_name, 'a') as hdf:
        # 创建'observation'组，如果不存在
        if 'observations' not in hdf:
            observation_group = hdf.create_group('observations')
        else:
            observation_group = hdf['observations']
        # 创建'image'组，如果不存在
        if 'images' not in observation_group:
            image_group = observation_group.create_group('images')
        else:
            image_group = observation_group['images']
        # 创建或覆盖'goal_images'数据集
        if 'goal_images' in image_group:
            del image_group['goal_images']  # 删除已存在的数据集
        image_group.create_dataset('goal_images', data=goal_images)

def get_goal_image(file_name, len_steps, goal_image_camera):
    with h5py.File(file_name, 'r') as hdf:
        if 'observations' in hdf:
            observation_group = hdf['observations']
            # 检查是否存在'image'组
            if 'images' in observation_group:
                image_group = observation_group['images']
                # 检查是否存在'front'数据集
                if goal_image_camera in image_group:
                    front_dataset = image_group[goal_image_camera]
                    # 读取并打印特定位置的数据
                    goal_image = front_dataset[len_steps - 1]
                else:
                    print("Dataset 'front' not found in 'image'.")
            else:
                print("Group 'image' not found in 'observation'.")
        else:
            print("Group 'observation' not found in the HDF5 file.")
    return goal_image

def list_all_hdf5_files(directory):
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(file)
    return hdf5_files


if __name__ == '__main__':
    main()


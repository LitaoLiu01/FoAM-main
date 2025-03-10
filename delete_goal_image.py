import h5py
import numpy as np
import os

# 打开已经存在的HDF5文件
class Args:
    def __init__(self):
        self.max_steps = 750
        self.data_dir = '/mnt/nas/personal/litaoliu/ACT_Goal_Demonstration_0525/data_mix20'
        self.goal_image_camera = 'angle'
        # self.goal_image_frame = 660

def main():
    args = Args()
    for i in range(50):
        # 打开一个存在的 HDF5 文件
        data_dir = args.data_dir
        file_name = os.path.join(data_dir, f'episode_{i}.hdf5')
        delete_goal_image(file_name)
        print(f'delete the goal images in episode_{i}.hdf5')


def delete_goal_image(file_name):
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

def get_goal_image(file_name, len_steps, goal_image_camera):
    with h5py.File(file_name, 'r') as hdf:
        # 列出所有的组
        # 列出所有的组
        # print("Keys: %s" % list(hdf.keys()))
        # 获取特定的数据集
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


if __name__ == '__main__':
    main()


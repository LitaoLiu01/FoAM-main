import subprocess
import os

# docker exec -it TrainandInfer /bin/bash
# cd MultiPolicyTrainandInfer_frozenbackbone_0826/

class Args:
    def __init__(self):
        self.ckpt_dir = f'/workspace/ckpt/FoAM_wo_MPH_SimALL_cs100_0820_last_goal_img_frozen_backbone/'
        self.log_prefix = f'FoAM_wo_MPH_SimALL_cs100_0820_last_goal_img_frozen_backbone'
        self.GPU = '4090'
        self.chunk_size = 100
        self.TASK_NAME = [
            'sim_pick_left_blue_block',
            'sim_pick_left_green_block',
            'sim_pick_left_red_block',
            'sim_pick_left_yellow_block',
            'sim_pick_right_blue_block',
            'sim_pick_right_green_block',
            'sim_pick_right_red_block',
            'sim_pick_right_yellow_block',

            'sim_close_cabinet_bottom_drawer',
            'sim_close_cabinet_middle_drawer',
            'sim_close_cabinet_top_drawer',

            'sim_left_arm_put_apple_to_cabinet_bottom_drawer',
            'sim_left_arm_put_banana_to_cabinet_bottom_drawer',
            'sim_left_arm_put_blue_bottle_to_cabinet_bottom_drawer',
            'sim_left_arm_put_green_bottle_to_cabinet_bottom_drawer',

            'sim_right_arm_put_apple_to_cabinet_bottom_drawer',
            'sim_right_arm_put_banana_to_cabinet_bottom_drawer',
            'sim_right_arm_put_blue_bottle_to_cabinet_bottom_drawer',
            'sim_right_arm_put_green_bottle_to_cabinet_bottom_drawer',

            'sim_dual_arm_put_apple_to_cabinet_bottom_drawer',
            'sim_dual_arm_put_banana_to_cabinet_bottom_drawer',
            'sim_dual_arm_put_blue_bottle_to_cabinet_bottom_drawer',
            'sim_dual_arm_put_green_bottle_to_cabinet_bottom_drawer',

            'sim_open_cabinet_bottom_drawer',
            'sim_open_cabinet_middle_drawer',
            'sim_open_cabinet_top_drawer',

            'sim_put_black_stapler_to_basket',
            'sim_put_camera_to_basket',
            'sim_put_green_stapler_to_basket',
            'sim_put_hammer_to_basket',

            'sim_put_black_stapler_to_locker_top_layer',
            'sim_put_camera_to_locker_top_layer',
            'sim_put_green_stapler_to_locker_top_layer',
            'sim_put_hammer_to_locker_top_layer',

            'sim_put_apple_to_pan',
            'sim_put_duck_to_pan',
            'sim_put_pig_to_pan',
            'sim_put_teapot_to_pan',

            'sim_put_black_stapler_to_locker_middle_layer',
            'sim_put_blue_block_to_locker_middle_layer',

            'sim_left_arm_blue_box_cover_bottom_left_marker',
            'sim_left_arm_blue_box_cover_bottom_right_marker',
            'sim_left_arm_blue_box_cover_upper_left_marker',
            'sim_left_arm_blue_box_cover_upper_right_marker',
            'sim_left_arm_green_box_cover_bottom_left_marker',
            'sim_left_arm_green_box_cover_bottom_right_marker',
            'sim_left_arm_green_box_cover_upper_left_marker',
            'sim_left_arm_green_box_cover_upper_right_marker',
            'sim_left_arm_red_box_cover_bottom_left_marker',
            'sim_left_arm_red_box_cover_bottom_right_marker',
            'sim_left_arm_red_box_cover_upper_left_marker',
            'sim_left_arm_red_box_cover_upper_right_marker',
            'sim_left_arm_yellow_box_cover_bottom_left_marker',
            'sim_left_arm_yellow_box_cover_bottom_right_marker',
            'sim_left_arm_yellow_box_cover_upper_left_marker',
            'sim_left_arm_yellow_box_cover_upper_right_marker',

            'sim_right_arm_blue_box_cover_bottom_left_marker',
            'sim_right_arm_blue_box_cover_bottom_right_marker',
            'sim_right_arm_blue_box_cover_upper_left_marker',
            'sim_right_arm_blue_box_cover_upper_right_marker',
            'sim_right_arm_green_box_cover_bottom_left_marker',
            'sim_right_arm_green_box_cover_bottom_right_marker',
            'sim_right_arm_green_box_cover_upper_left_marker',
            'sim_right_arm_green_box_cover_upper_right_marker',
            'sim_right_arm_red_box_cover_bottom_left_marker',
            'sim_right_arm_red_box_cover_bottom_right_marker',
            'sim_right_arm_red_box_cover_upper_left_marker',
            'sim_right_arm_red_box_cover_upper_right_marker',

            'sim_right_arm_yellow_box_cover_bottom_left_marker',
            'sim_right_arm_yellow_box_cover_bottom_right_marker',
            'sim_right_arm_yellow_box_cover_upper_left_marker',
            'sim_right_arm_yellow_box_cover_upper_right_marker',

            'sim_transfer_left_blue_block_to_right',
            'sim_transfer_left_green_block_to_right',
            'sim_transfer_left_red_block_to_right',
            'sim_transfer_left_yellow_block_to_right',
            'sim_transfer_right_blue_block_to_left',
            'sim_transfer_right_green_block_to_left',
            'sim_transfer_right_red_block_to_left',
            'sim_transfer_right_yellow_block_to_left'
        ]
        self.is_SupplmentaryExp = True
        self.SupplmentaryExp_Tasks = [
            'sim_left_arm_put_banana_to_cabinet_bottom_drawer',
            'sim_left_arm_put_blue_bottle_to_cabinet_bottom_drawer',
            'sim_left_arm_put_green_bottle_to_cabinet_bottom_drawer',
            'sim_right_arm_put_apple_to_cabinet_bottom_drawer',
            'sim_right_arm_put_blue_bottle_to_cabinet_bottom_drawer',
            'sim_right_arm_put_green_bottle_to_cabinet_bottom_drawer'
        ]

def main():
    args = Args()
    max_task_num = 0
    if args.is_SupplmentaryExp:
        TASK_NAME = args.SupplmentaryExp_Tasks
    else:
        TASK_NAME = args.TASK_NAME
    if args.GPU == '4090':
        max_task_num = 6
    elif args.GPU == 'H100':
        max_task_num = 40
    else:
        NotImplementedError
    counts = 0
    GPUS_NUM = 0
    for task in TASK_NAME:
        command = f'CUDA_VISIBLE_DEVICES={GPUS_NUM} nohup python imitate_episodes.py ' \
                  f'--ambiguity_env_name SimOpenDrawer ' \
                  f'--dataset_dir /data/litao.liu/CASIA_Intern/data_OpenDrawer_0525/data_SimOpenDrawer_ALL ' \
                  f'--ckpt_dir {args.ckpt_dir} ' \
                  f'--policy_class FoAM_wo_MPH ' \
                  f'--kl_weight 10 ' \
                  f'--chunk_size {args.chunk_size} ' \
                  f'--hidden_dim 512 ' \
                  f'--batch_size 8 ' \
                  f'--dim_feedforward 3200 ' \
                  f'--seed 0 ' \
                  f'--num_epochs 2000 ' \
                  f'--lr 1e-5 ' \
                  f'--multi_task ' \
                  f'--run_name multi_task_run ' \
                  f'--use_goal_img ' \
                  f'--eval ' \
                  f'--temporal_agg ' \
                  f'--task_name {task} ' \
                  f'> {args.log_prefix}_infer_{task}.log 2>&1&'

        print(command)

        if counts >= max_task_num:
            GPUS_NUM += 1
            counts = 0

        subprocess.run(command, shell=True)
        counts += 1

if __name__ == '__main__':
    main()





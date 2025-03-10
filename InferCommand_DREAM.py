import subprocess
import os

# docker exec -it TrainandInfer /bin/bash
# cd MultiPolicyTrainandInfer_frozenbackbone_0826/

class Args:
    def __init__(self):
        self.ckpt_dir = f'/workspace/ckpt/09111726_Sim_VLM_FoAM_w1_Lastimg_42Service'
        self.log_prefix = f'09111726_Sim_VLM_FoAM_w1_Lastimg_42Service'
        self.GPU = '4090'
        self.TASK_NAME = [
            'sim_pick_left_green_block',
            'sim_pick_right_green_block',

            'sim_put_green_stapler_to_locker_top_layer',
            'sim_put_hammer_to_locker_top_layer',

            'sim_put_apple_to_pan',

            'sim_transfer_left_green_block_to_right',
            'sim_transfer_right_yellow_block_to_left'

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
            'sim_transfer_right_yellow_block_to_left',
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
        self.huber_weight = 4
        self.is_SupplmentaryExp = False
        self.SupplmentaryExp_Tasks = [

        ]

def main():
    args = Args()
    max_task_num = 0
    if args.is_SupplmentaryExp:
        TASK_NAME = args.SupplmentaryExp_Tasks
    else:
        TASK_NAME = args.TASK_NAME
    if args.GPU == '4090':
        max_task_num = 10
    elif args.GPU == 'H100':
        max_task_num = 40
    else:
        NotImplementedError
    counts = 0
    GPUS_NUM = 0
    for task in TASK_NAME:
        command = f'CUDA_VISIBLE_DEVICES={GPUS_NUM + 1} nohup python imitate_episodes.py ' \
                  f'--ambiguity_env_name SimOpenDrawer ' \
                  f'--dataset_dir /data/litao.liu/CASIA_Intern/data_OpenDrawer_0525/data_SimOpenDrawer_ALL ' \
                  f'--ckpt_dir {args.ckpt_dir} ' \
                  f'--policy_class DREAM ' \
                  f'--kl_weight 10 ' \
                  f'--chunk_size 450 ' \
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
                  f'--huber_weight {args.huber_weight} ' \
                  f'--task_name {task} ' \
                  f'> {args.log_prefix}_infer_{task}.log 2>&1&'

        if counts >= max_task_num:
            GPUS_NUM += 1
            counts = 0

        print(command)

        subprocess.run(command, shell=True)
        counts += 1

if __name__ == '__main__':
    main()





<div align="center">

# FoAM: Foresight-Augmented Multi-Task Imitation Policy for Robotic Manipulation

<!-- [Delin Qu*](https://github.com/DelinQu), [HaomingSong*](https://github.com/HaomingSong), [Qizhi Chen*](https://github.com/Tavish9), [Yuanqi Yao](https://scholar.google.com/citations?user=s482QHoAAAAJ&hl=zh-CN), [Xinyi Ye](https://scholar.google.com/citations?user=GlYeyfoAAAAJ&hl=zh-CN), [Yan Ding](https://yding25.com), [Zhigang Wang](https://scholar.google.com/citations?user=cw3EaAYAAAAJ&hl=zh-CN), [JiaYuan Gu](https://cseweb.ucsd.edu/~jigu/), [Bin Zhao](https://scholar.google.com/citations?hl=zh-CN&user=DQB0hqwAAAAJ), [Xuelong Li†](https://scholar.google.com/citations?user=ahUibskAAAAJ)

Shanghai AI Laboratory, ShanghaiTech, TeleAI -->

<!-- <div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/930e6814-8a9f-43e1-a284-118a5732daa4">
  <br>
</div> -->

[\[📄Paper\]](https://arxiv.org/abs/2409.19528)  [\[🔥Project Page\]](https://projfoam.github.io/) [\[🚀 FoAM-benchmark\]](https://github.com/LitaoLiu01/FoAM-benchmark) 
 [\[🎄Custom Dataset\]](https://github.com/ProjFOAM/FoAM-dataset)

</div>

## Installation
If you want to use FoAM, you need to install the required packages:
```bash
conda create -n foam python=3.8.10
conda activate foam
```

Install torch.
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install packages from `requirements.txt` file. 

```bash
pip install -r requirements.txt
```
## Reproduce FoAM
### Prepare Datasets
First, generate data based on any one or more scenarios in the [FoAM Benchmark](https://github.com/LitaoLiu01/FoAM-benchmark).
Store the data in the folder `\train_data`. Its storage structure is shown below.

```md
# Data Storage Structure

- **train_data/** (Main directory for training data)
  - **scenarios_1/** (Scenario 1)
    - **subtask_1/** (Subtask 1)
      - **episodes_0-49/** (Contains episodes 0-49)
    - **subtask_2/** (Subtask 2)
      - **episodes_0-49/** (Contains episodes 0-49)
    - ...
  - **scenarios_2/** (Scenario 2)
    - **subtask_1/** (Subtask 1)
      - **episodes_0-49/**
    - **subtask_2/** (Subtask 2)
      - **episodes_0-49/**
    - ...
  - ...
```

### Training 
Run the `imitate_episodes.py` file.
```bash
--ambiguity_env_name SimOpenDrawer 
--dataset_dir /home/liulitao/Desktop/FoAM-main/train_data 
--ckpt_dir /home/liulitao/Desktop/FoAM-main/ckpt 
--policy_class FoAM 
--kl_weight 10 
--chunk_size 450 
--hidden_dim 512 
--batch_size 16 
--dim_feedforward 3200 
--seed 0 
--num_epochs 2000 
--lr 1e-5 
--multi_task 
--task_name sim_open_cabinet_bottom_drawer 
--run_name multi_task_run 
--huber_weight 0.001 
--use_redundant_task_emb -
-use_goal_img 
```

### Inference 
Run the `imitate_episodes.py` file.
```bash
--ambiguity_env_name SimOpenDrawer 
--dataset_dir /home/liulitao/Desktop/FoAM-main/train_data 
--ckpt_dir /home/liulitao/Desktop/FoAM-main/ckpt 
--policy_class FoAM 
--kl_weight 10 
--chunk_size 450 
--hidden_dim 512 
--batch_size 16 
--dim_feedforward 3200 
--seed 0 
--num_epochs 2000 
--lr 1e-5 
--multi_task 
--task_name sim_open_cabinet_bottom_drawer 
--run_name multi_task_run 
--huber_weight 0.001 
--use_redundant_task_emb -
-use_goal_img 
--eval
```

## FAQs
If you encounter any issues, feel free to open an issue on GitHub or reach out through discussions or directly contact [Litao Liu](mailto:litao.liu@example.com). We appreciate your feedback and contributions! 🚀

## License

This project is released under the [MIT license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{liu2024foamforesightaugmentedmultitaskimitation,
            title={FoAM: Foresight-Augmented Multi-Task Imitation Policy for Robotic Manipulation},
            author={Litao Liu and Wentao Wang and Yifan Han and Zhuoli Xie and Pengfei Yi and Junyan Li and Yi Qin and Wenzhao Lian},
            year={2024},
            eprint={2409.19528},
            archivePrefix={arXiv},
            primaryClass={cs.RO},
            url={https://arxiv.org/abs/2409.19528},
        }
```


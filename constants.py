import pathlib
import json
# from InferCommand_DREAM import
### Task parameters
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/SimEnvs/' # note: absolute path
DT = 0.02


def GetTextEmbeddings(is_use_redundant_task_emb):
    if is_use_redundant_task_emb:
        # 读取 embeddings_data.json 文件
        with open('TEXT_EMBEDDINGS_80.json', 'r') as json_file:
            TEXT_EMBEDDINGS = json.load(json_file)
    else:
        # 读取 embeddings_data.json 文件
        with open('TEXT_EMBEDDINGS.json', 'r') as json_file:
            TEXT_EMBEDDINGS = json.load(json_file)

    return TEXT_EMBEDDINGS

# 训练的时候需要修改这里，以确定是否开启redundant language模式，推理的时候不用管，
# You need to modify this during training to determine whether to enable redundant language mode. This is not necessary during inference.
# 通过是否设置参数args['use_redundant_task_emb']来控制推理的时候是否使用redundant task emb。
# Whether to use redundant task emb during inference is controlled by setting the parameter args['use_redundant_task_emb'].
is_use_redundant_task_emb = False # TODO hardcode

TEXT_EMBEDDINGS = GetTextEmbeddings(is_use_redundant_task_emb)

CAMERA_NAMES = ['head']

SIM_AMBIGUITY_NAMES = {
    'SimOpenDrawer': {
        'dataset_dir': '/home/liulitao/Desktop/FoAM-main/train_data/',
        'num_episodes': 50,
        'episode_len': 450,
        'camera_names': CAMERA_NAMES
    },
    'SimALLTasks': {
        'dataset_dir': '/data/litao.liu/CASIA_Intern/ALL_DATA/data_SimOpenDrawer/data_SimALL/',
        'num_episodes': 4000,
        'episode_len': 450,
        'camera_names': CAMERA_NAMES
    },
}

SIM_TASK_CONFIGS = {
    'sim_OpenCabinetDrawer_ACTGoalImg': {
        'dataset_dir': '/data/litao.liu/CASIA_Intern/ALL_DATA/data_SimOpenDrawer/data_SimOpenDrawer_ALL/',
        'num_episodes': 4000,
        'episode_len': 450,
        'camera_names': ['head', 'goal_images']
    },
}

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
TASKS = [
        # Simple Level 15
        "Pick left blue block",
        "Pick left green block",
        "Pick left red block",
        "Pick left yellow block",
        "Pick right blue block",
        "Pick right green block",
        "Pick right red block",
        "Pick right yellow block",
        "Close cabinet bottom drawer",
        "Close cabinet middle drawer",
        "Close cabinet top drawer",
        # Same with Dual Arm Task II
        "Put apple to cabinet bottom drawer",
        "Put banana to cabinet bottom drawer",
        "Put blue bottle to cabinet bottom drawer",
        "Put green bottle to cabinet bottom drawer",

        # Normal Level 17
        "Open cabinet bottom drawer",
        "Open cabinet middle drawer",
        "Open cabinet top drawer",
        "Put black stapler to basket",
        "Put camera to basket",
        "Put green stapler to basket",
        "Put hammer to basket",
        "Put black stapler to locker top layer",
        "Put camera to locker top layer",
        "Put green stapler to locker top layer",
        "Put hammer to locker top layer",
        "Put apple to pan",
        "Put duck to pan",
        "Put pig to pan",
        "Put teapot to pan",
        "Put black stapler to locker middle layer",
        "Put blue block to locker middle layer",

        # Difficult Level 16
        "Pick blue box to cover lower left marker",
        "Pick blue box to cover lower right marker",
        "Pick blue box to cover upper left marker",
        "Pick blue box to cover upper right marker",
        "Pick green box to cover lower left marker",
        "Pick green box to cover lower right marker",
        "Pick green box to cover upper left marker",
        "Pick green box to cover upper right marker",
        "Pick red box to cover lower left marker",
        "Pick red box to cover lower right marker",
        "Pick red box to cover upper left marker",
        "Pick red box to cover upper right marker",
        "Pick yellow box to cover lower left marker",
        "Pick yellow box to cover lower right marker",
        "Pick yellow box to cover upper left marker",
        "Pick yellow box to cover upper right marker",

        # Dual Arm Tasks 8
        "Transfer left blue block to right",
        "Transfer left green block to right",
        "Transfer left red block to right",
        "Transfer left yellow block to right",
        "Transfer right blue block to left",
        "Transfer right green block to left",
        "Transfer right red block to left",
        "Transfer right yellow block to left",
        ]

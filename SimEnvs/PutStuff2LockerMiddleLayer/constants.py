import pathlib

### Task parameters
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/model/' # note: absolute path
DT = 0.04

DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {

    'sim_put_black_stapler_to_locker_middle_layer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },

    'sim_put_black_stapler_to_locker_bottom_layer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },

    'sim_put_blue_block_to_locker_bottom_layer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_blue_block_to_locker_middle_layer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
}

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
import pathlib

### Task parameters
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/model/' # note: absolute path
DT = 0.02

DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {
    'sim_transfer_right_red_block_to_left': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_right_yellow_block_to_left': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_right_blue_block_to_left': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_right_green_block_to_left': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_left_red_block_to_right': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_left_yellow_block_to_right': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_left_blue_block_to_right': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_transfer_left_green_block_to_right': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
}

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
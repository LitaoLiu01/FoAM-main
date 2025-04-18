import pathlib

### Task parameters
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/model/' # note: absolute path
DT = 0.02

DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {
    'sim_pick_the_right_red_block': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_right_yellow_block': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_right_blue_block': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_right_green_block': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_left_red_block': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_left_yellow_block': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_left_blue_block': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_pick_the_left_green_block': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
}

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
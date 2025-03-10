import pathlib

### Task parameters
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/model/' # note: absolute path
DT = 0.02

DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {
    'sim_put_blue_bottle_to_bottom_drawer': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_green_bottle_to_bottom_drawer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_banana_to_bottom_drawer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_apple_to_bottom_drawer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_blue_bottle_to_middle_drawer': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_green_bottle_to_middle_drawer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_banana_to_middle_drawer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_put_apple_to_middle_drawer': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
}

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
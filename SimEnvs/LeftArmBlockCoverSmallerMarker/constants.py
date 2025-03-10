import pathlib

### Task parameters
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/model/' # note: absolute path
DT = 0.02

DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {
    'sim_red_box_cover_upper_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_yellow_box_cover_upper_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_blue_box_cover_upper_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_green_box_cover_upper_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_red_box_cover_upper_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_yellow_box_cover_upper_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_blue_box_cover_upper_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_green_box_cover_upper_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_red_box_cover_bottom_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_yellow_box_cover_bottom_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_blue_box_cover_bottom_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_green_box_cover_bottom_left_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_red_box_cover_bottom_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_yellow_box_cover_bottom_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_blue_box_cover_bottom_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
    'sim_green_box_cover_bottom_right_marker': {
        'dataset_dir': DATA_DIR + '/sim_open_cabinet_drawer',
        'num_episodes': 2,
        'episode_len': 450,
        'camera_names': ['head']
    },
}

START_ARM_POSE = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
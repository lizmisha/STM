import os
import random

import cv2
import torch
import numpy as np
import yaml
from shutil import copytree, copy


def set_global_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_global_configs(seed: int, max_threads: int):
    set_global_seed(seed)
    cv2.setNumThreads(max_threads)
    torch.set_num_threads(max_threads)


def get_config(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config


# def save_train_files(config_path: str, experiment_path: str):
#     dir_path = parse_dir_path(config_path)
#     config_dir_name = config_path.split('/')[-2]
#
#     copytree(os.path.join(dir_path, config_dir_name),
#              os.path.join(experiment_path, 'train_files', config_dir_name),
#              copy_function=copy,
#              dirs_exist_ok=True)
#
#     copytree(os.path.join(dir_path, 'src'),
#              os.path.join(experiment_path, 'train_files/src'),
#              copy_function=copy,
#              dirs_exist_ok=True)

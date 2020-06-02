import argparse
import os

import torch

from src.models.callbacks import Callbacks, CheckpointSaver, WanDB


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='/home/liz/GitLab/REGINA/modelling/notebooks/ocr/configs/train_config.yaml'
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=2
    )
    return parser.parse_args()


def create_callbacks(experiment_name: str, dumps: dict) -> Callbacks:
    log_dir = os.path.join(dumps['path'], experiment_name, 'logs')
    weights_dir = os.path.join(dumps['path'], experiment_name, 'weights')
    return Callbacks(
        [
            WanDB(experiment_name, log_dir),
            CheckpointSaver(
                metric_name='mAP',
                save_dir=weights_dir,
                save_name='epoch_{epoch}_mAP_{mAP}.pth',
                num_checkpoints=3,
                mode='max'
            )
        ]
    )

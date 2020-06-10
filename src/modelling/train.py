import argparse
import os
from pprint import pprint

import torch

from src.modelling.callbacks import Callbacks, CheckpointSaver, WanDB
from src.modelling.factory import DataFactory, Factory
from src.modelling.modelling_utils import get_config, set_global_configs, save_train_files
from src.modelling.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='/home/liz/STM/configs/train_config.yaml'
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=3
    )
    return parser.parse_args()


def create_callbacks(experiment_params: dict, target_metric: str) -> Callbacks:
    log_dir = os.path.join(experiment_params['dump_path'], experiment_params['experiment_name'], 'logs')
    weights_dir = os.path.join(experiment_params['dump_path'], experiment_params['experiment_name'], 'weights')
    return Callbacks(
        [
            WanDB(experiment_params['project_name'], experiment_params['experiment_name'], log_dir),
            CheckpointSaver(
                metric_name=target_metric,
                save_dir=weights_dir,
                save_name='epoch_{epoch}_' + target_metric + '_{' + target_metric + '}.pth',
                num_checkpoints=1,
                mode='max'
            )
        ]
    )


def main():
    args = parse_args()
    config = get_config(args.config)
    pprint(config)

    set_global_configs(config['experiment_params']['seed'], config['data_params']['num_workers'])
    save_train_files(args.config, os.path.join(config['experiment_params']['dump_path'],
                                               config['experiment_params']['experiment_name']))

    device = torch.device('cuda', args.cuda)
    factory = Factory(config['train_params'])
    data_factory = DataFactory(config['data_params'])
    callbacks = create_callbacks(config['experiment_params'], config['train_params']['target_metric'])

    trainer = Runner(callbacks=callbacks, factory=factory, device=device, stages=config['stages'])
    trainer.fit(data_factory)


if __name__ == '__main__':
    main()

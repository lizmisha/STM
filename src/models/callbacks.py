import os
from queue import PriorityQueue

import torch
import wandb

from src.interfaces.callback import Callback


class Callbacks(Callback):

    def __init__(self, callbacks):
        super().__init__()
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []

    def set_runner(self, runner):
        super().set_runner(runner)
        for callback in self.callbacks:
            callback.set_runner(runner)

    def on_batch_begin(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(i, **kwargs)

    def on_batch_end(self, i, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(i, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch)

    def on_stage_begin(self):
        for callback in self.callbacks:
            callback.on_stage_begin()

    def on_stage_end(self):
        for callback in self.callbacks:
            callback.on_stage_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


class CheckpointSaver(Callback):

    def __init__(self, save_dir: str, save_name: str, num_checkpoints: int, mode: str, metric_name: str):
        super().__init__()
        self.mode = mode
        self.save_name = save_name
        self._best_checkpoints_queue = PriorityQueue(num_checkpoints)
        self._metric_name = metric_name
        self.save_dir = save_dir

    def on_train_begin(self):
        os.makedirs(self.save_dir, exist_ok=True)

    def on_stage_begin(self):
        os.makedirs(os.path.join(self.save_dir, self.runner.current_stage_name), exist_ok=True)
        while not self._best_checkpoints_queue.empty():
            self._best_checkpoints_queue.get()

    def save_checkpoint(self, epoch, path):
        if hasattr(self.runner.model, 'module'):
            state_dict = self.runner.model.module.state_dict()
        else:
            state_dict = self.runner.model.state_dict()
        torch.save(state_dict, path)

    def on_epoch_end(self, epoch):
        metric = self.metrics.valid_metrics[self._metric_name]
        new_path_to_save = os.path.join(
            self.save_dir,
            self.runner.current_stage_name,
            self.save_name.format(epoch=epoch, metric=f'{metric:.5}')
        )
        if self._try_update_best_losses(metric, new_path_to_save):
            self.save_checkpoint(epoch=epoch, path=new_path_to_save)

    def _try_update_best_losses(self, metric, new_path_to_save):
        if self.mode == 'min':
            metric = -metric
        if not self._best_checkpoints_queue.full():
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        min_metric, min_metric_path = self._best_checkpoints_queue.get()

        if min_metric < metric:
            os.remove(min_metric_path)
            self._best_checkpoints_queue.put((metric, new_path_to_save))
            return True

        self._best_checkpoints_queue.put((min_metric, min_metric_path))
        return False

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_stage_end(self):
        pass

    def on_train_end(self):
        pass

    def on_batch_begin(self, i, **kwargs):
        pass


class WanDB(Callback):

    def __init__(self, project_name: str, log_dir: str):
        super().__init__()

        self.project_name = project_name
        self.log_dir = log_dir
        self.logger = None

    def on_batch_begin(self, i, **kwargs):
        pass

    def on_batch_end(self, i, **kwargs):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, epoch: int):
        for k, v in self.metrics.train_metrics.items():
            self.logger.log({f'train_{k}': float(v), 'epoch': epoch})

        for k, v in self.metrics.valid_metrics.items():
            self.logger.log({f'valid_{k}': float(v), 'epoch': epoch})

        self.logger.join()

    def on_stage_begin(self):
        self.logger = wandb.init(project=self.project_name, dir=self.log_dir, name=self.runner.current_stage_name)
        self.logger.watch(self.runner.model)

    def on_stage_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

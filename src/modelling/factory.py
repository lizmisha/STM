import pydoc

import torch

from src.data.make_dataset import make_data


class Metrics:

    def __init__(self, functions):
        self.functions = functions
        self.best_score = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.valid_metrics = {}


class Factory:
    def __init__(self, params: dict):
        self.params = params

    def make_model(self, stage, device) -> torch.nn.Module:
        model_name = self.params['model']
        if 'model_params' in self.params:
            model = pydoc.locate(model_name)(**self.params['model_params'])
        else:
            model = pydoc.locate(model_name)

        if isinstance(stage.get('weights', None), str):
            model.load_state_dict(torch.load(stage['weights']))
        return model.to(device)

    @staticmethod
    def make_optimizer(model: torch.nn.Module, stage: dict) -> torch.optim.Optimizer:
        return getattr(torch.optim, stage['optimizer'])(params=model.get_trainable_parameters(),
                                                        **stage['optimizer_params'])

    @staticmethod
    def make_scheduler(optimizer, stage):
        return getattr(torch.optim.lr_scheduler, stage['scheduler'])(optimizer=optimizer, **stage['scheduler_params'])

    @staticmethod
    def make_loss(stage, device):
        if '.' not in stage['loss']:
            return getattr(torch.nn, stage['loss'])().to(device)

        return pydoc.locate(stage['loss'])().to(device)

    def make_metrics(self) -> Metrics:
        return Metrics(
            {
                params: pydoc.locate(metric)(**params)
                for metric, params in self.params['metrics'].items()
            }
        )


class DataFactory:
    def __init__(self, params: dict):
        self.params = params

    def make_train_loader(self):
        return make_data(**self.params, mode='train')

    def make_valid_loader(self):
        return make_data(**self.params, mode='valid')

    def make_test_loader(self):
        return make_data(**self.params, mode='test')

from abc import abstractmethod


class Callback:
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        self.runner = None
        self.metrics = None

    def set_runner(self, runner):
        self.runner = runner
        self.metrics = runner.metrics

    @abstractmethod
    def on_batch_begin(self, i, **kwargs): pass

    @abstractmethod
    def on_batch_end(self, i, **kwargs): pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int): pass

    @abstractmethod
    def on_epoch_end(self, epoch: int): pass

    @abstractmethod
    def on_stage_begin(self): pass

    @abstractmethod
    def on_stage_end(self): pass

    @abstractmethod
    def on_train_begin(self): pass

    @abstractmethod
    def on_train_end(self): pass

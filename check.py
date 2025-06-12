from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl


class DelayedModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if start_epoch < 1:
            raise ValueError("start_epoch must be at least 1.")
        self.start_epoch = start_epoch

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch + 1 >= self.start_epoch:
            super().on_train_epoch_end(trainer, pl_module)
        else:
            pass


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, start_epoch: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if start_epoch < 1:
            raise ValueError("start_epoch must be at least 1.")
        self.start_epoch = start_epoch
        self._stopped = True

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch + 1 >= self.start_epoch:
            self._stopped = False
            super().on_validation_epoch_end(trainer, pl_module)
        else:
            self._stopped = True

# import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics

import lightning.pytorch as pl
from lightning.pytorch import Trainer

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor

def get_trainer(epochs: int) -> Trainer:
    # generate the callbacks
    early_stopping = EarlyStopping('val/loss', patience=3)
    model_checkpoint = ModelCheckpoint(monitor="val/loss", mode="min", filename="Ep{epoch:02d}-val_loss{val/loss:.2f}-best", auto_insert_metric_name=False)
    # create the trainer
    trainer = Trainer(
        max_epochs=epochs,
        deterministic=True,
        callbacks=[early_stopping, model_checkpoint, RichProgressBar(leave=True), LearningRateMonitor(logging_interval='epoch')],
        log_every_n_steps=100
    )
    return trainer

# def fit_trainer(trainer: Trainer, model: pl.LightningModule, dataset: pl.LightningDataModule) -> Trainer:
#     # fit the trainer
#     trainer.fit(model, datamodule=dataset)
#     return trainer

# def test_trainer(trainer: Trainer, dataset: pl.LightningDataModule):
#     trainer.test(datamodule=dataset)
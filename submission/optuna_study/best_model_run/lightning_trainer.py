# import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics

import lightning.pytorch as pl
from lightning.pytorch import Trainer

class LightningTrainerModule(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float, weight_decay: float = 1e-4) -> None:
        # lightning code
        super().__init__()
        # save some of our parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        # save the lightning module specific variables
        self.automatic_optimization = True
        # define loss and metrics
        self._loss_fn = torch.nn.CrossEntropyLoss()
        self._train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self._val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self._test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        # define model
        self.model = model
        # define the results
        self.train_results = []
        self.test_results = []

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        loss = self._loss_fn(preds, y)
        self._train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/accuracy", self._train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        loss = self._loss_fn(preds, y)
        self._val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/accuracy", self._val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return

    def test_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)
        loss = self._loss_fn(preds, y)
        self._test_acc(preds, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/accuracy", self._test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def on_train_epoch_end(self):
        train_acc = self._train_acc.compute().detach().cpu().numpy() 
        self.train_results.append(train_acc)

    def on_validation_epoch_end(self):
        val_acc = self._val_acc.compute().detach().cpu().numpy()
        self.test_results.append(val_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
        return [optimizer], [{"scheduler":scheduler,"interval":"epoch","monitor":"val/loss"}]
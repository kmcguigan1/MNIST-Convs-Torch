import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

# in the future I want to implement Cross Fold Validation in a way that we can easily 
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str="./", batch_size=64, normalize=True, rotation_degrees=0, gaussian_blur=(3,0.2), seed=1999):
        # setup the experiment
        super().__init__()
        # save the info we need
        self.save_hyperparameters()
        # save the arrays
        self.train_data = None
        self.test_data = None
        # create the transforms
        self.train_augmentations = [transforms.ToTensor(),]
        self.test_augmentations = [transforms.ToTensor(),]
        if(normalize):
            self.train_augmentations.append(transforms.Normalize((0.1307,), (0.3081,)))
            self.test_augmentations.append(transforms.Normalize((0.1307,), (0.3081,)))
        if(rotation_degrees is not None and rotation_degrees != 0):
            self.train_augmentations.append(transforms.RandomRotation(degrees=rotation_degrees))
        if(gaussian_blur is not None):
            self.train_augmentations.append(transforms.GaussianBlur(gaussian_blur[0], gaussian_blur[1]))
        self.train_augmentations = transforms.Compose(self.train_augmentations)
        self.test_augmentations = transforms.Compose(self.test_augmentations)

    def prepare_data(self):
        # download
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if(self.train_data is None or self.test_data is None):
            self.train_data = MNIST(self.hparams.data_dir, train=True, transform=self.train_augmentations, download=True)
            self.test_data = MNIST(self.hparams.data_dir, train=False, transform=self.test_augmentations, download=True)

        # # Assign train/val datasets for use in dataloaders
        # if stage == "fit":
        #     self.train_data = MNIST(self.hparams.data_dir, train=True, transform=self.train_augmentations)
        #     self.test_data = MNIST(self.hparams.data_dir, train=False, transform=self.test_augmentations)

        # # Assign test dataset for use in dataloader(s)
        # if stage == "test":
        #     self.test_data = MNIST(self.hparams.data_dir, train=False, transform=self.test_augmentations)

        # if stage == "predict":
        #     self.test_data = MNIST(self.hparams.data_dir, train=False, transform=self.test_augmentations)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size*2, pin_memory=True, num_workers=4, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size*2, pin_memory=True, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size*2, pin_memory=True, num_workers=4)
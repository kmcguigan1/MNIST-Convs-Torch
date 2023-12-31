import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms

from sklearn.model_selection import KFold

from dataclasses import dataclass

@dataclass
class MNISTDataModuleConfig:
    pass

# in the future I want to implement Cross Fold Validation in a way that we can easily 
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str="./", batch_size=32, normalize=True, seed=1999, k=1, folds=5):
        # setup the experiment
        super().__init__()
        # save the hyperparameters
        self.save_hyperparameters(logger=False)
        # create the transforms
        if(self.hparams.normalize):
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            # cross_val_splitter = KFold(n_splits=5, shuffle=True, random_state=self.hparams.split_seed)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(self.hparams.seed)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.hparams.batch_size, pin_memory=True, num_workers=4)
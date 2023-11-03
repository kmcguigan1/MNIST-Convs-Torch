from lightning_trainer import LightningTrainerModule
from lightning_data import MNISTDataModule
from lightning_utils import get_trainer

from model import LilyModel

import gc
import numpy as np

import matplotlib.pyplot as plt

def run_model():
    data_module = MNISTDataModule(normalize=False, rotation_degrees=12, gaussian_blur=(3,0.0420))
    model = LilyModel(dropout_frac=0.0815)
    trainer_module = LightningTrainerModule(model, 0.0021, weight_decay=3.9109e-6)
    trainer = get_trainer(30)
    trainer.fit(model=trainer_module, datamodule=data_module)
    outputs = trainer.test(model=trainer_module, datamodule=data_module, ckpt_path="best")[0]

    train_acc = np.array(trainer_module.train_results)
    test_acc = np.array(trainer_module.test_results)

    print("\n\n")
    print("----------------------------------")
    print(outputs)
    print("----------------------------------\n\n")
    return outputs['test/accuracy'], train_acc, test_acc

if __name__ == '__main__':
    final_test_acc, train_acc, test_acc = run_model()
    test_acc = test_acc[1:]
    print(train_acc, train_acc[0], type(train_acc[0]))
    fig, ax = plt.subplots(1,1,figsize=(18,10))
    print(final_test_acc)
    fig.suptitle(f'Final Test Acc: {100*final_test_acc:.2f}')
    ax.plot(np.arange(train_acc.shape[0]), 100*train_acc, label="Train")
    ax.plot(np.arange(test_acc.shape[0]), 100*test_acc, label="Test")
    ax.legend(loc="best")
    plt.show()

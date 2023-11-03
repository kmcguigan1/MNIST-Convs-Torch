from lightning_trainer import LightningTrainerModule
from lightning_data import MNISTDataModule
from lightning_utils import get_trainer

from model import ResdiualModel

import optuna
from optuna.trial import TrialState

import gc

STUDY_NAME = 'lily-model-study'

def run_model(normalize_data, rotation_degrees, blur_sigma, learning_rate, weight_decay, drop_frac):
    data_module = MNISTDataModule(normalize=normalize_data, rotation_degrees=rotation_degrees, gaussian_blur=(3,blur_sigma))
    model = LilyModel(dropout_frac=drop_frac)
    trainer_module = LightningTrainerModule(model, learning_rate, weight_decay=weight_decay)
    trainer = get_trainer(25)
    trainer.fit(model=trainer_module, datamodule=data_module)
    outputs = trainer.test(model=trainer_module, datamodule=data_module, ckpt_path="best")[0]
    print("\n\n")
    print("----------------------------------")
    print(outputs)
    print(type(outputs))
    print("----------------------------------\n\n")
    return outputs['test/accuracy']

def objective(trial):
    # get the data parameters
    normalize_data = trial.suggest_categorical('normalize_data', [True, False])
    rotation_degrees = trial.suggest_int('rotation_degrees', low=0, high=25)
    blur_sigma = trial.suggest_float('blur_sigma', low=0.0, high=0.75)
    # get the training parameters
    learning_rate = trial.suggest_float('learning_rate', low=0.0001, high=0.01)
    weight_decay = trial.suggest_float('weight_decay', low=0.0, high=1e-4)
    # get the model parameters
    drop_frac = trial.suggest_float('drop_fract', low=0.0, high=0.5)
    # dispatch a model with these parameters
    test_acc = run_model(
        normalize_data, 
        rotation_degrees,
        blur_sigma,
        learning_rate, 
        weight_decay, 
        drop_frac
    )
    return test_acc

def run_optuna_study():
    study = optuna.create_study(direction='maximize', storage=f'sqlite:///{STUDY_NAME}_optuna.db', load_if_exists=True, study_name=STUDY_NAME)
    study.optimize(objective, n_trials=50)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("study_results_resid.csv")

def print_optuna_study():
    study = optuna.create_study(direction='maximize', storage=f'sqlite:///{STUDY_NAME}_optuna.db', load_if_exists=True, study_name=STUDY_NAME)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def save_optuna_study():
    study = optuna.create_study(direction='maximize', storage=f'sqlite:///{STUDY_NAME}_optuna.db', load_if_exists=True, study_name=STUDY_NAME)
    
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("study_results.csv")

if __name__ == '__main__':
    run_optuna_study()
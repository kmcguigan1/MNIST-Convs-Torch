from lightning_trainer import LightningTrainerModule
from lightning_data import MNISTDataModule
from lightning_utils import get_trainer

from model import ResdiualModel

import optuna
from optuna.trial import TrialState

import gc

STUDY_NAME = 'resid-study'

# def run_model(normalize_data, rotation_degrees, blur_sigma, learning_rate, weight_decay, drop_frac):
#     data_module = MNISTDataModule(normalize=normalize_data, rotation_degrees=rotation_degrees, gaussian_blur=(3,blur_sigma))
#     model = LilyModel(dropout_frac=drop_frac)
#     trainer_module = LightningTrainerModule(model, learning_rate, weight_decay=weight_decay)
#     trainer = get_trainer(25)
#     trainer.fit(model=trainer_module, datamodule=data_module)
#     outputs = trainer.test(model=trainer_module, datamodule=data_module, ckpt_path="best")[0]
#     print("\n\n")
#     print("----------------------------------")
#     print(outputs)
#     print(type(outputs))
#     print("----------------------------------\n\n")
#     return outputs['test/accuracy']

def run_model(
    normalize_data, 
    rotation_degrees, 
    blur_sigma, 
    learning_rate, 
    weight_decay, 
    num_conv_layers, 
    n_filters, 
    num_dense_layers, 
    n_nodes, 
    drop_frac
):
    data_module = MNISTDataModule(normalize=normalize_data, rotation_degrees=rotation_degrees, gaussian_blur=(3,blur_sigma))
    model = ResdiualModel(n_filters, n_nodes, num_conv_layers, num_dense_layers, drop_frac)
    trainer_module = LightningTrainerModule(model, learning_rate, weight_decay=weight_decay)
    trainer = get_trainer(25)
    trainer.fit(model=trainer_module, datamodule=data_module)
    outputs = trainer.test(model=trainer_module, datamodule=data_module, ckpt_path="best")[0]
    print("\n\n")
    print("----------------------------------")
    print(outputs)
    print(type(outputs))
    print("----------------------------------\n\n")
    # clean up the run
    del data_module
    del model
    del trainer_module
    del trainer
    gc.collect()
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
    num_conv_layers = trial.suggest_int('num_layers', low=1, high=6)
    n_filters = 2**trial.suggest_int('n_filters', low=4, high=6)
    num_dense_layers = trial.suggest_int('num_dense_layers', low=1, high=3)
    n_nodes = 2**trial.suggest_int('n_nodes', low=4, high=6)
    drop_frac = trial.suggest_float('drop_fract', low=0.0, high=0.5)
    # dispatch a model with these parameters
    test_acc = run_model(
        normalize_data, 
        rotation_degrees,
        blur_sigma,
        learning_rate, 
        weight_decay, 
        num_conv_layers, 
        n_filters,
        num_dense_layers,
        n_nodes,
        drop_frac
    )
    return test_acc

def run_optuna_study():
    study = optuna.create_study(direction='maximize', storage=f'sqlite:///{STUDY_NAME}_optuna.db', load_if_exists=True, study_name=STUDY_NAME)
    study.optimize(objective, n_trials=50)

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

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("study_results_resid.csv")

def save_optuna_study():
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

    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv("study_results.csv")


# def main():
#     data_module = MNISTDataModule()
#     model = ConvolutionalModel()
#     trainer_module = LightningTrainerModule(model)
#     trainer = get_trainer(25)
#     trainer.fit(model=trainer_module, datamodule=data_module)


if __name__ == '__main__':
    run_optuna_study()

# def run_conv_model(num_blocks, layers_per_block, starting_channels, dense_layers, starting_dense_nodes):
#     data_module = MNISTDataModule()
#     model = ConvolutionalModel(num_blocks=num_blocks, layers_per_block=layers_per_block, starting_channels=starting_channels, dense_layers=dense_layers, starting_dense_nodes=starting_dense_nodes)
#     trainer_module = LightningTrainerModule(model)
#     trainer = get_trainer(2)
#     trainer.fit(model=trainer_module, datamodule=data_module)
#     test_results = trainer.test(model=model, datamodule=data_module)
#     print(test_results)

# def objective(trial):
#     # get the recomended parameters
#     num_blocks = trial.suggest_int('num_blocks', low=1, high=4)
#     layers_per_block = trial.suggest_int('layers_per_block', low=1, high=4)
#     starting_channels = 2**trial.suggest_int('starting_channels', low=3, high=7)
#     dense_layers = trial.suggest_int('dense_layers', low=1, high=4)
#     starting_dense_nodes = 2**trial.suggest_int('starting_dense_nodes', low=4, high=7)
#     # dispatch a model with these parameters
#     test_acc = run_conv_model(num_blocks, layers_per_block, starting_channels, dense_layers, starting_dense_nodes)
#     return test_acc
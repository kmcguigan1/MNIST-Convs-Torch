import torch
import torch.nn as nn
import torch.nn.functional as F

from optuna.trial import Trial

from layers import Convolution, FullyConnected

class ConvolutionalNetwork(nn.Module):
    def __init__(self, trial: Trial, input_shape: tuple = (28,28)):
        # get the convolutions
        in_features, im_size = 1, 28
        self.convs = []
        n_convs = trial.suggest_int("n_convs", low=1, high=6)
        for idx in range(n_convs):
            # get the convolution parameters
            out_features = trial.suggest_int(f"filter_count_{idx}", low=0, high=10)
            kernel = trial.suggest_int(f"kernel_size_{idx}", low=1, high=7, step=2)
            use_batch_norm = trial.suggest_categorical(f"user_batch_norm_{idx}", [True, False])
            dropout_frac = trial.suggest_float(f'dropout_frac_{idx}', low=0.0, high=0.5, step=0.1)
            stride = trial.suggest_int(f'stride_{idx}', low=1, high=2)
            dilation = trial.suggest_int(f'dilation_{idx}', low=1, high=2)
            # add conv to layers
            self.convs.append(Convolution(
                in_features, 
                2**out_features, 
                kernel_size=kernel,
                stride=stride, 
                dilation=dilation, 
                activation_name='relu', 
                use_batch_norm=use_batch_norm, 
                dropout_frac=dropout_frac
            ))
            # update what we know
            in_features = 2**out_features
            im_size /= stride
        
        # get the dense layers
        in_features *= im_size * im_size
        self.dense = []
        n_dense = trial.suggest_int("n_dense", low=0, high=4)
        for idx in range(n_dense):
            # get dense params
            out_features = trial.suggest_int(f"dense_count_{idx}", low=0, high=10)
            use_batch_norm = trial.suggest_categorical(f"use_batch_norm_{idx+n_conv}", [True, False])
            dropout_frac = trial.suggest_float(f'dropout_frac_{idx+n_conv}', low=0.0, high=0.5, step=0.1)
            # add the dense layer
            self.dense.append(FullyConnected(
                in_features,
                out_features,
                activation_name='relu',
                use_batch_norm=use_batch_norm,
                dropout_frac=dropout_frac
            ))
            # update what we know
            in_features = 2**out_features
        # create the final projection layer
        self.projection = nn.Linear(in_features, 10)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.flatten(x)
        for dense in self.dense:
            x = dense(x)
        return self.projection(x)



        


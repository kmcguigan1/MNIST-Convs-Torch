import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION_MAP = {
    'relu': nn.ReLU(),
}

class LinearActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ActivationLayer(nn.Module):
    def __init__(self, activation_name='none'):
        self.use_activation = activation_name != 'none'
        if(self.use_activation):
            self.activation = ACTIVATION_MAP[activation_name]

    def forward(self, x):
        if(self.use_activation):
            x = self.activation(x)
        return x

class FullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels, activation_name='none', use_batch_norm=False, dropout_frac=0.0):
        self.dense = nn.Linear(in_channels, out_channels)
        # get the activation layer
        self.activation = ActivationLayer(activation_name)
        # get the batchnorm layer
        self.use_batch_norm = use_batch_norm
        if(self.use_batch_norm):
            self.bn = nn.BatchNorm1d(out_channels)
        self.use_dropout = dropout_frac > 0
        if(self.use_dropout):
            self.drop = nn.Dropout(dropout_frac)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        if(self.use_batch_norm):
            x = self.bn(x)
        if(self.use_dropout):
            x = self.drop(x)
        return x

class Pooling(nn.Module):
    def __init__(self, kernel_size=1, kind='max'):
        self.use_pooling = stride > 1
        if(self.use_pooling):
            if(kind == 'max'):
                self.pool = nn.MaxPool2d(kernel_size, padding='same')
            elif(kind == 'average'):
                self.pool = nn.AvgPool2d(kernel_size, padding='same')
            else:
                raise Exception(f"Invalid Pooling kind of {kind} specified to Pooling layer")
    
    def forward(self, x):
        if(self.use_pooling):
            x = self.pool(x)
        return x

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, activation_name='none', use_batch_norm=False, dropout_frac=0.0, pooling_size=1, pooling_kind='max'):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding='same', groups=groups)
        self.activation = ActivationLayer(activation_name)
        self.use_batch_norm = use_batch_norm
        if(self.use_batch_norm):
            self.bn = nn.BatchNorm2d(out_channels)
        self.use_dropout = dropout_frac > 0
        if(self.use_dropout):
            self.drop = nn.Dropout(dropout_frac)
        self.pooling = Pooling(kernel_size=pooling_size, kind=pooling_kind)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        if(self.use_batch_norm):
            x = self.bn(x)
        if(self.use_dropout):
            x = self.drop(x)
        return x
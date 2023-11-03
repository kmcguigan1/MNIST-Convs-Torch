import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvLayer, ResidConvLayer, FullyConnectedLayer, ResidBlock, ConvBlock

class LilyModel(nn.Module):
    def __init__(self, dropout_frac):
        super().__init__()
        # convolution
        self.input_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.input_norm = nn.BatchNorm2d(32)
        self.input_relu = nn.ReLU()
        self.input_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.hidden_conv = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.hidden_norm = nn.BatchNorm2d(64)
        self.hidden_relu = nn.ReLU()
        self.hidden_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.output_conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.output_norm = nn.BatchNorm2d(128)
        self.output_relu = nn.ReLU()
        self.output_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.output_dropout = nn.Dropout(dropout_frac)

        # classification
        self.input_dense = nn.Linear(128,64)
        self.input_dense_norm = nn.BatchNorm1d(64)
        self.input_dense_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_frac)
        self.output_dense = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_norm(x)
        x = self.input_relu(x)
        x = self.input_pool(x)

        x = self.hidden_conv(x)
        x = self.hidden_norm(x)
        x = self.hidden_relu(x)
        x = self.hidden_pool(x)

        x = self.output_conv(x)
        x = self.output_norm(x)
        x = self.output_relu(x)
        x = self.output_pool(x)
        x = self.output_dropout(x)

        x = x.squeeze()

        x = self.input_dense(x)
        x = self.input_dense_norm(x)
        x = self.input_dense_relu(x)
        x = self.input_dropout(x)

        x = self.output_dense(x)
        x = self.softmax(x)

        return x


class ResdiualModel(nn.Module):
    def __init__(self, n_filters, n_nodes, n_conv_layers, n_dense_layers, dropout_frac):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            ResidConvLayer(1, n_filters) if idx == 0 else ResidConvLayer(n_filters, n_filters) for idx in range(n_conv_layers)
        ])
        
        self.output_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.output_dropout = nn.Dropout(dropout_frac)

        self.dense_layers = nn.ModuleList([
            FullyConnectedLayer(n_filters, n_nodes, drop_frac=dropout_frac) if idx == 0 else FullyConnectedLayer(n_nodes, n_nodes, drop_frac=dropout_frac) for idx in range(n_dense_layers)
        ])

        self.proj = nn.Linear(n_nodes, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for block in self.conv_layers:
            x = block(x)

        x = self.output_pool(x)
        x = x.squeeze()
        x = self.output_dropout(x)

        for block in self.dense_layers:
            x = block(x)

        x = self.proj(x)
        x = self.softmax(x)
        return x





class ResidualConvolutionalModel(nn.Module):
    def __init__(
        self, num_blocks=3, layers_per_block:int=2,
        kernel_size=3, starting_channels=32, 
        skip_downsample_method='stride_conv',
        dense_layers=2, starting_dense_nodes=128
    ):
        super().__init__()
        self.resid_layers = []
        current_channels = 1
        next_channels = starting_channels
        for idx in range(num_blocks):
            self.resid_layers.append(ResidBlock(current_channels, next_channels, layer_count=layers_per_block, kernel_size=kernel_size, skip_downsample_method=skip_downsample_method))
            current_channels = next_channels
            next_channels *= 2
        self.resid_layers = nn.ModuleList(self.resid_layers)

        self.output_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.output_dropout = nn.Dropout(0.2)

        self.dense_layers = []
        next_channels = starting_dense_nodes
        for idx in range(dense_layers):
            if(idx == dense_layers - 1):
                next_channels = 10
            self.dense_layers.append(FullyConnectedLayer(current_channels, next_channels))
            current_channels = next_channels
            next_channels = next_channels // 2
        self.dense_layers = nn.ModuleList(self.dense_layers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for block in self.resid_layers:
            x = block(x)

        x = self.output_pool(x)
        x = x.squeeze()
        x = self.output_dropout(x)

        for block in self.dense_layers:
            x = block(x)

        x = self.softmax(x)
        return x


class ConvolutionalModel(nn.Module):
    def __init__(
        self, num_blocks=3, layers_per_block:int=1,
        kernel_size=3, starting_channels=32,
        dense_layers=2, starting_dense_nodes=128
    ):
        super().__init__()
        self.conv_layers = []
        current_channels = 1
        next_channels = starting_channels
        for idx in range(num_blocks):
            self.conv_layers.append(ConvBlock(current_channels, next_channels, layer_count=layers_per_block, kernel_size=kernel_size))
            current_channels = next_channels
            next_channels *= 2
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.output_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.output_dropout = nn.Dropout(0.2)

        self.dense_layers = []
        next_channels = starting_dense_nodes
        for idx in range(dense_layers):
            if(idx == dense_layers - 1):
                next_channels = 10
            self.dense_layers.append(FullyConnectedLayer(current_channels, next_channels))
            current_channels = next_channels
            next_channels = next_channels // 2
        self.dense_layers = nn.ModuleList(self.dense_layers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for block in self.conv_layers:
            x = block(x)

        x = self.output_pool(x)
        x = x.squeeze()
        x = self.output_dropout(x)

        for block in self.dense_layers:
            x = block(x)

        x = self.softmax(x)
        return x
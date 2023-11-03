import torch
import torch.nn as nn
import torch.nn.functional as F

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
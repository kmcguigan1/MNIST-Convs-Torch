import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # create the conv layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_count, kernel_size=3):
        super().__init__()
        self.blocks = []
        for idx in range(layer_count):
            if(idx == 0):
                self.blocks.append(ConvLayer(
                    in_channels, out_channels, 
                    kernel_size=kernel_size, stride=2
                ))
            else:
                self.blocks.append(ConvLayer(
                    out_channels, out_channels, 
                    kernel_size=kernel_size
                ))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResidConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, skip_downsample_method='stride_conv'):
        super().__init__()
        # save some info we will need
        self.in_channels, self.out_channels = in_channels, out_channels
        # self.skip_downsample_method = skip_downsample_method
        # self.skip_channel_method = skip_channel_method
        # validate that the use supplied information makes sense
        assert self.out_channels >= self.in_channels
        # first transform block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        # second transform block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        # check if we have an identity block and need to downsample the skip connection
        self.downsample_skip_connection = False
        if(stride > 1):
            self.downsample_skip_connection = True
            if(skip_downsample_method == 'max_pool'):
                self.skip_downsampler = nn.MaxPool2d(kernel_size=stride, stride=stride)
            elif(skip_downsample_method == 'stride_conv'):
                self.skip_downsampler = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)

    def _transform_skip_connections(self, inputs):
        # check if we need to downsample the skip connections that we have
        if(self.downsample_skip_connection):
            inputs = self.skip_downsampler(inputs)
        # check if we need to pad with zero channels
        if(self.out_channels > self.in_channels):
            shape = inputs.shape
            padding = torch.zeros(size=(shape[0], self.out_channels - self.in_channels, shape[2], shape[3]), device=inputs.device)
            inputs = torch.concat((inputs, padding), axis=1)
        return inputs
    
    def forward(self, inputs):
        # conv block one
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.bn1(x)
        # conv block two
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        # return the skip connection plus what was learned by this block
        return x + self._transform_skip_connections(inputs)

class ResidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer_count, kernel_size=3, skip_downsample_method='stride_conv'):
        super().__init__()
        self.blocks = []
        for idx in range(layer_count):
            if(idx == 0):
                self.blocks.append(ResidConvLayer(
                    in_channels, out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, 
                    skip_downsample_method=skip_downsample_method
                ))
            else:
                self.blocks.append(ResidConvLayer(
                    out_channels, out_channels, 
                    kernel_size=kernel_size,  
                    skip_downsample_method=skip_downsample_method
                ))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FullyConnectedLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_frac=0.2):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(drop_frac)

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.bn(x)
        return self.drop(x)
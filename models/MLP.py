import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnected(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', batch_norm=False):
        self.dense = nn.Linear(in_channels, out_channels)
        self.activation = activation
        self.batch_norm = batch_norm
        if(self.batch_norm):
            self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.dense(x)
        if(self.activation == 'relu'):
            x = F.relu(x)
        if(self.batch_norm):
            x = self.bn(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, init_nodes=128, node_mult_per_layer=0.5, hidden_layers=4):
        super(MLP, self).__init__()
        # find the different node counts at the different blocks
        in_nodes, out_nodes = in_channels, init_nodes
        self.blocks = []
        for idx in range(hidden_layers):
            self.blocks.append(FC(in_nodes, out_nodes))
            in_nodes = out_nodes
            out_nodes = int(out_nodes * (node_mult_per_layer**idx))
        # create the projection layer
        self.projection = nn.Linear(n_nodes[-1], out_channels)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.projection(x)
        return x
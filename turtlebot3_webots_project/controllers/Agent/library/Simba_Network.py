import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

import numpy as np
from typing import Type

# Orthogonal initialization function
def orthogonal_init(gain=1.0):
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return init_weights

class MLPBlock(nn.Module):
    def __init__(self, 
                 hidden_dim,
                 activation: Type[nn.Module] = nn.ReLU,
                 dtype=torch.float32):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype)
        self.act = activation()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 activation: Type[nn.Module] = nn.ReLU,
                 dtype=torch.float32):
        super(ResidualBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype)
        self.act = activation()

        # Apply He Normal initialization to the dense layers
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='tanh' if activation == nn.Tanh else 'relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='tanh' if activation == nn.Tanh else 'relu')

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return res + x

class SACEncoder(nn.Module):
    def __init__(self, 
                 block_type,
                 input_dim, 
                 num_blocks, 
                 hidden_dim, 
                 dtype=torch.float32):
        super(SACEncoder, self).__init__()
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        if self.block_type == "mlp":
            self.encoder = MLPBlock(hidden_dim,
                                    activation=nn.ReLU,
                                    dtype=dtype)
        elif self.block_type == "residual":
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.fc.apply(orthogonal_init(1.0))
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden_dim,
                               activation=nn.ReLU, 
                               dtype=dtype) for _ in range(self.num_blocks)]
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.block_type == "mlp":
            x = self.encoder(x)
        elif self.block_type == "residual":
            x = self.fc(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
        return x

class PPOEncoder(nn.Module):
    def __init__(self, 
                 block_type,
                 input_dim, 
                 num_blocks, 
                 hidden_dim, 
                 dtype=torch.float32):
        super(PPOEncoder, self).__init__()
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        if self.block_type == "mlp":
            self.encoder = MLPBlock(hidden_dim, 
                                    activation=nn.Tanh,
                                    dtype=dtype)
        elif self.block_type == "residual":
            self.fc = nn.Linear(input_dim, hidden_dim)
            self.fc.apply(orthogonal_init(1.0))
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden_dim, 
                               activation=nn.Tanh,
                               dtype=dtype) for _ in range(self.num_blocks)]
            )
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if self.block_type == "mlp":
            x = self.encoder(x)
        elif self.block_type == "residual":
            x = self.fc(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
        return x

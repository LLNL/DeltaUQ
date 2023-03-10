# Copyright 2022 Lawrence Livermore National Security, LLC and other
# Authors: Rushil Anirudh, Jayaraman J. Thiagarajan. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import numpy as np
from math import pi
from math import cos
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from collections import OrderedDict

config = {}
config['activation'] = 'tanh'
config['hidden_dim'] = 256
config['n_layers'] = 3

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu', ff = False, mapsize = 128, scale = 2**-8, drp = -1, bn = False, config=config):
        super(MLP, self).__init__()
        if ff:
            config['ff'] = True

        self.config = config
        self.device = device

        if ff:
            config['scale'] = scale
            mn = -scale
            mx = scale
            self.B = torch.rand((mapsize, input_dim)) * (mx - mn) + mn
            self.B = self.B.to(self.device)
            input_dim = 4 * mapsize ## FF + deltaenc
        else:   
            input_dim = int(2 * input_dim)

        layers = [MLPLayer(self.config['activation'], input_dim, self.config['hidden_dim'], is_first=True, is_last=False, drp = -1, bn=False)]

        for i in range(1, self.config['n_layers'] - 1):
            layers.append(MLPLayer(self.config['activation'], self.config['hidden_dim'], self.config['hidden_dim'], is_first=False, is_last=False, drp=drp, bn=bn))

        layers.append(MLPLayer('identity', self.config['hidden_dim'], output_dim, is_first=False, is_last=True, drp = -1, bn= False))

        self.mlp = nn.Sequential(*layers)

    def input_mapping(self,x):
        x_proj = (2. * np.pi * x) @ self.B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self,x):
        out = self.mlp(x)
        return out

class MLPLayer(nn.Module):
    def __init__(self, activation, input_dim, output_dim, is_first=False, is_last=False, drp = -1, bn=False):
        super().__init__()
        self.is_first = is_first
        self.is_last = is_last
        self.bn = False
        self.drp = False

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'identity':
            self.activation_fn = nn.Identity()
        else:
            raise NotImplementedError("Only 'relu', 'tanh' and 'identity' activations are supported")

        self.linear = nn.Linear(input_dim, output_dim)

        if drp > 0:
            self.dropout = nn.Dropout(drp)
            self.drp = True
        if bn:
            self.bn = True
            self.batchnorm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        if self.is_first or self.is_last:
            return x
        else:
            if self.bn:
                x = self.batchnorm(x)  
            if self.drp:
                x = self.dropout(x)
            return x

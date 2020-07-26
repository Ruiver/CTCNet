"""
author: ouyangtianxiong
date:2019/12/24
des: this file contains basic neural network module that can be used as fundamental component
"""
import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y) ** 2 + torch.relu(x + y)


class FCNet(nn.Module):
    def __init__(self, in_size, out_size, activate=None, drop=0.0):
        super(FCNet, self).__init__()
        #self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.lin = nn.Linear(in_size, out_size)
        self.drop_value = drop
        self.drop = nn.Dropout(drop)

        # in case of using upper character by mistake
        self.activate = activate.lower() if (activate is not None) else None
        if activate == 'relu':
            self.ac_fn = nn.ReLU()
        elif activate == 'sigmoid':
            self.ac_fn = nn.Sigmoid()
        elif activate == 'tanh':
            self.ac_fn = nn.Tanh()

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)

        x = self.lin(x)

        if self.activate is not None:
            x = self.ac_fn(x)
        return x

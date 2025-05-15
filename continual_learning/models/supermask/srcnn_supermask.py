"""
Filename: ./Supermasks/models/SRCNN_Supermask.py
Author: Vincenzo Nannetti
Date: 14/03/2025
Description: Supermask super resolution convolutional network.

Usage:


Dependencies:
    - PyTorch
"""
import torch.nn as nn
from continual_learning.models.layers.mask_conv import MaskConv

class SRCNN_Supermask(nn.Module):
    def __init__(self, num_tasks):
        super(SRCNN_Supermask, self).__init__()
        self.conv1 = MaskConv(in_channels=3,  out_channels=64 ,kernel_size=9, padding=4, num_tasks=num_tasks)
        self.conv2 = MaskConv(in_channels=64, out_channels=32, kernel_size=1, padding=0, num_tasks=num_tasks)
        self.conv3 = MaskConv(in_channels=32, out_channels=3,  kernel_size=5, padding=2, num_tasks=num_tasks)

        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


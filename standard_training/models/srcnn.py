"""
    Filename: ./Super-Resolution/models/SRCNN.py
Author: Vincenzo Nannetti
Date: 04/03/2025
Description: Super-Resolution Convolutional Neural Network Model Definition

Usage:


Dependencies:
    - PyTorch
"""
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64,kernel_size=9,padding=4)
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0)
        self.conv3 = nn.Conv2d(32, 2,kernel_size=5,padding=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
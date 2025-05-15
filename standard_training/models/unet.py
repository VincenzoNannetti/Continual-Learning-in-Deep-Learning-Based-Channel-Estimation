"""
Filename: ./Autoencoder/Architectures/UNetModel.py
Author: Vincenzo Nannetti
Date: 30/03/2025
Description: U-Net Inspired Autoencoder for Channel Estimation

Usage:
    Standard UNet architecture to be used with the SRCNN model
    or as a standalone network for channel estimation.

Dependencies:
    - PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetModel(nn.Module):
    def __init__(self, in_channels=2, base_features=16):
        super(UNetModel, self).__init__()
        
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_features, 3, padding=1)
        self.enc2 = nn.Conv2d(base_features, base_features*2, 3, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(base_features*2, base_features*2, 3, padding=1)
        
        # Decoder with skip connections
        self.dec2 = nn.Conv2d(base_features*4, base_features, 3, padding=1) # base_features*4 due to skip connection
        self.dec1 = nn.Conv2d(base_features*2, in_channels, 3, padding=1)  # base_features*2 due to skip connection
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Get original dimensions
        _, _, h, w = x.size()
        
        # Encode
        e1 = self.relu(self.enc1(x))
        e1_pool = F.max_pool2d(e1, 2, 2)
        e2 = self.relu(self.enc2(e1_pool))
        e2_pool = F.max_pool2d(e2, 2, 2)
        
        # Bottleneck
        b = self.relu(self.bottleneck(e2_pool))
        
        # Decode with skip connections - use interpolate instead of upsample for exact sizing
        b_up = F.interpolate(b, size=e2.size()[2:], mode='bilinear', align_corners=True)
        d2_in = torch.cat([b_up, e2], dim=1)  # Skip connection
        d2 = self.relu(self.dec2(d2_in))
        
        d2_up = F.interpolate(d2, size=e1.size()[2:], mode='bilinear', align_corners=True)
        d1_in = torch.cat([d2_up, e1], dim=1)  # Skip connection
        d1 = self.dec1(d1_in)
        
        # Ensure output has the same dimensions as input
        if d1.size()[2:] != x.size()[2:]:
            d1 = F.interpolate(d1, size=(h, w), mode='bilinear', align_corners=True)
        
        return d1 
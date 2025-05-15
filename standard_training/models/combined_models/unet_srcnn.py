"""
Filename: ./Autoencoder/Architectures/UNetCombinedModel.py
Author: Vincenzo Nannetti
Date: 30/03/2025
Description: Combined model that uses a UNet architecture as preprocessing
             step before feeding into the super-resolution network.

Usage:
    This model provides an end-to-end solution for channel estimation,
    combining the denoising capabilities of a UNet with the 
    super-resolution benefits of SRCNN.

Dependencies:
    - PyTorch
    - UNetModel
    - SRCNN
"""
import torch
import torch.nn as nn
from ..srcnn import SRCNN
from ..unet import UNetModel
class UNetCombinedModel(nn.Module):
    def __init__(self, pretrained_unet=None, pretrained_srcnn=None, unet_args={}):
        """
        Initialise the combined model
        
        Args:
            pretrained_unet: Path to pretrained UNet model weights
            pretrained_srcnn: Path to pretrained SRCNN model weights
            unet_args (dict): Arguments to pass to the UNetModel constructor (e.g., {'base_features': 32}).
        """
        super(UNetCombinedModel, self).__init__()
        
        # Initialise the UNet, passing through any specific arguments
        # Ensure default in_channels=2 if not provided in unet_args
        unet_constructor_args = {'in_channels': 2} 
        unet_constructor_args.update(unet_args) # Overwrite defaults with passed args
        self.unet = UNetModel(**unet_constructor_args)
            
        # Initialise the SRCNN (assuming it takes no args)
        self.srcnn = SRCNN()
        
        # Load pretrained weights if provided
        if pretrained_unet:
            unet_state = torch.load(pretrained_unet)
            if 'model_state_dict' in unet_state:
                self.unet.load_state_dict(unet_state['model_state_dict'])
            else:
                self.unet.load_state_dict(unet_state)
                
        if pretrained_srcnn:
            srcnn_state = torch.load(pretrained_srcnn)
            if 'model_state_dict' in srcnn_state:
                self.srcnn.load_state_dict(srcnn_state['model_state_dict'])
            else:
                self.srcnn.load_state_dict(srcnn_state)
    
    def forward(self, x):
        """
        Forward pass through the combined model
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            Output tensor after passing through UNet and SRCNN
        """
        # Pass through UNet first
        denoised = self.unet(x)
        
        # Pass denoised output through SRCNN
        output = self.srcnn(denoised)
        
        return output
    
    def freeze_unet(self):
        """Freeze the UNet weights to only train the SRCNN part"""
        for param in self.unet.parameters():
            param.requires_grad = False
    
    def freeze_srcnn(self):
        """Freeze the SRCNN weights to only train the UNet part"""
        for param in self.srcnn.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all weights for fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
            
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_submodule_parameters(self):
        """
        Count parameters for each submodule separately
        
        Returns:
            dict: Dictionary with parameter counts for each submodule
        """
        unet_params = sum(p.numel() for p in self.unet.parameters())
        srcnn_params = sum(p.numel() for p in self.srcnn.parameters())
        
        return {
            'unet': unet_params,
            'srcnn': srcnn_params,
            'total': unet_params + srcnn_params
        } 
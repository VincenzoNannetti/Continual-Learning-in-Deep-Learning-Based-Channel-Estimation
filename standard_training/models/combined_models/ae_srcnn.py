"""
Filename: ./Autoencoder/Architectures/CombinedModel.py
Author: Vincenzo Nannetti
Date: 26/03/2025
Description: Combined model that uses a denoising autoencoder as a preprocessing 
             step before feeding into the super-resolution network.

Usage:
    This model provides an end-to-end solution for channel estimation,
    combining the denoising capabilities of an autoencoder with the 
    super-resolution benefits of SRCNN.

Dependencies:
    - PyTorch
    - DenoisingAutoencoder
    - SRCNN
"""
import torch
import torch.nn as nn
from ..srcnn import SRCNN
from ..denoising_autoencoder import DenoisingAutoencoder, DenoisingResAutoencoder

class CombinedModel_AESRCNN(nn.Module):
    def __init__(self, autoencoder_type="residual", pretrained_autoencoder=None, pretrained_srcnn=None):
        """
        Initialize the combined model
        
        Args:
            autoencoder_type: Type of autoencoder to use ("basic" or "residual")
            pretrained_autoencoder: Path to pretrained model weights (contains both AE and SRCNN)
            pretrained_srcnn: Not used when loading complete model (kept for backwards compatibility)
        """
        super(CombinedModel_AESRCNN, self).__init__()
        
        # Initialize the autoencoder
        if autoencoder_type == "basic":
            self.autoencoder = DenoisingAutoencoder(input_channels=2)
        else:
            self.autoencoder = DenoisingResAutoencoder(input_channels=2)
            
        # Initialize the SRCNN
        self.srcnn = SRCNN()
        
        # Load pretrained weights if provided
        if pretrained_autoencoder:
            try:
                print(f"Loading pretrained model from: {pretrained_autoencoder}")
                state_dict = torch.load(pretrained_autoencoder)
                
                # Handle different state dict formats
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Load the complete model state dict
                self.load_state_dict(state_dict)
                print("Successfully loaded combined model weights")
            except Exception as e:
                print(f"Error loading pretrained model: {str(e)}")
                raise
    
    def forward(self, x):
        """
        Forward pass through the combined model
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            Output tensor after passing through autoencoder and SRCNN
        """
        # Pass through autoencoder first
        denoised = self.autoencoder(x)
        
        # Pass denoised output through SRCNN
        output = self.srcnn(denoised)
        
        return output
    
    def freeze_autoencoder(self):
        """Freeze the autoencoder weights to only train the SRCNN part"""
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def freeze_srcnn(self):
        """Freeze the SRCNN weights to only train the autoencoder part"""
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
        autoencoder_params = sum(p.numel() for p in self.autoencoder.parameters())
        srcnn_params = sum(p.numel() for p in self.srcnn.parameters())
        
        return {
            'autoencoder': autoencoder_params,
            'srcnn': srcnn_params,
            'total': autoencoder_params + srcnn_params
        } 
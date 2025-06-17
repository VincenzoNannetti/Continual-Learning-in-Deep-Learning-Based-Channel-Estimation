"""
Filename: ./Autoencoder/Architectures/UNetModel.py
Author: Vincenzo Nannetti
Date: 30/03/2025
Description: U-Net Inspired Autoencoder for Channel Estimation

Usage:
    Standard UNet architecture to be used with the SRCNN model
    or as a standalone network for channel estimation.
    Now supports configurable depth and activation functions with proper powers-of-2 architecture.

Dependencies:
    - PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetModel(nn.Module):
    def __init__(self, in_channels=2, base_features=16, use_batch_norm=False, 
                 depth=3, activation='leakyrelu', use_leaky_relu=False, leaky_slope=0.01, verbose=False):
        super(UNetModel, self).__init__()
        
        # Validation
        assert depth >= 1, "UNetModel depth must be at least 1"
        assert base_features > 0, "base_features must be positive"
        
        self.use_batch_norm = use_batch_norm
        self.depth = depth
        self.in_channels = in_channels
        self.base_features = base_features
                    
        self.activation_type = activation.lower()
        
        # Determine activation function
        if self.activation_type == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=leaky_slope)
        elif self.activation_type == 'gelu':
            self.activation = nn.GELU()
        elif self.activation_type == 'swish' or self.activation_type == 'silu':
            self.activation = nn.SiLU()
        elif self.activation_type == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ReLU()  # Default fallback
            self.activation_type = 'relu'
            
        if verbose:
            print(f"UNet using {self.activation_type.upper()} activation with depth {depth}")
        
        # Build encoder layers with proper powers of 2
        self.encoder_convs = nn.ModuleList()
        self.encoder_bns = nn.ModuleList() if use_batch_norm else None
        
        current_channels = in_channels
        self.encoder_channels = []  # Track encoder output channels for skip connections
        
        for i in range(depth):
            out_channels = base_features * (2 ** i)  # 16, 32, 64, 128, ...
            conv = nn.Conv2d(current_channels, out_channels, 3, padding=1)
            self.encoder_convs.append(conv)
            self.encoder_channels.append(out_channels)
            
            if use_batch_norm:
                self.encoder_bns.append(nn.BatchNorm2d(out_channels))
                
            current_channels = out_channels
        
        # Bottleneck - one level deeper in channel dimension
        bottleneck_channels = base_features * (2 ** depth)  # Next power of 2
        self.bottleneck = nn.Conv2d(current_channels, bottleneck_channels, 3, padding=1)
        if use_batch_norm:
            self.bottleneck_bn = nn.BatchNorm2d(bottleneck_channels)
        
        # Build decoder layers with symmetric powers of 2
        self.decoder_convs = nn.ModuleList()
        self.decoder_reduce_convs = nn.ModuleList()  # To reduce concatenated channels back to powers of 2
        self.decoder_bns = nn.ModuleList() if use_batch_norm else None
        self.decoder_reduce_bns = nn.ModuleList() if use_batch_norm else None
        
        current_channels = bottleneck_channels
        
        for i in range(depth):
            # Calculate skip connection channels from corresponding encoder
            skip_channels = self.encoder_channels[-(i+1)]  # Reverse order
            concat_channels = current_channels + skip_channels  # After concatenation
            
            # Target output channels - symmetric powers of 2
            if i == depth - 1:  # Final layer
                target_channels = base_features  # Will be reduced to in_channels separately
            else:
                target_channels = base_features * (2 ** (depth - 1 - i))
            
            # First conv: reduce concatenated channels to target channels (maintaining powers of 2)
            reduce_conv = nn.Conv2d(concat_channels, target_channels, 3, padding=1)
            self.decoder_reduce_convs.append(reduce_conv)
            
            # Second conv: refine features (channel count stays the same)
            refine_conv = nn.Conv2d(target_channels, target_channels, 3, padding=1)
            self.decoder_convs.append(refine_conv)
            
            # Batch norm for both convs (except final output)
            if use_batch_norm:
                self.decoder_reduce_bns.append(nn.BatchNorm2d(target_channels))
                if i < depth - 1:  # No BN on final layer
                    self.decoder_bns.append(nn.BatchNorm2d(target_channels))
                else:
                    self.decoder_bns.append(None)  # Placeholder
                
            current_channels = target_channels
        
        # Final 1x1 conv to map base_features → in_channels
        self.final_conv = nn.Conv2d(base_features, in_channels, 1, padding=0)
    
    def forward(self, x):
        # Get original dimensions
        _, _, h, w = x.size()
        
        # Encoder path - store features for skip connections
        encoder_features = []
        current = x
        
        for i, conv in enumerate(self.encoder_convs):
            current = conv(current)
            
            if self.use_batch_norm and self.encoder_bns is not None:
                current = self.encoder_bns[i](current)
                
            current = self.activation(current)
            encoder_features.append(current)  # Store for skip connection
            
            # Apply pooling except for the last encoder layer
            if i < len(self.encoder_convs) - 1:
                current = F.max_pool2d(current, 2, 2)
        
        # Bottleneck
        current = self.bottleneck(current)
        if self.use_batch_norm and hasattr(self, 'bottleneck_bn'):
            current = self.bottleneck_bn(current)
        current = self.activation(current)
        
        # Decoder path with skip connections and symmetric channel reduction
        for i in range(self.depth):
            # Upsample to match the corresponding encoder feature size
            encoder_feature = encoder_features[-(i+1)]  # Get corresponding encoder feature
            current = F.interpolate(current, size=encoder_feature.size()[2:], mode='bilinear', align_corners=True)
            
            # Concatenate with skip connection
            current = torch.cat([current, encoder_feature], dim=1)
            
            # First conv: reduce concatenated channels to target power-of-2
            current = self.decoder_reduce_convs[i](current)
            if self.use_batch_norm and self.decoder_reduce_bns is not None:
                current = self.decoder_reduce_bns[i](current)
            current = self.activation(current)
            
            # Second conv: refine features
            current = self.decoder_convs[i](current)
            
            # Apply batch norm and activation (except for final layer)
            if i < self.depth - 1:
                if self.use_batch_norm and self.decoder_bns is not None and self.decoder_bns[i] is not None:
                    current = self.decoder_bns[i](current)
                current = self.activation(current)
        
        # Final 1x1 conv to map to output channels
        current = self.final_conv(current)
        
        # Ensure output has the same dimensions as input
        if current.size()[2:] != x.size()[2:]:
            current = F.interpolate(current, size=(h, w), mode='bilinear', align_corners=True)
        
        return current 
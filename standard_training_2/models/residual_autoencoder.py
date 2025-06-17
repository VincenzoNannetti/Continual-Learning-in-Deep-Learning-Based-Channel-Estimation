import torch
import torch.nn as nn

class DenoisingResAutoencoder(nn.Module):
    # version with residual connections

    def __init__(self, input_channels=3, feature_maps=32, kernel_size=3, dropout_rate=0.1):
        super(DenoisingResAutoencoder, self).__init__()
        
        padding = kernel_size // 2
        
        # Encoder blocks with residual connections
        self.enc1_conv = nn.Conv2d(input_channels, feature_maps, kernel_size, padding=padding)
        self.enc1_bn   = nn.BatchNorm2d(feature_maps)
        self.enc1_act  = nn.LeakyReLU(0.2, inplace=True)
        
        self.enc2_conv = nn.Conv2d(feature_maps, feature_maps*2, kernel_size, padding=padding)
        self.enc2_bn   = nn.BatchNorm2d(feature_maps*2)
        self.enc2_act  = nn.LeakyReLU(0.2, inplace=True)
        
        self.enc3_conv = nn.Conv2d(feature_maps*2, feature_maps*4, kernel_size, padding=padding)
        self.enc3_bn   = nn.BatchNorm2d(feature_maps*4)
        self.enc3_act  = nn.LeakyReLU(0.2, inplace=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_maps*4, feature_maps*4, kernel_size, padding=padding),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder blocks with residual connections and skip connections from encoder
        self.dec3_conv = nn.Conv2d(feature_maps*8, feature_maps*2, kernel_size, padding=padding)
        self.dec3_bn   = nn.BatchNorm2d(feature_maps*2)
        self.dec3_act  = nn.LeakyReLU(0.2, inplace=True)
        
        self.dec2_conv = nn.Conv2d(feature_maps*4, feature_maps, kernel_size, padding=padding)
        self.dec2_bn   = nn.BatchNorm2d(feature_maps)
        self.dec2_act  = nn.LeakyReLU(0.2, inplace=True)
        
        self.dec1_conv = nn.Conv2d(feature_maps*2, input_channels, kernel_size, padding=padding)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # Encoder with residual connections and storing activations for skip connections
        e1 = self.enc1_act(self.enc1_bn(self.enc1_conv(x)))
        e1 = self.dropout(e1)
        
        e2 = self.enc2_act(self.enc2_bn(self.enc2_conv(e1)))
        e2 = self.dropout(e2)
        
        e3 = self.enc3_act(self.enc3_bn(self.enc3_conv(e2)))
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder with skip connections
        # Concatenate bottleneck with encoder output
        d3 = torch.cat([b, e3], dim=1)
        d3 = self.dec3_act(self.dec3_bn(self.dec3_conv(d3)))
        d3 = self.dropout(d3)
        
        # Concatenate with encoder output
        d2 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2_act(self.dec2_bn(self.dec2_conv(d2)))
        d2 = self.dropout(d2)
        
        # Concatenate with encoder output
        d1 = torch.cat([d2, e1], dim=1)
        out = self.dec1_conv(d1)
        
        # Add residual connection from input to output
        return out + x  # Residual learning - learn the noise rather than denoised signal 
    
    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
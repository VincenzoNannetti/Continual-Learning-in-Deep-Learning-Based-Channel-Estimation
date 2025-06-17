import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels=2, feature_maps=32, kernel_size=3, dropout_rate=0.2):
        """
        Denoising Autoencoder optimised for channel matrices
        
        Args:
            input_channels: Number of input channels (typically 2 for complex data - real, imaginary)
            feature_maps: Base number of feature maps (filters)
            kernel_size: Size of convolutional kernels
            dropout_rate: Dropout rate for regularization
        """
        super(DenoisingAutoencoder, self).__init__()
        
        padding = kernel_size // 2  # Same padding
        
        # Encoder pathway
        self.encoder = nn.Sequential(
            # First encoding block
            nn.Conv2d(input_channels, feature_maps, kernel_size, padding=padding),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Second encoding block - increase feature maps
            nn.Conv2d(feature_maps, feature_maps*2, kernel_size, padding=padding),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Third encoding block - bottleneck
            nn.Conv2d(feature_maps*2, feature_maps*4, kernel_size, padding=padding),
            nn.BatchNorm2d(feature_maps*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder pathway
        self.decoder = nn.Sequential(
            # First decoding block
            nn.Conv2d(feature_maps*4, feature_maps*2, kernel_size, padding=padding),
            nn.BatchNorm2d(feature_maps*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Second decoding block
            nn.Conv2d(feature_maps*2, feature_maps, kernel_size, padding=padding),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Output layer - reconstruct input channels
            nn.Conv2d(feature_maps, input_channels, kernel_size, padding=padding)
        )
        
    def forward(self, x):
        """Forward pass through the network"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_encoded_representation(self, x):
        """Get the bottleneck representation for external use"""
        return self.encoder(x)
    
    def count_parameters(self):
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
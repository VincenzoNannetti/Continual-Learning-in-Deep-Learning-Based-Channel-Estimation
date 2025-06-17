import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self, channels=[64, 32], kernels=[9, 1, 5]):
        super(SRCNN, self).__init__()
        
        # Ensure we have the right number of channels and kernels
        assert len(channels) == 2, "channels should contain exactly 2 values [conv1_out, conv2_out]"
        assert len(kernels) == 3, "kernels should contain exactly 3 values [conv1_kernel, conv2_kernel, conv3_kernel]"
        
        # Calculate padding to maintain spatial dimensions
        # For odd kernel sizes: padding = (kernel_size - 1) // 2
        pad1 = (kernels[0] - 1) // 2
        pad2 = (kernels[1] - 1) // 2
        pad3 = (kernels[2] - 1) // 2
        
        self.conv1 = nn.Conv2d(2, channels[0], kernel_size=kernels[0], padding=pad1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=kernels[1], padding=pad2)
        self.conv3 = nn.Conv2d(channels[1], 2, kernel_size=kernels[2], padding=pad3)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
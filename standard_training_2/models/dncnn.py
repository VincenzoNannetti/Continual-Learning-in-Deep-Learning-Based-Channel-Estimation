import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, num_channels=2):
        super(DnCNN, self).__init__()

        layers = [nn.Conv2d(num_channels, 64, kernel_size=3, padding=1, bias=False),nn.ReLU(inplace=True)]

        for _ in range(18):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, num_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise
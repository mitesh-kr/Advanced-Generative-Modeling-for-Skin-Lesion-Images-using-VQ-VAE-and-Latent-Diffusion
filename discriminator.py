import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.im_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            activation
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            activation
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            activation
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Identity()  # No activation or batch normalization for the last layer
        )

    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

"""
Defines a lightweight 2D U-Net and Dice loss for medical image segmentation tasks
Author: Isaac Tiang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic 2-layer convolution block with normalization & ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


# 2D U-Net model
class UNet2D(nn.Module):
    """
    A simplified 2D U-Net with skip connections
    Each downsampling halves the resolution, and
    each upsampling restores it via transposed convs
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 2, base_channels)

        # Output
        self.output_layer = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Down path
        d1 = self.enc1(x)
        d2 = self.enc2(self.pool(d1))
        d3 = self.enc3(self.pool(d2))
        d4 = self.enc4(self.pool(d3))

        # Up path with skip connections
        x = self.up1(d4)
        x = self.dec1(torch.cat([x, d3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, d2], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, d1], dim=1))

        return self.output_layer(x)


# Dice Loss implementation
class DiceLoss(nn.Module):
    def __init__(self, n_classes=6, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Remove channel dim if present
        if targets.ndim == 4:
            targets = targets.squeeze(1)
        # One-hot encode targets
        one_hot = F.one_hot(targets, num_classes=self.n_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        inputs = torch.softmax(inputs, dim=1)
        dims = (0, 2, 3)
        intersection = torch.sum(inputs * one_hot, dims)
        cardinality = torch.sum(inputs + one_hot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()

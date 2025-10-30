"""
Defines a lightweight 2D U-Net and Dice loss for medical image segmentation tasks
Author: Isaac Tiang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Residual block with dropout (2D)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # Match channels if needed
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


# Improved 2D U-Net
class ImprovedUNet2D(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, base_channels=32, dropout_prob=0.3):
        super(ImprovedUNet2D, self).__init__()

        # Encoder
        self.encoder1 = ResidualBlock(in_channels, base_channels, dropout_prob)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ResidualBlock(base_channels, base_channels * 2, dropout_prob)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ResidualBlock(base_channels * 2, base_channels * 4, dropout_prob)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ResidualBlock(base_channels * 4, base_channels * 8, dropout_prob)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualBlock(base_channels * 8, base_channels * 4, dropout_prob)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(base_channels * 4, base_channels * 2, dropout_prob)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(base_channels * 2, base_channels, dropout_prob)

        # Output
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoding path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Decoding path with skip connections
        dec3 = self.upconv3(enc4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


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

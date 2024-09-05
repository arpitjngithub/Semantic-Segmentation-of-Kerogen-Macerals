# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional block to be used in autoencoder and U-Net
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# Encoder block: Conv block followed by maxpooling
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p

# Decoder block for autoencoder (no skip connections)
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels, out_channels)
        
    def forward(self, x):
        x = self.upconv(x)
        x = self.conv(x)
        return x

# Encoder will be the same for Autoencoder and U-net
# We are getting both conv output and maxpool output for convenience.
# we will ignore conv output for Autoencoder. It acts as skip connections for U-Net
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        self.bridge = ConvBlock(512, 1024)
        
    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)
        b = self.bridge(p4)
        return s1, s2, s3, s4, b

# Decoder for Autoencoder ONLY
class AutoencoderDecoder(nn.Module):
    def __init__(self):
        super(AutoencoderDecoder, self).__init__()
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.output = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = torch.sigmoid(self.output(x))
        return x

# Use encoder and decoder blocks to build the autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = AutoencoderDecoder()
        
    def forward(self, x):
        _, _, _, _, encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Decoder block for U-Net
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)
        
    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x

# Build U-Net using the blocks
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.dec1 = UNetDecoderBlock(1024, 512)
        self.dec2 = UNetDecoderBlock(512, 256)
        self.dec3 = UNetDecoderBlock(256, 128)
        self.dec4 = UNetDecoderBlock(128, 64)
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        s1, s2, s3, s4, b = self.encoder(x)
        x = self.dec1(b, s4)
        x = self.dec2(x, s3)
        x = self.dec3(x, s2)
        x = self.dec4(x, s1)
        x = torch.sigmoid(self.output(x))
        return x

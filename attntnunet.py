import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import random




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=False):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=False, dropout=0.0):
        super(ResConvBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, padding, batch_norm)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, padding, batch_norm)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut(x)
        if self.batch_norm:
            shortcut = self.batch_norm(shortcut)

        conv = self.conv1(x)
        if self.dropout:
            conv = self.dropout(conv)
        conv = self.conv2(conv)

        res_path = conv + shortcut
        res_path = self.relu(res_path)
        return res_path

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0)
        self.phi_g = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.up(self.phi_g(g))
        concat_xg = theta_x + phi_g
        act_xg = self.relu(concat_xg)
        psi = self.sigmoid(self.psi(act_xg))
        upsample_psi = self.up(psi)
        upsample_psi = upsample_psi.expand_as(x)
        return upsample_psi * x

class AttentionResUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, dropout_rate=0.0, batch_norm=True):
        super(AttentionResUNet, self).__init__()
        self.filter_num = 64
        self.filter_size = 3
        self.up_samp_size = 2

        self.down1 = ResConvBlock(input_channels, self.filter_num, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = ResConvBlock(self.filter_num, self.filter_num * 2, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = ResConvBlock(self.filter_num * 2, self.filter_num * 4, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = ResConvBlock(self.filter_num * 4, self.filter_num * 8, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.center = ResConvBlock(self.filter_num * 8, self.filter_num * 16, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)

        self.gate4 = ConvBlock(self.filter_num * 16, self.filter_num * 8, kernel_size=1, padding=0, batch_norm=batch_norm)
        self.att4 = AttentionBlock(self.filter_num * 8, self.filter_num * 8, self.filter_num * 4)
        self.up4 = nn.ConvTranspose2d(self.filter_num * 16, self.filter_num * 8, kernel_size=2, stride=2)
        self.up_conv4 = ResConvBlock(self.filter_num * 16, self.filter_num * 8, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)

        self.gate3 = ConvBlock(self.filter_num * 8, self.filter_num * 4, kernel_size=1, padding=0, batch_norm=batch_norm)
        self.att3 = AttentionBlock(self.filter_num * 4, self.filter_num * 4, self.filter_num * 2)
        self.up3 = nn.ConvTranspose2d(self.filter_num * 8, self.filter_num * 4, kernel_size=2, stride=2)
        self.up_conv3 = ResConvBlock(self.filter_num * 8, self.filter_num * 4, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)

        self.gate2 = ConvBlock(self.filter_num * 4, self.filter_num * 2, kernel_size=1, padding=0, batch_norm=batch_norm)
        self.att2 = AttentionBlock(self.filter_num * 2, self.filter_num * 2, self.filter_num)
        self.up2 = nn.ConvTranspose2d(self.filter_num * 4, self.filter_num * 2, kernel_size=2, stride=2)
        self.up_conv2 = ResConvBlock(self.filter_num * 4, self.filter_num * 2, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)

        self.gate1 = ConvBlock(self.filter_num * 2, self.filter_num, kernel_size=1, padding=0, batch_norm=batch_norm)
        self.att1 = AttentionBlock(self.filter_num, self.filter_num, self.filter_num // 2)
        self.up1 = nn.ConvTranspose2d(self.filter_num * 2, self.filter_num, kernel_size=2, stride=2)
        self.up_conv1 = ResConvBlock(self.filter_num * 2, self.filter_num, self.filter_size, batch_norm=batch_norm, dropout=dropout_rate)

        self.final_conv = nn.Conv2d(self.filter_num, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        down1 = self.down1(x)
        pool1 = self.pool1(down1)
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)
        down4 = self.down4(pool3)
        pool4 = self.pool4(down4)
        center = self.center(pool4)

        gate4 = self.gate4(center)
        att4 = self.att4(down4, gate4)
        up4 = self.up4(center)
        up4 = torch.cat([up4, att4], dim=1)
        up_conv4 = self.up_conv4(up4)

        gate3 = self.gate3(up_conv4)
        att3 = self.att3(down3, gate3)
        up3 = self.up3(up_conv4)
        up3 = torch.cat([up3, att3], dim=1)
        up_conv3 = self.up_conv3(up3)

        gate2 = self.gate2(up_conv3)
        att2 = self.att2(down2, gate2)
        up2 = self.up2(up_conv3)
        up2 = torch.cat([up2, att2], dim=1)
        up_conv2 = self.up_conv2(up2)

        gate1 = self.gate1(up_conv2)
        att1 = self.att1(down1, gate1)
        up1 = self.up1(up_conv2)
        up1 = torch.cat([up1, att1], dim=1)
        up_conv1 = self.up_conv1(up1)

        final = self.final_conv(up_conv1)
        return self.sigmoid(final)

"""Resnet implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import numpy as np

def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    return getattr(m, class_name)


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride, linear_output=False):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)
        self.linear_output = linear_output

    def forward(self, x):
        p = int(np.floor((self.kernel_size - 1) / 2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        if self.linear_output:
            return x
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class ResUNet(nn.Module):
    def __init__(self, encoder='resnet50', pretrained=True, num_in_layers=4, num_out_layers=2, linear_output=False):
        super(ResUNet, self).__init__()
        self.pretrained = pretrained
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        if num_in_layers != 3:  # Number of input channels
            self.firstconv = nn.Conv2d(num_in_layers, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        else:
            self.firstconv = resnet.conv1 # H/2

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool # H/4

        # encoder
        self.encoder1 = resnet.layer1 # H/4
        self.encoder2 = resnet.layer2 # H/8
        self.encoder3 = resnet.layer3 # H/16

        # decoder
        self.upconv4 = upconv(filters[2], 512, 3, 2)
        self.iconv4 = conv(filters[1] + 512, 512, 3, 1)

        self.upconv3 = upconv(512, 256, 3, 2)
        self.iconv3 = conv(filters[0] + 256, 256, 3, 1)

        self.outconv = conv(256, num_out_layers, 1, 1, linear_output=linear_output)
        # self.outconv = conv(256, num_out_layers, 1, 1

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x, return_ft=False):
        # encoding
        x1 = self.firstconv(x)
        x1 = self.firstbn(x1)
        x1 = self.firstrelu(x1)
        x1 = self.firstmaxpool(x1)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        # decoding
        x = self.upconv4(x4)
        x = self.skipconnect(x3, x)
        x = self.iconv4(x)

        x = self.upconv3(x)
        x = self.skipconnect(x2, x)
        ft = self.iconv3(x)
        x = self.outconv(ft)
        if return_ft:
            return x, ft
        else:
            return x


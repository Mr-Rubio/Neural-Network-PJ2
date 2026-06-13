"""Custom CNN for CIFAR-10 classification."""

import numpy as np
import torch.nn as nn

from utils.nn import init_weights_


def get_number_of_parameters(model):
    return sum(np.prod(p.shape).item() for p in model.parameters())


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation='relu'):
        super().__init__()
        act = nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            act,
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    Custom CNN with Conv2d, BatchNorm, ReLU/GELU, MaxPool, Dropout, and Linear layers.
    Supports base and wide channel configurations.
    """

    def __init__(self, num_classes=10, channels=(64, 128, 256), fc_dim=512,
                 activation='relu', dropout=0.5):
        super().__init__()
        c1, c2, c3 = channels
        self.stage1 = nn.Sequential(
            ConvBlock(3, c1, activation),
            ConvBlock(c1, c1, activation),
            nn.MaxPool2d(2, 2),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(c1, c2, activation),
            ConvBlock(c2, c2, activation),
            nn.MaxPool2d(2, 2),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(c2, c3, activation),
            ConvBlock(c3, c3, activation),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * 4 * 4, fc_dim),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)

    def _init_weights(self):
        for m in self.modules():
            init_weights_(m)

    @property
    def first_conv(self):
        return self.stage1[0].block[0]


def build_model(variant='base', activation='relu', dropout=0.5):
    configs = {
        'base': {'channels': (64, 128, 256), 'fc_dim': 512},
        'wide': {'channels': (128, 256, 512), 'fc_dim': 512},
    }
    cfg = configs.get(variant, configs['base'])
    return CustomCNN(activation=activation, dropout=dropout, **cfg)

from __future__ import annotations
import torch
from torch import nn
from torchvision.models import resnet18, mobilenet_v2


class ConvBN(nn.Sequential):
    def __init__(self, a, b, stride=1, groups=1):
        super().__init__(nn.Conv2d(a, b, 3, stride=stride, padding=1, groups=groups, bias=False),
                         nn.BatchNorm2d(b), nn.ReLU(inplace=False))


class LeNetSmall(nn.Module):
    """LeNet-inspired ReLU CNN; NOT an exact reproduction of historical LeNet-5."""
    def __init__(self, channels=1, classes=10, width=16, dropout=.2):
        super().__init__()
        self.features = nn.Sequential(ConvBN(channels, width), nn.MaxPool2d(2),
                                      ConvBN(width, width*2), nn.MaxPool2d(2),
                                      nn.AdaptiveAvgPool2d((7, 7)))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(width*2*7*7, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, classes))

    def forward(self, x):
        return self.head(self.features(x))


class TinyVGG(nn.Module):
    """Three small VGG-style blocks; NOT VGG16."""
    def __init__(self, channels=3, classes=10, width=16, dropout=.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBN(channels, width), ConvBN(width, width), nn.MaxPool2d(2),
            ConvBN(width, width*2), ConvBN(width*2, width*2), nn.MaxPool2d(2),
            ConvBN(width*2, width*4), ConvBN(width*4, width*4), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width*4, classes))

    def forward(self, x):
        return self.head(self.features(x))


class Separable(nn.Sequential):
    def __init__(self, a, b, stride=1):
        super().__init__(ConvBN(a, a, stride, groups=a),
                         nn.Conv2d(a, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU())


class TinyMobile(nn.Module):
    """Small depthwise-separable MobileNet-inspired model, random initialization."""
    def __init__(self, channels=3, classes=2, width=24, dropout=.2):
        super().__init__()
        self.features = nn.Sequential(ConvBN(channels, width, 2),
            Separable(width, width*2), Separable(width*2, width*4, 2),
            Separable(width*4, width*4), Separable(width*4, width*6, 2),
            Separable(width*6, width*6), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(width*6, classes))

    def forward(self, x):
        return self.head(self.features(x))


class DenseLayer(nn.Module):
    def __init__(self, channels, growth):
        super().__init__()
        self.f = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(),
                               nn.Conv2d(channels, growth, 3, padding=1, bias=False))

    def forward(self, x):
        return torch.cat((x, self.f(x)), dim=1)


class TinyDenseNet(nn.Module):
    """Three dense blocks, four layers/block, growth=12; NOT DenseNet121."""
    def __init__(self, channels=3, classes=10, dropout=.2):
        super().__init__()
        n = 24
        layers = [nn.Conv2d(channels, n, 3, padding=1, bias=False)]
        for block in range(3):
            for _ in range(4):
                layers.append(DenseLayer(n, 12))
                n += 12
            if block != 2:
                out = n // 2
                layers += [nn.BatchNorm2d(n), nn.ReLU(), nn.Conv2d(n, out, 1, bias=False),
                           nn.AvgPool2d(2)]
                n = out
        layers += [nn.BatchNorm2d(n), nn.ReLU(), nn.AdaptiveAvgPool2d(1)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(n, classes))

    def forward(self, x):
        return self.head(self.features(x))


NAMES = {"lenet_small": "LeNet-inspired small CNN", "tiny_vgg": "Tiny VGG-style CNN",
         "tiny_mobile": "Tiny MobileNet-style CNN", "resnet18_cifar": "ResNet18 (CIFAR stem)",
         "mobilenet_v2_half": "MobileNetV2 width=0.5 (CIFAR stem)",
         "tiny_densenet": "Tiny DenseNet (3 blocks, growth=12)"}


def build_model(name, channels, classes, width=16, dropout=.2):
    if name == "lenet_small":
        return LeNetSmall(channels, classes, width, dropout)
    if name == "tiny_vgg":
        return TinyVGG(channels, classes, width, dropout)
    if name == "tiny_mobile":
        return TinyMobile(channels, classes, width, dropout)
    if name == "tiny_densenet":
        return TinyDenseNet(channels, classes, dropout)
    if name == "resnet18_cifar":
        m = resnet18(weights=None)  # Deliberately no hidden weight download.
        m.conv1 = nn.Conv2d(channels, 64, 3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, classes))
        return m
    if name == "mobilenet_v2_half":
        m = mobilenet_v2(weights=None, width_mult=.5, num_classes=classes, dropout=dropout)
        old = m.features[0][0]
        m.features[0][0] = nn.Conv2d(channels, old.out_channels, 3, stride=1, padding=1, bias=False)
        return m
    raise ValueError(f"Unknown model {name}; choose from {list(NAMES)}")

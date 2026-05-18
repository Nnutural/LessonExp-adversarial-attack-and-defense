"""Model definitions."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18

from .constants import CIFAR10_MEAN, CIFAR10_STD


class Normalize(nn.Module):
    """Channel-wise input normalization stored as model buffers."""

    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class CifarResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.normalize = Normalize(CIFAR10_MEAN, CIFAR10_STD)
        self.backbone = resnet18(weights=None, num_classes=num_classes)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))


def build_resnet18(num_classes: int = 10) -> CifarResNet18:
    return CifarResNet18(num_classes=num_classes)

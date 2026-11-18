"""PyTorch models for CIFAR-10 experiments."""

from typing import cast

import torch
from torch import nn


class LinearBaseline(nn.Module):
    """Single linear layer applied to flattened RGB images."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.classifier(images))


class SimpleCNN(nn.Module):
    """Small convolutional neural network for CIFAR-10."""

    def __init__(
        self,
        channels: tuple[int, int] = (32, 64),
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if any(channel <= 0 for channel in channels):
            raise ValueError("Channel counts must be positive")

        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[1] * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = cast(torch.Tensor, self.features(images))
        return cast(torch.Tensor, self.classifier(features))


def build_model(
    model_type: str,
    channels: list[int] | None = None,
    num_classes: int = 10,
) -> nn.Module:
    """Build a model using the experiment configuration."""

    if model_type == "linear":
        return LinearBaseline(num_classes=num_classes)

    if model_type == "simple_cnn":
        if channels is None:
            channels = [32, 64]

        if len(channels) != 2:
            raise ValueError("SimpleCNN requires exactly two channel counts")

        return SimpleCNN(
            channels=(channels[0], channels[1]),
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model type: {model_type}")

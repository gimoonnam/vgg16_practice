import torch
import torch.nn as nn


class ResNetLayer(nn.Module):
    """
    This a class for defining the N filters
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        batch_normalization: bool = True,
    ):
        super().__init__()
        model = []
        model.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(1, 1),
                stride=stride,
            )
        )

        if batch_normalization:
            model.append(nn.BatchNorm2d(out_channels))

        model.append(nn.ReLU(True))

        self.conv = nn.Sequential(*model)

    def forward(self, x):
        return self.conv(x)



import torch
import torch.nn as nn


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=7,
        stride=1
    ):
        super().__init__()

        padding = kernel_size // 2

        # Depthwise convolution:
        # one spatial filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(in_channels)

        # Pointwise convolution:
        # mixes information between channels
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)

        return x


class MobileNetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):
        super().__init__()

        self.conv = DepthwiseSeparableConv1D(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride
        )

    def forward(self, x):
        return self.conv(x)


class MobileNet1D(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.features = nn.Sequential(

            MobileNetBlock1D(
                32, 32, stride=1
            ),

            MobileNetBlock1D(
                32, 64, stride=2
            ),

            MobileNetBlock1D(
                64, 64, stride=1
            ),

            MobileNetBlock1D(
                64, 128, stride=2
            ),

            MobileNetBlock1D(
                128, 128, stride=1
            ),

            MobileNetBlock1D(
                128, 256, stride=2
            ),

            MobileNetBlock1D(
                256, 256, stride=1
            ),

            MobileNetBlock1D(
                256, 512, stride=2
            ),

            MobileNetBlock1D(
                512, 512, stride=1
            )
        )

        # Compact feature representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.global_pool(x)

        return self.regressor(x)
import torch
import torch.nn as nn
import yaml

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        # If dimensions change, transform the residual connection.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()

        # Initial feature extraction
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
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual stages
        self.layer1 = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 32)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock1D(64, 128, stride=2),
            ResidualBlock1D(128, 128)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock1D(128, 256, stride=2),
            ResidualBlock1D(256, 256)
        )

        # Global pooling gives a compact representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 256-dimensional embedding
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)

        return self.regressor(x)
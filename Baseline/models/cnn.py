import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 (large receptive field)
            nn.Conv1d(1, 16, kernel_size=101, padding=50),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=51, padding=25),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(32, 64, kernel_size=25, padding=12),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(64, 64, kernel_size=15, padding=7),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.MaxPool1d(2),

            # Block 5 (refinement)
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.01),

            # Global pooling
            nn.AdaptiveAvgPool1d(256)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 256, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
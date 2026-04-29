import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(32, 64, kernel_size=31, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(64, 64, kernel_size=31, stride=1, padding=15),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Compress to fixed size
            nn.AdaptiveAvgPool1d(256)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # predicts log10(nu_max)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
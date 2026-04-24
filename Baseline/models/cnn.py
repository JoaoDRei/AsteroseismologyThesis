import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Flatten(),
            nn.Linear(32 * 507, 128), # Note: 507 depends on your input length
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.conv(x)
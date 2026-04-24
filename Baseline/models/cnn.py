import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            # This ensures the output is always 512 units wide before the Linear layer
            nn.AdaptiveAvgPool1d(512) 
        )
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 512, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size=35000):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 1)  # predicts log10(nu_max)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
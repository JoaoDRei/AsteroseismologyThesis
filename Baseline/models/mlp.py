import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size=35000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1) # Outputs log(v_max)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten (Batch, 1, Len) -> (Batch, Len)
        return self.net(x)
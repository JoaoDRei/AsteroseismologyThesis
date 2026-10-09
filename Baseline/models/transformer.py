import torch
import torch.nn as nn

class SimpleTransformer(nn.Module):
    def __init__(self, seq_length, d_model=64, nhead=4, num_layers=3, output_dim=1):
        super().__init__()

        # Step 1: project scalar → embedding
        self.input_proj = nn.Linear(1, d_model)

        # Step 2: positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, d_model))

        # Step 3: transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Step 4: regression head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        # x: (B, 1, L)
        x = x.transpose(1, 2)       # → (B, L, 1)
        x = self.input_proj(x)      # → (B, L, D)

        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer(x)     # → (B, L, D)

        x = x.mean(dim=1)           # global pooling

        return self.regressor(x)
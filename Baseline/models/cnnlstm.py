import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN + LSTM model for 1D stellar light curves / power spectra.

    The CNN extracts local features and progressively downsamples
    the input sequence. The LSTM then models dependencies between
    the resulting higher-level feature vectors.
    """

    def __init__(self, output_dim=1, hidden_size=128, num_layers=2):
        super().__init__()

        # ---------------------------------------------------------
        # 1. CNN feature extractor
        # ---------------------------------------------------------
        self.cnn = nn.Sequential(

            # Input:  (B, 1, ~65000)
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01, inplace=True),

            # ~65000 -> ~32500
            nn.MaxPool1d(kernel_size=2),

            # ~32500 -> ~16250
            nn.Conv1d(
                32, 64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),

            # ~16250 -> ~8125
            nn.MaxPool1d(kernel_size=2),

            # ~8125 -> ~4063
            nn.Conv1d(
                64, 128,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),

            # ~4063 -> ~2032
            nn.MaxPool1d(kernel_size=2),

            # ~2032 -> ~1016
            nn.Conv1d(
                128, 128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False
            ),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ---------------------------------------------------------
        # 2. Reduce sequence length before LSTM
        # ---------------------------------------------------------
        #
        # We don't want to give the LSTM ~1000-2000 time steps
        # unnecessarily. Adaptive pooling gives us a controlled
        # sequence length.
        #
        # Output shape:
        # (B, 128, 256)
        #
        self.sequence_pool = nn.AdaptiveAvgPool1d(256)

        # ---------------------------------------------------------
        # 3. LSTM
        # ---------------------------------------------------------
        #
        # After permutation:
        # (B, 256, 128)
        #
        # Therefore:
        #   sequence length = 256
        #   input features   = 128
        #
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # ---------------------------------------------------------
        # 4. Prediction head
        # ---------------------------------------------------------
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):

        # ---------------------------------------------------------
        # CNN
        # ---------------------------------------------------------
        # (B, 1, L)
        x = self.cnn(x)

        # (B, 128, L')
        x = self.sequence_pool(x)

        # ---------------------------------------------------------
        # Prepare for LSTM
        # ---------------------------------------------------------
        # LSTM expects:
        # (batch, sequence, features)
        #
        # Currently:
        # (B, 128, 256)
        #
        # Change to:
        # (B, 256, 128)
        x = x.transpose(1, 2)

        # ---------------------------------------------------------
        # LSTM
        # ---------------------------------------------------------
        x, (h_n, c_n) = self.lstm(x)

        # x:
        # (B, 256, hidden_size)
        #
        # Take the representation from the final sequence step.
        x = x[:, -1, :]

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------
        return self.regressor(x)
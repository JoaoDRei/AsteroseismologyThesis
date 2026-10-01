import torch
import torch.nn as nn


class CNNLSTMBidirectional(nn.Module):
    """
    CNN + bidirectional LSTM model for 1D stellar
    light curves / power spectra.

    The CNN extracts local features and downsamples
    the input sequence. The bidirectional LSTM then
    models dependencies using information from both
    directions along the sequence.
    """

    def __init__(
        self,
        output_dim=1,
        hidden_size=128,
        num_layers=2
    ):
        super().__init__()

        # ---------------------------------------------------------
        # 1. CNN feature extractor
        # ---------------------------------------------------------

        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=15,
                stride=2,
                padding=7,
                bias=False
            ),

            nn.BatchNorm1d(32),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                32,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            nn.BatchNorm1d(64),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            nn.BatchNorm1d(128),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),

            nn.MaxPool1d(
                kernel_size=2
            ),

            nn.Conv1d(
                128,
                128,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False
            ),

            nn.BatchNorm1d(128),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),
        )

        # ---------------------------------------------------------
        # 2. Reduce sequence length
        # ---------------------------------------------------------

        self.sequence_pool = nn.AdaptiveAvgPool1d(
            256
        )

        # ---------------------------------------------------------
        # 3. Bidirectional LSTM
        # ---------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                0.1
                if num_layers > 1
                else 0.0
            ),
            bidirectional=True
        )

        # Bidirectional LSTM produces
        #
        # hidden_size forward
        # +
        # hidden_size backward

        lstm_output_size = hidden_size * 2

        # ---------------------------------------------------------
        # 4. Prediction head
        # ---------------------------------------------------------

        self.regressor = nn.Sequential(

            nn.Linear(
                lstm_output_size,
                128
            ),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),

            nn.Dropout(0.1),

            nn.Linear(
                128,
                output_dim
            )
        )

    def forward(self, x):

        # ---------------------------------------------------------
        # CNN
        # ---------------------------------------------------------

        x = self.cnn(x)

        # (B, 128, L')
        x = self.sequence_pool(x)

        # (B, 128, 256)
        x = x.transpose(1, 2)

        # (B, 256, 128)
        # ---------------------------------------------------------
        # Bidirectional LSTM
        # ---------------------------------------------------------

        x, (h_n, c_n) = self.lstm(x)

        # h_n:
        #
        # (num_layers * 2, B, hidden_size)
        #
        # For num_layers=2:
        #
        # h_n[0] = layer 1 forward
        # h_n[1] = layer 1 backward
        # h_n[2] = layer 2 forward
        # h_n[3] = layer 2 backward

        h_forward = h_n[-2]
        h_backward = h_n[-1]

        # (B, hidden_size * 2)

        x = torch.cat(
            [h_forward, h_backward],
            dim=1
        )

        # ---------------------------------------------------------
        # Prediction
        # ---------------------------------------------------------

        return self.regressor(x)
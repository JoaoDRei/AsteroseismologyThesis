import torch
import torch.nn as nn


class CNNBiLSTMEncoder(nn.Module):
    """
    CNN + bidirectional LSTM encoder.

    The CNN extracts local features and progressively downsamples
    the input sequence. The bidirectional LSTM then integrates
    information from both directions.

    The final hidden states of the last LSTM layer are concatenated
    to form the embedding.

    With hidden_size=128:
        embedding_dim = 128 * 2 = 256
    """

    def __init__(
        self,
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
            nn.LeakyReLU(0.01, inplace=True),

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
            nn.LeakyReLU(0.01, inplace=True),

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
            nn.LeakyReLU(0.01, inplace=True),

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
            nn.LeakyReLU(0.01, inplace=True),
        )

        # ---------------------------------------------------------
        # 2. Control sequence length
        # ---------------------------------------------------------

        self.sequence_pool = nn.AdaptiveAvgPool1d(256)

        # ---------------------------------------------------------
        # 3. Bidirectional LSTM
        # ---------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                0.1 if num_layers > 1 else 0.0
            ),
            bidirectional=True
        )

        # Forward + backward
        self.embedding_dim = hidden_size * 2

    def encode(self, x):
        """
        Convert an input light curve into a fixed-dimensional
        embedding.

        Input:
            (B, 1, L)

        Output:
            (B, embedding_dim)
        """

        # CNN
        x = self.cnn(x)

        # (B, 128, L') -> (B, 128, 256)
        x = self.sequence_pool(x)

        # LSTM expects (B, sequence, features)
        x = x.transpose(1, 2)

        # Bidirectional LSTM
        _, (h_n, _) = self.lstm(x)

        # Last LSTM layer:
        #
        # h_n[-2] = forward
        # h_n[-1] = backward

        h_forward = h_n[-2]
        h_backward = h_n[-1]

        embedding = torch.cat(
            [h_forward, h_backward],
            dim=1
        )

        return embedding

    def forward(self, x):
        return self.encode(x)


class CNNBiLSTM(nn.Module):
    """
    CNN + bidirectional LSTM model for supervised prediction.

    This is the supervised version of CNNBiLSTMEncoder.
    """

    def __init__(
        self,
        output_dim=1,
        hidden_size=128,
        num_layers=2
    ):
        super().__init__()

        self.encoder = CNNBiLSTMEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(

            nn.Linear(
                self.encoder.embedding_dim,
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

        embedding = self.encoder(x)

        return self.regressor(embedding)
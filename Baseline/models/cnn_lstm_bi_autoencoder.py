import torch
import torch.nn as nn

from .cnn_lstm_bi import CNNBiLSTMEncoder


class CNNBiLSTMAutoencoder(nn.Module):
    """
    CNN + bidirectional LSTM autoencoder.

    Encoder:
        light curve -> embedding

    Decoder:
        embedding -> reconstructed coarse light curve
    """

    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        reconstruction_length=4096,
        latent_dim=None,
        preserve_short_sequences=False,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------

        self.encoder = CNNBiLSTMEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            preserve_short_sequences=preserve_short_sequences,
        )

        encoder_dim = self.encoder.embedding_dim
        self.latent_dim = (
            encoder_dim if latent_dim is None else int(latent_dim)
        )
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        self.bottleneck = (
            nn.Identity()
            if self.latent_dim == encoder_dim
            else nn.Linear(encoder_dim, self.latent_dim)
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------

        self.decoder = nn.Sequential(

            nn.Linear(
                self.latent_dim,
                512
            ),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),

            nn.Linear(
                512,
                1024
            ),

            nn.LeakyReLU(
                0.01,
                inplace=True
            ),

            nn.Linear(
                1024,
                reconstruction_length
            )
        )

    def forward(self, x):

        embedding = self.bottleneck(self.encoder(x))

        reconstruction = self.decoder(
            embedding
        )

        return reconstruction, embedding
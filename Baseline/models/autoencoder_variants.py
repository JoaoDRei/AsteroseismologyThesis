import torch.nn as nn
import torch.nn.functional as F

from .cnn_lstm_bi_autoencoder import CNNBiLSTMAutoencoder


class ConvolutionalAutoencoder(nn.Module):
    """Convolutional encoder with a compact dense decoder."""

    def __init__(self, latent_dim, reconstruction_length):
        super().__init__()
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")

        self.encoder_features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01, inplace=True),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )
        self.encoder_projection = nn.Linear(128 * 16, self.latent_dim)
        self.decoder = _make_decoder(self.latent_dim, reconstruction_length)

    def forward(self, x):
        embedding = self.encoder_projection(self.encoder_features(x))
        return self.decoder(embedding), embedding


class FullyConnectedAutoencoder(nn.Module):
    """Fully connected autoencoder operating on the pooled reconstruction grid."""

    def __init__(self, latent_dim, reconstruction_length, hidden_size=1024):
        super().__init__()
        self.latent_dim = int(latent_dim)
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if reconstruction_length <= 0 or hidden_size <= 0:
            raise ValueError("reconstruction_length and hidden_size must be positive.")

        self.reconstruction_length = int(reconstruction_length)
        self.encoder = nn.Sequential(
            nn.Linear(self.reconstruction_length, hidden_size),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden_size, self.latent_dim),
        )
        self.decoder = _make_decoder(self.latent_dim, self.reconstruction_length)

    def forward(self, x):
        pooled = F.adaptive_avg_pool1d(x, self.reconstruction_length).flatten(1)
        embedding = self.encoder(pooled)
        return self.decoder(embedding), embedding


def _make_decoder(latent_dim, reconstruction_length):
    if reconstruction_length <= 0:
        raise ValueError("reconstruction_length must be positive.")
    return nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Linear(1024, reconstruction_length),
    )


def build_autoencoder(
    architecture,
    hidden_size,
    num_layers,
    reconstruction_length,
    latent_dim,
    preserve_short_sequences=False,
):
    if architecture == "cnn_bilstm":
        return CNNBiLSTMAutoencoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            reconstruction_length=reconstruction_length,
            latent_dim=latent_dim,
            preserve_short_sequences=preserve_short_sequences,
        )
    if architecture == "convolutional":
        return ConvolutionalAutoencoder(
            latent_dim=latent_dim,
            reconstruction_length=reconstruction_length,
        )
    if architecture == "fully_connected":
        return FullyConnectedAutoencoder(
            latent_dim=latent_dim,
            reconstruction_length=reconstruction_length,
            hidden_size=1024,
        )
    raise ValueError(f"Unsupported autoencoder architecture: {architecture}")

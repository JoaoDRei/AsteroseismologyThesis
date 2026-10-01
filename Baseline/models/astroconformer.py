import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rotary Positional Embedding
# ============================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embedding (RoPE).

    Generates cos/sin embeddings for the sequence positions.
    """

    def __init__(self, dim, base=10000):
        super().__init__()

        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        """
        Returns:
            cos: (1, 1, seq_len, dim)
            sin: (1, 1, seq_len, dim)
        """

        positions = torch.arange(
            seq_len,
            device=device,
            dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum(
            "i,j->ij",
            positions,
            self.inv_freq
        )

        # Duplicate frequencies so that the last dimension
        # has size `dim`.
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos, sin


def rotate_half(x):
    """
    Rotate the last dimension by 90 degrees.
    """

    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q, k, cos, sin):
    """
    Apply rotary positional embedding to q and k.
    """

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


# ============================================================
# Convolutional Patch / Token Embedding
# ============================================================

class ConvEmbedding(nn.Module):
    """
    Converts the long 1D input sequence into a shorter sequence
    of learned feature vectors.

    Example:

        (B, 1, 65000)
              ↓
        Conv1D(stride=64)
              ↓
        (B, encoder_dim, ~1015)
              ↓
        (B, ~1015, encoder_dim)
    """

    def __init__(
        self,
        in_channels=1,
        encoder_dim=128,
        stride=64
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=encoder_dim,
            kernel_size=stride,
            stride=stride,
            padding=0,
            bias=True
        )

        self.bn = nn.BatchNorm1d(encoder_dim)

        self.activation = nn.SiLU()

    def forward(self, x):
        """
        Input:
            x: (B, 1, L)

        Output:
            x: (B, L_tokens, encoder_dim)
        """

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        return x


# ============================================================
# Feed Forward Network
# ============================================================

class FeedForward(nn.Module):
    """
    Conformer feed-forward module.

    Uses the usual 4x expansion.
    """

    def __init__(
        self,
        encoder_dim,
        dropout=0.1,
        expansion_factor=4
    ):
        super().__init__()

        hidden_dim = encoder_dim * expansion_factor

        self.net = nn.Sequential(
            nn.LayerNorm(encoder_dim),

            nn.Linear(
                encoder_dim,
                hidden_dim
            ),

            nn.SiLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                encoder_dim
            ),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Multi-Head Self-Attention with RoPE
# ============================================================

class RotarySelfAttention(nn.Module):
    """
    Multi-head self-attention with rotary positional embeddings.
    """

    def __init__(
        self,
        encoder_dim=128,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        assert encoder_dim % num_heads == 0, (
            "encoder_dim must be divisible by num_heads"
        )

        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dim = encoder_dim // num_heads

        # Rotate half of each attention head.
        self.rotary_dim = self.head_dim // 2

        self.query = nn.Linear(
            encoder_dim,
            encoder_dim
        )

        self.key = nn.Linear(
            encoder_dim,
            encoder_dim
        )

        self.value = nn.Linear(
            encoder_dim,
            encoder_dim
        )

        self.output = nn.Linear(
            encoder_dim,
            encoder_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.rotary_embedding = RotaryEmbedding(
            self.rotary_dim
        )

    def forward(self, x):
        """
        x:
            (B, L, encoder_dim)

        returns:
            (B, L, encoder_dim)
        """

        B, L, C = x.shape

        # ----------------------------------------------------
        # Linear projections
        # ----------------------------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # ----------------------------------------------------
        # Split into attention heads
        # ----------------------------------------------------

        q = q.view(
            B, L,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, L,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, L,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Shape:
        #
        # (B, heads, L, head_dim)

        # ----------------------------------------------------
        # Rotary positional embedding
        # ----------------------------------------------------

        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]

        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]

        cos, sin = self.rotary_embedding(
            L,
            x.device
        )

        q_rot, k_rot = apply_rotary(
            q_rot,
            k_rot,
            cos,
            sin
        )

        q = torch.cat(
            [q_rot, q_pass],
            dim=-1
        )

        k = torch.cat(
            [k_rot, k_pass],
            dim=-1
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        x = torch.matmul(
            attention,
            v
        )

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        x = x.transpose(1, 2).contiguous()

        x = x.view(
            B,
            L,
            C
        )

        x = self.output(x)

        return x


# ============================================================
# Conformer Convolution Module
# ============================================================

class ConformerConv(nn.Module):
    """
    Local convolutional module inside the Conformer block.

    This is deliberately close to the convolutional component
    of the original implementation, while keeping it simple.
    """

    def __init__(
        self,
        encoder_dim=128,
        kernel_size=31,
        dropout=0.1
    ):
        super().__init__()

        assert kernel_size % 2 == 1, (
            "kernel_size should be odd for same padding"
        )

        self.norm = nn.LayerNorm(encoder_dim)

        # Depthwise convolution.
        self.depthwise = nn.Conv1d(
            encoder_dim,
            encoder_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=encoder_dim,
            bias=False
        )

        self.bn = nn.BatchNorm1d(
            encoder_dim
        )

        self.activation = nn.SiLU()

        self.pointwise = nn.Conv1d(
            encoder_dim,
            encoder_dim,
            kernel_size=1
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input/output:
            (B, L, C)
        """

        x = self.norm(x)

        # (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)

        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pointwise(x)

        x = self.dropout(x)

        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)

        return x


# ============================================================
# Conformer Block
# ============================================================

class ConformerBlock(nn.Module):
    """
    One Conformer block:

        FFN
         ↓
        MHSA
         ↓
        Conv
         ↓
        FFN

    with residual connections around each component.
    """

    def __init__(
        self,
        encoder_dim=128,
        num_heads=8,
        dropout=0.1,
        conv_kernel_size=31
    ):
        super().__init__()

        self.ffn1 = FeedForward(
            encoder_dim=encoder_dim,
            dropout=dropout
        )

        self.attention_norm = nn.LayerNorm(
            encoder_dim
        )

        self.attention = RotarySelfAttention(
            encoder_dim=encoder_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.conv = ConformerConv(
            encoder_dim=encoder_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout
        )

        self.ffn2 = FeedForward(
            encoder_dim=encoder_dim,
            dropout=dropout
        )

        self.final_norm = nn.LayerNorm(
            encoder_dim
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Feed Forward 1
        # ----------------------------------------------------

        x = x + self.ffn1(x)

        # ----------------------------------------------------
        # Multi-Head Self Attention
        # ----------------------------------------------------

        residual = x

        x = self.attention_norm(x)

        x = self.attention(x)

        x = self.attention_dropout(x)

        x = residual + x

        # ----------------------------------------------------
        # Convolution
        # ----------------------------------------------------

        x = x + self.conv(x)

        # ----------------------------------------------------
        # Feed Forward 2
        # ----------------------------------------------------

        x = x + self.ffn2(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(x)

        return x


# ============================================================
# AstroConformer
# ============================================================

class AstroConformer(nn.Module):
    """
    AstroConformer-inspired model for stellar light curves
    and power spectra.

    Input:
        (B, 1, L)

    Output:
        (B, output_dim)

    Default configuration is intended as a first experimental
    baseline rather than an exact reproduction of the paper.
    """

    def __init__(
        self,
        output_dim=1,
        in_channels=1,
        encoder_dim=128,
        num_heads=8,
        num_layers=4,
        stride=64,
        conv_kernel_size=31,
        dropout=0.1
    ):
        super().__init__()

        self.encoder_dim = encoder_dim

        # ----------------------------------------------------
        # Input embedding
        # ----------------------------------------------------

        self.extractor = ConvEmbedding(
            in_channels=in_channels,
            encoder_dim=encoder_dim,
            stride=stride
        )

        # ----------------------------------------------------
        # Conformer encoder
        # ----------------------------------------------------

        self.encoder = nn.ModuleList([
            ConformerBlock(
                encoder_dim=encoder_dim,
                num_heads=num_heads,
                dropout=dropout,
                conv_kernel_size=conv_kernel_size
            )
            for _ in range(num_layers)
        ])

        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.pred_layer = nn.Sequential(
            nn.Linear(
                encoder_dim,
                encoder_dim
            ),

            nn.SiLU(),

            nn.Dropout(0.3),

            nn.Linear(
                encoder_dim,
                output_dim
            )
        )

    def forward(self, x):
        """
        Input:
            x: (B, 1, L)

        Output:
            (B, output_dim)
        """

        # ----------------------------------------------------
        # Convolutional tokenization
        # ----------------------------------------------------

        x = self.extractor(x)

        # x:
        # (B, L_tokens, encoder_dim)

        # ----------------------------------------------------
        # Conformer blocks
        # ----------------------------------------------------

        for block in self.encoder:
            x = block(x)

        # ----------------------------------------------------
        # Global sequence pooling
        # ----------------------------------------------------

        x = x.mean(dim=1)

        # x:
        # (B, encoder_dim)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        x = self.pred_layer(x)

        return x
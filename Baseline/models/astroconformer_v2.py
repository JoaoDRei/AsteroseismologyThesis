import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Rotary Positional Embedding
# ============================================================

class RotaryEmbedding(nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()

        self.dim = dim

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        self.register_buffer(
            "inv_freq",
            inv_freq
        )

    def forward(self, seq_len, device):

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

        emb = torch.cat(
            [freqs, freqs],
            dim=-1
        )

        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos, sin


def rotate_half(x):

    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]

    return torch.cat(
        [-x2, x1],
        dim=-1
    )


def apply_rotary(q, k, cos, sin):

    q = (
        q * cos
        + rotate_half(q) * sin
    )

    k = (
        k * cos
        + rotate_half(k) * sin
    )

    return q, k


# ============================================================
# Dense Patch Embedding
# ============================================================

class DensePatchEmbedding(nn.Module):
    """
    Splits the input sequence into non-overlapping patches
    and projects each patch with a Linear layer.

    Example:

        (B, 1, 65000)
              ↓
        remove/pad to multiple of patch_size
              ↓
        (B, ~1015, 64)
              ↓
        Linear(64, encoder_dim)
              ↓
        (B, ~1015, encoder_dim)
    """

    def __init__(
        self,
        patch_size=64,
        encoder_dim=128
    ):
        super().__init__()

        self.patch_size = patch_size

        self.projection = nn.Linear(
            patch_size,
            encoder_dim
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Input
        # ----------------------------------------------------

        # Expected:
        # (B, 1, L)

        B, C, L = x.shape

        if C != 1:
            raise ValueError(
                "DensePatchEmbedding expects one input channel."
            )

        x = x.squeeze(1)

        # ----------------------------------------------------
        # Make length divisible by patch size
        # ----------------------------------------------------

        remainder = L % self.patch_size

        if remainder != 0:

            padding = self.patch_size - remainder

            x = F.pad(
                x,
                (0, padding)
            )

        # ----------------------------------------------------
        # Split into patches
        # ----------------------------------------------------

        # (B, L)
        #
        # →
        #
        # (B, N_patches, patch_size)

        x = x.unfold(
            dimension=1,
            size=self.patch_size,
            step=self.patch_size
        )

        # ----------------------------------------------------
        # Linear projection
        # ----------------------------------------------------

        x = self.projection(x)

        # Result:
        #
        # (B, N_patches, encoder_dim)

        return x


# ============================================================
# Feed Forward
# ============================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        encoder_dim,
        dropout=0.1,
        expansion_factor=4
    ):
        super().__init__()

        hidden_dim = (
            encoder_dim * expansion_factor
        )

        self.net = nn.Sequential(

            nn.LayerNorm(
                encoder_dim
            ),

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
# Rotary Self Attention
# ============================================================

class RotarySelfAttention(nn.Module):

    def __init__(
        self,
        encoder_dim=128,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        assert encoder_dim % num_heads == 0

        self.encoder_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dim = (
            encoder_dim // num_heads
        )

        self.rotary_dim = (
            self.head_dim // 2
        )

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

        self.dropout = nn.Dropout(
            dropout
        )

        self.rotary_embedding = (
            RotaryEmbedding(
                self.rotary_dim
            )
        )

    def forward(self, x):

        B, L, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # ----------------------------------------------------
        # Split heads
        # ----------------------------------------------------

        q = q.view(
            B,
            L,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            L,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            L,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Rotary embedding
        # ----------------------------------------------------

        q_rot = q[
            ..., :self.rotary_dim
        ]

        q_pass = q[
            ..., self.rotary_dim:
        ]

        k_rot = k[
            ..., :self.rotary_dim
        ]

        k_pass = k[
            ..., self.rotary_dim:
        ]

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

        scores = scores / math.sqrt(
            self.head_dim
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(
            attention
        )

        x = torch.matmul(
            attention,
            v
        )

        # ----------------------------------------------------
        # Merge heads
        # ----------------------------------------------------

        x = x.transpose(
            1,
            2
        ).contiguous()

        x = x.view(
            B,
            L,
            C
        )

        return self.output(x)


# ============================================================
# Conformer Convolution
# ============================================================

class ConformerConv(nn.Module):

    def __init__(
        self,
        encoder_dim=128,
        kernel_size=31,
        dropout=0.1
    ):
        super().__init__()

        assert kernel_size % 2 == 1

        self.norm = nn.LayerNorm(
            encoder_dim
        )

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

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        x = self.norm(x)

        # (B, L, C)
        # →
        # (B, C, L)

        x = x.transpose(
            1,
            2
        )

        x = self.depthwise(x)

        x = self.bn(x)

        x = self.activation(x)

        x = self.pointwise(x)

        x = self.dropout(x)

        # (B, C, L)
        # →
        # (B, L, C)

        x = x.transpose(
            1,
            2
        )

        return x


# ============================================================
# Conformer Block
# ============================================================

class ConformerBlock(nn.Module):

    def __init__(
        self,
        encoder_dim=128,
        num_heads=8,
        dropout=0.1,
        conv_kernel_size=31
    ):
        super().__init__()

        self.ffn1 = FeedForward(
            encoder_dim,
            dropout
        )

        self.attention_norm = nn.LayerNorm(
            encoder_dim
        )

        self.attention = RotarySelfAttention(
            encoder_dim,
            num_heads,
            dropout
        )

        self.attention_dropout = nn.Dropout(
            dropout
        )

        self.conv = ConformerConv(
            encoder_dim,
            conv_kernel_size,
            dropout
        )

        self.ffn2 = FeedForward(
            encoder_dim,
            dropout
        )

        self.final_norm = nn.LayerNorm(
            encoder_dim
        )

    def forward(self, x):

        # FFN
        x = x + self.ffn1(x)

        # Attention
        residual = x

        x = self.attention_norm(x)

        x = self.attention(x)

        x = self.attention_dropout(x)

        x = residual + x

        # Convolution
        x = x + self.conv(x)

        # FFN
        x = x + self.ffn2(x)

        # Final normalization
        x = self.final_norm(x)

        return x


# ============================================================
# AstroConformer V2
# ============================================================

class AstroConformerV2(nn.Module):
    """
    AstroConformer-inspired model using dense patch embedding.

    Unlike V1, there is NO convolutional input embedding.

    Raw sequence
        ↓
    non-overlapping patches
        ↓
    Linear projection
        ↓
    Conformer
        ↓
    mean pooling
        ↓
    prediction
    """

    def __init__(
        self,
        output_dim=1,
        patch_size=64,
        encoder_dim=128,
        num_heads=8,
        num_layers=4,
        conv_kernel_size=31,
        dropout=0.1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.encoder_dim = encoder_dim

        # ----------------------------------------------------
        # Dense patch embedding
        # ----------------------------------------------------

        self.extractor = DensePatchEmbedding(
            patch_size=patch_size,
            encoder_dim=encoder_dim
        )

        # ----------------------------------------------------
        # Conformer
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

        # ----------------------------------------------------
        # Dense patch embedding
        # ----------------------------------------------------

        x = self.extractor(x)

        # (B, N_patches, encoder_dim)

        # ----------------------------------------------------
        # Conformer
        # ----------------------------------------------------

        for block in self.encoder:
            x = block(x)

        # ----------------------------------------------------
        # Global mean pooling
        # ----------------------------------------------------

        x = x.mean(dim=1)

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        x = self.pred_layer(x)

        return x
"""Task-adapted neural baselines for the QORT-Net comparison.

All models accept a causal window shaped [batch, time, features] and return one
normalized RUL estimate. Forecasting architectures are adapted only at the
scalar prediction head; their defining representation mechanism is retained.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 48, dropout: float = 0.15, **_):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.input_projection(x))
        return self.head(encoded[:, -1]).squeeze(-1)


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.depthwise = CausalConv1d(
            channels, channels, kernel_size=3, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value, gate = self.pointwise(self.depthwise(x)).chunk(2, dim=1)
        x = self.dropout(self.output(value * torch.sigmoid(gate)))
        return F.gelu(self.norm(x + residual))


class ResidualShrinkageBlock1d(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        reduced = max(channels // 4, 4)
        self.threshold = nn.Sequential(
            nn.Linear(channels, reduced), nn.ReLU(),
            nn.Linear(reduced, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.features(x)
        magnitude = residual.abs()
        threshold = magnitude.mean(dim=2) * self.threshold(magnitude.mean(dim=2))
        shrunk = torch.sign(residual) * F.relu(magnitude - threshold.unsqueeze(-1))
        return F.relu(x + shrunk)


class DRSNTCNRegressor(nn.Module):
    """PCA is fitted outside the network on training bearings only."""

    def __init__(self, input_dim: int, hidden_dim: int = 48, dropout: float = 0.15, **_):
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        self.tcn = nn.Sequential(
            TCNBlock(hidden_dim, 1, dropout), TCNBlock(hidden_dim, 2, dropout)
        )
        self.shrinkage = nn.Sequential(
            *[ResidualShrinkageBlock1d(hidden_dim, dropout) for _ in range(4)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.tcn(self.input_projection(x.transpose(1, 2)))
        return self.head(self.shrinkage(encoded)).squeeze(-1)


class PatchTSTRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 48, patch_size: int = 4,
        window: int = 20, dropout: float = 0.15, **_,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = math.ceil(window / patch_size)
        self.patch_embedding = nn.Linear(patch_size, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        padding = (-length) % self.patch_size
        if padding:
            x = torch.cat([x[:, :1].expand(-1, padding, -1), x], dim=1)
        patches = x.unfold(1, self.patch_size, self.patch_size).permute(0, 2, 1, 3)
        tokens = self.patch_embedding(patches).reshape(batch * channels, self.num_patches, -1)
        tokens = self.encoder(tokens + self.position)
        representation = tokens.mean(dim=1).reshape(batch, channels, -1).mean(dim=1)
        return self.head(representation).squeeze(-1)


def orthonormal_polynomial_basis(length: int, order: int) -> torch.Tensor:
    t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    vander = np.stack([t**degree for degree in range(order + 1)], axis=1)
    q, r = np.linalg.qr(vander)
    q = q * np.sign(np.diag(r))
    return torch.from_numpy(np.ascontiguousarray(q.T)).float()


class FiLMRULRegressor(nn.Module):
    """Scalar-RUL adaptation of Legendre projection plus Fourier denoising."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 48, window: int = 20,
        dropout: float = 0.15, **_,
    ):
        super().__init__()
        order = min(11, window - 1)
        self.register_buffer("basis", orthonormal_polynomial_basis(window, order))
        frequency_bins = (order + 1) // 2 + 1
        rank = max(4, hidden_dim // 8)
        self.left = nn.Parameter(torch.randn(frequency_bins, rank) * 0.05)
        self.right = nn.Parameter(torch.randn(rank, input_dim) * 0.05)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = torch.einsum("blc,kl->bkc", x, self.basis)
        # cuFFT has limited half-precision support for non-power-of-two lengths.
        spectrum = torch.fft.rfft(memory.float(), dim=1)
        shaping = torch.sigmoid(self.left @ self.right).unsqueeze(0)
        filtered = torch.fft.irfft(spectrum * shaping, n=memory.shape[1], dim=1)
        tokens = self.embedding(filtered)
        representation = torch.cat([tokens.mean(dim=1), tokens[:, -1]], dim=-1)
        return self.head(representation).squeeze(-1)


class FilterNetRULRegressor(nn.Module):
    """Plain and contextual shaping filters followed by a scalar RUL head."""

    def __init__(
        self, input_dim: int, hidden_dim: int = 48, window: int = 20,
        dropout: float = 0.15, **_,
    ):
        super().__init__()
        self.window = window
        frequency_bins = window // 2 + 1
        self.plain_filter = nn.Parameter(torch.zeros(frequency_bins, input_dim, 2))
        nn.init.normal_(self.plain_filter, std=0.04)
        contextual_hidden = max(8, frequency_bins)
        self.context_filter = nn.Sequential(
            nn.Linear(frequency_bins, contextual_hidden), nn.GELU(),
            nn.Linear(contextual_hidden, frequency_bins), nn.Sigmoid(),
        )
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x.float(), dim=1)
        plain = torch.view_as_complex(self.plain_filter.contiguous()).unsqueeze(0)
        energy = torch.log1p(spectrum.abs().pow(2).mean(dim=2))
        contextual = self.context_filter(energy).unsqueeze(-1)
        filtered = torch.fft.irfft(spectrum * (1.0 + plain) * contextual, n=self.window, dim=1)
        tokens = self.embedding(filtered + x)
        representation = torch.cat(
            [tokens.mean(dim=1), tokens.std(dim=1, unbiased=False), tokens[:, -1]], dim=-1
        )
        return self.head(representation).squeeze(-1)


class MixLinearRULRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 48, patch_size: int = 4,
        window: int = 20, dropout: float = 0.15, **_,
    ):
        super().__init__()
        if window % patch_size:
            raise ValueError("window must be divisible by patch_size")
        self.patch_size = patch_size
        self.segments = window // patch_size
        positions = torch.arange(window, dtype=torch.float32)
        frequencies = torch.arange(window, dtype=torch.float32).unsqueeze(1)
        basis = torch.cos(math.pi / window * (positions + 0.5) * frequencies)
        basis[0] *= math.sqrt(1.0 / window)
        basis[1:] *= math.sqrt(2.0 / window)
        self.register_buffer("dct_basis", basis[:4])
        segment_axis = torch.arange(patch_size, dtype=torch.float32)
        self.register_buffer("segment_axis", segment_axis - segment_axis.mean())
        feature_dim = input_dim * (self.segments * 2 + 4)
        bottleneck = max(hidden_dim // 2, 16)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, bottleneck), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        segments = x.reshape(batch, self.segments, self.patch_size, channels)
        intercept = segments.mean(dim=2)
        slope = torch.einsum("bspc,p->bsc", segments, self.segment_axis)
        slope = slope / self.segment_axis.square().sum()
        spectrum = torch.einsum("kt,btc->bkc", self.dct_basis, x)
        features = torch.cat(
            [intercept.flatten(1), slope.flatten(1), spectrum.flatten(1)], dim=1
        )
        return self.head(features).squeeze(-1)


class LocalVariatePatchBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        self.pointwise = nn.Conv2d(d_model, d_model * 2, 1)
        self.output = nn.Conv2d(d_model, d_model, 1)
        self.norm = nn.GroupNorm(1, d_model)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        residual = field
        value, gate = self.pointwise(self.depthwise(field)).chunk(2, dim=1)
        field = self.dropout(self.output(value * torch.sigmoid(gate)))
        return F.gelu(self.norm(field + residual))


class VPNetRULRegressor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 48, patch_size: int = 4,
        dropout: float = 0.15, **_,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_size, hidden_dim)
        self.blocks = nn.Sequential(
            LocalVariatePatchBlock(hidden_dim, dropout),
            LocalVariatePatchBlock(hidden_dim, dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, length, _ = x.shape
        padding = (-length) % self.patch_size
        if padding:
            x = torch.cat([x[:, :1].expand(-1, padding, -1), x], dim=1)
        patches = x.unfold(1, self.patch_size, self.patch_size)
        field = self.patch_embedding(patches).permute(0, 3, 2, 1).contiguous()
        return self.head(self.blocks(field).mean(dim=(2, 3))).squeeze(-1)


MODEL_FACTORIES = {
    "gru": GRURegressor,
    "pca_drsn_tcn": DRSNTCNRegressor,
    "patchtst": PatchTSTRegressor,
    "film": FiLMRULRegressor,
    "filternet": FilterNetRULRegressor,
    "mixlinear": MixLinearRULRegressor,
    "vpnet": VPNetRULRegressor,
}


def make_model(name: str, **kwargs) -> nn.Module:
    try:
        return MODEL_FACTORIES[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"Unknown baseline: {name}") from exc


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

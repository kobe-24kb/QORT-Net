"""QORT-Net: Quantile-guided Orthogonal Reliability Token Network."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def discrete_legendre_basis(length: int, order: int) -> torch.Tensor:
    t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    vander = np.stack([t**degree for degree in range(order + 1)], axis=1)
    q, r = np.linalg.qr(vander)
    q = q * np.sign(np.diag(r))
    return torch.from_numpy(np.ascontiguousarray(q.T)).float()


class OrthogonalSpectralTokenizer(nn.Module):
    """A causal window becomes trend, residual, endpoint and spectral tokens."""

    def __init__(self, window: int, order: int, spectral_bands: int, mode: str = "full"):
        super().__init__()
        if mode not in {"full", "no_spectral", "no_trend", "raw"}:
            raise ValueError(f"Unknown token mode: {mode}")
        self.window = window
        self.order = order
        self.mode = mode
        # Ablation modes: "no_spectral" drops the learned spectral tokens,
        # "no_trend" drops the orthogonal trend coefficients and residual RMS,
        # "raw" replaces all summaries by the chronological snapshots.
        self.spectral_bands = 0 if mode in {"no_spectral", "raw"} else spectral_bands
        self.register_buffer("basis", discrete_legendre_basis(window, order))
        self.filters = nn.Parameter(torch.randn(self.spectral_bands, window // 2 + 1) * 0.15)

    @property
    def token_count(self) -> int:
        if self.mode == "raw":
            return self.window
        if self.mode == "no_trend":
            return 2 + self.spectral_bands
        return self.order + 4 + self.spectral_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw":
            return x
        if self.mode == "no_trend":
            power = torch.log1p(torch.fft.rfft(x, dim=1).abs().pow(2))
            spectral = torch.einsum("bfc,mf->bmc", power, torch.softmax(self.filters, dim=-1))
            return torch.cat(
                [(x[:, -1] - x[:, 0]).unsqueeze(1), x[:, -1].unsqueeze(1), spectral], dim=1
            )
        coefficients = torch.einsum("blc,kl->bkc", x, self.basis)
        reconstruction = torch.einsum("bkc,kl->blc", coefficients, self.basis)
        residual = x - reconstruction
        volatility = torch.sqrt(residual.pow(2).mean(dim=1) + 1e-8)
        endpoint_rate = x[:, -1] - x[:, 0]
        endpoint = x[:, -1]
        power = torch.log1p(torch.fft.rfft(x, dim=1).abs().pow(2))
        filters = torch.softmax(self.filters, dim=-1)
        spectral = torch.einsum("bfc,mf->bmc", power, filters)
        return torch.cat(
            [
                coefficients,
                volatility.unsqueeze(1),
                endpoint_rate.unsqueeze(1),
                endpoint.unsqueeze(1),
                spectral,
            ],
            dim=1,
        )


class TokenMixerBlock(nn.Module):
    def __init__(self, tokens: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mix = nn.Linear(tokens, tokens)
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mixed = self.token_mix(self.token_norm(z).transpose(1, 2)).transpose(1, 2)
        z = z + self.dropout(mixed)
        return z + self.dropout(self.channel_mix(self.channel_norm(z)))


class OrthogonalTokenEncoder(nn.Module):
    """One compact degradation expert with a fixed causal receptive field."""

    def __init__(
        self,
        input_dim: int,
        window: int,
        order: int,
        spectral_bands: int,
        hidden_dim: int,
        dropout: float,
        token_mode: str = "full",
    ):
        super().__init__()
        self.window = window
        self.tokenizer = OrthogonalSpectralTokenizer(window, order, spectral_bands, token_mode)
        tokens = self.tokenizer.token_count
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.token_embedding = nn.Parameter(torch.zeros(tokens, hidden_dim))
        nn.init.normal_(self.token_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [TokenMixerBlock(tokens, hidden_dim, dropout) for _ in range(2)]
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, -self.window :]
        z = self.embed(self.tokenizer(x)) + self.token_embedding
        for block in self.blocks:
            z = block(z)
        return self.pool_norm(z.mean(dim=1))


class QORTNet(nn.Module):
    """Prior--vibration fusion with explicit training-fold lifetime hypotheses.

    Only the modules used by the selected variant are instantiated, so the
    parameter count and the random-number sequence used for initialization
    correspond exactly to the evaluated architecture.
    """

    def __init__(
        self,
        input_dim: int,
        number_of_priors: int = 3,
        window: int = 40,
        short_window: int = 20,
        hidden_dim: int = 32,
        dropout: float = 0.10,
        gate_variation: float = 1.5,
        use_relative: bool = True,
        use_quantile_bank: bool = True,
        use_disagreement_gate: bool = True,
        vibration_only: bool = False,
        prior_only: bool = False,
        router_mode: str = "cv",
        router_value: float = 0.5,
        router_center: float = 0.390,
        router_scale: float = 0.012,
        gate_base: str = "coupled",
        gate_base_value: float = 0.575,
        token_mode: str = "full",
        use_optimism_term: bool = True,
    ):
        super().__init__()
        self.use_optimism_term = use_optimism_term
        if router_mode not in {"cv", "fixed"}:
            raise ValueError(f"Unknown router mode: {router_mode}")
        if gate_base not in {"coupled", "constant"}:
            raise ValueError(f"Unknown gate base: {gate_base}")
        self.use_relative = use_relative
        self.use_quantile_bank = use_quantile_bank
        self.use_disagreement_gate = use_disagreement_gate
        self.vibration_only = vibration_only
        self.prior_only = prior_only
        self.gate_variation = gate_variation
        self.router_mode = router_mode
        self.router_value = float(router_value)
        self.router_center = float(router_center)
        self.router_scale = float(router_scale)
        self.gate_base = gate_base
        self.gate_base_value = float(gate_base_value)

        if prior_only:
            # The prior-only control receives no vibration-derived input: only
            # the three causal priors, age, training-fold lifetime CV and the
            # prior dispersion enter its selector.
            self.prior_only_selector = nn.Sequential(
                nn.Linear(number_of_priors + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, number_of_priors),
            )
            nn.init.zeros_(self.prior_only_selector[-1].weight)
            nn.init.zeros_(self.prior_only_selector[-1].bias)
            return

        self.high_encoder = OrthogonalTokenEncoder(
            input_dim=input_dim,
            window=window,
            order=3,
            spectral_bands=4,
            hidden_dim=24,
            dropout=dropout,
            token_mode=token_mode,
        )
        self.low_encoder = OrthogonalTokenEncoder(
            input_dim=input_dim,
            window=short_window,
            order=1,
            spectral_bands=4,
            hidden_dim=48,
            dropout=dropout,
            token_mode=token_mode,
        )
        self.high_head = nn.Sequential(
            nn.Linear(24, 24),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )
        self.low_head = nn.Sequential(
            nn.Linear(48, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )
        if vibration_only:
            return
        self.fusion_projection = nn.Sequential(
            nn.Linear(24 + 48, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if use_quantile_bank:
            self.prior_selector = nn.Sequential(
                nn.Linear(hidden_dim + number_of_priors + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, number_of_priors),
            )
            nn.init.zeros_(self.prior_selector[-1].weight)
            nn.init.zeros_(self.prior_selector[-1].bias)
        if use_disagreement_gate:
            self.local_gate = nn.Sequential(
                nn.Linear(hidden_dim + 7, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.local_gate[-1].weight)
            nn.init.zeros_(self.local_gate[-1].bias)
            # A positive-only scale implements risk-asymmetric disagreement:
            # an optimistic prior raises the gate logit toward vibration.
            if use_optimism_term:
                self.optimism_scale_raw = nn.Parameter(torch.tensor(4.99))

    def regime_weight(self, lifetime_cv: torch.Tensor) -> torch.Tensor:
        if self.router_mode == "fixed":
            return torch.full_like(lifetime_cv, self.router_value).squeeze(-1)
        return torch.sigmoid(
            (lifetime_cv - self.router_center) / self.router_scale
        ).squeeze(-1)

    def forward(
        self,
        absolute: torch.Tensor,
        relative: torch.Tensor,
        priors: torch.Tensor,
        age: torch.Tensor,
        lifetime_cv: torch.Tensor,
    ):
        prior_dispersion = priors.std(dim=-1, keepdim=True, unbiased=False)
        regime_weight = self.regime_weight(lifetime_cv)
        if self.prior_only:
            selector_input = torch.cat(
                [priors, age, lifetime_cv, prior_dispersion], dim=-1
            )
            selector_weights = torch.softmax(
                self.prior_only_selector(selector_input), dim=-1
            )
            mixed_prior = (selector_weights * priors).sum(dim=-1)
            zeros = torch.zeros_like(mixed_prior)
            return mixed_prior, {
                "vibration_prediction": zeros,
                "mixed_prior": mixed_prior,
                "selector_weights": selector_weights,
                "gate": zeros,
                "prior_reliability": torch.ones_like(mixed_prior),
                "regime_weight": regime_weight,
                "high_prediction": zeros,
                "low_prediction": zeros,
                "optimistic_conflict": zeros,
                "gate_activation": zeros,
            }
        high_input = relative if self.use_relative else absolute
        high_representation = self.high_encoder(high_input)
        low_representation = self.low_encoder(absolute)
        high_prediction = torch.sigmoid(self.high_head(high_representation)).squeeze(-1)
        low_prediction = torch.sigmoid(self.low_head(low_representation)).squeeze(-1)
        vibration = (
            regime_weight * high_prediction
            + (1.0 - regime_weight) * low_prediction
        )
        middle = priors.shape[1] // 2
        if self.vibration_only:
            ones = torch.ones_like(vibration)
            zeros = torch.zeros_like(vibration)
            fixed_weights = torch.zeros_like(priors)
            fixed_weights[:, middle] = 1.0
            return vibration, {
                "vibration_prediction": vibration,
                "mixed_prior": priors[:, middle],
                "selector_weights": fixed_weights,
                "gate": ones,
                "prior_reliability": 1.0 - regime_weight,
                "regime_weight": regime_weight,
                "high_prediction": high_prediction,
                "low_prediction": low_prediction,
                "optimistic_conflict": zeros,
                "gate_activation": zeros,
            }
        representation = self.fusion_projection(
            torch.cat([high_representation, low_representation], dim=-1)
        )
        if self.use_quantile_bank:
            selector_input = torch.cat(
                [representation, priors, age, lifetime_cv, prior_dispersion], dim=-1
            )
            selector_weights = torch.softmax(self.prior_selector(selector_input), dim=-1)
            mixed_prior = (selector_weights * priors).sum(dim=-1)
        else:
            mixed_prior = priors[:, middle]
            selector_weights = torch.zeros_like(priors)
            selector_weights[:, middle] = 1.0
        disagreement = (vibration - mixed_prior).abs().unsqueeze(-1)
        optimistic_conflict = torch.relu(mixed_prior - vibration).unsqueeze(-1)
        if self.gate_base == "constant":
            base_gate = torch.full_like(disagreement, self.gate_base_value)
        else:
            base_gate = (0.20 + 0.75 * regime_weight.unsqueeze(-1)).clamp(0.20, 0.95)
        if self.use_disagreement_gate:
            selector_entropy = -(
                selector_weights.clamp_min(1e-8) * selector_weights.clamp_min(1e-8).log()
            ).sum(dim=-1, keepdim=True) / math.log(max(priors.shape[1], 2))
            gate_features = torch.cat(
                [
                    representation,
                    vibration.unsqueeze(-1),
                    mixed_prior.unsqueeze(-1),
                    disagreement,
                    prior_dispersion,
                    age,
                    lifetime_cv,
                    selector_entropy,
                ],
                dim=-1,
            )
            local = torch.tanh(self.local_gate(gate_features))
            logit = torch.logit(base_gate) + self.gate_variation * local
            if self.use_optimism_term:
                optimism_scale = torch.nn.functional.softplus(self.optimism_scale_raw)
                logit = logit + optimism_scale * optimistic_conflict
            gate = torch.sigmoid(logit)
        else:
            gate = base_gate
        gate = gate.squeeze(-1)
        prediction = gate * vibration + (1.0 - gate) * mixed_prior
        return prediction, {
            "vibration_prediction": vibration,
            "mixed_prior": mixed_prior,
            "selector_weights": selector_weights,
            "gate": gate,
            "prior_reliability": 1.0 - regime_weight,
            "regime_weight": regime_weight,
            "high_prediction": high_prediction,
            "low_prediction": low_prediction,
            "optimistic_conflict": optimistic_conflict.squeeze(-1),
            "gate_activation": torch.ones_like(vibration)
            if self.use_disagreement_gate
            else torch.zeros_like(vibration),
        }

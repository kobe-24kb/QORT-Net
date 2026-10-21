"""Leakage-safe data protocol shared by all QORT-Net experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
DEFAULT_FEATURES = HERE.parent / "data" / "features"
# Development subsets (used while the architecture was designed) and
# held-out operating conditions (first evaluated after the code was frozen).
DATASET_BEARINGS = {
    "xjtu": [f"Bearing2_{index}" for index in range(1, 6)],
    "phm": [f"Bearing1_{index}" for index in range(1, 8)],
    "xjtu_c1": [f"Bearing1_{index}" for index in range(1, 6)],
    "phm_c2": [f"Bearing2_{index}" for index in range(1, 8)],
    "phm60": [f"Bearing1_{index}" for index in range(1, 8)],
}
SAMPLE_INTERVAL_SECONDS = {
    "xjtu": 60.0,
    "phm": 10.0,
    "xjtu_c1": 60.0,
    "phm_c2": 10.0,
    "phm60": 60.0,
}
DATASET_GROUPS = {
    "both": ("xjtu", "phm"),
    "development": ("xjtu", "phm"),
    "external": ("xjtu_c1", "phm_c2"),
    "all": ("xjtu", "phm", "xjtu_c1", "phm_c2"),
}


def resolve_datasets(spec: str) -> tuple[str, ...]:
    if spec in DATASET_GROUPS:
        return DATASET_GROUPS[spec]
    names = tuple(item.strip() for item in spec.split(",") if item.strip())
    unknown = [name for name in names if name not in DATASET_BEARINGS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}")
    return names


@dataclass
class SequenceData:
    name: str
    x: np.ndarray
    y: np.ndarray


class RobustTrainScaler:
    """Median/IQR scaler fitted exclusively on training bearings."""

    def fit(self, arrays: list[np.ndarray]):
        values = np.concatenate(arrays, axis=0).astype(np.float64)
        self.center = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        scale = q75 - q25
        standard_deviation = np.std(values, axis=0)
        scale = np.where(scale > 1e-8, scale, standard_deviation)
        self.scale = np.where(scale > 1e-8, scale, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (values - self.center) / self.scale
        return np.clip(transformed, -8.0, 8.0).astype(np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_sequences(dataset: str, feature_dir: Path = DEFAULT_FEATURES) -> list[SequenceData]:
    sequences = []
    for bearing in DATASET_BEARINGS[dataset]:
        path = feature_dir / f"{dataset}_{bearing}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing feature cache: {path}")
        with np.load(path) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Non-finite value in {path}")
        sequences.append(SequenceData(bearing, x, y))
    return sequences


def healthy_reference_center(
    sequences: list[SequenceData], snapshots: int = 5
) -> list[SequenceData]:
    """Construct a causal healthy-relative view from commissioning snapshots."""
    if snapshots <= 0:
        return [SequenceData(s.name, s.x.copy(), s.y.copy()) for s in sequences]
    centered = []
    for sequence in sequences:
        count = min(snapshots, len(sequence.x))
        fixed_reference = np.median(sequence.x[:count], axis=0)
        values = (sequence.x - fixed_reference).astype(np.float32)
        for index in range(count - 1):
            causal_reference = np.median(sequence.x[: index + 1], axis=0)
            values[index] = sequence.x[index] - causal_reference
        centered.append(SequenceData(sequence.name, values, sequence.y.copy()))
    return centered


def transform_sequences(
    sequences: list[SequenceData], scaler: RobustTrainScaler
) -> list[SequenceData]:
    return [
        SequenceData(sequence.name, scaler.transform(sequence.x), sequence.y.copy())
        for sequence in sequences
    ]


def lifetime_cv(sequences: list[SequenceData]) -> float:
    lengths = np.asarray([len(sequence.y) for sequence in sequences], dtype=np.float64)
    return float(lengths.std(ddof=0) / max(lengths.mean(), 1.0))


def lifetime_quantiles(
    sequences: list[SequenceData], quantiles: tuple[float, ...]
) -> np.ndarray:
    lengths = np.asarray([len(sequence.y) for sequence in sequences], dtype=np.float64)
    values = np.quantile(lengths, quantiles)
    return np.maximum(values, 2.0).astype(np.float32)


def _window_view(values: np.ndarray, window: int) -> np.ndarray:
    padded = np.concatenate(
        [np.repeat(values[:1], window - 1, axis=0), values], axis=0
    )
    view = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=window, axis=0
    ).transpose(0, 2, 1)
    return np.ascontiguousarray(view, dtype=np.float32)


class QORTWindowDataset(Dataset):
    """Paired absolute/relative windows and training-only lifetime priors."""

    def __init__(
        self,
        absolute_sequences: list[SequenceData],
        relative_sequences: list[SequenceData],
        window: int,
        reference_lives: np.ndarray,
    ):
        if [s.name for s in absolute_sequences] != [s.name for s in relative_sequences]:
            raise ValueError("Absolute and relative sequence orders do not match")
        absolute_windows, relative_windows = [], []
        labels, priors, ages, prior_classes = [], [], [], []
        weights, bearing_ids = [], []
        reference_lives = np.asarray(reference_lives, dtype=np.float32)
        age_denominator = max(float(np.median(reference_lives)) - 1.0, 1.0)
        for bearing_index, (absolute, relative) in enumerate(
            zip(absolute_sequences, relative_sequences)
        ):
            n = len(absolute.y)
            elapsed = np.arange(n, dtype=np.float32)
            prior = np.stack(
                [
                    np.clip(1.0 - elapsed / max(float(life) - 1.0, 1.0), 0.0, 1.0)
                    for life in reference_lives
                ],
                axis=1,
            ).astype(np.float32)
            closest = int(np.argmin(np.abs(reference_lives - float(n))))
            absolute_windows.append(_window_view(absolute.x, window))
            relative_windows.append(_window_view(relative.x, window))
            labels.append(absolute.y.astype(np.float32))
            priors.append(prior)
            ages.append(np.clip(elapsed / age_denominator, 0.0, 2.0)[:, None])
            prior_classes.append(np.full(n, closest, dtype=np.int64))
            weights.append(np.full(n, 1.0 / n, dtype=np.float64))
            bearing_ids.append(np.full(n, bearing_index, dtype=np.int64))
        self.absolute = torch.from_numpy(np.concatenate(absolute_windows))
        self.relative = torch.from_numpy(np.concatenate(relative_windows))
        self.y = torch.from_numpy(np.concatenate(labels))
        self.priors = torch.from_numpy(np.concatenate(priors))
        self.age = torch.from_numpy(np.concatenate(ages))
        self.prior_class = torch.from_numpy(np.concatenate(prior_classes))
        self.weights = np.concatenate(weights)
        self.bearing_ids = np.concatenate(bearing_ids)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return (
            self.absolute[index],
            self.relative[index],
            self.priors[index],
            self.age[index],
            self.prior_class[index],
            self.y[index],
        )


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    error = prediction - target
    differences = np.diff(prediction)
    late_start = int(round(len(target) * 0.7))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "late_mae": float(np.mean(np.abs(error[late_start:]))),
        "monotonic_violation": float(np.mean(differences > 1e-6))
        if len(target) > 1
        else 0.0,
    }


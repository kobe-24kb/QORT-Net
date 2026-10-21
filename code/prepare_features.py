"""Extract a unified, leakage-free feature cache for XJTU-SY and PHM2012.

Each raw vibration snapshot is converted independently.  No full-life smoothing,
normalization, degradation-onset detection, or target information is used here.
Those operations must be fitted inside each training fold.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis, skew
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
HEXIN = HERE.parent
DEFAULT_XJTU = HEXIN / "XJTU" / "raw" / "XJTU-SY_Bearing_Datasets" / "37.5Hz11kN"
DEFAULT_PHM = (
    HEXIN
    / "2012PHM"
    / "数据集预处理"
    / "phm-ieee-2012-data-challenge-dataset-master"
)
DEFAULT_OUTPUT = HERE.parent / "data" / "features"


TIME_FEATURES = [
    "mean",
    "std",
    "rms",
    "abs_mean",
    "peak",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "impulse_factor",
    "shape_factor",
    "clearance_factor",
]
FREQ_FEATURES = [
    "spectral_centroid",
    "spectral_bandwidth",
    "rms_frequency",
    "spectral_entropy",
    "spectral_flatness",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
]
FEATURE_NAMES = [
    f"{axis}_{name}"
    for axis in ("horizontal", "vertical")
    for name in (*TIME_FEATURES, *FREQ_FEATURES)
] + ["cross_channel_correlation", "rms_log_ratio"]


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / max(abs(den), eps))


def channel_features(signal: np.ndarray) -> list[float]:
    signal = np.asarray(signal, dtype=np.float64)
    eps = 1e-12
    mean = float(np.mean(signal))
    std = float(np.std(signal))
    rms = float(np.sqrt(np.mean(signal * signal)))
    abs_mean = float(np.mean(np.abs(signal)))
    peak = float(np.max(np.abs(signal)))
    peak_to_peak = float(np.ptp(signal))
    skewness = float(np.nan_to_num(skew(signal, bias=False), nan=0.0))
    kurt = float(np.nan_to_num(kurtosis(signal, fisher=True, bias=False), nan=0.0))
    crest = _safe_ratio(peak, rms)
    impulse = _safe_ratio(peak, abs_mean)
    shape = _safe_ratio(rms, abs_mean)
    root_abs_mean = float(np.mean(np.sqrt(np.abs(signal) + eps)))
    clearance = _safe_ratio(peak, root_abs_mean * root_abs_mean)

    centered = signal - mean
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    if power.size > 1:
        power[0] = 0.0
    total = float(np.sum(power)) + eps
    probability = power / total
    frequency = np.linspace(0.0, 1.0, power.size, dtype=np.float64)
    centroid = float(np.sum(frequency * probability))
    bandwidth = float(np.sqrt(np.sum(((frequency - centroid) ** 2) * probability)))
    rms_frequency = float(np.sqrt(np.sum((frequency**2) * probability)))
    nonzero = probability[probability > eps]
    spectral_entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(max(power.size, 2)))
    positive_power = power[1:] if power.size > 1 else power
    spectral_flatness = float(
        np.exp(np.mean(np.log(positive_power + eps))) / (np.mean(positive_power) + eps)
    )
    low = float(np.sum(probability[frequency < 0.20]))
    mid = float(np.sum(probability[(frequency >= 0.20) & (frequency < 0.50)]))
    high = float(np.sum(probability[frequency >= 0.50]))

    return [
        mean,
        std,
        rms,
        abs_mean,
        peak,
        peak_to_peak,
        skewness,
        kurt,
        crest,
        impulse,
        shape,
        clearance,
        centroid,
        bandwidth,
        rms_frequency,
        spectral_entropy,
        spectral_flatness,
        low,
        mid,
        high,
    ]


def snapshot_features(signals: np.ndarray) -> np.ndarray:
    if signals.ndim != 2 or signals.shape[1] != 2:
        raise ValueError(f"Expected [samples, 2] vibration signal, got {signals.shape}")
    horizontal = np.asarray(signals[:, 0], dtype=np.float64)
    vertical = np.asarray(signals[:, 1], dtype=np.float64)
    horizontal_features = channel_features(horizontal)
    vertical_features = channel_features(vertical)
    if np.std(horizontal) < 1e-12 or np.std(vertical) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(horizontal, vertical)[0, 1])
    rms_h = horizontal_features[2]
    rms_v = vertical_features[2]
    rms_log_ratio = float(np.log((rms_h + 1e-12) / (rms_v + 1e-12)))
    result = np.asarray(
        [*horizontal_features, *vertical_features, correlation, rms_log_ratio],
        dtype=np.float32,
    )
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)


def read_xjtu(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)[:, :2]


def read_phm(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    delimiter = ";" if ";" in first_line else ","
    values = np.loadtxt(path, delimiter=delimiter, dtype=np.float32, ndmin=2)
    return values[:, -2:]


def numeric_order(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else 0


def process_bearing(
    dataset: str,
    bearing: str,
    folder: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, object]:
    if dataset.startswith("xjtu"):
        files = sorted(folder.glob("*.csv"), key=numeric_order)
        reader = read_xjtu
        sample_interval_seconds = 60.0
    else:
        files = sorted(folder.glob("acc_*.csv"), key=numeric_order)
        reader = read_phm
        sample_interval_seconds = 10.0
    if not files:
        raise FileNotFoundError(f"No acceleration CSV files found in {folder}")

    def convert(path: Path) -> np.ndarray:
        return snapshot_features(reader(path))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        iterator = pool.map(convert, files)
        features = list(
            tqdm(iterator, total=len(files), desc=f"{dataset}:{bearing}", leave=False)
        )
    x = np.stack(features).astype(np.float32)
    n = len(files)
    y = np.linspace(1.0, 0.0, n, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{bearing}.npz"
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        feature_names=np.asarray(FEATURE_NAMES),
        sample_interval_seconds=np.float32(sample_interval_seconds),
    )
    return {
        "dataset": dataset,
        "bearing": bearing,
        "snapshots": n,
        "features": x.shape[1],
        "output": str(output_path),
    }


def build_jobs(dataset: str, xjtu_root: Path, phm_root: Path):
    if dataset in {"both", "xjtu"}:
        for index in range(1, 6):
            bearing = f"Bearing2_{index}"
            yield "xjtu", bearing, xjtu_root / bearing
    if dataset in {"both", "phm"}:
        for index in range(1, 8):
            bearing = f"Bearing1_{index}"
            split = "Learning_set" if index <= 2 else "Full_Test_Set"
            yield "phm", bearing, phm_root / split / bearing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("both", "xjtu", "phm"), default="both")
    parser.add_argument("--xjtu-root", type=Path, default=DEFAULT_XJTU)
    parser.add_argument("--phm-root", type=Path, default=DEFAULT_PHM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    for dataset, bearing, folder in build_jobs(args.dataset, args.xjtu_root, args.phm_root):
        cache_path = args.output / f"{dataset}_{bearing}.npz"
        if cache_path.exists() and not args.overwrite:
            with np.load(cache_path) as cached:
                records.append(
                    {
                        "dataset": dataset,
                        "bearing": bearing,
                        "snapshots": int(cached["x"].shape[0]),
                        "features": int(cached["x"].shape[1]),
                        "output": str(cache_path),
                        "cached": True,
                    }
                )
            continue
        records.append(
            process_bearing(dataset, bearing, folder, args.output, args.workers)
        )

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

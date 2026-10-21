"""Deterministic and linear sanity-check baselines under cyclic LOBO."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from protocol import (
    DEFAULT_FEATURES,
    RobustTrainScaler,
    lifetime_quantiles,
    load_sequences,
    resolve_datasets,
    metrics,
    transform_sequences,
)


HERE = Path(__file__).resolve().parent


def median_clock(length: int, median_life: float) -> np.ndarray:
    elapsed = np.arange(length, dtype=np.float32)
    return np.clip(1.0 - elapsed / max(median_life - 1.0, 1.0), 0.0, 1.0)


def run(args) -> pd.DataFrame:
    datasets = resolve_datasets(args.dataset)
    records = []
    prediction_dir = args.output / f"{args.tag}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        raw = load_sequences(dataset_name, args.features)
        for test_index, raw_test in enumerate(raw):
            validation_index = (test_index + 1) % len(raw)
            train_raw = [
                sequence for index, sequence in enumerate(raw)
                if index not in {test_index, validation_index}
            ]
            scaler = RobustTrainScaler().fit([sequence.x for sequence in train_raw])
            train = transform_sequences(train_raw, scaler)
            validation = transform_sequences([raw[validation_index]], scaler)[0]
            test = transform_sequences([raw_test], scaler)[0]
            median_life = float(lifetime_quantiles(train_raw, (0.5,))[0])
            deterministic = {
                "constant_05": np.full(len(test.y), 0.5, dtype=np.float32),
                "median_lifetime_clock": median_clock(len(test.y), median_life),
            }
            for model_name, prediction in deterministic.items():
                record = {
                    "dataset": dataset_name,
                    "test_bearing": raw_test.name,
                    "validation_bearing": raw[validation_index].name,
                    "train_bearings": ",".join(s.name for s in train_raw),
                    "model": model_name,
                    "seed": 0,
                    **metrics(test.y, prediction),
                    "parameters": 0,
                    "selected_alpha": np.nan,
                    "reference_l50": median_life,
                }
                records.append(record)
                np.savez_compressed(
                    prediction_dir / f"{dataset_name}_{raw_test.name}_{model_name}.npz",
                    target=test.y, prediction=prediction,
                )

            x_train = np.concatenate([s.x for s in train], axis=0)
            y_train = np.concatenate([s.y for s in train], axis=0)
            best_alpha, best_mae = None, float("inf")
            for alpha in args.alphas:
                candidate = Ridge(alpha=alpha).fit(x_train, y_train)
                val_prediction = np.clip(candidate.predict(validation.x), 0.0, 1.0)
                val_mae = float(np.mean(np.abs(val_prediction - validation.y)))
                if val_mae < best_mae:
                    best_mae, best_alpha = val_mae, alpha
            model = Ridge(alpha=best_alpha).fit(x_train, y_train)
            prediction = np.clip(model.predict(test.x), 0.0, 1.0)
            records.append({
                "dataset": dataset_name,
                "test_bearing": raw_test.name,
                "validation_bearing": raw[validation_index].name,
                "train_bearings": ",".join(s.name for s in train_raw),
                "model": "ridge_linear_vibration",
                "seed": 0,
                **metrics(test.y, prediction),
                "parameters": int(model.coef_.size + 1),
                "selected_alpha": best_alpha,
                "reference_l50": median_life,
            })
            np.savez_compressed(
                prediction_dir / f"{dataset_name}_{raw_test.name}_ridge_linear_vibration.npz",
                target=test.y, prediction=prediction,
            )
            pd.DataFrame(records).to_csv(args.output / f"{args.tag}_results.csv", index=False)
    return pd.DataFrame(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results")
    parser.add_argument("--tag", default="trivial_baselines")
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    frame = run(arguments)
    print(frame.groupby(["dataset", "model"])[["mae", "rmse"]].mean())

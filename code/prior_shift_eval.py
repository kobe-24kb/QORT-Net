"""Inference-time lifetime-prior misspecification test using archived checkpoints.

For every fold and seed, the training-fold reference lifetimes L20/L50/L80 are
multiplied by a factor f before the age and prior channels are formed; the
trained weights are not changed. The training-fold CV is scale invariant and is
therefore unchanged. The same perturbation is applied to the information-equated
GRU and to the zero-learning median-lifetime clock, so the test asks how much of
each model's accuracy survives a population prior that is systematically too
short (f < 1) or too long (f > 1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from protocol import (
    QORTWindowDataset,
    RobustTrainScaler,
    healthy_reference_center,
    lifetime_cv,
    lifetime_quantiles,
    load_sequences,
    metrics,
    resolve_datasets,
    transform_sequences,
)
from qort_net import QORTNet
from run_experiments import predict as qort_predict
from run_targeted_baselines import WindowDataset, append_time_prior_channels
from run_targeted_baselines import predict as baseline_predict
from targeted_baselines import make_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def cyclic_folds(count: int):
    for test_index in range(count):
        validation_index = (test_index + 1) % count
        yield test_index, validation_index, [
            i for i in range(count) if i not in {test_index, validation_index}
        ]


def clock(length: int, life: float) -> np.ndarray:
    elapsed = np.arange(length, dtype=np.float32)
    return np.clip(1.0 - elapsed / max(life - 1.0, 1.0), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--factors", default="0.7,0.85,1.0,1.15,1.3")
    parser.add_argument("--seeds", default="7,21,42")
    parser.add_argument("--qort-tag", default="main_qort")
    parser.add_argument("--gru-tag", default="main_equated")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "prior_shift_results.csv")
    args = parser.parse_args()
    factors = [float(v) for v in args.factors.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for dataset in resolve_datasets(args.dataset):
        absolute = load_sequences(dataset)
        relative = healthy_reference_center(absolute, 5)
        for test_index, validation_index, train in cyclic_folds(len(absolute)):
            train_abs = [absolute[i] for i in train]
            train_rel = [relative[i] for i in train]
            abs_scaler = RobustTrainScaler().fit([s.x for s in train_abs])
            rel_scaler = RobustTrainScaler().fit([s.x for s in train_rel])
            test_abs = transform_sequences([absolute[test_index]], abs_scaler)[0]
            test_rel = transform_sequences([relative[test_index]], rel_scaler)[0]
            lives = lifetime_quantiles(train_abs, (0.2, 0.5, 0.8))
            dispersion = lifetime_cv(train_abs)
            name = absolute[test_index].name
            for factor in factors:
                shifted = (lives * factor).astype(np.float32)
                target = test_abs.y
                records.append({"dataset": dataset, "test_bearing": name, "factor": factor,
                                "model": "median_lifetime_clock", "seed": 0,
                                **metrics(target, clock(len(target), float(shifted[1])))})
                qort_data = QORTWindowDataset([test_abs], [test_rel], 40, shifted)
                for seed in seeds:
                    state = torch.load(
                        ROOT / "checkpoints" / args.qort_tag / f"{dataset}_{name}_qort_net_seed{seed}.pt",
                        map_location=device,
                    )
                    model = QORTNet(input_dim=test_abs.x.shape[1]).to(device)
                    model.load_state_dict(state)
                    output = qort_predict(model, qort_data, device, 512, dispersion)
                    prediction = np.clip(output["prediction"], 0.0, 1.0)
                    records.append({"dataset": dataset, "test_bearing": name, "factor": factor,
                                    "model": "qort_net", "seed": seed,
                                    **metrics(target, prediction),
                                    "mean_gate": float(np.mean(output["gate"]))})
                    # Information-equated GRU with the same shifted channels.
                    base_scaler = RobustTrainScaler().fit([s.x for s in train_abs])
                    test_base = transform_sequences([absolute[test_index]], base_scaler)[0]
                    test_base = append_time_prior_channels([test_base], shifted, dispersion)[0]
                    gru = make_model("gru", input_dim=test_base.x.shape[1], hidden_dim=48,
                                     patch_size=4, window=20, dropout=0.15).to(device)
                    gru.load_state_dict(torch.load(
                        ROOT / "checkpoints" / args.gru_tag / f"{dataset}_{name}_gru_time_prior_s{seed}.pt",
                        map_location=device,
                    ))
                    prediction = baseline_predict(gru, WindowDataset([test_base], 20), device, 512)
                    records.append({"dataset": dataset, "test_bearing": name, "factor": factor,
                                    "model": "gru_time_prior", "seed": seed,
                                    **metrics(target, prediction)})
            print(dataset, name, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output, index=False)
    print(frame.groupby(["dataset", "model", "factor"]).mae.mean().unstack())


if __name__ == "__main__":
    main()

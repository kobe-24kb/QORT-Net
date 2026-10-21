"""Cross-condition transfer with the frozen QORT-Net release (descriptive study).

A model is fitted on one operating condition and applied, unchanged, to every
bearing of another condition of the same platform. Scaling, lifetime quantiles,
lifetime CV, early stopping and checkpoint selection use source bearings only.
"""

from __future__ import annotations

import argparse
import json
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
    transform_sequences,
)
from qort_net import QORTNet
from run_experiments import build_parser as qort_parser
from run_experiments import predict as qort_predict
from run_experiments import train_one
from run_targeted_baselines import (
    WindowDataset,
    append_time_prior_channels,
    build_parser as baseline_parser,
    predict as baseline_predict,
    train_fold,
)
from targeted_baselines import make_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIRECTIONS = (("xjtu", "xjtu_c1"), ("xjtu_c1", "xjtu"), ("phm", "phm_c2"), ("phm_c2", "phm"))


def clock(length: int, life: float) -> np.ndarray:
    elapsed = np.arange(length, dtype=np.float32)
    return np.clip(1.0 - elapsed / max(life - 1.0, 1.0), 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="7,21,42")
    parser.add_argument("--tag", default="transfer")
    cli = parser.parse_args()
    seeds = [int(v) for v in cli.seeds.split(",")]
    qargs = qort_parser().parse_args([])
    bargs = baseline_parser().parse_args(["--information-equated"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / f"{cli.tag}_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source, target in DIRECTIONS:
        src_abs = load_sequences(source)
        src_rel = healthy_reference_center(src_abs, qargs.healthy_reference)
        tgt_abs = load_sequences(target)
        tgt_rel = healthy_reference_center(tgt_abs, qargs.healthy_reference)
        train_abs, val_abs = src_abs[:-1], src_abs[-1]
        train_rel, val_rel = src_rel[:-1], src_rel[-1]
        abs_scaler = RobustTrainScaler().fit([s.x for s in train_abs])
        rel_scaler = RobustTrainScaler().fit([s.x for s in train_rel])
        lives = lifetime_quantiles(train_abs, (0.2, 0.5, 0.8))
        dispersion = lifetime_cv(train_abs)
        a_train = transform_sequences(train_abs, abs_scaler)
        r_train = transform_sequences(train_rel, rel_scaler)
        a_val = transform_sequences([val_abs], abs_scaler)[0]
        r_val = transform_sequences([val_rel], rel_scaler)[0]
        a_tgt = transform_sequences(tgt_abs, abs_scaler)
        r_tgt = transform_sequences(tgt_rel, rel_scaler)
        common = {"source": source, "target": target, "validation_bearing": val_abs.name,
                  "lifetime_cv": dispersion, "reference_lives": json.dumps(lives.tolist())}

        for test in a_tgt:
            records.append({**common, "test_bearing": test.name, "model": "median_lifetime_clock",
                            "seed": 0, **metrics(test.y, clock(len(test.y), float(lives[1])))})

        for prior_only in (False, True):
            qargs.prior_only = prior_only
            name = "qort_prior_only" if prior_only else "qort_net"
            for seed in seeds:
                result = train_one(a_train, r_train, a_val, r_val, a_tgt[0], r_tgt[0],
                                   lives, dispersion, device, seed, qargs)
                model = QORTNet(input_dim=a_train[0].x.shape[1], prior_only=prior_only).to(device)
                model.load_state_dict(result["state_dict"])
                for a_test, r_test in zip(a_tgt, r_tgt):
                    data = QORTWindowDataset([a_test], [r_test], qargs.window, lives)
                    output = qort_predict(model, data, device, 512, dispersion)
                    prediction = np.clip(output["prediction"], 0.0, 1.0)
                    records.append({**common, "test_bearing": a_test.name, "model": name,
                                    "seed": seed, **metrics(a_test.y, prediction),
                                    "best_val_mae": result["best_val_mae"]})
                    np.savez_compressed(out_dir / f"{source}_to_{target}_{a_test.name}_{name}_seed{seed}.npz",
                                        target=a_test.y, prediction=prediction,
                                        mixed_prior=output["mixed_prior"], gate=output["gate"])
                print(source, target, name, seed, flush=True)

        b_train = append_time_prior_channels(a_train, lives, dispersion)
        b_val = append_time_prior_channels([a_val], lives, dispersion)[0]
        b_tgt = append_time_prior_channels(a_tgt, lives, dispersion)
        bargs.information_equated = False  # channels are appended above
        for model_name in ("gru", "film"):
            for seed in seeds:
                result, _ = train_fold(model_name, b_train, b_val, b_tgt[0], bargs, device, seed)
                model = make_model(model_name, input_dim=b_train[0].x.shape[1], hidden_dim=bargs.hidden_dim,
                                   patch_size=bargs.patch_size, window=bargs.window,
                                   dropout=bargs.dropout).to(device)
                model.load_state_dict(result["state_dict"])
                for test in b_tgt:
                    prediction = baseline_predict(model, WindowDataset([test], bargs.window), device, 512)
                    records.append({**common, "test_bearing": test.name,
                                    "model": f"{model_name}_time_prior", "seed": seed,
                                    **metrics(test.y, prediction), "best_val_mae": result["best_val_mae"]})
                print(source, target, model_name, seed, flush=True)
        pd.DataFrame(records).to_csv(ROOT / "results" / f"{cli.tag}_results.csv", index=False)
    frame = pd.DataFrame(records)
    print(frame.groupby(["source", "target", "model"]).mae.mean().unstack())


if __name__ == "__main__":
    main()

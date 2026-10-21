"""Train and evaluate QORT-Net under cyclic leave-one-bearing-out testing."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from protocol import (
    DEFAULT_FEATURES,
    QORTWindowDataset,
    RobustTrainScaler,
    healthy_reference_center,
    lifetime_cv,
    lifetime_quantiles,
    load_sequences,
    resolve_datasets,
    metrics,
    set_seed,
    transform_sequences,
)
from qort_net import QORTNet


HERE = Path(__file__).resolve().parent


@torch.no_grad()
def predict(model, dataset, device, batch_size, dispersion):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output = {
        "prediction": [],
        "vibration_prediction": [],
        "mixed_prior": [],
        "selector_weights": [],
        "gate": [],
        "prior_reliability": [],
        "regime_weight": [],
        "high_prediction": [],
        "low_prediction": [],
        "optimistic_conflict": [],
        "gate_activation": [],
        "priors": [],
    }
    model.eval()
    for absolute, relative, priors, age, _, _ in loader:
        absolute = absolute.to(device, non_blocking=True)
        relative = relative.to(device, non_blocking=True)
        priors_device = priors.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        cv = torch.full((len(absolute), 1), dispersion, device=device)
        prediction, auxiliary = model(absolute, relative, priors_device, age, cv)
        output["prediction"].append(prediction.float().cpu().numpy())
        for key in (
            "vibration_prediction",
            "mixed_prior",
            "selector_weights",
            "gate",
            "prior_reliability",
            "regime_weight",
            "high_prediction",
            "low_prediction",
            "optimistic_conflict",
            "gate_activation",
        ):
            output[key].append(auxiliary[key].float().cpu().numpy())
        output["priors"].append(priors.numpy())
    return {key: np.concatenate(value) for key, value in output.items()}


def train_one(
    absolute_train,
    relative_train,
    absolute_validation,
    relative_validation,
    absolute_test,
    relative_test,
    reference_lives,
    dispersion,
    device,
    seed,
    args,
):
    set_seed(seed)
    train_dataset = QORTWindowDataset(
        absolute_train, relative_train, args.window, reference_lives
    )
    validation_dataset = QORTWindowDataset(
        [absolute_validation], [relative_validation], args.window, reference_lives
    )
    test_dataset = QORTWindowDataset(
        [absolute_test], [relative_test], args.window, reference_lives
    )
    sampler = WeightedRandomSampler(
        train_dataset.weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )
    model = QORTNet(
        input_dim=absolute_train[0].x.shape[1],
        number_of_priors=len(reference_lives),
        window=args.window,
        short_window=args.short_window,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gate_variation=args.gate_variation,
        use_relative=not args.no_relative_view,
        use_quantile_bank=not args.single_prior,
        use_disagreement_gate=not args.static_gate,
        vibration_only=args.vibration_only,
        prior_only=args.prior_only,
        router_mode=args.router_mode,
        router_value=args.router_value,
        gate_base=args.gate_base,
        gate_base_value=args.gate_base_value,
        token_mode=args.token_mode,
        use_optimism_term=not args.no_optimism_term,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=0.08)
    use_amp = device.type == "cuda"
    amp_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best_state, best_validation, stale = None, float("inf"), 0
    epochs_run = 0

    for epoch in range(args.epochs):
        model.train()
        for absolute, relative, priors, age, prior_class, target in loader:
            absolute = absolute.to(device, non_blocking=True)
            relative = relative.to(device, non_blocking=True)
            priors = priors.to(device, non_blocking=True)
            age = age.to(device, non_blocking=True)
            prior_class = prior_class.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            cv = torch.full((len(absolute), 1), dispersion, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                prediction, auxiliary = model(absolute, relative, priors, age, cv)
                loss = criterion(prediction, target)
                if not args.prior_only:
                    loss = loss + args.lambda_vibration * criterion(
                        auxiliary["vibration_prediction"], target
                    )
                if not args.single_prior and not args.vibration_only:
                    loss = loss + args.lambda_selector * F.nll_loss(
                        auxiliary["selector_weights"].clamp_min(1e-8).log(),
                        prior_class,
                    )
                if args.lambda_over > 0:
                    terminal_weight = (1.0 - target).clamp(0.0, 1.0).pow(2)
                    overestimate = torch.relu(prediction - target)
                    loss = loss + args.lambda_over * (
                        terminal_weight * overestimate.pow(2)
                    ).mean()
                if args.lambda_gate > 0 and not args.static_gate and not args.prior_only:
                    gate = auxiliary["gate"]
                    loss = loss + args.lambda_gate * (
                        gate - gate.mean()
                    ).pow(2).mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        validation = predict(
            model, validation_dataset, device, args.batch_size, dispersion
        )
        validation_mae = float(
            np.mean(np.abs(validation["prediction"] - validation_dataset.y.numpy()))
        )
        scheduler.step(validation_mae)
        epochs_run = epoch + 1
        if validation_mae < best_validation - 1e-5:
            best_validation = validation_mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test = predict(model, test_dataset, device, args.batch_size, dispersion)
    test["prediction"] = np.clip(test["prediction"], 0.0, 1.0)
    test["target"] = test_dataset.y.numpy()
    test["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    test["best_val_mae"] = best_validation
    test["epochs_run"] = epochs_run
    test["state_dict"] = best_state
    return test


def experiment_name(args) -> str:
    if args.model_name:
        return args.model_name
    if args.prior_only:
        return "qort_prior_only"
    if args.vibration_only:
        return "qort_vibration_only"
    if args.single_prior:
        return "qort_single_prior"
    if args.no_relative_view:
        return "qort_no_relative_view"
    if args.static_gate:
        return "qort_static_gate"
    return "qort_net"


def run(args) -> pd.DataFrame:
    datasets = resolve_datasets(args.dataset)
    seeds = [int(value) for value in args.seeds.split(",")]
    quantiles = tuple(float(value) for value in args.quantiles.split(","))
    if len(quantiles) != 3:
        raise ValueError("QORT-Net currently requires exactly three lifetime quantiles")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = experiment_name(args)
    prediction_dir = args.output / f"{args.tag}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = HERE.parent / "checkpoints" / args.tag
    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print(json.dumps({"device": str(device), "model": model_name}), flush=True)

    for dataset_name in datasets:
        source_absolute = load_sequences(dataset_name, args.features)
        source_relative = healthy_reference_center(
            source_absolute, args.healthy_reference
        )
        for test_index, source_test in enumerate(source_absolute):
            validation_index = (test_index + 1) % len(source_absolute)
            train_indices = [
                index
                for index in range(len(source_absolute))
                if index not in {test_index, validation_index}
            ]
            subsets = [(None, train_indices)]
            if args.train_subsets:
                subsets = [
                    (dropped, [index for index in train_indices if index != dropped])
                    for dropped in train_indices
                ]
            for dropped, subset_indices in subsets:
                fold_records = run_fold(
                    args, dataset_name, source_absolute, source_relative,
                    test_index, validation_index, subset_indices, dropped,
                    seeds, quantiles, device, model_name, prediction_dir,
                    checkpoint_dir,
                )
                records.extend(fold_records)
                pd.DataFrame(records).to_csv(
                    args.output / f"{args.tag}_results.csv", index=False
                )
    return pd.DataFrame(records)


def run_fold(
    args, dataset_name, source_absolute, source_relative, test_index,
    validation_index, train_indices, dropped, seeds, quantiles, device,
    model_name, prediction_dir, checkpoint_dir,
):
    records = []
    source_test = source_absolute[test_index]
    if True:
        if True:
            raw_absolute_train = [source_absolute[index] for index in train_indices]
            raw_relative_train = [source_relative[index] for index in train_indices]
            absolute_scaler = RobustTrainScaler().fit(
                [sequence.x for sequence in raw_absolute_train]
            )
            relative_scaler = RobustTrainScaler().fit(
                [sequence.x for sequence in raw_relative_train]
            )
            absolute_train = transform_sequences(raw_absolute_train, absolute_scaler)
            relative_train = transform_sequences(raw_relative_train, relative_scaler)
            absolute_validation = transform_sequences(
                [source_absolute[validation_index]], absolute_scaler
            )[0]
            relative_validation = transform_sequences(
                [source_relative[validation_index]], relative_scaler
            )[0]
            absolute_test = transform_sequences(
                [source_absolute[test_index]], absolute_scaler
            )[0]
            relative_test = transform_sequences(
                [source_relative[test_index]], relative_scaler
            )[0]
            reference_lives = lifetime_quantiles(raw_absolute_train, quantiles)
            dispersion = lifetime_cv(raw_absolute_train)

            for seed in seeds:
                started = time.perf_counter()
                result = train_one(
                    absolute_train,
                    relative_train,
                    absolute_validation,
                    relative_validation,
                    absolute_test,
                    relative_test,
                    reference_lives,
                    dispersion,
                    device,
                    seed,
                    args,
                )
                suffix = "" if dropped is None else f"_drop{source_absolute[dropped].name}"
                record = {
                    "dataset": dataset_name,
                    "test_bearing": source_test.name,
                    "validation_bearing": source_absolute[validation_index].name,
                    "train_bearings": ",".join(source_absolute[i].name for i in train_indices),
                    "dropped_train_bearing": ""
                    if dropped is None else source_absolute[dropped].name,
                    "model": model_name,
                    "seed": seed,
                    "regime_weight": float(np.mean(result["regime_weight"])),
                    **metrics(result["target"], result["prediction"]),
                    "vibration_mae": metrics(
                        result["target"], result["vibration_prediction"]
                    )["mae"],
                    "prior_mae": metrics(result["target"], result["mixed_prior"])[
                        "mae"
                    ],
                    "mean_gate": float(np.mean(result["gate"])),
                    "gate_std": float(np.std(result["gate"])),
                    "lifetime_cv": dispersion,
                    "reference_lives": json.dumps(reference_lives.tolist()),
                    "best_val_mae": result["best_val_mae"],
                    "epochs_run": result["epochs_run"],
                    "parameters": result["parameters"],
                    "train_seconds": time.perf_counter() - started,
                }
                records.append(record)
                payload = {
                    key: value
                    for key, value in result.items()
                    if key
                    in {
                        "target",
                        "prediction",
                        "vibration_prediction",
                        "mixed_prior",
                        "selector_weights",
                        "gate",
                        "prior_reliability",
                        "regime_weight",
                        "high_prediction",
                        "low_prediction",
                        "optimistic_conflict",
                        "gate_activation",
                        "priors",
                    }
                }
                np.savez_compressed(
                    prediction_dir
                    / f"{dataset_name}_{source_test.name}_{model_name}{suffix}_seed{seed}.npz",
                    **payload,
                )
                if args.save_checkpoints:
                    torch.save(
                        result["state_dict"],
                        checkpoint_dir
                        / f"{dataset_name}_{source_test.name}_{model_name}{suffix}_seed{seed}.pt",
                    )
                print(json.dumps(record, ensure_ascii=False), flush=True)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results")
    parser.add_argument("--tag", default="qort_main_3seed")
    parser.add_argument("--seeds", default="7,21,42")
    parser.add_argument("--quantiles", default="0.2,0.5,0.8")
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--gate-variation", type=float, default=1.5)
    parser.add_argument("--healthy-reference", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-vibration", type=float, default=0.20)
    parser.add_argument("--lambda-selector", type=float, default=0.02)
    parser.add_argument("--lambda-over", type=float, default=0.05)
    parser.add_argument("--lambda-gate", type=float, default=0.002)
    parser.add_argument("--single-prior", action="store_true")
    parser.add_argument("--no-relative-view", action="store_true")
    parser.add_argument("--static-gate", action="store_true")
    parser.add_argument("--vibration-only", action="store_true")
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--router-mode", choices=("cv", "fixed"), default="cv")
    parser.add_argument("--router-value", type=float, default=0.5)
    parser.add_argument("--gate-base", choices=("coupled", "constant"), default="coupled")
    parser.add_argument("--gate-base-value", type=float, default=0.575)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--train-subsets", action="store_true")
    parser.add_argument("--token-mode", choices=("full", "no_spectral", "no_trend", "raw"),
                        default="full")
    parser.add_argument("--no-optimism-term", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.tag}_config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    frame = run(args)
    columns = ["mae", "rmse", "late_mae", "monotonic_violation"]
    print(frame.groupby(["dataset", "model"])[columns].agg(["mean", "std"]))


if __name__ == "__main__":
    main()

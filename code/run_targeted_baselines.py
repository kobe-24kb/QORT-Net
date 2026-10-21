"""Run model-specific baselines under the same cyclic LOBO protocol."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from protocol import (
    DATASET_BEARINGS,
    DEFAULT_FEATURES,
    RobustTrainScaler,
    SequenceData,
    lifetime_cv,
    lifetime_quantiles,
    load_sequences,
    resolve_datasets,
    metrics,
    set_seed,
    transform_sequences,
)
from targeted_baselines import make_model, parameter_count


HERE = Path(__file__).resolve().parent


def append_time_prior_channels(
    sequences: list[SequenceData], reference_lives: np.ndarray, dispersion: float
) -> list[SequenceData]:
    """Append the same causal age/prior/CV channels used by QORT-Net."""
    reference_lives = np.asarray(reference_lives, dtype=np.float32)
    age_denominator = max(float(np.median(reference_lives)) - 1.0, 1.0)
    augmented = []
    for sequence in sequences:
        elapsed = np.arange(len(sequence.y), dtype=np.float32)
        age = np.clip(elapsed / age_denominator, 0.0, 2.0)[:, None]
        priors = np.stack(
            [
                np.clip(1.0 - elapsed / max(float(life) - 1.0, 1.0), 0.0, 1.0)
                for life in reference_lives
            ],
            axis=1,
        ).astype(np.float32)
        cv = np.full((len(sequence.y), 1), dispersion, dtype=np.float32)
        x = np.concatenate([sequence.x, age, priors, cv], axis=1).astype(np.float32)
        augmented.append(SequenceData(sequence.name, x, sequence.y.copy()))
    return augmented


class WindowDataset(Dataset):
    def __init__(self, sequences: list[SequenceData], window: int):
        windows, labels, weights = [], [], []
        for sequence in sequences:
            padded = np.concatenate(
                [np.repeat(sequence.x[:1], window - 1, axis=0), sequence.x], axis=0
            )
            view = np.lib.stride_tricks.sliding_window_view(
                padded, window_shape=window, axis=0
            ).transpose(0, 2, 1)
            windows.append(np.ascontiguousarray(view, dtype=np.float32))
            labels.append(sequence.y.astype(np.float32))
            weights.append(np.full(len(sequence.y), 1.0 / len(sequence.y)))
        self.x = torch.from_numpy(np.concatenate(windows))
        self.y = torch.from_numpy(np.concatenate(labels))
        self.weights = np.concatenate(weights)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    output = []
    for x, _ in loader:
        output.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(output)


def apply_training_pca(train, validation, test):
    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(np.concatenate([sequence.x for sequence in train], axis=0))

    def convert(sequence):
        return SequenceData(
            sequence.name, pca.transform(sequence.x).astype(np.float32), sequence.y.copy()
        )

    return [convert(s) for s in train], convert(validation), convert(test), int(pca.n_components_)


def train_fold(model_name, train, validation, test, args, device, seed):
    set_seed(seed)
    pca_components = None
    if args.information_equated:
        reference_lives = lifetime_quantiles(train, (0.2, 0.5, 0.8))
        dispersion = lifetime_cv(train)
        train = append_time_prior_channels(train, reference_lives, dispersion)
        validation = append_time_prior_channels(
            [validation], reference_lives, dispersion
        )[0]
        test = append_time_prior_channels([test], reference_lives, dispersion)[0]
    if model_name == "pca_drsn_tcn":
        train, validation, test, pca_components = apply_training_pca(
            train, validation, test
        )
    train_data = WindowDataset(train, args.window)
    validation_data = WindowDataset([validation], args.window)
    test_data = WindowDataset([test], args.window)
    sampler = WeightedRandomSampler(
        train_data.weights, len(train_data), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler,
        pin_memory=device.type == "cuda",
    )
    model = make_model(
        model_name, input_dim=train[0].x.shape[1], hidden_dim=args.hidden_dim,
        patch_size=args.patch_size, window=args.window, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.plateau_patience, min_lr=1e-5
    )
    criterion = nn.SmoothL1Loss(beta=0.08)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    best_state, best_validation, stale = None, float("inf"), 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction_batch = model(x)
                loss = criterion(prediction_batch, y)
                if args.lambda_over > 0:
                    terminal_weight = (1.0 - y).clamp(0.0, 1.0).pow(2)
                    overestimate = torch.relu(prediction_batch - y)
                    loss = loss + args.lambda_over * (
                        terminal_weight * overestimate.pow(2)
                    ).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        validation_prediction = predict(model, validation_data, device, args.batch_size)
        validation_mae = float(np.mean(np.abs(validation_prediction - validation.y)))
        scheduler.step(validation_mae)
        if validation_mae < best_validation - 1e-5:
            best_validation = validation_mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.load_state_dict(best_state)
    prediction = predict(model, test_data, device, args.batch_size)
    result = metrics(test.y, prediction)
    result.update(
        best_val_mae=best_validation, epochs_run=epoch + 1,
        parameters=parameter_count(model), train_seconds=time.perf_counter() - started,
    )
    if pca_components is not None:
        result["pca_components"] = pca_components
    result["state_dict"] = best_state
    return result, prediction


def run(args):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    datasets = list(resolve_datasets(args.dataset))
    checkpoint_dir = HERE.parent / "checkpoints" / args.tag
    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",")]
    records = []
    prediction_dir = args.output / f"{args.tag}_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    for dataset_name in datasets:
        raw_sequences = load_sequences(dataset_name, args.features)
        for test_index, raw_test in enumerate(raw_sequences):
            validation_index = (test_index + 1) % len(raw_sequences)
            raw_validation = raw_sequences[validation_index]
            raw_train = [
                sequence for index, sequence in enumerate(raw_sequences)
                if index not in {test_index, validation_index}
            ]
            scaler = RobustTrainScaler().fit([sequence.x for sequence in raw_train])
            train = transform_sequences(raw_train, scaler)
            validation = transform_sequences([raw_validation], scaler)[0]
            test = transform_sequences([raw_test], scaler)[0]
            for model_name in models:
                for seed in seeds:
                    print(dataset_name, raw_test.name, model_name, seed, flush=True)
                    result, prediction = train_fold(
                        model_name, train, validation, test, args, device, seed
                    )
                    state_dict = result.pop("state_dict")
                    label = (
                        f"{model_name}{'_time_prior' if args.information_equated else ''}"
                        f"{args.model_suffix}"
                    )
                    if args.save_checkpoints:
                        torch.save(
                            state_dict,
                            checkpoint_dir / f"{dataset_name}_{raw_test.name}_{label}_s{seed}.pt",
                        )
                    record = {
                        "dataset": dataset_name, "test_bearing": raw_test.name,
                        "validation_bearing": raw_validation.name,
                        "train_bearings": ",".join(s.name for s in raw_train),
                        "model": label, "seed": seed, **result,
                    }
                    records.append(record)
                    np.savez_compressed(
                        prediction_dir / f"{dataset_name}_{raw_test.name}_{label}_s{seed}.npz",
                        target=raw_test.y, prediction=prediction,
                    )
                    pd.DataFrame(records).to_csv(
                        args.output / f"{args.tag}_results.csv", index=False
                    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / f"{args.tag}_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return pd.DataFrame(records)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=HERE.parent / "results")
    parser.add_argument("--tag", default="targeted_baselines_3seed")
    parser.add_argument("--models", default="film,filternet")
    parser.add_argument("--seeds", default="7,21,42")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--information-equated", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--lambda-over", type=float, default=0.0)
    parser.add_argument("--model-suffix", default="")
    parser.add_argument("--save-checkpoints", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

"""Check distributed feature caches, fold isolation and model variants on CPU."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from protocol import DATASET_BEARINGS, load_sequences
from prior_shift_eval import cyclic_folds
from qort_net import QORTNet

def main():
    torch.set_num_threads(1)
    count = 0
    for dataset in DATASET_BEARINGS:
        sequences = load_sequences(dataset)
        for sequence in sequences:
            assert sequence.x.ndim == 2 and sequence.x.shape[1] == 42
            assert len(sequence.x) == len(sequence.y) and len(sequence.y) > 1
            assert np.isclose(sequence.y[0], 1) and np.isclose(sequence.y[-1], 0)
            count += 1
        for test, valid, train in cyclic_folds(len(sequences)):
            assert test != valid and test not in train and valid not in train
            assert sorted([test, valid, *train]) == list(range(len(sequences)))
    absolute = torch.randn(4, 40, 42)
    relative = torch.randn(4, 40, 42)
    priors = torch.tensor([[0.2, 0.5, 0.8]]).repeat(4, 1)
    age = torch.full((4, 1), 0.25)
    cv = torch.full((4, 1), 0.39)
    variants = [
        {}, {"token_mode": "no_spectral"}, {"token_mode": "no_trend"},
        {"token_mode": "raw"}, {"use_optimism_term": False},
        {"use_relative": False}, {"use_quantile_bank": False},
        {"use_disagreement_gate": False}, {"vibration_only": True},
        {"prior_only": True},
    ]
    with torch.inference_mode():
        for options in variants:
            model = QORTNet(input_dim=42, **options).eval()
            prediction, details = model(absolute, relative, priors, age, cv)
            assert prediction.shape == (4,) and torch.isfinite(prediction).all()
            assert ((prediction >= 0) & (prediction <= 1)).all()
            assert details["gate"].shape == (4,)
    print(f"PASS: {count} feature caches, isolated cyclic folds and {len(variants)} model variants.")

if __name__ == "__main__":
    main()

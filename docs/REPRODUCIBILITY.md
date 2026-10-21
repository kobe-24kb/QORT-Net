# Reproducibility and provenance

The repository contains the QORT-Net v1.1 research release and a small portability patch.
`results/release_manifest_v1.0.json` and `results/release_manifest_v1.1.json` retain the
original freeze timestamps and hashes. They describe the archived research releases.

The GitHub packaging changes `code/prepare_features.py`,
`code/extract_condition_features.py` and `code/queues/common.sh` for portable paths,
explicit raw-data arguments and queue error handling. Model computation, training,
metrics, feature arrays, checkpoints and recorded predictions are retained.
`docs/packaging_changes.json` lists every modified original file with before/after hashes.
`docs/repository_manifest.json` records the distributed files. Run
`python scripts/verify_release.py` to check their integrity using only Python's standard library.

## Data protocol

The order of bearings is defined in `code/protocol.py`. Fold i tests bearing i,
validates on bearing (i + 1) modulo N, and trains on the remaining bearings.
Scaling and lifetime quantiles are fitted on training bearings. The fixed seeds are
7, 21 and 42. `all` selects xjtu, phm, xjtu_c1 and phm_c2; phm60 is evaluated separately.

Feature caches, archived checkpoints, per-bearing predictions, aggregate result CSVs
and figure/table source data are distributed together. Raw vibration snapshots are
obtained from the dataset providers. `data/raw/` and new logs are ignored by Git.

## Verification

```bash
python scripts/verify_release.py
python scripts/smoke_check.py
python code/run_trivial_baselines.py --dataset xjtu --output verification_output --tag trivial_check
python code/run_experiments.py --dataset xjtu --seeds 7 --epochs 1 --output verification_output --tag training_check
python code/prior_shift_eval.py --dataset xjtu --seeds 7 --factors 1.0 --output verification_output/prior_shift_check.csv
```

The one-epoch command checks the training pipeline; it does not reproduce the paper's
converged results. The prior-shift command checks archived QORT-Net and GRU inference.
Full reproduction uses the original queue settings listed in the main README.

The reference environment uses Python 3.10 and PyTorch 2.9.0 with CUDA 13.0 on an
NVIDIA GeForce RTX 5060 Ti. CPU execution is available through `--device cpu` for the
training scripts. Numerical agreement across hardware and library builds should be
evaluated with the recorded metrics and seeds.

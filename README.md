# QORT-Net: fusing healthy-relative vibration representations with training-fold lifetime priors for bearing remaining useful life prediction

This repository distributes the QORT-Net v1.1 research code, feature caches,
archived model checkpoints, experiment results and figure/table source data for
the manuscript submitted to *Measurement Science and Technology*.

QORT-Net combines absolute and healthy-relative vibration views with a bank of
lifetime priors computed from training bearings. The release includes component
ablations, information-equated and vibration-only baselines, cross-condition
transfer, prior misspecification and training-subset sensitivity studies.

## Repository layout

```text
code/          Model, training, preprocessing, analysis and experiment queues
data/features/ Precomputed per-snapshot feature caches
checkpoints/   Archived trained model state dictionaries
results/       Configurations, metrics, predictions and prespecified plans
source_data/   Source tables for the manuscript figures and statistical analyses
scripts/       Release integrity and model/data smoke checks
docs/          Reproducibility notes, provenance and file hashes
```

## Installation

Use Python 3.10. Create and activate an isolated environment:

```bash
conda create -n qort-net python=3.10 -y
conda activate qort-net
```

Install PyTorch for the target platform. The reference experiments used PyTorch
2.9.0 with CUDA 13.0 on an NVIDIA GeForce RTX 5060 Ti:

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
```

For CPU execution, replace the first command with:

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

Commands below run from the repository root. On Windows, individual Python
commands work in PowerShell; the experiment queues require Bash (for example,
Git Bash). Activate the environment before running a queue. Set `PY` to override
the Python executable when needed.

## Quick check

The included feature caches allow evaluation without downloading raw data.

```bash
python scripts/verify_release.py
python scripts/smoke_check.py
python code/run_trivial_baselines.py --dataset xjtu --output verification_output --tag quickcheck
```

The quick check writes new outputs under the ignored `verification_output/`
directory. See [reproducibility notes](docs/REPRODUCIBILITY.md) for training and
archived-checkpoint inference checks.

## Datasets

The caches cover XJTU-SY conditions 1 and 2, PRONOSTIA conditions 1 and 2, and a
60-second PRONOSTIA condition-1 variant. Raw vibration snapshots are distributed
by the [XJTU-SY dataset provider](https://biaowang.tech/xjtu-sy-bearing-datasets/)
and the [IEEE PHM 2012 dataset archive](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset).

To rebuild the caches, replace the example roots with the downloaded dataset paths:

```bash
python code/prepare_features.py --xjtu-root /path/to/XJTU-SY/37.5Hz11kN --phm-root /path/to/PHM2012
python code/extract_condition_features.py --condition xjtu_c1 --xjtu-root /path/to/XJTU-SY
python code/extract_condition_features.py --condition phm_c2 --phm-root /path/to/PHM2012
python code/extract_condition_features.py --condition phm60
```

`PHM2012` must contain `Learning_set/` and `Full_Test_Set/`. The XJTU-SY root
for additional conditions must contain `35Hz12kN/` and `37.5Hz11kN/`.
All extraction commands write to `data/features/` by default.

## Reproduce the experiments

```bash
bash code/queues/A_main.sh
bash code/queues/B1_ablations.sh
bash code/queues/B2_router.sh
bash code/queues/C_baselines.sh
bash code/queues/D_grid_subsets.sh
bash code/queues/E_transfer.sh
bash code/queues/F_prior_shift.sh
bash code/queues/G_no_optimism.sh
bash code/queues/G_no_spectral.sh
bash code/queues/G_no_trend.sh
bash code/queues/G_raw.sh
```

| Queue | Experiment |
| --- | --- |
| A | Main QORT-Net, prior-only and information-equated controls; sampling intervals and windows |
| B1 / B2 | Component ablations; router and gate variants |
| C | Vibration-only, budget-matched and grid-searched baselines |
| D | QORT-Net grid and training-subset perturbation |
| E / F | Cross-condition transfer; inference-time prior misspecification |
| G | Core tokenizer and optimism-term ablations |

The recorded outputs are included. Re-running a queue writes to its original
`results/` tags and may replace existing outputs. For exploratory Python runs,
use a separate `--output` directory and `--tag` where the entry point supports them.

Generate analysis tables and figures:

```bash
python code/analyze_resubmission.py
python code/analyze_sensitivity.py
python code/make_numbers.py
python code/make_resubmission_figures.py
```

These commands write source tables to `source_data/`, LaTeX tables to
`manuscript/tables/`, and figures to `figures/`.

## Protocol and release history

Seeds 7, 21 and 42 are fixed for Python, NumPy and PyTorch. Cyclic
leave-one-bearing-out folds use the bearing order in `code/protocol.py`.
Each result row records the test, validation and training bearings. Training-only
scaling and lifetime priors are shared across the relevant comparisons.

The original analysis plan and freeze manifests remain in `results/`:

- [Prespecified analysis plan](results/PRESPECIFIED_ANALYSIS_PLAN.md)
- [Core-ablation addendum](results/PRESPECIFIED_ADDENDUM_core_ablations.md)
- [Transfer addendum](results/PRESPECIFIED_ADDENDUM_transfer.md)
- [v1.0 freeze manifest](results/release_manifest_v1.0.json)
- [v1.1 freeze manifest](results/release_manifest_v1.1.json)

The GitHub packaging corrects execution paths and adds documentation and checks.
See the [changelog](CHANGELOG.md) and [provenance notes](docs/REPRODUCIBILITY.md).

## License and citation

The software is released under the [MIT License](LICENSE). Dataset terms are
set by their original providers. The final publication citation will be added
when available.

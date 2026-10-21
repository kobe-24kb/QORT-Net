# Code directory

See the [repository README](../README.md) for installation, data and execution commands.
Run Python entry points from the repository root as `python code/<script>.py`.
Run queues as `bash code/queues/<queue>.sh` with the desired environment activated.

| Module | Purpose |
| --- | --- |
| `prepare_features.py`, `extract_condition_features.py` | Per-snapshot descriptor extraction |
| `protocol.py` | Splits, scaling, windows, priors and metrics |
| `qort_net.py` | QORT-Net and its ablation variants |
| `targeted_baselines.py` | Local scalar-RUL baseline adaptations |
| `run_experiments.py` | Training, ablations and sensitivity runs |
| `run_targeted_baselines.py`, `run_trivial_baselines.py` | Baseline evaluation |
| `run_transfer.py`, `prior_shift_eval.py` | Transfer and prior misspecification |
| `analyze_resubmission.py`, `analyze_sensitivity.py` | Statistics and source data |
| `make_numbers.py`, `make_resubmission_figures.py` | Tables and data figures |
| `freeze_release.py` | Research-source freeze manifests |
| `queues/` | Original experiment settings with a portable launcher |

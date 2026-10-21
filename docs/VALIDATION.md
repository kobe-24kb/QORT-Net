# Validation report

Validated on 2026-10-21 with Python 3.10.19, PyTorch 2.9.0+cu130 and an available CUDA device.

| Check | Result |
| --- | --- |
| Feature-cache loading | 31 caches; 42 finite descriptors per snapshot; normalized target endpoints |
| Cyclic data splits | Train, validation and test bearings are disjoint in every fold |
| Model forward pass | 10 default/ablation variants return finite, bounded RUL predictions on CPU |
| Training pipeline | Five XJTU folds, seed 7, one epoch each completed on CUDA |
| Trivial baselines | Constant, median-lifetime clock and ridge completed on five XJTU folds |
| Archived checkpoint inference | QORT-Net and information-equated GRU completed on five XJTU folds, seed 7, prior factor 1.0 |
| Archived metric agreement | 15 rows matched; maximum MAE difference 0; maximum RMSE difference 0 |
| Python syntax | All code and verification scripts compile |
| Bash syntax | Every experiment queue parses successfully |
| README navigation | Local Markdown targets exist |
| Repository integrity | Distributed files recorded in the SHA-256 manifest |

The one-epoch run validates training execution. Full three-seed retraining of all
experiments was not repeated during repository packaging. Original recorded results
and checkpoints are included for inspection and reproduction.

# Pre-specified analysis plan for the frozen QORT-Net release

Written on 2026-10-17, after the code was frozen and before any model was trained or
evaluated on the held-out operating conditions. The SHA-256 hashes of this file, the
source code and the feature caches are recorded in `release_manifest.json`.

## 1. Frozen implementation

* Source: `code/` of this package (release v1.0). The unused prior-only selector and the
  unused dual-reference encoder were removed before freezing; each variant now
  instantiates only the modules it uses (full model: 36 643 parameters).
* All QORT-Net settings are the defaults of `run_experiments.py`: windows 40/20,
  quantiles 0.2/0.5/0.8, router centre 0.390 and scale 0.012, gate variation 1.5,
  base gate 0.20 + 0.75r, loss weights 0.20/0.02/0.05/0.002, AdamW (1e-3, 1e-4),
  batch 128, at most 120 epochs, patience 18, plateau patience 5.
* Information-equated baselines use the defaults of `run_targeted_baselines.py`
  (window 20, width 48, dropout 0.15, at most 100 epochs, patience 14).
* Seeds 7, 21 and 42. No setting is changed after any held-out result is seen.

## 2. Data sets

* Development subsets (used while the architecture was designed; results remain
  exploratory): XJTU-SY condition 2 (Bearing2_1–2_5) and PRONOSTIA condition 1
  (Bearing1_1–1_7).
* Held-out operating conditions (never used during development): XJTU-SY condition 1,
  35 Hz / 12 kN (Bearing1_1–1_5), and PRONOSTIA condition 2, 1650 r/min / 4200 N
  (Bearing2_1–2_7). PRONOSTIA condition 3 (three bearings) is excluded a priori because
  cyclic LOBO would leave a single training bearing, making the three lifetime quantiles
  identical and the training-fold CV zero.
* Protocol: cyclic LOBO within each condition, in the bearing order listed above;
  scaling, lifetime quantiles and CV fitted on training bearings only.

## 3. Primary hypothesis (held-out conditions)

H1: on the 12 held-out bearings, QORT-Net has lower bearing-level MAE than the
information-equated GRU (GRU + time/prior).

* Unit of analysis: bearing; metrics are averaged over the three seeds within a bearing.
* Effect size: mean paired difference Δ = MAE(QORT-Net) − MAE(GRU + time/prior), with the
  median difference and the number of bearings favouring QORT-Net.
* Tests: exact two-sided sign-flip permutation test of the mean difference (all 2^12
  sign assignments) and exact two-sided Wilcoxon signed-rank test.
* Interval: 95 % percentile bootstrap, resampling bearings within each condition
  (stratified), 20 000 resamples, random seed 20261017.

## 4. Secondary analyses

* QORT-Net versus FiLM-inspired + time/prior, PatchTST + time/prior, prior-only QORT
  and the median-lifetime clock on the held-out conditions; Holm correction across the
  five comparisons of sections 3–4.
* All 24 bearings (development + held-out): linear mixed-effects model of the per-seed
  paired difference with a fixed intercept, a fixed held-out indicator and a random
  intercept for bearing nested within condition. Reported as a sensitivity analysis.
* Three-seed component ablations, router/gate variants, training-subset perturbation,
  window-length and acquisition-interval sensitivity are descriptive; they are not
  used to select or modify the reported model.

## 5. Reporting rule

All held-out results are reported irrespective of direction or magnitude. Development
results are labelled exploratory because the development history used those bearings.

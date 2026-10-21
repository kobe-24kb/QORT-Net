# Addendum 2: core-component ablations (added after the held-out evaluation)

Recorded on 2026-10-17, after the held-out results were known, in response to an internal review
noting that the orthogonal-spectral tokens and the optimism term lacked direct ablations. These
experiments are descriptive; they do not change the reported model or any pre-specified test.

* Code: release v1.0 extended with two options that leave the default computation unchanged
  (`--token-mode`, `--no-optimism-term`, release v1.1). Re-running the default configuration on
  XJTU-SY C2 with seed 7 reproduced the archived v1.0 metrics exactly (maximum MAE difference 0).
* Variants, each on all four conditions with seeds 7, 21 and 42 and otherwise frozen settings:
  1. `no_spectral`: both experts without the four learned spectral tokens;
  2. `no_trend`: both experts without the orthogonal trend coefficients and residual RMS
     (endpoint change, endpoint state and spectral tokens retained);
  3. `raw`: the orthogonal-spectral tokenizer replaced by the chronological snapshots of each
     window as tokens (40 and 20 tokens), with the same embedding and mixer;
  4. `no_optimism`: the learned local gate correction retained, the optimistic-conflict term removed.
* Reporting: bearing-level MAE averaged over seeds and paired differences from the full model, as
  for the other ablations; all results are reported.

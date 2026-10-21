# Addendum 1 to the pre-specified analysis plan: cross-condition transfer

Written on 2026-10-17 before any transfer run was executed. Uses the same frozen
release (`release_manifest.json`); the new runner `code/run_transfer.py` only calls
the frozen training and prediction functions.

* Directions (same platform, different load/speed and lifetime distribution):
  XJTU-SY C2 -> C1, XJTU-SY C1 -> C2, PRONOSTIA C1 -> C2, PRONOSTIA C2 -> C1.
* Source protocol: the last bearing of the source condition (in the order of
  `DATASET_BEARINGS`) is the validation bearing; all other source bearings are training
  bearings. Scalers, lifetime quantiles and CV come from the source training bearings.
  No target-condition data are used for fitting, model selection or early stopping.
* Models: QORT-Net, prior-only QORT, GRU + time/prior, FiLM-inspired + time/prior and
  the median-lifetime clock; seeds 7, 21 and 42; every target bearing is tested.
* Reporting: bearing-level MAE averaged over seeds, per direction; descriptive only, no
  hypothesis test. All results are reported irrespective of direction.

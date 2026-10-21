"""Small-sample lifetime-quantile sensitivity and prior-misspecification summaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_resubmission import DATASET_LABEL, LENGTHS, LABEL, table_env, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
SOURCE = ROOT / "source_data"


def clock_mae(length: int, life: float) -> float:
    elapsed = np.arange(length, dtype=np.float64)
    target = np.linspace(1.0, 0.0, length)
    prediction = np.clip(1.0 - elapsed / max(life - 1.0, 1.0), 0.0, 1.0)
    return float(np.mean(np.abs(prediction - target)))


def subset_analysis() -> None:
    full = pd.read_csv(RESULTS / "main_qort_results.csv")
    full_prior = pd.read_csv(RESULTS / "main_prior_only_results.csv")
    sub = pd.read_csv(RESULTS / "subsets_qort_results.csv")
    sub_prior = pd.read_csv(RESULTS / "subsets_prior_only_results.csv")
    rows = []
    for frame_full, frame_sub, model in ((full, sub, "qort_net"), (full_prior, sub_prior, "qort_prior_only")):
        frame_full = frame_full[frame_full.dataset.isin(["xjtu", "phm"])]
        ref = frame_full.groupby(["dataset", "test_bearing"]).agg(
            mae_full=("mae", "mean"), lives_full=("reference_lives", "first")).reset_index()
        per = frame_sub.groupby(["dataset", "test_bearing", "dropped_train_bearing"]).agg(
            mae=("mae", "mean"), lives=("reference_lives", "first"), cv=("lifetime_cv", "first"),
            r=("regime_weight", "first")).reset_index()
        per = per.merge(ref, on=["dataset", "test_bearing"])
        per["model"] = model
        rows.append(per)
    per = pd.concat(rows, ignore_index=True)
    clock_rows = []
    for _, r in per[per.model == "qort_net"].iterrows():
        T = LENGTHS[(r.dataset, r.test_bearing)]
        lives = json.loads(r.lives)
        lives_full = json.loads(r.lives_full)
        clock_rows.append({**r.to_dict(), "model": "median_lifetime_clock",
                           "mae": clock_mae(T, lives[1]), "mae_full": clock_mae(T, lives_full[1])})
    per = pd.concat([per, pd.DataFrame(clock_rows)], ignore_index=True)
    per["L50"] = per.lives.map(lambda s: json.loads(s)[1])
    per["L50_full"] = per.lives_full.map(lambda s: json.loads(s)[1])
    per["L20"] = per.lives.map(lambda s: json.loads(s)[0])
    per["L80"] = per.lives.map(lambda s: json.loads(s)[2])
    per["rel_L50_change"] = (per.L50 - per.L50_full) / per.L50_full
    per["delta_mae"] = per.mae - per.mae_full
    per.to_csv(SOURCE / "training_subset_sensitivity.csv", index=False)

    summary = []
    for (dataset, model), g in per.groupby(["dataset", "model"]):
        fold_sd = g.groupby("test_bearing").mae.std(ddof=0).mean()
        rho, p = stats.spearmanr(np.abs(g.rel_L50_change), np.abs(g.delta_mae))
        summary.append({
            "dataset": dataset, "model": model, "subsets": len(g),
            "mae_full": g.groupby("test_bearing").mae_full.first().mean(),
            "mae_subset_mean": g.mae.mean(), "mean_abs_delta": np.abs(g.delta_mae).mean(),
            "max_delta": g.delta_mae.max(), "within_fold_sd": fold_sd,
            "mean_abs_rel_L50_change": np.abs(g.rel_L50_change).mean(),
            "max_abs_rel_L50_change": np.abs(g.rel_L50_change).max(),
            "spearman_absL50_absdelta": rho, "spearman_p": p,
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(SOURCE / "training_subset_summary.csv", index=False)
    header = ("Condition & Model & Full-train MAE & Drop-one MAE & Mean $|\\Delta|$ & Max $\\Delta$ & "
              "Within-fold SD")
    rows_tex = []
    for dataset in ("xjtu", "phm"):
        for model in ("qort_net", "qort_prior_only", "median_lifetime_clock"):
            r = summary[(summary.dataset == dataset) & (summary.model == model)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows_tex.append(
                f"{DATASET_LABEL[dataset]} & {LABEL[model]} & {r.mae_full:.3f} & {r.mae_subset_mean:.3f} & "
                f"{r.mean_abs_delta:.3f} & ${r.max_delta:+.3f}$ & {r.within_fold_sd:.3f}\\\\")
        rows_tex.append("\\addlinespace")
    write("subset_sensitivity.tex", table_env(
        "tab:subsets",
        "Sensitivity to the composition of the training set. For every fold, each training bearing is removed in turn (XJTU-SY C2: 15 two-bearing training sets; PHM2012 C1: 35 four-bearing training sets), the scalers, lifetime quantiles and CV are re-estimated and the model is retrained with three seeds. $\\Delta$ is the change in test MAE relative to the full training set of the same fold.",
        "l X c c c c c", header, rows_tex[:-1]))


def prior_shift_analysis() -> None:
    path = RESULTS / "prior_shift_results.csv"
    if not path.exists():
        print("[missing] prior_shift_results.csv")
        return
    shift = pd.read_csv(path)
    main = pd.read_csv(RESULTS / "main_qort_results.csv")
    check = shift[(shift.model == "qort_net") & (shift.factor == 1.0)].merge(
        main[["dataset", "test_bearing", "seed", "mae"]], on=["dataset", "test_bearing", "seed"],
        suffixes=("_reloaded", "_original"))
    verification = {
        "runs": int(len(check)),
        "max_abs_mae_difference": float(np.abs(check.mae_reloaded - check.mae_original).max()),
    }
    (SOURCE / "checkpoint_reload_verification.json").write_text(json.dumps(verification, indent=2))
    bearing = shift.groupby(["dataset", "test_bearing", "model", "factor"]).mae.mean().reset_index()
    bearing.to_csv(SOURCE / "prior_shift_bearing.csv", index=False)
    summary = bearing.groupby(["dataset", "model", "factor"]).mae.mean().unstack("factor")
    summary.to_csv(SOURCE / "prior_shift_summary.csv")
    factors = list(summary.columns)
    header = "Condition & Model & " + " & ".join(f"$f={f:g}$" for f in factors)
    rows_tex = []
    for dataset in ("xjtu", "phm", "xjtu_c1", "phm_c2"):
        for model in ("qort_net", "gru_time_prior", "median_lifetime_clock"):
            if (dataset, model) not in summary.index:
                continue
            values = summary.loc[(dataset, model)]
            rows_tex.append(f"{DATASET_LABEL[dataset]} & {LABEL[model]} & "
                            + " & ".join(f"{v:.3f}" for v in values) + "\\\\")
        rows_tex.append("\\addlinespace")
    write("prior_shift.tex", table_env(
        "tab:priorshift",
        "Inference-time misspecification of the lifetime prior. The training-fold reference lifetimes are multiplied by $f$ before the age and prior channels are formed; trained weights are unchanged. Entries are bearing-mean MAE (three seeds for the learned models).",
        "l X " + "c " * len(factors), header, rows_tex[:-1]))
    print(json.dumps(verification))


def transfer_analysis() -> None:
    path = RESULTS / "transfer_results.csv"
    if not path.exists():
        print("[missing] transfer_results.csv")
        return
    frame = pd.read_csv(path)
    bearing = frame.groupby(["source", "target", "test_bearing", "model"]).mae.mean().reset_index()
    bearing.to_csv(SOURCE / "transfer_bearing.csv", index=False)
    summary = bearing.groupby(["source", "target", "model"]).mae.mean().unstack("model")
    within = pd.read_csv(SOURCE / "bearing_level_metrics.csv")
    within = within.groupby(["dataset", "model"]).mae.mean().unstack("model")
    summary.to_csv(SOURCE / "transfer_summary.csv")
    models = ["qort_net", "qort_prior_only", "median_lifetime_clock", "gru_time_prior", "film_time_prior"]
    short = {"qort_net": "QORT-Net", "qort_prior_only": "Prior-only", "median_lifetime_clock": "Clock",
             "gru_time_prior": "GRU+TP", "film_time_prior": "FiLM+TP"}
    code = {"xjtu": "XJTU C2", "xjtu_c1": "XJTU C1", "phm": "PHM C1", "phm_c2": "PHM C2"}
    header = ("Direction & Source $[L_{20},L_{50},L_{80}]$ & " + " & ".join(short[m] for m in models)
              + " & Within target")
    rows_tex = []
    for (source, target), row in summary.iterrows():
        lives = json.loads(frame[(frame.source == source) & (frame.target == target)].reference_lives.iloc[0])
        values = [row.get(m, np.nan) for m in models]
        best = np.nanmin(values)
        cells = [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}" for v in values]
        rows_tex.append(f"{code[source]}$\\rightarrow${code[target].split()[1]} & "
                        f"[{lives[0]:.0f}, {lives[1]:.0f}, {lives[2]:.0f}] & " + " & ".join(cells)
                        + f" & {within.loc[target, 'qort_net']:.3f}\\\\")
    write("transfer.tex", table_env(
        "tab:transfer",
        "Cross-condition transfer (descriptive). Models are fitted and early-stopped on one condition and applied unchanged to all bearings of another condition of the same platform; reference lives are the source-condition $[L_{20},L_{50},L_{80}]$ in snapshots. Entries are bearing-mean MAE over three seeds; the last column repeats the within-condition QORT-Net result for the target.",
        "l X c c c c c c", header, rows_tex, size="\\scriptsize"))


if __name__ == "__main__":
    subset_analysis()
    prior_shift_analysis()
    transfer_analysis()

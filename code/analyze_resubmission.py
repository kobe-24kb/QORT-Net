"""Aggregate all frozen-release experiments into source data and manuscript tables.

Every number reported in the resubmitted manuscript is produced here from the
per-run CSV files and per-snapshot prediction arrays in ``results/``.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from protocol import DATASET_BEARINGS, SAMPLE_INTERVAL_SECONDS, load_sequences

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
SOURCE = ROOT / "source_data"
TABLES = ROOT / "manuscript" / "tables"
SOURCE.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20261017
BOOT = 20000

DEV = ("xjtu", "phm")
HELD = ("xjtu_c1", "phm_c2")
ALL = DEV + HELD
SET_OF = {"xjtu": "development", "phm": "development",
          "xjtu_c1": "held-out", "phm_c2": "held-out", "phm60": "interval"}
DATASET_LABEL = {
    "xjtu": "XJTU-SY C2", "phm": "PHM2012 C1",
    "xjtu_c1": "XJTU-SY C1", "phm_c2": "PHM2012 C2", "phm60": "PHM2012 C1 (60 s)",
}
LABEL = {
    "qort_net": "QORT-Net",
    "qort_prior_only": "Prior-only QORT",
    "median_lifetime_clock": "Median-lifetime clock",
    "constant_05": "Constant 0.5",
    "ridge_linear_vibration": "Ridge (vibration)",
    "gru_time_prior": "GRU + time/prior",
    "film_time_prior": "FiLM-inspired + time/prior",
    "patchtst_time_prior": "PatchTST + time/prior",
    "gru": "GRU", "film": "FiLM-inspired", "filternet": "FilterNet-inspired",
    "patchtst": "PatchTST", "pca_drsn_tcn": "PCA--DRSN--TCN",
    "mixlinear": "MixLinear-inspired", "vpnet": "VPNet-inspired",
    "qort_single_prior": "Single $q_{50}$ prior",
    "qort_static_gate": "Static gate ($g_t=g_0$)",
    "qort_no_relative_view": "No healthy-relative view",
    "qort_vibration_only": "Vibration only",
    "qort_no_aux": "No auxiliary losses",
    "qort_no_spectral": "No spectral tokens",
    "qort_no_trend": "No trend/residual tokens",
    "qort_raw": "Chronological tokens (no tokenizer)",
    "qort_no_optimism": "No optimistic-conflict term",
    "qort_router_r0": "$r=0$ (absolute expert)",
    "qort_router_r05": "$r=0.5$ (equal mixing)",
    "qort_router_r1": "$r=1$ (relative expert)",
    "qort_gate_base_const": "Constant base gate $g_0=0.575$",
    "gru_time_prior_budget": "GRU + time/prior (budget-matched)",
    "film_time_prior_budget": "FiLM-inspired + time/prior (budget-matched)",
}


# ------------------------------------------------------------------ loading
def read(tag: str) -> pd.DataFrame:
    path = RESULTS / f"{tag}_results.csv"
    if not path.exists():
        print(f"[missing] {path.name}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["tag"] = tag
    return frame


def lengths() -> dict[tuple[str, str], int]:
    out = {}
    for dataset in list(ALL) + ["phm60"]:
        for sequence in load_sequences(dataset):
            out[(dataset, sequence.name)] = len(sequence.y)
    return out


LENGTHS = lengths()


def bearing_level(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "test_bearing", "model"]
    agg = frame.groupby(keys).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), late_mae=("late_mae", "mean"),
        violation=("monotonic_violation", "mean"),
        seed_sd=("mae", lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0),
        runs=("mae", "size"),
    ).reset_index()
    agg["set"] = agg.dataset.map(SET_OF)
    agg["T"] = [LENGTHS[(d, b)] for d, b in zip(agg.dataset, agg.test_bearing)]
    agg["minutes"] = [
        m * (t - 1) * SAMPLE_INTERVAL_SECONDS[d] / 60.0
        for m, t, d in zip(agg.mae, agg["T"], agg.dataset)
    ]
    return agg


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(BOOT, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def dataset_summary(bearings: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for (dataset, model), group in bearings.groupby(["dataset", "model"]):
        low, high = bootstrap_mean_ci(group.mae.values, rng)
        rows.append({
            "dataset": dataset, "model": model, "n_bearings": len(group),
            "mae": group.mae.mean(), "mae_sd": group.mae.std(ddof=1),
            "mae_ci_low": low, "mae_ci_high": high,
            "rmse": group.rmse.mean(), "rmse_sd": group.rmse.std(ddof=1),
            "late_mae": group.late_mae.mean(), "violation": group.violation.mean(),
            "minutes": group.minutes.mean(), "seed_sd": group.seed_sd.mean(),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- statistics
def sign_flip_p(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    observed = abs(diff.mean())
    n = len(diff)
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    null = np.abs((signs * diff).mean(axis=1))
    return float(np.mean(null >= observed - 1e-12))


def stratified_bootstrap(diff: np.ndarray, strata: np.ndarray, rng) -> tuple[float, float]:
    diff = np.asarray(diff, dtype=float)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    total = len(diff)
    means = np.empty(BOOT)
    for b in range(BOOT):
        picks = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in groups])
        means[b] = diff[picks].sum() / total
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(bearings: pd.DataFrame, a: str, b: str, datasets, scope: str, rng) -> dict:
    sub = bearings[bearings.dataset.isin(datasets)]
    left = sub[sub.model == a].set_index(["dataset", "test_bearing"]).mae
    right = sub[sub.model == b].set_index(["dataset", "test_bearing"]).mae
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna()
    if joined.empty:
        return {}
    diff = (joined.a - joined.b).values
    strata = np.array([d for d, _ in joined.index])
    n = len(diff)
    wins = int((diff < 0).sum())
    try:
        wil = stats.wilcoxon(diff, alternative="two-sided", method="exact").pvalue
        wil_one = stats.wilcoxon(diff, alternative="less", method="exact").pvalue
    except ValueError:
        wil, wil_one = float("nan"), float("nan")
    ranks = stats.rankdata(np.abs(diff))
    rank_biserial = float((ranks[diff < 0].sum() - ranks[diff > 0].sum()) / ranks.sum())
    low, high = stratified_bootstrap(diff, strata, rng)
    return {
        "scope": scope, "model": a, "comparator": b, "n": n, "wins": wins,
        "mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
        "rank_biserial": rank_biserial, "ci_low": low, "ci_high": high,
        "p_signflip": sign_flip_p(diff), "p_wilcoxon": float(wil),
        "p_wilcoxon_one_sided": float(wil_one),
        "min_p_wilcoxon": float(2.0 / 2 ** n),
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def mixed_model(runs: pd.DataFrame, a: str, b: str) -> dict:
    import statsmodels.formula.api as smf

    sub = runs[runs.dataset.isin(ALL)]
    left = sub[sub.model == a].set_index(["dataset", "test_bearing", "seed"]).mae
    right = sub[sub.model == b].set_index(["dataset", "test_bearing", "seed"]).mae
    joined = pd.concat([left.rename("a"), right.rename("b")], axis=1).dropna().reset_index()
    joined["diff"] = joined.a - joined.b
    joined["held_out"] = joined.dataset.isin(HELD).astype(float)
    joined["bearing"] = joined.dataset + ":" + joined.test_bearing
    fit = smf.mixedlm("diff ~ held_out", joined, groups=joined["bearing"]).fit(reml=True)
    ci = fit.conf_int()
    dev = fit.params["Intercept"]
    contrast = np.array([1.0, 1.0])
    cov = fit.cov_params().loc[["Intercept", "held_out"], ["Intercept", "held_out"]].values
    held = float(contrast @ fit.params[["Intercept", "held_out"]].values)
    held_se = float(np.sqrt(contrast @ cov @ contrast))
    return {
        "comparison": f"{a} - {b}", "observations": len(joined),
        "bearings": joined.bearing.nunique(),
        "development_effect": float(dev),
        "development_ci_low": float(ci.loc["Intercept", 0]),
        "development_ci_high": float(ci.loc["Intercept", 1]),
        "held_out_effect": held, "held_out_ci_low": held - 1.96 * held_se,
        "held_out_ci_high": held + 1.96 * held_se,
        "held_out_shift": float(fit.params["held_out"]),
        "held_out_shift_p": float(fit.pvalues["held_out"]),
        "bearing_variance": float(fit.cov_re.iloc[0, 0]),
        "residual_variance": float(fit.scale),
    }


# ---------------------------------------------------------------- formatting
def f3(value) -> str:
    return "--" if pd.isna(value) else f"{value:.3f}"


def signed(value) -> str:
    return "--" if pd.isna(value) else f"{value:+.3f}"


def pfmt(value) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.4f}" if value >= 0.0001 else "<0.0001"


def write(name: str, text: str) -> None:
    (TABLES / name).write_text(text, encoding="utf-8")
    print(f"[table] {name}")


def table_env(label, caption, colspec, header, rows, size="\\scriptsize") -> str:
    body = "\n".join(rows)
    colspec = colspec.replace("X", ">{\\raggedright\\arraybackslash}X")
    return (
        "\\begin{table}[H]\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n{size}\n"
        "\\renewcommand{\\arraystretch}{1.2}\n"
        f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}\n\\toprule\n{header}\\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabularx}}\n\\end{{table}}\n"
    )


# ------------------------------------------------------------- predictions
def prediction_path(dataset, bearing, model, seed):
    """Locate the archived per-snapshot prediction file of one run."""
    candidates = {
        "qort_net": RESULTS / "main_qort_predictions" / f"{dataset}_{bearing}_qort_net_seed{seed}.npz",
        "qort_prior_only": RESULTS / "main_prior_only_predictions" / f"{dataset}_{bearing}_qort_prior_only_seed{seed}.npz",
        "qort_vibration_only": RESULTS / "abl_vibration_only_predictions" / f"{dataset}_{bearing}_qort_vibration_only_seed{seed}.npz",
        "gru_time_prior": RESULTS / "main_equated_predictions" / f"{dataset}_{bearing}_gru_time_prior_s{seed}.npz",
        "film_time_prior": RESULTS / "main_equated_predictions" / f"{dataset}_{bearing}_film_time_prior_s{seed}.npz",
        "patchtst_time_prior": RESULTS / "main_equated_predictions" / f"{dataset}_{bearing}_patchtst_time_prior_s{seed}.npz",
        "median_lifetime_clock": RESULTS / "trivial_predictions" / f"{dataset}_{bearing}_median_lifetime_clock.npz",
    }
    return candidates[model]


def load_prediction(dataset, bearing, model, seed):
    path = prediction_path(dataset, bearing, model, seed)
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def mean_trajectory(dataset, bearing, model, seeds=(7, 21, 42)):
    if model == "median_lifetime_clock":
        data = load_prediction(dataset, bearing, model, 0)
        return data["target"], data["prediction"]
    arrays = [load_prediction(dataset, bearing, model, s) for s in seeds]
    return arrays[0]["target"], np.mean([a["prediction"] for a in arrays], axis=0)


STAGES = (("early", 0.0, 0.3), ("middle", 0.3, 0.7), ("late", 0.7, 1.0001))


def stage_errors(target, prediction):
    n = len(target)
    position = np.arange(n) / max(n - 1, 1)
    out = {}
    for name, low, high in STAGES:
        mask = (position >= low) & (position < high)
        out[name] = float(np.mean(np.abs(prediction[mask] - target[mask]))) if mask.any() else np.nan
    return out


# =================================================================== main
def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    tags = {
        "main": ["trivial", "main_qort", "main_prior_only", "main_equated", "main_vibration"],
        "ablation": ["abl_single_prior", "abl_static_gate", "abl_no_relative_view",
                     "abl_vibration_only", "abl_no_aux", "abl_no_spectral", "abl_no_trend",
                     "abl_raw", "abl_no_optimism", "router_r0", "router_r05",
                     "router_r1", "gate_base_const"],
        "budget": ["budget_equated"],
        "interval": ["phm60_qort", "phm60_prior_only", "phm60_equated"],
        "window": ["window_20_10", "window_80_40"],
    }
    runs = pd.concat([read(t) for group in tags.values() for t in group], ignore_index=True)
    runs = runs[runs.get("dropped_train_bearing", pd.Series("", index=runs.index)).fillna("") == ""]
    runs.to_csv(SOURCE / "all_run_records.csv", index=False)
    bearings = bearing_level(runs)
    bearings.to_csv(SOURCE / "bearing_level_metrics.csv", index=False)
    summary = dataset_summary(bearings)
    summary.to_csv(SOURCE / "dataset_summary.csv", index=False)

    # ---------------------------------------------------- paired inference
    comparators = ["gru_time_prior", "film_time_prior", "patchtst_time_prior",
                   "qort_prior_only", "median_lifetime_clock"]
    rows = []
    for scope, datasets in (("held-out", HELD), ("xjtu_c1", ("xjtu_c1",)), ("phm_c2", ("phm_c2",)),
                            ("development", DEV), ("xjtu", ("xjtu",)), ("phm", ("phm",)),
                            ("all", ALL)):
        for comparator in comparators:
            result = paired(bearings, "qort_net", comparator, datasets, scope, rng)
            if result:
                rows.append(result)
    pairs = pd.DataFrame(rows)
    for scope in ("held-out", "development", "all"):
        mask = pairs.scope == scope
        if mask.sum():
            pairs.loc[mask, "p_signflip_holm"] = holm(pairs.loc[mask, "p_signflip"].tolist())
            pairs.loc[mask, "p_wilcoxon_holm"] = holm(pairs.loc[mask, "p_wilcoxon"].tolist())
    pairs.to_csv(SOURCE / "paired_inference.csv", index=False)

    mixed = []
    for comparator in ("gru_time_prior", "film_time_prior"):
        try:
            mixed.append(mixed_model(runs, "qort_net", comparator))
        except Exception as error:  # pragma: no cover - reported, not hidden
            print("mixed model failed", comparator, error)
    pd.DataFrame(mixed).to_csv(SOURCE / "mixed_effects.csv", index=False)

    # ------------------------------------------------------- main tables
    main_models = ["qort_net", "qort_prior_only", "median_lifetime_clock",
                   "gru_time_prior", "film_time_prior", "patchtst_time_prior",
                   "ridge_linear_vibration", "constant_05"]
    vib_models = ["gru", "film", "filternet", "patchtst", "pca_drsn_tcn", "mixlinear", "vpnet"]

    def cell(dataset, model, field="mae"):
        row = summary[(summary.dataset == dataset) & (summary.model == model)]
        if row.empty:
            return "--"
        r = row.iloc[0]
        return f"${r[field]:.3f}\\pm{r[field + '_sd']:.3f}$"

    def ci(dataset, model):
        row = summary[(summary.dataset == dataset) & (summary.model == model)]
        if row.empty:
            return "--"
        r = row.iloc[0]
        return f"$[{r.mae_ci_low:.3f}, {r.mae_ci_high:.3f}]$"

    for set_name, datasets, label, caption in (
        ("development", DEV, "tab:main_dev",
         "Development subsets (exploratory). Bearing-level MAE after averaging three seeds within each bearing; mean $\\pm$ between-bearing SD with 95\\% bearing bootstrap CI. All models were retrained with the frozen release."),
        ("held-out", HELD, "tab:main_heldout",
         "Held-out operating conditions, evaluated once with the frozen release and pre-specified analysis. Entries as in table~\\ref{tab:main_dev}."),
    ):
        header = ("Model & " + " & ".join(
            f"{DATASET_LABEL[d]} MAE & 95\\% CI" for d in datasets))
        rows_tex = []
        for model in main_models:
            rows_tex.append(LABEL[model] + " & " + " & ".join(
                f"{cell(d, model)} & {ci(d, model)}" for d in datasets) + "\\\\")
        write(f"main_{set_name.replace('-', '')}.tex",
              table_env(label, caption, "X c c c c", header, rows_tex))

    # complete comparator table: MAE/RMSE for all four conditions
    header = "Model & " + " & ".join(DATASET_LABEL[d] for d in ALL)
    rows_tex = []
    for block in (vib_models, ["qort_net", "gru_time_prior", "film_time_prior", "patchtst_time_prior"],
                  ["qort_prior_only", "median_lifetime_clock", "ridge_linear_vibration", "constant_05"]):
        for model in block:
            vals = []
            for d in ALL:
                row = summary[(summary.dataset == d) & (summary.model == model)]
                vals.append("--" if row.empty else f"{row.iloc[0].mae:.3f}/{row.iloc[0].rmse:.3f}")
            rows_tex.append(LABEL[model] + " & " + " & ".join(vals) + "\\\\")
        rows_tex.append("\\addlinespace")
    rows_tex = rows_tex[:-1]
    write("complete_errors.tex", table_env(
        "tab:completeerrors",
        "Complete comparator results as MAE/RMSE (bearing means after averaging seeds within bearing). The first seven rows use vibration only; time/prior rows receive the five causal age, prior and lifetime-CV channels used by QORT-Net. RMSE is averaged over runs.",
        "X c c c c", header, rows_tex))

    # paired inference table
    def prow(scope_label, scope, comparator):
        r = pairs[(pairs.scope == scope) & (pairs.comparator == comparator)]
        if r.empty:
            return None
        r = r.iloc[0]
        holm_text = f"{pfmt(r.p_signflip_holm)}" if not pd.isna(r.get("p_signflip_holm", np.nan)) else "--"
        return (f"{scope_label} & {LABEL[comparator]} & {int(r.n)} & {int(r.wins)}/{int(r.n)} & "
                f"${r.mean_diff:+.3f}$ & ${r.median_diff:+.3f}$ & $[{r.ci_low:+.3f}, {r.ci_high:+.3f}]$ & "
                f"{pfmt(r.p_signflip)} & {pfmt(r.p_wilcoxon)} & {holm_text}\\\\")

    rows_tex = []
    for scope_label, scope in (("Held-out", "held-out"), ("Development", "development"), ("All 24", "all")):
        for comparator in comparators:
            row = prow(scope_label, scope, comparator)
            if row:
                rows_tex.append(row)
        rows_tex.append("\\addlinespace")
    write("paired.tex", table_env(
        "tab:paired",
        "Bearing-paired comparisons of QORT-Net with each comparator (difference = QORT-Net minus comparator; negative favours QORT-Net). CIs use 20\\,000 bootstrap resamples stratified by condition. $p_{\\mathrm{perm}}$: exact two-sided sign-flip permutation test of the mean; $p_{\\mathrm{W}}$: exact two-sided Wilcoxon test; Holm adjustment is applied to $p_{\\mathrm{perm}}$ within each scope. The held-out comparison with GRU + time/prior is the pre-specified primary test.",
        "l X c c c c c c c c",
        "Scope & Comparator & $n$ & Wins & Mean $\\Delta$ & Median $\\Delta$ & 95\\% CI (mean) & $p_{\\mathrm{perm}}$ & $p_{\\mathrm{W}}$ & Holm",
        rows_tex[:-1]))

    # ------------------------------------------------------ ablations
    ablations = ["qort_net", "qort_no_spectral", "qort_no_trend", "qort_raw", "qort_single_prior",
                 "qort_static_gate", "qort_no_optimism", "qort_no_relative_view",
                 "qort_vibration_only", "qort_no_aux", "qort_prior_only"]
    routers = ["qort_net", "qort_router_r0", "qort_router_r05", "qort_router_r1", "qort_gate_base_const"]
    ab_rows = []
    for model in ablations + routers[1:]:
        for d in ALL:
            r = summary[(summary.dataset == d) & (summary.model == model)]
            if r.empty:
                continue
            pr = paired(bearings, model, "qort_net", (d,), d, rng) if model != "qort_net" else {}
            ab_rows.append({"model": model, "dataset": d, "mae": r.iloc[0].mae, "rmse": r.iloc[0].rmse,
                            "seed_sd": r.iloc[0].seed_sd,
                            "diff_vs_full": pr.get("mean_diff", 0.0),
                            "worse_bearings": (pr.get("n", 0) - pr.get("wins", 0)) if pr else 0,
                            "n": pr.get("n", r.iloc[0].n_bearings)})
    ab = pd.DataFrame(ab_rows)
    ab.to_csv(SOURCE / "ablation_three_seed.csv", index=False)

    def ab_table(models, name, label, caption):
        header = "Variant & " + " & ".join(DATASET_LABEL[d] for d in ALL) + " & Pooled $\\Delta$"
        rows_tex = []
        for model in models:
            cells = []
            for d in ALL:
                r = ab[(ab.model == model) & (ab.dataset == d)]
                if r.empty:
                    cells.append("--")
                elif model == "qort_net":
                    cells.append(f"{r.iloc[0].mae:.3f}")
                else:
                    cells.append(f"{r.iloc[0].mae:.3f} (${r.iloc[0].diff_vs_full:+.3f}$)")
            if model == "qort_net":
                pooled = "--"
            else:
                pr = paired(bearings, model, "qort_net", ALL, "all", rng)
                pooled = f"${pr['mean_diff']:+.3f}$ ({pr['n'] - pr['wins']}/{pr['n']})" if pr else "--"
            rows_tex.append(f"{LABEL[model]} & " + " & ".join(cells) + f" & {pooled}\\\\")
        write(name, table_env(label, caption, "X c c c c c", header, rows_tex))

    ab_table(ablations, "ablation.tex", "tab:ablation",
             "Three-seed component ablations on all four conditions. Entries are bearing-mean MAE with the paired difference from the full model in parentheses (positive = removal increases error). The last column pools the 24 bearings: mean paired difference and, in parentheses, the number of bearings on which the variant has a higher MAE than the full model.")
    ab_table(routers, "router_variants.tex", "tab:routervariants",
             "Expert-weight and base-gate variants (three seeds). Fixed $r$ replaces equation~(\\ref{eq:router}); with the coupled base gate, $g_0=0.20+0.75r$ changes with $r$. The constant-base variant keeps the CV router but fixes $g_0=0.575$. Entries as in table~\\ref{tab:ablation}.")

    # router fold values for all conditions
    fold_rows = []
    qort_runs = runs[runs.model == "qort_net"]
    for (d, b), g in qort_runs.groupby(["dataset", "test_bearing"]):
        first = g.iloc[0]
        fold_rows.append({"dataset": d, "test_bearing": b, "validation_bearing": first.validation_bearing,
                          "train_bearings": first.train_bearings, "lifetime_cv": first.lifetime_cv,
                          "regime_weight": first.regime_weight,
                          "reference_lives": first.reference_lives, "T": LENGTHS[(d, b)]})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(SOURCE / "fold_router_values.csv", index=False)

    # -------------------------------- life-stage and atypical-lifetime analysis
    stage_rows, atypical_rows = [], []
    stage_models = ["qort_net", "qort_prior_only", "qort_vibration_only", "median_lifetime_clock",
                    "gru_time_prior", "film_time_prior"]
    for d in ALL:
        for _, fold in folds[folds.dataset == d].iterrows():
            b = fold.test_bearing
            lives = json.loads(fold.reference_lives)
            deviation = (fold["T"] - lives[1]) / lives[1]
            for model in stage_models:
                try:
                    target, prediction = mean_trajectory(d, b, model)
                except FileNotFoundError:
                    continue
                errors = stage_errors(target, prediction)
                stage_rows.append({"dataset": d, "test_bearing": b, "model": model, **errors,
                                   "lifetime_deviation": deviation})
            # vibration correction relative to the selected prior
            seeds = [load_prediction(d, b, "qort_net", s) for s in (7, 21, 42)]
            target = seeds[0]["target"]
            correction = np.mean([np.abs(s["prediction"] - s["mixed_prior"]) for s in seeds], axis=0)
            gate = np.mean([s["gate"] for s in seeds], axis=0)
            corr_stage = stage_errors(np.zeros_like(correction), correction)
            gate_stage = stage_errors(np.zeros_like(gate), gate)
            atypical_rows.append({
                "dataset": d, "test_bearing": b, "T": fold["T"], "L50": lives[1],
                "lifetime_deviation": deviation, "abs_deviation": abs(deviation),
                **{f"correction_{k}": v for k, v in corr_stage.items()},
                **{f"gate_{k}": v for k, v in gate_stage.items()},
            })
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(SOURCE / "life_stage_errors.csv", index=False)
    atyp = pd.DataFrame(atypical_rows)
    wide = bearings.pivot_table(index=["dataset", "test_bearing"], columns="model", values="mae").reset_index()
    atyp = atyp.merge(wide[["dataset", "test_bearing", "qort_net", "qort_prior_only",
                            "median_lifetime_clock", "gru_time_prior", "film_time_prior"]
                           + [c for c in ("qort_single_prior", "qort_static_gate") if c in wide]],
                      on=["dataset", "test_bearing"], how="left")
    atyp["gain_vs_clock"] = atyp.median_lifetime_clock - atyp.qort_net
    atyp["gain_vs_prior_only"] = atyp.qort_prior_only - atyp.qort_net
    if "qort_single_prior" in atyp:
        atyp["single_minus_full"] = atyp.qort_single_prior - atyp.qort_net
    if "qort_static_gate" in atyp:
        atyp["static_minus_full"] = atyp.qort_static_gate - atyp.qort_net
    atyp.to_csv(SOURCE / "atypical_lifetime_analysis.csv", index=False)
    corr = {}
    for column in ("gain_vs_clock", "gain_vs_prior_only", "single_minus_full", "static_minus_full"):
        if column in atyp:
            rho, p = stats.spearmanr(atyp.abs_deviation, atyp[column])
            corr[column] = {"spearman_rho": float(rho), "p": float(p), "n": int(len(atyp))}
    corr["error_vs_abs_deviation"] = {}
    for column in ("qort_net", "median_lifetime_clock", "qort_prior_only", "gru_time_prior", "film_time_prior"):
        rho, p = stats.spearmanr(atyp.abs_deviation, atyp[column])
        corr["error_vs_abs_deviation"][column] = {"spearman_rho": float(rho), "p": float(p)}
    atypical_mask = atyp.abs_deviation > 0.25
    corr["groups"] = {
        "threshold": 0.25,
        "atypical_n": int(atypical_mask.sum()), "typical_n": int((~atypical_mask).sum()),
        "atypical_gain_vs_clock": float(atyp.loc[atypical_mask, "gain_vs_clock"].mean()),
        "typical_gain_vs_clock": float(atyp.loc[~atypical_mask, "gain_vs_clock"].mean()),
        "atypical_gain_vs_prior_only": float(atyp.loc[atypical_mask, "gain_vs_prior_only"].mean()),
        "typical_gain_vs_prior_only": float(atyp.loc[~atypical_mask, "gain_vs_prior_only"].mean()),
    }
    (SOURCE / "atypical_correlations.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")

    # life-stage table: pooled over the 24 bearings, bearing-equal weighting
    header = "Model & Early (0--30\\%) & Middle (30--70\\%) & Late (70--100\\%)"
    rows_tex = []
    for model in stage_models:
        g = stages[stages.model == model]
        if g.empty:
            continue
        vals = [f"{g[s].mean():.3f}" for s, _, _ in STAGES]
        rows_tex.append(f"{LABEL[model]} & " + " & ".join(vals) + "\\\\")
    write("life_stage.tex", table_env(
        "tab:lifestage",
        "Life-stage MAE pooled over all 24 bearings (bearing-equal weighting, three-seed mean trajectories). Stages are fractions of each bearing's recorded life.",
        "X c c c", header, rows_tex))

    # ------------------------------------------ FiLM versus QORT diagnosis
    film_rows = []
    for d in ALL:
        for b in DATASET_BEARINGS[d]:
            try:
                target, clock = mean_trajectory(d, b, "median_lifetime_clock")
            except FileNotFoundError:
                continue
            row = {"dataset": d, "test_bearing": b}
            for model in ("qort_net", "film_time_prior", "gru_time_prior"):
                _, prediction = mean_trajectory(d, b, model)
                row[f"{model}_mae"] = float(np.mean(np.abs(prediction - target)))
                row[f"{model}_dist_clock"] = float(np.mean(np.abs(prediction - clock)))
                row[f"{model}_tv"] = float(np.mean(np.abs(np.diff(prediction))))
            row["clock_mae"] = float(np.mean(np.abs(clock - target)))
            film_rows.append(row)
    film = pd.DataFrame(film_rows)
    film.to_csv(SOURCE / "film_versus_qort_diagnosis.csv", index=False)

    # ------------------------------------------------ budget and grid study
    grid_frames = [read(f"grid_qort_lr{lr}_do{do}") for lr in ("0.001", "0.0005") for do in ("0.10", "0.20")
                   if not (lr == "0.001" and do == "0.10")]
    grid_frames += [read(f"grid_equated_lr{lr}_do{do}") for lr in ("0.001", "0.0005") for do in ("0.15", "0.25")
                    if not (lr == "0.001" and do == "0.15")]
    grid = pd.concat([f for f in grid_frames if not f.empty] + [
        runs[(runs.dataset.isin(DEV)) & runs.model.isin(["qort_net", "gru_time_prior", "film_time_prior"])]
    ], ignore_index=True)
    grid["family"] = grid.model.str.replace(r"_lr.*$", "", regex=True)
    selected = (grid.sort_values("best_val_mae")
                .groupby(["family", "dataset", "test_bearing", "seed"]).head(1))
    selected.to_csv(SOURCE / "grid_selected_runs.csv", index=False)
    sel_bearing = selected.groupby(["family", "dataset", "test_bearing"]).mae.mean().reset_index()
    budget_rows = []
    for family, label in (("qort_net", "QORT-Net"), ("gru_time_prior", "GRU + time/prior"),
                          ("film_time_prior", "FiLM-inspired + time/prior")):
        row = {"model": label}
        for d in DEV:
            default = summary[(summary.dataset == d) & (summary.model == family)]
            row[f"{d}_default"] = default.iloc[0].mae if not default.empty else np.nan
            budget = summary[(summary.dataset == d) & (summary.model == f"{family}_budget")]
            row[f"{d}_budget"] = budget.iloc[0].mae if not budget.empty else np.nan
            g = sel_bearing[(sel_bearing.family == family) & (sel_bearing.dataset == d)]
            row[f"{d}_grid"] = g.mae.mean() if not g.empty else np.nan
        for d in HELD:
            default = summary[(summary.dataset == d) & (summary.model == family)]
            row[f"{d}_default"] = default.iloc[0].mae if not default.empty else np.nan
            budget = summary[(summary.dataset == d) & (summary.model == f"{family}_budget")]
            row[f"{d}_budget"] = budget.iloc[0].mae if not budget.empty else np.nan
        budget_rows.append(row)
    budget = pd.DataFrame(budget_rows)
    budget.to_csv(SOURCE / "budget_and_grid.csv", index=False)
    header = ("Model & \\multicolumn{3}{c}{XJTU-SY C2} & \\multicolumn{3}{c}{PHM2012 C1} & "
              "\\multicolumn{2}{c}{XJTU-SY C1} & \\multicolumn{2}{c}{PHM2012 C2}\\\\\n"
              " & Def. & Budget & Grid & Def. & Budget & Grid & Def. & Budget & Def. & Budget")
    rows_tex = []
    for _, r in budget.iterrows():
        rows_tex.append(" & ".join([r.model] + [f3(r[f"{d}_{k}"]) for d in DEV for k in ("default", "budget", "grid")]
                                   + [f3(r[f"{d}_{k}"]) for d in HELD for k in ("default", "budget")]) + "\\\\")
    write("budget.tex", table_env(
        "tab:budget",
        "Effect of optimization budget. Def.: default configuration. Budget: baselines given QORT-Net's 40-snapshot context, epoch limit (120), patience (18), plateau patience (5) and terminal over-estimation loss (QORT-Net is unchanged). Grid: for each fold and seed, the configuration with the lowest validation MAE among four learning-rate/dropout settings is selected, using the same grid size for every model.",
        "l c c c c c c c c c c", header, rows_tex, size="\\tiny"))

    # --------------------------------------- window and interval sensitivity
    sens_rows = []
    for model, label in (("qort_w20_10", "QORT-Net, windows 20/10"), ("qort_net", "QORT-Net, windows 40/20"),
                         ("qort_w80_40", "QORT-Net, windows 80/40")):
        row = {"setting": label}
        for d in DEV:
            r = summary[(summary.dataset == d) & (summary.model == model)]
            row[d] = r.iloc[0].mae if not r.empty else np.nan
        sens_rows.append(row)
    pd.DataFrame(sens_rows).to_csv(SOURCE / "window_sensitivity.csv", index=False)
    interval_rows = []
    for model in ("qort_net", "qort_prior_only", "median_lifetime_clock", "gru_time_prior", "film_time_prior"):
        row = {"model": LABEL[model]}
        for d in ("phm", "phm60"):
            r = summary[(summary.dataset == d) & (summary.model == model)]
            row[d] = r.iloc[0].mae if not r.empty else np.nan
            row[f"{d}_minutes"] = r.iloc[0].minutes if not r.empty else np.nan
        interval_rows.append(row)
    pd.DataFrame(interval_rows).to_csv(SOURCE / "interval_sensitivity.csv", index=False)

    # ----------------------------------------------- validation-test audit
    audit = []
    for (d, model), g in runs[runs.dataset.isin(ALL)].groupby(["dataset", "model"]):
        if g.best_val_mae.notna().sum() >= 6 and g.mae.nunique() > 1:
            rho, _ = stats.spearmanr(g.best_val_mae, g.mae)
            audit.append({"dataset": d, "model": model, "n": len(g), "spearman": rho})
    pd.DataFrame(audit).to_csv(SOURCE / "validation_test_audit.csv", index=False)

    # ------------------------------------------------------- cost table
    cost = (runs[runs.dataset.isin(ALL)]
            .groupby(["model", "dataset"]).agg(parameters=("parameters", "median"),
                                               seconds=("train_seconds", "mean"),
                                               epochs=("epochs_run", "mean")).reset_index())
    cost.to_csv(SOURCE / "computational_cost.csv", index=False)

    # ------------------------------------------------ per-bearing table
    header = ("Condition & Bearing & $T_b$ & $d_b$ & QORT-Net & Prior-only & Clock & GRU+TP & FiLM+TP & "
              "QORT (min)")
    rows_tex = []
    for d in ALL:
        for _, fold in folds[folds.dataset == d].sort_values("test_bearing").iterrows():
            b = fold.test_bearing
            lives = json.loads(fold.reference_lives)
            sel = bearings[(bearings.dataset == d) & (bearings.test_bearing == b)].set_index("model")
            vals = [sel.mae.get(m, np.nan) for m in ("qort_net", "qort_prior_only", "median_lifetime_clock",
                                                     "gru_time_prior", "film_time_prior")]
            best = np.nanmin(vals)
            cells = [f"\\textbf{{{v:.3f}}}" if abs(v - best) < 5e-4 else f"{v:.3f}" for v in vals]
            rows_tex.append(f"{DATASET_LABEL[d]} & {b.replace('_', chr(92) + '_')} & {fold['T']} & "
                            f"${(fold['T'] - lives[1]) / lives[1]:+.2f}$ & " + " & ".join(cells)
                            + f" & {sel.minutes.get('qort_net', np.nan):.1f}\\\\")
        rows_tex.append("\\addlinespace")
    write("bearing_errors.tex", table_env(
        "tab:bearingerrors",
        "Bearing-level MAE (three-seed means) for all 24 test bearings of the four conditions. $T_b$: recorded snapshots; $d_b=(T_b-L_{50})/L_{50}$: deviation of the test lifetime from the training-fold median life. The lowest MAE in each row is bold. The last column is the retrospective minute conversion of QORT-Net's error (equation~(\\ref{eq:minutes})).",
        "l X c c c c c c c c", header, rows_tex[:-1], size="\\tiny"))

    # -------------------------------------- physical time and late-life
    header = "Model & " + " & ".join(DATASET_LABEL[d] for d in ALL)
    rows_tex = []
    for field, title in (("minutes", "Physical MAE (min)"), ("late_mae", "Final-30\\% MAE"),
                         ("violation", "Monotonic violation rate")):
        rows_tex.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for model in ["qort_net", "qort_prior_only", "median_lifetime_clock", "gru_time_prior",
                      "film_time_prior", "patchtst_time_prior"]:
            vals = []
            for d in ALL:
                r = summary[(summary.dataset == d) & (summary.model == model)]
                vals.append("--" if r.empty else (f"{r.iloc[0][field]:.1f}" if field == "minutes"
                                                  else f"{r.iloc[0][field]:.3f}"))
            rows_tex.append(f"\\quad {LABEL[model]} & " + " & ".join(vals) + "\\\\")
    write("latelife.tex", table_env(
        "tab:latelife",
        "Retrospective physical-time error, late-life error and monotonic violation rate (bearing means). Minutes use each completed test bearing's recorded duration and are not available online.",
        "X c c c c", header, rows_tex))

    # ------------------------------------------- window / interval tables
    header = "Setting & " + " & ".join(f"{DATASET_LABEL[d]}" for d in DEV)
    rows_tex = [f"{r['setting']} & " + " & ".join(f3(r[d]) for d in DEV) + "\\\\" for r in sens_rows]
    header2 = "Model & 10 s MAE & 60 s MAE & 10 s (min) & 60 s (min)"
    rows_tex2 = [f"{r['model']} & {f3(r['phm'])} & {f3(r['phm60'])} & "
                 f"{r['phm_minutes']:.1f} & {r['phm60_minutes']:.1f}\\\\" for r in interval_rows
                 if not pd.isna(r["phm"])]
    write("window_interval.tex",
          table_env("tab:window", "QORT-Net window-length sensitivity on the development subsets (three seeds; long/short windows in snapshots). The 40/20 setting is the frozen configuration.",
                    "X c c", header, rows_tex)
          + table_env("tab:interval", "Acquisition-interval sensitivity on PHM2012 C1. The 60 s variant keeps every sixth snapshot, anchored at the final record, so that windows of 40/20 snapshots span the same 40/20 min as on XJTU-SY; models are retrained with the frozen settings.",
                      "X c c c c", header2, rows_tex2))

    # --------------------------------------------------------- cost table
    header = "Model & Parameters & " + " & ".join(f"{DATASET_LABEL[d]} (s)" for d in ALL)
    rows_tex = []
    for model in ["qort_net", "qort_prior_only", "gru_time_prior", "film_time_prior", "patchtst_time_prior",
                  "gru", "film", "filternet", "patchtst", "pca_drsn_tcn", "mixlinear", "vpnet"]:
        g = cost[cost.model == model].set_index("dataset")
        if g.empty:
            continue
        params = int(g.parameters.median())
        rows_tex.append(f"{LABEL[model]} & {params:,}".replace(",", "\\,") + " & "
                        + " & ".join(f"{g.seconds.get(d, np.nan):.1f}" for d in ALL) + "\\\\")
    write("recorded_cost.tex", table_env(
        "tab:recordedcost",
        "Parameter count and mean elapsed fitting time per fold-seed run (five experiment streams shared one RTX 5060 Ti GPU, so times are indicative only). Parameter counts of the inspired baselines depend on the input width and are given for the development runs.",
        "X c c c c c", header, rows_tex))

    print("\nPrimary held-out comparison:")
    print(pairs[(pairs.scope == "held-out")][["comparator", "n", "wins", "mean_diff", "ci_low",
                                              "ci_high", "p_signflip", "p_wilcoxon"]].round(4))


if __name__ == "__main__":
    main()

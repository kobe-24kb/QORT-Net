"""Data figures for the resubmitted manuscript (frozen-release results only)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from analyze_resubmission import (
    ALL, DATASET_LABEL, DEV, HELD, LABEL, LENGTHS, load_prediction, mean_trajectory,
)
from protocol import DATASET_BEARINGS, SAMPLE_INTERVAL_SECONDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "source_data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "qort_net": "#6F3C97",
    "qort_prior_only": "#B89CC9",
    "median_lifetime_clock": "#7A7F87",
    "constant_05": "#B7BBC1",
    "ridge_linear_vibration": "#9A6B4F",
    "gru_time_prior": "#3F77B5",
    "film_time_prior": "#E07A5F",
    "patchtst_time_prior": "#2A9D8F",
}
SET_COLOR = {"development": "#6F3C97", "held-out": "#E07A5F"}
CODE = {"xjtu": "XJTU-C2", "phm": "PHM-C1", "xjtu_c1": "XJTU-C1", "phm_c2": "PHM-C2"}
SHORT = {
    "qort_net": "QORT-Net", "qort_prior_only": "Prior-only", "median_lifetime_clock": "Clock",
    "gru_time_prior": "GRU+TP", "film_time_prior": "FiLM+TP", "patchtst_time_prior": "PatchTST+TP",
    "ridge_linear_vibration": "Ridge", "constant_05": "Const.",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
})


def export(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("[figure]", stem)


def panel(ax, label: str, x=-0.14) -> None:
    ax.text(x, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=9)


# ------------------------------------------------------------------ figure 5
def protocol_figure() -> None:
    folds = pd.read_csv(SOURCE / "fold_router_values.csv")
    fig = plt.figure(figsize=(7.2, 2.9), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0))
    ax = fig.add_subplot(grid[0])
    x, ticks, labels = 0, [], []
    for d in ALL:
        hours = [LENGTHS[(d, b)] * SAMPLE_INTERVAL_SECONDS[d] / 3600 for b in DATASET_BEARINGS[d]]
        positions = np.arange(x, x + len(hours))
        color = SET_COLOR["development" if d in DEV else "held-out"]
        ax.bar(positions, hours, color=color, alpha=0.85 if d in DEV else 0.75, width=0.75)
        ticks.extend(positions)
        labels.extend(b.replace("Bearing", "") for b in DATASET_BEARINGS[d])
        ax.text(positions.mean(), max(hours) + 0.4, DATASET_LABEL[d], ha="center", fontsize=6.8)
        x += len(hours) + 1
    ax.set_xticks(ticks, labels, rotation=90, fontsize=6)
    ax.set_ylabel("Recorded test duration (h)")
    ax.set_title("Run-to-failure units: development (purple) and held-out (orange)")
    panel(ax, "(a)", -0.08)

    ax = fig.add_subplot(grid[1])
    cv = np.linspace(0.0, 1.0, 400)
    ax.plot(cv, 1 / (1 + np.exp(-(cv - 0.390) / 0.012)), color="0.35", lw=1.0,
            label=r"$r=\sigma\{(\mathrm{CV}-0.390)/0.012\}$")
    markers = {"xjtu": "o", "phm": "s", "xjtu_c1": "^", "phm_c2": "D"}
    for d in ALL:
        g = folds[folds.dataset == d]
        ax.scatter(g.lifetime_cv, g.regime_weight, s=18, marker=markers[d],
                   color=SET_COLOR["development" if d in DEV else "held-out"],
                   edgecolor="white", linewidth=0.4, label=DATASET_LABEL[d], zorder=3)
    ax.set_xlabel(r"Training-fold lifetime CV")
    ax.set_ylabel(r"Expert weight $r$")
    ax.set_title("Fold-level expert weights (all 24 folds)")
    ax.legend(loc="center right", fontsize=6)
    panel(ax, "(b)")
    export(fig, "Fig5_units_and_router")


# ------------------------------------------------------------------ figure 6
def comparison_figure() -> None:
    summary = pd.read_csv(SOURCE / "dataset_summary.csv")
    pairs = pd.read_csv(SOURCE / "bearing_level_metrics.csv")
    models = ["qort_net", "qort_prior_only", "median_lifetime_clock",
              "gru_time_prior", "film_time_prior", "patchtst_time_prior"]
    fig = plt.figure(figsize=(7.2, 4.9), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 0.95))
    for i, d in enumerate(ALL):
        ax = fig.add_subplot(grid[0, i])
        sub = summary[(summary.dataset == d) & summary.model.isin(models)].set_index("model").loc[models]
        y = np.arange(len(models))[::-1]
        ax.errorbar(sub.mae, y, xerr=[sub.mae - sub.mae_ci_low, sub.mae_ci_high - sub.mae],
                    fmt="none", ecolor="0.6", elinewidth=0.8, capsize=1.5)
        ax.scatter(sub.mae, y, c=[COLORS[m] for m in models], s=22, zorder=3)
        ax.set_yticks(y, [SHORT[m] for m in models] if i == 0 else [""] * len(models))
        ax.set_xlabel("Bearing-level MAE")
        ax.set_title(DATASET_LABEL[d] + (" (dev.)" if d in DEV else " (held-out)"), fontsize=7.8)
        ax.set_xlim(left=0)
        panel(ax, f"({'abcd'[i]})", -0.16 if i else -0.55)
    ax = fig.add_subplot(grid[1, :])
    wide = pairs.pivot_table(index=["dataset", "test_bearing"], columns="model", values="mae").reset_index()
    wide = wide[wide.dataset.isin(ALL)]
    wide["diff"] = wide.qort_net - wide.gru_time_prior
    wide["order"] = wide.dataset.map({d: k for k, d in enumerate(ALL)})
    wide = wide.sort_values(["order", "test_bearing"]).reset_index(drop=True)
    colors = [SET_COLOR["development" if d in DEV else "held-out"] for d in wide.dataset]
    ax.bar(np.arange(len(wide)), wide["diff"], color=colors, width=0.7)
    ax.axhline(0, color="0.2", lw=0.6)
    ax.set_xticks(np.arange(len(wide)),
                  [f"{CODE[d]} {b.replace('Bearing', '')}"
                   for d, b in zip(wide.dataset, wide.test_bearing)], rotation=90, fontsize=6)
    ax.set_ylabel(r"MAE$_{\mathrm{QORT}}$ $-$ MAE$_{\mathrm{GRU+TP}}$")
    ax.set_title("Paired bearing-level differences (negative favours QORT-Net); purple: development, orange: held-out")
    panel(ax, "(e)", -0.06)
    export(fig, "Fig8_information_equated")


# ----------------------------------------------------------- trajectories
def trajectory_figure(datasets, stem, ncols) -> None:
    items = [(d, b) for d in datasets for b in DATASET_BEARINGS[d]]
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 1.75 * nrows + 0.35),
                             constrained_layout=True, squeeze=False)
    models = ["gru_time_prior", "film_time_prior", "patchtst_time_prior"]
    for ax, (d, b) in zip(axes.flat, items):
        target, clock = mean_trajectory(d, b, "median_lifetime_clock")
        t = np.arange(len(target)) * SAMPLE_INTERVAL_SECONDS[d] / 3600
        ax.plot(t, target, color="black", lw=1.0, label="Target")
        ax.plot(t, clock, color=COLORS["median_lifetime_clock"], lw=0.9, ls="--", label="Median-life clock")
        for model in models:
            _, pred = mean_trajectory(d, b, model)
            ax.plot(t, pred, color=COLORS[model], lw=0.7, alpha=0.85, label=SHORT[model])
        seeds = np.array([load_prediction(d, b, "qort_net", s)["prediction"] for s in (7, 21, 42)])
        ax.fill_between(t, seeds.min(0), seeds.max(0), color=COLORS["qort_net"], alpha=0.18, lw=0)
        ax.plot(t, seeds.mean(0), color=COLORS["qort_net"], lw=1.0, label="QORT-Net")
        ax.set_title(f"{DATASET_LABEL[d]} {b}", fontsize=7.4)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("Time (h)", fontsize=6.8)
        ax.set_ylabel("Normalized RUL", fontsize=6.8)
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    export(fig, stem)


# --------------------------------------------------------- ablation figure
def ablation_figure() -> None:
    ab = pd.read_csv(SOURCE / "ablation_three_seed.csv")
    variants = ["qort_no_spectral", "qort_no_trend", "qort_raw", "qort_single_prior", "qort_static_gate",
                "qort_no_optimism", "qort_no_relative_view", "qort_vibration_only", "qort_no_aux",
                "qort_router_r0", "qort_router_r05", "qort_router_r1", "qort_gate_base_const"]
    names = {"qort_no_spectral": "No spectral tokens", "qort_no_trend": "No trend/residual",
             "qort_raw": "Chronological tokens", "qort_no_optimism": "No conflict term",
             "qort_single_prior": r"Single $q_{50}$ prior", "qort_static_gate": "Static gate",
             "qort_no_relative_view": "No relative view", "qort_vibration_only": "Vibration only",
             "qort_no_aux": "No auxiliary losses", "qort_router_r0": r"$r=0$",
             "qort_router_r05": r"$r=0.5$", "qort_router_r1": r"$r=1$",
             "qort_gate_base_const": r"Constant $g_0$"}
    fig, ax = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
    markers = {"xjtu": "o", "phm": "s", "xjtu_c1": "^", "phm_c2": "D"}
    offsets = {"xjtu": -0.24, "phm": -0.08, "xjtu_c1": 0.08, "phm_c2": 0.24}
    for d in ALL:
        g = ab[ab.dataset == d].set_index("model")
        values = [g.loc[v, "diff_vs_full"] if v in g.index else np.nan for v in variants]
        ax.scatter(np.arange(len(variants)) + offsets[d], values, marker=markers[d], s=22,
                   color=SET_COLOR["development" if d in DEV else "held-out"],
                   edgecolor="white", linewidth=0.4, label=DATASET_LABEL[d], zorder=3)
    ax.axhline(0, color="0.3", lw=0.6)
    for k in range(len(variants)):
        ax.axvspan(k - 0.45, k + 0.45, color="0.95", zorder=0)
    ax.set_xticks(np.arange(len(variants)), [names[v] for v in variants], rotation=35, ha="right",
                  fontsize=6.5)
    shown = ab[ab.model.isin(variants)].diff_vs_full
    ax.set_ylim(top=float(np.nanmax(shown)) + 0.03)
    ax.set_ylabel(r"MAE$_{\mathrm{variant}}$ $-$ MAE$_{\mathrm{full}}$")
    ax.set_title("Three-seed component, expert-weight and gate-base variants (positive: variant is worse)")
    ax.legend(ncol=4, loc="upper left")
    export(fig, "Fig11_ablations")


# ------------------------------------------------- sensitivity figure
def sensitivity_figure() -> None:
    subsets = pd.read_csv(SOURCE / "training_subset_sensitivity.csv")
    shift_path = SOURCE / "prior_shift_bearing.csv"
    fig = plt.figure(figsize=(7.2, 2.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.2, 1.0, 1.0))
    ax = fig.add_subplot(grid[0])
    models = ["qort_net", "qort_prior_only", "median_lifetime_clock"]
    positions = []
    for i, d in enumerate(("xjtu", "phm")):
        for j, m in enumerate(models):
            values = subsets[(subsets.dataset == d) & (subsets.model == m)].delta_mae.values
            pos = i * 4 + j
            positions.append(pos)
            bp = ax.boxplot(values, positions=[pos], widths=0.6, patch_artist=True, showfliers=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS[m]); patch.set_alpha(0.55)
            for element in ("medians", "whiskers", "caps"):
                for line in bp[element]:
                    line.set_color("0.25"); line.set_linewidth(0.7)
            ax.scatter(np.full(len(values), pos) + np.random.default_rng(1).uniform(-0.15, 0.15, len(values)),
                       values, s=4, color=COLORS[m], alpha=0.8, zorder=3)
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_xticks([1, 5], [DATASET_LABEL["xjtu"], DATASET_LABEL["phm"]])
    ax.set_ylabel(r"$\Delta$MAE after dropping one training bearing")
    ax.set_title("Training-set composition")
    handles = [mpl.patches.Patch(color=COLORS[m], alpha=0.6, label=SHORT[m]) for m in models]
    ax.legend(handles=handles, loc="upper left", fontsize=6)
    panel(ax, "(a)")
    if shift_path.exists():
        shift = pd.read_csv(shift_path)
        for k, group in enumerate((DEV, HELD)):
            ax = fig.add_subplot(grid[k + 1])
            g = shift[shift.dataset.isin(group)].groupby(["model", "factor"]).mae.mean().reset_index()
            for m in ("qort_net", "gru_time_prior", "median_lifetime_clock"):
                h = g[g.model == m]
                ax.plot(h.factor, h.mae, marker="o", ms=3, color=COLORS[m], label=SHORT[m])
            ax.axvline(1.0, color="0.7", lw=0.6, ls=":")
            ax.set_xlabel(r"Reference-life scale factor $f$")
            ax.set_ylabel("Mean MAE (bearing-equal)")
            ax.set_title("Prior misspecification: " + ("development" if k == 0 else "held-out"))
            ax.legend(fontsize=6)
            panel(ax, "(b)" if k == 0 else "(c)")
    export(fig, "Fig12_sensitivity")


# --------------------------------------------- atypical / life-stage figure
def atypical_figure() -> None:
    atyp = pd.read_csv(SOURCE / "atypical_lifetime_analysis.csv")
    stages = pd.read_csv(SOURCE / "life_stage_errors.csv")
    fig = plt.figure(figsize=(7.2, 2.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.1, 1.1, 0.9))
    ax = fig.add_subplot(grid[0])
    for d in ALL:
        g = atyp[atyp.dataset == d]
        ax.scatter(g.lifetime_deviation, g.gain_vs_clock, s=20,
                   color=SET_COLOR["development" if d in DEV else "held-out"],
                   marker={"xjtu": "o", "phm": "s", "xjtu_c1": "^", "phm_c2": "D"}[d],
                   edgecolor="white", linewidth=0.4, label=DATASET_LABEL[d])
    ax.axhline(0, color="0.3", lw=0.6); ax.axvline(0, color="0.8", lw=0.6)
    ax.set_xlabel(r"Lifetime deviation $(T_b-L_{50})/L_{50}$")
    ax.set_ylabel(r"MAE$_{\mathrm{clock}}$ $-$ MAE$_{\mathrm{QORT}}$")
    ax.set_title("Gain over the clock vs. atypicality")
    ax.legend(fontsize=5.8, loc="upper right")
    panel(ax, "(a)")
    ax = fig.add_subplot(grid[1])
    models = ["qort_net", "qort_prior_only", "median_lifetime_clock", "gru_time_prior", "film_time_prior"]
    width = 0.16
    for k, m in enumerate(models):
        g = stages[stages.model == m]
        means = [g[s].mean() for s in ("early", "middle", "late")]
        ax.bar(np.arange(3) + (k - 2) * width, means, width=width, color=COLORS[m], label=SHORT[m])
    ax.set_xticks(range(3), ["Early\n0–30%", "Middle\n30–70%", "Late\n70–100%"])
    ax.set_ylabel("MAE (24 bearings)")
    ax.set_title("Life-stage error")
    ax.legend(fontsize=5.8, ncol=2)
    panel(ax, "(b)")
    ax = fig.add_subplot(grid[2])
    for column, label, color in (("correction", r"$|\hat y_t-p_t|$", COLORS["qort_net"]),
                                 ("gate", r"gate $g_t$", "#2A9D8F")):
        for group, ls in ((False, "-"), (True, "--")):
            g = atyp[(atyp.abs_deviation > 0.25) == group]
            values = [g[f"{column}_{s}"].mean() for s in ("early", "middle", "late")]
            ax.plot(range(3), values, marker="o", ms=3, ls=ls, color=color,
                    label=f"{label}, {'atypical' if group else 'typical'}")
    ax.set_xticks(range(3), ["Early", "Middle", "Late"])
    ax.set_title("Vibration correction by stage")
    ax.legend(fontsize=5.6)
    panel(ax, "(c)")
    export(fig, "Fig10_atypical_lifestage")


if __name__ == "__main__":
    protocol_figure()
    comparison_figure()
    trajectory_figure(("xjtu",), "Fig6_xjtu_c2_trajectories", 3)
    trajectory_figure(("phm",), "Fig7_phm_c1_trajectories", 4)
    trajectory_figure(HELD, "Fig9_heldout_trajectories", 4)
    ablation_figure()
    sensitivity_figure()
    atypical_figure()

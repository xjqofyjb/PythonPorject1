"""Build publication-focused composite figures for the paper."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from analysis.style import set_style

# === universal style ===
UNIVERSAL_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}
plt.rcParams.update(UNIVERSAL_STYLE)
mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

COLORS = {
    "CG": "#2166AC",
    "MILP300": "#4393C3",
    "MILP60": "#92C5DE",
    "FO": "#D6604D",
    "RollingH": "#F4A582",
    "RestrictedCG": "#B2ABD2",
    "FIFO": "#878787",
    "Greedy": "#BABABA",
}

METHOD_ORDER = ["cg", "restricted_cg", "rcg_random", "rcg_arrival", "rolling_horizon", "fix_and_optimize", "milp300", "milp60", "greedy", "fifo"]
METHOD_LABELS = {
    "cg": "CG+IR",
    "restricted_cg": "Restricted-CG",
    "rcg_random": "Restricted CG (Random 50%)",
    "rcg_arrival": "Restricted CG (First 50%)",
    "rolling_horizon": "Rolling-Horizon MILP",
    "fix_and_optimize": "Fix-and-Optimize",
    "milp300": "MILP-300s",
    "milp60": "MILP-60s",
    "greedy": "Greedy",
    "fifo": "FIFO",
    "cg_basic": "CG-Basic",
    "cg_warm": "CG+Warm",
    "cg_stab": "CG+Stab",
    "cg_multik": "CG+MultiCol",
    "cg_full": "CG-Full",
}
METHOD_COLORS = {
    "cg": COLORS["CG"],
    "restricted_cg": COLORS["RestrictedCG"],
    "rcg_random": COLORS["RestrictedCG"],
    "rcg_arrival": COLORS["RestrictedCG"],
    "rolling_horizon": COLORS["RollingH"],
    "fix_and_optimize": COLORS["FO"],
    "milp300": COLORS["MILP300"],
    "milp60": COLORS["MILP60"],
    "greedy": COLORS["Greedy"],
    "fifo": COLORS["FIFO"],
    "cg_basic": COLORS["CG"],
    "cg_warm": COLORS["FO"],
    "cg_stab": COLORS["RollingH"],
    "cg_multik": COLORS["RestrictedCG"],
    "cg_full": COLORS["MILP300"],
}
SCENARIO_LABELS = {"U": "Uniform", "P": "Peaked", "L": "Long service"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, outdir: Path, name: str) -> None:
    ensure_dir(outdir)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)


def place_shared_legend(fig: plt.Figure, handles: list, labels: list, *, threshold: int = 6) -> tuple[str, Dict[str, object]]:
    if len(labels) > threshold:
        ncol = min(len(labels), 5)
        return "bottom", {"loc": "lower center", "ncol": ncol, "bbox_to_anchor": (0.5, -0.02)}
    ncol = min(len(labels), 5)
    return "top", {"loc": "upper center", "ncol": ncol, "bbox_to_anchor": (0.5, 1.02)}


def add_panel_labels(axes: Iterable[plt.Axes]) -> None:
    for label, ax in zip("abcdefghijklmnopqrstuvwxyz", axes):
        ax.text(
            0.5,
            -0.16,
            f"({label})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="normal",
            va="top",
            ha="center",
        )


def mean_ci(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    mean = float(clean.mean())
    if len(clean) <= 1:
        return mean, 0.0
    ci = 1.96 * float(clean.std(ddof=1)) / np.sqrt(len(clean))
    return mean, ci


def summarize(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        mean, ci = mean_ci(group[value_col])
        row[f"{value_col}_mean"] = mean
        row[f"{value_col}_ci"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def line_with_ci(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, yci: str, method: str) -> None:
    if df.empty:
        return
    ax.plot(
        df[x],
        df[y],
        marker="o",
        color=METHOD_COLORS.get(method, "#333333"),
        label=METHOD_LABELS.get(method, method),
    )
    ax.fill_between(
        df[x],
        df[y] - df[yci],
        df[y] + df[yci],
        color=METHOD_COLORS.get(method, "#333333"),
        alpha=0.15,
    )


def build_main_performance(main_df: pd.DataFrame, outdir: Path) -> None:
    summary_obj = summarize(main_df, ["N", "method"], "obj")
    summary_runtime = summarize(main_df, ["N", "method"], "runtime_total")
    cg_effort = main_df[main_df["method"] == "cg"].groupby(["N"], dropna=False).agg(
        num_iters_mean=("num_iters", "mean"),
        num_columns_added_mean=("num_columns_added", "mean"),
    ).reset_index()

    cg_ref = summary_obj[summary_obj["method"] == "cg"][["N", "obj_mean"]].rename(columns={"obj_mean": "cg_obj"})
    rel_gap = summary_obj.merge(cg_ref, on="N", how="left")
    rel_gap["gap_pct_mean"] = (rel_gap["obj_mean"] - rel_gap["cg_obj"]) / rel_gap["cg_obj"] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))
    flat_axes = axes.ravel()

    for method in METHOD_ORDER:
        obj_m = summary_obj[summary_obj["method"] == method].sort_values("N")
        run_m = summary_runtime[summary_runtime["method"] == method].sort_values("N")
        gap_m = rel_gap[rel_gap["method"] == method].sort_values("N")
        if obj_m.empty:
            continue
        line_with_ci(flat_axes[0], obj_m, "N", "obj_mean", "obj_ci", method)
        line_with_ci(flat_axes[2], run_m, "N", "runtime_total_mean", "runtime_total_ci", method)
        if method != "cg":
            flat_axes[1].plot(
                gap_m["N"],
                gap_m["gap_pct_mean"],
                marker="o",
                color=METHOD_COLORS.get(method, "#333333"),
                label=METHOD_LABELS.get(method, method),
            )

    flat_axes[0].set_xlabel("Number of ships $N$")
    flat_axes[0].set_ylabel("Objective value")

    flat_axes[1].set_xlabel("Number of ships $N$")
    flat_axes[1].set_ylabel("Gap to CG+IR (%)")
    flat_axes[1].axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")

    flat_axes[2].set_xlabel("Number of ships $N$")
    flat_axes[2].set_ylabel("Runtime (s)")
    flat_axes[2].set_yscale("log")

    cg_gap = main_df[main_df["method"] == "cg"].copy()
    has_gap_pct = "gap_pct" in cg_gap.columns and cg_gap["gap_pct"].notna().any()
    if has_gap_pct:
        gap_summary = summarize(cg_gap, ["N"], "gap_pct").sort_values("N")
        flat_axes[3].bar(
            gap_summary["N"],
            gap_summary["gap_pct_mean"],
            yerr=gap_summary["gap_pct_ci"],
            color=METHOD_COLORS["cg"],
            alpha=0.84,
            width=28,
            capsize=3,
        )
        for _, row in gap_summary.iterrows():
            val = float(row["gap_pct_mean"])
            flat_axes[3].text(
                row["N"],
                val + max(float(row["gap_pct_ci"]), 0.002) + 0.002,
                f"{val:.3f}%",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )
        upper = max(0.10, float((gap_summary["gap_pct_mean"] + gap_summary["gap_pct_ci"]).max()) * 1.28)
        flat_axes[3].set_ylim(0.0, upper)
        flat_axes[3].set_xlabel("Number of ships $N$")
        flat_axes[3].set_ylabel("IRMP vs LP bound (%)")
    else:
        flat_axes[3].plot(
            cg_effort["N"],
            cg_effort["num_iters_mean"],
            color=METHOD_COLORS["cg"],
            marker="o",
            label="CG iterations",
        )
        effort_twin = flat_axes[3].twinx()
        effort_twin.bar(
            cg_effort["N"],
            cg_effort["num_columns_added_mean"],
            color=METHOD_COLORS["cg"],
            alpha=0.16,
            width=28,
            label="Generated columns",
        )
        effort_twin.grid(False)
        effort_twin.set_ylabel("Generated columns")

        flat_axes[3].set_xlabel("Number of ships $N$")
        flat_axes[3].set_ylabel("CG iterations")
        effort_handles, effort_labels = flat_axes[3].get_legend_handles_labels()
        twin_handles, twin_labels = effort_twin.get_legend_handles_labels()
        flat_axes[3].legend(
            effort_handles + twin_handles,
            effort_labels + twin_labels,
            loc="upper left",
            fontsize=8.5,
            frameon=False,
        )

    add_panel_labels(flat_axes)
    handles, labels = flat_axes[0].get_legend_handles_labels()
    _, legend_kwargs = place_shared_legend(fig, handles, labels)
    fig.legend(handles, labels, **legend_kwargs)
    if legend_kwargs["loc"] == "lower center":
        fig.tight_layout(rect=(0, 0.14, 1, 1))
    else:
        fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    save_fig(fig, outdir, "Fig_Paper_Main_Performance")


def build_scenario_mechanism(scenario_df: pd.DataFrame | None, mechanism_df: pd.DataFrame | None, outdir: Path) -> None:
    if scenario_df is None and mechanism_df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    add_panel_labels(axes)

    if scenario_df is not None:
        scen = summarize(scenario_df, ["scenario", "method"], "obj")
        scenarios = [s for s in ["U", "P", "L"] if s in scen["scenario"].unique()]
        methods = [m for m in METHOD_ORDER if m in scen["method"].unique()]
        x = np.arange(len(scenarios))
        width = min(0.82 / max(len(methods), 1), 0.22)
        for idx, method in enumerate(methods):
            block = scen[scen["method"] == method].set_index("scenario").reindex(scenarios).reset_index()
            axes[0].bar(
                x + (idx - (len(methods) - 1) / 2) * width,
                block["obj_mean"],
                yerr=block["obj_ci"],
                width=width,
                color=METHOD_COLORS.get(method, "#333333"),
                label=METHOD_LABELS.get(method, method),
                alpha=0.92,
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios])
        axes[0].set_ylabel("Objective value")
    else:
        axes[0].axis("off")

    if mechanism_df is not None:
        mechanisms = [m for m in ["hybrid", "battery_only", "shore_only"] if m in mechanism_df["mechanism"].unique()]
        methods = [
            m
            for m in METHOD_ORDER
            if m in mechanism_df["method"].unique() and not m.startswith("milp")
        ]
        hybrid = mechanism_df[mechanism_df["mechanism"] == "hybrid"][["seed", "method", "obj"]].rename(columns={"obj": "hybrid_obj"})
        mech_rel = mechanism_df.merge(hybrid, on=["seed", "method"], how="left")
        mech_rel["cost_penalty_pct"] = (mech_rel["obj"] - mech_rel["hybrid_obj"]) / mech_rel["hybrid_obj"] * 100.0
        mech_summary = summarize(mech_rel, ["mechanism", "method"], "cost_penalty_pct")
        x = np.arange(len(mechanisms))
        width = min(0.82 / max(len(methods), 1), 0.24)
        for idx, method in enumerate(methods):
            block = mech_summary[mech_summary["method"] == method].set_index("mechanism").reindex(mechanisms).reset_index()
            axes[1].bar(
                x + (idx - (len(methods) - 1) / 2) * width,
                block["cost_penalty_pct_mean"],
                yerr=block["cost_penalty_pct_ci"],
                width=width,
                color=METHOD_COLORS.get(method, "#333333"),
                label=METHOD_LABELS.get(method, method),
                alpha=0.92,
            )
        axes[1].axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["Hybrid\n(baseline)", "Battery only", "Shore only"])
        axes[1].set_ylabel("Cost penalty vs hybrid (%)")
    else:
        axes[1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        _, legend_kwargs = place_shared_legend(fig, handles, labels)
        fig.legend(handles, labels, **legend_kwargs)
        if legend_kwargs["loc"] == "lower center":
            fig.tight_layout(rect=(0, 0.16, 1, 1))
        else:
            fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    else:
        fig.tight_layout()
    save_fig(fig, outdir, "Fig_Paper_Scenario_Mechanism")


def build_simops_figure(simops_df: pd.DataFrame, outdir: Path) -> None:
    summary_obj = summarize(simops_df, ["N", "operation_mode", "method"], "obj")
    summary_brown = summarize(simops_df, ["N", "operation_mode", "method"], "brown_ratio")
    summary_mask = summarize(simops_df, ["N", "operation_mode", "method"], "avg_masking_rate")

    fig = plt.figure(figsize=(11.2, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.30, wspace=0.25)
    ax_top = fig.add_subplot(grid[0, :])
    ax_bl = fig.add_subplot(grid[1, 0])
    ax_br = fig.add_subplot(grid[1, 1])

    methods = [
        m
        for m in METHOD_ORDER
        if m in simops_df["method"].unique() and not m.startswith("milp")
    ]
    for method in methods:
        sim = summary_obj[(summary_obj["method"] == method) & (summary_obj["operation_mode"] == "simops")].sort_values("N")
        seq = summary_obj[(summary_obj["method"] == method) & (summary_obj["operation_mode"] == "sequential")].sort_values("N")
        if sim.empty or seq.empty:
            continue
        merged = sim.merge(seq[["N", "obj_mean"]], on="N", suffixes=("_sim", "_seq"))
        merged["saving_pct"] = (merged["obj_mean_seq"] - merged["obj_mean_sim"]) / merged["obj_mean_seq"] * 100.0
        ax_top.plot(
            merged["N"],
            merged["saving_pct"],
            marker="o",
            color=METHOD_COLORS.get(method, "#333333"),
            label=METHOD_LABELS.get(method, method),
            linewidth=2.2 if method == "cg" else 1.7,
            markersize=5.2 if method == "cg" else 4.5,
        )

        sim_brown = summary_brown[(summary_brown["method"] == method) & (summary_brown["operation_mode"] == "simops")].sort_values("N")
        seq_brown = summary_brown[(summary_brown["method"] == method) & (summary_brown["operation_mode"] == "sequential")].sort_values("N")
        if not sim_brown.empty and not seq_brown.empty:
            brown = sim_brown.merge(seq_brown[["N", "brown_ratio_mean"]], on="N", suffixes=("_sim", "_seq"))
            brown["brown_reduction_pp"] = (brown["brown_ratio_mean_seq"] - brown["brown_ratio_mean_sim"]) * 100.0
            ax_bl.plot(
                brown["N"],
                brown["brown_reduction_pp"],
                marker="o",
                color=METHOD_COLORS.get(method, "#333333"),
                label=METHOD_LABELS.get(method, method),
                linewidth=2.2 if method == "cg" else 1.7,
                markersize=5.2 if method == "cg" else 4.5,
            )

    cg_mask = summary_mask[
        (summary_mask["method"] == "cg") &
        (summary_mask["operation_mode"] == "simops")
    ].sort_values("N")
    if not cg_mask.empty:
        ax_br.plot(
            cg_mask["N"],
            cg_mask["avg_masking_rate_mean"],
            marker="o",
            color=METHOD_COLORS["cg"],
            linewidth=2.0,
            label="CG+IR",
        )
        ax_br.fill_between(
            cg_mask["N"],
            cg_mask["avg_masking_rate_mean"] - cg_mask["avg_masking_rate_ci"],
            cg_mask["avg_masking_rate_mean"] + cg_mask["avg_masking_rate_ci"],
            color=METHOD_COLORS["cg"],
            alpha=0.15,
        )
        for _, row in cg_mask.iterrows():
            ax_br.annotate(
                f"{row['avg_masking_rate_mean']:.2f}",
                xy=(row["N"], row["avg_masking_rate_mean"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )

    ax_top.set_xlabel("Number of ships $N$")
    ax_top.set_ylabel("Savings vs sequential (%)")
    ax_top.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")

    ax_bl.set_xlabel("Number of ships $N$")
    ax_bl.set_ylabel("Sequential - SIMOPS (percentage points)")
    ax_bl.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--")

    ax_br.set_xlabel("Number of ships $N$")
    ax_br.set_ylabel("Average masking rate")
    ax_br.set_ylim(0.0, 1.05)
    ax_br.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    add_panel_labels([ax_top, ax_bl, ax_br])
    handles, labels = ax_top.get_legend_handles_labels()
    if handles:
        _, legend_kwargs = place_shared_legend(fig, handles, labels)
        fig.legend(handles, labels, **legend_kwargs)
        if legend_kwargs["loc"] == "lower center":
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.96, hspace=0.42, wspace=0.28)
        else:
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.88, hspace=0.42, wspace=0.28)
    else:
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.94, hspace=0.42, wspace=0.28)
    save_fig(fig, outdir, "Fig_Paper_SIMOPS")


def build_sensitivity_figure(sens_df: pd.DataFrame, outdir: Path) -> None:
    cg_df = sens_df[sens_df["method"] == "cg"].copy()
    if cg_df.empty:
        cg_df = sens_df.copy()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    param_specs = [
        ("battery_cost", "Battery service cost ($/kWh)"),
        ("shore_cap", "Number of shore-power berths"),
        ("deadline_tightness", "Deadline tightness factor"),
    ]

    for ax, (param_name, xlabel) in zip(axes, param_specs):
        block = cg_df[cg_df["param_name"] == param_name].copy()
        if block.empty:
            ax.axis("off")
            continue
        obj = summarize(block, ["param_value"], "obj").sort_values("param_value")
        brown = summarize(block, ["param_value"], "brown_ratio").sort_values("param_value") if "brown_ratio" in block.columns else None

        ax.plot(obj["param_value"], obj["obj_mean"], color="#0F4C81", marker="o")
        ax.fill_between(obj["param_value"], obj["obj_mean"] - obj["obj_ci"], obj["obj_mean"] + obj["obj_ci"], color="#0F4C81", alpha=0.16)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Objective value")
        if brown is not None and not brown.empty:
            obj_lookup = obj.set_index("param_value")["obj_mean"]
            for _, row in brown.iterrows():
                share = float(row["brown_ratio_mean"])
                if share <= 0.005:
                    continue
                x_pos = row["param_value"]
                y_pos = float(obj_lookup.loc[x_pos]) if x_pos in obj_lookup.index else float(obj["obj_mean"].max())
                ax.annotate(
                    f"AE: {share * 100:.1f}%",
                    xy=(x_pos, y_pos),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.0,
                    color="#B85C38",
                )

    add_panel_labels(axes)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_fig(fig, outdir, "Fig_Paper_Sensitivity")


def build_ablation_figure(ablation_df: pd.DataFrame, outdir: Path) -> None:
    variants = ["cg_basic", "cg_warm", "cg_stab", "cg_multik", "cg_full"]
    block = ablation_df[ablation_df["method"].isin(variants)].copy()
    if block.empty:
        return

    summary = block.groupby("method", dropna=False).agg(
        runtime_mean=("runtime_total", "mean"),
        num_iters_mean=("num_iters", "mean"),
        num_columns_added_mean=("num_columns_added", "mean"),
    ).reset_index()
    summary["label"] = summary["method"].map(METHOD_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.0))
    metrics = [
        ("runtime_mean", "Runtime (s)", ""),
        ("num_iters_mean", "Iterations", ""),
        ("num_columns_added_mean", "Columns", ""),
    ]

    x = np.arange(len(summary))
    colors = [METHOD_COLORS.get(m, "#333333") for m in summary["method"]]
    for ax, (metric, ylabel, _title) in zip(axes, metrics):
        ax.bar(x, summary[metric], color=colors, alpha=0.94)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["label"], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
    add_panel_labels(axes)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save_fig(fig, outdir, "Fig_Paper_Ablation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build composite paper figures.")
    parser.add_argument("--results_dir", default="results", help="Directory containing experiment CSV outputs")
    parser.add_argument("--outdir", default="figs/paper", help="Output directory for paper figures")
    args = parser.parse_args()

    set_style()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)

    main_df = load_csv(results_dir / "results_main_rigorous.csv")
    scenario_df = load_csv(results_dir / "results_scenario_rigorous.csv")
    mechanism_df = load_csv(results_dir / "results_mechanism_rigorous.csv")
    simops_df = load_csv(results_dir / "results_simops_rigorous.csv")
    sensitivity_df = load_csv(results_dir / "results_sensitivity_rigorous.csv")
    ablation_df = load_csv(results_dir / "results_ablation_rigorous.csv")

    if main_df is not None:
        build_main_performance(main_df, outdir)
    build_scenario_mechanism(scenario_df, mechanism_df, outdir)
    if simops_df is not None:
        build_simops_figure(simops_df, outdir)
    if sensitivity_df is not None:
        build_sensitivity_figure(sensitivity_df, outdir)
    if ablation_df is not None:
        build_ablation_figure(ablation_df, outdir)


if __name__ == "__main__":
    main()

"""Generate publication-ready plots from results CSV."""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.style import set_style


METHOD_ORDER = ["cg", "restricted_cg", "rcg_random", "rcg_arrival", "rolling_horizon", "fix_and_optimize", "fifo", "greedy", "milp300", "milp60"]
PAPER_METHODS = ["cg_basic", "cg_warm", "cg_stab", "cg_multik", "cg_full", "restricted_cg", "rcg_random", "rcg_arrival", "rolling_horizon", "fix_and_optimize", "fifo", "greedy", "milp300", "milp60"]
SCENARIO_LABELS = {
    "U": "Uniform",
    "P": "Peaked Arrivals",
    "L": "Long Service",
}
COLORS = {
    "cg": "#1f77b4",
    "cg_basic": "#1f77b4",
    "cg_warm": "#ff7f0e",
    "cg_stab": "#2ca02c",
    "cg_multik": "#d62728",
    "cg_full": "#9467bd",
    "restricted_cg": "#17becf",
    "rcg_random": "#17becf",
    "rcg_arrival": "#bc80bd",
    "rolling_horizon": "#4C956C",
    "fix_and_optimize": "#6C5B7B",
    "fifo": "#8c564b",
    "greedy": "#e377c2",
    "milp300": "#7f7f7f",
    "milp60": "#bcbd22",
}
MARKERS = {
    "cg": "o",
    "cg_basic": "o",
    "cg_warm": "s",
    "cg_stab": "^",
    "cg_multik": "D",
    "cg_full": "P",
    "restricted_cg": "h",
    "rcg_random": "h",
    "rcg_arrival": "*",
    "rolling_horizon": "P",
    "fix_and_optimize": "D",
    "fifo": "v",
    "greedy": "X",
    "milp300": ">",
    "milp60": "<",
}

LINESTYLES = {
    "simops": "-",
    "sequential": "--",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_fig(fig, outdir: str, name: str) -> None:
    ensure_dir(outdir)
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    pdf_path = os.path.join(outdir, f"{name}.pdf")
    fig.savefig(pdf_path)
    plt.close(fig)


def _type_codes(df: pd.DataFrame) -> List[str]:
    codes = []
    for col in df.columns:
        if col.startswith("type_") and col.endswith("_count"):
            codes.append(col.split("_")[1])
    return sorted(set(codes))


def summarize(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols + ["method"], dropna=False)
    agg = grouped.agg(
        obj_mean=("obj", "mean"),
        obj_std=("obj", "std"),
        runtime_mean=("runtime_total", "mean"),
        runtime_std=("runtime_total", "std"),
    )
    return agg.reset_index()


def plot_lines(df: pd.DataFrame, x_col: str, y_col: str, y_std: str, ax, ylabel: str, logy: bool = False):
    for method in METHOD_ORDER:
        g = df[df["method"] == method]
        if g.empty:
            continue
        ax.plot(g[x_col], g[y_col], label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
        if y_std in g.columns:
            ax.fill_between(g[x_col], g[y_col] - g[y_std], g[y_col] + g[y_std], alpha=0.2, color=COLORS.get(method, None))
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)


def plot_main(df: pd.DataFrame, outdir: str, logy: bool = False) -> None:
    summary = summarize(df, ["N"]).sort_values("N")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    plot_lines(summary, "N", "obj_mean", "obj_std", ax, ylabel="Objective")
    ax.set_title("Objective vs N")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save_fig(fig, outdir, "Fig_Obj_U")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    plot_lines(summary, "N", "runtime_mean", "runtime_std", ax, ylabel="Runtime (s)", logy=logy)
    ax.set_title("Runtime vs N")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save_fig(fig, outdir, "Fig_Runtime_U")

    cg_df = df[df["method"] == "cg"].groupby(["N"]).agg(
        pricing_calls_mean=("num_pricing_calls", "mean"),
        iters_mean=("num_iters", "mean"),
        pricing_time_share_mean=("pricing_time_share", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharex=True)
    axes[0].plot(cg_df["N"], cg_df["pricing_calls_mean"], marker="o", color=COLORS["cg"])
    axes[0].set_title("Pricing Calls")
    axes[0].set_ylabel("Count")
    axes[1].plot(cg_df["N"], cg_df["iters_mean"], marker="o", color=COLORS["cg"])
    axes[1].set_title("CG Iterations")
    axes[2].plot(cg_df["N"], cg_df["pricing_time_share_mean"], marker="o", color=COLORS["cg"])
    axes[2].set_title("Pricing Time Share")
    axes[2].set_ylabel("Ratio")
    for ax in axes:
        ax.set_xlabel("N")
        ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_CG_Metrics_U")

    # Combined panel: Obj + Runtime + CG metrics (2x2 layout)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    plot_lines(summary, "N", "obj_mean", "obj_std", axes[0, 0], ylabel="Objective")
    axes[0, 0].set_title("Objective vs N")
    plot_lines(summary, "N", "runtime_mean", "runtime_std", axes[0, 1], ylabel="Runtime (s)", logy=logy)
    axes[0, 1].set_title("Runtime vs N")
    axes[1, 0].plot(cg_df["N"], cg_df["pricing_calls_mean"], marker="o", color=COLORS["cg"])
    axes[1, 0].set_title("Pricing Calls")
    axes[1, 0].set_xlabel("N")
    axes[1, 0].set_ylabel("Count")
    axes[1, 1].plot(cg_df["N"], cg_df["pricing_time_share_mean"], marker="o", color=COLORS["cg"])
    axes[1, 1].set_title("Pricing Time Share")
    axes[1, 1].set_xlabel("N")
    axes[1, 1].set_ylabel("Ratio")
    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save_fig(fig, outdir, "Fig_Main_Combined_U")


def plot_mechanism(df: pd.DataFrame, outdir: str) -> None:
    summary = df.groupby(["mechanism", "method"]).agg(obj_mean=("obj", "mean"), obj_std=("obj", "std"))
    summary = summary.reset_index()

    mechanisms = summary["mechanism"].unique().tolist()
    x = np.arange(len(mechanisms))
    width = 0.18

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, method in enumerate(METHOD_ORDER):
        g = summary[summary["method"] == method]
        if g.empty:
            continue
        idx = [mechanisms.index(m) for m in g["mechanism"]]
        ax.bar(x + (i - 1.5) * width, g["obj_mean"], width=width, label=method, color=COLORS.get(method, None), yerr=g["obj_std"])
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms)
    ax.set_ylabel("Objective")
    ax.set_title("Mechanism Comparison")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Mechanism_U")

    ratio_summary = df.groupby(["mechanism", "method"]).agg(
        shore_ratio_mean=("shore_ratio", "mean"),
        battery_ratio_mean=("battery_ratio", "mean"),
        brown_ratio_mean=("brown_ratio", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    methods = [m for m in ["cg", "fifo", "greedy"] if m in ratio_summary["method"].unique()]
    x = np.arange(len(mechanisms))
    width = 0.22
    for i, method in enumerate(methods):
        g = ratio_summary[ratio_summary["method"] == method]
        if g.empty:
            continue
        idx = [mechanisms.index(m) for m in g["mechanism"]]
        shore = g["shore_ratio_mean"].to_numpy()
        battery = g["battery_ratio_mean"].to_numpy()
        brown = g["brown_ratio_mean"].to_numpy()

        base = np.zeros_like(shore)
        ax.bar(x + (i - 1) * width, shore, width=width, label=f"{method}-shore", color="#4c78a8")
        base = shore
        ax.bar(x + (i - 1) * width, battery, width=width, bottom=base, label=f"{method}-battery", color="#f58518")
        base = base + battery
        ax.bar(x + (i - 1) * width, brown, width=width, bottom=base, label=f"{method}-brown", color="#54a24b")

    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms)
    ax.set_ylabel("Mode Ratio")
    ax.set_title("Mode Usage by Mechanism")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Mechanism_Modes_U")


def plot_sensitivity(df: pd.DataFrame, outdir: str) -> None:
    for param_name in df["param_name"].dropna().unique():
        sub = df[df["param_name"] == param_name]
        summary = summarize(sub, ["param_value"]).sort_values("param_value")

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        plot_lines(summary, "param_value", "obj_mean", "obj_std", ax, ylabel="Objective")
        ax.set_title(f"Sensitivity: {param_name}")
        ax.set_xlabel(param_name)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        name_map = {
            "battery_cost": "Fig_Sens_BatteryCost",
            "shore_cap": "Fig_Sens_ShoreCap",
            "deadline_tightness": "Fig_Sens_Deadline",
        }
        save_fig(fig, outdir, name_map.get(param_name, f"Fig_Sens_{param_name}"))


def plot_type_breakdown(df: pd.DataFrame, outdir: str) -> None:
    type_codes = _type_codes(df)
    if not type_codes or "method" not in df.columns:
        return
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(type_codes))
    width = min(0.8 / len(methods), 0.25)
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        values = []
        for t_code in type_codes:
            total_col = f"type_{t_code}_cost_total"
            count_col = f"type_{t_code}_count"
            if total_col not in sub.columns or count_col not in sub.columns:
                values.append(np.nan)
                continue
            denom = sub[count_col].sum()
            values.append(sub[total_col].sum() / denom if denom > 0 else np.nan)
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, values, width=width, label=method, color=COLORS.get(method, None))
    ax.set_xticks(x)
    ax.set_xticklabels(type_codes)
    ax.set_ylabel("Avg Cost per Ship")
    ax.set_title("Cost by Ship Type")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Type_Cost")

    # Mode ratios by type (use CG if available, else first method)
    method_pick = "cg" if "cg" in df["method"].unique() else methods[0]
    sub = df[df["method"] == method_pick]
    shore_vals = []
    battery_vals = []
    brown_vals = []
    for t_code in type_codes:
        shore_col = f"type_{t_code}_shore_ratio"
        battery_col = f"type_{t_code}_battery_ratio"
        brown_col = f"type_{t_code}_brown_ratio"
        if shore_col not in sub.columns or battery_col not in sub.columns or brown_col not in sub.columns:
            shore_vals.append(np.nan)
            battery_vals.append(np.nan)
            brown_vals.append(np.nan)
            continue
        shore_vals.append(sub[shore_col].mean())
        battery_vals.append(sub[battery_col].mean())
        brown_vals.append(sub[brown_col].mean())

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(type_codes))
    ax.bar(x, shore_vals, label="shore", color="#4c78a8")
    base = np.array(shore_vals, dtype=float)
    ax.bar(x, battery_vals, bottom=base, label="battery", color="#f58518")
    base = base + np.array(battery_vals, dtype=float)
    ax.bar(x, brown_vals, bottom=base, label="brown", color="#54a24b")
    ax.set_xticks(x)
    ax.set_xticklabels(type_codes)
    ax.set_ylabel("Mode Ratio")
    ax.set_title(f"Mode Ratios by Type ({method_pick})")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Type_Modes")

def _scenario_order(df: pd.DataFrame) -> List[str]:
    preferred = [s for s in ["U", "P", "L"] if s in df["scenario"].unique()]
    remaining = [s for s in df["scenario"].unique() if s not in preferred]
    return preferred + sorted(remaining)


def plot_scenario(df: pd.DataFrame, outdir: str) -> None:
    if "scenario" not in df.columns:
        return
    summary = df.groupby(["scenario", "method"], dropna=False).agg(
        obj_mean=("obj", "mean"),
        obj_std=("obj", "std"),
        runtime_mean=("runtime_total", "mean"),
        runtime_std=("runtime_total", "std"),
    ).reset_index()

    scenarios = _scenario_order(summary)
    methods = [m for m in METHOD_ORDER if m in summary["method"].unique()]
    if not methods:
        return
    x = np.arange(len(scenarios))
    width = min(0.8 / len(methods), 0.22)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, method in enumerate(methods):
        g = summary[summary["method"] == method]
        if g.empty:
            continue
        g = g.set_index("scenario").reindex(scenarios).reset_index()
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, g["obj_mean"], width=width, label=method, color=COLORS.get(method, None), yerr=g["obj_std"])
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios])
    ax.set_ylabel("Objective")
    ax.set_title("Scenario Comparison (Objective)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Scenario_Obj")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, method in enumerate(methods):
        g = summary[summary["method"] == method]
        if g.empty:
            continue
        g = g.set_index("scenario").reindex(scenarios).reset_index()
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, g["runtime_mean"], width=width, label=method, color=COLORS.get(method, None), yerr=g["runtime_std"])
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios])
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Scenario Comparison (Runtime)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Scenario_Runtime")

    ratio_cols = ["shore_ratio", "battery_ratio", "brown_ratio"]
    if all(c in df.columns for c in ratio_cols):
        ratio = df.groupby(["scenario", "method"]).agg(
            shore_ratio_mean=("shore_ratio", "mean"),
            battery_ratio_mean=("battery_ratio", "mean"),
            brown_ratio_mean=("brown_ratio", "mean"),
        ).reset_index()
        methods_ratio = [m for m in ["cg", "fifo", "greedy"] if m in ratio["method"].unique()]
        if methods_ratio:
            fig, ax = plt.subplots(figsize=(6.6, 3.8))
            x = np.arange(len(scenarios))
            width = min(0.7 / len(methods_ratio), 0.22)
            for i, method in enumerate(methods_ratio):
                g = ratio[ratio["method"] == method]
                if g.empty:
                    continue
                g = g.set_index("scenario").reindex(scenarios).reset_index()
                shore = g["shore_ratio_mean"].to_numpy()
                battery = g["battery_ratio_mean"].to_numpy()
                brown = g["brown_ratio_mean"].to_numpy()
                base = np.zeros_like(shore)
                offset = (i - (len(methods_ratio) - 1) / 2) * width
                ax.bar(x + offset, shore, width=width, label=f"{method}-shore", color="#4c78a8")
                base = shore
                ax.bar(x + offset, battery, width=width, bottom=base, label=f"{method}-battery", color="#f58518")
                base = base + battery
                ax.bar(x + offset, brown, width=width, bottom=base, label=f"{method}-brown", color="#54a24b")
            ax.set_xticks(x)
            ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios])
            ax.set_ylabel("Mode Ratio")
            ax.set_title("Scenario Comparison (Mode Ratios)")
            ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
            ax.grid(True, axis="y", alpha=0.3)
            save_fig(fig, outdir, "Fig_Scenario_Modes")


def _load_trace_frames(traces_dir: str, n_value: int) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(traces_dir):
        return frames
    for fname in os.listdir(traces_dir):
        if not fname.endswith(".csv"):
            continue
        if f"N{n_value}_" not in fname:
            continue
        method = None
        for m in PAPER_METHODS + METHOD_ORDER:
            if fname.endswith(f"_{m}.csv"):
                method = m
                break
        if method is None:
            method = fname.split("_")[-1].replace(".csv", "")
        path = os.path.join(traces_dir, fname)
        df = pd.read_csv(path)
        frames.setdefault(method, []).append(df)
    out: Dict[str, pd.DataFrame] = {}
    for method, items in frames.items():
        concat = pd.concat(items, ignore_index=True)
        out[method] = concat
    return out


def plot_convergence(traces_dir: str, outdir: str, n_value: int = 500) -> None:
    frames = _load_trace_frames(traces_dir, n_value)
    if not frames:
        return

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in PAPER_METHODS:
        if method not in frames:
            continue
        df = frames[method]
        grouped = df.groupby("iteration").agg(
            wall_time_mean=("wall_time", "mean"),
            best_primal_mean=("best_primal_obj", "mean"),
        ).reset_index()
        ax.plot(
            grouped["wall_time_mean"],
            grouped["best_primal_mean"],
            label=method,
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, None),
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Best Primal Objective")
    ax.set_title(f"Convergence (N={n_value})")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Convergence_N500")


def plot_cg_traces(traces_dir: str, outdir: str, n_value: int = 500) -> None:
    frames = _load_trace_frames(traces_dir, n_value)
    if not frames:
        return

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True)
    for method in PAPER_METHODS:
        if not method.startswith("cg"):
            continue
        if method not in frames:
            continue
        df = frames[method]
        grouped = df.groupby("iteration").agg(
            rmp_obj_mean=("rmp_obj", "mean"),
            pricing_calls_mean=("pricing_calls_cum", "mean"),
            min_rc_mean=("min_reduced_cost_last", "mean"),
        ).reset_index()
        axes[0].plot(grouped["iteration"], grouped["rmp_obj_mean"], label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
        axes[1].plot(grouped["iteration"], grouped["pricing_calls_mean"], label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
        axes[2].plot(grouped["iteration"], grouped["min_rc_mean"], label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
    axes[0].set_title("RMP Objective")
    axes[1].set_title("Pricing Calls")
    axes[2].set_title("Min Reduced Cost")
    axes[0].set_ylabel("Value")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.grid(True, axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save_fig(fig, outdir, "Fig_CG_Traces_N500")


def plot_ablation(df: pd.DataFrame, outdir: str) -> None:
    ablation_methods = [m for m in ["cg_basic", "cg_warm", "cg_stab", "cg_multik", "cg_full"] if m in df["method"].unique()]
    if not ablation_methods:
        return
    grouped = df[df["method"].isin(ablation_methods)].groupby("method").agg(
        iters_mean=("num_iters", "mean"),
        runtime_mean=("runtime_total", "mean"),
        pricing_calls_mean=("num_pricing_calls", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    x = np.arange(len(grouped))
    axes[0].bar(x, grouped["iters_mean"], color=[COLORS.get(m, "#333333") for m in grouped["method"]])
    axes[1].bar(x, grouped["runtime_mean"], color=[COLORS.get(m, "#333333") for m in grouped["method"]])
    axes[2].bar(x, grouped["pricing_calls_mean"], color=[COLORS.get(m, "#333333") for m in grouped["method"]])
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(grouped["method"], rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("Iterations")
    axes[1].set_title("Runtime (s)")
    axes[2].set_title("Pricing Calls")
    fig.tight_layout()
    save_fig(fig, outdir, "Fig_Ablation_Bars")


def plot_paper(df: pd.DataFrame, outdir: str, traces_dir: str) -> None:
    summary = summarize(df, ["N"]).sort_values("N")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in PAPER_METHODS:
        g = summary[summary["method"] == method]
        if g.empty:
            continue
        ax.plot(g["N"], g["obj_mean"], label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
        ax.fill_between(g["N"], g["obj_mean"] - g["obj_std"], g["obj_mean"] + g["obj_std"], alpha=0.2, color=COLORS.get(method, None))
    ax.set_xlabel("N")
    ax.set_ylabel("Objective")
    ax.set_title("Objective vs N")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Paper_Obj")

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in PAPER_METHODS:
        g = summary[summary["method"] == method]
        if g.empty:
            continue
        ax.plot(g["N"], g["runtime_mean"], label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
        ax.fill_between(g["N"], g["runtime_mean"] - g["runtime_std"], g["runtime_mean"] + g["runtime_std"], alpha=0.2, color=COLORS.get(method, None))
    ax.set_xlabel("N")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime vs N")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_Paper_Runtime")

    plot_convergence(traces_dir, outdir, n_value=500)
    plot_cg_traces(traces_dir, outdir, n_value=500)
    plot_ablation(df, outdir)


def plot_simops(df: pd.DataFrame, outdir: str) -> None:
    if "operation_mode" not in df.columns:
        return

    summary = df.groupby(["N", "operation_mode", "method"], dropna=False).agg(
        obj_mean=("obj", "mean"),
        obj_std=("obj", "std"),
        stay_mean=("avg_stay_time", "mean"),
        stay_std=("avg_stay_time", "std"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in df["method"].unique():
        for mode in ["simops", "sequential"]:
            g = summary[(summary["method"] == method) & (summary["operation_mode"] == mode)]
            if g.empty:
                continue
            ax.plot(
                g["N"],
                g["obj_mean"],
                label=f"{method}-{mode}",
                marker=MARKERS.get(method, "o"),
                linestyle=LINESTYLES.get(mode, "-"),
                color=COLORS.get(method, None),
            )
            ax.fill_between(
                g["N"],
                g["obj_mean"] - g["obj_std"],
                g["obj_mean"] + g["obj_std"],
                alpha=0.2,
                color=COLORS.get(method, None),
            )
    ax.set_xlabel("N")
    ax.set_ylabel("Objective")
    ax.set_title("SIMOPS vs Sequential Cost")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_SIMOPS_Cost_Comparison")

    # Cost savings (%)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in df["method"].unique():
        sim = summary[(summary["method"] == method) & (summary["operation_mode"] == "simops")]
        seq = summary[(summary["method"] == method) & (summary["operation_mode"] == "sequential")]
        if sim.empty or seq.empty:
            continue
        merged = sim.merge(seq, on=["N", "method"], suffixes=("_sim", "_seq"))
        savings = (merged["obj_mean_seq"] - merged["obj_mean_sim"]) / merged["obj_mean_seq"] * 100.0
        ax.plot(merged["N"], savings, label=method, marker=MARKERS.get(method, "o"), color=COLORS.get(method, None))
    ax.set_xlabel("N")
    ax.set_ylabel("Cost Savings (%)")
    ax.set_title("SIMOPS Cost Savings")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_SIMOPS_Cost_Savings")

    # Stay time comparison
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for method in df["method"].unique():
        for mode in ["simops", "sequential"]:
            g = summary[(summary["method"] == method) & (summary["operation_mode"] == mode)]
            if g.empty:
                continue
            ax.plot(
                g["N"],
                g["stay_mean"],
                label=f"{method}-{mode}",
                marker=MARKERS.get(method, "o"),
                linestyle=LINESTYLES.get(mode, "-"),
                color=COLORS.get(method, None),
            )
            ax.fill_between(
                g["N"],
                g["stay_mean"] - g["stay_std"],
                g["stay_mean"] + g["stay_std"],
                alpha=0.2,
                color=COLORS.get(method, None),
            )
    ax.set_xlabel("N")
    ax.set_ylabel("Avg Stay Time")
    ax.set_title("SIMOPS Stay Time")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, outdir, "Fig_SIMOPS_Stay_Time")

    # Mode distribution (stacked) averaged across N/seeds
    ratio_cols = ["shore_ratio", "battery_ratio", "brown_ratio"]
    if all(c in df.columns for c in ratio_cols):
        ratio = df.groupby(["operation_mode", "method"]).agg(
            shore_ratio_mean=("shore_ratio", "mean"),
            battery_ratio_mean=("battery_ratio", "mean"),
            brown_ratio_mean=("brown_ratio", "mean"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        methods = ratio["method"].unique().tolist()
        x = np.arange(len(methods))
        width = 0.35
        for i, mode in enumerate(["simops", "sequential"]):
            g = ratio[ratio["operation_mode"] == mode]
            if g.empty:
                continue
            idx = [methods.index(m) for m in g["method"]]
            shore = g["shore_ratio_mean"].to_numpy()
            battery = g["battery_ratio_mean"].to_numpy()
            brown = g["brown_ratio_mean"].to_numpy()
            base = np.zeros_like(shore)
            offset = (i - 0.5) * width
            ax.bar(x + offset, shore, width=width, label=f"{mode}-shore", color="#4c78a8")
            base = shore
            ax.bar(x + offset, battery, width=width, bottom=base, label=f"{mode}-battery", color="#f58518")
            base = base + battery
            ax.bar(x + offset, brown, width=width, bottom=base, label=f"{mode}-brown", color="#54a24b")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("Mode Ratio")
        ax.set_title("Mode Distribution (SIMOPS vs Sequential)")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        ax.grid(True, axis="y", alpha=0.3)
        save_fig(fig, outdir, "Fig_SIMOPS_Mode_Distribution")

    # Masking distribution (simops only)
    if "avg_masking_rate" in df.columns:
        simops = df[df["operation_mode"] == "simops"]
        if not simops.empty:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            ax.hist(simops["avg_masking_rate"], bins=12, color="#4c78a8", alpha=0.8)
            ax.set_xlabel("Average Masking Rate")
            ax.set_ylabel("Count")
            ax.set_title("Masking Rate Distribution (SIMOPS)")
            ax.grid(True, axis="y", alpha=0.3)
            save_fig(fig, outdir, "Fig_SIMOPS_Masking_Distribution")

    # Masking by ship type (SIMOPS)
    type_codes = _type_codes(df)
    if type_codes:
        simops = df[df["operation_mode"] == "simops"]
        if not simops.empty:
            method_pick = "cg" if "cg" in simops["method"].unique() else simops["method"].iloc[0]
            sub = simops[simops["method"] == method_pick]
            full_vals = []
            partial_vals = []
            for t_code in type_codes:
                full_col = f"num_fully_masked_type_{t_code}"
                part_col = f"num_partially_masked_type_{t_code}"
                count_col = f"type_{t_code}_count"
                if full_col not in sub.columns or part_col not in sub.columns or count_col not in sub.columns:
                    full_vals.append(np.nan)
                    partial_vals.append(np.nan)
                    continue
                denom = sub[count_col].sum()
                full_vals.append(sub[full_col].sum() / denom if denom > 0 else np.nan)
                partial_vals.append(sub[part_col].sum() / denom if denom > 0 else np.nan)

            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            x = np.arange(len(type_codes))
            ax.bar(x, full_vals, label="fully masked", color="#4c78a8")
            base = np.array(full_vals, dtype=float)
            ax.bar(x, partial_vals, bottom=base, label="partially masked", color="#f58518")
            ax.set_xticks(x)
            ax.set_xticklabels(type_codes)
            ax.set_ylabel("Share")
            ax.set_title(f"Masking by Type ({method_pick})")
            ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
            ax.grid(True, axis="y", alpha=0.3)
            save_fig(fig, outdir, "Fig_SIMOPS_Masking_By_Type")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make plots from results.")
    parser.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    parser.add_argument("--outdir", required=True, help="Output directory for figures")
    parser.add_argument("--experiment", default="", help="main/mechanism/sensitivity/ablation/paper; inferred if empty")
    parser.add_argument("--logy", action="store_true", help="Use log-scale for runtime plots")
    parser.add_argument("--traces_dir", default="results/traces", help="Directory with CG trace CSVs")
    args = parser.parse_args()

    set_style()
    df = pd.read_csv(args.inp)

    experiment = args.experiment
    if not experiment:
        if "simops" in os.path.basename(args.inp):
            experiment = "simops"
        elif "mechanism" in os.path.basename(args.inp):
            experiment = "mechanism"
        elif "scenario" in os.path.basename(args.inp):
            experiment = "scenario"
        elif "sensitivity" in os.path.basename(args.inp):
            experiment = "sensitivity"
        elif "ablation" in os.path.basename(args.inp):
            experiment = "ablation"
        else:
            experiment = "main"

    if experiment == "main":
        plot_main(df, args.outdir, logy=args.logy)
    elif experiment == "mechanism":
        plot_mechanism(df, args.outdir)
    elif experiment == "scenario":
        plot_scenario(df, args.outdir)
    elif experiment == "sensitivity":
        plot_sensitivity(df, args.outdir)
    elif experiment in ("ablation", "paper"):
        plot_paper(df, args.outdir, args.traces_dir)
    elif experiment == "simops":
        plot_simops(df, args.outdir)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    if _type_codes(df):
        plot_type_breakdown(df, args.outdir)


if __name__ == "__main__":
    main()

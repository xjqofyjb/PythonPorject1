"""Figure 3: main comparative results."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import DOUBLE_COLUMN_QUAD, METHOD_COLORS, METHOD_LABELS, METHOD_LINESTYLES, METHOD_MARKERS, METHOD_ORDER, REFERENCE_COLOR, REFERENCE_LINESTYLE
from .utils import ExportedFigure, add_panel_labels, apply_common_axis_format, apply_cost_formatter, compact_legend, export_figure, load_csv, set_x_axis_label, start_figure, summarize


METHODS = METHOD_ORDER


def build() -> ExportedFigure:
    start_figure()
    df = load_csv("results/results_main_rigorous.csv")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COLUMN_QUAD.width, DOUBLE_COLUMN_QUAD.height))
    flat = axes.ravel()

    summary_obj = summarize(df, ["N", "method"], "obj")
    summary_runtime = summarize(df, ["N", "method"], "runtime_total")
    ref = summary_obj[summary_obj["method"] == "cg"][["N", "obj_mean"]].rename(columns={"obj_mean": "cg_obj"})
    rel = summary_obj.merge(ref, on="N", how="left")
    rel["gap_to_cg_pct"] = (rel["obj_mean"] - rel["cg_obj"]) / rel["cg_obj"] * 100.0

    for method in METHODS:
        block_obj = summary_obj[summary_obj["method"] == method].sort_values("N")
        if block_obj.empty:
            continue
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method] or None
        line = METHOD_LINESTYLES[method]

        flat[0].plot(block_obj["N"], block_obj["obj_mean"], color=color, marker=marker, linestyle=line, label=METHOD_LABELS[method])
        flat[0].fill_between(
            block_obj["N"],
            block_obj["obj_mean"] - block_obj["obj_ci"],
            block_obj["obj_mean"] + block_obj["obj_ci"],
            color=color,
            alpha=0.14,
        )
        rel_block = rel[rel["method"] == method].sort_values("N")
        if method != "cg" and not rel_block.empty:
            flat[1].plot(rel_block["N"], rel_block["gap_to_cg_pct"], color=color, marker=marker, linestyle=line)
        run_block = summary_runtime[summary_runtime["method"] == method].sort_values("N")
        if not run_block.empty:
            flat[2].plot(run_block["N"], run_block["runtime_total_mean"], color=color, marker=marker, linestyle=line)
            flat[2].fill_between(
                run_block["N"],
                np.maximum(run_block["runtime_total_mean"] - run_block["runtime_total_ci"], 1e-4),
                run_block["runtime_total_mean"] + run_block["runtime_total_ci"],
                color=color,
                alpha=0.14,
            )

    cg_gap = summarize(df[df["method"] == "cg"], ["N"], "gap_pct").sort_values("N")
    flat[3].bar(
        cg_gap["N"],
        cg_gap["gap_pct_mean"],
        width=30,
        color=METHOD_COLORS["cg"],
        edgecolor=METHOD_COLORS["cg"],
        alpha=0.8,
        yerr=cg_gap["gap_pct_ci"],
        error_kw={"elinewidth": 0.8, "ecolor": "#666666", "capsize": 2},
    )
    flat[3].set_facecolor("#FCFCFC")

    flat[0].set_ylabel("Total cost ($)")
    flat[1].set_ylabel("Relative gap to CG+IR (%)")
    flat[1].axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    set_x_axis_label(flat[2], "Number of ships $N$")
    flat[2].set_ylabel("Wall-clock runtime (s)")
    flat[2].set_yscale("log")
    set_x_axis_label(flat[3], "Number of ships $N$")
    flat[3].set_ylabel("CG integrality gap (%)")
    for ax in flat:
        apply_common_axis_format(ax)
    apply_cost_formatter(flat[0])

    handles, labels = flat[0].get_legend_handles_labels()
    compact_legend(fig, handles, labels, preferred="top")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.16, hspace=0.42, wspace=0.32)
    add_panel_labels(fig, flat, pad=[0.010, 0.010, 0.012, 0.012])
    return export_figure(fig, "Fig3_main_results")


if __name__ == "__main__":
    build()

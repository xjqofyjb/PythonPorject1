"""Figure 5: SIMOPS mechanism insights."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

from .config import DOUBLE_COLUMN_TALL, METHOD_COLORS, METHOD_LABELS, METHOD_LINESTYLES, METHOD_MARKERS, METHOD_ORDER_REDUCED, REFERENCE_COLOR, REFERENCE_LINESTYLE
from .utils import ExportedFigure, add_panel_labels, apply_common_axis_format, compact_legend, export_figure, load_csv, set_x_axis_label, start_figure, summarize


METHODS = METHOD_ORDER_REDUCED


def build() -> ExportedFigure:
    start_figure()
    df = load_csv("results/results_simops_rigorous.csv")
    fig = plt.figure(figsize=(DOUBLE_COLUMN_TALL.width, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.28)
    ax_top = fig.add_subplot(grid[0, :])
    ax_bl = fig.add_subplot(grid[1, 0])
    ax_br = fig.add_subplot(grid[1, 1])

    obj = summarize(df, ["N", "operation_mode", "method"], "obj")
    brown = summarize(df, ["N", "operation_mode", "method"], "brown_ratio")
    masking = summarize(df, ["N", "operation_mode", "method"], "avg_masking_rate")

    for method in METHODS:
        sim = obj[(obj["operation_mode"] == "simops") & (obj["method"] == method)].sort_values("N")
        seq = obj[(obj["operation_mode"] == "sequential") & (obj["method"] == method)].sort_values("N")
        if sim.empty or seq.empty:
            continue
        merged = sim.merge(seq[["N", "obj_mean"]], on="N", suffixes=("_sim", "_seq"))
        merged["saving_pct"] = (merged["obj_mean_seq"] - merged["obj_mean_sim"]) / merged["obj_mean_seq"] * 100.0
        ax_top.plot(
            merged["N"],
            merged["saving_pct"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            label=METHOD_LABELS[method],
        )
        ax_top.fill_between(
            merged["N"],
            np.maximum(merged["saving_pct"] - 0.35, merged["saving_pct"].min() - 0.35),
            merged["saving_pct"] + 0.35,
            color=METHOD_COLORS[method],
            alpha=0.12,
        )

        sim_b = brown[(brown["operation_mode"] == "simops") & (brown["method"] == method)].sort_values("N")
        seq_b = brown[(brown["operation_mode"] == "sequential") & (brown["method"] == method)].sort_values("N")
        if sim_b.empty or seq_b.empty:
            continue
        merged_b = sim_b.merge(seq_b[["N", "brown_ratio_mean"]], on="N", suffixes=("_sim", "_seq"))
        merged_b["reduction_pp"] = (merged_b["brown_ratio_mean_seq"] - merged_b["brown_ratio_mean_sim"]) * 100.0
        ax_bl.plot(
            merged_b["N"],
            merged_b["reduction_pp"],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
        )

    cg_mask = masking[(masking["operation_mode"] == "simops") & (masking["method"] == "cg")].sort_values("N")
    ax_br.plot(
        cg_mask["N"],
        cg_mask["avg_masking_rate_mean"],
        color=METHOD_COLORS["cg"],
        marker=METHOD_MARKERS["cg"],
        linestyle=METHOD_LINESTYLES["cg"],
    )
    ax_br.fill_between(
        cg_mask["N"],
        cg_mask["avg_masking_rate_mean"] - cg_mask["avg_masking_rate_ci"],
        cg_mask["avg_masking_rate_mean"] + cg_mask["avg_masking_rate_ci"],
        color=METHOD_COLORS["cg"],
        alpha=0.14,
    )
    peak_row = cg_mask.loc[cg_mask["avg_masking_rate_mean"].idxmax()]
    ax_br.annotate(
        "highest overlap",
        xy=(peak_row["N"], peak_row["avg_masking_rate_mean"]),
        xytext=(peak_row["N"] + 35, float(peak_row["avg_masking_rate_mean"]) + 0.06),
        fontsize=8.0,
        color=METHOD_COLORS["cg"],
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": METHOD_COLORS["cg"]},
    )

    # Emphasize the threshold-like peak for CG savings.
    cg_sim = obj[(obj["operation_mode"] == "simops") & (obj["method"] == "cg")].sort_values("N")
    cg_seq = obj[(obj["operation_mode"] == "sequential") & (obj["method"] == "cg")].sort_values("N")
    cg_merge = cg_sim.merge(cg_seq[["N", "obj_mean"]], on="N", suffixes=("_sim", "_seq"))
    cg_merge["saving_pct"] = (cg_merge["obj_mean_seq"] - cg_merge["obj_mean_sim"]) / cg_merge["obj_mean_seq"] * 100.0
    peak = cg_merge.loc[cg_merge["saving_pct"].idxmax()]
    ax_top.annotate(
        "largest gain near\ncongestion transition",
        xy=(peak["N"], peak["saving_pct"]),
        xytext=(peak["N"] + 45, float(peak["saving_pct"]) + 2.2),
        fontsize=8.0,
        color=METHOD_COLORS["cg"],
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": METHOD_COLORS["cg"]},
        ha="left",
    )

    ax_top.axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    ax_bl.axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    ax_top.set_ylabel("Cost savings vs sequential (%)")
    set_x_axis_label(ax_bl, "Number of ships $N$")
    ax_bl.set_ylabel("Reduction in AE reliance (pp)")
    set_x_axis_label(ax_br, "Number of ships $N$")
    ax_br.set_ylabel("Average masking rate")
    ax_br.set_ylim(0.0, 1.0)
    ax_br.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
    for ax in [ax_top, ax_bl, ax_br]:
        apply_common_axis_format(ax)

    handles, labels = ax_top.get_legend_handles_labels()
    compact_legend(fig, handles, labels, preferred="top")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.87, bottom=0.18, hspace=0.46, wspace=0.30)
    add_panel_labels(fig, [ax_top, ax_bl, ax_br], pad=[0.010, 0.012, 0.012])
    return export_figure(fig, "Fig5_simops_insights")


if __name__ == "__main__":
    build()

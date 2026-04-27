"""Figure 4: scenarios and mechanisms."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import DOUBLE_COLUMN, MECHANISM_LABELS, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, REFERENCE_COLOR, REFERENCE_LINESTYLE, SCENARIO_LABELS
from .utils import ExportedFigure, add_panel_labels, apply_common_axis_format, apply_cost_formatter, compact_legend, export_figure, load_csv, start_figure, summarize


METHODS = METHOD_ORDER


def build() -> ExportedFigure:
    start_figure()
    scenario_df = load_csv("results/results_scenario_rigorous.csv")
    mechanism_df = load_csv("results/results_mechanism_rigorous.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN.width, 3.65))

    scen = summarize(scenario_df, ["scenario", "method"], "obj")
    scenarios = [s for s in ["U", "P", "L"] if s in scen["scenario"].unique()]
    methods_available = [method for method in METHODS if method in scen["method"].unique()]
    x = np.arange(len(scenarios))
    width = min(0.82 / max(len(methods_available), 1), 0.10)
    for idx, method in enumerate(methods_available):
        block = scen[scen["method"] == method].set_index("scenario").reindex(scenarios).reset_index()
        if block.empty:
            continue
        axes[0].bar(
            x + (idx - (len(methods_available) - 1) / 2) * width,
            block["obj_mean"],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            yerr=block["obj_ci"],
            error_kw={"elinewidth": 0.7, "ecolor": "#666666", "capsize": 2},
            label=METHOD_LABELS[method],
            alpha=0.95,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_ylabel("Total cost ($)")

    hybrid = mechanism_df[mechanism_df["mechanism"] == "hybrid"][["seed", "method", "obj"]].rename(columns={"obj": "hybrid_obj"})
    rel = mechanism_df.merge(hybrid, on=["seed", "method"], how="left")
    rel["penalty_pct"] = (rel["obj"] - rel["hybrid_obj"]) / rel["hybrid_obj"] * 100.0
    mech = summarize(rel, ["mechanism", "method"], "penalty_pct")
    mechs = [m for m in ["hybrid", "battery_only", "shore_only"] if m in mech["mechanism"].unique()]
    me_methods = [method for method in METHODS if method in mech["method"].unique()]
    x2 = np.arange(len(mechs))
    width2 = min(0.82 / max(len(me_methods), 1), 0.10)
    for idx, method in enumerate(me_methods):
        block = mech[mech["method"] == method].set_index("mechanism").reindex(mechs).reset_index()
        if block.empty:
            continue
        axes[1].bar(
            x2 + (idx - (len(me_methods) - 1) / 2) * width2,
            block["penalty_pct_mean"],
            width=width2,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            yerr=block["penalty_pct_ci"],
            error_kw={"elinewidth": 0.7, "ecolor": "#666666", "capsize": 2},
            alpha=0.95,
        )
    axes[1].axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([MECHANISM_LABELS[m] for m in mechs])
    axes[1].set_ylabel("Cost penalty vs hybrid (%)")
    for ax in axes:
        apply_common_axis_format(ax)
    apply_cost_formatter(axes[0])

    handles, labels = axes[0].get_legend_handles_labels()
    compact_legend(fig, handles, labels, preferred="top")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.16, wspace=0.26)
    add_panel_labels(fig, axes, pad=0.012)
    return export_figure(fig, "Fig4_scenarios_mechanisms")


if __name__ == "__main__":
    build()

"""Figure 6: sensitivity analysis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .config import DOUBLE_COLUMN, FILL_COLORS, METHOD_COLORS
from .utils import ExportedFigure, add_panel_labels, apply_common_axis_format, apply_cost_formatter, compact_legend, export_figure, load_csv, set_x_axis_label, start_figure, style_secondary_axis, summarize


PARAM_SPECS = [
    ("battery_cost", "Battery cost\n($/kWh)"),
    ("shore_cap", "Shore berths"),
    ("deadline_tightness", "Deadline\nfactor"),
]


def build() -> ExportedFigure:
    start_figure()
    df = load_csv("results/results_sensitivity_rigorous.csv")
    cg = df[df["method"] == "cg"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN.width, 3.3))

    left_limits: list[tuple[float, float]] = []
    right_axes: list[plt.Axes] = []
    for idx, (ax, (param_name, xlabel)) in enumerate(zip(axes, PARAM_SPECS)):
        block = cg[cg["param_name"] == param_name].copy()
        obj = summarize(block, ["param_value"], "obj").sort_values("param_value")
        brown = summarize(block, ["param_value"], "brown_ratio").sort_values("param_value")

        ax.plot(obj["param_value"], obj["obj_mean"], color=METHOD_COLORS["cg"], marker="o", label="Total cost")
        ax.fill_between(
            obj["param_value"],
            obj["obj_mean"] - obj["obj_ci"],
            obj["obj_mean"] + obj["obj_ci"],
            color=FILL_COLORS["blue"],
            alpha=0.18,
        )
        set_x_axis_label(ax, xlabel)
        if idx == 0:
            ax.set_ylabel("Total cost ($)")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        apply_common_axis_format(ax)
        apply_cost_formatter(ax)

        twin = ax.twinx()
        twin.plot(
            brown["param_value"],
            brown["brown_ratio_mean"] * 100.0,
            color="#E5988A",
            marker="s",
            linestyle="--",
            linewidth=1.3,
            label="AE share",
        )
        if brown["brown_ratio_ci"].notna().any():
            twin.fill_between(
                brown["param_value"],
                np.maximum((brown["brown_ratio_mean"] - brown["brown_ratio_ci"]) * 100.0, 0.0),
                np.minimum((brown["brown_ratio_mean"] + brown["brown_ratio_ci"]) * 100.0, 100.0),
                color=FILL_COLORS["salmon"],
                alpha=0.16,
            )
        twin.set_ylim(0.0, 100.0)
        if idx == len(axes) - 1:
            style_secondary_axis(twin, "#E5988A", "AE share (%)")
        else:
            twin.tick_params(axis="y", colors="#E5988A", labelsize=8.3, labelright=False)
            twin.spines["right"].set_visible(True)
            twin.spines["right"].set_color("#E5988A")
            twin.grid(False)
            twin.minorticks_off()

        left_limits.append((float((obj["obj_mean"] - obj["obj_ci"]).min()), float((obj["obj_mean"] + obj["obj_ci"]).max())))
        right_axes.append(twin)

    span_low = min(low for low, _ in left_limits)
    span_high = max(high for _, high in left_limits)
    padding = (span_high - span_low) * 0.06
    for ax in axes:
        ax.set_ylim(span_low - padding, span_high + padding)

    legend_handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS["cg"], marker="o", linestyle="-", label="Total cost"),
        plt.Line2D([0], [0], color="#E5988A", marker="s", linestyle="--", label="AE share"),
    ]
    compact_legend(fig, legend_handles, [h.get_label() for h in legend_handles], preferred="top")
    fig.subplots_adjust(left=0.09, right=0.95, top=0.83, bottom=0.32, wspace=0.20)
    add_panel_labels(fig, axes, pad=0.012)
    return export_figure(fig, "Fig6_sensitivity")


if __name__ == "__main__":
    build()

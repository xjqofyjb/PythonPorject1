"""Figure 8: robustness under arrival perturbation."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .config import DOUBLE_COLUMN, FILL_COLORS, MODE_COLORS
from .utils import ExportedFigure, add_figure_legend, add_panel_labels, apply_common_axis_format, apply_cost_formatter, export_figure, load_csv, set_x_axis_label, start_figure, style_secondary_axis


def build() -> ExportedFigure:
    start_figure()
    summary = load_csv("results/robustness_fixed_deadline_summary.csv")
    detail = load_csv("results/robustness_fixed_deadline.csv")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN.width, 3.25))
    ax_cost, ax_modes = axes

    x = summary["delta"]
    ax_cost.plot(x, summary["mean_cost"], color="#4C78A8", marker="o", label="Total cost")
    ax_cost.fill_between(
        x,
        summary["mean_cost"] - summary["std_cost"],
        summary["mean_cost"] + summary["std_cost"],
        color=FILL_COLORS["blue"],
        alpha=0.18,
    )
    ax_cost.plot(x, summary["mean_energy_cost"], color=MODE_COLORS["SP"], marker="s", label="Energy cost")
    set_x_axis_label(ax_cost, "Perturbation\namplitude $\\Delta$ (h)")
    ax_cost.set_ylabel("Total / energy cost ($)")
    apply_common_axis_format(ax_cost)
    apply_cost_formatter(ax_cost)
    ax_cost.set_ylim(175000, 192500)

    ax_delay = ax_cost.twinx()
    ax_delay.errorbar(
        x,
        summary["mean_delay_cost"],
        yerr=summary["std_delay_cost"],
        color=MODE_COLORS["AE"],
        marker="^",
        linestyle="--",
        linewidth=1.2,
        capsize=2.5,
        label="Delay cost",
    )
    ax_delay.set_ylim(0.0, max(160.0, float((summary["mean_delay_cost"] + summary["std_delay_cost"]).max()) * 1.2))
    style_secondary_axis(ax_delay, MODE_COLORS["AE"], "Delay cost ($)")

    grouped = detail.groupby("delta", as_index=False).agg(
        sp_share_mean=("sp_share", "mean"),
        sp_share_std=("sp_share", "std"),
        bs_share_mean=("bs_share", "mean"),
        bs_share_std=("bs_share", "std"),
        ae_share_mean=("ae_share", "mean"),
        ae_share_std=("ae_share", "std"),
    )
    for prefix, color, marker, label in [
        ("sp", MODE_COLORS["SP"], "o", "Shore power"),
        ("bs", MODE_COLORS["BS"], "s", "Battery swap"),
        ("ae", MODE_COLORS["AE"], "^", "AE"),
    ]:
        mean = grouped[f"{prefix}_share_mean"] * 100.0
        std = grouped[f"{prefix}_share_std"].fillna(0.0) * 100.0
        ax_modes.plot(grouped["delta"], mean, color=color, marker=marker, label=label)
        ax_modes.fill_between(grouped["delta"], mean - std, mean + std, color=color, alpha=0.08)
    set_x_axis_label(ax_modes, "Perturbation\namplitude $\\Delta$ (h)")
    ax_modes.set_ylabel("Service-mode share (%)")
    ax_modes.set_ylim(0.0, 100.0)
    apply_common_axis_format(ax_modes)

    ax_cost.text(1.10, 188300, "system cost remains\nwithin a narrow band", fontsize=7.9, color="#4C78A8")
    ax_modes.text(1.12, 83.0, "green service mix is preserved", fontsize=7.9, color=MODE_COLORS["BS"])

    handles1, labels1 = ax_cost.get_legend_handles_labels()
    handles_delay, labels_delay = ax_delay.get_legend_handles_labels()
    handles2, labels2 = ax_modes.get_legend_handles_labels()
    add_figure_legend(fig, handles1 + handles_delay + handles2, labels1 + labels_delay + labels2, ncol=3)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.31, wspace=0.34)
    add_panel_labels(fig, axes, pad=0.012)
    return export_figure(fig, "Fig8_robustness")


if __name__ == "__main__":
    build()

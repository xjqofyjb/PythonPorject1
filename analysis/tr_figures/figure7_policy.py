"""Figure 7: dual-axis carbon-price policy figure."""
from __future__ import annotations

import matplotlib.pyplot as plt

from .config import DOUBLE_COLUMN, COLORS_DUAL, FILL_COLORS, REFERENCE_COLOR
from .utils import ExportedFigure, add_figure_legend, apply_common_axis_format, apply_cost_formatter, export_figure, load_csv, make_style_handles, set_x_axis_label, start_figure, style_secondary_axis


def build() -> ExportedFigure:
    start_figure()
    df = load_csv("results/carbon_price_dual_summary.csv")
    fig, ax_cost = plt.subplots(figsize=(DOUBLE_COLUMN.width, 3.2))
    ax_ae = ax_cost.twinx()

    adequate = df[df["capacity_config"] == "adequate"].sort_values("carbon_price")
    constrained = df[df["capacity_config"] == "constrained"].sort_values("carbon_price")

    ax_cost.plot(
        adequate["carbon_price"],
        adequate["mean_cost"],
        color=COLORS_DUAL["adequate_cost"],
        marker="o",
        linestyle="-",
        label="Total cost",
    )
    ax_cost.plot(
        constrained["carbon_price"],
        constrained["mean_cost"],
        color=COLORS_DUAL["constrained_cost"],
        marker="o",
        linestyle="--",
    )
    ax_cost.fill_between(
        constrained["carbon_price"],
        constrained["mean_cost"] - constrained["std_cost"],
        constrained["mean_cost"] + constrained["std_cost"],
        color=FILL_COLORS["blue"],
        alpha=0.18,
    )

    ax_ae.plot(
        adequate["carbon_price"],
        adequate["mean_ae_share"] * 100.0,
        color=COLORS_DUAL["adequate_ae"],
        marker="s",
        linestyle="-",
        label="AE share",
    )
    ax_ae.plot(
        constrained["carbon_price"],
        constrained["mean_ae_share"] * 100.0,
        color=COLORS_DUAL["constrained_ae"],
        marker="s",
        linestyle="--",
    )
    ax_ae.fill_between(
        constrained["carbon_price"],
        (constrained["mean_ae_share"] - constrained["std_ae_share"]) * 100.0,
        (constrained["mean_ae_share"] + constrained["std_ae_share"]) * 100.0,
        color=FILL_COLORS["salmon"],
        alpha=0.16,
    )

    reference_specs = [
        (100, "EU ETS ($100)"),
        (200, "Baseline ($200)"),
        (380, "IMO 2027 RU ($380)"),
    ]
    for xpos, label in reference_specs:
        ax_cost.axvline(xpos, color=REFERENCE_COLOR, linestyle=":", linewidth=0.85)
        ax_cost.text(
            xpos,
            1.02,
            label,
            transform=ax_cost.get_xaxis_transform(),
            fontsize=7.9,
            ha="center",
            va="bottom",
            color="#666666",
        )

    set_x_axis_label(ax_cost, "Carbon price ($ per t CO2)")
    ax_cost.set_ylabel("Total cost ($)", color=COLORS_DUAL["adequate_cost"])
    ax_cost.tick_params(axis="y", colors=COLORS_DUAL["adequate_cost"])
    ax_cost.spines["left"].set_color(COLORS_DUAL["adequate_cost"])
    ax_cost.set_xlim(float(df["carbon_price"].min()) - 8, float(df["carbon_price"].max()) + 8)
    ax_cost.set_ylim(175000, 360000)
    apply_common_axis_format(ax_cost)
    apply_cost_formatter(ax_cost)

    ax_ae.set_ylim(-1.0, 52.0)
    style_secondary_axis(ax_ae, COLORS_DUAL["adequate_ae"], "AE mode share (%)")

    ax_cost.text(257, 188200, "adequate capacity remains flat", fontsize=7.8, color=COLORS_DUAL["adequate_cost"])
    ax_ae.text(248, 45.2, "constrained capacity:\nsharp cost rise, limited AE decline", fontsize=7.8, color=COLORS_DUAL["constrained_ae"])

    legend_handles = make_style_handles(
        metric_colors={"Total cost": COLORS_DUAL["adequate_cost"], "AE share": COLORS_DUAL["adequate_ae"]},
        style_labels={"Adequate capacity": "-", "Constrained capacity": "--"},
    )
    add_figure_legend(fig, legend_handles, [handle.get_label() for handle in legend_handles], ncol=4)
    fig.subplots_adjust(left=0.11, right=0.89, top=0.84, bottom=0.19)
    return export_figure(fig, "Fig7_policy_carbon")


if __name__ == "__main__":
    build()

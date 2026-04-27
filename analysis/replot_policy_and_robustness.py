"""Redraw Fig. 7-8 with a cleaner TRD-like visual language."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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

COLORS_DUAL = {
    "adequate_cost": "#2166AC",
    "adequate_ae": "#D6604D",
    "constrained_cost": "#2166AC",
    "constrained_ae": "#D6604D",
}

COLORS_MODE = {
    "SP": "#2166AC",
    "BS": "#1B7837",
    "AE": "#D6604D",
}

FIG_DOUBLE = (7.4, 3.7)
GRAY = "#7A7A7A"

CARBON_PRICES = np.array([100, 140, 200, 260, 320, 380], dtype=float)
ADEQUATE_COST_MEAN = np.array([183337.84] * 6, dtype=float)
CONSTRAINED_COST_MEAN = np.array([226376.38, 243131.69, 268150.13, 292944.79, 317371.35, 341432.08], dtype=float)
CONSTRAINED_COST_STD = np.array([5111.06, 5654.37, 6474.68, 7293.34, 8167.83, 9077.59], dtype=float)
ADEQUATE_AE_MEAN = np.array([0.0] * 6, dtype=float)
CONSTRAINED_AE_MEAN = np.array([46.2, 45.9, 45.6, 45.0, 44.4, 43.6], dtype=float)
CONSTRAINED_AE_STD = np.array([1.3, 1.4, 1.3, 1.6, 1.3, 1.1], dtype=float)

DELTAS = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
COST_CHANGE_PCT = np.array([0.0, -0.036, -0.122, -0.137, -0.122], dtype=float)
SP_SHARE = np.array([14.9, 14.9, 14.9, 14.9, 14.7], dtype=float)
BS_SHARE = np.array([85.1, 85.1, 85.1, 85.1, 85.3], dtype=float)
AE_SHARE = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def add_panel_labels_below(fig: plt.Figure, axes: list[plt.Axes], labels: list[str]) -> None:
    for ax, label in zip(axes, labels):
        bbox = ax.get_position()
        x_center = (bbox.x0 + bbox.x1) / 2
        fig.text(x_center, bbox.y0 - 0.065, label, ha="center", va="top", fontsize=10)


def draw_carbon_price_dual_layer(outdir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE, sharex=True)

    for ax in (ax1, ax2):
        for x_anchor in (100, 200, 380):
            ax.axvline(x_anchor, color=GRAY, alpha=0.4, linewidth=0.8, linestyle=":")

    ax1.plot(CARBON_PRICES, ADEQUATE_COST_MEAN, color=COLORS_DUAL["adequate_cost"], marker="o", linestyle="-")
    ax1.plot(CARBON_PRICES, CONSTRAINED_COST_MEAN, color=COLORS_DUAL["constrained_cost"], marker="s", linestyle="--")
    ax1.fill_between(
        CARBON_PRICES,
        CONSTRAINED_COST_MEAN - CONSTRAINED_COST_STD,
        CONSTRAINED_COST_MEAN + CONSTRAINED_COST_STD,
        color=COLORS_DUAL["constrained_cost"],
        alpha=0.12,
    )
    ax1.set_xlabel(r"Carbon price ($/tonne CO$_2$)")
    ax1.set_ylabel("Total cost ($)")
    ax1.set_ylim(170000, 360000)
    for x_anchor, text in ((100, "EU ETS"), (200, "Baseline"), (380, "IMO 2027")):
        ax1.text(
            x_anchor,
            1.03,
            text,
            transform=ax1.get_xaxis_transform(),
            fontsize=8.8,
            color=GRAY,
            ha="center",
            va="bottom",
            clip_on=False,
        )

    ax2.plot(CARBON_PRICES, ADEQUATE_AE_MEAN, color=COLORS_DUAL["adequate_ae"], marker="o", linestyle="-")
    ax2.plot(CARBON_PRICES, CONSTRAINED_AE_MEAN, color=COLORS_DUAL["constrained_ae"], marker="s", linestyle="--")
    ax2.fill_between(
        CARBON_PRICES,
        CONSTRAINED_AE_MEAN - CONSTRAINED_AE_STD,
        CONSTRAINED_AE_MEAN + CONSTRAINED_AE_STD,
        color=COLORS_DUAL["constrained_ae"],
        alpha=0.12,
    )
    ax2.set_xlabel(r"Carbon price ($/tonne CO$_2$)")
    ax2.set_ylabel("AE mode share (%)")
    ax2.set_ylim(-2, 50)
    ax2.spines["right"].set_visible(True)
    ax2.text(126, 2.2, "AE = 0% (all green)", fontsize=8.2, color=COLORS_DUAL["adequate_ae"], ha="left", va="bottom")
    ax2.text(248, 41.0, "46.2% to 43.6%", fontsize=8.2, color=COLORS_DUAL["constrained_ae"], ha="left", va="bottom")

    handles = [
        Line2D([0], [0], color=COLORS_DUAL["adequate_cost"], marker="o", linestyle="-", label="Adequate capacity (2SP+2BS)"),
        Line2D([0], [0], color=COLORS_DUAL["constrained_cost"], marker="s", linestyle="--", label="Constrained capacity (1SP+1BS)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.015))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.84, bottom=0.33, wspace=0.30)
    add_panel_labels_below(fig, [ax1, ax2], ["(a)", "(b)"])

    ensure_dir(outdir)
    pdf_path = outdir / "Fig_Carbon_Price_DualLayer.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def draw_robustness_fixed_deadline(outdir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    ax1.axhspan(-2.3, 2.3, color=GRAY, alpha=0.08)
    ax1.axhline(0.0, color=GRAY, linestyle="--", alpha=0.5, linewidth=1.0)
    ax1.plot(DELTAS, COST_CHANGE_PCT, color=COLORS_MODE["SP"], marker="o", linestyle="-")
    for x, y in zip(DELTAS, COST_CHANGE_PCT):
        ax1.text(x, y + 0.017, f"{y:.3f}%", color=COLORS_MODE["SP"], fontsize=8.0, ha="center", va="bottom")
    ax1.annotate(
        "Delta = 2h: -0.12%",
        xy=(2.0, -0.122),
        xytext=(1.40, -0.21),
        fontsize=8.2,
        color=COLORS_MODE["SP"],
        arrowprops={"arrowstyle": "-", "color": COLORS_MODE["SP"], "lw": 0.8},
        ha="left",
        va="center",
    )
    ax1.text(0.5, 0.90, "Cross-seed std approx. +/-2.3%", transform=ax1.transAxes, fontsize=8.1, color=GRAY, ha="center")
    ax1.set_xlabel(r"Perturbation amplitude $\Delta$ (hours)")
    ax1.set_ylabel("Cost change relative to baseline (%)")
    ax1.set_ylim(-0.4, 0.2)

    ax2.plot(DELTAS, SP_SHARE, color=COLORS_MODE["SP"], marker="o", linestyle="-", label="SP")
    ax2.plot(DELTAS, BS_SHARE, color=COLORS_MODE["BS"], marker="o", linestyle="-", label="BS")
    ax2.plot(DELTAS, AE_SHARE, color=COLORS_MODE["AE"], marker="o", linestyle="-", label="AE")
    ax2.text(0.60, 87.5, "BS nearly unchanged", fontsize=8.1, color=COLORS_MODE["BS"], ha="left")
    ax2.text(0.60, 17.0, "SP nearly unchanged", fontsize=8.1, color=COLORS_MODE["SP"], ha="left")
    ax2.text(0.60, 2.6, "AE = 0% throughout", fontsize=8.1, color=COLORS_MODE["AE"], ha="left")
    ax2.set_xlabel(r"Perturbation amplitude $\Delta$ (hours)")
    ax2.set_ylabel("Mode share (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right")

    fig.text(0.26, 0.035, "Note: delay cost < 0.02% of total in all cases", fontsize=8, style="italic", ha="center")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.30, wspace=0.32)
    add_panel_labels_below(fig, [ax1, ax2], ["(a)", "(b)"])

    ensure_dir(outdir)
    pdf_path = outdir / "Fig_Robustness_FixedDeadline.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    outdir = Path("figs/paper")
    carbon_path = draw_carbon_price_dual_layer(outdir)
    robustness_path = draw_robustness_fixed_deadline(outdir)
    print(f"Saved: {carbon_path}")
    print(f"Saved: {robustness_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tr_figures.config import METHOD_COLORS, MODE_COLORS, REFERENCE_COLOR, REFERENCE_LINESTYLE, SINGLE_COLUMN
from analysis.tr_figures.utils import apply_common_axis_format, set_x_axis_label, start_figure


CONFIG_COLORS = {
    "loose": METHOD_COLORS["cg"],
    "tight": "#B85C38",
}
CONFIG_LABELS = {
    "loose": "Loose deadline",
    "tight": "Tight deadline",
}
MODE_STYLE = {
    "loose": "-",
    "tight": "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dual-configuration Fig. 8 for Experiment 2.")
    parser.add_argument("--comparison-csv", required=True, help="Wide aggregated comparison CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for figure files.")
    return parser.parse_args()


def _plot_cost_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["delta"].to_numpy(dtype=float)
    ax_delay = ax.twinx()

    for config in ["loose", "tight"]:
        total_mean = summary[f"{config}_objective_mean"].to_numpy(dtype=float)
        total_std = summary[f"{config}_objective_std"].to_numpy(dtype=float)
        energy_mean = summary[f"{config}_energy_cost_mean"].to_numpy(dtype=float)
        energy_std = summary[f"{config}_energy_cost_std"].to_numpy(dtype=float)
        delay_mean = summary[f"{config}_delay_cost_mean"].to_numpy(dtype=float)
        delay_std = summary[f"{config}_delay_cost_std"].to_numpy(dtype=float)

        color = CONFIG_COLORS[config]
        style = MODE_STYLE[config]
        label_prefix = CONFIG_LABELS[config]

        ax.errorbar(
            x,
            total_mean,
            yerr=total_std,
            color=color,
            marker="o",
            linestyle=style,
            linewidth=1.7,
            capsize=2.5,
            label=f"{label_prefix}: total",
        )
        ax.errorbar(
            x,
            energy_mean,
            yerr=energy_std,
            color=color,
            marker="s",
            linestyle=style,
            linewidth=1.4,
            alpha=0.80,
            capsize=2.5,
            label=f"{label_prefix}: energy",
        )
        ax_delay.errorbar(
            x,
            delay_mean,
            yerr=delay_std,
            color=color,
            marker="^",
            linestyle=style,
            linewidth=1.4,
            alpha=0.90,
            capsize=2.5,
            label=f"{label_prefix}: delay",
        )

    set_x_axis_label(ax, "Perturbation amplitude $\\Delta$ (h)")
    ax.set_ylabel("Total / energy cost ($)")
    ax_delay.set_ylabel("Delay cost ($)")
    apply_common_axis_format(ax)
    ax_delay.grid(False)

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax_delay.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", fontsize=7.0, ncol=2)


def _plot_mode_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = summary["delta"].to_numpy(dtype=float)
    mode_defs = [
        ("sp", "SP share", MODE_COLORS["SP"], "o"),
        ("bs", "BS share", MODE_COLORS["BS"], "s"),
        ("ae", "AE share", MODE_COLORS["AE"], "^"),
    ]

    for mode_prefix, mode_label, color, marker in mode_defs:
        for config in ["loose", "tight"]:
            mean = summary[f"{config}_{mode_prefix}_share_mean"].to_numpy(dtype=float) * 100.0
            std = summary[f"{config}_{mode_prefix}_share_std"].to_numpy(dtype=float) * 100.0
            ax.plot(
                x,
                mean,
                color=color,
                marker=marker,
                linestyle=MODE_STYLE[config],
                linewidth=1.6,
                label=f"{mode_label} ({CONFIG_LABELS[config]})",
            )
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.06)

    set_x_axis_label(ax, "Perturbation amplitude $\\Delta$ (h)")
    ax.set_ylabel("Service-mode share (%)")
    ax.set_ylim(0.0, 100.0)
    apply_common_axis_format(ax)
    ax.legend(loc="upper right", fontsize=7.0, ncol=2)


def draw_fig8_dual(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    start_figure()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35))
    _plot_cost_panel(axes[0], summary)
    _plot_mode_panel(axes[1], summary)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.24, wspace=0.36)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig8_dual_config.pdf"
    png_path = output_dir / "fig8_dual_config.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def draw_fig8_prime(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    start_figure()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN.width, SINGLE_COLUMN.height))

    points: list[tuple[float, float, str]] = []
    for config in ["loose", "tight"]:
        x = summary[f"{config}_relative_delta"].to_numpy(dtype=float)
        y = summary[f"{config}_cost_change_pct"].to_numpy(dtype=float)
        points.extend((float(xv), float(yv), config) for xv, yv in zip(x, y))
        ax.scatter(
            x,
            y,
            color=CONFIG_COLORS[config],
            marker="o" if config == "loose" else "s",
            s=36,
            label=CONFIG_LABELS[config],
        )

    all_x = np.array([pt[0] for pt in points], dtype=float)
    all_y = np.array([pt[1] for pt in points], dtype=float)
    if len(all_x) >= 2:
        coef = np.polyfit(all_x, all_y, deg=1)
        fit_x = np.linspace(float(all_x.min()), float(all_x.max()), 100)
        fit_y = coef[0] * fit_x + coef[1]
        ax.plot(fit_x, fit_y, color=REFERENCE_COLOR, linestyle="-", linewidth=1.2, label="Linear fit")

    ax.axvline(1.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    ax.text(1.02, float(all_y.max()) * 0.75 if len(all_y) else 0.5, "relative perturbation = 1", fontsize=7.2, color="#555555")
    set_x_axis_label(ax, r"Relative perturbation amplitude $\Delta / \bar{s}$")
    ax.set_ylabel("Cost change vs baseline (%)")
    apply_common_axis_format(ax)
    ax.legend(loc="best", fontsize=7.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig8_prime_robustness_boundary.pdf"
    png_path = output_dir / "fig8_prime_robustness_boundary.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.comparison_csv)
    output_dir = Path(args.output_dir)
    draw_fig8_dual(summary, output_dir)
    draw_fig8_prime(summary, output_dir)


if __name__ == "__main__":
    main()

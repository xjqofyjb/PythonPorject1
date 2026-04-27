"""Figure 2: mechanism Gantt chart."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from src.instances import generate_instance
from src.metrics import compute_simops_metrics
from src.runner import build_cg_cfg, get_solver, load_config

from .config import DOUBLE_COLUMN_TALL, MODE_COLORS, REFERENCE_COLOR
from .utils import ExportedFigure, add_figure_legend, add_major_grid, add_panel_labels, export_figure, set_x_axis_label, start_figure


def _solve_schedule(config_path: str, n_vessels: int = 8, seed: int = 2):
    cfg = load_config(config_path)
    params = cfg.get("params", {})
    instance = generate_instance(n_vessels, seed, "U", "hybrid", params)
    method_cfg = {
        "method": "cg",
        "operation_mode": "simops",
        "return_schedule": True,
    }
    method_cfg.update(build_cg_cfg(cfg, "cg"))
    solver = get_solver("cg")
    simops = solver(instance, method_cfg, logger=None)
    method_cfg["operation_mode"] = "sequential"
    sequential = solver(instance, method_cfg, logger=None)
    return instance, sequential, simops


def _add_rect(ax: plt.Axes, x: float, y: float, width: float, height: float, *, face: str, edge: str, hatch: str | None = None, alpha: float = 1.0) -> None:
    if width <= 0:
        return
    rect = Rectangle(
        (x, y),
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.6,
        hatch=hatch,
        alpha=alpha,
        joinstyle="miter",
    )
    ax.add_patch(rect)


def _mode_color(mode: str) -> str:
    return MODE_COLORS["SP"] if mode == "shore" else MODE_COLORS["BS"]


def _summary_box(ax: plt.Axes, *, result: dict, metrics: dict[str, float]) -> None:
    summary = "\n".join(
        [
            f"Objective: ${result['obj']:,.0f}",
            f"Average stay: {metrics['avg_stay_time']:.2f} h",
            f"Masking rate: {metrics['avg_masking_rate'] * 100:.1f}%",
        ]
    )
    ax.text(
        1.01,
        0.86,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.9,
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "boxstyle": "square,pad=0.28"},
    )


def _plot_panel(ax: plt.Axes, instance, result: dict, *, operation_label: str, operation_mode: str, order: list[int]) -> None:
    schedule = result["schedule"]
    cargo_h = 0.58
    service_h = 0.28
    metrics = compute_simops_metrics(
        instance,
        operation_mode,
        service_start_times=np.asarray(schedule["service_start_times"], dtype=float),
        service_durations=np.asarray(schedule["service_durations"], dtype=float),
    )

    for lane, vessel_idx in enumerate(order):
        y = lane
        arrival = float(instance.arrival_times[vessel_idx])
        cargo_duration = float(instance.cargo_times[vessel_idx])
        cargo_end = arrival + cargo_duration
        _add_rect(
            ax,
            arrival,
            y - cargo_h / 2,
            cargo_duration,
            cargo_h,
            face=MODE_COLORS["cargo"],
            edge=MODE_COLORS["cargo_edge"],
        )

        mode = str(schedule["modes"][vessel_idx])
        service_start = float(schedule["service_start_times"][vessel_idx])
        service_duration = float(schedule["service_durations"][vessel_idx])
        service_end = service_start + service_duration

        if service_duration > 0 and mode in {"shore", "battery"}:
            color = _mode_color(mode)
            overlap_start = max(arrival, service_start)
            overlap_end = min(cargo_end, service_end)
            masked_width = max(0.0, overlap_end - overlap_start)
            _add_rect(
                ax,
                service_start,
                y - service_h / 2,
                service_duration,
                service_h,
                face=color,
                edge=color,
                alpha=0.18,
            )
            _add_rect(
                ax,
                overlap_start,
                y - service_h / 2,
                masked_width,
                service_h,
                face=color,
                edge=color,
                alpha=0.92,
            )
            if service_start < overlap_start:
                _add_rect(
                    ax,
                    service_start,
                    y - service_h / 2,
                    overlap_start - service_start,
                    service_h,
                    face=color,
                    edge=color,
                    hatch="////",
                    alpha=0.32,
                )
            if overlap_end < service_end:
                _add_rect(
                    ax,
                    overlap_end,
                    y - service_h / 2,
                    service_end - overlap_end,
                    service_h,
                    face=color,
                    edge=color,
                    hatch="////",
                    alpha=0.32,
                )
        elif mode == "brown":
            ax.scatter([cargo_end], [y], s=20, color=MODE_COLORS["AE"], zorder=4)

        ax.vlines(
            float(instance.deadlines[vessel_idx]),
            y - 0.24,
            y + 0.24,
            color=REFERENCE_COLOR,
            linewidth=0.7,
            linestyles=":",
        )

    _summary_box(ax, result=result, metrics=metrics)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{lane + 1} - Type {instance.ship_types[vessel_idx]}" for lane, vessel_idx in enumerate(order)])
    ax.set_ylabel("Vessel order / type")
    ax.set_xlim(0, max(float(instance.deadlines[vessel_idx]) for vessel_idx in order) + 2.5)
    ax.set_ylim(-0.7, len(order) - 0.25)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    add_major_grid(ax)


def build(config_path: str = "configs/simops.yaml") -> ExportedFigure:
    start_figure()
    instance, sequential, simops = _solve_schedule(config_path)
    order = [int(idx) for idx in np.argsort(instance.arrival_times)]

    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COLUMN_TALL.width, 5.05), sharex=True)
    _plot_panel(axes[0], instance, sequential, operation_label="Sequential operation", operation_mode="sequential", order=order)
    _plot_panel(axes[1], instance, simops, operation_label="SIMOPS operation", operation_mode="simops", order=order)
    set_x_axis_label(axes[1], "Time (hours)")

    legend_handles = [
        Patch(facecolor=MODE_COLORS["cargo"], edgecolor=MODE_COLORS["cargo_edge"], label="Cargo-handling window"),
        Patch(facecolor=MODE_COLORS["SP"], edgecolor=MODE_COLORS["SP"], label="Shore power"),
        Patch(facecolor=MODE_COLORS["BS"], edgecolor=MODE_COLORS["BS"], label="Battery swap"),
        Patch(facecolor="white", edgecolor="#666666", hatch="////", label="Exposed service segment"),
        plt.Line2D([0], [0], color=REFERENCE_COLOR, linestyle=":", linewidth=0.9, label="Departure deadline"),
    ]
    add_figure_legend(fig, legend_handles, [handle.get_label() for handle in legend_handles], ncol=3)
    fig.subplots_adjust(left=0.14, right=0.80, top=0.89, bottom=0.16, hspace=0.26)
    add_panel_labels(fig, axes, ["Sequential operation", "SIMOPS operation"], pad=[0.010, 0.012])
    return export_figure(fig, "Fig2_mechanism_gantt")


if __name__ == "__main__":
    build()

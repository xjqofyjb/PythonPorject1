"""Generate a publication-grade SIMOPS Gantt comparison figure."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch

from analysis.style import set_style
from src.instances import Instance, generate_instance
from src.metrics import compute_simops_metrics
from src.runner import build_cg_cfg, get_solver, load_config


MODE_COLORS = {
    "shore": "#4C956C",
    "battery": "#C95A49",
    "brown": "#7C6148",
}
CARGO_FACE = "#E6E0D8"
CARGO_EDGE = "#B7ADA3"
TEXT_MUTED = "#575757"


@dataclass
class PanelStats:
    title: str
    objective: float
    delay_cost: float
    avg_stay_time: float
    avg_masking_rate: float


def _save_fig(fig: plt.Figure, outdir: str, name: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, f"{name}.pdf"))
    plt.close(fig)


def _lighten(color: str, factor: float = 0.45) -> tuple[float, float, float]:
    import matplotlib.colors as mcolors

    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * factor)


def _add_round_bar(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.9,
    alpha: float = 1.0,
    hatch: str | None = None,
    zorder: float = 2.0,
) -> None:
    if width <= 1e-9:
        return
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=linewidth,
        facecolor=facecolor,
        edgecolor=edgecolor,
        hatch=hatch,
        alpha=alpha,
        mutation_aspect=1.0,
        zorder=zorder,
    )
    ax.add_patch(patch)


def _solve_schedule(
    instance: Instance,
    cfg: Dict[str, Any],
    method: str,
    operation_mode: str,
) -> Dict[str, Any]:
    solver = get_solver(method)
    solver_cfg: Dict[str, Any] = {
        "method": method,
        "operation_mode": operation_mode,
        "return_schedule": True,
    }
    if method.startswith("cg") or method.startswith("rcg") or method == "restricted_cg":
        solver_cfg.update(build_cg_cfg(cfg, method))
    result = solver(instance, solver_cfg, logger=None)
    schedule = result.get("schedule")
    if not schedule:
        raise RuntimeError("Schedule not returned; set return_schedule in solver.")
    return result


def _masked_interval(
    arrival: float,
    cargo: float,
    start: float,
    duration: float,
) -> tuple[float | None, float]:
    if duration <= 0:
        return None, 0.0
    cargo_end = arrival + cargo
    service_end = start + duration
    overlap_start = max(arrival, start)
    overlap_end = min(cargo_end, service_end)
    if overlap_end <= overlap_start:
        return None, 0.0
    return overlap_start, overlap_end - overlap_start


def _panel_box_text(stats: PanelStats) -> str:
    return (
        f"Obj = {stats.objective / 1000:.2f}k\n"
        f"Delay = {stats.delay_cost / 1000:.2f}k\n"
        f"Avg. stay = {stats.avg_stay_time:.2f} h\n"
        f"Masking = {stats.avg_masking_rate:.2f}"
    )


def _plot_panel(
    ax: plt.Axes,
    instance: Instance,
    result: Dict[str, Any],
    order: List[int],
    stats: PanelStats,
    xlim: tuple[float, float],
) -> None:
    schedule = result["schedule"]
    cargo_height = 0.62
    service_height = 0.34
    lane_pad = 0.17

    for lane, ship_id in enumerate(order):
        y0 = lane - cargo_height / 2.0
        arrival = float(instance.arrival_times[ship_id])
        cargo = float(instance.cargo_times[ship_id])
        cargo_end = arrival + cargo

        _add_round_bar(
            ax,
            arrival,
            y0,
            cargo,
            cargo_height,
            facecolor=CARGO_FACE,
            edgecolor=CARGO_EDGE,
            linewidth=0.8,
            zorder=1.0,
        )

        start = float(schedule["service_start_times"][ship_id])
        duration = float(schedule["service_durations"][ship_id])
        mode = str(schedule["modes"][ship_id])
        service_y = lane - service_height / 2.0
        color = MODE_COLORS.get(mode, "#4C78A8")

        if duration > 0:
            service_end = start + duration
            masked_start, masked_width = _masked_interval(arrival, cargo, start, duration)

            if masked_start is None:
                _add_round_bar(
                    ax,
                    start,
                    service_y,
                    duration,
                    service_height,
                    facecolor=_lighten(color, 0.58),
                    edgecolor=color,
                    linewidth=1.0,
                    hatch="////",
                    zorder=3.0,
                )
            else:
                if start < masked_start:
                    _add_round_bar(
                        ax,
                        start,
                        service_y,
                        masked_start - start,
                        service_height,
                        facecolor=_lighten(color, 0.58),
                        edgecolor=color,
                        linewidth=1.0,
                        hatch="////",
                        zorder=3.0,
                    )
                _add_round_bar(
                    ax,
                    masked_start,
                    service_y,
                    masked_width,
                    service_height,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=4.0,
                )
                masked_end = masked_start + masked_width
                if masked_end < service_end:
                    _add_round_bar(
                        ax,
                        masked_end,
                        service_y,
                        service_end - masked_end,
                        service_height,
                        facecolor=_lighten(color, 0.58),
                        edgecolor=color,
                        linewidth=1.0,
                        hatch="////",
                        zorder=3.0,
                    )
        elif mode == "brown":
            ax.scatter(
                [cargo_end],
                [lane],
                s=44,
                marker="D",
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=5.0,
            )

        ax.vlines(
            float(instance.deadlines[ship_id]),
            lane - 0.27,
            lane + 0.27,
            color="#8E8E8E",
            linewidth=0.9,
            linestyles=(0, (2, 2)),
            zorder=0.6,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.invert_yaxis()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"Ship {ship_id} ({instance.ship_types[ship_id]})" for ship_id in order])
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.text(
        0.985,
        0.04,
        _panel_box_text(stats),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.3,
        color=TEXT_MUTED,
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#D1D1D1",
            "alpha": 0.97,
        },
    )


def _build_stats(instance: Instance, result: Dict[str, Any], title: str, operation_mode: str) -> PanelStats:
    schedule = result["schedule"]
    metrics = compute_simops_metrics(
        instance,
        operation_mode,
        service_start_times=np.asarray(schedule["service_start_times"], dtype=float),
        service_durations=np.asarray(schedule["service_durations"], dtype=float),
    )
    return PanelStats(
        title=title,
        objective=float(result["obj"]),
        delay_cost=float(result.get("cost_delay", 0.0)),
        avg_stay_time=float(metrics.get("avg_stay_time", np.nan)),
        avg_masking_rate=float(metrics.get("avg_masking_rate", 0.0)),
    )


def _determine_xlim(instance: Instance, results: List[Dict[str, Any]], order: List[int]) -> tuple[float, float]:
    min_time = min(float(instance.arrival_times[i]) for i in order)
    max_candidates = []
    for result in results:
        schedule = result["schedule"]
        for ship_id in order:
            start = float(schedule["service_start_times"][ship_id])
            dur = float(schedule["service_durations"][ship_id])
            cargo_end = float(instance.arrival_times[ship_id] + instance.cargo_times[ship_id])
            max_candidates.append(max(cargo_end, start + dur))
    max_time = max(max_candidates) if max_candidates else min_time + 1.0
    left = max(0.0, np.floor((min_time - 0.75) / 2.0) * 2.0)
    right = np.ceil((max_time + 1.25) / 2.0) * 2.0
    return left, right


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a top-journal SIMOPS Gantt comparison figure.")
    parser.add_argument("--config", default="configs/simops.yaml", help="Path to YAML config")
    parser.add_argument("--N", type=int, default=8, help="Instance size")
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--scenario", default="U", help="Scenario code")
    parser.add_argument("--mechanism", default="hybrid", help="Mechanism type")
    parser.add_argument("--method", default="cg", help="Solver method")
    parser.add_argument("--outdir", default="figs/simops", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    params = cfg.get("params", {})
    instance = generate_instance(args.N, args.seed, args.scenario, args.mechanism, params)

    set_style()

    sequential = _solve_schedule(instance, cfg, args.method, "sequential")
    simops = _solve_schedule(instance, cfg, args.method, "simops")

    order = [int(i) for i in np.argsort(instance.arrival_times)]
    xlim = _determine_xlim(instance, [sequential, simops], order)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.6, 6.8),
        sharex=True,
        gridspec_kw={"hspace": 0.10},
    )

    stats_seq = _build_stats(instance, sequential, "Sequential operation", "sequential")
    stats_sim = _build_stats(instance, simops, "SIMOPS operation", "simops")

    _plot_panel(axes[0], instance, sequential, order, stats_seq, xlim)
    _plot_panel(axes[1], instance, simops, order, stats_sim, xlim)

    axes[1].set_xlabel("Time (hours)")
    for ax in axes:
        ax.set_ylabel("Arrival-ordered ships")
        xticks = np.arange(xlim[0], xlim[1] + 0.1, 4.0)
        ax.set_xticks(xticks)

    legend_handles = [
        Patch(facecolor=CARGO_FACE, edgecolor=CARGO_EDGE, label="Cargo handling window"),
        Patch(facecolor=MODE_COLORS["shore"], edgecolor=MODE_COLORS["shore"], label="Shore power"),
        Patch(facecolor=MODE_COLORS["battery"], edgecolor=MODE_COLORS["battery"], label="Battery swap"),
        Patch(
            facecolor="#FFFFFF",
            edgecolor="#777777",
            hatch="////",
            label="Unmasked service segment",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    fig.text(
        0.5,
        0.01,
        "Bars extending beyond the gray cargo window indicate service time that remains exposed after cargo completion.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=TEXT_MUTED,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.89, bottom=0.09, hspace=0.14)
    _save_fig(fig, args.outdir, "Fig_SIMOPS_Gantt_Comparison")


if __name__ == "__main__":
    main()

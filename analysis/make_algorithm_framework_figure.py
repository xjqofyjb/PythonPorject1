"""Draw the decomposition-based solution framework figure.

The figure is intentionally rendered from code so the manuscript asset can be
regenerated without hand-editing drawing software.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "revised" / "tr_style"
RES_DIR = ROOT / "results" / "revised" / "tr_style"
STEM = "fig_algorithm_framework"

CAPTION = (
    "Decomposition-based solution framework. The algorithm starts from feasible "
    "SP, BS, and AE service-plan columns and solves a restricted master problem. "
    "Dual prices are passed to mode-specific pricing problems to generate "
    "negative reduced-cost columns. For N <= 100, full pricing is continued "
    "until LP convergence and the reported gap is a Full-CG LP-IP gap. For "
    "large-scale instances, a strengthened budgeted-CG protocol with "
    "incumbent-column injection is used, and the reported gap is a "
    "generated-pool LP-IP gap rather than a complete-column global optimality "
    "certificate. The final integer schedule is recovered by solving the IRMP "
    "on the generated column pool."
)


COLORS = {
    "input": "#eeeeee",
    "rmp": "#dbeaf7",
    "pricing": "#fde6cc",
    "diagnostic": "#fff4c2",
    "recovery": "#dff0df",
    "border": "#444444",
    "arrow": "#333333",
    "muted": "#666666",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.default": "regular",
        }
    )


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    color: str,
) -> dict[str, tuple[float, float]]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=COLORS["border"],
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.035,
        title,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color="#111111",
        zorder=3,
    )
    body = "\n".join(lines)
    title_lines = title.count("\n") + 1
    ax.text(
        x + 0.018,
        y + h - 0.044 - 0.034 * title_lines,
        body,
        ha="left",
        va="top",
        fontsize=7.15,
        linespacing=1.32,
        color="#222222",
        zorder=3,
    )
    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2, y + h / 2),
    }


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str | None = None,
    *,
    dashed: bool = False,
    rad: float = 0.0,
    label_xy: tuple[float, float] | None = None,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["arrow"],
            "linewidth": 0.9,
            "linestyle": (0, (4, 3)) if dashed else "solid",
            "mutation_scale": 10,
            "shrinkA": 3,
            "shrinkB": 3,
            "connectionstyle": f"arc3,rad={rad}",
        },
        zorder=4,
    )
    if label:
        lx, ly = label_xy if label_xy else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(
            lx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.92},
            zorder=5,
        )


def elbow_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    dashed: bool = False,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    if len(points) < 2:
        return
    xs = [p[0] for p in points[:-1]]
    ys = [p[1] for p in points[:-1]]
    ax.plot(
        xs,
        ys,
        color=COLORS["arrow"],
        linewidth=0.9,
        linestyle=(0, (4, 3)) if dashed else "solid",
        zorder=4,
    )
    ax.annotate(
        "",
        xy=points[-1],
        xytext=points[-2],
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["arrow"],
            "linewidth": 0.9,
            "linestyle": (0, (4, 3)) if dashed else "solid",
            "mutation_scale": 10,
            "shrinkA": 0,
            "shrinkB": 3,
        },
        zorder=4,
    )
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            ha="center",
            va="center",
            fontsize=6.8,
            color=COLORS["muted"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.92},
            zorder=5,
        )


def draw() -> None:
    configure_style()
    fig, ax = plt.subplots(figsize=(10.6, 5.9), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Decomposition-based solution framework and solution-quality diagnostics",
        ha="center",
        va="top",
        fontsize=11.0,
        fontweight="bold",
        color="#111111",
    )

    input_box = box(
        ax,
        0.035,
        0.58,
        0.27,
        0.285,
        "1. Input and feasible service plans",
        [
            r"Vessel data and compatibility",
            r"SP/BS capacities",
            "Feasible start sets",
            "Initial AE/SP/BS columns",
        ],
        COLORS["input"],
    )

    rmp_box = box(
        ax,
        0.37,
        0.62,
        0.255,
        0.235,
        "2. Restricted master\nproblem (RMP)",
        [
            "Assignment constraints",
            "SP/BS capacity constraints",
            r"LP duals: $\pi_i, \rho_{kt}, \eta_t$",
        ],
        COLORS["rmp"],
    )

    pricing_box = box(
        ax,
        0.37,
        0.335,
        0.255,
        0.215,
        "3. Mode-specific pricing",
        [
            "SP pricing",
            "BS pricing",
            "AE fallback pricing",
            "Negative reduced-cost columns",
        ],
        COLORS["pricing"],
    )

    diag_box = box(
        ax,
        0.705,
        0.585,
        0.26,
        0.285,
        "5. Stopping and\nsolution-quality logic",
        [
            r"$N \leq 100$: full pricing",
            "Full-CG LP-IP gap",
            r"$N=200,500$: budgeted CG",
            "Pool LP-IP gap",
            "No global certificate",
        ],
        COLORS["diagnostic"],
    )

    irmp_box = box(
        ax,
        0.705,
        0.345,
        0.26,
        0.145,
        "6. Integer recovery",
        [
            "Solve IRMP",
            "Recover integer schedule",
        ],
        COLORS["recovery"],
    )

    output_box = box(
        ax,
        0.705,
        0.065,
        0.26,
        0.20,
        "7. Outputs and\ndiagnostics",
        [
            "Cost and mode shares",
            "Delay and masking rate",
            "Gap type",
            "SIMOPS dominance check",
        ],
        COLORS["recovery"],
    )

    loop_box = box(
        ax,
        0.37,
        0.105,
        0.255,
        0.155,
        "4. Column generation loop",
        [
            "Add columns",
            "Resolve RMP",
            "Repeat until stop",
        ],
        "#f8f8f8",
    )

    inject_box = box(
        ax,
        0.035,
        0.20,
        0.27,
        0.185,
        "Optional injected columns",
        [
            "Heuristic incumbents",
            "Sequential-to-SIMOPS columns",
            "Feasibility checked",
        ],
        COLORS["input"],
    )

    arrow(ax, input_box["right"], rmp_box["left"], "initial columns", label_xy=(0.335, 0.755))
    arrow(ax, rmp_box["bottom"], pricing_box["top"], "dual prices", label_xy=(0.535, 0.585))
    arrow(ax, pricing_box["bottom"], loop_box["top"], "negative reduced-cost columns", label_xy=(0.535, 0.305))
    elbow_arrow(
        ax,
        [loop_box["left"], (0.335, loop_box["left"][1]), (0.335, rmp_box["left"][1]), rmp_box["left"]],
        label="repeat",
        label_xy=(0.317, 0.45),
    )
    arrow(ax, rmp_box["right"], diag_box["left"])
    arrow(ax, diag_box["bottom"], irmp_box["top"], "final pool")
    arrow(ax, irmp_box["bottom"], output_box["top"])

    arrow(
        ax,
        inject_box["top"],
        (0.40, 0.63),
        dashed=True,
        rad=-0.10,
    )
    elbow_arrow(
        ax,
        [inject_box["right"], (0.335, inject_box["right"][1]), (0.335, loop_box["left"][1]), loop_box["left"]],
        dashed=True,
    )

    ax.text(
        0.5,
        0.02,
        "Solid arrows: column generation and recovery flow. Dashed arrows: optional injected incumbent or sequential columns.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=COLORS["muted"],
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG_DIR / f"{STEM}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    (RES_DIR / f"{STEM}_caption.txt").write_text(CAPTION + "\n", encoding="utf-8")


if __name__ == "__main__":
    draw()

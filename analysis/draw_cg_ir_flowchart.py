"""Draw a publication-ready CG+IR workflow figure using matplotlib."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon


CM_TO_IN = 1.0 / 2.54


START_END_COLOR = "#1a3a5c"
RECT_COLOR = "#e8f0f7"
DECISION_COLOR = "#f0a500"
PHASE1_COLOR = "#3b6c9e"
PHASE2_COLOR = "#3b8b69"
ARROW_COLOR = "#000000"


@dataclass
class Node:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def top(self) -> tuple[float, float]:
        return self.cx, self.cy + self.h / 2

    @property
    def bottom(self) -> tuple[float, float]:
        return self.cx, self.cy - self.h / 2

    @property
    def left(self) -> tuple[float, float]:
        return self.cx - self.w / 2, self.cy

    @property
    def right(self) -> tuple[float, float]:
        return self.cx + self.w / 2, self.cy


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
        }
    )


def draw_rectangle(ax: plt.Axes, node: Node, text: str, facecolor: str = RECT_COLOR, textcolor: str = "black") -> None:
    patch = FancyBboxPatch(
        (node.cx - node.w / 2, node.cy - node.h / 2),
        node.w,
        node.h,
        boxstyle="round,pad=0.007,rounding_size=0.015",
        linewidth=0.9,
        edgecolor=ARROW_COLOR,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(node.cx, node.cy, text, ha="center", va="center", color=textcolor, multialignment="center", zorder=3)


def draw_ellipse(ax: plt.Axes, node: Node, text: str) -> None:
    patch = Ellipse(
        (node.cx, node.cy),
        node.w,
        node.h,
        linewidth=0.9,
        edgecolor=START_END_COLOR,
        facecolor=START_END_COLOR,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(node.cx, node.cy, text, ha="center", va="center", color="white", multialignment="center", zorder=3)


def draw_diamond(ax: plt.Axes, node: Node, text: str) -> None:
    vertices = [
        (node.cx, node.cy + node.h / 2),
        (node.cx + node.w / 2, node.cy),
        (node.cx, node.cy - node.h / 2),
        (node.cx - node.w / 2, node.cy),
    ]
    patch = Polygon(vertices, closed=True, linewidth=0.9, edgecolor=DECISION_COLOR, facecolor=DECISION_COLOR, zorder=2)
    ax.add_patch(patch)
    ax.text(node.cx, node.cy, text, ha="center", va="center", color="white", multialignment="center", zorder=3)


def draw_phase_box(ax: plt.Axes, x: float, y: float, w: float, h: float, color: str, label: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.0,
        edgecolor=color,
        facecolor="none",
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x - 0.03,
        y + h / 2,
        label,
        rotation=90,
        ha="center",
        va="center",
        color=color,
        fontsize=8.0,
        fontweight="bold",
    )


def arrow(
    ax: plt.Axes,
    xy_from: tuple[float, float],
    xy_to: tuple[float, float],
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    connectionstyle: str = "arc3",
) -> None:
    ax.annotate(
        "",
        xy=xy_to,
        xytext=xy_from,
        arrowprops=dict(
            arrowstyle="-|>",
            color=ARROW_COLOR,
            linewidth=0.95,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=9.5,
            connectionstyle=connectionstyle,
        ),
        zorder=4,
    )
    if label and label_xy:
        ax.text(label_xy[0], label_xy[1], label, fontsize=7.5, ha="center", va="center", color="black")


def build_figure(outdir: Path, stem: str) -> Path:
    configure_style()

    fig_w = 8.5 * CM_TO_IN
    fig_h = 17.8 * CM_TO_IN
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_phase_box(ax, 0.12, 0.50, 0.82, 0.44, PHASE1_COLOR, "Phase 1: LP Iteration (Column Generation)")
    draw_phase_box(ax, 0.12, 0.09, 0.82, 0.34, PHASE2_COLOR, "Phase 2: Integer Recovery (IRMP)")

    start = Node(0.57, 0.92, 0.68, 0.065)
    rmp = Node(0.57, 0.81, 0.62, 0.068)
    pricing = Node(0.57, 0.69, 0.68, 0.105)
    decision1 = Node(0.57, 0.56, 0.30, 0.09)
    add_col = Node(0.20, 0.56, 0.24, 0.06)
    lp_conv = Node(0.57, 0.45, 0.46, 0.06)
    irmp = Node(0.57, 0.34, 0.62, 0.075)
    gap = Node(0.57, 0.235, 0.58, 0.07)
    decision2 = Node(0.57, 0.13, 0.22, 0.085)
    certified = Node(0.25, 0.13, 0.24, 0.058)
    worst_case = Node(0.84, 0.13, 0.24, 0.07)
    output = Node(0.57, 0.035, 0.50, 0.055)

    draw_ellipse(ax, start, "Start: Initialize column pool\n$\\hat{P}_i^{(0)}$ with AE fallback +\ngreedy SP/BS columns")
    draw_rectangle(ax, rmp, "Solve LP relaxation of RMP\n$\\rightarrow$ obtain dual variables\n$(\\pi^*,\\ \\mu^*,\\ \\nu^*)$")
    draw_rectangle(
        ax,
        pricing,
        "Solve pricing subproblem $PP_i$ for each ship $i$:\ncompute reduced cost\n$\\bar{\\sigma}_i^*=\\min\\{\\sigma_i^{SP}(k,t),\\ \\sigma_i^{BS}(t),\\ \\sigma_i^{AE}\\}$",
    )
    draw_diamond(ax, decision1, "Any $\\bar{\\sigma}_i^* < -\\epsilon$ ?")
    draw_rectangle(ax, add_col, "Add new column\nto $\\hat{P}_i$")
    draw_rectangle(ax, lp_conv, "LP converged:\n$Z_{LP}^*$ = lower bound")
    draw_rectangle(ax, irmp, "Solve IRMP on column pool $\\hat{P}_i$\nwith integrality constraints\n$\\rightarrow\\ Z_{IP}^{IRMP}$")
    draw_rectangle(ax, gap, "Compute integrality gap:\n$Gap=(Z_{IP}^{IRMP}-Z_{LP}^*)/Z_{LP}^*\\times100\\%$")
    draw_diamond(ax, decision2, "$Gap = 0$ ?")
    draw_rectangle(ax, certified, "Optimal solution\ncertified")
    draw_rectangle(ax, worst_case, "Gap provides\nworst-case quality bound")
    draw_ellipse(ax, output, "Output: integer solution + gap certificate")

    arrow(ax, start.bottom, rmp.top)
    arrow(ax, rmp.bottom, pricing.top)
    arrow(ax, pricing.bottom, decision1.top)
    arrow(ax, decision1.bottom, lp_conv.top, label="No", label_xy=(0.61, 0.505))
    arrow(ax, decision1.left, add_col.right, label="Yes", label_xy=(0.385, 0.595))
    arrow(ax, add_col.top, (rmp.left[0], rmp.left[1] + 0.005), connectionstyle="angle3,angleA=90,angleB=180")
    arrow(ax, lp_conv.bottom, irmp.top)
    arrow(ax, irmp.bottom, gap.top)
    arrow(ax, gap.bottom, decision2.top)
    arrow(ax, decision2.left, certified.right, label="Yes", label_xy=(0.41, 0.165))
    arrow(ax, decision2.right, worst_case.left, label="No", label_xy=(0.72, 0.165))
    arrow(ax, certified.bottom, (output.top[0] - 0.11, output.top[1]), connectionstyle="angle3,angleA=-90,angleB=180")
    arrow(ax, worst_case.bottom, (output.top[0] + 0.11, output.top[1]), connectionstyle="angle3,angleA=-90,angleB=0")

    fig.tight_layout(pad=0.15)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the CG+IR workflow figure for paper submission.")
    parser.add_argument("--outdir", default="figs/paper", help="Output directory")
    parser.add_argument("--stem", default="Fig_CG_IR_Workflow", help="Output file stem")
    args = parser.parse_args()

    pdf_path = build_figure(Path(args.outdir), args.stem)
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

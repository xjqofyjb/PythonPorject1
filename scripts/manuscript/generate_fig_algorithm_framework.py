"""
Render the Section-4 algorithm-framework schematic for the manuscript.

Clean Transportation Research / Elsevier-style flowchart of the
decomposition-based solution framework.

Design goals (v3):
  - Single, clearly-vertical main flow (input -> RMP -> pricing ->
    CG-loop/stopping -> IRMP -> outputs).
  - Right-angle (L-shaped) arrows so nothing crosses over a box.
  - One dedicated rail on the right for the CG loop ("add columns; repeat").
  - Optional dashed flows (sequential-column / incumbent-column injection)
    parked on the left rail with clear, readable italic labels.
  - Thin borders, small arrow heads, muted fills, no gradients/shadows.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_FIG_DIR = os.path.join(ROOT, "figures", "revised", "tr_style")
OUT_DAT_DIR = os.path.join(ROOT, "results", "revised", "tr_style")

# ----- Muted palette ------------------------------------------------------
COLORS = {
    "input":      {"fill": "#EFEFEF", "edge": "#9A9A9A"},  # light gray
    "rmp":        {"fill": "#DCE6F2", "edge": "#5C7CA0"},  # light blue
    "pricing":    {"fill": "#FCE5CD", "edge": "#C97A29"},  # light orange
    "diagnostic": {"fill": "#FFF2CC", "edge": "#B58B19"},  # light yellow
    "ir":         {"fill": "#D9EAD3", "edge": "#5A8A4F"},  # light green
    "output":     {"fill": "#D9EAD3", "edge": "#5A8A4F"},  # light green
}

# Arrow tuning - small heads, thin lines
HEAD_LEN, HEAD_WID = 4.0, 2.4
LW       = 0.7
MUT      = 7
ARROW_C  = "0.30"
DASH_C   = "0.55"
DASH_LS  = (0, (4, 3))


def add_box(ax, *, x, y, w, h, kind, title, body_lines,
            title_size=10.0, body_size=8.2):
    fill = COLORS[kind]["fill"]
    edge = COLORS[kind]["edge"]
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010,rounding_size=0.10",
        linewidth=0.85, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.16, title,
            ha="center", va="top", fontsize=title_size, fontweight="bold",
            color="0.10", zorder=3)
    ax.text(x + w / 2, y + h - 0.42, "\n".join(body_lines),
            ha="center", va="top", fontsize=body_size, color="0.18",
            zorder=3, linespacing=1.30)


def straight(ax, src, dst, *, dashed=False):
    """Plain vertical/horizontal arrow."""
    color = DASH_C if dashed else ARROW_C
    ls = DASH_LS if dashed else "-"
    arr = FancyArrowPatch(
        src, dst,
        arrowstyle=f"->,head_length={HEAD_LEN},head_width={HEAD_WID}",
        connectionstyle="arc3,rad=0",
        linewidth=LW, color=color, linestyle=ls,
        zorder=4, mutation_scale=MUT,
    )
    ax.add_patch(arr)


def angle_arrow(ax, src, dst, *, angleA=-90, angleB=180, rad=4, dashed=False):
    """Right-angle arrow (L-shape) from src to dst."""
    color = DASH_C if dashed else ARROW_C
    ls = DASH_LS if dashed else "-"
    arr = FancyArrowPatch(
        src, dst,
        arrowstyle=f"->,head_length={HEAD_LEN},head_width={HEAD_WID}",
        connectionstyle=f"angle,angleA={angleA},angleB={angleB},rad={rad}",
        linewidth=LW, color=color, linestyle=ls,
        zorder=4, mutation_scale=MUT,
    )
    ax.add_patch(arr)


def label(ax, x, y, text, *, color="0.20", size=7.6, italic=False, halo=True):
    style = "italic" if italic else "normal"
    bbox = (dict(boxstyle="round,pad=0.18", facecolor="white",
                 edgecolor="none", alpha=0.95)
            if halo else None)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=size, color=color, style=style,
            bbox=bbox, zorder=5)


def build_figure():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    fig, ax = plt.subplots(figsize=(11.6, 9.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.2, 13.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    # Layout: main column [3.2 .. 14.0] = width 10.8
    L_RAIL    = 1.95          # left rail x for dashed labels
    MAIN_X    = 3.20
    MAIN_W    = 10.80
    MAIN_R    = MAIN_X + MAIN_W       # 14.00
    R_RAIL    = MAIN_R + 1.10         # 15.10  -> right rail for CG loop label

    # ---------------- 1. Input -------------------------------------------
    add_box(ax, x=MAIN_X, y=11.20, w=MAIN_W, h=1.55, kind="input",
            title="1. Input data and feasible service plans",
            body_lines=[
                r"Vessel data:  $A_i$ arrival,  $E_i$ EFT,  $T_i^{\mathrm{cargo}}$,  $D_i$ deadline,  $Q_{ik}$ SP-compatibility",
                r"Resource capacities:  $K^{SP}$ (shore-power berths),  $K^{BS}$ (battery swap units)",
                r"Initial columns:  AE-fallback,  feasible SP / BS service plans  (warm-start pool $\mathcal{P}^{(0)}$)",
            ])

    # ---------------- 2. RMP ---------------------------------------------
    rmp_y, rmp_h = 8.85, 1.95
    add_box(ax, x=MAIN_X, y=rmp_y, w=MAIN_W, h=rmp_h, kind="rmp",
            title="2. Restricted master problem (RMP) — LP relaxation",
            body_lines=[
                r"Assignment / convexity:  $\sum_{p \in \mathcal{P}_i} \lambda_{ip} = 1$",
                r"SP / BS capacity:  $\sum_{i,p} a^{SP}_{ipkt}\,\lambda_{ip} \leq K^{SP}_{kt}$,   $\sum_{i,p} a^{BS}_{ipt}\,\lambda_{ip} \leq K^{BS}_{t}$",
                r"Solve LP relaxation  $\Rightarrow$  duals  $\pi_i$  (assignment),  $\rho_{kt}$  (SP),  $\eta_t$  (BS)",
            ])

    # ---------------- 3. Pricing problems --------------------------------
    pricing_y, pricing_h = 6.30, 1.85
    pw = 3.30
    gap = (MAIN_W - 3 * pw) / 2.0
    px1 = MAIN_X
    px2 = px1 + pw + gap
    px3 = px2 + pw + gap
    add_box(ax, x=px1, y=pricing_y, w=pw, h=pricing_h, kind="pricing",
            title="3a. SP pricing",
            body_lines=[
                r"Find berth $k$, start $t$",
                r"with reduced cost",
                r"$\bar{c}^{SP}_{ip} = c_{ip} - \pi_i - \!\sum_{k,t}\!\rho_{kt}\,a^{SP}_{ipkt} < 0$",
            ])
    add_box(ax, x=px2, y=pricing_y, w=pw, h=pricing_h, kind="pricing",
            title="3b. BS pricing",
            body_lines=[
                r"Find start $t$",
                r"with reduced cost",
                r"$\bar{c}^{BS}_{ip} = c_{ip} - \pi_i - \!\sum_{t}\!\eta_{t}\,a^{BS}_{ipt} < 0$",
            ])
    add_box(ax, x=px3, y=pricing_y, w=pw, h=pricing_h, kind="pricing",
            title="3c. AE pricing  (fallback)",
            body_lines=[
                r"Always feasible;",
                r"reduced cost",
                r"$\bar{c}^{AE}_{ip} = c^{AE}_{ip} - \pi_i$",
            ])

    # ---------------- 4./5. CG loop & stopping ---------------------------
    diag_y, diag_h = 3.55, 2.10
    add_box(ax, x=MAIN_X, y=diag_y, w=MAIN_W, h=diag_h, kind="diagnostic",
            title="4./5. Column-generation loop  &  stopping / solution-quality logic",
            body_lines=[
                r"Add columns with $\bar{c}_{ip} < 0$ to RMP and re-solve until stop:",
                r"$\bullet$  $N \leq 100$:   full pricing convergence  $\Rightarrow$  Full-CG LP-IP gap",
                r"$\bullet$  $N = 200, 500$:   strengthened budgeted CG  +  incumbent-column injection  +  objective stabilization",
                r"$\Rightarrow$  Pool LP-IP gap   (NOT a complete-column global optimality certificate)",
            ])

    # ---------------- 6. Integer recovery --------------------------------
    ir_y, ir_h = 1.45, 1.45
    ir_w = 6.6
    ir_x = MAIN_X + (MAIN_W - ir_w) / 2.0
    add_box(ax, x=ir_x, y=ir_y, w=ir_w, h=ir_h, kind="ir",
            title="6. Integer recovery (IRMP)",
            body_lines=[
                r"Solve IP on final column pool  $\mathcal{P}^{(*)} \subseteq \bigcup_i \mathcal{P}_i$",
                r"$\Rightarrow$  integer schedule  $\{\lambda^{*}_{ip}\}$",
            ])

    # ---------------- 7. Outputs -----------------------------------------
    out_y, out_h = -0.10, 1.30
    add_box(ax, x=MAIN_X, y=out_y, w=MAIN_W, h=out_h, kind="output",
            title="7. Outputs",
            body_lines=[
                r"Total cost  $\bullet$  SP / BS / AE shares  $\bullet$  Average delay  $\bullet$  Masking rate",
                r"Gap-type label (Full-CG LP-IP  vs  Pool LP-IP)  $\bullet$  Dominance diagnostics for SIMOPS vs sequential",
            ])

    # ============================ Arrows =================================
    cx_main = MAIN_X + MAIN_W / 2.0
    rmp_top = rmp_y + rmp_h
    rmp_bot = rmp_y
    pric_top = pricing_y + pricing_h
    pric_bot = pricing_y
    diag_top = diag_y + diag_h
    diag_bot = diag_y
    ir_top   = ir_y + ir_h
    ir_bot   = ir_y
    in_bot   = 11.20
    out_top  = out_y + out_h

    # Input -> RMP  (vertical)
    straight(ax, (cx_main, in_bot), (cx_main, rmp_top))

    # RMP -> three pricing boxes (vertical to each)
    pric_centers = [px1 + pw / 2, px2 + pw / 2, px3 + pw / 2]
    for cx in pric_centers:
        straight(ax, (cx, rmp_bot), (cx, pric_top))

    # Single dual-prices label centred between RMP and pricing
    label(ax, cx_main, (rmp_bot + pric_top) / 2,
          r"duals  $\pi_i,\ \rho_{kt},\ \eta_t$", size=7.9)

    # Pricing -> diagnostic block (vertical from each)
    for cx in pric_centers:
        straight(ax, (cx, pric_bot), (cx, diag_top))

    # ----- CG loop: L-shape on the right rail (clean, no curves) ---------
    # diagnostic right-edge -> up via right rail -> RMP right-edge
    cg_y_top = rmp_y + rmp_h * 0.45
    cg_y_bot = diag_y + diag_h * 0.55
    rail_x   = MAIN_R + 0.40
    # Down-going from diagnostic -> rail (horizontal)
    arr1 = FancyArrowPatch(
        (MAIN_R, cg_y_bot), (rail_x, cg_y_bot),
        arrowstyle="-",
        connectionstyle="arc3,rad=0",
        linewidth=LW, color=ARROW_C, zorder=4,
    )
    ax.add_patch(arr1)
    # Vertical along the rail
    arr2 = FancyArrowPatch(
        (rail_x, cg_y_bot), (rail_x, cg_y_top),
        arrowstyle="-",
        connectionstyle="arc3,rad=0",
        linewidth=LW, color=ARROW_C, zorder=4,
    )
    ax.add_patch(arr2)
    # Horizontal back to RMP, with arrowhead
    arr3 = FancyArrowPatch(
        (rail_x, cg_y_top), (MAIN_R, cg_y_top),
        arrowstyle=f"->,head_length={HEAD_LEN},head_width={HEAD_WID}",
        connectionstyle="arc3,rad=0",
        linewidth=LW, color=ARROW_C, zorder=4, mutation_scale=MUT,
    )
    ax.add_patch(arr3)
    # Loop label, vertically centred along the rail
    label(ax, rail_x + 0.05, (cg_y_top + cg_y_bot) / 2,
          "add columns;\nrepeat until\nstop",
          size=7.7)

    # Diagnostic -> IRMP (vertical)
    straight(ax, (cx_main, diag_bot), (cx_main, ir_top))

    # IRMP -> Outputs (vertical)
    straight(ax, (cx_main, ir_bot), (cx_main, out_top))

    # ---------- Optional dashed flows on the LEFT rail -------------------
    # (a) Sequential-column injection: input bottom-left -> RMP top-left, L-shape
    sci_x_in = MAIN_X + 0.35
    sci_x_rail = L_RAIL
    # down from input
    a = FancyArrowPatch((sci_x_in, in_bot), (sci_x_rail, in_bot),
                         arrowstyle="-", linewidth=LW,
                         color=DASH_C, linestyle=DASH_LS, zorder=4)
    ax.add_patch(a)
    a = FancyArrowPatch((sci_x_rail, in_bot), (sci_x_rail, rmp_top),
                         arrowstyle="-", linewidth=LW,
                         color=DASH_C, linestyle=DASH_LS, zorder=4)
    ax.add_patch(a)
    a = FancyArrowPatch(
        (sci_x_rail, rmp_top), (sci_x_in, rmp_top),
        arrowstyle=f"->,head_length={HEAD_LEN},head_width={HEAD_WID}",
        linewidth=LW, color=DASH_C, linestyle=DASH_LS,
        zorder=4, mutation_scale=MUT,
    )
    ax.add_patch(a)
    label(ax, sci_x_rail - 0.08, (in_bot + rmp_top) / 2,
          "sequential-column\ninjection (optional)",
          color=DASH_C, size=7.4, italic=True)

    # (b) Incumbent-column injection: IRMP top-left -> diagnostic bottom-left
    ici_x_box_diag = MAIN_X + 0.35
    ici_x_box_ir   = ir_x + 0.35
    a = FancyArrowPatch(
        (ici_x_box_ir, ir_top), (sci_x_rail, ir_top),
        arrowstyle="-", linewidth=LW,
        color=DASH_C, linestyle=DASH_LS, zorder=4,
    )
    ax.add_patch(a)
    a = FancyArrowPatch(
        (sci_x_rail, ir_top), (sci_x_rail, diag_bot),
        arrowstyle="-", linewidth=LW,
        color=DASH_C, linestyle=DASH_LS, zorder=4,
    )
    ax.add_patch(a)
    a = FancyArrowPatch(
        (sci_x_rail, diag_bot), (ici_x_box_diag, diag_bot),
        arrowstyle=f"->,head_length={HEAD_LEN},head_width={HEAD_WID}",
        linewidth=LW, color=DASH_C, linestyle=DASH_LS,
        zorder=4, mutation_scale=MUT,
    )
    ax.add_patch(a)
    label(ax, sci_x_rail - 0.08, (ir_top + diag_bot) / 2,
          "incumbent-column\ninjection (optional)",
          color=DASH_C, size=7.4, italic=True)

    # ----- Title --------------------------------------------------------
    fig.suptitle("Decomposition-based solution framework "
                 "and solution-quality diagnostics",
                 x=0.5, y=0.985, fontsize=12.5, fontweight="bold")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005)
    return fig


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DAT_DIR, exist_ok=True)

    fig = build_figure()
    base = os.path.join(OUT_FIG_DIR, "fig_algorithm_framework")
    fig.savefig(base + ".pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(base + ".svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Decomposition-based solution framework. The algorithm starts from "
        "feasible SP, BS, and AE service-plan columns and solves a "
        "restricted master problem. Dual prices are passed to mode-specific "
        "pricing problems to generate negative reduced-cost columns. For "
        "$N \\le 100$, full pricing is continued until LP convergence and "
        "the reported gap is a Full-CG LP-IP gap. For large-scale "
        "instances, a strengthened budgeted-CG protocol with "
        "incumbent-column injection is used, and the reported gap is a "
        "generated-pool LP-IP gap rather than a complete-column global "
        "optimality certificate. The final integer schedule is recovered "
        "by solving the IRMP on the generated column pool.\n"
    )
    with open(os.path.join(OUT_DAT_DIR, "fig_algorithm_framework_caption.txt"),
              "w", encoding="utf-8") as f:
        f.write(caption)

    print("OK")
    for ext in ("pdf", "png", "svg"):
        print(f"  {base}.{ext}")


if __name__ == "__main__":
    main()
ework "
                 "and solution-quality diagnostics",
                 x=0.5, y=0.985, fontsize=12.5, fontweight="bold")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005)
    return fig


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DAT_DIR, exist_ok=True)

    fig = build_figure()
    base = os.path.join(OUT_FIG_DIR, "fig_algorithm_framework")
    fig.savefig(base + ".pdf", facecolor="white", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(base + ".svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Decomposition-based solution framework. The algorithm starts from "
        "feasible SP, BS, and AE service-plan columns and solves a "
        "restricted master problem. Dual prices are passed to mode-specific "
        "pricing problems to generate negative reduced-cost columns. For "
        "$N \\le 100$, full pricing is continued until LP convergence and "
        "the reported gap is a Full-CG LP-IP gap. For large-scale "
        "instances, a strengthened budgeted-CG protocol with "
        "incumbent-column injection is used, and the reported gap is a "
        "generated-pool LP-IP gap rather than a complete-column global "
        "optimality certificate. The final integer schedule is recovered "
        "by solving the IRMP on the generated column pool.\n"
    )
    with open(os.path.join(OUT_DAT_DIR, "fig_algorithm_framework_caption.txt"),
              "w", encoding="utf-8") as f:
        f.write(caption)

    print("OK")
    for ext in ("pdf", "png", "svg"):
        print(f"  {base}.{ext}")


if __name__ == "__main__":
    main()

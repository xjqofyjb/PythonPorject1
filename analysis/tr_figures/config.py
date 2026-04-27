"""Centralized plotting configuration for TR-series journal figures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path("figures_tr_style")
PDF_DIR = ROOT / "pdf"
SVG_DIR = ROOT / "svg"
PNG_DIR = ROOT / "png"
ELSEVIER_TEMPLATE_DIR = Path(r"C:\Users\researcher\Desktop\Elsevier_s_CAS_LaTeX_Single_Column_Template")

METHOD_LABELS = {
    "cg": "CG+IR",
    "restricted_cg": "Restricted-CG",
    "rcg_random": "Restricted-CG",
    "rcg_arrival": "Restricted-CG",
    "rolling_horizon": "Rolling-Horizon MILP",
    "fix_and_optimize": "Fix-and-Optimize",
    "milp300": "MILP-300s",
    "milp60": "MILP-60s",
    "greedy": "Greedy",
    "fifo": "FIFO",
    "cg_basic": "CG-Basic",
    "cg_warm": "CG+Warm",
    "cg_stab": "CG+Stab",
    "cg_multik": "CG+MultiCol",
    "cg_full": "CG-Full",
}

METHOD_ORDER = [
    "cg",
    "restricted_cg",
    "rolling_horizon",
    "fix_and_optimize",
    "milp300",
    "milp60",
    "greedy",
    "fifo",
]

METHOD_ORDER_REDUCED = [
    "cg",
    "restricted_cg",
    "rolling_horizon",
    "fix_and_optimize",
    "greedy",
    "fifo",
]

METHOD_COLORS = {
    "cg": "#4C78A8",
    "restricted_cg": "#5B84B1",
    "rcg_random": "#5B84B1",
    "rcg_arrival": "#5B84B1",
    "rolling_horizon": "#7AA974",
    "fix_and_optimize": "#7E89B6",
    "milp300": "#6F6F6F",
    "milp60": "#9A9A9A",
    "greedy": "#E5988A",
    "fifo": "#728C7A",
    "cg_basic": "#4C78A8",
    "cg_warm": "#5B84B1",
    "cg_stab": "#7E89B6",
    "cg_multik": "#7AA974",
    "cg_full": "#E5988A",
}

METHOD_MARKERS = {
    "cg": "o",
    "restricted_cg": "s",
    "rcg_random": "s",
    "rcg_arrival": "s",
    "rolling_horizon": "^",
    "fix_and_optimize": "D",
    "milp300": None,
    "milp60": None,
    "greedy": "P",
    "fifo": "v",
    "cg_basic": "o",
    "cg_warm": "o",
    "cg_stab": "o",
    "cg_multik": "o",
    "cg_full": "o",
}

METHOD_LINESTYLES = {
    "cg": "-",
    "restricted_cg": "-",
    "rcg_random": "-",
    "rcg_arrival": "-",
    "rolling_horizon": "-",
    "fix_and_optimize": "-",
    "milp300": "-",
    "milp60": "-",
    "greedy": "-",
    "fifo": "-",
    "cg_basic": "-",
    "cg_warm": "--",
    "cg_stab": ":",
    "cg_multik": "-.",
    "cg_full": (0, (3, 1, 1, 1)),
}

REFERENCE_LINESTYLE = (0, (4, 2))
REFERENCE_COLOR = "#B8B8B8"

MODE_COLORS = {
    "SP": "#7AA974",
    "BS": "#E5988A",
    "AE": "#7F7F7F",
    "cargo": "#E5E2DE",
    "cargo_edge": "#B8B8B8",
}

COLORS_DUAL = {
    "adequate_cost": "#4C78A8",
    "adequate_ae": "#7AA974",
    "constrained_cost": "#5B84B1",
    "constrained_ae": "#E5988A",
}

FILL_COLORS = {
    "blue": "#D9E6EC",
    "salmon": "#F3D6CF",
    "green": "#DCE8D8",
    "gray": "#ECEAE7",
}

SCENARIO_LABELS = {"U": "Uniform", "P": "Peaked", "L": "Long service"}
MECHANISM_LABELS = {
    "hybrid": "Hybrid",
    "battery_only": "Battery only",
    "shore_only": "Shore only",
}


@dataclass(frozen=True)
class FigureSize:
    width: float
    height: float


SINGLE_COLUMN = FigureSize(3.35, 2.6)
DOUBLE_COLUMN = FigureSize(7.0, 3.35)
DOUBLE_COLUMN_TALL = FigureSize(7.0, 5.0)
DOUBLE_COLUMN_QUAD = FigureSize(7.1, 5.55)

LEGEND_BBOX_TOP = 1.01
LEGEND_EDGE = "#D0D0D0"
LEGEND_HANDLE_LENGTH = 1.9
LEGEND_COLUMN_SPACING = 0.9
LEGEND_BORDER_PAD = 0.45
LEGEND_HANDLE_TEXT_PAD = 0.6
PANEL_LABEL_FONT_SIZE = 8.7
PANEL_LABEL_PAD = 0.012
X_LABEL_SIZE = 8.8
X_LABEL_PAD = 9.0


UNIVERSAL_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 9.8,
    "axes.titlesize": 9.8,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "legend.fontsize": 7.9,
    "lines.linewidth": 1.75,
    "lines.markersize": 4.6,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "#D0D0D0",
    "legend.fancybox": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def apply_style() -> None:
    plt.rcParams.update(UNIVERSAL_STYLE)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

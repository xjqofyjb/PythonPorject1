"""Shared helpers for TR-style figure generation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .config import (
    LEGEND_BBOX_TOP,
    LEGEND_BORDER_PAD,
    LEGEND_COLUMN_SPACING,
    LEGEND_EDGE,
    LEGEND_HANDLE_LENGTH,
    LEGEND_HANDLE_TEXT_PAD,
    PANEL_LABEL_FONT_SIZE,
    PANEL_LABEL_PAD,
    PDF_DIR,
    PNG_DIR,
    REFERENCE_COLOR,
    SVG_DIR,
    X_LABEL_PAD,
    X_LABEL_SIZE,
    apply_style,
)


class MissingFigureInputError(RuntimeError):
    """Raised when a figure cannot be faithfully generated from available inputs."""


@dataclass
class ExportedFigure:
    pdf: Path
    svg: Path
    png: Path


def ensure_output_dirs() -> None:
    for path in (PDF_DIR, SVG_DIR, PNG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def export_figure(fig: plt.Figure, stem: str) -> ExportedFigure:
    ensure_output_dirs()
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    pdf_path = PDF_DIR / f"{stem}.pdf"
    svg_path = SVG_DIR / f"{stem}.svg"
    png_path = PNG_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    return ExportedFigure(pdf=pdf_path, svg=svg_path, png=png_path)


def load_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise MissingFigureInputError(f"Missing required data file: {file_path}")
    return pd.read_csv(file_path)


def resolve_existing_path(candidates: Sequence[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    rendered = ", ".join(str(Path(candidate)) for candidate in candidates)
    raise MissingFigureInputError(f"Missing required input file. Checked: {rendered}")


def mean_ci(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna()
    if clean.empty:
        return np.nan, np.nan
    mean = float(clean.mean())
    if len(clean) == 1:
        return mean, 0.0
    ci = 1.96 * float(clean.std(ddof=1)) / np.sqrt(len(clean))
    return mean, ci


def summarize(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, block in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        mean, ci = mean_ci(block[value_col])
        row[f"{value_col}_mean"] = mean
        row[f"{value_col}_ci"] = ci
        rows.append(row)
    return pd.DataFrame(rows)


def add_panel_labels(
    fig: plt.Figure,
    axes: Iterable[plt.Axes],
    subtitles: Sequence[str] | None = None,
    *,
    pad: float | Sequence[float] = PANEL_LABEL_PAD,
) -> None:
    axes_list = list(axes)
    subtitle_list = list(subtitles) if subtitles is not None else []
    pad_values = [pad] * len(axes_list) if not isinstance(pad, (list, tuple)) else list(pad)
    if len(pad_values) < len(axes_list):
        pad_values.extend([pad_values[-1]] * (len(axes_list) - len(pad_values)))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for idx, ax in enumerate(axes_list):
        label = chr(ord("a") + idx)
        text = f"({label})"
        if idx < len(subtitle_list) and subtitle_list[idx]:
            text = f"{text} {subtitle_list[idx]}"
        bbox = ax.get_position()
        tight_bbox = ax.get_tightbbox(renderer)
        tight_y0 = fig.transFigure.inverted().transform((0.0, tight_bbox.y0))[1]
        fig.text(
            (bbox.x0 + bbox.x1) / 2.0,
            tight_y0 - pad_values[idx],
            text,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONT_SIZE,
            fontweight="normal",
        )


def add_major_grid(ax: plt.Axes) -> None:
    ax.grid(True, axis="both", which="major", color="#D8D8D8", linewidth=0.45, alpha=0.75)


def hide_minor_grids(ax: plt.Axes) -> None:
    ax.minorticks_off()


def compact_legend(fig: plt.Figure, handles: list, labels: list, preferred: str = "top") -> None:
    if not handles:
        return
    loc = "upper center" if preferred == "top" else "lower center"
    bbox = (0.5, LEGEND_BBOX_TOP) if preferred == "top" else (0.5, -0.02)
    ncol = min(max(2, len(labels)), 4 if len(labels) <= 6 else 5)
    if len(labels) > 8 and preferred != "top":
        loc = "lower center"
        bbox = (0.5, -0.02)
    add_figure_legend(
        fig,
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox,
    )


def add_figure_legend(
    fig: plt.Figure,
    handles: list,
    labels: list,
    *,
    loc: str = "upper center",
    ncol: int = 3,
    bbox_to_anchor: tuple[float, float] = (0.5, LEGEND_BBOX_TOP),
) -> None:
    fig.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        edgecolor=LEGEND_EDGE,
        facecolor="white",
        handlelength=LEGEND_HANDLE_LENGTH,
        columnspacing=LEGEND_COLUMN_SPACING,
        borderpad=LEGEND_BORDER_PAD,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )


def make_style_handles(*, metric_colors: dict[str, str] | None = None, style_labels: dict[str, str] | None = None) -> list[Line2D]:
    handles: list[Line2D] = []
    if metric_colors:
        for label, color in metric_colors.items():
            handles.append(Line2D([0], [0], color=color, lw=1.8, marker="o", label=label))
    if style_labels:
        for label, linestyle in style_labels.items():
            handles.append(Line2D([0], [0], color="#555555", lw=1.8, linestyle=linestyle, label=label))
    return handles


def style_secondary_axis(ax: plt.Axes, color: str, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=color)
    ax.tick_params(axis="y", colors=color, labelsize=8.3)
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(color)
    ax.yaxis.label.set_color(color)
    ax.grid(False)
    ax.minorticks_off()


def set_x_axis_label(ax: plt.Axes, text: str, *, size: float = X_LABEL_SIZE, pad: float = X_LABEL_PAD) -> None:
    ax.set_xlabel(text)
    ax.xaxis.label.set_size(size)
    ax.xaxis.labelpad = pad


def format_thousands(x: float, _pos: float) -> str:
    return f"{x/1000:.0f}"


def apply_cost_formatter(ax: plt.Axes, *, thousands: bool = False) -> None:
    if thousands:
        ax.yaxis.set_major_formatter(FuncFormatter(format_thousands))
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:,.0f}"))


def apply_common_axis_format(ax: plt.Axes) -> None:
    hide_minor_grids(ax)
    add_major_grid(ax)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="both", width=0.75, length=3.6)


def start_figure() -> None:
    apply_style()

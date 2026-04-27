"""Figure 1: engineering-practice photo panel."""
from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .config import DOUBLE_COLUMN, ELSEVIER_TEMPLATE_DIR
from .utils import ExportedFigure, add_panel_labels, export_figure, resolve_existing_path, start_figure


def _center_crop(image: np.ndarray, target_ratio: float) -> np.ndarray:
    height, width = image.shape[:2]
    current_ratio = width / height
    if abs(current_ratio - target_ratio) < 1e-3:
        return image
    if current_ratio > target_ratio:
        new_width = int(round(height * target_ratio))
        margin = max((width - new_width) // 2, 0)
        return image[:, margin:margin + new_width]
    new_height = int(round(width / target_ratio))
    margin = max((height - new_height) // 2, 0)
    return image[margin:margin + new_height, :]


def build() -> ExportedFigure:
    start_figure()

    left = resolve_existing_path([Path("ltd.jpg"), ELSEVIER_TEMPLATE_DIR / "ltd.jpg"])
    right = resolve_existing_path([Path("cjhd.jpg"), ELSEVIER_TEMPLATE_DIR / "cjhd.jpg"])

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN.width, 2.85))
    target_ratio = 1.28
    images = [_center_crop(mpimg.imread(left), target_ratio), _center_crop(mpimg.imread(right), target_ratio)]
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#BEBEBE")
            spine.set_linewidth(0.6)
    fig.subplots_adjust(wspace=0.035, left=0.02, right=0.98, top=0.985, bottom=0.12)
    add_panel_labels(fig, axes, ["Shore power practice", "Battery swapping practice"], pad=0.012)
    return export_figure(fig, "Fig1_realworld_practice")


if __name__ == "__main__":
    build()

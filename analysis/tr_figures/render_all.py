"""Render the full TR-style figure set for Figures 2-8."""
from __future__ import annotations

from pathlib import Path

from .figure1_realworld import build as build_fig1
from .figure2_mechanism import build as build_fig2
from .figure3_main import build as build_fig3
from .figure4_scenarios import build as build_fig4
from .figure5_simops import build as build_fig5
from .figure6_sensitivity import build as build_fig6
from .figure7_policy import build as build_fig7
from .figure8_robustness import build as build_fig8


def main() -> None:
    exported = [
        ("Fig1", build_fig1()),
        ("Fig2", build_fig2()),
        ("Fig3", build_fig3()),
        ("Fig4", build_fig4()),
        ("Fig5", build_fig5()),
        ("Fig6", build_fig6()),
        ("Fig7", build_fig7()),
        ("Fig8", build_fig8()),
    ]
    for fig_id, outputs in exported:
        print(f"{fig_id}: {outputs.pdf}")
        print(f"{fig_id}: {outputs.svg}")
        print(f"{fig_id}: {outputs.png}")


if __name__ == "__main__":
    main()

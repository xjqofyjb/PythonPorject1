"""Run experiments and generate paper figures from a config."""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def _ensure_repo_root() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    repo = os.path.abspath(os.path.join(here, ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments and generate paper figures.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    _ensure_repo_root()

    from src.runner import load_config, run_experiment
    from analysis.make_plots import plot_paper
    from analysis.style import set_style

    cfg = load_config(args.config)
    run_experiment(args.config, cfg)

    set_style()
    df = pd.read_csv(cfg.get("output", "results/results_ablation.csv"))
    outdir = "figs/paper"
    traces_dir = cfg.get("trace_dir", "results/traces")
    plot_paper(df, outdir, traces_dir)


if __name__ == "__main__":
    main()
"""One-click pipeline runner."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


def run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


@dataclass(frozen=True)
class Stage:
    config: str
    csv: str
    experiment: str
    figdir: str
    table: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline.")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds")
    args = parser.parse_args()

    base = [sys.executable, "-m"]
    stages = [
        Stage("configs/main.yaml", "results/results_main_rigorous.csv", "main", "figs/main", "results/main_table.tex"),
        Stage("configs/scenario.yaml", "results/results_scenario_rigorous.csv", "scenario", "figs/scenario", "results/scenario_table.tex"),
        Stage("configs/simops.yaml", "results/results_simops_rigorous.csv", "simops", "figs/simops", "results/simops_table.tex"),
        Stage("configs/sensitivity.yaml", "results/results_sensitivity_rigorous.csv", "sensitivity", "figs/sensitivity", "results/sensitivity_table.tex"),
        Stage("configs/mechanism.yaml", "results/results_mechanism_rigorous.csv", "mechanism", "figs/mechanism", "results/mechanism_table.tex"),
        Stage("configs/ablation.yaml", "results/results_ablation_rigorous.csv", "paper", "figs/paper", None),
    ]

    for stage in stages:
        cmd = base + ["src.runner", "--config", stage.config]
        if args.seeds is not None:
            cmd += ["--seeds", str(args.seeds)]
        run(cmd)

        if stage.table is not None:
            run(base + ["analysis.make_tables", "--in", stage.csv, "--out", stage.table, "--experiment", stage.experiment])
        run(base + ["analysis.make_plots", "--in", stage.csv, "--outdir", stage.figdir, "--experiment", stage.experiment])

    run(base + ["analysis.build_paper_figures", "--results_dir", "results", "--outdir", "figs/paper"])


if __name__ == "__main__":
    main()

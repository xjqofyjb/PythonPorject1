# Experiment Pipeline (Publication-Ready)

This repo provides a reproducible, config-driven experimental pipeline with unified logging, CSV outputs, LaTeX tables, and publication-quality figures.

## Structure

```
configs/       # YAML configs
src/           # instance generation + solvers + runner
analysis/      # tables + plots + style
results/       # CSV outputs, logs, meta.json
figs/          # PDF/PNG figures
legacy/        # preserved original scripts
```

## Dependencies

Minimal:

```
pip install pyyaml numpy pandas matplotlib pulp
```

Optional (recommended for style):

```
pip install scienceplots
```

Optional (recommended for style):

## One-click pipeline

```
python -m pipeline
```

Override seed count (e.g., 5):

```
python -m pipeline --seeds 5
```

The pipeline now runs:

- main benchmark experiment
- scenario experiment
- SIMOPS vs sequential experiment
- sensitivity experiment
- mechanism experiment
- ablation experiment
- publication-focused composite figures under `figs/paper`

Detailed execution order and figure hierarchy:

```
EXPERIMENT_EXECUTION_PLAN.md
```

## Step-by-step commands (PowerShell)

Run main experiment:

```
python -m src.runner --config configs/main.yaml
```

Generate table and plots:

```
python -m analysis.make_tables --in results/results_main_rigorous.csv --out results/main_table.tex
python -m analysis.make_plots --in results/results_main_rigorous.csv --outdir figs/main
```

Run mechanism experiment:

```
python -m src.runner --config configs/mechanism.yaml
python -m analysis.make_tables --in results/results_mechanism_rigorous.csv --out results/mechanism_table.tex
python -m analysis.make_plots --in results/results_mechanism_rigorous.csv --outdir figs/mechanism --experiment mechanism
```

Run SIMOPS experiment:

```
python -m src.runner --config configs/simops.yaml
python -m analysis.make_tables --in results/results_simops_rigorous.csv --out results/simops_table.tex --experiment simops
python -m analysis.make_plots --in results/results_simops_rigorous.csv --outdir figs/simops --experiment simops
```

Run scenario experiment (ship structure):

```
python -m src.runner --config configs/scenario.yaml
python -m analysis.make_tables --in results/results_scenario_rigorous.csv --out results/scenario_table.tex
python -m analysis.make_plots --in results/results_scenario_rigorous.csv --outdir figs/scenario --experiment scenario
```

Run sensitivity experiment:

```
python -m src.runner --config configs/sensitivity.yaml
python -m analysis.make_tables --in results/results_sensitivity_rigorous.csv --out results/sensitivity_table.tex
python -m analysis.make_plots --in results/results_sensitivity_rigorous.csv --outdir figs/sensitivity --experiment sensitivity
```

Build composite paper figures:

```
python -m analysis.build_paper_figures --results_dir results --outdir figs/paper
```

Generate SIMOPS Gantt comparison (single instance):

```
python -m analysis.make_gantt --config configs/simops.yaml --N 10 --seed 1 --method cg --outdir figs/simops
```

## Paper / Ablation (Phase 2)

One-command run (includes traces + paper figures):

```
python src/run_experiments.py --config configs/ablation.yaml
```

Manual sequence:

```
python -m src.runner --config configs/ablation.yaml
python -m analysis.make_plots --in results/results_ablation.csv --outdir figs/paper --experiment paper --traces_dir results/traces
```

## Baselines

- FIFO: urgency-aware FIFO using (arrival_time, slack_time) ordering with mode selection.
- Greedy: per-ship myopic scheduling that enumerates feasible start times and modes to minimize incremental cost.
- MILP: Gurobi 60s / 300s time limit placeholders for small N.

## Reproducibility

- Seeds are controlled in YAML (or via `--seeds`).
- Every run writes to `results/meta.json` with config, timestamp, and platform details.
- All errors are logged to `results/logs/*.log` and written to the CSV `error` field.

## Notes

- MILP runs are automatically skipped for `N > 100` and marked as `skipped` in CSV.
- `pricing_model.pth` is optional; if missing or failing to load, CG falls back without blocking.
- Output figures are saved as both PDF (vector) and 300 dpi PNG.

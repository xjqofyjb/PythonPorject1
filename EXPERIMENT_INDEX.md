# Experiment and Output Index

Last updated: 2026-02-03

This document indexes all experiments, inputs, and outputs in this repo for paper writing or defense.

## 1) Experiments overview

### A) Main experiment (scaling study)
- Config: `configs/main.yaml`
- Run: `py -3.13 -m src.runner --config configs/main.yaml`
- Scope:
  - scenario: U
  - mechanism: hybrid
  - methods: cg, greedy, fifo, milp60, milp300
  - N: 20, 50, 100, 200, 500
  - seeds: 10
- Outputs:
  - CSV: `results/results_main_rigorous.csv`
  - Table: `results/main_table.tex`
  - Figures: `figs/main/Fig_Obj_U.*`, `figs/main/Fig_Runtime_U.*`, `figs/main/Fig_CG_Metrics_U.*`, `figs/main/Fig_Main_Combined_U.*`

### B) Sensitivity analysis
- Config: `configs/sensitivity.yaml`
- Run: `py -3.13 -m src.runner --config configs/sensitivity.yaml`
- Scope:
  - scenario: U
  - mechanism: hybrid
  - methods: cg, greedy, fifo, milp60, milp300
  - N: 100
  - seeds: 10
  - parameter sweeps:
    - battery_cost: 30, 40, 50, 60, 80, 100
    - shore_cap: 1, 2, 3, 5, 8
    - deadline_tightness: 0.8, 1.0, 1.2
- Outputs:
  - CSV: `results/results_sensitivity_rigorous.csv`
  - Table: `results/sensitivity_table.tex`
  - Figures: `figs/sensitivity/Fig_Sens_BatteryCost.*`, `figs/sensitivity/Fig_Sens_ShoreCap.*`, `figs/sensitivity/Fig_Sens_Deadline.*`

### C) Mechanism comparison
- Config: `configs/mechanism.yaml`
- Run: `py -3.13 -m src.runner --config configs/mechanism.yaml`
- Scope:
  - scenario: U
  - mechanisms: hybrid, battery_only, shore_only
  - methods: cg, greedy, fifo
  - N: 100
  - seeds: 10
- Outputs:
  - CSV: `results/results_mechanism_rigorous.csv`
  - Table: `results/mechanism_table.tex`
  - Figures: `figs/mechanism/Fig_Mechanism_U.*`, `figs/mechanism/Fig_Mechanism_Modes_U.*`

### D) Scenario (ship structure)
- Config: `configs/scenario.yaml`
- Run: `py -3.13 -m src.runner --config configs/scenario.yaml`
- Scope:
  - scenarios: U, P, L
  - mechanism: hybrid
  - methods: cg, greedy, fifo, milp60, milp300
  - N: 100
  - seeds: 10
- Outputs:
  - CSV: `results/results_scenario_rigorous.csv`
  - Table: `results/scenario_table.tex`
  - Figures: `figs/scenario/Fig_Scenario_Obj.*`, `figs/scenario/Fig_Scenario_Runtime.*`, `figs/scenario/Fig_Scenario_Modes.*`

### E) Ablation / paper figures
- Config: `configs/ablation.yaml`
- One-command run (includes figures): `py -3.13 src/run_experiments.py --config configs/ablation.yaml`
- Manual run (data only): `py -3.13 -m src.runner --config configs/ablation.yaml`
- Scope:
  - scenario: U
  - mechanism: hybrid
  - methods:
    - cg_basic, cg_warm, cg_stab, cg_multik, cg_full
    - fifo, greedy, milp60, milp300
  - N: 20, 50, 100, 200, 500
  - seeds: 5
- Outputs:
  - CSV: `results/results_ablation_rigorous.csv`
  - Traces: `results/traces/*.csv` (per-instance CG iteration logs)
  - Figures: `figs/paper/Fig_Paper_Obj.*`, `figs/paper/Fig_Paper_Runtime.*`, `figs/paper/Fig_Convergence_N500.*`, `figs/paper/Fig_CG_Traces_N500.*`, `figs/paper/Fig_Ablation_Bars.*`

### F) SIMOPS mechanism validation
- Config: `configs/simops.yaml`
- Run: `py -3.13 -m src.runner --config configs/simops.yaml`
- Scope:
  - scenario: U
  - operation_modes: simops, sequential
  - methods: cg, greedy, fifo
  - N: 25, 50, 100, 200, 500
  - seeds: 10
- Outputs:
  - CSV: `results/results_simops_rigorous.csv`
  - Table: `results/simops_table.tex`
  - Figures: `figs/simops/Fig_SIMOPS_Cost_Comparison.*`, `figs/simops/Fig_SIMOPS_Cost_Savings.*`, `figs/simops/Fig_SIMOPS_Stay_Time.*`, `figs/simops/Fig_SIMOPS_Mode_Distribution.*`, `figs/simops/Fig_SIMOPS_Masking_Distribution.*`

### G) SIMOPS (strong contrast)
- Config: `configs/simops_strong.yaml`
- Run: `py -3.13 -m src.runner --config configs/simops_strong.yaml`
- Scope:
  - scenario: U
  - operation_modes: simops, sequential
  - methods: cg, greedy, fifo
  - N: 25, 50, 100, 200, 500
  - seeds: 10
  - params (key changes): cargo_time_scale=1.2, deadline_tightness=1.1
- Outputs:
  - CSV: `results/results_simops_strong_rigorous.csv`
  - Table: `results/simops_strong_table.tex`
  - Figures: `figs/simops_strong/Fig_SIMOPS_Cost_Comparison.*`, `figs/simops_strong/Fig_SIMOPS_Cost_Savings.*`, `figs/simops_strong/Fig_SIMOPS_Stay_Time.*`, `figs/simops_strong/Fig_SIMOPS_Mode_Distribution.*`, `figs/simops_strong/Fig_SIMOPS_Masking_Distribution.*`

## 2) Tables index (LaTeX)
- Main: `results/main_table.tex` (from `results/results_main_rigorous.csv`)
- Sensitivity: `results/sensitivity_table.tex` (from `results/results_sensitivity_rigorous.csv`)
- Mechanism: `results/mechanism_table.tex` (from `results/results_mechanism_rigorous.csv`)
- Scenario: `results/scenario_table.tex` (from `results/results_scenario_rigorous.csv`)
- SIMOPS: `results/simops_table.tex` (from `results/results_simops_rigorous.csv`)
- SIMOPS (strong): `results/simops_strong_table.tex` (from `results/results_simops_strong_rigorous.csv`)

## 3) Figures index (purpose)
- Main scaling:
  - `Fig_Obj_U`: objective vs N (method comparison)
  - `Fig_Runtime_U`: runtime vs N
  - `Fig_CG_Metrics_U`: CG pricing calls, iterations, pricing time share vs N
  - `Fig_Main_Combined_U`: combined panel (objective, runtime, CG metrics)
- Sensitivity:
  - `Fig_Sens_BatteryCost`: objective vs battery_cost
  - `Fig_Sens_ShoreCap`: objective vs shore_cap
  - `Fig_Sens_Deadline`: objective vs deadline_tightness
- Mechanism:
  - `Fig_Mechanism_U`: objective by mechanism and method
  - `Fig_Mechanism_Modes_U`: mode ratios (shore/battery/hybrid) by mechanism
- Scenario:
  - `Fig_Scenario_Obj`: objective by scenario and method
  - `Fig_Scenario_Runtime`: runtime by scenario and method
  - `Fig_Scenario_Modes`: mode ratios by scenario and method
- Ship types:
  - `Fig_Type_Cost`: average cost per ship by type
  - `Fig_Type_Modes`: mode ratios by ship type
- Paper / ablation:
  - `Fig_Paper_Obj`: objective vs N (CG variants + baselines)
  - `Fig_Paper_Runtime`: runtime vs N (CG variants + baselines)
  - `Fig_Convergence_N500`: convergence over time (N=500)
  - `Fig_CG_Traces_N500`: CG trace diagnostics (RMP objective, pricing calls, min reduced cost)
  - `Fig_Ablation_Bars`: CG ablation bars (iterations, runtime, pricing calls)
- SIMOPS:
  - `Fig_SIMOPS_Cost_Comparison`: objective comparison (simops vs sequential)
  - `Fig_SIMOPS_Cost_Savings`: percent savings by N
  - `Fig_SIMOPS_Stay_Time`: average stay time comparison
  - `Fig_SIMOPS_Mode_Distribution`: average mode ratio comparison
  - `Fig_SIMOPS_Masking_Distribution`: masking rate histogram
  - `Fig_SIMOPS_Masking_By_Type`: masking rates by ship type
  - `Fig_SIMOPS_Gantt_Comparison`: Gantt chart (single instance)

## 4) Key metrics (CSV columns)
- Objective/efficiency: `obj`, `runtime_total`, `gap` (MILP), `status`, `error`
- CG-specific: `num_iters`, `num_pricing_calls`, `num_columns_added`, `runtime_pricing`, `pricing_time_share`, `min_reduced_cost_last`
- Mechanism usage: `mode_shore_count`, `mode_battery_count`, `mode_brown_count`, `shore_ratio`, `battery_ratio`, `brown_ratio`, `shore_utilization`
- Ship types: `type_{A,B,C,D}_count`, `type_{A,B,C,D}_cost_total`, `type_{A,B,C,D}_shore_ratio`, `type_{A,B,C,D}_battery_ratio`, `type_{A,B,C,D}_brown_ratio`
- Cost proxies (scaled): `cost_energy`, `cost_delay`
- SIMOPS: `operation_mode`, `avg_masking_rate`, `num_fully_masked`, `num_partially_masked`, `time_saved_total`, `avg_stay_time`

## 5) Data hygiene notes
- Duplicates were removed from main and ablation results on 2026-02-03.
- Legacy columns `iters`, `pricing_calls`, `runtime` were removed after backfilling `num_iters`, `num_pricing_calls`, `runtime_total`.
- Backups are preserved in `results/*.bak` and `results/*.bak2`.

## 6) Reproducibility
- Each run writes metadata to `results/meta.json` (config, timestamp, platform, git commit if available).
- All figures are saved as PDF and 300 dpi PNG.

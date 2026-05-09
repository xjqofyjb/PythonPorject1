# Figure 4 Data Sources

## CSV inputs
- `results/revised/scenario_comparison_raw.csv` -- corrected scenario
  benchmark at N = 100. CG+IR rows are validated as
  `cg_status = full_pricing_converged` and `gap_type = Full-CG LP-IP gap`.
- `results/revised/mechanism_comparison_raw.csv` -- corrected mechanism
  comparison at N = 100. CG+IR rows are validated as
  `cg_status = full_pricing_converged`. Cross-checked against
  `results/revised/mechanism_comparison_summary.csv`.

## Inclusion rules
- Old weak quick-CG outputs are NOT read; both source CSVs only contain
  full-pricing CG+IR rows after explicit validation.
- Method names are standardized: `fix_and_optimize` -> `Fix-and-Optimize`,
  `rolling_horizon` -> `Rolling-Horizon`.
- Panel (a) shows U / P / L scenarios for the five comparison methods
  (CG+IR, Rolling-Horizon, Fix-and-Optimize, FIFO, Greedy).
- Panel (b) shows the four service mechanisms (Hybrid, Battery-only,
  Shore-power only, Green-only no-AE) for the three methods that exercise
  the mechanism switch (CG+IR, Rolling-Horizon, Fix-and-Optimize).
- For each (mechanism, method) bar in panel (b), the cost penalty vs the
  Hybrid reference is computed per scenario and then averaged across U/P/L.
- Bars where every seed was infeasible (pure-mode green-only constraint)
  are drawn as a hatched grey patch and labelled `infeasible'.

## Aggregation
- Panel (a): per (scenario, method) mean and std of objective across seeds.
- Panel (b): per (scenario, method, mechanism) seed-mean of objective,
  divided by the same-(scenario, method) Hybrid mean to get a per-scenario
  cost-penalty percentage; then averaged across scenarios.

## Outputs
- `figures/revised/manuscript/fig4_scenario_mechanism_final.png`
- `figures/revised/manuscript/fig4_scenario_mechanism_final.pdf`
- `results/revised/manuscript/fig4_scenario_panel_data.csv`
- `results/revised/manuscript/fig4_mechanism_panel_data.csv`
- `results/revised/manuscript/fig4_caption.txt`
- `results/revised/manuscript/fig4_data_sources.md`

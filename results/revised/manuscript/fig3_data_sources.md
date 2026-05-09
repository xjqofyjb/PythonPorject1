# Figure 3 Data Sources

## CSV inputs
- `results/revised/table8_revised.csv` - corrected small-scale benchmark
  summary used for N = 20, 50, 100. CG+IR rows have
  `cg_status = full_pricing_converged` and `gap_type = Full-CG LP-IP gap`.
- `results/revised/final_check/table8_final_controlled.csv` - final
  controlled replacement summary used for N = 200 (scenarios U/P/L) and
  N = 500 (scenario U only). CG+IR rows have
  `gap_type = Pool LP-IP gap` (budgeted-stabilized pricing).

## Diagnostic / validation inputs
- `results/revised/final_check/table8_validation_report.md`
- `results/revised/final_check/final_check_diagnostic_report.md`

## Inclusion rules
- Old weak quick-CG outputs (e.g. `cg_status = budgeted_topK` CG+IR rows
  for N = 200 / N = 500 in `table8_revised.csv`) are NOT read.
- N = 20 / 50 / 100 rows are taken only from the corrected small-scale
  summary and are validated to be `full_pricing_converged`.
- N = 200 and N = 500 rows are taken only from the final controlled
  replacement summary and are validated to be `Pool LP-IP gap`.
- N = 500 includes scenario U only; P/L were not run in the controlled
  replacement due to the runtime budget.
- CG+IR gap labels are preserved in the figure: Full-CG LP-IP gap for
  N <= 100, Pool LP-IP gap for N = 200 and N = 500.

## Aggregation
- Per row, the relative gap to CG+IR is recomputed from `objective_mean`
  against the CG+IR objective at the same (N, scenario) - we do not rely
  on stale `gap_pct` columns.
- For N = 20, 50, 100 and N = 200, U/P/L are averaged equally per method
  to produce a per-(N, method) summary.
- For N = 500, only scenario U is available and is shown without averaging.

## Methods included
- CG+IR (the column-generation upper bound)
- Restricted-CG
- Rolling-Horizon
- Fix-and-Optimize
- FIFO
- Greedy

## Outputs
- `figures/revised/manuscript/fig3_benchmark_final.png`
- `figures/revised/manuscript/fig3_benchmark_final.pdf`
- `results/revised/manuscript/fig3_benchmark_final_plot_data.csv`
- `results/revised/manuscript/fig3_benchmark_per_scenario.csv`
- `results/revised/manuscript/fig3_benchmark_final_caption.txt`
- `results/revised/manuscript/fig3_data_sources.md`

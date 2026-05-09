# Figure 9 Data Sources

## CSV inputs
- `results/revised/arrival_perturbation_raw.csv` -- corrected
  arrival-perturbation runs at N = 100, scenario U. CG+IR rows are
  validated to be `cg_status = full_pricing_converged` and
  `gap_type = Full-CG LP-IP gap`.
- `results/revised/arrival_perturbation_summary.csv` -- pre-aggregated
  reference summary used as a cross-check.

## Inclusion rules
- Old / weak quick-CG outputs are NOT read; the script asserts
  full-pricing convergence on every CG+IR row before plotting.
- Only CG+IR rows are plotted; this figure shows the framework's
  intrinsic robustness to arrival perturbation, independent of
  baseline-method effects.
- Slack configurations: `loose` and `tight`.
- Perturbation types: `symmetric` and `one_sided_delay`.
- Perturbation amplitudes: $\Delta \in \{0, 1, 2\}$ hours.

## Aggregation
- Per (slack, perturbation_type, Delta): seed mean and standard deviation
  of total cost are computed from the raw CSV.
- Relative cost increase is computed against the same-(slack,
  perturbation_type) row at $\Delta = 0$.

## Outputs
- `figures/revised/manuscript/fig9_arrival_perturbation_final.png`
- `figures/revised/manuscript/fig9_arrival_perturbation_final.pdf`
- `results/revised/manuscript/fig9_panel_data.csv`
- `results/revised/manuscript/fig9_caption.txt`
- `results/revised/manuscript/fig9_data_sources.md`

# Final-Check Audit

- Python version: 3.13.9
- Solver version: Gurobi 13.0.0
- Git commit hash: 66ba66fd661bc96697c170bb875a87de1a646384
- Source folders used: `src/`, `experiments/`, `analysis/`, `tests/`
- New output roots: `results/revised/final_check/`, `figures/revised/final_check/`, `logs/revised/final_check/`

## Full-Controlled Files Read

- `results/revised/full_controlled/simops_dual_peak_full_raw.csv`
- `results/revised/full_controlled/simops_dual_peak_full_summary.csv`
- `results/revised/full_controlled/simops_dual_peak_diagnostics.csv`
- `results/revised/full_controlled/table8_final_controlled.csv`
- `results/revised/full_controlled/table8_final_controlled.tex`
- `results/revised/full_controlled/n500_table8_full_replacement_summary.csv`

## New Final-Check Experiments

- `metadata_display_final_check`
- `dual_peak_enrichment_final_check`
- `final_dual_peak_outputs`
- `bs_cost_threshold_fine_grid_final_check`
- `final_bs_outputs`
- `table8_final_check`

## CG Settings

- N <= 100: full pricing / complete generated pool for compact experimental instances; gap type is `Full-CG LP-IP gap` when pricing converges.
- N = 110, 125, 150 enrichment: strengthened budgeted CG, `min_iters=20`, `max_iters=50`, `pricing_top_k=10`, incumbent injection enabled, sequential selected columns injected into SIMOPS.
- N = 200: strengthened budgeted CG from full-controlled replacement, incumbent injection enabled, Pool LP-IP gap reported unless pricing truly converges.
- N = 500: strengthened budgeted CG from full-controlled replacement, incumbent injection enabled, Pool LP-IP gap reported unless pricing truly converges.

## Exclusions and Interpretation

- Old weak quick-CG results are excluded from final-check outputs.
- Large-scale pricing did not fully converge in the controlled N=200/N=500 runs.
- `Full-CG LP-IP gap` means pricing converged for the complete pricing scan used by the run.
- `Pool LP-IP gap` means the integer solution is compared only against the final generated-pool LP relaxation.

For N=200 and N=500, the reported large-scale gaps are generated-pool LP-IP gaps. They indicate pool integrality tightness and incumbent stability, not complete-column global optimality.

# Full Controlled Run Audit

- Git commit: `66ba66fd661bc96697c170bb875a87de1a646384`
- Python: `3.13.9`
- Solver: `Gurobi 13.0.0`
- Platform: `Windows-11-10.0.26100-SP0`
- Hardware: `Intel64 Family 6 Model 183 Stepping 1, GenuineIntel`

## Config Files Used

- `configs/main.yaml`
- `configs/simops.yaml`
- `configs/sensitivity.yaml`
- Controlled overrides in `experiments/run_revision_experiments.py`

## CG Settings

### N <= 100

- Full generated-pool solve with `use_full_pool_small=True`, `full_pool_n=100`.
- Report `cg_status=full_pricing_converged`, `gap_type=Full-CG LP-IP gap`, `pricing_converged=True` only when no negative reduced-cost columns remain under full generated pricing.

### N = 200

- Strengthened budgeted CG.
- `min_iters=20`, `max_iters=30`, `pricing_top_k=5`, `pricing_eps=1e-6`.
- Greedy/FIFO incumbent injection enabled.
- External incumbent injection available through `incumbent_solutions`.
- Status is `budgeted_stabilized` only if last-5 relative improvement `<1e-4` and `pool_gap_pct <0.01`; otherwise `budgeted_max_iter` unless pricing converges.

### N = 500

- Same strengthened budgeted CG as N=200.
- Results remain generated-pool evidence unless pricing converges.

## Incumbent Injection

- `use_incumbent_injection=True` injects Greedy and FIFO selected service-plan columns.
- SIMOPS dominance runs solve sequential first and inject selected sequential columns into SIMOPS.

## SIMOPS/Sequential Dominance Procedure

1. Solve sequential.
2. Extract selected sequential columns.
3. Solve SIMOPS with sequential incumbent columns injected.
4. Record `dominance_check_passed = simops_obj <= sequential_obj + tolerance`.

## Stopping Rules

- Pricing convergence: `num_negative_columns_scanned_last == 0` or `min_reduced_cost_scanned_last >= -pricing_tolerance`.
- Objective stabilization: `iteration >= min_iter`, `relative_improvement_last_5 < 1e-4`, and `pool_gap_pct < 0.01`.

## Gap Definitions

- `Full-CG LP-IP gap`: LP-IP gap after full pricing convergence.
- `Pool LP-IP gap`: LP-IP gap over the generated budgeted column pool only.

For N=200 and N=500, pricing convergence is not required for reporting, but pricing_converged must remain False unless no negative reduced-cost columns remain. These large-scale results are generated-pool, budgeted-stabilized solutions.

## Old Files That Must NOT Be Reused

- `results/revised/main_benchmark_raw.csv` for N=200/N=500 conclusions.
- `results/revised/simops_dual_peak_raw.csv` for N=200/N=500 dual-peak conclusions.
- Any old quick-CG N=200/N=500 rows not under `results/revised/full_controlled/`.
- Old figures under `figures/revised/` for controlled full-run conclusions.

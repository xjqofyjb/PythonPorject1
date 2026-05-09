# Final Controlled Diagnostic Report

Generated under `results/revised/full_controlled/`, `figures/revised/full_controlled/`, and `logs/revised/full_controlled/`.

## 1. Tests

All tests passed after metadata cleanup and table-generation updates:

```text
12 passed in 4.85s
```

The saved log is `logs/revised/full_controlled/pytest_before_full_run.txt`.

## 2. N=500 cautious expansion

The N=500 cautious expansion passed.

- Seeds tested: 1, 2, 3, 4, 5.
- CG+IR beat the best scalable baseline in 5/5 seeds.
- Objective stabilized rate: 1.000.
- Average `pool_gap_pct`: 0.000232.
- Pricing fully converged in any cautious-expansion run: no.
- Ready for 10-seed N=500 expansion: yes.

## 3. N=200 10-seed replacement

The N=200 replacement passed for scenarios U, P, and L with seeds 1-10.

- CG+IR was the best mean objective in all three scenarios.
- No baseline-win diagnostic rows were produced.
- CG metadata is reported as Pool LP-IP gap because the N=200 runs used strengthened budgeted CG.
- Pricing convergence rate for CG+IR was 0.0 in U, P, and L.
- Objective stabilized rate was 1.0 in U/P and 0.9 in L.

## 4. N=500 10-seed replacement

The N=500 replacement ran for scenario U with seeds 1-10.

- CG+IR beat the best scalable baseline in 10/10 seeds.
- CG+IR mean objective: 1,690,831.877.
- Best baseline mean objective: Fix-and-Optimize, 1,812,278.477.
- CG+IR objective stabilized rate: 1.0.
- CG+IR pricing convergence rate: 0.0.
- Mean `pool_gap_pct`: 0.000225.

## 5. Scalable baseline comparison

CG+IR beat or tied scalable baselines at N=200 and N=500 in the controlled replacement outputs.

- N=200 U/P/L: CG+IR had the lowest mean objective in every scenario.
- N=500 U: CG+IR beat the best scalable baseline in every seed.

## 6. Dual-peak dominance checks

Dual-peak full controlled rerun passed dominance checks for all N and seeds.

- `dominance_pass_rate` is 1.0 for every N.
- No `simops_dominance_failures.csv` was produced.
- Sequential columns were injected into the SIMOPS pool for the comparison protocol.

Revised SIMOPS savings by N:

| N | SIMOPS saving mean (%) |
|---:|---:|
| 25 | 20.291 |
| 50 | 8.370 |
| 75 | 11.257 |
| 90 | 17.510 |
| 100 | 19.709 |
| 110 | 19.042 |
| 125 | 15.916 |
| 150 | 12.706 |
| 200 | 8.839 |
| 500 | 3.936 |

The dual-peak pattern is still present in the controlled rerun: savings rise to a high region around N=90-110 and decline at larger N.

## 7. Large-scale pricing convergence

Pricing did not fully converge for the large-scale budgeted runs.

For large-scale instances, pricing did not fully converge in all runs. The reported large-scale gaps are generated-pool LP-IP gaps and should be interpreted as evidence of pool integrality tightness and incumbent stability, not as complete-column global optimality certificates.

## 8. Pool LP-IP gap interpretation

`Full-CG LP-IP gap` applies only when pricing fully converges and no negative reduced-cost columns remain within tolerance.

`Pool LP-IP gap` is computed between the final generated-pool LP objective and the IRMP objective:

```text
gap_pct = 100 * (irmp_obj - lp_obj_final_pool) / max(1, abs(irmp_obj))
```

For N=200 and N=500, this is not a global optimality gap over the complete column universe.

## 9. BS cost sensitivity

The BS cost sensitivity rerun generated raw, summary, PNG, and PDF outputs from CSV data.

The corrected cost scaling changes both cost levels and service structure at high C_BS values. In the controlled summary, CG+IR keeps high BS usage through C_BS = 0.80, but at C_BS = 1.00 and 1.20 the mean BS share drops to 0 and AE share rises to 0.876. This should be treated as a structural threshold rather than only a cost-level effect.

## 10. Readiness for manuscript update

The controlled outputs are ready as corrected numerical evidence for revising tables and figures, with two qualifications:

- The manuscript text should not claim global optimality for N=200 or N=500.
- Large-scale conclusions should explicitly say they are based on budgeted-stabilized generated-pool solutions with incumbent injection and dominance checks.

## 11. Remaining risks

- N=500 was run only for scenario U in the full replacement table due runtime budget; P and L remain optional extensions.
- Large-scale pricing scans still found negative reduced-cost columns, so the generated pool may be improvable.
- N=110, 125, and 150 dual-peak points are budgeted large runs with objective stabilization rates below 1.0 for some points.
- The table currently combines the controlled N=200 and N=500 replacement slices; it does not regenerate N=20, 50, or 100 in `full_controlled/`.

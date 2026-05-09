# Final-Check Diagnostic Report

## 1. Tests

All tests passed.

```text
12 passed in 4.84s
```

The saved test log is `logs/revised/final_check/pytest_final_check.txt`.

## 2. Metadata Display

Metadata display issues were fixed at the summary/display layer without changing raw solver data.

- Full-pricing N <= 100 rows now use `objective_stabilized_rate_display = N/A`.
- `stabilization_applicable = False` for full-pricing converged N <= 100 rows.
- Budgeted rows keep numeric objective-stabilization rates.

Validation output: `results/revised/final_check/metadata_display_check.csv`.

## 3. BS Delay and Emissions

BS delay and emissions fields were fixed for final-check BS outputs.

- `avg_delay_h_mean` is no longer NaN.
- `emissions_total_tCO2_mean` is computed from vessel energy and service mode.
- SP and BS use grid factor 0.445 kgCO2/kWh.
- AE uses 0.70 kgCO2/kWh.
- Emissions are not mode shares.

Validation output: `results/revised/final_check/bs_metric_validation.csv`.

## 4. N=110/125/150 Enrichment

The targeted enrichment passed dominance checks.

| N | Original saving (%) | Enriched saving (%) | Absolute change |
|---:|---:|---:|---:|
| 110 | 19.041895 | 19.181387 | 0.139492 |
| 125 | 15.915600 | 16.037829 | 0.122229 |
| 150 | 12.706133 | 12.809057 | 0.102925 |

Dominance pass rate remained 1.0 for N=110, 125, and 150.

## 5. Enrichment Decision

Enrichment did not materially change SIMOPS savings because all absolute changes were <= 0.5 percentage points.

Final dual-peak figure therefore keeps the original full-controlled values for N=110, 125, and 150, with enrichment used as stability evidence.

## 6. Final Dual-Peak Values

Final values used in `fig5_dual_peak_final`:

| N | SIMOPS saving mean (%) | Dominance pass rate | Gap type |
|---:|---:|---:|---|
| 25 | 20.291304 | 1.0 | Full-CG LP-IP gap |
| 50 | 8.369753 | 1.0 | Full-CG LP-IP gap |
| 75 | 11.256622 | 1.0 | Full-CG LP-IP gap |
| 90 | 17.509883 | 1.0 | Full-CG LP-IP gap |
| 100 | 19.708778 | 1.0 | Full-CG LP-IP gap |
| 110 | 19.041895 | 1.0 | Pool LP-IP gap |
| 125 | 15.915600 | 1.0 | Pool LP-IP gap |
| 150 | 12.706133 | 1.0 | Pool LP-IP gap |
| 200 | 8.838683 | 1.0 | Pool LP-IP gap |
| 500 | 3.936078 | 1.0 | Pool LP-IP gap |

The final dual-peak / dual-high-region pattern remains. The manuscript should avoid overstating this as a mathematically sharp peak; a better wording is “dual-high-region / threshold-sensitive SIMOPS value pattern.”

## 7. BS-AE Cost Threshold

The detected substitution threshold is C_BS = 0.90 $/kWh under the final-check fine grid.

At the threshold:

- Before threshold: mean BS share = 0.848, mean AE share = 0.013.
- At/after threshold: mean BS share = 0.000, mean AE share = 0.876.

This is an abrupt structural threshold. The final manuscript-supported statement is:

“Below the AE-equivalent threshold, battery cost mainly changes the cost level while preserving service structure; once the swapping unit cost crosses the fallback threshold, BS is displaced by AE, revealing a discontinuous substitution pattern.”

## 8. Table 8 Validation

Table 8 validation passed.

- Old quick results were excluded.
- N=200 and N=500 rows come from controlled replacement outputs.
- N=200 and N=500 CG+IR rows use `Pool LP-IP gap`.
- Method names are standardized.
- No all-NaN rows were found.
- N=500 is U-only; P/L were not run in the controlled replacement due runtime budget.

Validation output: `results/revised/final_check/table8_validation_report.md`.

## 9. Large-Scale Pricing

N=200 and N=500 remain correctly labeled as Pool LP-IP gap.

Large-scale pricing did not fully converge in the controlled replacement runs. Pricing convergence rate is 0.0 for the final N=200/N=500 CG+IR Table 8 rows and for N=110/125/150/200/500 final dual-peak rows.

For large-scale instances, pricing did not fully converge in all runs. The reported large-scale gaps are generated-pool LP-IP gaps and should be interpreted as evidence of pool integrality tightness and incumbent stability, not as complete-column global optimality certificates.

## 10. Final Manuscript Caveats

- Do not claim global optimality for N=200 or N=500.
- Describe N=200/N=500 as budgeted-stabilized generated-pool results.
- State that the Pool LP-IP gap is not a complete-column global optimality gap.
- State that N=500 final Table 8 replacement is scenario U only.
- For BS sensitivity, report the structural threshold at C_BS = 0.90 $/kWh rather than saying battery cost only changes cost levels.

## 11. Manuscript Readiness

The final-check outputs are manuscript-ready with the caveats above. No failed dominance checks, invalid emissions fields, or incorrect large-scale gap labels remain in the final-check deliverables.

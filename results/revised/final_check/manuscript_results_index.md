# Manuscript Results Index

Use only the files listed below for manuscript revision. These final-check outputs exclude old weak quick-CG results and preserve large-scale Pool LP-IP gap labeling.

## Tables

### Table 8 benchmark

- `results/revised/final_check/table8_final_controlled.tex`
- `results/revised/final_check/table8_final_controlled.csv`

Interpretation: controlled replacement benchmark for N=200 U/P/L and N=500 U. N=200 and N=500 CG+IR rows are generated-pool budgeted-CG results, not global optimality certificates.

### N=500 replacement

- `results/revised/final_check/n500_table8_full_replacement_summary.csv`

Interpretation: N=500 scenario U only. CG+IR beat the best scalable baseline in 10/10 seeds in the controlled replacement run.

### Dual peak

- `results/revised/final_check/simops_dual_peak_final_summary.csv`

Interpretation: SIMOPS dominance passed for every N and seed. The pattern is best described as a dual-high-region / threshold-sensitive SIMOPS value pattern, with high values around N=25 and N=90-110, then declining at N=200/500.

### BS sensitivity

- `results/revised/final_check/bs_cost_sensitivity_final_summary.csv`

Interpretation: below the fallback threshold, BS cost mainly changes total cost while preserving service structure; at the threshold, BS is displaced by AE.

### BS threshold

- `results/revised/final_check/bs_threshold_detection.csv`

Interpretation: the detected BS-AE substitution threshold is C_BS = 0.90 $/kWh in the tested grid.

## Figures

### Dual peak

- `figures/revised/final_check/fig5_dual_peak_final.pdf`
- `figures/revised/final_check/fig5_dual_peak_final.png`

### BS cost sensitivity

- `figures/revised/final_check/fig6a_bs_cost_sensitivity_final.pdf`
- `figures/revised/final_check/fig6a_bs_cost_sensitivity_final.png`

### BS mode-share threshold

- `figures/revised/final_check/fig6a_bs_cost_mode_share_final.pdf`
- `figures/revised/final_check/fig6a_bs_cost_mode_share_final.png`

## Diagnostics

- `results/revised/final_check/final_check_audit.md`
- `results/revised/final_check/table8_validation_report.md`
- `results/revised/final_check/dual_peak_enrichment_diagnostics.csv`
- `results/revised/final_check/final_check_diagnostic_report.md`

## Required Caveats

- N=500 is U-only in the final controlled Table 8 replacement.
- N=200 and N=500 should be labeled as Pool LP-IP gap unless full pricing convergence is later demonstrated.
- Large-scale pricing did not fully converge, so large-scale results should be described as budgeted-stabilized generated-pool evidence.
- BS/AE emissions use grid factor 0.445 kgCO2/kWh for SP and BS, and AE factor 0.70 kgCO2/kWh for AE.

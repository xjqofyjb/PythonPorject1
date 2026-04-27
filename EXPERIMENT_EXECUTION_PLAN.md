# Experiment Script Checklist And Execution Order

This project now has two output layers:

1. `analysis.make_plots`
   Generates experiment-specific diagnostic figures for each CSV.
2. `analysis.build_paper_figures`
   Generates composite, publication-focused figures intended for the paper body.

## Recommended run order

Run the experiments in the following order so the paper figures can reuse all result CSVs.

1. Main scalability and benchmark experiment
```powershell
python -m src.runner --config configs/main.yaml
python -m analysis.make_tables --in results/results_main_rigorous.csv --out results/main_table.tex --experiment main
python -m analysis.make_plots --in results/results_main_rigorous.csv --outdir figs/main --experiment main
```

2. Scenario stress-test experiment
```powershell
python -m src.runner --config configs/scenario.yaml
python -m analysis.make_tables --in results/results_scenario_rigorous.csv --out results/scenario_table.tex --experiment scenario
python -m analysis.make_plots --in results/results_scenario_rigorous.csv --outdir figs/scenario --experiment scenario
```

3. SIMOPS vs sequential mechanism experiment
```powershell
python -m src.runner --config configs/simops.yaml
python -m analysis.make_tables --in results/results_simops_rigorous.csv --out results/simops_table.tex --experiment simops
python -m analysis.make_plots --in results/results_simops_rigorous.csv --outdir figs/simops --experiment simops
```

4. Parameter sensitivity experiment
```powershell
python -m src.runner --config configs/sensitivity.yaml
python -m analysis.make_tables --in results/results_sensitivity_rigorous.csv --out results/sensitivity_table.tex --experiment sensitivity
python -m analysis.make_plots --in results/results_sensitivity_rigorous.csv --outdir figs/sensitivity --experiment sensitivity
```

5. Service-portfolio experiment
```powershell
python -m src.runner --config configs/mechanism.yaml
python -m analysis.make_tables --in results/results_mechanism_rigorous.csv --out results/mechanism_table.tex --experiment mechanism
python -m analysis.make_plots --in results/results_mechanism_rigorous.csv --outdir figs/mechanism --experiment mechanism
```

6. CG ablation experiment
```powershell
python -m src.runner --config configs/ablation.yaml
python -m analysis.make_plots --in results/results_ablation_rigorous.csv --outdir figs/paper --experiment paper --traces_dir results/traces
```

7. Paper-body composite figures
```powershell
python -m analysis.build_paper_figures --results_dir results --outdir figs/paper
```

## One-command full pipeline

```powershell
python -m pipeline
```

Optional seed override:

```powershell
python -m pipeline --seeds 5
```

## Which figures are intended for the paper body

- `figs/paper/Fig_Paper_Main_Performance.*`
  Main benchmark figure: total cost, gap to CG+IR, runtime, and solve success.
- `figs/paper/Fig_Paper_Scenario_Mechanism.*`
  Scenario stress-test and service-portfolio comparison.
- `figs/paper/Fig_Paper_SIMOPS.*`
  Core SIMOPS mechanism figure: savings, stay-time reduction, masking distribution, masking by type.
- `figs/paper/Fig_Paper_Sensitivity.*`
  Three-panel sensitivity figure for battery cost, shore capacity, and deadline tightness.
- `figs/paper/Fig_Paper_Ablation.*`
  CG variant comparison for appendix or robustness section.

## Suggested figure hierarchy for a top-tier journal submission

1. Main text:
   `Fig_Paper_Main_Performance`, `Fig_Paper_SIMOPS`, `Fig_Paper_Sensitivity`
2. Main text or online appendix:
   `Fig_Paper_Scenario_Mechanism`
3. Appendix:
   `Fig_Paper_Ablation` and experiment-specific diagnostic plots under `figs/main`, `figs/scenario`, `figs/simops`, `figs/sensitivity`, `figs/mechanism`

## Notes on visual standard

- All paper figures are exported as both PDF and high-resolution PNG.
- The composite paper figures use restrained colors, serif typography, panel labels, shared legends, and confidence bands to better match TRE/TRC/EJOR expectations.
- If you want an even more journal-specific look, the next refinement step should be tailoring figure width and font sizes to the exact target journal template.

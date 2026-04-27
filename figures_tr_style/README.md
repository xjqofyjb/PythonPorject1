# TR-style figure package

This folder contains the Transportation Research-style figure outputs and the manifest used to control regeneration.

## Output directories

- `pdf/`: vector PDF files for manuscript submission
- `svg/`: editable vector backups
- `png/`: 600 dpi preview images

## Data provenance by figure

- `Fig1`
  - Source: `ltd.jpg`, `cjhd.jpg`
  - Resolved from the current Elsevier draft directory at `C:\Users\researcher\Desktop\Elsevier_s_CAS_LaTeX_Single_Column_Template\`
  - Rendered as a unified two-panel evidence figure with matched crop ratio and panel height

- `Fig2`
  - Source: `configs/simops.yaml`, `src/instances.py`, `src/solvers/cg_solver.py`
  - Re-solved from the illustrative SIMOPS instance used in the paper to reconstruct sequential and SIMOPS schedules
  - Right-side summary boxes report objective value, average stay, and masking rate only

- `Fig3`
  - Source: `results/results_main_rigorous.csv`
  - Built directly from seed-level raw result rows

- `Fig4`
  - Source: `results/results_scenario_rigorous.csv`, `results/results_mechanism_rigorous.csv`
  - Built directly from seed-level raw result rows

- `Fig5`
  - Source: `results/results_simops_rigorous.csv`
  - Built directly from seed-level raw result rows

- `Fig6`
  - Source: `results/results_sensitivity_rigorous.csv`
  - Built directly from seed-level raw result rows
  - Uses matched left-axis styling for total cost and a consistent secondary axis for AE share

- `Fig7`
  - Source: `results/carbon_price_dual_summary.csv`
  - Built from the experiment summary file generated from seed-level carbon-price runs
  - Carbon-price reference lines correspond to EU ETS ($100), the paper baseline ($200), and the IMO 2027 RU benchmark ($380)

- `Fig8`
  - Source: `results/robustness_fixed_deadline_summary.csv`, `results/robustness_fixed_deadline.csv`
  - Built from the fixed-deadline robustness experiment outputs

## How to rerun

From the project root:

```powershell
C:\Users\researcher\miniconda3\python.exe -m analysis.tr_figures.render_all
```

## Script layout

- Shared configuration: `analysis/tr_figures/config.py`
- Shared utilities: `analysis/tr_figures/utils.py`
- Figure scripts:
  - `figure1_realworld.py`
  - `figure2_mechanism.py`
  - `figure3_main.py`
  - `figure4_scenarios.py`
  - `figure5_simops.py`
  - `figure6_sensitivity.py`
  - `figure7_policy.py`
  - `figure8_robustness.py`

## Assumptions and limitations

- Figure 1 depends on the original photo assets from the current manuscript directory. If those files are moved or renamed, rerun will stop rather than silently substitute other imagery.
- Figures 2-8 preserve the numerical trends in the current result files. No data were invented.
- Confidence information is shown only when raw repeated-seed outputs support it. Otherwise the plots use direct line encodings from the available summaries.

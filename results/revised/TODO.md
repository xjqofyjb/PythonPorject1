# Revised Experiment TODO

- Full experiment suite has not been executed in this turn; the files in this directory are quick-mode outputs unless noted otherwise.
- Quick `main_benchmark` contains honest `skipped` rows for N=500 Rolling-Horizon and Fix-and-Optimize. Run `--full` for those time-heavy baselines.
- `Green_only_no_AE` no-AE enforcement is implemented for CG columns. Rolling-Horizon and Fix-and-Optimize rows for this mechanism are explicitly marked `skipped`; do not interpret them as infeasible cases.
- Carbon and capacity quick runs use reduced quick grids described in `README.md`; run `--full` for the complete requested grids.
- The carbon emission values currently use an `emissions_proxy_kg` computed from available aggregate cost/energy outputs. If exact kWh-by-mode emissions are required, add schedule-level energy accounting by mode before manuscript use.
- Bootstrap confidence intervals for the full SIMOPS dual-peak figure are not computed in quick mode; compute them from the full 10- or 20-seed raw output before claiming statistical precision.

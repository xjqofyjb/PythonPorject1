# Revision Audit

## Files inspected

- `src/instances.py`: synthetic instance generation and time-slot conversion.
- `src/model_utils.py`: added shared revised model conventions.
- `src/solvers/milp_solver.py`: compact time-indexed MILP.
- `src/solvers/cg_solver.py`: column pool generation, RMP solve, pricing, and integer recovery.
- `src/solvers/rolling_horizon_solver.py`: rolling-horizon benchmark.
- `src/solvers/fix_and_optimize_solver.py`: fix-and-optimize benchmark using rolling-horizon window MILPs.
- `src/solvers/fifo_solver.py`: FIFO heuristic benchmark.
- `src/solvers/greedy_solver.py`: greedy heuristic benchmark.
- `src/metrics.py`: decomposition/type/SIMOPS metrics.
- `src/runner.py` and `src/run_experiments.py`: legacy experiment launchers.
- `configs/main.yaml`, `configs/simops.yaml`, `configs/sensitivity.yaml`: baseline parameter/config files.
- `analysis/make_tables.py`, `analysis/make_plots.py`, `analysis/tr_figures/*.py`: existing plotting/table-generation scripts.
- Existing results/figures were found under `results/`, `figs/`, and `figures_tr_style/`.

## Where `C_BS` is used

- Active code stores `C_BS` as `Instance.battery_cost`, loaded from config key `battery_cost`.
- `src/solvers/milp_solver.py`: objective and solution decomposition use `instance.battery_cost * instance.energy_kwh[i]`.
- `src/solvers/cg_solver.py`: BS column costs, pricing pool costs, greedy seed columns, and decomposition use `instance.battery_cost * energy`.
- `src/solvers/rolling_horizon_solver.py`: `C_BS` is passed into rolling-window models and evaluated as `c_bs * ship.energy_kwh`.
- `src/solvers/fix_and_optimize_solver.py`: passes `C_BS` to the rolling-window solver.
- `src/solvers/fifo_solver.py` and `src/solvers/greedy_solver.py`: local BS costs use `instance.battery_cost * demand`.
- `configs/*.yaml`: `battery_cost` values define the $/kWh coefficient; sensitivity configs vary this key.
- `experiments/run_revision_experiments.py`: revised sensitivity runner varies `battery_cost` and records it as `C_BS`.

## Where time discretization is implemented

- `src/model_utils.py`: fixed convention helpers for `dt = 0.25 h`, `horizon = ceil(48 / dt) = 192`, ceiling slot conversion, and SIMOPS/sequential start minima.
- `src/instances.py`: converts physical arrival, cargo, deadline, SP duration, and BS duration to integer slots using ceiling.
- `src/solvers/cg_solver.py`, `src/solvers/milp_solver.py`, `src/solvers/fifo_solver.py`, `src/solvers/greedy_solver.py`: feasible SP/BS start sets use arrival or arrival+cargo and enforce `start + duration <= horizon`.
- `src/solvers/rolling_horizon_solver.py`: rolling windows use the same horizon convention and same start minima.

## Where CG dual values are extracted

- `src/solvers/cg_solver.py`, `_solve_master`: PuLP constraint `.pi` values are read for ship assignment, shore capacity, and battery capacity constraints.
- `src/solvers/cg_solver.py`, `_positive_congestion_prices`: raw minimization `<=` capacity duals are converted to nonnegative congestion prices with `max(0, -raw_dual)`.
- `src/solvers/cg_solver.py`, `_reduced_cost`: pricing uses `generalized_cost - pi_i + congestion_sum`.

## Where gap metrics are computed

- `src/solvers/cg_solver.py`, near final result assembly: computes `lp_obj_final_pool`, `irmp_obj`, `gap_pct`, `best_reduced_cost_last`, `num_negative_columns_last`, `num_columns_total`, `iterations`, `cg_status`, and `gap_type`.
- Legacy plotting/table scripts consume `gap_pct`; revised outputs are generated from `results/revised/*.csv`.

## Where experiments are launched

- Legacy: `src/runner.py`, `src/run_experiments.py`, top-level `run_experiment_*.py`, and `pipeline.py`.
- Revised: `experiments/run_revision_experiments.py`, with outputs under `results/revised/` and `figures/revised/`.

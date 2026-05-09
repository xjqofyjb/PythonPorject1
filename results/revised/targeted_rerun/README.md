# Targeted Rerun Results

## Exact Commands Run

- `C:\Users\researcher\miniconda3\python.exe -m pytest -q`
- `python experiments/run_revision_experiments.py --experiment targeted_simops_dominance --N 200 --scenario U --seeds 1 2 --strong-cg`
- `python experiments/run_revision_experiments.py --experiment targeted_N500_stress --N 500 --scenario U --seed 1 --strong-cg`
- `python experiments/run_revision_experiments.py --experiment targeted_table8_N200_replacement --N 200 --scenarios U P L --seeds 1 2 --strong-cg`

## Environment

- Python: `3.13.9`
- Gurobi: `13.0.0`
- Platform: `Windows-11-10.0.26100-SP0`

## Seeds

- SIMOPS dominance: seeds `1, 2`
- N=500 stress: seed `1`
- N=200 table replacement: scenarios `U, P, L`, seeds `1, 2`

## Large-Scale CG Protocol

- `N <= 100`: full generated-pool solve, reported as `Full-CG LP-IP gap` only when pricing converges.
- `N = 200, 500`: strengthened budgeted CG with `min_iters=20`, `max_iters=30`, `pricing_top_k=5`, Greedy/FIFO incumbent injection, and optional sequential-solution injection for SIMOPS dominance checks.
- Large-scale non-converged runs are labeled `Pool LP-IP gap`, not global optimality certificates.

## Gap Interpretation

- `Full-CG LP-IP gap`: complete pricing converged for the generated column universe.
- `Pool LP-IP gap`: IRMP gap against the LP over the generated budgeted pool only.

## Targeted Outcomes

- Tests passed: `True`
- N=200 SIMOPS dominance passed: `True`
- N=200 SIMOPS savings: seed 1: 9.172%, seed 2: 8.944%
- N=500 CG+IR objective: `1698230.332`
- N=500 best scalable baseline objective: `1812962.492`
- N=500 CG+IR beat scalable baselines: `True`
- Any large-scale pricing fully converged: `False`
- Any large-scale objective stabilized: `True`

Pricing did not fully converge for the large-scale budgeted evidence unless `pricing_converged=True` is shown in the raw CSV. These results are budgeted generated-pool solutions, not global optimality certificates.

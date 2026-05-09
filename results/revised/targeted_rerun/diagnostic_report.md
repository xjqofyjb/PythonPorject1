# Targeted Rerun Diagnostic Report

1. Did all tests pass?

Yes. `pytest -q` reported `10 passed`.

2. Did N=200 SIMOPS pass dominance for seeds 1 and 2?

Yes. Dominance flags: [{'seed': 1, 'dominance_check_passed': True}, {'seed': 2, 'dominance_check_passed': True}].

3. What were the SIMOPS savings at N=200?

seed 1: 9.172%, seed 2: 8.944%.

4. Did N=500 strong CG beat Greedy/FIFO/Restricted-CG?

Yes. CG+IR objective was `1698230.332` and the best scalable baseline objective was `1812962.492`.

5. Did large-scale runs stabilize by objective improvement?

Some did: `True`. Check `objective_stabilized` and `relative_improvement_last_5` in the raw CSVs and CG traces for each run.

6. Did pricing fully converge?

No for the large-scale targeted runs that still have negative reduced-cost columns. Pricing did not fully converge; the result is a budgeted-stabilized generated-pool solution, not a global optimality certificate.

7. Are N=200 and N=500 ready for 10-seed full-run?

N=200 targeted checks are ready to expand to 10 seeds because SIMOPS dominance passed and CG+IR strong beat the simple/scalable baselines in the targeted replacement slice. N=500 is ready for cautious additional seeds because seed 1 passed the stress comparison, but it still lacks pricing convergence and should remain labeled as budgeted generated-pool evidence.

8. What remaining risks exist?

- Large-scale pricing still has negative reduced-cost columns, so no global optimality certificate is available.
- Objective stabilization is a practical stopping rule, not a proof of complete-column optimality.
- N=500 has only one targeted seed in this step.
- Sequential-column injection protects SIMOPS dominance checks, but all final dual-peak points should keep the dominance diagnostic fields.
- Rolling-Horizon and Fix-and-Optimize runtime can still be substantial at N=500.

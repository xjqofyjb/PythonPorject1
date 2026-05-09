from src.instances import generate_instance
from src.solvers import cg_solver


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def test_simops_not_worse_than_sequential_small_full_pool():
    params = {
        "time_step_hours": 0.25,
        "horizon_hours": 48.0,
        "arrival_window_hours": 8.0,
        "shore_power_kw": 900.0,
        "battery_swap_hours": 0.75,
        "battery_cost": 0.45,
        "shore_cost": 0.15,
        "brown_cost": 0.90,
        "shore_cap": 2,
        "battery_slots": 2,
    }
    inst = generate_instance(10, 1, "U", "hybrid", params)
    base_cfg = {"method": "cg", "cg": {"use_full_pool_small": True, "full_pool_n": 100, "time_limit": 20}}
    seq = cg_solver.solve(inst, {**base_cfg, "operation_mode": "sequential"}, DummyLogger())
    sim = cg_solver.solve(
        inst,
        {**base_cfg, "operation_mode": "simops", "cg": {**base_cfg["cg"], "incumbent_solutions": [seq]}},
        DummyLogger(),
    )

    assert sim["obj"] <= seq["obj"] + 1e-6
    for result, mode in [(seq, "sequential"), (sim, "simops")]:
        assert result["cg_status"] == "full_pricing_converged"
        assert result["gap_type"] == "Full-CG LP-IP gap"
        assert result["full_pricing_converged"] is True

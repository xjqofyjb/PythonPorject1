from src.instances import generate_instance
from src.solvers import cg_solver


class DummyLogger:
    def info(self, *args, **kwargs):
        pass

    def exception(self, *args, **kwargs):
        pass


def test_cg_gap_metadata_for_full_pool_small_instance():
    params = {
        "time_step_hours": 0.25,
        "horizon_hours": 48.0,
        "arrival_window_hours": 8.0,
        "shore_power_kw": 900.0,
        "battery_swap_hours": 0.75,
        "shore_cap": 1,
        "battery_slots": 1,
    }
    inst = generate_instance(2, 1, "U", "hybrid", params)
    result = cg_solver.solve(
        inst,
        {
            "method": "cg",
            "cg": {"use_full_pool_small": True, "full_pool_n": 100, "time_limit": 10},
        },
        DummyLogger(),
    )
    assert result["cg_status"] == "full_pricing_converged"
    assert result["gap_type"] == "Full-CG LP-IP gap"
    assert "lp_obj_final_pool" in result
    assert "irmp_obj" in result
    assert result["gap_pct"] >= 0

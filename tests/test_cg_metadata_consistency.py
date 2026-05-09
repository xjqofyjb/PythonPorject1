from src.instances import generate_instance
from src.solvers import cg_solver


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def test_budgeted_metadata_consistency_with_negative_columns():
    inst = generate_instance(
        30,
        1,
        "U",
        "hybrid",
        {
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
        },
    )
    result = cg_solver.solve(
        inst,
        {
            "method": "cg",
            "operation_mode": "simops",
            "cg": {
                "use_full_pool_small": False,
                "max_iters": 2,
                "pricing_top_k": 2,
                "time_limit": 20,
                "pricing_eps": 1e-6,
                "use_incumbent_injection": True,
            },
        },
        DummyLogger(),
    )
    tol = 1e-6
    if result["num_negative_columns_scanned_last"] > 0:
        assert result["pricing_converged"] is False
    if result["pricing_converged"]:
        assert (
            result["num_negative_columns_scanned_last"] == 0
            or result["min_reduced_cost_scanned_last"] >= -tol
        )


def test_status_gap_type_consistency():
    inst = generate_instance(5, 1, "U", "hybrid", {"time_step_hours": 0.25, "horizon_hours": 48.0})
    full = cg_solver.solve(
        inst,
        {"method": "cg", "operation_mode": "simops", "cg": {"use_full_pool_small": True, "full_pool_n": 100}},
        DummyLogger(),
    )
    assert full["cg_status"] == "full_pricing_converged"
    assert full["gap_type"] == "Full-CG LP-IP gap"
    assert full["pricing_converged"] is True

    budgeted = cg_solver.solve(
        inst,
        {
            "method": "cg",
            "operation_mode": "simops",
            "cg": {
                "use_full_pool_small": False,
                "max_iters": 1,
                "min_iters": 1,
                "pricing_top_k": 1,
                "stabilization_rel_improvement": 1.0,
                "stabilization_gap_pct": 100.0,
            },
        },
        DummyLogger(),
    )
    if budgeted["cg_status"] == "budgeted_stabilized":
        assert budgeted["gap_type"] == "Pool LP-IP gap"
        assert budgeted["objective_stabilized"] is True

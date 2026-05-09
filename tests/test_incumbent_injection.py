from src.instances import generate_instance
from src.solvers import cg_solver, greedy_solver


class DummyLogger:
    def info(self, *args, **kwargs):
        pass


def test_injected_greedy_incumbent_not_worse_than_greedy():
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
    inst = generate_instance(20, 1, "U", "hybrid", params)
    greedy = greedy_solver.solve(inst, {"operation_mode": "simops", "return_schedule": True}, DummyLogger())
    greedy["method"] = "Greedy"

    columns, ship_cols, _ = cg_solver._build_columns(inst, "simops")
    injected_columns = cg_solver.convert_solution_to_columns(inst, greedy, "simops", columns)
    assert injected_columns

    cg = cg_solver.solve(
        inst,
        {
            "method": "cg",
            "operation_mode": "simops",
            "cg": {
                "use_full_pool_small": False,
                "max_iters": 4,
                "pricing_top_k": 3,
                "time_limit": 20,
                "use_incumbent_injection": True,
                "incumbent_solutions": [greedy],
            },
        },
        DummyLogger(),
    )
    assert cg["obj"] <= greedy["obj"] + 1e-6
    assert cg["injected_columns_count"] > 0

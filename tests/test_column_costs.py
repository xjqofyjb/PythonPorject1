import pytest

from src.instances import generate_instance
from src.solvers.cg_solver import _build_columns


def test_battery_columns_store_direct_cost_scaled_by_energy():
    inst = generate_instance(
        8,
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
    columns, _, _ = _build_columns(inst, "simops")
    battery_cols = [col for col in columns if col["mode"] == "battery"]
    assert battery_cols
    for col in battery_cols:
        i = col["ship"]
        assert col["direct_cost"] == pytest.approx(inst.battery_cost * inst.energy_kwh[i])
        assert col["cost"] == pytest.approx(col["direct_cost"] + col["delay_cost"])

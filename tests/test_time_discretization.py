import math

from src.instances import generate_instance
from src.model_utils import horizon_slots
from src.solvers.cg_solver import _build_columns


BASE_PARAMS = {
    "time_step_hours": 0.25,
    "horizon_hours": 48.0,
    "arrival_window_hours": 8.0,
    "shore_power_kw": 900.0,
    "battery_swap_hours": 0.75,
    "shore_cap": 2,
    "battery_slots": 2,
}


def test_horizon_and_durations_are_slot_based():
    inst = generate_instance(5, 1, "U", "hybrid", BASE_PARAMS)
    assert horizon_slots(BASE_PARAMS) == 192
    assert inst.horizon_steps == 192
    assert set(inst.bs_duration_steps.tolist()) == {3}
    for energy, duration in zip(inst.energy_kwh, inst.sp_duration_steps):
        assert duration == math.ceil(energy / (inst.shore_power_kw * inst.dt_hours))


def test_generated_columns_respect_horizon_and_start_min():
    inst = generate_instance(8, 2, "U", "hybrid", BASE_PARAMS)
    for op_mode in ["simops", "sequential"]:
        columns, _, horizon = _build_columns(inst, op_mode)
        assert horizon == 192
        for col in columns:
            if col["mode"] == "brown":
                continue
            i = col["ship"]
            required = inst.arrival_steps[i]
            if op_mode == "sequential":
                required += inst.cargo_steps[i]
            assert col["start"] >= required
            assert col["start"] + col["duration"] <= horizon

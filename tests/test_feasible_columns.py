from src.instances import generate_instance
from src.solvers.cg_solver import _build_columns


def test_no_sp_columns_when_incompatible_but_ae_exists():
    params = {
        "time_step_hours": 0.25,
        "horizon_hours": 48.0,
        "arrival_window_hours": 8.0,
        "shore_power_kw": 900.0,
        "battery_swap_hours": 0.75,
        "shore_cap": 2,
        "battery_slots": 2,
        "ship_type_defs": {
            "X": {
                "prob": 1.0,
                "energy_range": [4000.0, 4000.0],
                "delay": 1000.0,
                "shore": 0,
                "cargo_range": [6.0, 6.0],
            }
        },
    }
    inst = generate_instance(3, 1, "U", "hybrid", params)
    columns, ship_cols, _ = _build_columns(inst, "simops")

    for i, col_ids in enumerate(ship_cols):
        own_cols = [columns[j] for j in col_ids]
        assert not any(col["mode"] == "shore" for col in own_cols)
        ae_cols = [col for col in own_cols if col["mode"] == "brown"]
        assert len(ae_cols) == 1
        assert ae_cols[0]["shore_berth_use"] == []
        assert ae_cols[0]["battery_use"] == []

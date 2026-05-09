"""Instance generation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from src.model_utils import ceil_slots, horizon_slots


@dataclass
class Instance:
    """Container for a single experiment instance."""
    N: int
    seed: int
    scenario: str
    mechanism: str
    params: Dict[str, Any]
    demand: np.ndarray
    arrival_times: np.ndarray
    service_times: np.ndarray
    cargo_times: np.ndarray
    deadlines: np.ndarray
    shore_cap: float          # kept for backward compat; = number of berths
    shore_berths: int           # number of shore-power berths |K^SP|
    battery_slots: int
    battery_cost: float
    shore_cost: float
    brown_cost: float
    dt_hours: float
    arrival_steps: np.ndarray
    cargo_steps: np.ndarray
    deadline_steps: np.ndarray
    sp_duration_steps: np.ndarray
    bs_duration_steps: np.ndarray
    ship_types: np.ndarray
    shore_compatible: np.ndarray
    delay_costs: np.ndarray
    energy_kwh: np.ndarray
    shore_power_kw: float
    battery_swap_hours: float
    horizon_steps: int


def generate_instance(
    N: int,
    seed: int,
    scenario: str,
    mechanism: str,
    params: Dict[str, Any],
) -> Instance:
    """Generate a synthetic instance with configurable heterogeneity.

    The generator is deterministic given (N, seed, params).
    """
    rng = np.random.default_rng(seed)
    scenario_code = scenario.upper()

    dt_hours = float(params.get("time_step_hours", 0.25))
    horizon_steps = horizon_slots(params)
    arrival_window_hours = float(params.get("arrival_window_hours", 8.0))
    shore_power_kw = float(params.get("shore_power_kw", 900.0))
    battery_swap_hours = float(params.get("battery_swap_hours", 0.75))

    type_defs = {
        "A": {"prob": 0.20, "energy_range": (6000.0, 8000.0), "delay": 2000.0, "shore": 1, "cargo_range": (6.0, 24.0)},
        "B": {"prob": 0.50, "energy_range": (3000.0, 5000.0), "delay": 800.0, "shore": 1, "cargo_range": (6.0, 24.0)},
        "C": {"prob": 0.15, "energy_range": (3000.0, 4000.0), "delay": 500.0, "shore": 0, "cargo_range": (6.0, 24.0)},
        "D": {"prob": 0.15, "energy_range": (4000.0, 4000.0), "delay": 1500.0, "shore": 1, "cargo_range": (6.0, 12.0)},
    }
    type_defs = params.get("ship_type_defs", type_defs)

    types = np.array(list(type_defs.keys()))
    probs = np.array([type_defs[t]["prob"] for t in types], dtype=float)
    probs = probs / probs.sum()
    ship_types = rng.choice(types, size=N, p=probs)

    if scenario_code == "P":
        peak_mean_hours = float(params.get("peaked_arrival_mean_hours", arrival_window_hours / 2.0))
        peak_std_hours = float(params.get("peaked_arrival_std_hours", 1.0))
        uniform_mix = float(params.get("peaked_arrival_uniform_mix", 0.30))
        arrival_times = np.clip(
            rng.normal(peak_mean_hours, peak_std_hours, size=N),
            0.0,
            arrival_window_hours,
        )
        mix_mask = rng.uniform(0.0, 1.0, size=N) < uniform_mix
        arrival_times[mix_mask] = rng.uniform(0.0, arrival_window_hours, size=int(mix_mask.sum()))
    else:
        arrival_times = rng.uniform(0.0, arrival_window_hours, size=N)

    long_service_scale = float(params.get("long_service_scale", 1.0))
    long_service_shift_hours = float(params.get("long_service_shift_hours", 4.0))
    energy_kwh = np.zeros(N, dtype=float)
    cargo_times = np.zeros(N, dtype=float)
    delay_costs = np.zeros(N, dtype=float)
    shore_compatible = np.zeros(N, dtype=int)
    for idx, t_code in enumerate(ship_types):
        spec = type_defs[t_code]
        e_low, e_high = spec["energy_range"]
        energy_kwh[idx] = rng.uniform(e_low, e_high)
        delay_costs[idx] = float(spec["delay"])
        shore_compatible[idx] = int(spec["shore"])

        c_low, c_high = spec["cargo_range"]
        if scenario_code == "L":
            c_low = c_low * long_service_scale + long_service_shift_hours
            c_high = c_high * long_service_scale + long_service_shift_hours
        cargo_times[idx] = rng.uniform(c_low, c_high)

    slack_base = rng.uniform(2.0, 6.0, size=N)
    tightness = float(params.get("deadline_tightness", 1.0))
    slack = slack_base * tightness
    deadlines = arrival_times + cargo_times + slack

    shore_cap = float(params.get("shore_cap", 2.0))
    battery_slots = int(params.get("battery_slots", 2))
    battery_cost = float(params.get("battery_cost", 0.45))
    shore_cost = float(params.get("shore_cost", 0.15))
    brown_cost = float(params.get("brown_cost", 0.9))

    # Mechanism switches.
    if mechanism == "battery_only":
        shore_cap = 0.0
    elif mechanism == "shore_only":
        battery_slots = 0
        battery_cost = battery_cost * 1.4
    elif mechanism == "no_brown":
        brown_cost = brown_cost * 10.0

    arrival_steps = np.array([ceil_slots(x, dt_hours) for x in arrival_times], dtype=int)
    cargo_steps = np.array([ceil_slots(x, dt_hours) for x in cargo_times], dtype=int)
    deadline_steps = np.array([ceil_slots(x, dt_hours) for x in deadlines], dtype=int)
    sp_duration_steps = np.array([ceil_slots(x / shore_power_kw, dt_hours) for x in energy_kwh], dtype=int)
    bs_duration_steps = np.full(N, ceil_slots(battery_swap_hours, dt_hours), dtype=int)

    return Instance(
        N=N,
        seed=seed,
        scenario=scenario,
        mechanism=mechanism,
        params=params,
        demand=energy_kwh,
        arrival_times=arrival_times,
        service_times=sp_duration_steps.astype(float) * dt_hours,
        cargo_times=cargo_times,
        deadlines=deadlines,
        shore_cap=shore_cap,
        shore_berths=int(shore_cap),
        battery_slots=battery_slots,
        battery_cost=battery_cost,
        shore_cost=shore_cost,
        brown_cost=brown_cost,
        dt_hours=dt_hours,
        arrival_steps=arrival_steps,
        cargo_steps=cargo_steps,
        deadline_steps=deadline_steps,
        sp_duration_steps=sp_duration_steps,
        bs_duration_steps=bs_duration_steps,
        ship_types=ship_types,
        shore_compatible=shore_compatible,
        delay_costs=delay_costs,
        energy_kwh=energy_kwh,
        shore_power_kw=shore_power_kw,
        battery_swap_hours=battery_swap_hours,
        horizon_steps=horizon_steps,
    )

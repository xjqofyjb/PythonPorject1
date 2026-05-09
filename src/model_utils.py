"""Shared model conventions for the revised experiments."""
from __future__ import annotations

import math
from typing import Any, Mapping


DEFAULT_DT_HOURS = 0.25
DEFAULT_HORIZON_HOURS = 48.0


def horizon_slots(params: Mapping[str, Any] | None = None) -> int:
    """Return the fixed optimization horizon in integer slots."""
    params = params or {}
    dt = float(params.get("time_step_hours", params.get("delta_t", DEFAULT_DT_HOURS)))
    horizon_h = float(params.get("horizon_hours", params.get("T_horizon", params.get("T_horizon_hours", DEFAULT_HORIZON_HOURS))))
    return int(math.ceil(horizon_h / dt))


def ceil_slots(hours: float, dt_hours: float) -> int:
    """Convert a physical duration in hours to integer time slots."""
    return int(math.ceil(float(hours) / float(dt_hours)))


def energy_direct_cost(unit_cost_per_kwh: float, energy_kwh: float) -> float:
    """Direct service cost when the cost coefficient is in $/kWh."""
    return float(unit_cost_per_kwh) * float(energy_kwh)


def operation_start_min(arrival_slot: int, cargo_slots: int, operation_mode: str) -> int:
    """Earliest feasible SP/BS start slot under SIMOPS or sequential operation."""
    if str(operation_mode).lower() == "sequential":
        return int(arrival_slot) + int(cargo_slots)
    return int(arrival_slot)

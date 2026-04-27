"""Resource-aware greedy solver.

Updated to use per-berth shore power tracking consistent with
the corrected model (per-berth exclusive constraint).
"""
from __future__ import annotations

import time
from typing import Dict, Any

import numpy as np

from src.instances import Instance
from src.metrics import compute_simops_metrics, compute_type_breakdown


def solve(instance: Instance, cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Solve instance using a myopic, resource-aware greedy heuristic."""
    t0 = time.perf_counter()
    K_SP = int(instance.shore_berths) if hasattr(instance, 'shore_berths') else int(instance.shore_cap)
    battery_cap = int(instance.battery_slots)
    brown_available = bool(instance.params.get("brown_available", True))

    dt = float(instance.dt_hours)
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps
    sp_dur = instance.sp_duration_steps
    bs_dur = instance.bs_duration_steps

    horizon = int(max(np.max(deadlines + sp_dur + 2), np.max(arrival + cargo + sp_dur + 2)))
    horizon = max(horizon, int(np.max(deadlines)) + 5)

    # Per-berth shore power usage tracking
    shore_usage = np.zeros((max(K_SP, 1), horizon), dtype=int)
    battery_usage = np.zeros(horizon, dtype=int)

    order = np.argsort(arrival)
    mode_counts = {"shore": 0, "battery": 0, "brown": 0}
    infeasible_jobs = 0
    total_cost = 0.0
    energy_cost_total = 0.0
    delay_cost_total = 0.0
    service_starts = np.full(instance.N, np.nan)
    service_durs = np.zeros(instance.N, dtype=float)
    start_steps = np.zeros(instance.N, dtype=int)
    duration_steps = np.zeros(instance.N, dtype=int)
    modes = ["brown"] * instance.N

    def feasible_shore(start: int, duration: int) -> int:
        """Check if any shore berth is available. Returns berth index or -1."""
        end = start + duration
        if end > horizon or K_SP <= 0:
            return -1
        for k in range(K_SP):
            if not np.any(shore_usage[k, start:end]):
                return k
        return -1

    def feasible_battery(start: int, duration: int) -> bool:
        end = start + duration
        if end > horizon or battery_cap <= 0:
            return False
        if np.any(battery_usage[start:end] >= battery_cap):
            return False
        return True

    def reserve_shore(k: int, start: int, duration: int) -> None:
        shore_usage[k, start:start + duration] = 1

    def reserve_battery(start: int, duration: int) -> None:
        battery_usage[start:start + duration] += 1

    for i in order:
        op_mode = cfg.get("operation_mode", "simops")
        if op_mode == "sequential":
            start_min = int(arrival[i] + cargo[i])
        else:
            start_min = int(arrival[i])

        best = None
        demand = instance.energy_kwh[i]
        delay_cost = instance.delay_costs[i]

        if instance.shore_compatible[i]:
            duration = int(sp_dur[i])
            for t in range(start_min, horizon - duration + 1):
                k = feasible_shore(t, duration)
                if k < 0:
                    continue
                completion = max(arrival[i] + cargo[i], t + duration)
                tardy = max(0, completion - deadlines[i])
                cost = instance.shore_cost * demand + delay_cost * dt * tardy
                if best is None or cost < best[0]:
                    best = (cost, "shore", t, duration, k)

        duration = int(bs_dur[i])
        for t in range(start_min, horizon - duration + 1):
            if not feasible_battery(t, duration):
                continue
            completion = max(arrival[i] + cargo[i], t + duration)
            tardy = max(0, completion - deadlines[i])
            cost = instance.battery_cost * demand + delay_cost * dt * tardy
            if best is None or cost < best[0]:
                best = (cost, "battery", t, duration, -1)

        if brown_available:
            completion = int(arrival[i] + cargo[i])
            tardy = max(0, completion - deadlines[i])
            cost = instance.brown_cost * demand + delay_cost * dt * tardy
            if best is None or cost < best[0]:
                best = (cost, "brown", int(arrival[i]), 0, -1)

        if best is None:
            infeasible_jobs += 1
            continue

        cost, mode, start, duration, berth_k = best
        if mode == "shore":
            reserve_shore(berth_k, start, duration)
        elif mode == "battery":
            reserve_battery(start, duration)
        service_starts[i] = float(start) * dt
        service_durs[i] = float(duration) * dt
        start_steps[i] = int(start)
        duration_steps[i] = int(duration)
        modes[i] = mode
        total_cost += cost
        if mode == "shore":
            energy_cost_total += instance.shore_cost * demand
        elif mode == "battery":
            energy_cost_total += instance.battery_cost * demand
        else:
            energy_cost_total += instance.brown_cost * demand
        completion = max(arrival[i] + cargo[i], start + duration)
        tardy = max(0, completion - deadlines[i])
        delay_cost_total += delay_cost * dt * tardy
        mode_counts[mode] += 1

    status = "ok" if infeasible_jobs == 0 else "infeasible"
    t1 = time.perf_counter()
    result = {
        "obj": float(total_cost),
        "runtime_total": float(t1 - t0),
        "status": status,
        "mechanism_counts": {
            "shore": mode_counts["shore"],
            "battery": mode_counts["battery"],
            "brown": mode_counts["brown"],
        },
        "infeasible_jobs": infeasible_jobs,
        "cost_energy": float(energy_cost_total),
        "cost_delay": float(delay_cost_total),
    }
    result.update(compute_type_breakdown(instance, modes, start_steps, duration_steps))
    if "operation_mode" in cfg:
        simops = compute_simops_metrics(
            instance,
            cfg.get("operation_mode", "simops"),
            service_start_times=service_starts,
            service_durations=service_durs,
        )
        result.update(simops)
    if cfg.get("return_schedule"):
        result["schedule"] = {
            "service_start_times": service_starts.tolist(),
            "service_durations": service_durs.tolist(),
            "modes": modes,
        }
    return result
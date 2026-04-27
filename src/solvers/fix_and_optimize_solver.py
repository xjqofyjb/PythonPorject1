"""Fix-and-optimize baseline for port energy scheduling."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Set

import numpy as np

from src.instances import Instance
from src.metrics import compute_simops_metrics, compute_type_breakdown
from src.solvers.rolling_horizon_solver import (
    _build_ships_from_instance,
    _compute_horizon_steps,
    _greedy_window_solution,
    solve_window_milp,
)


def _build_fixed_usage(
    decisions: Dict[int, Dict[str, Any]],
    free_ship_ids: Set[int],
    horizon_steps: int,
    k_sp: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed occupancies contributed by ships outside the current neighborhood."""
    fixed_sp = np.zeros((max(k_sp, 1), horizon_steps), dtype=int)
    fixed_bs = np.zeros(horizon_steps, dtype=int)

    for ship_id, dec in decisions.items():
        if ship_id in free_ship_ids:
            continue
        mode = dec.get("mode")
        start = int(dec.get("start_step", 0))
        duration = int(dec.get("duration_step", 0))
        if duration <= 0:
            continue
        end = min(horizon_steps, start + duration)
        if mode == "SP":
            berth_k = int(dec.get("berth_k", -1))
            if 0 <= berth_k < max(k_sp, 1):
                fixed_sp[berth_k, start:end] = 1
        elif mode == "BS":
            fixed_bs[start:end] += 1
    return fixed_sp, fixed_bs


def _aggregate_results(
    instance: Instance,
    ships,
    committed_results: Dict[int, Dict[str, Any]],
    operation_mode: str,
    runtime_total: float,
) -> Dict[str, Any]:
    dt = float(instance.dt_hours)
    mode_map = {"SP": "shore", "BS": "battery", "AE": "brown"}

    total_obj = sum(dec["obj"] for dec in committed_results.values())
    total_energy = sum(dec["cost_energy"] for dec in committed_results.values())
    total_delay = sum(dec["cost_delay"] for dec in committed_results.values())

    mode_assignments = {ship_id: dec["mode"] for ship_id, dec in committed_results.items()}
    start_times = {ship_id: dec["start_step"] * dt for ship_id, dec in committed_results.items()}
    departure_times = {ship_id: dec["departure_step"] * dt for ship_id, dec in committed_results.items()}
    mode_counts = {"shore": 0, "battery": 0, "brown": 0}
    for dec in committed_results.values():
        mode_counts[mode_map[dec["mode"]]] += 1

    start_steps = np.zeros(instance.N, dtype=int)
    duration_steps = np.zeros(instance.N, dtype=int)
    service_starts = np.full(instance.N, np.nan)
    service_durations = np.zeros(instance.N, dtype=float)
    modes: List[str] = ["brown"] * instance.N

    for ship in ships:
        dec = committed_results.get(ship.ship_id)
        if dec is None:
            continue
        mode = mode_map[dec["mode"]]
        start_steps[ship.ship_id] = int(dec["start_step"])
        duration_steps[ship.ship_id] = int(dec["duration_step"])
        service_starts[ship.ship_id] = float(dec["start_step"]) * dt
        service_durations[ship.ship_id] = float(dec["duration_step"]) * dt
        modes[ship.ship_id] = mode

    result = {
        "obj": float(total_obj),
        "runtime_total": float(runtime_total),
        "mode_assignments": mode_assignments,
        "start_times": start_times,
        "departure_times": departure_times,
        "success": len(committed_results) == len(ships),
        "method": "fix_and_optimize",
        "mechanism_counts": mode_counts,
        "cost_energy": float(total_energy),
        "cost_delay": float(total_delay),
        "committed_results": committed_results,
        "status": "ok" if len(committed_results) == len(ships) else "infeasible",
        "infeasible_jobs": int(instance.N - len(committed_results)),
    }
    result.update(compute_type_breakdown(instance, modes, start_steps, duration_steps))
    result.update(
        compute_simops_metrics(
            instance,
            operation_mode,
            service_start_times=service_starts,
            service_durations=service_durations,
        )
    )
    return result


def fix_and_optimize(
    instance: Instance,
    block_size: int = 10,
    step_size: int = 5,
    max_passes: int = 2,
    time_limit: int = 30,
    operation_mode: str = "simops",
) -> Dict[str, Any]:
    """Large-neighborhood improvement using exact MILP on moving ship blocks."""
    t0 = time.perf_counter()
    ships = _build_ships_from_instance(instance)
    ships_sorted = sorted(ships, key=lambda s: (s.arrival_step, s.ship_id))

    horizon_steps = _compute_horizon_steps(ships_sorted, {"delta_t": instance.dt_hours, "T_horizon": instance.params.get("T_horizon", instance.params.get("T_horizon_hours", 48.0))})
    params = {
        "K_SP": int(instance.shore_berths) if hasattr(instance, "shore_berths") else int(instance.shore_cap),
        "K_BS": int(instance.battery_slots),
        "C_SP": float(instance.shore_cost),
        "C_BS": float(instance.battery_cost),
        "C_AE": float(instance.brown_cost),
        "T_BS": float(instance.battery_swap_hours),
        "P_SP": float(instance.shore_power_kw),
        "delta_t": float(instance.dt_hours),
        "T_horizon": float(instance.params.get("T_horizon", instance.params.get("T_horizon_hours", 48.0))),
        "T_horizon_steps": int(horizon_steps),
    }

    zero_sp = np.zeros((max(params["K_SP"], 1), horizon_steps), dtype=int)
    zero_bs = np.zeros(horizon_steps, dtype=int)
    committed_results = _greedy_window_solution(ships_sorted, zero_sp, zero_bs, params, operation_mode)

    block_size = max(1, min(int(block_size), len(ships_sorted)))
    step_size = max(1, int(step_size))
    max_passes = max(1, int(max_passes))
    improve_tol = 1e-6

    for _ in range(max_passes):
        num_accepted = 0
        for start_idx in range(0, len(ships_sorted), step_size):
            block = ships_sorted[start_idx:start_idx + block_size]
            if not block:
                break

            free_ship_ids = {ship.ship_id for ship in block}
            fixed_sp, fixed_bs = _build_fixed_usage(
                committed_results,
                free_ship_ids,
                horizon_steps,
                int(params["K_SP"]),
            )
            window_result = solve_window_milp(
                block,
                fixed_sp,
                fixed_bs,
                params,
                time_limit=time_limit,
                operation_mode=operation_mode,
            )
            decisions = window_result.get("decisions") or {}
            if len(decisions) != len(block):
                continue

            old_obj = sum(float(committed_results[ship.ship_id]["obj"]) for ship in block if ship.ship_id in committed_results)
            new_obj = sum(float(decisions[ship.ship_id]["obj"]) for ship in block)
            if new_obj <= old_obj - improve_tol:
                for ship in block:
                    committed_results[ship.ship_id] = decisions[ship.ship_id]
                num_accepted += 1

        if num_accepted == 0:
            break

    return _aggregate_results(instance, ships_sorted, committed_results, operation_mode, time.perf_counter() - t0)


def solve(instance: Instance, cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Adapter to the existing experiment framework."""
    fao_cfg = dict(cfg.get("fix_and_optimize", {}))
    block_size = int(fao_cfg.get("block_size", 10))
    step_size = int(fao_cfg.get("step_size", max(1, block_size // 2)))
    max_passes = int(fao_cfg.get("max_passes", 2))
    time_limit = int(fao_cfg.get("time_limit", 30))
    operation_mode = str(cfg.get("operation_mode", "simops"))

    result = fix_and_optimize(
        instance,
        block_size=block_size,
        step_size=step_size,
        max_passes=max_passes,
        time_limit=time_limit,
        operation_mode=operation_mode,
    )
    logger.info(
        "Fix-and-optimize solved N=%s with block=%s step=%s passes=%s obj=%.2f runtime=%.2fs",
        instance.N,
        block_size,
        step_size,
        max_passes,
        float(result["obj"]),
        float(result["runtime_total"]),
    )
    return result

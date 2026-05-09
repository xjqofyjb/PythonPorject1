"""Metrics computation."""
from __future__ import annotations

from typing import Dict, Any, List

import numpy as np

from src.instances import Instance


def compute_cost_components(instance: Instance, obj: float) -> Dict[str, Any]:
    """Compute simple cost decomposition proxies."""
    demand_sum = float(instance.energy_kwh.sum()) if hasattr(instance, "energy_kwh") else float(instance.demand.sum())
    energy_cost = demand_sum * float(instance.shore_cost)
    delay_cost = max(0.0, demand_sum - instance.shore_cap * 40.0) * 0.01

    total = energy_cost + delay_cost
    scale = obj / total if total > 0 else 1.0
    return {
        "cost_energy": energy_cost * scale,
        "cost_delay": delay_cost * scale,
    }


def compute_mechanism_metrics(instance: Instance) -> Dict[str, Any]:
    """Mechanism-specific indicators."""
    demand_sum = float(instance.energy_kwh.sum()) if hasattr(instance, "energy_kwh") else float(instance.demand.sum())
    shore_util = 0.0
    if instance.shore_cap > 0:
        horizon_steps = int(np.max(instance.deadline_steps)) if hasattr(instance, "deadline_steps") else 1
        total_capacity = instance.shore_cap * instance.shore_power_kw * instance.dt_hours * horizon_steps
        if total_capacity > 0:
            shore_util = min(1.0, demand_sum / total_capacity)
    return {
        "shore_utilization": shore_util,
    }


def compute_mode_ratios(mechanism_counts: Dict[str, int], total_jobs: int) -> Dict[str, Any]:
    """Compute mode usage ratios from counts."""
    if total_jobs <= 0:
        return {"shore_ratio": np.nan, "battery_ratio": np.nan, "brown_ratio": np.nan}
    shore = mechanism_counts.get("shore", 0)
    battery = mechanism_counts.get("battery", 0)
    brown = mechanism_counts.get("brown", 0)
    return {
        "shore_ratio": shore / total_jobs,
        "battery_ratio": battery / total_jobs,
        "brown_ratio": brown / total_jobs,
    }


def compute_solution_operational_metrics(
    instance: Instance,
    schedule: Dict[str, Any] | None,
    operation_mode: str = "simops",
    grid_emission_factor_kg_per_kwh: float = 0.445,
    ae_emission_factor_kg_per_kwh: float = 0.70,
) -> Dict[str, Any]:
    """Compute delay and emissions from a per-vessel returned schedule."""
    if not schedule:
        return {
            "avg_delay_h": np.nan,
            "delay_cost": np.nan,
            "emissions_total_kg": np.nan,
            "emissions_total_tCO2": np.nan,
        }

    modes = list(schedule.get("modes") or [])
    starts = np.asarray(schedule.get("service_start_times") or [], dtype=float)
    durs = np.asarray(schedule.get("service_durations") or [], dtype=float)
    if len(modes) != instance.N or len(starts) != instance.N or len(durs) != instance.N:
        return {
            "avg_delay_h": np.nan,
            "delay_cost": np.nan,
            "emissions_total_kg": np.nan,
            "emissions_total_tCO2": np.nan,
        }

    delays = np.zeros(instance.N, dtype=float)
    emissions_kg = np.zeros(instance.N, dtype=float)
    for i, mode in enumerate(modes):
        cargo_completion = float(instance.arrival_times[i] + instance.cargo_times[i])
        service_completion = float(starts[i] + durs[i])
        if mode == "brown":
            departure = cargo_completion
            emissions_factor = ae_emission_factor_kg_per_kwh
        else:
            departure = service_completion if operation_mode == "sequential" else max(cargo_completion, service_completion)
            emissions_factor = grid_emission_factor_kg_per_kwh
        delays[i] = max(0.0, departure - float(instance.deadlines[i]))
        emissions_kg[i] = float(instance.energy_kwh[i]) * emissions_factor

    delay_cost_total = float(np.sum(delays * instance.delay_costs))
    emissions_total_kg = float(np.sum(emissions_kg))
    return {
        "avg_delay_h": float(np.mean(delays)) if instance.N else 0.0,
        "delay_cost": delay_cost_total,
        "emissions_total_kg": emissions_total_kg,
        "emissions_total_tCO2": emissions_total_kg / 1000.0,
    }


def compute_simops_metrics(
    instance: Instance,
    operation_mode: str,
    service_start_times: np.ndarray | None = None,
    service_durations: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute SIMOPS masking metrics from service/cargo overlap."""
    if operation_mode != "simops":
        if instance.N <= 0:
            return {
                "num_fully_masked": 0,
                "num_partially_masked": 0,
                "avg_masking_rate": 0.0,
                "time_saved_total": 0.0,
                "avg_stay_time": 0.0,
            }
        if service_start_times is None:
            service_start_times = instance.arrival_times + instance.cargo_times
        if service_durations is None:
            service_durations = instance.service_times
        if np.isnan(service_start_times).any():
            fill = instance.arrival_times + instance.cargo_times
            service_start_times = np.where(np.isnan(service_start_times), fill, service_start_times)
        stay_time = (service_start_times - instance.arrival_times) + service_durations
        base = {
            "num_fully_masked": 0,
            "num_partially_masked": 0,
            "avg_masking_rate": 0.0,
            "time_saved_total": 0.0,
            "avg_stay_time": float(np.mean(stay_time)),
        }
        if hasattr(instance, "ship_types"):
            types = np.asarray(instance.ship_types)
            for t_code in sorted(set(types)):
                base[f"num_fully_masked_type_{t_code}"] = 0
                base[f"num_partially_masked_type_{t_code}"] = 0
                base[f"avg_masking_rate_type_{t_code}"] = 0.0
        return base

    N = instance.N
    if N <= 0:
        return {
            "num_fully_masked": 0,
            "num_partially_masked": 0,
            "avg_masking_rate": 0.0,
            "time_saved_total": 0.0,
            "avg_stay_time": 0.0,
        }

    if service_start_times is None:
        service_start_times = instance.arrival_times
    if service_durations is None:
        service_durations = instance.service_times
    if np.isnan(service_start_times).any():
        fill = instance.arrival_times
        service_start_times = np.where(np.isnan(service_start_times), fill, service_start_times)

    cargo_starts = instance.arrival_times
    cargo_ends = cargo_starts + instance.cargo_times
    service_ends = service_start_times + service_durations

    overlap = np.maximum(0.0, np.minimum(cargo_ends, service_ends) - np.maximum(cargo_starts, service_start_times))
    masking_rate = np.divide(
        overlap,
        service_durations,
        out=np.zeros_like(overlap, dtype=float),
        where=service_durations > 0,
    )

    num_fully = int(np.sum(masking_rate >= 0.99))
    num_partial = int(np.sum((masking_rate > 0.0) & (masking_rate < 0.99)))
    avg_mask = float(np.mean(masking_rate))
    time_saved_total = float(np.sum(overlap))

    stay_time = np.maximum(instance.cargo_times, (service_start_times - instance.arrival_times) + service_durations)
    avg_stay = float(np.mean(stay_time))
    result = {
        "num_fully_masked": num_fully,
        "num_partially_masked": num_partial,
        "avg_masking_rate": avg_mask,
        "time_saved_total": time_saved_total,
        "avg_stay_time": avg_stay,
    }

    # Per-type masking counts for analysis.
    if hasattr(instance, "ship_types"):
        types = np.asarray(instance.ship_types)
        for t_code in sorted(set(types)):
            mask = types == t_code
            if not np.any(mask):
                result[f"num_fully_masked_type_{t_code}"] = 0
                result[f"num_partially_masked_type_{t_code}"] = 0
                result[f"avg_masking_rate_type_{t_code}"] = 0.0
                continue
            t_rates = masking_rate[mask]
            result[f"num_fully_masked_type_{t_code}"] = int(np.sum(t_rates >= 0.99))
            result[f"num_partially_masked_type_{t_code}"] = int(np.sum((t_rates > 0.0) & (t_rates < 0.99)))
            result[f"avg_masking_rate_type_{t_code}"] = float(np.mean(t_rates))

    return result


def compute_type_breakdown(
    instance: Instance,
    modes: List[str],
    start_steps: np.ndarray,
    duration_steps: np.ndarray,
) -> Dict[str, Any]:
    """Compute per-ship-type cost and mode ratios."""
    types = np.asarray(instance.ship_types)
    unique_types = sorted(set(types))
    out: Dict[str, Any] = {}
    dt = float(instance.dt_hours)
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps

    modes_arr = np.asarray(modes)
    for t_code in unique_types:
        mask = types == t_code
        count = int(np.sum(mask))
        out[f"type_{t_code}_count"] = count
        if count == 0:
            out[f"type_{t_code}_cost_energy"] = 0.0
            out[f"type_{t_code}_cost_delay"] = 0.0
            out[f"type_{t_code}_cost_total"] = 0.0
            out[f"type_{t_code}_shore_ratio"] = 0.0
            out[f"type_{t_code}_battery_ratio"] = 0.0
            out[f"type_{t_code}_brown_ratio"] = 0.0
            out[f"type_{t_code}_avg_tardy_hours"] = 0.0
            continue

        energy = instance.energy_kwh[mask]
        delay_costs = instance.delay_costs[mask]
        modes_t = modes_arr[mask]
        start_t = start_steps[mask]
        dur_t = duration_steps[mask]

        completion = np.maximum(arrival[mask] + cargo[mask], start_t + dur_t)
        tardy = np.maximum(0.0, completion - deadlines[mask])

        rate = np.where(
            modes_t == "shore",
            instance.shore_cost,
            np.where(modes_t == "battery", instance.battery_cost, instance.brown_cost),
        )
        cost_energy = float(np.sum(rate * energy))
        cost_delay = float(np.sum(delay_costs * dt * tardy))

        out[f"type_{t_code}_cost_energy"] = cost_energy
        out[f"type_{t_code}_cost_delay"] = cost_delay
        out[f"type_{t_code}_cost_total"] = cost_energy + cost_delay

        out[f"type_{t_code}_shore_ratio"] = float(np.mean(modes_t == "shore"))
        out[f"type_{t_code}_battery_ratio"] = float(np.mean(modes_t == "battery"))
        out[f"type_{t_code}_brown_ratio"] = float(np.mean(modes_t == "brown"))
        out[f"type_{t_code}_avg_tardy_hours"] = float(np.mean(tardy) * dt)

    return out

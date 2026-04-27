from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.instances import Instance
from src.metrics import compute_mode_ratios
from src.solvers.cg_solver import (
    _build_columns,
    _build_greedy_seed_column_ids,
    _select_pricing_ship_subset,
    _solve_master,
)


def _plan_key(col: dict[str, Any]) -> tuple[int, str, int, int]:
    return (int(col["ship"]), str(col["mode"]), int(col["start"]), int(col["berth"]))


def _mode_label(mode: str) -> str:
    if mode == "shore":
        return "SP"
    if mode == "battery":
        return "BS"
    return "AE"


def _compute_reduced_cost(col: dict[str, Any], duals: dict[str, Any]) -> float:
    rc = float(col["cost"]) - float(duals["ship"].get(int(col["ship"]), 0.0))
    rc -= sum(float(duals["shore"].get((k, t), 0.0)) for (k, t) in col.get("shore_berth_use", []))
    rc -= sum(float(duals["battery"].get(t, 0.0)) for t in col.get("battery_use", []))
    return float(rc)


def _make_pool(columns_all: list[dict[str, Any]], ship_cols_all: list[list[int]], active_ids: set[int]) -> tuple[list[dict[str, Any]], list[list[int]], list[int]]:
    active_list = sorted(active_ids)
    columns = [columns_all[j] for j in active_list]
    old_to_new = {old: new for new, old in enumerate(active_list)}
    ship_cols = [[old_to_new[j] for j in ship_cols_all[i] if j in active_ids] for i in range(len(ship_cols_all))]
    return columns, ship_cols, active_list


def _brown_fallback(ship_cols: list[list[int]], columns: list[dict[str, Any]]) -> dict[int, float]:
    solution = {j: 0.0 for j in range(len(columns))}
    for col_ids in ship_cols:
        for j in col_ids:
            if columns[j]["mode"] == "brown":
                solution[j] = 1.0
                break
    return solution


def _extract_solution(
    instance: Instance,
    columns: list[dict[str, Any]],
    ship_cols: list[list[int]],
    solution: dict[int, float],
    lp_obj: float,
    lp_status: str,
    ip_status: str,
) -> dict[str, Any]:
    dt = float(instance.dt_hours)
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps

    mode_counts = {"shore": 0, "battery": 0, "brown": 0}
    cost_energy = 0.0
    cost_delay = 0.0
    chosen_plan_keys: list[tuple[int, str, int, int]] = []
    chosen_modes: list[str] = []
    chosen_starts: list[int] = []
    chosen_berths: list[int] = []

    for i, col_ids in enumerate(ship_cols):
        chosen = None
        for j in col_ids:
            if solution.get(j, 0.0) > 0.5:
                chosen = columns[j]
                break
        if chosen is None:
            chosen = min((columns[j] for j in col_ids), key=lambda c: c["cost"])

        mode = str(chosen["mode"])
        chosen_plan_keys.append(_plan_key(chosen))
        chosen_modes.append(mode)
        chosen_starts.append(int(chosen["start"]))
        chosen_berths.append(int(chosen["berth"]))
        mode_counts[mode] += 1

        energy = float(instance.energy_kwh[i])
        if mode == "shore":
            cost_energy += float(instance.shore_cost) * energy
        elif mode == "battery":
            cost_energy += float(instance.battery_cost) * energy
        else:
            cost_energy += float(instance.brown_cost) * energy

        completion = max(int(arrival[i] + cargo[i]), int(chosen["start"] + chosen["duration"]))
        tardy = max(0.0, completion - int(deadlines[i]))
        cost_delay += float(instance.delay_costs[i]) * dt * tardy

    objective = float(cost_energy + cost_delay)
    internal_gap_pct = np.nan
    if np.isfinite(lp_obj) and abs(lp_obj) > 1e-9:
        internal_gap_pct = max(0.0, (objective - float(lp_obj)) / float(lp_obj) * 100.0)

    shares = compute_mode_ratios(mode_counts, instance.N)
    return {
        "objective": objective,
        "lp_lower_bound": float(lp_obj) if np.isfinite(lp_obj) else np.nan,
        "internal_gap_pct": float(internal_gap_pct) if np.isfinite(internal_gap_pct) else np.nan,
        "lp_status": str(lp_status),
        "ip_status": str(ip_status),
        "success": ip_status in ("Optimal", "Not Solved", "Feasible"),
        "plan_keys": chosen_plan_keys,
        "mode_assignments": chosen_modes,
        "start_steps": chosen_starts,
        "berths": chosen_berths,
        "mode_distribution": {
            "SP": float(shares.get("shore_ratio", np.nan)),
            "BS": float(shares.get("battery_ratio", np.nan)),
            "AE": float(shares.get("brown_ratio", np.nan)),
        },
        "mechanism_counts": {key: int(val) for key, val in mode_counts.items()},
        "cost_energy": float(cost_energy),
        "cost_delay": float(cost_delay),
    }


def _solve_irmp_on_active_ids(
    instance: Instance,
    columns_all: list[dict[str, Any]],
    ship_cols_all: list[list[int]],
    active_ids: set[int],
    horizon: int,
    time_limit: int,
) -> dict[str, Any]:
    K_SP = int(instance.shore_berths) if hasattr(instance, "shore_berths") else int(instance.shore_cap)
    battery_cap = int(instance.battery_slots)
    columns, ship_cols, active_list = _make_pool(columns_all, ship_cols_all, active_ids)

    obj_lp, _, _, status_lp = _solve_master(
        columns,
        ship_cols,
        horizon,
        K_SP,
        battery_cap,
        relax=True,
        time_limit=time_limit,
    )
    _, solution, _, status_ip = _solve_master(
        columns,
        ship_cols,
        horizon,
        K_SP,
        battery_cap,
        relax=False,
        time_limit=time_limit,
    )
    if status_ip not in ("Optimal", "Not Solved", "Feasible"):
        solution = _brown_fallback(ship_cols, columns)

    extracted = _extract_solution(instance, columns, ship_cols, solution, obj_lp, status_lp, status_ip)
    extracted["pool_size"] = int(len(active_list))
    extracted["active_column_ids"] = active_list
    return extracted


def _run_baseline_pool(
    instance: Instance,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    op_mode = cfg.get("operation_mode", "simops")
    cg_cfg = cfg.get("cg", {})
    max_iters = int(cg_cfg.get("max_iters", 20))
    time_limit = int(cg_cfg.get("time_limit", 60))
    pricing_top_k = int(cg_cfg.get("pricing_top_k", 3))
    pricing_eps = float(cg_cfg.get("pricing_eps", 1e-6))
    use_full_pool_small = bool(cg_cfg.get("use_full_pool_small", True))
    full_pool_n = int(cg_cfg.get("full_pool_n", 100))
    method = str(cfg.get("method", "cg")).lower()

    restricted_cfg = dict(cg_cfg.get("restricted_pricing", {}))
    is_restricted = method.startswith("rcg") or bool(restricted_cfg.get("enabled", False))
    if is_restricted:
        restricted_cfg.setdefault("enabled", True)
        restricted_cfg.setdefault("fraction", 0.5)
        restricted_cfg.setdefault("selection", "random")
        restricted_cfg.setdefault("max_iters", 5)
        use_full_pool_small = False
        max_iters = int(restricted_cfg["max_iters"])

    columns_all, ship_cols_all, horizon = _build_columns(instance, op_mode)
    K_SP = int(instance.shore_berths) if hasattr(instance, "shore_berths") else int(instance.shore_cap)
    battery_cap = int(instance.battery_slots)

    if use_full_pool_small and instance.N <= full_pool_n:
        active_ids = set(range(len(columns_all)))
        obj_lp, _, duals, status_lp = _solve_master(
            columns_all,
            ship_cols_all,
            horizon,
            K_SP,
            battery_cap,
            relax=True,
            time_limit=time_limit,
        )
        baseline = _solve_irmp_on_active_ids(instance, columns_all, ship_cols_all, active_ids, horizon, time_limit)
        baseline.update(
            {
                "duals": duals,
                "baseline_pool_complete": True,
                "used_full_pool_small": True,
                "pricing_calls": 0,
                "num_columns_added_during_cg": 0,
                "min_reduced_cost_last": 0.0,
                "num_iters": 1,
                "lp_status": status_lp,
            }
        )
    else:
        if is_restricted:
            active_ids = _build_greedy_seed_column_ids(instance, columns_all, op_mode)
            subset_fraction = float(restricted_cfg.get("fraction", 0.5))
            subset_rule = str(restricted_cfg.get("selection", "random")).lower()
            subset_rng = np.random.default_rng(instance.seed)
        else:
            active_ids = set()
            for col_ids in ship_cols_all:
                brown_id = next(j for j in col_ids if columns_all[j]["mode"] == "brown")
                active_ids.add(brown_id)

        pricing_calls = 0
        num_added = 0
        min_rc = 0.0
        last_status_lp = ""
        duals: dict[str, Any] = {"ship": {}, "shore": {}, "battery": {}}
        num_iters = 0

        for it in range(1, max_iters + 1):
            columns, ship_cols, _ = _make_pool(columns_all, ship_cols_all, active_ids)
            obj_lp, _, duals, status_lp = _solve_master(
                columns,
                ship_cols,
                horizon,
                K_SP,
                battery_cap,
                relax=True,
                time_limit=time_limit,
            )
            last_status_lp = status_lp
            num_iters = it
            if status_lp not in ("Optimal", "Not Solved"):
                break

            if is_restricted:
                pricing_ship_indices = _select_pricing_ship_subset(instance, subset_fraction, subset_rule, subset_rng)
            else:
                pricing_ship_indices = list(range(instance.N))

            new_cols: list[int] = []
            min_rc_iter = 0.0
            for i in pricing_ship_indices:
                rc_list: list[tuple[float, int]] = []
                pi = float(duals["ship"].get(i, 0.0))
                pricing_calls += 1
                for col_id in ship_cols_all[i]:
                    col = columns_all[col_id]
                    rc = float(col["cost"]) - pi
                    rc -= sum(float(duals["shore"].get((k, t), 0.0)) for (k, t) in col.get("shore_berth_use", []))
                    rc -= sum(float(duals["battery"].get(t, 0.0)) for t in col["battery_use"])
                    if rc < -pricing_eps:
                        rc_list.append((rc, col_id))
                rc_list.sort(key=lambda pair: pair[0])
                for rc, col_id in rc_list[:pricing_top_k]:
                    if col_id not in active_ids:
                        new_cols.append(col_id)
                        min_rc_iter = min(min_rc_iter, float(rc))

            min_rc = float(min_rc_iter)
            if not new_cols:
                break
            for col_id in new_cols:
                active_ids.add(col_id)
            num_added += len(new_cols)

        baseline = _solve_irmp_on_active_ids(instance, columns_all, ship_cols_all, active_ids, horizon, time_limit)
        baseline.update(
            {
                "duals": duals,
                "baseline_pool_complete": len(active_ids) == len(columns_all),
                "used_full_pool_small": False,
                "pricing_calls": int(pricing_calls),
                "num_columns_added_during_cg": int(num_added),
                "min_reduced_cost_last": float(min_rc),
                "num_iters": int(num_iters),
                "lp_status": last_status_lp or baseline["lp_status"],
            }
        )

    pool_mode_counts = {"SP": 0, "BS": 0, "AE": 0}
    for col_id in baseline["active_column_ids"]:
        pool_mode_counts[_mode_label(columns_all[col_id]["mode"])] += 1

    baseline.update(
        {
            "columns_all": columns_all,
            "ship_cols_all": ship_cols_all,
            "horizon": int(horizon),
            "time_limit": int(time_limit),
            "baseline_pool_size": int(len(baseline["active_column_ids"])),
            "full_column_pool_size": int(len(columns_all)),
            "baseline_pool_mode_counts": pool_mode_counts,
        }
    )
    return baseline


@dataclass
class EnrichmentResult:
    active_ids: set[int]
    added_ids: list[int]
    added_mode_counts: dict[str, int]
    epsilon_abs: float


def enrich_column_pool(
    instance: Instance,
    baseline_snapshot: dict[str, Any],
    epsilon_loose: float,
    *,
    positive_tol: float = 0.0,
) -> EnrichmentResult:
    del instance
    baseline_ids = set(int(j) for j in baseline_snapshot["active_column_ids"])
    duals = baseline_snapshot["duals"]
    columns_all = baseline_snapshot["columns_all"]

    added_ids: list[int] = []
    added_mode_counts = {"SP": 0, "BS": 0, "AE": 0}
    for col_id, col in enumerate(columns_all):
        if col_id in baseline_ids:
            continue
        rc = _compute_reduced_cost(col, duals)
        if -float(epsilon_loose) <= rc <= float(positive_tol):
            added_ids.append(col_id)
            added_mode_counts[_mode_label(str(col["mode"]))] += 1

    enriched_ids = set(baseline_ids)
    enriched_ids.update(added_ids)
    return EnrichmentResult(
        active_ids=enriched_ids,
        added_ids=added_ids,
        added_mode_counts=added_mode_counts,
        epsilon_abs=float(epsilon_loose),
    )


def capture_baseline_pool(instance: Instance, cfg: dict[str, Any]) -> dict[str, Any]:
    return _run_baseline_pool(instance, cfg)


def solve_enriched_from_baseline(
    instance: Instance,
    baseline: dict[str, Any],
    epsilon_ratio: float,
) -> dict[str, Any]:
    epsilon_abs = float(epsilon_ratio) * float(baseline["objective"])
    enrichment = enrich_column_pool(instance, baseline, epsilon_abs)

    enriched = _solve_irmp_on_active_ids(
        instance,
        baseline["columns_all"],
        baseline["ship_cols_all"],
        enrichment.active_ids,
        int(baseline["horizon"]),
        int(baseline["time_limit"]),
    )
    baseline_plans = baseline["plan_keys"]
    enriched_plans = enriched["plan_keys"]
    changed = [idx for idx, (left, right) in enumerate(zip(baseline_plans, enriched_plans), start=1) if left != right]
    improvement_pct = max(0.0, (float(baseline["objective"]) - float(enriched["objective"])) / max(float(baseline["objective"]), 1e-9) * 100.0)

    enriched.update(
        {
            "epsilon_ratio": float(epsilon_ratio),
            "epsilon_abs": float(enrichment.epsilon_abs),
            "enriched_pool_size": int(len(enrichment.active_ids)),
            "columns_added": int(len(enrichment.added_ids)),
            "columns_by_mode_added": enrichment.added_mode_counts,
            "solutions_identical": len(changed) == 0,
            "n_vessels_changed": int(len(changed)),
            "changed_vessels": changed,
            "improvement_pct": float(improvement_pct),
        }
    )
    return enriched


def solve_baseline_and_enriched(
    instance: Instance,
    cfg: dict[str, Any],
    epsilon_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = capture_baseline_pool(instance, cfg)
    enriched = solve_enriched_from_baseline(instance, baseline, epsilon_ratio)
    return baseline, enriched

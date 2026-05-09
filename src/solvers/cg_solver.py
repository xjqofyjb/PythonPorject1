"""Column generation (CG) solver for the set-partitioning master problem.

Aligned with the corrected model:
  - Shore power columns include berth assignment (k)
  - RMP uses per-berth capacity constraints: sum_j (col j occupies berth k at t) <= 1
"""
from __future__ import annotations

import os
import time
from typing import Dict, Any, List, Tuple

import numpy as np

from src.instances import Instance
from src.io import ensure_dir, write_trace_csv
from src.metrics import compute_simops_metrics, compute_type_breakdown
from src.model_utils import energy_direct_cost, horizon_slots, operation_start_min


def _make_solver_candidates(pulp_module, time_limit: int) -> List[Any]:
    """Prefer stronger commercial solvers when available; fall back to CBC."""
    candidates: List[Any] = []

    gurobi_cmd_cls = getattr(pulp_module, "GUROBI_CMD", None)
    if gurobi_cmd_cls is not None:
        try:
            solver = gurobi_cmd_cls(msg=False, timeLimit=time_limit, gapRel=0.0, gapAbs=0.0, keepFiles=True)
            if solver.available():
                candidates.append(solver)
        except Exception:
            pass

    gurobi_cls = getattr(pulp_module, "GUROBI", None)
    if gurobi_cls is not None:
        try:
            solver = gurobi_cls(msg=False, timeLimit=time_limit, gapRel=0.0)
            if solver.available():
                candidates.append(solver)
        except Exception:
            pass

    try:
        solver = pulp_module.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        if solver.available():
            candidates.append(solver)
    except Exception:
        pass

    return candidates


def _column_key(col: Dict[str, Any]) -> Tuple[int, str, int, int, int]:
    return (
        int(col["ship"]),
        str(col["mode"]),
        int(col.get("berth", -1)),
        int(col["start"]),
        int(col["duration"]),
    )


def _make_column(instance: Instance, i: int, mode: str, start: int, duration: int, berth: int, op_mode: str) -> Dict[str, Any] | None:
    """Build one service-plan column if it obeys the revised feasibility rules."""
    horizon = int(getattr(instance, "horizon_steps", horizon_slots(instance.params)))
    start_min = operation_start_min(int(instance.arrival_steps[i]), int(instance.cargo_steps[i]), op_mode)
    if mode != "brown" and start < start_min:
        return None
    if start + duration > horizon:
        return None
    if mode == "shore":
        if not bool(instance.shore_compatible[i]):
            return None
        K_SP = int(instance.shore_berths) if hasattr(instance, "shore_berths") else int(instance.shore_cap)
        if berth < 0 or berth >= K_SP:
            return None
        direct_cost = energy_direct_cost(instance.shore_cost, instance.energy_kwh[i])
        shore_use = [(berth, tt) for tt in range(start, start + duration)]
        battery_use: List[int] = []
    elif mode == "battery":
        if int(instance.battery_slots) <= 0:
            return None
        direct_cost = energy_direct_cost(instance.battery_cost, instance.energy_kwh[i])
        berth = -1
        shore_use = []
        battery_use = list(range(start, start + duration))
    elif mode == "brown":
        if not bool(instance.params.get("brown_available", True)):
            return None
        start = int(instance.arrival_steps[i])
        duration = 0
        berth = -1
        direct_cost = energy_direct_cost(instance.brown_cost, instance.energy_kwh[i])
        shore_use = []
        battery_use = []
    else:
        return None

    completion = max(int(instance.arrival_steps[i] + instance.cargo_steps[i]), int(start + duration))
    if mode == "brown":
        completion = int(instance.arrival_steps[i] + instance.cargo_steps[i])
    tardy = max(0, completion - int(instance.deadline_steps[i]))
    delay_cost = float(instance.delay_costs[i]) * float(instance.dt_hours) * tardy
    return {
        "ship": int(i),
        "mode": mode,
        "start": int(start),
        "duration": int(duration),
        "berth": int(berth),
        "shore_berth_use": shore_use,
        "battery_use": battery_use,
        "direct_cost": float(direct_cost),
        "delay_cost": float(delay_cost),
        "cost": float(direct_cost + delay_cost),
        "completion": int(completion),
        "tardy": int(tardy),
    }


def _build_columns(instance: Instance, op_mode: str) -> Tuple[List[Dict[str, Any]], List[List[int]], int]:
    dt = float(instance.dt_hours)
    N = instance.N
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps
    sp_dur = instance.sp_duration_steps
    bs_dur = instance.bs_duration_steps

    K_SP = int(instance.shore_berths) if hasattr(instance, 'shore_berths') else int(instance.shore_cap)

    horizon = int(getattr(instance, "horizon_steps", horizon_slots(instance.params)))

    columns: List[Dict[str, Any]] = []
    ship_cols: List[List[int]] = [[] for _ in range(N)]
    brown_available = bool(instance.params.get("brown_available", True))

    for i in range(N):
        start_min = operation_start_min(int(arrival[i]), int(cargo[i]), op_mode)

        # Brown (AE) column
        if brown_available:
            col = _make_column(instance, i, "brown", int(arrival[i]), 0, -1, op_mode)
            if col is None:
                continue
            col_id = len(columns)
            columns.append(col)
            ship_cols[i].append(col_id)

        # Shore power columns: one column per (berth k, start time t)
        if instance.shore_compatible[i] and K_SP > 0:
            end_sp = horizon - int(sp_dur[i])
            for k in range(K_SP):
                for t in range(start_min, end_sp + 1):
                    col = _make_column(instance, i, "shore", int(t), int(sp_dur[i]), int(k), op_mode)
                    if col is None:
                        continue
                    col_id = len(columns)
                    columns.append(col)
                    ship_cols[i].append(col_id)

        # Battery swap columns
        end_bs = horizon - int(bs_dur[i])
        for t in range(start_min, end_bs + 1):
            col = _make_column(instance, i, "battery", int(t), int(bs_dur[i]), -1, op_mode)
            if col is None:
                continue
            col_id = len(columns)
            columns.append(col)
            ship_cols[i].append(col_id)

    return columns, ship_cols, horizon


def _build_column_lookup(columns: List[Dict[str, Any]]) -> Dict[Tuple[int, str, int, int], int]:
    lookup: Dict[Tuple[int, str, int, int], int] = {}
    for j, col in enumerate(columns):
        key = (int(col["ship"]), str(col["mode"]), int(col["start"]), int(col["berth"]))
        lookup[key] = j
    return lookup


def _build_unique_lookup(columns: List[Dict[str, Any]]) -> Dict[Tuple[int, str, int, int, int], int]:
    return {_column_key(col): j for j, col in enumerate(columns)}


def convert_solution_to_columns(
    instance: Instance,
    solution: Dict[str, Any],
    operation_mode: str,
    existing_columns: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Convert a returned schedule into service-plan columns."""
    schedule = solution.get("schedule", solution)
    modes = schedule.get("modes") or []
    starts_h = schedule.get("service_start_times") or schedule.get("start_times") or []
    durs_h = schedule.get("service_durations") or []
    berths = schedule.get("service_berths") or schedule.get("berths") or [-1] * len(modes)
    if not modes or not starts_h:
        return []

    existing = _build_unique_lookup(existing_columns or [])
    converted: List[Dict[str, Any]] = []
    for i, mode_raw in enumerate(modes):
        mode_map = {"SP": "shore", "BS": "battery", "AE": "brown"}
        mode = mode_map.get(str(mode_raw), str(mode_raw))
        if mode not in {"shore", "battery", "brown"}:
            continue
        start = int(round(float(starts_h[i]) / float(instance.dt_hours))) if i < len(starts_h) else int(instance.arrival_steps[i])
        if mode == "shore":
            duration = int(instance.sp_duration_steps[i])
        elif mode == "battery":
            duration = int(instance.bs_duration_steps[i])
        else:
            duration = 0
            start = int(instance.arrival_steps[i])
        berth = int(berths[i]) if i < len(berths) and berths[i] is not None else -1
        if mode == "shore" and berth < 0:
            # Reuse any feasible berth if the heuristic did not expose berth ids.
            berth = 0
        col = _make_column(instance, i, mode, start, duration, berth, operation_mode)
        if col is None:
            continue
        if _column_key(col) in existing:
            converted.append(existing_columns[existing[_column_key(col)]])  # type: ignore[index]
        else:
            converted.append(col)
    return converted


def inject_incumbent_solution_columns(
    instance: Instance,
    operation_mode: str,
    incumbent_solutions: List[Dict[str, Any]],
    existing_columns: List[Dict[str, Any]],
    ship_cols: List[List[int]],
) -> Tuple[List[Dict[str, Any]], List[List[int]], Dict[str, Any]]:
    """Inject feasible columns selected by incumbent schedules."""
    lookup = _build_unique_lookup(existing_columns)
    injected_by_method: Dict[str, int] = {}
    for incumbent in incumbent_solutions:
        method_name = str(incumbent.get("method", incumbent.get("name", "incumbent")))
        added = 0
        for col in convert_solution_to_columns(instance, incumbent, operation_mode, existing_columns):
            key = _column_key(col)
            if key in lookup:
                continue
            col_id = len(existing_columns)
            existing_columns.append(col)
            ship_cols[int(col["ship"])].append(col_id)
            lookup[key] = col_id
            added += 1
        injected_by_method[method_name] = injected_by_method.get(method_name, 0) + added
    return existing_columns, ship_cols, {
        "injected_columns_count": int(sum(injected_by_method.values())),
        "injected_incumbent_methods": ";".join(f"{k}:{v}" for k, v in injected_by_method.items() if v > 0),
    }


def _incumbent_active_ids(columns_all: List[Dict[str, Any]], incumbent_columns: List[Dict[str, Any]]) -> set[int]:
    lookup = _build_unique_lookup(columns_all)
    return {lookup[_column_key(col)] for col in incumbent_columns if _column_key(col) in lookup}


def _build_greedy_seed_column_ids(
    instance: Instance,
    columns_all: List[Dict[str, Any]],
    op_mode: str,
) -> set[int]:
    """Build one feasible seed column per ship using the same logic as the greedy baseline."""
    K_SP = int(instance.shore_berths) if hasattr(instance, "shore_berths") else int(instance.shore_cap)
    battery_cap = int(instance.battery_slots)
    brown_available = bool(instance.params.get("brown_available", True))

    dt = float(instance.dt_hours)
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps
    sp_dur = instance.sp_duration_steps
    bs_dur = instance.bs_duration_steps

    horizon = int(getattr(instance, "horizon_steps", horizon_slots(instance.params)))

    shore_usage = np.zeros((max(K_SP, 1), horizon), dtype=int)
    battery_usage = np.zeros(horizon, dtype=int)
    lookup = _build_column_lookup(columns_all)
    active_cols: set[int] = set()

    def feasible_shore(start: int, duration: int) -> int:
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
        return not np.any(battery_usage[start:end] >= battery_cap)

    order = np.argsort(arrival)
    for i in order:
        start_min = operation_start_min(int(arrival[i]), int(cargo[i]), op_mode)

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
                cost = energy_direct_cost(instance.shore_cost, demand) + delay_cost * dt * tardy
                if best is None or cost < best[0]:
                    best = (cost, "shore", t, duration, k)

        duration = int(bs_dur[i])
        for t in range(start_min, horizon - duration + 1):
            if not feasible_battery(t, duration):
                continue
            completion = max(arrival[i] + cargo[i], t + duration)
            tardy = max(0, completion - deadlines[i])
            cost = energy_direct_cost(instance.battery_cost, demand) + delay_cost * dt * tardy
            if best is None or cost < best[0]:
                best = (cost, "battery", t, duration, -1)

        if brown_available:
            completion = int(arrival[i] + cargo[i])
            tardy = max(0, completion - deadlines[i])
            cost = energy_direct_cost(instance.brown_cost, demand) + delay_cost * dt * tardy
            if best is None or cost < best[0]:
                best = (cost, "brown", int(arrival[i]), 0, -1)

        if best is None:
            best = (0.0, "brown", int(arrival[i]), 0, -1)

        _, mode, start, duration, berth = best
        col_id = lookup[(int(i), str(mode), int(start), int(berth))]
        active_cols.add(col_id)

        if mode == "shore" and berth >= 0:
            shore_usage[berth, start:start + duration] = 1
        elif mode == "battery":
            battery_usage[start:start + duration] += 1

    return active_cols


def _select_pricing_ship_subset(
    instance: Instance,
    fraction: float,
    selection_rule: str,
    rng: np.random.Generator,
) -> List[int]:
    n_pick = max(1, int(np.ceil(instance.N * fraction)))
    if selection_rule == "arrival":
        order = np.argsort(instance.arrival_steps)
        return [int(i) for i in order[:n_pick]]
    selected = rng.choice(instance.N, size=n_pick, replace=False)
    return [int(i) for i in np.sort(selected)]


def _positive_congestion_prices(duals: Dict[str, Any]) -> Tuple[Dict[Tuple[int, int], float], Dict[int, float]]:
    """Convert raw minimization <= duals to nonnegative congestion prices."""
    rho = {key: max(0.0, -float(value)) for key, value in duals.get("shore", {}).items()}
    eta = {int(key): max(0.0, -float(value)) for key, value in duals.get("battery", {}).items()}
    return rho, eta


def _reduced_cost(col: Dict[str, Any], pi_i: float, rho: Dict[Tuple[int, int], float], eta: Dict[int, float]) -> float:
    """Reduced cost using positive congestion prices with plus signs."""
    shore_congestion = sum(rho.get((int(k), int(t)), 0.0) for (k, t) in col.get("shore_berth_use", []))
    battery_congestion = sum(eta.get(int(t), 0.0) for t in col.get("battery_use", []))
    return float(col["cost"]) - float(pi_i) + shore_congestion + battery_congestion


def _solve_master(
    columns: List[Dict[str, Any]],
    ship_cols: List[List[int]],
    horizon: int,
    K_SP: int,
    battery_cap: int,
    relax: bool,
    time_limit: int,
) -> Tuple[float, Dict[int, float], Dict[str, Any], str]:
    try:
        import pulp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PuLP is required for CG master. Install with: pip install pulp") from exc

    prob = pulp.LpProblem("cg_master", pulp.LpMinimize)
    cat = "Continuous" if relax else "Binary"
    x = {j: pulp.LpVariable(f"x_{j}", lowBound=0, upBound=1, cat=cat) for j in range(len(columns))}

    # Ship assignment constraints: each ship assigned exactly one column
    ship_cons = {}
    for i, col_ids in enumerate(ship_cols):
        cons = pulp.lpSum(x[j] for j in col_ids) == 1
        prob += cons
        ship_cons[i] = cons

    # Per-berth shore power capacity: sum_j (col j occupies berth k at t) <= 1
    shore_cons = {}  # keyed by (k, t)
    if K_SP > 0:
        for k in range(K_SP):
            for t in range(horizon):
                cols_using = [j for j, c in enumerate(columns) if (k, t) in c.get("shore_berth_use", [])]
                if cols_using:
                    cons = pulp.lpSum(x[j] for j in cols_using) <= 1
                    prob += cons
                    shore_cons[k, t] = cons

    # Battery swap capacity (aggregated)
    battery_cons = {}
    for t in range(horizon):
        battery_cols = [j for j, c in enumerate(columns) if t in c["battery_use"]]
        if battery_cols:
            cons_batt = pulp.lpSum(x[j] for j in battery_cols) <= battery_cap
            prob += cons_batt
            battery_cons[t] = cons_batt

    prob += pulp.lpSum(columns[j]["cost"] * x[j] for j in range(len(columns)))

    def _solve_with_fallback() -> None:
        last_exc: Exception | None = None
        for solver_obj in _make_solver_candidates(pulp, time_limit):
            try:
                prob.solve(solver_obj)
                return
            except Exception as exc:
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No available MILP/LP solver found for CG master.")

    def _duals_missing() -> bool:
        for cons in ship_cons.values():
            if cons.pi is None:
                return True
        for cons in shore_cons.values():
            if cons.pi is None:
                return True
        for cons in battery_cons.values():
            if cons.pi is None:
                return True
        return False

    _solve_with_fallback()
    if relax and _duals_missing():
        # Fallback to CBC to ensure duals are available for pricing.
        prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))

    status = pulp.LpStatus.get(prob.status, "unknown")
    obj_val = pulp.value(prob.objective) if prob.objective is not None else None
    obj = float(obj_val) if obj_val is not None else float("nan")
    solution = {}
    for j in x:
        raw_value = pulp.value(x[j])
        solution[j] = float(raw_value) if raw_value is not None else 0.0

    duals = {"ship": {}, "shore": {}, "battery": {}}
    if relax:
        for i, cons in ship_cons.items():
            duals["ship"][i] = float(cons.pi)
        for (k, t), cons in shore_cons.items():
            duals["shore"][k, t] = float(cons.pi)
        for t, cons in battery_cons.items():
            duals["battery"][t] = float(cons.pi)
    return obj, solution, duals, status


def solve(instance: Instance, cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Solve instance using column generation and integerization."""
    t0 = time.perf_counter()
    op_mode = cfg.get("operation_mode", "simops")
    cg_cfg = cfg.get("cg", {})
    max_iters = int(cg_cfg.get("max_iters", 20))
    time_limit = int(cg_cfg.get("time_limit", 60))
    pricing_top_k = int(cg_cfg.get("pricing_top_k", 3))
    pricing_eps = float(cg_cfg.get("pricing_eps", 1e-6))
    min_iters = int(cg_cfg.get("min_iters", 0))
    stabilization_window = int(cg_cfg.get("stabilization_window", 5))
    stabilization_tol = float(cg_cfg.get("stabilization_rel_improvement", 1e-4))
    stabilization_gap_pct = float(cg_cfg.get("stabilization_gap_pct", 0.01))
    use_incumbent_injection = bool(cg_cfg.get("use_incumbent_injection", False))
    use_full_pool_small = bool(cg_cfg.get("use_full_pool_small", True))
    full_pool_n = int(cg_cfg.get("full_pool_n", 100))
    method = str(cfg.get("method", "cg")).lower()

    restricted_cfg = dict(cg_cfg.get("restricted_pricing", {}))
    is_restricted = method.startswith("rcg") or bool(restricted_cfg.get("enabled", False))
    if is_restricted:
        restricted_cfg.setdefault("enabled", True)
        restricted_cfg.setdefault("fraction", 0.5)
        if "arrival" in method:
            restricted_cfg.setdefault("selection", "arrival")
        else:
            restricted_cfg.setdefault("selection", "random")
        restricted_cfg.setdefault("max_iters", 5)
        use_full_pool_small = False
        max_iters = int(restricted_cfg["max_iters"])

    K_SP = int(instance.shore_berths) if hasattr(instance, 'shore_berths') else int(instance.shore_cap)

    columns_all, ship_cols_all, horizon = _build_columns(instance, op_mode)
    injection_meta = {"injected_columns_count": 0, "injected_incumbent_methods": ""}
    objective_stabilized = False
    relative_improvement_last_5 = float("nan")

    final_lp_obj = float("nan")

    if use_full_pool_small and instance.N <= full_pool_n:
        obj_lp, _, duals, status_lp = _solve_master(
            columns_all,
            ship_cols_all,
            horizon,
            K_SP,
            int(instance.battery_slots),
            relax=True,
            time_limit=time_limit,
        )
        final_lp_obj = float(obj_lp)
        obj_ip, solution, _, status_ip = _solve_master(
            columns_all,
            ship_cols_all,
            horizon,
            K_SP,
            int(instance.battery_slots),
            relax=False,
            time_limit=time_limit,
        )
        trace_rows = [
            {
                "iteration": 1,
                "wall_time": time.perf_counter() - t0,
                "rmp_lp_obj": obj_lp,
                "lp_obj_final_pool_if_available": obj_lp,
                "irmp_obj_checkpoint_if_solved": obj_ip,
                "columns_added_this_iter": 0,
                "total_columns": len(columns_all),
                "pricing_calls_cum": 0,
                "min_reduced_cost_scanned_last": 0.0,
                "num_negative_columns_scanned_last": 0,
                "num_columns_added_last": 0,
                "min_reduced_cost_added_last": 0.0,
                "min_reduced_cost_last": 0.0,
                "num_negative_columns_last": 0,
                "pool_gap_pct_if_available": 100.0 * max(0.0, (obj_ip - obj_lp) / max(1.0, abs(obj_ip))),
                "relative_improvement_last_5": 0.0,
                "pricing_converged": True,
                "objective_stabilized": False,
            }
        ]
        pricing_calls = 0
        num_added = 0
        min_rc_scanned_last = 0.0
        min_rc_added_last = 0.0
        num_negative_last = 0
        num_columns_added_last = 0
        pricing_time = 0.0
        columns = columns_all
        ship_cols = ship_cols_all
    else:
        if is_restricted:
            active_cols = _build_greedy_seed_column_ids(instance, columns_all, op_mode)
            subset_fraction = float(restricted_cfg.get("fraction", 0.5))
            subset_rule = str(restricted_cfg.get("selection", "random")).lower()
            subset_rng = np.random.default_rng(instance.seed)
        else:
            # Start with the lowest-cost column per ship (brown).
            active_cols = set()
            for col_ids in ship_cols_all:
                brown_id = next(j for j in col_ids if columns_all[j]["mode"] == "brown")
                active_cols.add(brown_id)

        incumbent_solutions = list(cg_cfg.get("incumbent_solutions", []))
        if use_incumbent_injection:
            try:
                from src.solvers import greedy_solver, fifo_solver
                inc_cfg = {"operation_mode": op_mode, "return_schedule": True}
                greedy_sol = greedy_solver.solve(instance, inc_cfg, logger)
                greedy_sol["method"] = "Greedy"
                fifo_sol = fifo_solver.solve(instance, inc_cfg, logger)
                fifo_sol["method"] = "FIFO"
                incumbent_solutions.extend([greedy_sol, fifo_sol])
            except Exception as exc:
                logger.info("Incumbent injection heuristic generation failed: %s", exc)
        if incumbent_solutions:
            inc_cols: List[Dict[str, Any]] = []
            for incumbent in incumbent_solutions:
                inc_cols.extend(convert_solution_to_columns(instance, incumbent, op_mode, columns_all))
            injected_ids = _incumbent_active_ids(columns_all, inc_cols)
            active_cols.update(injected_ids)
            by_method: Dict[str, int] = {}
            for incumbent in incumbent_solutions:
                method_name = str(incumbent.get("method", incumbent.get("name", "incumbent")))
                count = len(_incumbent_active_ids(columns_all, convert_solution_to_columns(instance, incumbent, op_mode, columns_all)))
                by_method[method_name] = by_method.get(method_name, 0) + count
            injection_meta = {
                "injected_columns_count": int(len(injected_ids)),
                "injected_incumbent_methods": ";".join(f"{k}:{v}" for k, v in by_method.items() if v > 0),
            }

        pricing_calls = 0
        num_added = 0
        min_rc_scanned_last = 0.0
        min_rc_added_last = 0.0
        num_negative_last = 0
        num_columns_added_last = 0
        trace_rows = []
        pricing_time = 0.0
        for it in range(1, max_iters + 1):
            active_list = sorted(active_cols)
            columns = [columns_all[j] for j in active_list]
            map_old_to_new = {old: new for new, old in enumerate(active_list)}
            ship_cols = [
                [map_old_to_new[j] for j in ship_cols_all[i] if j in active_cols]
                for i in range(instance.N)
            ]

            obj_lp, _, duals, status_lp = _solve_master(
                columns,
                ship_cols,
                horizon,
                K_SP,
                int(instance.battery_slots),
                relax=True,
                time_limit=time_limit,
            )
            if status_lp not in ("Optimal", "Not Solved"):
                break

            # Pricing: find columns with negative reduced cost.
            # Raw <= duals in a minimization model are converted to positive
            # congestion prices and then added to the generalized cost.
            rho, eta = _positive_congestion_prices(duals)
            new_cols = []
            pricing_start = time.perf_counter()
            min_rc_scanned_iter = float("inf")
            min_rc_added_iter = float("inf")
            num_negative_iter = 0
            if is_restricted:
                pricing_ship_indices = _select_pricing_ship_subset(instance, subset_fraction, subset_rule, subset_rng)
            else:
                pricing_ship_indices = list(range(instance.N))

            for i in pricing_ship_indices:
                col_ids = ship_cols_all[i]
                pricing_calls += 1
                rc_list = []
                pi = duals["ship"].get(i, 0.0)
                for col_id in col_ids:
                    col = columns_all[col_id]
                    rc = _reduced_cost(col, pi, rho, eta)
                    min_rc_scanned_iter = min(min_rc_scanned_iter, rc)
                    if rc < -pricing_eps:
                        num_negative_iter += 1
                        rc_list.append((rc, col_id))
                rc_list.sort(key=lambda x: x[0])
                for rc, col_id in rc_list[:pricing_top_k]:
                    if col_id not in active_cols:
                        new_cols.append(col_id)
                        min_rc_added_iter = min(min_rc_added_iter, rc)
            if not np.isfinite(min_rc_scanned_iter):
                min_rc_scanned_iter = 0.0
            if not np.isfinite(min_rc_added_iter):
                min_rc_added_iter = 0.0
            min_rc_scanned_last = float(min_rc_scanned_iter)
            min_rc_added_last = float(min_rc_added_iter)
            num_negative_last = int(num_negative_iter)
            num_columns_added_last = int(len(new_cols))
            pricing_time += time.perf_counter() - pricing_start

            if not new_cols:
                relative_improvement_last_5 = 0.0
                pricing_converged_iter = bool(num_negative_iter == 0 or min_rc_scanned_last >= -pricing_eps)
                trace_rows.append(
                    {
                        "iteration": it,
                        "wall_time": time.perf_counter() - t0,
                        "rmp_lp_obj": obj_lp,
                        "lp_obj_final_pool_if_available": np.nan,
                        "irmp_obj_checkpoint_if_solved": np.nan,
                        "columns_added_this_iter": 0,
                        "total_columns": len(active_cols),
                        "pricing_calls_cum": pricing_calls,
                        "min_reduced_cost_scanned_last": min_rc_scanned_last,
                        "num_negative_columns_scanned_last": num_negative_iter,
                        "num_columns_added_last": 0,
                        "min_reduced_cost_added_last": min_rc_added_last,
                        "min_reduced_cost_last": min_rc_scanned_last,
                        "num_negative_columns_last": num_negative_iter,
                        "pool_gap_pct_if_available": np.nan,
                        "relative_improvement_last_5": relative_improvement_last_5,
                        "pricing_converged": pricing_converged_iter,
                        "objective_stabilized": False,
                    }
                )
                break

            for col_id in new_cols:
                active_cols.add(col_id)
            num_added += len(new_cols)
            if len(trace_rows) >= stabilization_window:
                prev = float(trace_rows[-stabilization_window]["rmp_lp_obj"])
                relative_improvement_last_5 = max(0.0, (prev - float(obj_lp)) / max(1.0, abs(float(prev))))
            else:
                relative_improvement_last_5 = float("nan")
            trace_rows.append(
                {
                    "iteration": it,
                    "wall_time": time.perf_counter() - t0,
                    "rmp_lp_obj": obj_lp,
                    "lp_obj_final_pool_if_available": np.nan,
                    "irmp_obj_checkpoint_if_solved": np.nan,
                    "columns_added_this_iter": len(new_cols),
                    "total_columns": len(active_cols),
                    "pricing_calls_cum": pricing_calls,
                    "min_reduced_cost_scanned_last": min_rc_scanned_last,
                    "num_negative_columns_scanned_last": num_negative_iter,
                    "num_columns_added_last": len(new_cols),
                    "min_reduced_cost_added_last": min_rc_added_last,
                    "min_reduced_cost_last": min_rc_scanned_last,
                    "num_negative_columns_last": num_negative_iter,
                    "pool_gap_pct_if_available": np.nan,
                    "relative_improvement_last_5": relative_improvement_last_5,
                    "pricing_converged": False,
                    "objective_stabilized": False,
                }
            )
            if it >= min_iters and np.isfinite(relative_improvement_last_5) and relative_improvement_last_5 < stabilization_tol:
                objective_stabilized = True
                trace_rows[-1]["objective_stabilized"] = True
                break

        active_list = sorted(active_cols)
        columns = [columns_all[j] for j in active_list]
        map_old_to_new = {old: new for new, old in enumerate(active_list)}
        ship_cols = [
            [map_old_to_new[j] for j in ship_cols_all[i] if j in active_cols]
            for i in range(instance.N)
        ]

        final_lp_obj, _, _, _ = _solve_master(
            columns,
            ship_cols,
            horizon,
            K_SP,
            int(instance.battery_slots),
            relax=True,
            time_limit=time_limit,
        )
        obj_ip, solution, _, status_ip = _solve_master(
            columns,
            ship_cols,
            horizon,
            K_SP,
            int(instance.battery_slots),
            relax=False,
            time_limit=time_limit,
        )
        if status_ip not in ("Optimal", "Not Solved", "Feasible"):
            # Fallback to brown-only schedule if IP fails.
            solution = {j: 0.0 for j in range(len(columns))}
            for i, col_ids in enumerate(ship_cols):
                for j in col_ids:
                    if columns[j]["mode"] == "brown":
                        solution[j] = 1.0
                        break
        if trace_rows:
            pool_gap_checkpoint = float("nan")
            if np.isfinite(final_lp_obj) and np.isfinite(obj_ip):
                pool_gap_checkpoint = 100.0 * max(0.0, float((obj_ip - final_lp_obj) / max(1.0, abs(obj_ip))))
            trace_rows[-1]["lp_obj_final_pool_if_available"] = float(final_lp_obj)
            trace_rows[-1]["irmp_obj_checkpoint_if_solved"] = float(obj_ip)
            trace_rows[-1]["pool_gap_pct_if_available"] = pool_gap_checkpoint
            if trace_rows[-1].get("objective_stabilized") and np.isfinite(pool_gap_checkpoint) and pool_gap_checkpoint >= stabilization_gap_pct:
                objective_stabilized = False
                trace_rows[-1]["objective_stabilized"] = False

    # ---- Extract solution ----
    service_starts = np.full(instance.N, np.nan)
    service_durs = np.zeros(instance.N, dtype=float)
    start_steps = np.zeros(instance.N, dtype=int)
    duration_steps = np.zeros(instance.N, dtype=int)
    service_berths = np.full(instance.N, -1, dtype=int)
    modes: List[str] = ["brown"] * instance.N
    mode_counts = {"shore": 0, "battery": 0, "brown": 0}
    cost_energy_val = 0.0
    cost_delay_val = 0.0
    dt = float(instance.dt_hours)
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps

    for i, col_ids in enumerate(ship_cols):
        chosen = None
        for j in col_ids:
            if solution.get(j, 0.0) > 0.5:
                chosen = columns[j]
                break
        if chosen is None:
            chosen = min((columns[j] for j in col_ids), key=lambda c: c["cost"])

        mode = chosen["mode"]
        mode_counts[mode] += 1
        service_starts[i] = float(chosen["start"]) * dt
        service_durs[i] = float(chosen["duration"]) * dt
        start_steps[i] = int(chosen["start"])
        duration_steps[i] = int(chosen["duration"])
        service_berths[i] = int(chosen.get("berth", -1))
        modes[i] = mode

        energy = instance.energy_kwh[i]
        if mode == "shore":
            cost_energy_val += instance.shore_cost * energy
        elif mode == "battery":
            cost_energy_val += instance.battery_cost * energy
        else:
            cost_energy_val += instance.brown_cost * energy

        completion = max(arrival[i] + cargo[i], chosen["start"] + chosen["duration"])
        tardy = max(0.0, completion - deadlines[i])
        cost_delay_val += instance.delay_costs[i] * dt * tardy

    runtime_total = time.perf_counter() - t0
    pricing_time_share = pricing_time / runtime_total if runtime_total > 0 else 0.0

    trace_dir = cfg.get("trace_dir")
    instance_id = cfg.get("instance_id")
    method = cfg.get("method", "cg")
    if trace_dir and instance_id:
        ensure_dir(trace_dir)
        trace_path = os.path.join(trace_dir, f"{instance_id}_{method}.csv")
        write_trace_csv(trace_path, trace_rows)

    total_cost_val = float(cost_energy_val + cost_delay_val)
    gap = float("nan")
    gap_pct = float("nan")
    if np.isfinite(final_lp_obj) and abs(final_lp_obj) > 1e-9:
        gap_pct = 100.0 * max(0.0, float((total_cost_val - final_lp_obj) / max(1.0, abs(total_cost_val))))
        gap = gap_pct / 100.0

    full_pricing_converged = bool(use_full_pool_small and instance.N <= full_pool_n)
    pricing_converged = full_pricing_converged
    if full_pricing_converged:
        cg_status = "full_pricing_converged"
        gap_type = "Full-CG LP-IP gap"
        min_reduced_cost = 0.0
    else:
        converged_budget = bool(num_negative_last == 0 or (np.isfinite(min_rc_scanned_last) and min_rc_scanned_last >= -pricing_eps))
        pricing_converged = converged_budget
        if converged_budget:
            cg_status = "full_pricing_converged"
            gap_type = "Full-CG LP-IP gap"
        elif objective_stabilized:
            cg_status = "budgeted_stabilized"
            gap_type = "Pool LP-IP gap"
        else:
            cg_status = "budgeted_max_iter"
            gap_type = "Pool LP-IP gap"
        min_reduced_cost = float(min_rc_scanned_last)

    result = {
        "obj": total_cost_val,
        "runtime_total": float(runtime_total),
        "runtime_pricing": float(pricing_time),
        "gap": gap,
        "gap_pct": gap_pct,
        "lp_lower_bound": float(final_lp_obj) if np.isfinite(final_lp_obj) else float("nan"),
        "lp_obj_final_pool": float(final_lp_obj) if np.isfinite(final_lp_obj) else float("nan"),
        "irmp_obj": total_cost_val,
        "pool_gap_pct": gap_pct,
        "min_reduced_cost_scanned_last": float(min_rc_scanned_last),
        "num_negative_columns_scanned_last": int(num_negative_last),
        "num_columns_added_last": int(num_columns_added_last),
        "min_reduced_cost_added_last": float(min_rc_added_last),
        "best_reduced_cost_last": float(min_rc_scanned_last),
        "num_negative_columns_last": int(num_negative_last),
        "num_columns_total": len(columns),
        "iterations": len(trace_rows),
        "runtime_sec": float(runtime_total),
        "cg_status": cg_status,
        "gap_type": gap_type,
        "full_pricing_converged": pricing_converged,
        "pricing_converged": pricing_converged,
        "objective_stabilized": bool(objective_stabilized),
        "relative_improvement_last_5": float(relative_improvement_last_5) if np.isfinite(relative_improvement_last_5) else float("nan"),
        "injected_columns_count": int(injection_meta["injected_columns_count"]),
        "injected_incumbent_methods": str(injection_meta["injected_incumbent_methods"]),
        "min_reduced_cost": min_reduced_cost,
        "status": "ok",
        "operation_mode": op_mode,
        "lp_status": status_lp if "status_lp" in locals() else "",
        "ip_status": status_ip if "status_ip" in locals() else "",
        "success": status_ip in ("Optimal", "Not Solved", "Feasible") if "status_ip" in locals() else True,
        "num_iters": len(trace_rows),
        "num_pricing_calls": pricing_calls,
        "num_columns_added": num_added,
        "n_columns_generated": len(columns),
        "min_reduced_cost_last": float(min_rc_scanned_last),
        "pricing_time_share": float(pricing_time_share),
        "mechanism_counts": mode_counts,
        "infeasible_jobs": 0,
        "cost_energy": cost_energy_val,
        "cost_delay": cost_delay_val,
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
            "service_berths": service_berths.tolist(),
            "modes": modes,
        }
    return result

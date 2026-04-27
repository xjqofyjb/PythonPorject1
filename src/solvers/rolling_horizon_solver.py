"""Rolling-horizon MILP baseline for port energy scheduling."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.instances import Instance
from src.metrics import compute_simops_metrics, compute_type_breakdown


@dataclass(frozen=True)
class Ship:
    """Rolling-horizon ship record in time-step units."""
    ship_id: int
    arrival_step: int
    cargo_step: int
    deadline_step: int
    energy_kwh: float
    delay_cost: float
    ship_type: str
    shore_compatible: bool
    sp_duration_step: int
    bs_duration_step: int


def _make_solver_candidates(pulp_module, time_limit: int) -> List[Tuple[str, Any]]:
    """Prefer stronger commercial backends when available; fall back to CBC."""
    candidates: List[Tuple[str, Any]] = []

    gurobi_cmd_cls = getattr(pulp_module, "GUROBI_CMD", None)
    if gurobi_cmd_cls is not None:
        try:
            solver = gurobi_cmd_cls(msg=False, timeLimit=time_limit, gapRel=0.0, gapAbs=0.0, keepFiles=True)
            if solver.available():
                candidates.append(("gurobi_cmd", solver))
        except Exception:
            pass

    gurobi_cls = getattr(pulp_module, "GUROBI", None)
    if gurobi_cls is not None:
        try:
            solver = gurobi_cls(msg=False, timeLimit=time_limit, gapRel=0.0)
            if solver.available():
                candidates.append(("gurobi", solver))
        except Exception:
            pass

    try:
        solver = pulp_module.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        if solver.available():
            candidates.append(("cbc", solver))
    except Exception:
        pass

    return candidates


def _build_ships_from_instance(instance: Instance) -> List[Ship]:
    ships: List[Ship] = []
    for i in range(instance.N):
        ships.append(
            Ship(
                ship_id=int(i),
                arrival_step=int(instance.arrival_steps[i]),
                cargo_step=int(instance.cargo_steps[i]),
                deadline_step=int(instance.deadline_steps[i]),
                energy_kwh=float(instance.energy_kwh[i]),
                delay_cost=float(instance.delay_costs[i]),
                ship_type=str(instance.ship_types[i]),
                shore_compatible=bool(instance.shore_compatible[i]),
                sp_duration_step=int(instance.sp_duration_steps[i]),
                bs_duration_step=int(instance.bs_duration_steps[i]),
            )
        )
    return ships


def _compute_horizon_steps(ships: List[Ship], params: Dict[str, Any]) -> int:
    dt = float(params.get("delta_t", 0.25))
    horizon_hours = float(params.get("T_horizon", params.get("T_horizon_hours", 48.0)))
    horizon_steps = int(math.ceil(horizon_hours / dt))
    if ships:
        max_finish = max(
            max(ship.arrival_step + ship.cargo_step, ship.deadline_step) + max(ship.sp_duration_step, ship.bs_duration_step) + 5
            for ship in ships
        )
        horizon_steps = max(horizon_steps, int(max_finish))
    return horizon_steps


def _evaluate_assignment(ship: Ship, mode: str, start_step: int, dt: float, c_sp: float, c_bs: float, c_ae: float) -> Tuple[float, float, float]:
    """Return total/energy/delay cost for a chosen assignment."""
    if mode == "SP":
        duration = ship.sp_duration_step
        energy_cost = c_sp * ship.energy_kwh
    elif mode == "BS":
        duration = ship.bs_duration_step
        energy_cost = c_bs * ship.energy_kwh
    else:
        duration = 0
        energy_cost = c_ae * ship.energy_kwh
        start_step = ship.arrival_step

    departure_step = max(ship.arrival_step + ship.cargo_step, start_step + duration)
    tardy_step = max(0, departure_step - ship.deadline_step)
    delay_cost = ship.delay_cost * dt * tardy_step
    return energy_cost + delay_cost, energy_cost, delay_cost


def _greedy_window_solution(
    window_ships: List[Ship],
    fixed_sp: np.ndarray,
    fixed_bs: np.ndarray,
    params: Dict[str, Any],
    operation_mode: str,
) -> Dict[int, Dict[str, Any]]:
    """Fallback greedy for one rolling window."""
    k_sp = int(params["K_SP"])
    k_bs = int(params["K_BS"])
    c_sp = float(params["C_SP"])
    c_bs = float(params["C_BS"])
    c_ae = float(params["C_AE"])
    dt = float(params["delta_t"])
    horizon = int(params["T_horizon_steps"])

    shore_usage = fixed_sp.copy()
    battery_usage = fixed_bs.copy()
    decisions: Dict[int, Dict[str, Any]] = {}

    def feasible_shore(start: int, dur: int) -> int:
        end = start + dur
        if end > horizon or k_sp <= 0:
            return -1
        for berth in range(k_sp):
            if not np.any(shore_usage[berth, start:end]):
                return berth
        return -1

    def feasible_bs(start: int, dur: int) -> bool:
        end = start + dur
        if end > horizon or k_bs <= 0:
            return False
        return not np.any(battery_usage[start:end] >= k_bs)

    for ship in sorted(window_ships, key=lambda s: (s.arrival_step, s.ship_id)):
        start_min = ship.arrival_step if operation_mode == "simops" else ship.arrival_step + ship.cargo_step
        best = None

        if ship.shore_compatible:
            dur = ship.sp_duration_step
            for start in range(start_min, max(start_min, horizon - dur + 1)):
                berth = feasible_shore(start, dur)
                if berth < 0:
                    continue
                total_cost, energy_cost, delay_cost = _evaluate_assignment(ship, "SP", start, dt, c_sp, c_bs, c_ae)
                if best is None or total_cost < best["obj"]:
                    departure = max(ship.arrival_step + ship.cargo_step, start + dur)
                    best = {
                        "ship_id": ship.ship_id,
                        "mode": "SP",
                        "start_step": int(start),
                        "berth_k": int(berth),
                        "duration_step": int(dur),
                        "departure_step": int(departure),
                        "obj": float(total_cost),
                        "cost_energy": float(energy_cost),
                        "cost_delay": float(delay_cost),
                    }

        dur = ship.bs_duration_step
        for start in range(start_min, max(start_min, horizon - dur + 1)):
            if not feasible_bs(start, dur):
                continue
            total_cost, energy_cost, delay_cost = _evaluate_assignment(ship, "BS", start, dt, c_sp, c_bs, c_ae)
            if best is None or total_cost < best["obj"]:
                departure = max(ship.arrival_step + ship.cargo_step, start + dur)
                best = {
                    "ship_id": ship.ship_id,
                    "mode": "BS",
                    "start_step": int(start),
                    "berth_k": -1,
                    "duration_step": int(dur),
                    "departure_step": int(departure),
                    "obj": float(total_cost),
                    "cost_energy": float(energy_cost),
                    "cost_delay": float(delay_cost),
                }

        total_cost, energy_cost, delay_cost = _evaluate_assignment(ship, "AE", ship.arrival_step, dt, c_sp, c_bs, c_ae)
        if best is None or total_cost < best["obj"]:
            departure = ship.arrival_step + ship.cargo_step
            best = {
                "ship_id": ship.ship_id,
                "mode": "AE",
                "start_step": int(ship.arrival_step),
                "berth_k": -1,
                "duration_step": 0,
                "departure_step": int(departure),
                "obj": float(total_cost),
                "cost_energy": float(energy_cost),
                "cost_delay": float(delay_cost),
            }

        decisions[ship.ship_id] = best
        if best["mode"] == "SP" and best["berth_k"] >= 0:
            shore_usage[best["berth_k"], best["start_step"]:best["start_step"] + best["duration_step"]] = 1
        elif best["mode"] == "BS":
            battery_usage[best["start_step"]:best["start_step"] + best["duration_step"]] += 1

    return decisions


def _extract_window_solution(
    window_ships: List[Ship],
    x_sp: Dict[Tuple[int, int], Any],
    y_bs: Dict[int, Any],
    s_sp: Dict[Tuple[int, int, int], Any],
    b_bs: Dict[Tuple[int, int], Any],
    start_sp: Dict[int, List[int]],
    start_bs: Dict[int, List[int]],
    arrival: Dict[int, int],
    cargo: Dict[int, int],
    dt: float,
    c_sp: float,
    c_bs: float,
    c_ae: float,
    pulp_module,
) -> Dict[int, Dict[str, Any]] | None:
    decisions: Dict[int, Dict[str, Any]] = {}
    for ship in window_ships:
        i = ship.ship_id
        mode = "AE"
        start = ship.arrival_step
        berth_k = -1
        duration = 0

        for k in range(int(max([kk for (_, kk) in x_sp.keys()], default=-1)) + 1):
            var = x_sp.get((i, k))
            if var is not None and pulp_module.value(var) is not None and pulp_module.value(var) > 0.5:
                mode = "SP"
                berth_k = k
                duration = ship.sp_duration_step
                for t in start_sp.get(i, []):
                    svar = s_sp.get((i, k, t))
                    if svar is not None and pulp_module.value(svar) is not None and pulp_module.value(svar) > 0.5:
                        start = t
                        break
                break

        if mode == "AE":
            y_var = y_bs.get(i)
            if y_var is not None and pulp_module.value(y_var) is not None and pulp_module.value(y_var) > 0.5:
                mode = "BS"
                duration = ship.bs_duration_step
                for t in start_bs.get(i, []):
                    bvar = b_bs.get((i, t))
                    if bvar is not None and pulp_module.value(bvar) is not None and pulp_module.value(bvar) > 0.5:
                        start = t
                        break

        departure = max(arrival[i] + cargo[i], start + duration)
        total_cost, energy_cost, delay_cost = _evaluate_assignment(ship, mode, start, dt, c_sp, c_bs, c_ae)
        decisions[i] = {
            "ship_id": i,
            "mode": mode,
            "start_step": int(start),
            "berth_k": int(berth_k),
            "duration_step": int(duration),
            "departure_step": int(departure),
            "obj": float(total_cost),
            "cost_energy": float(energy_cost),
            "cost_delay": float(delay_cost),
        }

    if len(decisions) != len(window_ships):
        return None
    return decisions


def solve_window_milp(
    window_ships: List[Ship],
    fixed_sp: np.ndarray,
    fixed_bs: np.ndarray,
    params: Dict[str, Any],
    time_limit: int,
    operation_mode: str = "simops",
) -> Dict[str, Any]:
    """Solve one rolling window with fixed prior occupancy deducted from capacity."""
    try:
        import pulp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PuLP is required for rolling-horizon MILP. Install with: pip install pulp") from exc

    t0 = time.perf_counter()
    k_sp = int(params["K_SP"])
    k_bs = int(params["K_BS"])
    c_sp = float(params["C_SP"])
    c_bs = float(params["C_BS"])
    c_ae = float(params["C_AE"])
    dt = float(params["delta_t"])
    horizon = int(params["T_horizon_steps"])

    ship_ids = [ship.ship_id for ship in window_ships]
    arrival = {ship.ship_id: int(ship.arrival_step) for ship in window_ships}
    cargo = {ship.ship_id: int(ship.cargo_step) for ship in window_ships}
    deadlines = {ship.ship_id: int(ship.deadline_step) for ship in window_ships}

    start_sp: Dict[int, List[int]] = {}
    start_bs: Dict[int, List[int]] = {}
    for ship in window_ships:
        start_min = ship.arrival_step if operation_mode == "simops" else ship.arrival_step + ship.cargo_step
        start_sp[ship.ship_id] = []
        start_bs[ship.ship_id] = []

        if ship.shore_compatible and k_sp > 0:
            last_sp = horizon - ship.sp_duration_step
            if last_sp >= start_min:
                start_sp[ship.ship_id] = list(range(start_min, last_sp + 1))

        last_bs = horizon - ship.bs_duration_step
        if last_bs >= start_min:
            start_bs[ship.ship_id] = list(range(start_min, last_bs + 1))

    prob = pulp.LpProblem("rolling_horizon_window", pulp.LpMinimize)

    x_sp: Dict[Tuple[int, int], Any] = {}
    y_bs: Dict[int, Any] = {}
    z_ae: Dict[int, Any] = {}
    w: Dict[Tuple[int, int], Any] = {}
    s_sp: Dict[Tuple[int, int, int], Any] = {}
    b_bs: Dict[Tuple[int, int], Any] = {}
    L: Dict[int, Any] = {}
    delta: Dict[int, Any] = {}

    for ship in window_ships:
        i = ship.ship_id
        y_bs[i] = pulp.LpVariable(f"y_bs_{i}", cat="Binary")
        z_ae[i] = pulp.LpVariable(f"z_ae_{i}", cat="Binary")
        L[i] = pulp.LpVariable(f"L_{i}", lowBound=0)
        delta[i] = pulp.LpVariable(f"delta_{i}", lowBound=0)

        if ship.shore_compatible and k_sp > 0:
            for k in range(k_sp):
                x_sp[i, k] = pulp.LpVariable(f"x_sp_{i}_{k}", cat="Binary")

        for t in sorted(set(start_sp[i]) | set(start_bs[i])):
            w[i, t] = pulp.LpVariable(f"w_{i}_{t}", cat="Binary")

        for k in range(k_sp):
            if not ship.shore_compatible:
                continue
            for t in start_sp[i]:
                s_sp[i, k, t] = pulp.LpVariable(f"s_sp_{i}_{k}_{t}", cat="Binary")

        for t in start_bs[i]:
            b_bs[i, t] = pulp.LpVariable(f"b_bs_{i}_{t}", cat="Binary")

    for ship in window_ships:
        i = ship.ship_id
        shore_sum = pulp.lpSum(x_sp[i, k] for k in range(k_sp) if (i, k) in x_sp)
        prob += shore_sum + y_bs[i] + z_ae[i] == 1

        w_sum = pulp.lpSum(w[i, t] for t in sorted(set(start_sp[i]) | set(start_bs[i])))
        prob += w_sum == shore_sum + y_bs[i]

        for k in range(k_sp):
            if not ship.shore_compatible:
                continue
            for t in start_sp[i]:
                s = s_sp[i, k, t]
                prob += s <= x_sp[i, k]
                prob += s <= w[i, t]
                prob += s >= x_sp[i, k] + w[i, t] - 1
            if start_sp[i]:
                prob += pulp.lpSum(s_sp[i, k, t] for t in start_sp[i]) == x_sp[i, k]

        for t in start_bs[i]:
            b = b_bs[i, t]
            prob += b <= y_bs[i]
            prob += b <= w[i, t]
            prob += b >= y_bs[i] + w[i, t] - 1
        if start_bs[i]:
            prob += pulp.lpSum(b_bs[i, t] for t in start_bs[i]) == y_bs[i]

        prob += L[i] >= arrival[i] + cargo[i]
        service_finish = (
            pulp.lpSum(t * w[i, t] for t in sorted(set(start_sp[i]) | set(start_bs[i])))
            + ship.sp_duration_step * shore_sum
            + ship.bs_duration_step * y_bs[i]
        )
        prob += L[i] >= service_finish
        prob += delta[i] >= L[i] - deadlines[i]

    for k in range(k_sp):
        for t in range(horizon):
            rhs = 1 - int(fixed_sp[k, t])
            if rhs < 0:
                rhs = 0
            occupancy = []
            for ship in window_ships:
                if not ship.shore_compatible:
                    continue
                dur = ship.sp_duration_step
                for tau in range(max(0, t - dur + 1), t + 1):
                    if (ship.ship_id, k, tau) in s_sp:
                        occupancy.append(s_sp[ship.ship_id, k, tau])
            if occupancy:
                prob += pulp.lpSum(occupancy) <= rhs

    for t in range(horizon):
        rhs = k_bs - int(fixed_bs[t])
        if rhs < 0:
            rhs = 0
        occupancy_bs = []
        for ship in window_ships:
            dur = ship.bs_duration_step
            for tau in range(max(0, t - dur + 1), t + 1):
                if (ship.ship_id, tau) in b_bs:
                    occupancy_bs.append(b_bs[ship.ship_id, tau])
        if occupancy_bs:
            prob += pulp.lpSum(occupancy_bs) <= rhs

    prob += pulp.lpSum(
        c_sp * ship.energy_kwh * pulp.lpSum(x_sp[ship.ship_id, k] for k in range(k_sp) if (ship.ship_id, k) in x_sp)
        + c_bs * ship.energy_kwh * y_bs[ship.ship_id]
        + c_ae * ship.energy_kwh * z_ae[ship.ship_id]
        + ship.delay_cost * dt * delta[ship.ship_id]
        for ship in window_ships
    )

    solver_name = "unknown"
    solved = False
    last_exc: Exception | None = None
    for candidate_name, solver in _make_solver_candidates(pulp, time_limit):
        try:
            prob.solve(solver)
            solver_name = candidate_name
            solved = True
            break
        except Exception as exc:
            last_exc = exc
            continue
    if not solved:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No available MILP solver found for rolling-horizon window.")

    status = pulp.LpStatus.get(prob.status, "unknown")
    obj_val = pulp.value(prob.objective) if prob.objective is not None else None
    decisions = _extract_window_solution(
        window_ships,
        x_sp,
        y_bs,
        s_sp,
        b_bs,
        start_sp,
        start_bs,
        arrival,
        cargo,
        dt,
        c_sp,
        c_bs,
        c_ae,
        pulp,
    )

    used_fallback = False
    if decisions is None or obj_val is None:
        decisions = _greedy_window_solution(window_ships, fixed_sp, fixed_bs, params, operation_mode)
        obj_val = sum(dec["obj"] for dec in decisions.values())
        used_fallback = True

    return {
        "status": status,
        "success": decisions is not None,
        "obj": float(obj_val),
        "decisions": decisions,
        "runtime_total": float(time.perf_counter() - t0),
        "solver_backend": solver_name,
        "used_fallback": used_fallback,
    }


def rolling_horizon_milp(
    ships: List[Ship],
    params: Dict[str, Any],
    window_size: int = 10,
    commit_size: int = 5,
    time_limit: int = 30,
    operation_mode: str = "simops",
) -> Dict[str, Any]:
    """Rolling-horizon MILP with fixed-capacity carry-over between windows."""
    t0 = time.perf_counter()
    ships_sorted = sorted(ships, key=lambda s: (s.arrival_step, s.ship_id))
    horizon_steps = _compute_horizon_steps(ships_sorted, params)
    params = dict(params)
    params["T_horizon_steps"] = horizon_steps

    k_sp = int(params["K_SP"])
    dt = float(params["delta_t"])

    fixed_sp = np.zeros((max(k_sp, 1), horizon_steps), dtype=int)
    fixed_bs = np.zeros(horizon_steps, dtype=int)
    committed_results: Dict[int, Dict[str, Any]] = {}

    cursor = 0
    while cursor < len(ships_sorted):
        window_ships = ships_sorted[cursor: cursor + window_size]
        if not window_ships:
            break

        window_result = solve_window_milp(window_ships, fixed_sp, fixed_bs, params, time_limit, operation_mode=operation_mode)
        decisions = window_result["decisions"]

        commit_count = min(commit_size, len(window_ships))
        commit_ships = window_ships[:commit_count]
        for ship in commit_ships:
            dec = decisions[ship.ship_id]
            committed_results[ship.ship_id] = dec
            if dec["mode"] == "SP" and dec["berth_k"] >= 0:
                fixed_sp[dec["berth_k"], dec["start_step"]:dec["start_step"] + dec["duration_step"]] = 1
            elif dec["mode"] == "BS":
                fixed_bs[dec["start_step"]:dec["start_step"] + dec["duration_step"]] += 1

        cursor += commit_count

    total_obj = sum(dec["obj"] for dec in committed_results.values())
    total_energy = sum(dec["cost_energy"] for dec in committed_results.values())
    total_delay = sum(dec["cost_delay"] for dec in committed_results.values())

    mode_map = {"SP": "shore", "BS": "battery", "AE": "brown"}
    mode_assignments = {ship_id: dec["mode"] for ship_id, dec in committed_results.items()}
    start_times = {ship_id: dec["start_step"] * dt for ship_id, dec in committed_results.items()}
    departure_times = {ship_id: dec["departure_step"] * dt for ship_id, dec in committed_results.items()}
    mode_counts = {"shore": 0, "battery": 0, "brown": 0}
    for dec in committed_results.values():
        mode_counts[mode_map[dec["mode"]]] += 1

    return {
        "obj": float(total_obj),
        "runtime_total": float(time.perf_counter() - t0),
        "mode_assignments": mode_assignments,
        "start_times": start_times,
        "departure_times": departure_times,
        "success": len(committed_results) == len(ships_sorted),
        "method": "rolling_horizon",
        "mechanism_counts": mode_counts,
        "cost_energy": float(total_energy),
        "cost_delay": float(total_delay),
        "committed_results": committed_results,
    }


def solve(instance: Instance, cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Adapter to the existing experiment framework."""
    rh_cfg = dict(cfg.get("rolling_horizon", {}))
    window_size = int(rh_cfg.get("window_size", 10))
    commit_size = int(rh_cfg.get("commit_size", max(1, window_size // 2)))
    time_limit = int(rh_cfg.get("time_limit", 30))
    operation_mode = str(cfg.get("operation_mode", "simops"))

    horizon_hours = float(instance.params.get("T_horizon", instance.params.get("T_horizon_hours", 48.0)))
    params = {
        "K_SP": int(instance.shore_berths) if hasattr(instance, "shore_berths") else int(instance.shore_cap),
        "K_BS": int(instance.battery_slots),
        "C_SP": float(instance.shore_cost),
        "C_BS": float(instance.battery_cost),
        "C_AE": float(instance.brown_cost),
        "T_BS": float(instance.battery_swap_hours),
        "P_SP": float(instance.shore_power_kw),
        "delta_t": float(instance.dt_hours),
        "T_horizon": horizon_hours,
    }

    ships = _build_ships_from_instance(instance)
    result = rolling_horizon_milp(
        ships,
        params,
        window_size=window_size,
        commit_size=commit_size,
        time_limit=time_limit,
        operation_mode=operation_mode,
    )

    start_steps = np.zeros(instance.N, dtype=int)
    duration_steps = np.zeros(instance.N, dtype=int)
    service_starts = np.full(instance.N, np.nan)
    service_durations = np.zeros(instance.N, dtype=float)
    modes: List[str] = ["brown"] * instance.N

    for ship in ships:
        dec = result["committed_results"].get(ship.ship_id)
        if dec is None:
            continue
        mode = {"SP": "shore", "BS": "battery", "AE": "brown"}[dec["mode"]]
        start_steps[ship.ship_id] = int(dec["start_step"])
        duration_steps[ship.ship_id] = int(dec["duration_step"])
        service_starts[ship.ship_id] = float(dec["start_step"]) * instance.dt_hours
        service_durations[ship.ship_id] = float(dec["duration_step"]) * instance.dt_hours
        modes[ship.ship_id] = mode

    result.update(
        {
            "status": "ok" if result["success"] else "infeasible",
            "infeasible_jobs": int(instance.N - len(result["committed_results"])),
        }
    )
    result.update(compute_type_breakdown(instance, modes, start_steps, duration_steps))
    simops = compute_simops_metrics(
        instance,
        operation_mode,
        service_start_times=service_starts,
        service_durations=service_durations,
    )
    result.update(simops)
    if cfg.get("return_schedule"):
        result["schedule"] = {
            "service_start_times": service_starts.tolist(),
            "service_durations": service_durations.tolist(),
            "modes": modes,
        }
    return result

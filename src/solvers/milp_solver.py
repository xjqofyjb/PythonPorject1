"""MILP solver for the time-indexed formulation.

Aligned with the corrected model in the paper:
  - Per-berth shore power variables s_{ikt} with McCormick linearization
  - Per-berth exclusive capacity constraint: sum_i u_{ikt} <= 1, forall k, t
"""
from __future__ import annotations

import time
from typing import Dict, Any, List, Tuple

import numpy as np

from src.instances import Instance
from src.metrics import compute_simops_metrics, compute_type_breakdown
from src.model_utils import horizon_slots, operation_start_min


def _make_solver_candidates(pulp_module, time_limit: int) -> List[Tuple[str, Any]]:
    """Prefer exact commercial backends when available; fall back to CBC."""
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


def _build_cover_index(start_times: List[List[int]], durations: np.ndarray, horizon: int) -> List[List[List[Tuple[int, int]]]]:
    cover: List[List[List[Tuple[int, int]]]] = [[] for _ in range(horizon)]
    for i, times in enumerate(start_times):
        dur = int(durations[i])
        for t0 in times:
            for t in range(t0, min(horizon, t0 + dur)):
                cover[t].append((i, t0))
    return cover


def solve(instance: Instance, cfg: Dict[str, Any], logger) -> Dict[str, Any]:
    """Solve instance using a time-indexed MILP with per-berth shore power model."""
    t0 = time.perf_counter()
    try:
        import pulp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PuLP is required for MILP. Install with: pip install pulp") from exc

    method = cfg.get("method", "milp60")
    time_limit = int(method.replace("milp", "")) if method.startswith("milp") else 60
    op_mode = cfg.get("operation_mode", "simops")

    dt = float(instance.dt_hours)
    N = instance.N
    arrival = instance.arrival_steps
    cargo = instance.cargo_steps
    deadlines = instance.deadline_steps
    sp_dur = instance.sp_duration_steps
    bs_dur = instance.bs_duration_steps

    # Number of shore power berths |K^SP|
    K_SP = int(instance.shore_berths) if hasattr(instance, 'shore_berths') else int(instance.shore_cap)
    K_BS = int(instance.battery_slots)

    horizon = int(getattr(instance, "horizon_steps", horizon_slots(instance.params)))

    # Feasible start times for each ship
    start_sp: List[List[int]] = []
    start_bs: List[List[int]] = []
    for i in range(N):
        start_min = operation_start_min(int(arrival[i]), int(cargo[i]), op_mode)
        start_max_sp = horizon - int(sp_dur[i])
        start_max_bs = horizon - int(bs_dur[i])
        sp_times = list(range(start_min, start_max_sp + 1)) if start_max_sp >= start_min else []
        bs_times = list(range(start_min, start_max_bs + 1)) if start_max_bs >= start_min else []
        start_sp.append(sp_times if instance.shore_compatible[i] else [])
        start_bs.append(bs_times)

    # Cover index for battery swapping (aggregated, unchanged)
    cover_bs = _build_cover_index(start_bs, bs_dur, horizon)

    prob = pulp.LpProblem("port_energy_milp", pulp.LpMinimize)

    # ---- Decision variables ----

    # x_sp[i,k]: ship i assigned to shore berth k (binary)
    x_sp = {}
    if K_SP > 0:
        for i in range(N):
            if instance.shore_compatible[i]:
                for k in range(K_SP):
                    x_sp[i, k] = pulp.LpVariable(f"x_sp_{i}_{k}", cat="Binary")

    # y_bs[i]: ship i uses battery swapping (binary)
    y_bs = {i: pulp.LpVariable(f"y_bs_{i}", cat="Binary") for i in range(N)}

    # z[i]: ship i uses brown/AE (binary)
    z = {i: pulp.LpVariable(f"z_{i}", cat="Binary") for i in range(N)}

    # w[i,t]: ship i starts energy service at time t (binary)
    w = {}
    for i in range(N):
        all_times = sorted(set(start_sp[i]) | set(start_bs[i]))
        for t in all_times:
            w[i, t] = pulp.LpVariable(f"w_{i}_{t}", cat="Binary")

    # s_sp[i,k,t]: ship i starts shore power at berth k, time t (McCormick auxiliary)
    s_sp = {}
    if K_SP > 0:
        for i in range(N):
            if instance.shore_compatible[i]:
                for k in range(K_SP):
                    for t in start_sp[i]:
                        s_sp[i, k, t] = pulp.LpVariable(f"s_sp_{i}_{k}_{t}", cat="Binary")

    # b_bs[i,t]: ship i starts battery swap at time t (McCormick auxiliary)
    b_bs = {}
    for i in range(N):
        for t in start_bs[i]:
            b_bs[i, t] = pulp.LpVariable(f"b_bs_{i}_{t}", cat="Binary")

    # L[i]: departure time, delta[i]: tardiness
    L = {i: pulp.LpVariable(f"L_{i}", lowBound=0) for i in range(N)}
    delta = {i: pulp.LpVariable(f"delta_{i}", lowBound=0) for i in range(N)}

    # ---- Constraints ----

    for i in range(N):
        # (1) Mode selection: sum_k x_sp[i,k] + y_bs[i] + z[i] = 1
        shore_sum = pulp.lpSum(x_sp[i, k] for k in range(K_SP) if (i, k) in x_sp)
        prob += shore_sum + y_bs[i] + z[i] == 1

        # Non-shore-compatible ships cannot use shore power
        if not instance.shore_compatible[i]:
            prob += shore_sum == 0

        # (2) Service start time uniqueness: sum_t w[i,t] = sum_k x_sp[i,k] + y_bs[i]
        w_sum = pulp.lpSum(w[i, t] for t in sorted(set(start_sp[i]) | set(start_bs[i])))
        prob += w_sum == shore_sum + y_bs[i]

        # (4) McCormick linearization for shore power: s_sp[i,k,t] = x_sp[i,k] * w[i,t]
        if instance.shore_compatible[i] and K_SP > 0:
            for k in range(K_SP):
                for t in start_sp[i]:
                    s = s_sp[i, k, t]
                    x = x_sp[i, k]
                    ww = w.get((i, t))
                    if ww is None:
                        prob += s == 0
                        continue
                    prob += s <= x
                    prob += s <= ww
                    prob += s >= x + ww - 1
                # Valid inequality: sum_t s_sp[i,k,t] = x_sp[i,k]
                prob += pulp.lpSum(s_sp[i, k, t] for t in start_sp[i]) == x_sp[i, k]

        # (5) McCormick linearization for battery swap: b_bs[i,t] = y_bs[i] * w[i,t]
        for t in start_bs[i]:
            b = b_bs[i, t]
            ww = w.get((i, t))
            if ww is None:
                prob += b == 0
                continue
            prob += b <= y_bs[i]
            prob += b <= ww
            prob += b >= y_bs[i] + ww - 1
        # Valid inequality: sum_t b_bs[i,t] = y_bs[i]
        prob += pulp.lpSum(b_bs[i, t] for t in start_bs[i]) == y_bs[i]

        # (7) SIMOPS completion time constraints
        prob += L[i] >= int(arrival[i] + cargo[i])
        # Service finish: L[i] >= sum_t t*w[i,t] + d^SP * sum_k x_sp[i,k] + d^BS * y_bs[i]
        service_finish = (
            pulp.lpSum(t * w[i, t] for t in sorted(set(start_sp[i]) | set(start_bs[i])) if (i, t) in w)
            + int(sp_dur[i]) * shore_sum
            + int(bs_dur[i]) * y_bs[i]
        )
        prob += L[i] >= service_finish

        # (8) Tardiness
        prob += delta[i] >= L[i] - int(deadlines[i])

    # (6) Per-berth shore power capacity: sum_i u[i,k,t] <= 1, forall k, t
    # u[i,k,t] = sum_{tau=max(0, t-d_i^SP+1)}^{t} s_sp[i,k,tau]
    if K_SP > 0:
        for k in range(K_SP):
            for t in range(horizon):
                occupancy = []
                for i in range(N):
                    if not instance.shore_compatible[i]:
                        continue
                    dur = int(sp_dur[i])
                    for tau in range(max(0, t - dur + 1), t + 1):
                        if (i, k, tau) in s_sp:
                            occupancy.append(s_sp[i, k, tau])
                if occupancy:
                    prob += pulp.lpSum(occupancy) <= 1

    # Battery swap capacity (aggregated, unchanged): sum_i v[i,t] <= K_BS
    for t in range(horizon):
        occupancy_bs = []
        for i in range(N):
            dur = int(bs_dur[i])
            for tau in range(max(0, t - dur + 1), t + 1):
                if (i, tau) in b_bs:
                    occupancy_bs.append(b_bs[i, tau])
        if occupancy_bs:
            prob += pulp.lpSum(occupancy_bs) <= K_BS

    # ---- Objective function ----
    energy_cost = []
    delay_cost = []
    for i in range(N):
        shore_sum = pulp.lpSum(x_sp[i, k] for k in range(K_SP) if (i, k) in x_sp)
        energy_cost.append(
            instance.shore_cost * instance.energy_kwh[i] * shore_sum
            + instance.battery_cost * instance.energy_kwh[i] * y_bs[i]
            + instance.brown_cost * instance.energy_kwh[i] * z[i]
        )
        delay_cost.append(instance.delay_costs[i] * dt * delta[i])

    prob += pulp.lpSum(energy_cost) + pulp.lpSum(delay_cost)

    # ---- Solve ----
    last_exc: Exception | None = None
    solved = False
    solver_name = "unknown"
    solver_candidates = _make_solver_candidates(pulp, time_limit)

    for candidate_name, solver in solver_candidates:
        try:
            prob.solve(solver)
            solved = True
            solver_name = candidate_name
            break
        except Exception as exc:
            last_exc = exc
            continue

    if not solved:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("No available MILP solver found.")

    status = pulp.LpStatus.get(prob.status, "unknown")
    runtime_total = time.perf_counter() - t0
    obj = float(pulp.value(prob.objective)) if prob.objective is not None else float("nan")
    status_label = "ok" if status == "Optimal" else status.lower()
    # Some external solver wrappers can surface a time-limited incumbent as "Optimal".
    if status == "Optimal" and time_limit > 0 and runtime_total >= 0.95 * time_limit:
        status_label = "not solved"

    # ---- Extract solution ----
    service_starts = np.full(N, np.nan)
    service_durs = np.zeros(N, dtype=float)
    start_steps = np.zeros(N, dtype=int)
    duration_steps = np.zeros(N, dtype=int)
    modes: List[str] = ["brown"] * N
    mode_counts = {"shore": 0, "battery": 0, "brown": 0}
    cost_energy_val = 0.0
    cost_delay_val = 0.0

    for i in range(N):
        chosen = "brown"
        start = None
        dur_steps = 0

        # Check shore power assignment
        for k in range(K_SP):
            if (i, k) not in x_sp:
                continue
            if pulp.value(x_sp[i, k]) > 0.5:
                chosen = "shore"
                dur_steps = int(sp_dur[i])
                # Find start time from s_sp
                for t in start_sp[i]:
                    if (i, k, t) in s_sp and pulp.value(s_sp[i, k, t]) > 0.5:
                        start = t
                        break
                # Fallback: find start from w[i,t] if s_sp missed due to precision
                if start is None:
                    all_times = sorted(set(start_sp[i]) | set(start_bs[i]))
                    for t in all_times:
                        if (i, t) in w and pulp.value(w[i, t]) > 0.5:
                            start = t
                            break
                if start is None:
                    # Last resort: earliest feasible time
                    start = start_sp[i][0] if start_sp[i] else int(arrival[i])
                break

        # Check battery swap
        if chosen == "brown":
            if pulp.value(y_bs[i]) > 0.5:
                chosen = "battery"
                dur_steps = int(bs_dur[i])
                for t in start_bs[i]:
                    if (i, t) in b_bs and pulp.value(b_bs[i, t]) > 0.5:
                        start = t
                        break
                # Fallback
                if start is None:
                    all_times = sorted(set(start_sp[i]) | set(start_bs[i]))
                    for t in all_times:
                        if (i, t) in w and pulp.value(w[i, t]) > 0.5:
                            start = t
                            break
                if start is None:
                    start = start_bs[i][0] if start_bs[i] else int(arrival[i])

        if chosen == "brown":
            start = int(arrival[i])
            dur_steps = 0

        mode_counts[chosen] += 1
        service_starts[i] = float(start) * dt
        service_durs[i] = float(dur_steps) * dt
        start_steps[i] = int(start)
        duration_steps[i] = int(dur_steps)
        modes[i] = chosen

        energy = instance.energy_kwh[i]
        if chosen == "shore":
            cost_energy_val += instance.shore_cost * energy
        elif chosen == "battery":
            cost_energy_val += instance.battery_cost * energy
        else:
            cost_energy_val += instance.brown_cost * energy

        completion = max(arrival[i] + cargo[i], start + dur_steps)
        tardy = max(0.0, completion - deadlines[i])
        cost_delay_val += instance.delay_costs[i] * dt * tardy

    result = {
        "obj": float(obj),
        "runtime_total": float(runtime_total),
        "status": status_label,
        "time_limit": time_limit,
        "mechanism_counts": mode_counts,
        "infeasible_jobs": 0,
        "cost_energy": cost_energy_val,
        "cost_delay": cost_delay_val,
        "solver_backend": solver_name,
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

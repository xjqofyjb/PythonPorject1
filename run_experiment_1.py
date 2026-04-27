from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.instances import Instance, generate_instance
from src.metrics import compute_mode_ratios
from src.runner import build_method_cfg, expand_seeds, load_config
from src.solvers import cg_solver


TARGET_NS = [75, 125, 150]
TARGET_SCENARIO = "U"
TARGET_METHOD = "cg"
TARGET_OPERATION_MODES = ["simops", "sequential"]


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_stamp()}] {message}\n")


def summarize_type_mix(instance: Instance) -> dict[str, Any]:
    ship_types = np.asarray(instance.ship_types)
    unique, counts = np.unique(ship_types, return_counts=True)
    type_counts = {str(t): int(c) for t, c in zip(unique, counts)}
    type_ratios = {str(t): round(int(c) / instance.N, 4) for t, c in zip(unique, counts)}
    return {
        "ship_type_counts": type_counts,
        "ship_type_ratios": type_ratios,
        "arrival_time_range": [float(instance.arrival_times.min()), float(instance.arrival_times.max())],
        "cargo_time_range": [float(instance.cargo_times.min()), float(instance.cargo_times.max())],
        "deadline_range": [float(instance.deadlines.min()), float(instance.deadlines.max())],
        "energy_range": [float(instance.energy_kwh.min()), float(instance.energy_kwh.max())],
        "shore_compatible_ratio": float(np.mean(instance.shore_compatible)),
    }


def save_instance(instance: Instance, path: Path) -> None:
    payload = json_ready(asdict(instance))
    payload["audit"] = summarize_type_mix(instance)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def compute_departure_times(instance: Instance, start_times: np.ndarray, durations: np.ndarray) -> np.ndarray:
    cargo_finish = instance.arrival_times + instance.cargo_times
    return np.maximum(cargo_finish, start_times + durations)


def compute_shore_utilization(instance: Instance, modes: list[str], duration_steps: np.ndarray, start_steps: np.ndarray) -> float:
    if instance.shore_berths <= 0:
        return 0.0
    shore_steps = int(np.sum(duration_steps[np.asarray(modes) == "shore"]))
    completion_horizon = int(
        max(
            np.max(instance.deadline_steps),
            np.max(start_steps + duration_steps) if len(start_steps) else 0,
            np.max(instance.arrival_steps + instance.cargo_steps),
        )
    )
    completion_horizon = max(completion_horizon, 1)
    denom = instance.shore_berths * completion_horizon
    return float(shore_steps / denom) if denom > 0 else 0.0


def build_raw_payload(
    instance: Instance,
    operation_mode: str,
    result: dict[str, Any],
    trace_dir: Path,
) -> dict[str, Any]:
    schedule = result.get("schedule", {})
    start_times = np.asarray(schedule.get("service_start_times", []), dtype=float)
    durations = np.asarray(schedule.get("service_durations", []), dtype=float)
    modes = list(schedule.get("modes", []))
    start_steps = np.rint(start_times / instance.dt_hours).astype(int) if len(start_times) else np.array([], dtype=int)
    duration_steps = np.rint(durations / instance.dt_hours).astype(int) if len(durations) else np.array([], dtype=int)
    departures = compute_departure_times(instance, start_times, durations) if len(start_times) else np.array([], dtype=float)
    ratios = compute_mode_ratios(result.get("mechanism_counts", {}), instance.N)

    payload = {
        "experiment": "experiment_1_fill_points",
        "scenario": TARGET_SCENARIO,
        "mechanism": instance.mechanism,
        "method": TARGET_METHOD,
        "operation_mode": operation_mode,
        "N": instance.N,
        "seed": instance.seed,
        "objective": float(result.get("obj", np.nan)),
        "runtime": float(result.get("runtime_total", np.nan)),
        "runtime_pricing": float(result.get("runtime_pricing", np.nan)),
        "cg_iterations": int(result.get("num_iters", 0) or 0),
        "internal_gap": float(result.get("gap_pct", np.nan)),
        "lp_lower_bound": float(result.get("lp_lower_bound", np.nan)),
        "n_columns_generated": int(result.get("n_columns_generated", 0) or 0),
        "n_columns_added": int(result.get("num_columns_added", 0) or 0),
        "num_pricing_calls": int(result.get("num_pricing_calls", 0) or 0),
        "sp_share": float(ratios.get("shore_ratio", np.nan)),
        "bs_share": float(ratios.get("battery_ratio", np.nan)),
        "ae_share": float(ratios.get("brown_ratio", np.nan)),
        "sp_utilization": compute_shore_utilization(instance, modes, duration_steps, start_steps),
        "avg_stay_time": float(result.get("avg_stay_time", np.nan)),
        "masking_rate": float(result.get("avg_masking_rate", np.nan)),
        "delay_cost": float(result.get("cost_delay", np.nan)),
        "energy_cost": float(result.get("cost_energy", np.nan)),
        "success": bool(result.get("success", False)),
        "status": str(result.get("status", "")),
        "lp_status": str(result.get("lp_status", "")),
        "ip_status": str(result.get("ip_status", "")),
        "trace_file": str(trace_dir / f"exp1_N{instance.N}_seed{instance.seed}_sc{TARGET_SCENARIO}_op{operation_mode}_{TARGET_METHOD}.csv"),
        "mode_assignments": modes,
        "start_times": start_times.tolist(),
        "service_durations": durations.tolist(),
        "departure_times": departures.tolist(),
        "instance_audit": summarize_type_mix(instance),
    }
    return payload


def run_single_case(
    config_path: str,
    results_dir: str,
    N: int,
    seed: int,
    operation_mode: str,
    force: bool,
) -> dict[str, Any]:
    results_root = Path(results_dir)
    raw_dir = results_root / "raw"
    trace_dir = results_root / "traces"
    raw_path = raw_dir / f"N{N}_seed{seed}_{operation_mode}.json"
    if raw_path.exists() and not force:
        with raw_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    cfg = load_config(config_path)
    instance = generate_instance(
        N,
        seed,
        TARGET_SCENARIO,
        cfg.get("mechanism", "hybrid"),
        cfg.get("params", {}),
    )
    method_cfg = build_method_cfg(
        cfg,
        TARGET_METHOD,
        str(trace_dir),
        f"exp1_N{N}_seed{seed}_sc{TARGET_SCENARIO}_op{operation_mode}",
    )
    method_cfg["operation_mode"] = operation_mode
    method_cfg["return_schedule"] = True

    try:
        result = cg_solver.solve(instance, method_cfg, logger=None)
        payload = build_raw_payload(instance, operation_mode, result, trace_dir)
    except Exception as exc:
        payload = {
            "experiment": "experiment_1_fill_points",
            "scenario": TARGET_SCENARIO,
            "mechanism": instance.mechanism,
            "method": TARGET_METHOD,
            "operation_mode": operation_mode,
            "N": N,
            "seed": seed,
            "success": False,
            "status": "error",
            "error": str(exc),
        }

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)
    return payload


def scale_summary(rows: list[dict[str, Any]], N: int) -> dict[str, float]:
    sim = [row for row in rows if row.get("operation_mode") == "simops" and row.get("success")]
    seq = [row for row in rows if row.get("operation_mode") == "sequential" and row.get("success")]
    if len(sim) != len(seq) or not sim:
        raise RuntimeError(f"N={N} results are incomplete: simops={len(sim)}, sequential={len(seq)}")

    sim_obj = np.array([row["objective"] for row in sim], dtype=float)
    seq_obj = np.array([row["objective"] for row in seq], dtype=float)
    sim_gap = np.array([row["internal_gap"] for row in sim], dtype=float)
    sim_mask = np.array([row["masking_rate"] for row in sim], dtype=float)
    savings = (seq_obj.mean() - sim_obj.mean()) / seq_obj.mean() * 100.0
    return {
        "success_rate": sum(bool(row.get("success")) for row in rows) / len(rows),
        "simops_obj_mean": float(sim_obj.mean()),
        "sequential_obj_mean": float(seq_obj.mean()),
        "savings_pct": float(savings),
        "internal_gap_mean": float(np.nanmean(sim_gap)),
        "masking_rate_mean": float(np.nanmean(sim_mask)),
    }


def run_scale_batch(
    config_path: str,
    results_dir: Path,
    N: int,
    seeds: list[int],
    workers: int,
    force: bool,
    log_path: Path,
) -> list[dict[str, Any]]:
    append_log(log_path, f"Starting N={N} batch with seeds={seeds} and workers={workers}.")
    tasks = [(config_path, str(results_dir), N, seed, op_mode, force) for seed in seeds for op_mode in TARGET_OPERATION_MODES]

    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            rows.append(run_single_case(*task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single_case, *task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    
    return rows


def run_scale_batch_with_fallback(
    config_path: str,
    results_dir: Path,
    N: int,
    seeds: list[int],
    workers: int,
    force: bool,
    log_path: Path,
) -> list[dict[str, Any]]:
    try:
        rows = run_scale_batch(config_path, results_dir, N, seeds, workers, force, log_path)
    except PermissionError:
        append_log(log_path, f"Process-based parallelism failed for N={N}; falling back to ThreadPoolExecutor.")
        tasks = [(config_path, str(results_dir), N, seed, op_mode, force) for seed in seeds for op_mode in TARGET_OPERATION_MODES]
        rows = []
        if workers <= 1:
            for task in tasks:
                rows.append(run_single_case(*task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_single_case, *task) for task in tasks]
                for future in as_completed(futures):
                    rows.append(future.result())

    rows.sort(key=lambda item: (item["seed"], item["operation_mode"]))
    summary = scale_summary(rows, N)
    append_log(
        log_path,
        (
            f"N={N} completed. success_rate={summary['success_rate']:.2%}, "
            f"savings={summary['savings_pct']:.2f}%, "
            f"masking={summary['masking_rate_mean']:.3f}, "
            f"internal_gap={summary['internal_gap_mean']:.4f}%."
        ),
    )

    if summary["success_rate"] < 1.0:
        raise RuntimeError(f"N={N} did not finish successfully for all seeds.")
    if summary["internal_gap_mean"] > 0.05:
        raise RuntimeError(f"N={N} average internal gap {summary['internal_gap_mean']:.4f}% exceeds 0.05%.")
    if N == 75 and not (5.0 <= summary["savings_pct"] <= 25.0):
        raise RuntimeError(f"N=75 savings {summary['savings_pct']:.2f}% is outside the guardrail [5, 25].")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 1 fill-point CG+IR simulations.")
    parser.add_argument("--config", default="configs/simops.yaml", help="Base config path.")
    parser.add_argument("--results-dir", default="results/experiment_1_fill_points", help="Output directory.")
    parser.add_argument("--instance-dir", default="data/instances", help="Instance JSON directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers per scale.")
    parser.add_argument("--seeds", default=None, help="Optional seed override, e.g. 1:10.")
    parser.add_argument("--force", action="store_true", help="Recompute even if raw JSON already exists.")
    parser.add_argument("--Ns", default=None, help="Optional comma-separated subset, e.g. 75,125.")
    return parser.parse_args()


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    cfg = load_config(args.config)

    results_dir = Path(args.results_dir)
    instance_dir = Path(args.instance_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "raw").mkdir(parents=True, exist_ok=True)
    (results_dir / "traces").mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "run_log.txt"
    if args.force and log_path.exists():
        log_path.unlink()

    backup_dirs = sorted(path.name for path in Path("results").glob("backup*") if path.is_dir())
    append_log(log_path, "Experiment 1 execution started.")
    append_log(log_path, f"Using config={args.config}. Existing result backups={backup_dirs}.")

    seeds = expand_seeds(args.seeds if args.seeds is not None else cfg.get("seeds", 10))
    target_ns = TARGET_NS if not args.Ns else [int(token.strip()) for token in args.Ns.split(",") if token.strip()]
    append_log(log_path, f"Resolved seeds={seeds}; target scales={target_ns}; scenario={TARGET_SCENARIO}.")
    append_log(log_path, "Current codebase uses seeds 1..10 for the SIMOPS experiment; this run keeps that seed scheme.")

    for N in target_ns:
        for seed in seeds:
            instance = generate_instance(N, seed, TARGET_SCENARIO, cfg.get("mechanism", "hybrid"), cfg.get("params", {}))
            instance_path = instance_dir / f"N{N}_seed{seed}.json"
            if args.force or not instance_path.exists():
                save_instance(instance, instance_path)
        append_log(log_path, f"Saved/verified {len(seeds)} instances for N={N} under {instance_dir}.")

    all_rows: list[dict[str, Any]] = []
    workers = max(1, min(args.workers, len(seeds) * len(TARGET_OPERATION_MODES)))
    for N in target_ns:
        batch_rows = run_scale_batch_with_fallback(args.config, results_dir, N, seeds, workers, args.force, log_path)
        all_rows.extend(batch_rows)

    append_log(log_path, f"Experiment 1 finished successfully with {len(all_rows)} raw result files.")


if __name__ == "__main__":
    main()

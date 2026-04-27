from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.instances import Instance, generate_instance
from src.metrics import compute_mode_ratios
from src.runner import build_method_cfg, expand_seeds, load_config
from src.solvers import cg_solver


PERTURBATION_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]
LOOSE_SLACK_RANGE = (2.0, 6.0)
TIGHT_SLACK_CANDIDATES = [(0.5, 1.5), (0.75, 1.5)]
N_VALUE = 100
SCENARIO = "U"
METHOD = "cg"
OPERATION_MODE = "simops"


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_stamp()}] {message}\n")


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def summarize_type_mix(instance: Instance) -> dict[str, Any]:
    ship_types = np.asarray(instance.ship_types)
    unique, counts = np.unique(ship_types, return_counts=True)
    return {
        "ship_type_counts": {str(t): int(c) for t, c in zip(unique, counts)},
        "ship_type_ratios": {str(t): round(int(c) / instance.N, 4) for t, c in zip(unique, counts)},
        "arrival_time_range": [float(instance.arrival_times.min()), float(instance.arrival_times.max())],
        "cargo_time_range": [float(instance.cargo_times.min()), float(instance.cargo_times.max())],
        "deadline_range": [float(instance.deadlines.min()), float(instance.deadlines.max())],
    }


def build_instance_with_slack(
    base_instance: Instance,
    slack_range: tuple[float, float],
    *,
    slack_seed_offset: int = 200_000,
) -> Instance:
    low, high = slack_range
    rng = np.random.default_rng(int(base_instance.seed) + slack_seed_offset)
    slack = rng.uniform(low, high, size=base_instance.N)
    deadlines = base_instance.arrival_times + base_instance.cargo_times + slack
    deadline_steps = np.ceil(deadlines / float(base_instance.dt_hours)).astype(int)
    params = dict(base_instance.params)
    params["slack_range_hours"] = [float(low), float(high)]
    params["slack_seed_offset"] = int(slack_seed_offset)
    return replace(
        base_instance,
        deadlines=deadlines,
        deadline_steps=deadline_steps,
        params=params,
    )


def perturb_instance_fixed_deadline(instance: Instance, delta_hours: float) -> Instance:
    rng = np.random.default_rng(int(instance.seed) + 1000)
    perturb = rng.uniform(-delta_hours, delta_hours, size=instance.N)
    arrival_times = np.maximum(0.0, instance.arrival_times + perturb)
    arrival_steps = np.ceil(arrival_times / float(instance.dt_hours)).astype(int)
    return replace(
        instance,
        arrival_times=arrival_times,
        arrival_steps=arrival_steps,
    )


def save_instance_file(instance: Instance, path: Path, config_name: str, slack_range: tuple[float, float]) -> None:
    payload = json_ready(asdict(instance))
    payload["experiment_2_metadata"] = {
        "config_name": config_name,
        "slack_range_hours": list(slack_range),
        "audit": summarize_type_mix(instance),
    }
    save_json(path, payload)


def make_method_cfg(config: dict[str, Any], trace_dir: Path, instance_id: str) -> dict[str, Any]:
    cfg = build_method_cfg(config, METHOD, str(trace_dir), instance_id)
    cfg["operation_mode"] = OPERATION_MODE
    cfg["return_schedule"] = True
    return cfg


def extract_schedule_arrays(result: dict[str, Any], instance: Instance) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    schedule = result.get("schedule", {})
    start_times = np.asarray(schedule.get("service_start_times", []), dtype=float)
    durations = np.asarray(schedule.get("service_durations", []), dtype=float)
    modes = list(schedule.get("modes", []))
    if len(start_times) != instance.N or len(durations) != instance.N or len(modes) != instance.N:
        start_times = np.full(instance.N, np.nan)
        durations = np.zeros(instance.N, dtype=float)
        modes = ["unknown"] * instance.N
    start_steps = np.rint(start_times / float(instance.dt_hours)).astype(int)
    duration_steps = np.rint(durations / float(instance.dt_hours)).astype(int)
    return start_times, start_steps, duration_steps, modes


def compute_departures(instance: Instance, start_steps: np.ndarray, duration_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    completion_steps = np.maximum(instance.arrival_steps + instance.cargo_steps, start_steps + duration_steps)
    tardy_steps = np.maximum(0, completion_steps - instance.deadline_steps)
    return completion_steps, tardy_steps


def compute_shore_utilization(instance: Instance, modes: list[str], duration_steps: np.ndarray, start_steps: np.ndarray) -> float:
    if instance.shore_berths <= 0:
        return 0.0
    shore_steps = int(np.sum(duration_steps[np.asarray(modes) == "shore"]))
    completion_horizon = int(
        max(
            np.max(instance.deadline_steps),
            np.max(instance.arrival_steps + instance.cargo_steps),
            np.max(start_steps + duration_steps) if len(start_steps) else 0,
        )
    )
    completion_horizon = max(completion_horizon, 1)
    return float(shore_steps / (instance.shore_berths * completion_horizon))


def build_result_payload(
    *,
    config_name: str,
    slack_range: tuple[float, float],
    delta: float,
    instance: Instance,
    result: dict[str, Any],
    baseline_reference: dict[str, Any] | None,
    trace_dir: Path,
) -> dict[str, Any]:
    start_times, start_steps, duration_steps, modes = extract_schedule_arrays(result, instance)
    completion_steps, tardy_steps = compute_departures(instance, start_steps, duration_steps)
    departures = completion_steps * float(instance.dt_hours)
    shares = compute_mode_ratios(result.get("mechanism_counts", {}), instance.N)
    delayed = tardy_steps > 0
    feasibility_margin = float(np.min((instance.deadline_steps - completion_steps) * float(instance.dt_hours)))

    mode_switches = 0
    start_shifts = 0
    if baseline_reference is not None:
        baseline_modes = baseline_reference["mode_assignments"]
        baseline_starts = np.asarray(baseline_reference["start_times"], dtype=float)
        mode_switches = int(np.sum(np.asarray(modes) != np.asarray(baseline_modes)))
        start_shifts = int(np.sum(np.abs(start_times - baseline_starts) > 1e-9))

    payload = {
        "config_name": config_name,
        "slack_low": float(slack_range[0]),
        "slack_high": float(slack_range[1]),
        "slack_mean": float((slack_range[0] + slack_range[1]) / 2.0),
        "N": instance.N,
        "scenario": SCENARIO,
        "method": METHOD,
        "operation_mode": OPERATION_MODE,
        "seed": int(instance.seed),
        "delta": float(delta),
        "objective": float(result.get("obj", np.nan)),
        "runtime": float(result.get("runtime_total", np.nan)),
        "cg_iterations": int(result.get("num_iters", 0) or 0),
        "internal_gap": float(result.get("gap_pct", np.nan)),
        "lp_lower_bound": float(result.get("lp_lower_bound", np.nan)),
        "sp_share": float(shares.get("shore_ratio", np.nan)),
        "bs_share": float(shares.get("battery_ratio", np.nan)),
        "ae_share": float(shares.get("brown_ratio", np.nan)),
        "sp_utilization": compute_shore_utilization(instance, modes, duration_steps, start_steps),
        "avg_stay_time": float(result.get("avg_stay_time", np.nan)),
        "masking_rate": float(result.get("avg_masking_rate", np.nan)),
        "delay_cost": float(result.get("cost_delay", np.nan)),
        "delay_cost_pct": float(result.get("cost_delay", 0.0) / max(float(result.get("obj", np.nan)), 1e-9) * 100.0),
        "energy_cost": float(result.get("cost_energy", np.nan)),
        "n_delayed_vessels": int(np.sum(delayed)),
        "max_delay": float(np.max(tardy_steps) * float(instance.dt_hours)) if delayed.any() else 0.0,
        "avg_delay_of_delayed": float(np.mean(tardy_steps[delayed]) * float(instance.dt_hours)) if delayed.any() else 0.0,
        "n_mode_switches_vs_baseline": mode_switches,
        "n_start_time_shifts": start_shifts,
        "feasibility_margin": feasibility_margin,
        "success": bool(result.get("success", False)),
        "status": str(result.get("status", "")),
        "lp_status": str(result.get("lp_status", "")),
        "ip_status": str(result.get("ip_status", "")),
        "trace_file": str(trace_dir / f"exp2_{config_name}_N100_seed{instance.seed}_delta{delta:.1f}_{METHOD}.csv"),
        "mode_assignments": modes,
        "start_times": start_times.tolist(),
        "departure_times": departures.tolist(),
        "arrival_times": instance.arrival_times.tolist(),
        "deadlines": instance.deadlines.tolist(),
    }
    return payload


def run_single_seed_config(
    config_path: str,
    config_name: str,
    slack_range: tuple[float, float],
    seed: int,
    results_dir: str,
    save_instance_dir: str | None,
    force: bool,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    results_root = Path(results_dir)
    raw_dir = results_root / "raw"
    trace_dir = results_root / "traces"
    base = generate_instance(N_VALUE, int(seed), SCENARIO, config.get("mechanism", "hybrid"), dict(config.get("params", {})))
    if config_name == "tight":
        instance = build_instance_with_slack(base, slack_range)
        if save_instance_dir is not None:
            save_instance_file(instance, Path(save_instance_dir) / f"N100_tight_seed{seed}.json", config_name, slack_range)
    else:
        instance = base

    outputs: list[dict[str, Any]] = []
    baseline_payload: dict[str, Any] | None = None

    for delta in PERTURBATION_LEVELS:
        raw_name = f"N100_seed{seed}_delta{delta:.1f}_{config_name}.json"
        raw_path = raw_dir / raw_name
        if raw_path.exists() and not force:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            outputs.append(payload)
            if delta == 0.0:
                baseline_payload = payload
            continue

        perturbed = perturb_instance_fixed_deadline(instance, float(delta))
        method_cfg = make_method_cfg(config, trace_dir, f"exp2_{config_name}_N100_seed{seed}_delta{delta:.1f}")
        result = cg_solver.solve(perturbed, method_cfg, logger=None)

        if delta == 0.0:
            payload = build_result_payload(
                config_name=config_name,
                slack_range=slack_range,
                delta=float(delta),
                instance=perturbed,
                result=result,
                baseline_reference=None,
                trace_dir=trace_dir,
            )
            baseline_payload = payload
        else:
            payload = build_result_payload(
                config_name=config_name,
                slack_range=slack_range,
                delta=float(delta),
                instance=perturbed,
                result=result,
                baseline_reference=baseline_payload,
                trace_dir=trace_dir,
            )

        save_json(raw_path, payload)
        outputs.append(payload)

    outputs.sort(key=lambda row: row["delta"])
    return outputs


def parallel_map_seed_configs(
    tasks: list[tuple[str, str, tuple[float, float], int, str, str | None, bool]],
    workers: int,
) -> list[list[dict[str, Any]]]:
    if workers <= 1:
        return [run_single_seed_config(*task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single_seed_config, *task) for task in tasks]
            return [future.result() for future in as_completed(futures)]
    except PermissionError:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single_seed_config, *task) for task in tasks]
            return [future.result() for future in as_completed(futures)]


def run_precheck(
    config_path: str,
    slack_range: tuple[float, float],
    seeds: list[int],
    results_dir: Path,
    workers: int,
    force: bool,
    log_path: Path,
) -> pd.DataFrame:
    append_log(log_path, f"Running precheck for slack range {slack_range}.")
    tasks = [(config_path, "tight", slack_range, seed, str(results_dir), None, force) for seed in seeds]
    nested_rows = parallel_map_seed_configs(tasks, workers)
    rows = []
    for batch in nested_rows:
        baseline = next(row for row in batch if float(row["delta"]) == 0.0)
        flagged_ae = baseline["ae_share"] > 0.30
        flagged_delay = baseline["delay_cost_pct"] > 15.0
        flagged_gap = baseline["internal_gap"] > 0.5
        rows.append(
            {
                "seed": baseline["seed"],
                "slack_low": slack_range[0],
                "slack_high": slack_range[1],
                "objective": baseline["objective"],
                "ae_share": baseline["ae_share"],
                "delay_cost_pct": baseline["delay_cost_pct"],
                "internal_gap": baseline["internal_gap"],
                "flag_ae": flagged_ae,
                "flag_delay": flagged_delay,
                "flag_gap": flagged_gap,
                "flagged": bool(flagged_ae or flagged_delay or flagged_gap),
            }
        )
    precheck = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    precheck_dir = results_dir / "feasibility_check"
    precheck_dir.mkdir(parents=True, exist_ok=True)
    precheck.to_csv(precheck_dir / "precheck_results.csv", index=False)
    append_log(log_path, f"Precheck for {slack_range} flagged {int(precheck['flagged'].sum())} / {len(precheck)} seeds.")
    return precheck


def write_slack_decision(
    results_dir: Path,
    initial_range: tuple[float, float],
    final_range: tuple[float, float],
    initial_precheck: pd.DataFrame,
    final_precheck: pd.DataFrame,
) -> None:
    lines = [
        "# Final Slack Decision",
        "",
        f"- Initial slack range tested: U[{initial_range[0]}, {initial_range[1]}]",
        f"- Final slack range adopted: U[{final_range[0]}, {final_range[1]}]",
        f"- Initial flagged seeds: {int(initial_precheck['flagged'].sum())} / {len(initial_precheck)}",
        f"- Final flagged seeds: {int(final_precheck['flagged'].sum())} / {len(final_precheck)}",
        "",
        "## Final Baseline Statistics",
        "",
        f"- Mean AE share: {final_precheck['ae_share'].mean() * 100:.2f}%",
        f"- Mean delay-cost share: {final_precheck['delay_cost_pct'].mean():.2f}%",
        f"- Mean internal gap: {final_precheck['internal_gap'].mean():.4f}%",
    ]
    text = "\n".join(lines) + "\n"
    (results_dir / "feasibility_check").mkdir(parents=True, exist_ok=True)
    (results_dir / "feasibility_check" / "final_slack_decision.md").write_text(text, encoding="utf-8")
    (results_dir / "final_slack_decision.md").write_text(text, encoding="utf-8")


def choose_final_slack(
    config_path: str,
    seeds: list[int],
    results_dir: Path,
    workers: int,
    force: bool,
    log_path: Path,
) -> tuple[tuple[float, float], pd.DataFrame]:
    first = run_precheck(config_path, TIGHT_SLACK_CANDIDATES[0], seeds, results_dir, workers, force, log_path)
    if int(first["flagged"].sum()) < 3:
        write_slack_decision(results_dir, TIGHT_SLACK_CANDIDATES[0], TIGHT_SLACK_CANDIDATES[0], first, first)
        return TIGHT_SLACK_CANDIDATES[0], first

    append_log(log_path, "Initial tight slack precheck flagged >=3 seeds; retrying with relaxed lower bound 0.75.")
    second = run_precheck(config_path, TIGHT_SLACK_CANDIDATES[1], seeds, results_dir, workers, force, log_path)
    write_slack_decision(results_dir, TIGHT_SLACK_CANDIDATES[0], TIGHT_SLACK_CANDIDATES[1], first, second)
    if int(second["flagged"].sum()) >= 3:
        raise RuntimeError("Even slack U[0.75, 1.5] still flags >=3 seeds in precheck. Manual intervention required.")
    return TIGHT_SLACK_CANDIDATES[1], second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 2 tight-deadline perturbation study.")
    parser.add_argument("--config", default="configs/simops.yaml", help="Base config path.")
    parser.add_argument("--results-dir", default="results/experiment_2_tight_deadline", help="Results directory.")
    parser.add_argument("--instance-dir", default="data/instances", help="Instance directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers.")
    parser.add_argument("--seeds", default=None, help="Seed override, e.g. 1:10.")
    parser.add_argument("--force", action="store_true", help="Recompute raw JSON even if present.")
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
    (results_dir / "feasibility_check").mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"
    if args.force and log_path.exists():
        log_path.unlink()

    seeds = expand_seeds(args.seeds if args.seeds is not None else cfg.get("seeds", 10))
    workers = max(1, min(args.workers, len(seeds)))
    append_log(log_path, "Experiment 2 execution started.")
    append_log(log_path, f"Resolved seeds={seeds}; scenario={SCENARIO}; N={N_VALUE}.")

    final_slack_range, final_precheck = choose_final_slack(args.config, seeds, results_dir, workers, args.force, log_path)
    append_log(log_path, f"Final tight slack range selected: U[{final_slack_range[0]}, {final_slack_range[1]}].")
    append_log(
        log_path,
        (
            f"Precheck stats: AE mean={final_precheck['ae_share'].mean() * 100:.2f}%, "
            f"delay share mean={final_precheck['delay_cost_pct'].mean():.2f}%, "
            f"gap mean={final_precheck['internal_gap'].mean():.4f}%."
        ),
    )

    all_rows: list[dict[str, Any]] = []
    for config_name, slack_range in [("loose", LOOSE_SLACK_RANGE), ("tight", final_slack_range)]:
        append_log(log_path, f"Running full perturbation batch for {config_name} slack {slack_range}.")
        tasks = [
            (
                args.config,
                config_name,
                slack_range,
                seed,
                str(results_dir),
                str(instance_dir) if config_name == "tight" else None,
                args.force,
            )
            for seed in seeds
        ]
        nested_rows = parallel_map_seed_configs(tasks, workers)
        for batch in nested_rows:
            all_rows.extend(batch)

    all_df = pd.DataFrame(all_rows)
    all_df = all_df.sort_values(["config_name", "seed", "delta"]).reset_index(drop=True)
    all_df.to_csv(results_dir / "raw_results_index.csv", index=False)

    tight_delta_1 = all_df[(all_df["config_name"] == "tight") & (all_df["delta"] == 1.0)]
    delay_pct_mean = float(tight_delta_1["delay_cost_pct"].mean())
    ae_share_mean = float(tight_delta_1["ae_share"].mean() * 100.0)
    append_log(
        log_path,
        f"Checkpoint Δ=1.0h (tight): delay_cost_pct_mean={delay_pct_mean:.2f}%, AE_share_mean={ae_share_mean:.2f}%.",
    )
    if delay_pct_mean < 1.0:
        raise RuntimeError("Tight-deadline Δ=1.0h delay-cost share stayed below 1%; slack is still too loose.")
    if delay_pct_mean > 20.0:
        raise RuntimeError("Tight-deadline Δ=1.0h delay-cost share exceeded 20%; slack is too tight.")
    if ae_share_mean > 20.0:
        raise RuntimeError("Tight-deadline Δ=1.0h AE share exceeded 20%; structure is already unstable.")

    append_log(log_path, f"Experiment 2 finished successfully with {len(all_df)} raw result rows.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from column_enrichment import capture_baseline_pool, solve_enriched_from_baseline
from src.instances import generate_instance
from src.runner import build_method_cfg, expand_seeds, load_config


EPSILON_LEVELS = [0.005, 0.01, 0.02]
TARGET_NS = [20, 50, 100]
TARGET_SCENARIO = "U"
TARGET_METHOD = "cg"
TARGET_OPERATION_MODE = "simops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 column enrichment study.")
    parser.add_argument("--config", default="configs/main.yaml", help="Baseline config path.")
    parser.add_argument("--results-dir", default="results/experiment_3_column_enrichment", help="Output directory.")
    parser.add_argument("--workers", type=int, default=max(1, min(6, (multiprocessing.cpu_count() or 1))), help="Parallel workers.")
    parser.add_argument("--force", action="store_true", help="Recompute even if raw JSON exists.")
    return parser.parse_args()


def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_stamp()}] {message}\n")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
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


def epsilon_label(epsilon_ratio: float) -> str:
    return f"{epsilon_ratio * 100:.1f}pct"


def build_exact_milp_lookup() -> dict[tuple[int, int], float]:
    results_path = Path("results/results_main_rigorous.csv")
    table = pd.read_csv(results_path)
    subset = table[
        (table["scenario"] == TARGET_SCENARIO)
        & (table["mechanism"] == "hybrid")
        & (table["N"].isin([20, 50]))
        & (table["status"] == "ok")
        & (table["method"] == "milp300")
    ].copy()
    lookup: dict[tuple[int, int], float] = {}
    for _, row in subset.iterrows():
        lookup[(int(row["N"]), int(row["seed"]))] = float(row["obj"])
    return lookup


def build_method_config(config: dict[str, Any], instance_id: str) -> dict[str, Any]:
    cfg = build_method_cfg(config, TARGET_METHOD, trace_dir="", instance_id=instance_id)
    cfg["operation_mode"] = TARGET_OPERATION_MODE
    return cfg


def baseline_payload(
    *,
    N: int,
    seed: int,
    baseline: dict[str, Any],
    exact_milp: float | None,
) -> dict[str, Any]:
    payload = {
        "record_type": "baseline",
        "N": int(N),
        "seed": int(seed),
        "scenario": TARGET_SCENARIO,
        "method": TARGET_METHOD,
        "operation_mode": TARGET_OPERATION_MODE,
        "baseline_pool_size": int(baseline["baseline_pool_size"]),
        "full_column_pool_size": int(baseline["full_column_pool_size"]),
        "baseline_pool_complete": bool(baseline["baseline_pool_complete"]),
        "used_full_pool_small": bool(baseline["used_full_pool_small"]),
        "baseline_pool_mode_counts": baseline["baseline_pool_mode_counts"],
        "Z_baseline_IRMP": float(baseline["objective"]),
        "baseline_internal_gap": float(baseline["internal_gap_pct"]),
        "lp_lower_bound": float(baseline["lp_lower_bound"]),
        "baseline_lp_status": str(baseline["lp_status"]),
        "baseline_ip_status": str(baseline["ip_status"]),
        "baseline_success": bool(baseline["success"]),
        "num_iters": int(baseline["num_iters"]),
        "pricing_calls": int(baseline["pricing_calls"]),
        "num_columns_added_during_cg": int(baseline["num_columns_added_during_cg"]),
        "min_reduced_cost_last": float(baseline["min_reduced_cost_last"]),
        "mode_distribution_baseline": baseline["mode_distribution"],
        "cost_energy": float(baseline["cost_energy"]),
        "cost_delay": float(baseline["cost_delay"]),
        "Z_MILP_exact": float(exact_milp) if exact_milp is not None else np.nan,
        "gap_baseline_vs_MILP": (
            (float(baseline["objective"]) - float(exact_milp)) / float(exact_milp) * 100.0 if exact_milp is not None else np.nan
        ),
    }
    return payload


def enriched_payload(
    *,
    N: int,
    seed: int,
    baseline: dict[str, Any],
    enriched: dict[str, Any],
    exact_milp: float | None,
) -> dict[str, Any]:
    return {
        "record_type": "enriched",
        "N": int(N),
        "seed": int(seed),
        "scenario": TARGET_SCENARIO,
        "method": TARGET_METHOD,
        "operation_mode": TARGET_OPERATION_MODE,
        "epsilon_ratio": float(enriched["epsilon_ratio"]),
        "epsilon_pct": float(enriched["epsilon_ratio"] * 100.0),
        "epsilon_label": epsilon_label(float(enriched["epsilon_ratio"])),
        "epsilon_abs": float(enriched["epsilon_abs"]),
        "baseline_pool_size": int(baseline["baseline_pool_size"]),
        "full_column_pool_size": int(baseline["full_column_pool_size"]),
        "baseline_pool_complete": bool(baseline["baseline_pool_complete"]),
        "used_full_pool_small": bool(baseline["used_full_pool_small"]),
        "enriched_pool_size": int(enriched["enriched_pool_size"]),
        "columns_added": int(enriched["columns_added"]),
        "columns_by_mode_added": enriched["columns_by_mode_added"],
        "Z_baseline_IRMP": float(baseline["objective"]),
        "Z_enriched_IRMP": float(enriched["objective"]),
        "improvement_pct": float(enriched["improvement_pct"]),
        "baseline_internal_gap": float(baseline["internal_gap_pct"]),
        "enriched_internal_gap": float(enriched["internal_gap_pct"]),
        "lp_lower_bound": float(enriched["lp_lower_bound"]),
        "solutions_identical": bool(enriched["solutions_identical"]),
        "n_vessels_changed": int(enriched["n_vessels_changed"]),
        "changed_vessels": enriched["changed_vessels"],
        "mode_distribution_baseline": baseline["mode_distribution"],
        "mode_distribution_enriched": enriched["mode_distribution"],
        "cost_energy_baseline": float(baseline["cost_energy"]),
        "cost_delay_baseline": float(baseline["cost_delay"]),
        "cost_energy_enriched": float(enriched["cost_energy"]),
        "cost_delay_enriched": float(enriched["cost_delay"]),
        "baseline_lp_status": str(baseline["lp_status"]),
        "baseline_ip_status": str(baseline["ip_status"]),
        "enriched_lp_status": str(enriched["lp_status"]),
        "enriched_ip_status": str(enriched["ip_status"]),
        "baseline_success": bool(baseline["success"]),
        "enriched_success": bool(enriched["success"]),
        "Z_MILP_exact": float(exact_milp) if exact_milp is not None else np.nan,
        "gap_baseline_vs_MILP": (
            (float(baseline["objective"]) - float(exact_milp)) / float(exact_milp) * 100.0 if exact_milp is not None else np.nan
        ),
        "gap_enriched_vs_MILP": (
            (float(enriched["objective"]) - float(exact_milp)) / float(exact_milp) * 100.0 if exact_milp is not None else np.nan
        ),
    }


def run_single_instance(
    config_path: str,
    N: int,
    seed: int,
    results_dir: str,
    force: bool,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    exact_lookup = build_exact_milp_lookup()
    exact_milp = exact_lookup.get((int(N), int(seed)))
    mechanism = config.get("mechanism", "hybrid")
    params = dict(config.get("params", {}))
    instance = generate_instance(int(N), int(seed), TARGET_SCENARIO, mechanism, params)
    cfg = build_method_config(config, instance_id=f"exp3_N{N}_seed{seed}")

    root = Path(results_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = raw_dir / f"N{N}_seed{seed}_baseline.json"
    enriched_paths = {ratio: raw_dir / f"N{N}_seed{seed}_enriched_{epsilon_label(ratio)}.json" for ratio in EPSILON_LEVELS}
    if not force and baseline_path.exists() and all(path.exists() for path in enriched_paths.values()):
        rows = [json.loads(baseline_path.read_text(encoding="utf-8"))]
        rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in enriched_paths.values())
        return rows

    rows: list[dict[str, Any]] = []
    baseline = capture_baseline_pool(instance, cfg)
    baseline_row = baseline_payload(N=N, seed=seed, baseline=baseline, exact_milp=exact_milp)
    save_json(baseline_path, baseline_row)
    rows.append(baseline_row)
    for ratio in EPSILON_LEVELS:
        enriched = solve_enriched_from_baseline(instance, baseline, float(ratio))
        enriched_row = enriched_payload(N=N, seed=seed, baseline=baseline, enriched=enriched, exact_milp=exact_milp)
        save_json(enriched_paths[ratio], enriched_row)
        rows.append(enriched_row)
    return rows


def parallel_run(
    tasks: list[tuple[str, int, int, str, bool]],
    workers: int,
) -> list[list[dict[str, Any]]]:
    if workers <= 1:
        return [run_single_instance(*task) for task in tasks]
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single_instance, *task) for task in tasks]
            return [future.result() for future in as_completed(futures)]
    except PermissionError:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_single_instance, *task) for task in tasks]
            return [future.result() for future in as_completed(futures)]


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log.txt"

    config = load_config(args.config)
    seeds = expand_seeds(config.get("seeds", 10))
    tasks = [(args.config, int(N), int(seed), str(results_dir), bool(args.force)) for N in TARGET_NS for seed in seeds]

    append_log(log_path, "Step 1 complete: loaded config, matched target N in {20, 50, 100}, and located exact MILP references for N=20/50.")
    append_log(log_path, "Step 2 start: running baseline CG/IRMP snapshots and enriched-pool re-solves.")
    batches = parallel_run(tasks, workers=int(args.workers))

    rows = [row for batch in batches for row in batch]
    baseline_rows = [row for row in rows if row["record_type"] == "baseline"]
    enriched_rows = [row for row in rows if row["record_type"] == "enriched"]

    raw_index = pd.DataFrame(rows).sort_values(["record_type", "N", "seed", "epsilon_pct"], na_position="first").reset_index(drop=True)
    raw_index.to_csv(results_dir / "raw_results_index.csv", index=False)
    append_log(log_path, f"Step 2 complete: wrote {len(baseline_rows)} baseline records and {len(enriched_rows)} enriched records.")

    zero_added = sum(int(row["columns_added"]) == 0 for row in enriched_rows)
    all_full_pool = all(bool(row["baseline_pool_complete"]) for row in baseline_rows)
    append_log(
        log_path,
        f"Diagnostic: {zero_added}/{len(enriched_rows)} enriched runs added zero columns; baseline_pool_complete_all={all_full_pool}.",
    )

    any_large_improvement = any(float(row["improvement_pct"]) > 1.0 for row in enriched_rows)
    any_very_large = any(float(row["improvement_pct"]) > 5.0 for row in enriched_rows)
    any_beats_milp = any(
        pd.notna(row["Z_MILP_exact"]) and float(row["Z_enriched_IRMP"]) + 1e-6 < float(row["Z_MILP_exact"])
        for row in enriched_rows
    )
    if any_very_large or any_beats_milp:
        append_log(log_path, "Alert: delta-like signal detected. Review required before any paper claim is updated.")
    elif any_large_improvement:
        append_log(log_path, "Alert: gamma-like signal detected. Continue with analysis, but do not auto-strengthen claims.")
    else:
        append_log(log_path, "No gamma/delta signal detected during run stage.")

    append_log(log_path, "Step 3 pending: run analyze_experiment_3.py to aggregate figures, tables, and summary.")


if __name__ == "__main__":
    main()

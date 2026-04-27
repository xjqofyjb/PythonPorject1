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
from diagnostic_precheck import run_precheck
from src.instances import generate_instance
from src.runner import build_method_cfg, expand_seeds, load_config


EPSILON_LEVELS = [0.005, 0.01, 0.02]
TARGET_NS = [100, 200, 500]
OBJECTIVE_TOL = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 follow-up on iterative top-K pricing scales.")
    parser.add_argument("--config", default="configs/main.yaml", help="Main experiment config.")
    parser.add_argument("--results-dir", default="results/experiment_3_followup", help="Output directory.")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (multiprocessing.cpu_count() or 1))), help="Parallel workers.")
    parser.add_argument("--force", action="store_true", help="Recompute raw files even if they exist.")
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


def pricing_mode(N: int) -> str:
    return "top-K diag" if N == 100 else "top-K"


def raw_prefix(N: int, seed: int) -> str:
    return f"N100_diag_seed{seed}" if N == 100 else f"N{N}_seed{seed}"


def build_method_config(config: dict[str, Any], N: int, instance_id: str) -> dict[str, Any]:
    cfg = build_method_cfg(config, "cg", trace_dir="", instance_id=instance_id)
    cfg["operation_mode"] = "simops"
    cfg.setdefault("cg", {})
    if N == 100:
        cfg["cg"]["use_full_pool_small"] = False
    return cfg


def plan_key(col: dict[str, Any]) -> tuple[int, str, int, int]:
    return (int(col["ship"]), str(col["mode"]), int(col["start"]), int(col["berth"]))


def get_baseline_pool_keys(baseline: dict[str, Any]) -> set[tuple[int, str, int, int]]:
    columns_all = baseline["columns_all"]
    return {plan_key(columns_all[col_id]) for col_id in baseline["active_column_ids"]}


def build_baseline_payload(N: int, seed: int, baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_type": "baseline",
        "N": int(N),
        "seed": int(seed),
        "scenario": "U",
        "method": "cg",
        "pricing_mode": pricing_mode(N),
        "operation_mode": "simops",
        "baseline_pool_size": int(baseline["baseline_pool_size"]),
        "full_column_pool_size": int(baseline["full_column_pool_size"]),
        "baseline_pool_complete": bool(baseline["baseline_pool_complete"]),
        "used_full_pool_small": bool(baseline["used_full_pool_small"]),
        "num_iters": int(baseline["num_iters"]),
        "pricing_calls": int(baseline["pricing_calls"]),
        "num_columns_added_during_cg": int(baseline["num_columns_added_during_cg"]),
        "objective": float(baseline["objective"]),
        "internal_gap_pct": float(baseline["internal_gap_pct"]),
        "mode_distribution": baseline["mode_distribution"],
        "plan_keys": baseline["plan_keys"],
        "mode_assignments": baseline["mode_assignments"],
        "start_steps": baseline["start_steps"],
        "berths": baseline["berths"],
    }


def build_enriched_payload(
    N: int,
    seed: int,
    baseline: dict[str, Any],
    enriched: dict[str, Any],
    epsilon_ratio: float,
    rerun_triggered: bool,
) -> dict[str, Any]:
    baseline_plan_keys = baseline["plan_keys"]
    enriched_plan_keys = enriched["plan_keys"]
    n_plans_changed = sum(1 for left, right in zip(baseline_plan_keys, enriched_plan_keys) if left != right)
    n_mode_switches = sum(1 for left, right in zip(baseline["mode_assignments"], enriched["mode_assignments"]) if left != right)
    objective_diff = float(baseline["objective"]) - float(enriched["objective"])

    if abs(objective_diff) <= OBJECTIVE_TOL:
        objective_equivalence_type = "identical" if n_plans_changed == 0 else "alternative_optima"
    elif objective_diff > OBJECTIVE_TOL:
        objective_equivalence_type = "improved"
    else:
        raise RuntimeError(
            f"Unexpected enriched objective worse than baseline for N={N}, seed={seed}, eps={epsilon_ratio}: diff={objective_diff}"
        )

    baseline_pool_keys = get_baseline_pool_keys(baseline)
    columns_used_from_enrichment = sum(1 for key in enriched_plan_keys if tuple(key) not in baseline_pool_keys)

    return {
        "record_type": "enriched",
        "N": int(N),
        "seed": int(seed),
        "scenario": "U",
        "method": "cg",
        "pricing_mode": pricing_mode(N),
        "operation_mode": "simops",
        "epsilon_ratio": float(epsilon_ratio),
        "epsilon_pct": float(epsilon_ratio * 100.0),
        "epsilon_label": epsilon_label(epsilon_ratio),
        "epsilon_abs": float(enriched["epsilon_abs"]),
        "baseline_pool_size": int(baseline["baseline_pool_size"]),
        "full_column_pool_size": int(baseline["full_column_pool_size"]),
        "enriched_pool_size": int(enriched["enriched_pool_size"]),
        "columns_added": int(enriched["columns_added"]),
        "columns_by_mode_added": enriched["columns_by_mode_added"],
        "Z_baseline_IRMP": float(baseline["objective"]),
        "Z_enriched_IRMP": float(enriched["objective"]),
        "objective_diff": float(objective_diff),
        "improvement_pct": max(0.0, float(objective_diff) / max(float(baseline["objective"]), 1e-9) * 100.0),
        "baseline_internal_gap": float(baseline["internal_gap_pct"]),
        "enriched_internal_gap": float(enriched["internal_gap_pct"]),
        "n_plans_changed": int(n_plans_changed),
        "n_mode_switches": int(n_mode_switches),
        "objective_equivalence_type": objective_equivalence_type,
        "columns_used_from_enrichment": int(columns_used_from_enrichment),
        "enrichment_usage_ratio": float(columns_used_from_enrichment) / float(enriched["columns_added"]) if int(enriched["columns_added"]) > 0 else 0.0,
        "solutions_identical": bool(n_plans_changed == 0),
        "baseline_mode_distribution": baseline["mode_distribution"],
        "enriched_mode_distribution": enriched["mode_distribution"],
        "baseline_mode_assignments": baseline["mode_assignments"],
        "enriched_mode_assignments": enriched["mode_assignments"],
        "baseline_plan_keys": baseline_plan_keys,
        "enriched_plan_keys": enriched_plan_keys,
        "rerun_triggered": bool(rerun_triggered),
    }


def rerun_if_needed(instance: Any, baseline: dict[str, Any], epsilon_ratio: float, enriched: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    objective_diff = float(baseline["objective"]) - float(enriched["objective"])
    if abs(objective_diff) <= OBJECTIVE_TOL:
        return enriched, False
    rerun = solve_enriched_from_baseline(instance, baseline, epsilon_ratio)
    rerun_diff = float(baseline["objective"]) - float(rerun["objective"])
    if abs(rerun_diff) <= OBJECTIVE_TOL:
        return rerun, True
    if rerun_diff > OBJECTIVE_TOL:
        if float(rerun["objective"]) <= float(enriched["objective"]):
            return rerun, True
        return enriched, True
    raise RuntimeError(
        f"Rerun produced enriched objective worse than baseline for N={instance.N}, seed={instance.seed}, eps={epsilon_ratio}: diff={rerun_diff}"
    )


def run_single_instance(config_path: str, N: int, seed: int, results_dir: str, force: bool) -> list[dict[str, Any]]:
    config = load_config(config_path)
    instance = generate_instance(int(N), int(seed), "U", config.get("mechanism", "hybrid"), dict(config.get("params", {})))
    cfg = build_method_config(config, int(N), f"exp3_followup_N{N}_seed{seed}")

    root = Path(results_dir)
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = raw_prefix(int(N), int(seed))
    baseline_path = raw_dir / f"{prefix}_baseline.json"
    enriched_paths = {ratio: raw_dir / f"{prefix}_enriched_{epsilon_label(ratio)}.json" for ratio in EPSILON_LEVELS}

    if not force and baseline_path.exists() and all(path.exists() for path in enriched_paths.values()):
        rows = [json.loads(baseline_path.read_text(encoding="utf-8"))]
        rows.extend(json.loads(path.read_text(encoding="utf-8")) for path in enriched_paths.values())
        return rows

    baseline = capture_baseline_pool(instance, cfg)
    baseline_row = build_baseline_payload(int(N), int(seed), baseline)
    save_json(baseline_path, baseline_row)
    rows = [baseline_row]

    for ratio in EPSILON_LEVELS:
        enriched = solve_enriched_from_baseline(instance, baseline, float(ratio))
        enriched, rerun_triggered = rerun_if_needed(instance, baseline, float(ratio), enriched)
        enriched_row = build_enriched_payload(int(N), int(seed), baseline, enriched, float(ratio), rerun_triggered)
        save_json(enriched_paths[ratio], enriched_row)
        rows.append(enriched_row)
    return rows


def parallel_run(tasks: list[tuple[str, int, int, str, bool]], workers: int) -> list[list[dict[str, Any]]]:
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
    log_path = results_dir / "run_log.txt"

    precheck = run_precheck(args.config)
    precheck_dir = results_dir / "precheck"
    precheck_dir.mkdir(parents=True, exist_ok=True)
    (precheck_dir / "diagnostic_results.json").write_text(json.dumps(json_ready(precheck), indent=2, ensure_ascii=False), encoding="utf-8")
    if not precheck["passed"]:
        append_log(log_path, f"[PRECHECK FAILED] {precheck['hard_gate_failures']}")
        raise RuntimeError(f"Precheck failed: {precheck['hard_gate_failures']}")

    append_log(log_path, "[PRECHECK PASSED]")
    for record in precheck["records"]:
        append_log(
            log_path,
            f"  N={record['N']}: pool={record['baseline_pool_size']} / full={record['full_column_pool_size']} "
            f"(non-complete), iters={record['num_iters']}, used_full_pool_small={record['used_full_pool_small']}",
        )
    append_log(log_path, "All three scales confirmed to use iterative top-K pricing. Proceeding to Step 1.")

    config = load_config(args.config)
    seeds = expand_seeds(config.get("seeds", 10))
    tasks = [(args.config, int(N), int(seed), str(results_dir), bool(args.force)) for N in TARGET_NS for seed in seeds]
    append_log(log_path, f"Step 1/2 start: running {len(tasks)} baseline captures with shared enrichment sweeps.")

    batches = parallel_run(tasks, workers=int(args.workers))
    rows = [row for batch in batches for row in batch]
    pd.DataFrame(rows).sort_values(["record_type", "N", "seed", "epsilon_pct"], na_position="first").to_csv(results_dir / "raw_results_index.csv", index=False)

    baseline_count = sum(1 for row in rows if row["record_type"] == "baseline")
    enriched_rows = [row for row in rows if row["record_type"] == "enriched"]
    append_log(log_path, f"Step 2 complete: wrote {baseline_count} baseline records and {len(enriched_rows)} enriched records.")
    append_log(log_path, f"Equivalence-type totals: {pd.Series([row['objective_equivalence_type'] for row in enriched_rows]).value_counts().to_dict()}")

    n100_baseline = [row["objective"] for row in rows if row["record_type"] == "baseline" and int(row["N"]) == 100]
    if n100_baseline:
        main_df = pd.read_csv("results/results_main_rigorous.csv")
        full_mean = float(main_df[(main_df["scenario"] == "U") & (main_df["mechanism"] == config.get("mechanism", "hybrid")) & (main_df["method"] == "cg") & (main_df["N"] == 100)]["obj"].mean())
        topk_mean = float(np.mean(n100_baseline))
        diff_pct = (topk_mean - full_mean) / full_mean * 100.0
        append_log(log_path, f"N=100 top-K diagnostic vs full-pool mean objective diff = {diff_pct:.6f}%.")

    append_log(log_path, "Step 3 pending: run analyze_experiment_3_followup.py to build aggregated tables, figures, and summary.")


if __name__ == "__main__":
    main()

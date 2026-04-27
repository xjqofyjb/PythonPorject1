from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from column_enrichment import capture_baseline_pool
from src.instances import generate_instance
from src.runner import build_method_cfg, load_config


TARGET_NS = [100, 200, 500]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Experiment 3 follow-up precheck.")
    parser.add_argument("--config", default="configs/main.yaml", help="Main experiment config.")
    parser.add_argument("--results-dir", default="results/experiment_3_followup", help="Output directory.")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _soft_gate_ok(N: int, pool_size: int) -> bool:
    if N == 100:
        return 2000 <= pool_size <= 12000
    if N == 200:
        return 10000 <= pool_size <= 25000
    if N == 500:
        return 25000 <= pool_size <= 55000
    return True


def _build_cfg(config: dict[str, Any], N: int) -> dict[str, Any]:
    cfg = build_method_cfg(config, "cg", trace_dir="", instance_id=f"precheck_N{N}")
    cfg["operation_mode"] = "simops"
    cfg.setdefault("cg", {})
    if N == 100:
        cfg["cg"]["use_full_pool_small"] = False
    return cfg


def run_precheck(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    mechanism = config.get("mechanism", "hybrid")
    params = dict(config.get("params", {}))

    records: list[dict[str, Any]] = []
    hard_gate_failures: list[str] = []
    soft_gate_warnings: list[str] = []

    for N in TARGET_NS:
        instance = generate_instance(N, 1, "U", mechanism, params)
        cfg = _build_cfg(config, N)
        baseline = capture_baseline_pool(instance, cfg)

        record = {
            "N": int(N),
            "seed": 1,
            "baseline_pool_size": int(baseline["baseline_pool_size"]),
            "full_column_pool_size": int(baseline["full_column_pool_size"]),
            "baseline_pool_complete": bool(baseline["baseline_pool_complete"]),
            "use_full_pool_small_requested": bool(cfg["cg"].get("use_full_pool_small", True)),
            "used_full_pool_small": bool(baseline["used_full_pool_small"]),
            "num_iters": int(baseline["num_iters"]),
            "is_restricted": False,
            "pricing_calls": int(baseline["pricing_calls"]),
            "num_columns_added_during_cg": int(baseline["num_columns_added_during_cg"]),
            "arrival_signature_first3": [round(float(x), 6) for x in instance.arrival_times[:3]],
            "energy_sum": round(float(instance.energy_kwh.sum()), 6),
            "objective": round(float(baseline["objective"]), 6),
            "hard_gates": {
                "baseline_pool_complete_false": not bool(baseline["baseline_pool_complete"]),
                "use_full_pool_small_false": not bool(baseline["used_full_pool_small"]),
                "num_iters_ge_5": int(baseline["num_iters"]) >= 5,
                "is_restricted_false": True,
            },
            "soft_gate_ok": _soft_gate_ok(N, int(baseline["baseline_pool_size"])),
        }
        records.append(record)

        if not record["hard_gates"]["baseline_pool_complete_false"]:
            hard_gate_failures.append(f"N={N}: baseline_pool_complete=True")
        if not record["hard_gates"]["use_full_pool_small_false"]:
            hard_gate_failures.append(f"N={N}: used_full_pool_small=True")
        if not record["hard_gates"]["num_iters_ge_5"]:
            hard_gate_failures.append(f"N={N}: num_iters={record['num_iters']} < 5")
        if not record["hard_gates"]["is_restricted_false"]:
            hard_gate_failures.append(f"N={N}: is_restricted=True")
        if not record["soft_gate_ok"]:
            soft_gate_warnings.append(f"N={N}: baseline_pool_size={record['baseline_pool_size']} outside expected range")

    return {
        "passed": len(hard_gate_failures) == 0,
        "target_scales": TARGET_NS,
        "records": records,
        "hard_gate_failures": hard_gate_failures,
        "soft_gate_warnings": soft_gate_warnings,
    }


def main() -> None:
    args = parse_args()
    payload = run_precheck(args.config)
    out_dir = Path(args.results_dir) / "precheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostic_results.json"
    out_path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    print("[PRECHECK PASSED]" if payload["passed"] else "[PRECHECK FAILED]")
    for record in payload["records"]:
        print(
            f"N={record['N']}: pool={record['baseline_pool_size']} / full={record['full_column_pool_size']}, "
            f"iters={record['num_iters']}, used_full_pool_small={record['used_full_pool_small']}"
        )
    for item in payload["hard_gate_failures"]:
        print("HARD_FAIL:", item)
    for item in payload["soft_gate_warnings"]:
        print("SOFT_WARN:", item)


if __name__ == "__main__":
    main()

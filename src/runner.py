"""Experiment runner CLI."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List

import numpy as np

from src.instances import generate_instance
from src.metrics import compute_cost_components, compute_mechanism_metrics, compute_mode_ratios
from src.io import append_results, collect_meta, get_logger, write_meta
from src.solvers import (
    cg_solver,
    greedy_solver,
    fifo_solver,
    milp_solver,
    rolling_horizon_solver,
    fix_and_optimize_solver,
)


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from disk."""
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_seeds(seeds_cfg) -> List[int]:
    if isinstance(seeds_cfg, int):
        return list(range(1, seeds_cfg + 1))
    if isinstance(seeds_cfg, str):
        if ":" in seeds_cfg:
            start, end = seeds_cfg.split(":", 1)
            return list(range(int(start), int(end) + 1))
        return [int(seeds_cfg)]
    if isinstance(seeds_cfg, list):
        return [int(x) for x in seeds_cfg]
    return [1]


def expand_n_list(n_cfg) -> List[int]:
    if n_cfg is None:
        return []
    if isinstance(n_cfg, int):
        return [n_cfg]
    if isinstance(n_cfg, str):
        if ":" in n_cfg:
            start, end = n_cfg.split(":", 1)
            return list(range(int(start), int(end) + 1))
        return [int(n_cfg)]
    if isinstance(n_cfg, list):
        return [int(x) for x in n_cfg]
    return []


def get_solver(method: str):
    if method.startswith("cg") or method.startswith("rcg") or method == "restricted_cg":
        return cg_solver.solve
    if method in {"rolling_horizon", "rh_milp", "rolling_horizon_milp"}:
        return rolling_horizon_solver.solve
    if method in {"fix_and_optimize", "fao"}:
        return fix_and_optimize_solver.solve
    if method == "greedy":
        return greedy_solver.solve
    if method == "fifo":
        return fifo_solver.solve
    if method.startswith("milp"):
        return milp_solver.solve
    raise ValueError(f"Unknown method: {method}")


def base_row() -> Dict[str, Any]:
    return {
        "obj": np.nan,
        "runtime_total": np.nan,
        "runtime_pricing": np.nan,
        "gap": np.nan,
        "gap_pct": np.nan,
        "status": "error",
        "error": "",
        "num_iters": np.nan,
        "num_pricing_calls": np.nan,
        "num_columns_added": np.nan,
        "min_reduced_cost_last": np.nan,
        "pricing_time_share": np.nan,
        "cost_energy": np.nan,
        "cost_delay": np.nan,
        "shore_utilization": np.nan,
        "mode_shore_count": np.nan,
        "mode_battery_count": np.nan,
        "mode_brown_count": np.nan,
        "infeasible_jobs": 0,
        "shore_ratio": np.nan,
        "battery_ratio": np.nan,
        "brown_ratio": np.nan,
    }


def build_cg_cfg(config: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Build CG configuration for a given method variant."""
    base_cfg = config.get("cg", {})
    cfg = {"cg": dict(base_cfg)}

    if method == "cg_basic":
        cfg["cg"].update(
            {
            "warm_start": False,
            "stabilization": {"enabled": False, "lambda": base_cfg.get("stabilization", {}).get("lambda", 0.6)},
            "multi_column": {"enabled": False, "k": base_cfg.get("multi_column", {}).get("k", 3)},
            }
        )
    elif method == "cg_warm":
        cfg["cg"].update(
            {
            "warm_start": True,
            "stabilization": {"enabled": False, "lambda": base_cfg.get("stabilization", {}).get("lambda", 0.6)},
            "multi_column": {"enabled": False, "k": base_cfg.get("multi_column", {}).get("k", 3)},
            }
        )
    elif method == "cg_stab":
        cfg["cg"].update(
            {
            "warm_start": False,
            "stabilization": {"enabled": True, "lambda": base_cfg.get("stabilization", {}).get("lambda", 0.6)},
            "multi_column": {"enabled": False, "k": base_cfg.get("multi_column", {}).get("k", 3)},
            }
        )
    elif method == "cg_multik":
        cfg["cg"].update(
            {
            "warm_start": False,
            "stabilization": {"enabled": False, "lambda": base_cfg.get("stabilization", {}).get("lambda", 0.6)},
            "multi_column": {"enabled": True, "k": base_cfg.get("multi_column", {}).get("k", 3)},
            }
        )
    elif method == "cg_full":
        cfg["cg"].update(
            {
            "warm_start": True,
            "stabilization": {"enabled": True, "lambda": base_cfg.get("stabilization", {}).get("lambda", 0.6)},
            "multi_column": {"enabled": True, "k": base_cfg.get("multi_column", {}).get("k", 3)},
            }
        )
    elif method in {"rcg", "rcg_random", "rcg_arrival", "restricted_cg"}:
        restricted_cfg = dict(base_cfg.get("restricted_pricing", {}))
        restricted_cfg["enabled"] = True
        restricted_cfg.setdefault("fraction", 0.5)
        restricted_cfg.setdefault("max_iters", 5)
        if method == "rcg_arrival":
            restricted_cfg["selection"] = "arrival"
        else:
            restricted_cfg["selection"] = "random"
        cfg["cg"]["restricted_pricing"] = restricted_cfg
        cfg["cg"]["use_full_pool_small"] = False
    return cfg


def build_method_cfg(
    config: Dict[str, Any],
    method: str,
    trace_dir: str,
    instance_id: str,
) -> Dict[str, Any]:
    method_cfg: Dict[str, Any] = {"method": method}
    if method.startswith("cg") or method.startswith("rcg") or method == "restricted_cg":
        method_cfg.update(build_cg_cfg(config, method))
        method_cfg["trace_dir"] = trace_dir
        method_cfg["instance_id"] = instance_id
    elif method in {"rolling_horizon", "rh_milp", "rolling_horizon_milp"}:
        rh_cfg = dict(config.get("rolling_horizon", {}))
        rh_cfg.setdefault("window_size", 10)
        rh_cfg.setdefault("commit_size", max(1, rh_cfg["window_size"] // 2))
        rh_cfg.setdefault("time_limit", 30)
        method_cfg["rolling_horizon"] = rh_cfg
    elif method in {"fix_and_optimize", "fao"}:
        fao_cfg = dict(config.get("fix_and_optimize", {}))
        fao_cfg.setdefault("block_size", 10)
        fao_cfg.setdefault("step_size", max(1, fao_cfg["block_size"] // 2))
        fao_cfg.setdefault("max_passes", 2)
        fao_cfg.setdefault("time_limit", 30)
        method_cfg["fix_and_optimize"] = fao_cfg
    return method_cfg


def run_experiment(
    config_path: str,
    config: Dict[str, Any],
    seeds_override: str | int | None = None,
    n_override: str | int | None = None,
) -> None:
    output_path = config.get("output", "results/results.csv")
    log_dir = config.get("log_dir", "results/logs")
    logger = get_logger(log_dir, name="runner")

    if seeds_override is not None:
        config["seeds"] = seeds_override
    if n_override is not None:
        config["N_list"] = expand_n_list(n_override)

    seeds = expand_seeds(config.get("seeds", 10))
    scenarios = config.get("scenarios", ["U"])
    methods = [m.lower() for m in config.get("methods", [])]

    experiment = config.get("experiment", "main")
    params = config.get("params", {})

    logger.info("Experiment=%s | Seeds=%s | Methods=%s", experiment, seeds, methods)

    rows: List[Dict[str, Any]] = []
    trace_dir = config.get("trace_dir", "results/traces")

    def finalize_rows(batch: List[Dict[str, Any]]):
        if batch:
            append_results(output_path, batch)
            batch.clear()

    if experiment == "main":
        n_list = config.get("N_list", [20, 50, 100])
        mechanism = config.get("mechanism", "hybrid")
        for N in n_list:
            for seed in seeds:
                for scenario in scenarios:
                    instance = generate_instance(N, seed, scenario, mechanism, params)
                    for method in methods:
                        row = {
                            "experiment": experiment,
                            "scenario": scenario,
                            "mechanism": mechanism,
                            "N": N,
                            "seed": seed,
                            "method": method,
                            "param_name": "",
                            "param_value": np.nan,
                        }
                        row.update(base_row())
                        if method.startswith("milp") and N > 100:
                            row["status"] = "skipped"
                            row["error"] = "MILP skipped for N>100"
                            rows.append(row)
                            continue
                        try:
                            solve_fn = get_solver(method)
                            method_cfg = build_method_cfg(
                                config,
                                method,
                                trace_dir,
                                f"N{N}_seed{seed}_sc{scenario}",
                            )
                            sol = solve_fn(instance, method_cfg, logger)
                            row.update(sol)
                            if np.isnan(row["runtime_total"]) and "runtime" in sol:
                                row["runtime_total"] = sol["runtime"]
                            if "mechanism_counts" in sol:
                                counts = sol["mechanism_counts"]
                                row["mode_shore_count"] = counts.get("shore", 0)
                                row["mode_battery_count"] = counts.get("battery", 0)
                                row["mode_brown_count"] = counts.get("brown", 0)
                                row.update(compute_mode_ratios(counts, instance.N))
                            if "cost_energy" not in sol:
                                components = compute_cost_components(instance, row["obj"])
                                row.update(components)
                            row.update(compute_mechanism_metrics(instance))
                        except Exception as exc:
                            logger.exception("Run failed: N=%s seed=%s method=%s", N, seed, method)
                            row["error"] = str(exc)
                        rows.append(row)
            finalize_rows(rows)

    elif experiment == "mechanism":
        N = int(config.get("N", 100))
        mechanisms = config.get("mechanisms", ["hybrid", "battery_only", "shore_only"])
        for mechanism in mechanisms:
            for seed in seeds:
                for scenario in scenarios:
                    instance = generate_instance(N, seed, scenario, mechanism, params)
                    for method in methods:
                        row = {
                            "experiment": experiment,
                            "scenario": scenario,
                            "mechanism": mechanism,
                            "N": N,
                            "seed": seed,
                            "method": method,
                            "param_name": "mechanism",
                            "param_value": mechanism,
                        }
                        row.update(base_row())
                        try:
                            solve_fn = get_solver(method)
                            method_cfg = build_method_cfg(
                                config,
                                method,
                                trace_dir,
                                f"N{N}_seed{seed}_sc{scenario}_mech{mechanism}",
                            )
                            sol = solve_fn(instance, method_cfg, logger)
                            row.update(sol)
                            if np.isnan(row["runtime_total"]) and "runtime" in sol:
                                row["runtime_total"] = sol["runtime"]
                            if "mechanism_counts" in sol:
                                counts = sol["mechanism_counts"]
                                row["mode_shore_count"] = counts.get("shore", 0)
                                row["mode_battery_count"] = counts.get("battery", 0)
                                row["mode_brown_count"] = counts.get("brown", 0)
                                row.update(compute_mode_ratios(counts, instance.N))
                            if "cost_energy" not in sol:
                                components = compute_cost_components(instance, row["obj"])
                                row.update(components)
                            row.update(compute_mechanism_metrics(instance))
                        except Exception as exc:
                            logger.exception("Run failed: mech=%s seed=%s method=%s", mechanism, seed, method)
                            row["error"] = str(exc)
                        rows.append(row)
            finalize_rows(rows)

    elif experiment == "scenario":
        N = int(config.get("N", 100))
        mechanism = config.get("mechanism", "hybrid")
        for scenario in scenarios:
            for seed in seeds:
                instance = generate_instance(N, seed, scenario, mechanism, params)
                for method in methods:
                    row = {
                        "experiment": experiment,
                        "scenario": scenario,
                        "mechanism": mechanism,
                        "N": N,
                        "seed": seed,
                        "method": method,
                        "param_name": "scenario",
                        "param_value": scenario,
                    }
                    row.update(base_row())
                    try:
                        solve_fn = get_solver(method)
                        method_cfg = build_method_cfg(
                            config,
                            method,
                            trace_dir,
                            f"N{N}_seed{seed}_sc{scenario}",
                        )
                        sol = solve_fn(instance, method_cfg, logger)
                        row.update(sol)
                        if np.isnan(row["runtime_total"]) and "runtime" in sol:
                            row["runtime_total"] = sol["runtime"]
                        if "mechanism_counts" in sol:
                            counts = sol["mechanism_counts"]
                            row["mode_shore_count"] = counts.get("shore", 0)
                            row["mode_battery_count"] = counts.get("battery", 0)
                            row["mode_brown_count"] = counts.get("brown", 0)
                            row.update(compute_mode_ratios(counts, instance.N))
                        if "cost_energy" not in sol:
                            components = compute_cost_components(instance, row["obj"])
                            row.update(components)
                        row.update(compute_mechanism_metrics(instance))
                    except Exception as exc:
                        logger.exception("Run failed: scenario=%s seed=%s method=%s", scenario, seed, method)
                        row["error"] = str(exc)
                    rows.append(row)
            finalize_rows(rows)

    elif experiment == "sensitivity":
        N = int(config.get("N", 100))
        sensitivity = config.get("sensitivity", {})
        for param_name, values in sensitivity.items():
            for value in values:
                local_params = dict(params)
                local_params[param_name] = value
                for seed in seeds:
                    for scenario in scenarios:
                        instance = generate_instance(N, seed, scenario, config.get("mechanism", "hybrid"), local_params)
                        for method in methods:
                            row = {
                                "experiment": experiment,
                                "scenario": scenario,
                                "mechanism": config.get("mechanism", "hybrid"),
                                "N": N,
                                "seed": seed,
                                "method": method,
                                "param_name": param_name,
                                "param_value": value,
                            }
                            row.update(base_row())
                            try:
                                solve_fn = get_solver(method)
                                method_cfg = build_method_cfg(
                                    config,
                                    method,
                                    trace_dir,
                                    f"N{N}_seed{seed}_sc{scenario}_{param_name}{value}",
                                )
                                sol = solve_fn(instance, method_cfg, logger)
                                row.update(sol)
                                if np.isnan(row["runtime_total"]) and "runtime" in sol:
                                    row["runtime_total"] = sol["runtime"]
                                if "mechanism_counts" in sol:
                                    counts = sol["mechanism_counts"]
                                    row["mode_shore_count"] = counts.get("shore", 0)
                                    row["mode_battery_count"] = counts.get("battery", 0)
                                    row["mode_brown_count"] = counts.get("brown", 0)
                                    row.update(compute_mode_ratios(counts, instance.N))
                                if "cost_energy" not in sol:
                                    components = compute_cost_components(instance, row["obj"])
                                    row.update(components)
                                row.update(compute_mechanism_metrics(instance))
                            except Exception as exc:
                                logger.exception("Run failed: %s=%s seed=%s method=%s", param_name, value, seed, method)
                                row["error"] = str(exc)
                            rows.append(row)
                finalize_rows(rows)
    elif experiment == "simops":
        n_list = config.get("N_list", [25, 50, 100])
        operation_modes = config.get("operation_modes", ["simops", "sequential"])
        mechanism = config.get("mechanism", "hybrid")
        for N in n_list:
            for seed in seeds:
                for scenario in scenarios:
                    instance = generate_instance(N, seed, scenario, mechanism, params)
                    for op_mode in operation_modes:
                        for method in methods:
                            row = {
                                "experiment": experiment,
                                "scenario": scenario,
                                "mechanism": mechanism,
                                "operation_mode": op_mode,
                                "N": N,
                                "seed": seed,
                                "method": method,
                                "param_name": "operation_mode",
                                "param_value": op_mode,
                            }
                            row.update(base_row())
                            try:
                                solve_fn = get_solver(method)
                                method_cfg = build_method_cfg(
                                    config,
                                    method,
                                    trace_dir,
                                    f"N{N}_seed{seed}_sc{scenario}_op{op_mode}",
                                )
                                method_cfg["operation_mode"] = op_mode
                                sol = solve_fn(instance, method_cfg, logger)
                                row.update(sol)
                                if np.isnan(row["runtime_total"]) and "runtime" in sol:
                                    row["runtime_total"] = sol["runtime"]
                                if "mechanism_counts" in sol:
                                    counts = sol["mechanism_counts"]
                                    row["mode_shore_count"] = counts.get("shore", 0)
                                    row["mode_battery_count"] = counts.get("battery", 0)
                                    row["mode_brown_count"] = counts.get("brown", 0)
                                    row.update(compute_mode_ratios(counts, instance.N))
                                if "cost_energy" not in sol:
                                    components = compute_cost_components(instance, row["obj"])
                                    row.update(components)
                                row.update(compute_mechanism_metrics(instance))
                            except Exception as exc:
                                logger.exception(
                                    "Run failed: N=%s seed=%s op=%s method=%s",
                                    N,
                                    seed,
                                    op_mode,
                                    method,
                                )
                                row["error"] = str(exc)
                            rows.append(row)
            finalize_rows(rows)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    meta = collect_meta(config_path, config)
    write_meta(config.get("meta", "results/meta.json"), meta)
    logger.info("Completed. Results -> %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments from YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Override seeds: integer K, list in YAML, or range like 1:5.",
    )
    parser.add_argument(
        "--Ns",
        type=str,
        default=None,
        help="Override N list (main experiment): single N or range like 20:100.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_experiment(args.config, config, seeds_override=args.seeds, n_override=args.Ns)


if __name__ == "__main__":
    main()

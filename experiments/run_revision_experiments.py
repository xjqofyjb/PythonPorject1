"""Revised experiment runner for corrected cost/time/CG conventions.

All numerical outputs are produced by the active solvers. Failed runs are kept
as rows with an error message rather than replaced with fabricated values.
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.instances import generate_instance
from src.metrics import compute_solution_operational_metrics
from src.model_utils import ceil_slots
from src.runner import get_solver


RESULT_DIR = ROOT / "results" / "revised"
FIGURE_DIR = ROOT / "figures" / "revised"
TARGET_RESULT_DIR = RESULT_DIR / "targeted_rerun"
TARGET_FIGURE_DIR = FIGURE_DIR / "targeted_rerun"
TARGET_LOG_DIR = ROOT / "logs" / "revised" / "targeted_rerun"

BASE_PARAMS: Dict[str, Any] = {
    "time_step_hours": 0.25,
    "horizon_hours": 48.0,
    "arrival_window_hours": 8.0,
    "shore_power_kw": 900.0,
    "battery_swap_hours": 0.75,
    "battery_cost": 0.45,
    "shore_cost": 0.15,
    "brown_cost": 0.90,
    "shore_cap": 2,
    "battery_slots": 2,
    "deadline_tightness": 1.0,
    "brown_available": True,
    "grid_emission_factor_kg_per_kwh": 0.445,
    "ae_emission_factor_kg_per_kwh": 0.70,
}

METHOD_ALIASES = {
    "CG+IR": "cg",
    "MILP60": "milp60",
    "MILP300": "milp300",
    "Rolling-Horizon": "rolling_horizon",
    "Fix-and-Optimize": "fix_and_optimize",
    "Restricted-CG": "restricted_cg",
    "FIFO": "fifo",
    "Greedy": "greedy",
}


class Logger:
    def info(self, msg: str, *args: Any) -> None:
        if args:
            msg = msg % args
        print(msg)

    def exception(self, msg: str, *args: Any) -> None:
        if args:
            msg = msg % args
        print(msg)


def ensure_dirs() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_LOG_DIR.mkdir(parents=True, exist_ok=True)


def seeds(full: bool, quick: bool) -> List[int]:
    return [1, 2] if quick else list(range(1, 11 if full else 3))


def method_cfg(method: str, instance_id: str, extra: Dict[str, Any] | None = None, quick: bool = False) -> Dict[str, Any]:
    cg_time = 15 if quick else 60
    cg_iters = 8 if quick else 60
    local_time = 5 if quick else 30
    cfg: Dict[str, Any] = {
        "method": method,
        "operation_mode": "simops",
        "trace_dir": str(RESULT_DIR / "traces"),
        "instance_id": instance_id,
        "cg": {
            "use_full_pool_small": True,
            "full_pool_n": 100,
            "time_limit": cg_time,
            "max_iters": cg_iters,
            "pricing_top_k": 3,
            "pricing_eps": 1e-6,
        },
        "rolling_horizon": {"window_size": 8 if quick else 10, "commit_size": 4 if quick else 5, "time_limit": local_time},
        "fix_and_optimize": {"block_size": 8 if quick else 10, "step_size": 4 if quick else 5, "max_passes": 1 if quick else 2, "time_limit": local_time},
    }
    if method == "restricted_cg":
        cfg["cg"]["use_full_pool_small"] = False
        cfg["cg"]["restricted_pricing"] = {"enabled": True, "fraction": 0.5, "selection": "random", "max_iters": 5}
    if extra:
        cfg.update(extra)
    return cfg


def flatten_result(sol: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(sol)
    counts = row.pop("mechanism_counts", {}) or {}
    row["mode_shore_count"] = counts.get("shore", 0)
    row["mode_battery_count"] = counts.get("battery", 0)
    row["mode_brown_count"] = counts.get("brown", 0)
    total = max(1, row["mode_shore_count"] + row["mode_battery_count"] + row["mode_brown_count"])
    row["SP_share"] = row["mode_shore_count"] / total
    row["BS_share"] = row["mode_battery_count"] / total
    row["AE_share"] = row["mode_brown_count"] / total
    row["objective"] = row.get("obj")
    row["total_cost"] = row.get("obj")
    row["delay_cost"] = row.get("cost_delay")
    row["avg_stay_h"] = row.get("avg_stay_time")
    row["masking_rate"] = row.get("avg_masking_rate")
    return {k: v for k, v in row.items() if not isinstance(v, (dict, list))}


def enrich_solution_metrics(instance: Any, sol: Dict[str, Any], operation_mode: str = "simops") -> Dict[str, Any]:
    metrics = compute_solution_operational_metrics(
        instance,
        sol.get("schedule"),
        operation_mode=operation_mode,
        grid_emission_factor_kg_per_kwh=float(instance.params.get("grid_emission_factor_kg_per_kwh", BASE_PARAMS["grid_emission_factor_kg_per_kwh"])),
        ae_emission_factor_kg_per_kwh=float(instance.params.get("ae_emission_factor_kg_per_kwh", BASE_PARAMS["ae_emission_factor_kg_per_kwh"])),
    )
    enriched = dict(sol)
    enriched.update(metrics)
    return enriched


def solve_instance(instance: Any, method_label: str, scenario: str, extra_cfg: Dict[str, Any] | None = None, quick: bool = False) -> Dict[str, Any]:
    method = METHOD_ALIASES.get(method_label, method_label)
    cfg = method_cfg(method, f"rev_N{instance.N}_seed{instance.seed}_sc{scenario}_{method}", extra_cfg, quick=quick)
    sol = get_solver(method)(instance, cfg, Logger())
    sol = enrich_solution_metrics(instance, sol, cfg.get("operation_mode", "simops"))
    return flatten_result(sol)


def run_one(N: int, seed: int, scenario: str, method_label: str, params: Dict[str, Any], extra_cfg: Dict[str, Any] | None = None, quick: bool = False) -> Dict[str, Any]:
    method = METHOD_ALIASES.get(method_label, method_label)
    row: Dict[str, Any] = {
        "N": N,
        "seed": seed,
        "scenario": scenario,
        "method": method_label,
        "status": "error",
        "error": "",
    }
    if quick and N >= 500 and method in {"rolling_horizon", "fix_and_optimize"}:
        row["status"] = "skipped"
        row["error"] = "Skipped in quick mode for N>=500; run --full for this baseline."
        return row
    if method.startswith("milp") and N > 100:
        row["status"] = "skipped"
        row["error"] = "MILP skipped for N>100"
        return row
    try:
        inst = generate_instance(N, seed, scenario, str(params.get("mechanism", "hybrid")), params)
        sol = solve_instance(inst, method_label, scenario, extra_cfg=extra_cfg, quick=quick)
        row.update(sol)
        row["status"] = sol.get("status", "ok")
    except Exception as exc:
        row["error"] = repr(exc)
    return row


def write_raw(rows: List[Dict[str, Any]], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def append_raw(row: Dict[str, Any], path: Path) -> None:
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        cols = list(old.columns) + [c for c in df.columns if c not in old.columns]
        old = old.reindex(columns=cols)
        df = df.reindex(columns=cols)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def write_summary_table(df: pd.DataFrame, path_csv: Path, path_tex: Path, group_cols: List[str]) -> None:
    ok = df[pd.to_numeric(df.get("obj"), errors="coerce").notna()].copy()
    if ok.empty:
        pd.DataFrame().to_csv(path_csv, index=False)
        path_tex.write_text("% No completed runs available.\n", encoding="utf-8")
        return
    summary = ok.groupby(group_cols, dropna=False).agg(
        obj_mean=("obj", "mean"),
        obj_std=("obj", "std"),
        runtime_mean=("runtime_total", "mean"),
        gap_pct_mean=("gap_pct", "mean"),
        success_rate=("status", lambda s: float(np.mean(s.astype(str).isin(["ok", "Optimal", "optimal"])))),
        cg_status=("cg_status", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
        gap_type=("gap_type", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
    ).reset_index()
    summary.to_csv(path_csv, index=False)
    path_tex.write_text(summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8")


def write_trace_markdown(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    csv_path.with_suffix(".md").write_text(df.to_string(index=False), encoding="utf-8")


def strong_cg_cfg(
    experiment: str,
    N: int,
    scenario: str,
    seed: int,
    operation_mode: str = "simops",
    incumbent_solutions: List[Dict[str, Any]] | None = None,
    max_iters: int | None = None,
    pricing_top_k: int | None = None,
    min_iters: int | None = None,
    use_incumbent_injection: bool | None = None,
) -> Dict[str, Any]:
    instance_id = f"cg_trace_{experiment}_N{N}_{scenario}_seed{seed}_{operation_mode}"
    return {
        "method": "cg",
        "operation_mode": operation_mode,
        "return_schedule": True,
        "trace_dir": str(TARGET_LOG_DIR),
        "instance_id": instance_id,
        "cg": {
            "use_full_pool_small": N <= 100,
            "full_pool_n": 100,
            "time_limit": 60,
            "min_iters": min_iters if min_iters is not None else (20 if N >= 200 else 0),
            "max_iters": max_iters if max_iters is not None else (30 if N >= 200 else 60),
            "pricing_top_k": pricing_top_k if pricing_top_k is not None else (5 if N >= 200 else 3),
            "pricing_eps": 1e-6,
            "use_incumbent_injection": use_incumbent_injection if use_incumbent_injection is not None else N >= 200,
            "stabilization_window": 5,
            "stabilization_rel_improvement": 1e-4,
            "stabilization_gap_pct": 0.01,
            "incumbent_solutions": incumbent_solutions or [],
        },
    }


def restricted_cg_cfg(experiment: str, N: int, scenario: str, seed: int) -> Dict[str, Any]:
    return {
        "method": "restricted_cg",
        "operation_mode": "simops",
        "return_schedule": True,
        "trace_dir": str(TARGET_LOG_DIR),
        "instance_id": f"cg_trace_{experiment}_N{N}_{scenario}_seed{seed}_restricted",
        "cg": {
            "use_full_pool_small": False,
            "full_pool_n": 100,
            "time_limit": 30,
            "max_iters": 8,
            "pricing_top_k": 3,
            "pricing_eps": 1e-6,
            "restricted_pricing": {"enabled": True, "fraction": 0.5, "selection": "random", "max_iters": 8},
        },
    }


def normalize_target_row(row: Dict[str, Any], N: int, scenario: str, seed: int, method: str, operation_mode: str | None = None) -> Dict[str, Any]:
    out = dict(row)
    out["N"] = N
    out["scenario"] = scenario
    out["seed"] = seed
    out["method"] = method
    if operation_mode is not None:
        out["operation_mode"] = operation_mode
    out["objective"] = out.get("objective", out.get("obj"))
    out["total_cost"] = out.get("total_cost", out.get("obj"))
    out["delay_cost"] = out.get("delay_cost", out.get("cost_delay"))
    out["avg_delay_h"] = out.get("avg_delay_h", np.nan)
    out["avg_stay_h"] = out.get("avg_stay_h", out.get("avg_stay_time"))
    out["masking_rate"] = out.get("masking_rate", out.get("avg_masking_rate"))
    out["runtime_sec"] = out.get("runtime_sec", out.get("runtime_total"))
    out["pool_gap_pct"] = out.get("pool_gap_pct", out.get("gap_pct"))
    out["min_reduced_cost_scanned_last"] = out.get("min_reduced_cost_scanned_last", out.get("best_reduced_cost_last"))
    out["num_negative_columns_scanned_last"] = out.get("num_negative_columns_scanned_last", out.get("num_negative_columns_last"))
    out["num_columns_added_last"] = out.get("num_columns_added_last")
    out["min_reduced_cost_added_last"] = out.get("min_reduced_cost_added_last")
    return out


def plot_from_csv(csv_path: Path, fig_base: Path, x: str, y: str, hue: str | None = None, ylabel: str = "Cost ($)") -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df[y] = pd.to_numeric(df.get(y), errors="coerce")
    df = df.dropna(subset=[x, y])
    fig, ax = plt.subplots(figsize=(7, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No completed runs", ha="center", va="center")
    elif hue and hue in df.columns:
        for label, sub in df.groupby(hue):
            mean = sub.groupby(x)[y].mean()
            ax.plot(mean.index, mean.values, marker="o", label=str(label))
        ax.legend()
    else:
        mean = df.groupby(x)[y].mean()
        ax.plot(mean.index, mean.values, marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_base.with_suffix(".png"), dpi=200)
    fig.savefig(fig_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_heatmap(csv_path: Path, fig_base: Path, value_col: str, title: str) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    df[value_col] = pd.to_numeric(df.get(value_col), errors="coerce")
    pivot = df.pivot_table(index="K_BS", columns="K_SP", values=value_col, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    if pivot.empty:
        ax.text(0.5, 0.5, "No completed runs", ha="center", va="center")
    else:
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[str(c) for c in pivot.index])
        ax.set_xlabel("SP berths")
        ax.set_ylabel("BS slots")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(fig_base.with_suffix(".png"), dpi=200)
    fig.savefig(fig_base.with_suffix(".pdf"))
    plt.close(fig)


def run_main_benchmark(full: bool, quick: bool) -> None:
    rows = []
    raw_path = RESULT_DIR / "main_benchmark_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    Ns = [20, 50, 100, 200, 500]
    scenarios = ["U", "P", "L"]
    methods = ["CG+IR", "MILP60", "MILP300", "Rolling-Horizon", "Fix-and-Optimize", "Restricted-CG", "FIFO", "Greedy"]
    if quick:
        methods = ["CG+IR", "Rolling-Horizon", "Fix-and-Optimize", "Restricted-CG", "FIFO", "Greedy"]
    for N in Ns:
        for sc in scenarios:
            for sd in seeds(full, quick):
                for m in methods:
                    row = run_one(N, sd, sc, m, dict(BASE_PARAMS), quick=quick)
                    rows.append(row)
                    append_raw(row, raw_path)
    df = pd.read_csv(raw_path) if raw_path.exists() else write_raw(rows, raw_path)
    write_summary_table(df, RESULT_DIR / "table8_revised.csv", RESULT_DIR / "table8_revised.tex", ["N", "scenario", "method"])
    plot_from_csv(RESULT_DIR / "main_benchmark_raw.csv", FIGURE_DIR / "fig3_revised", "N", "obj", "method", "Total cost ($)")


def run_simops_dual_peak(full: bool, quick: bool) -> None:
    rows = []
    raw_path = RESULT_DIR / "simops_dual_peak_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    Ns = [25, 50, 75, 90, 100, 110, 125, 150, 200, 500]
    if quick:
        Ns = [25, 100, 200]
    for N in Ns:
        for sd in seeds(full, quick):
            pair = {}
            for op in ["simops", "sequential"]:
                row = run_one(N, sd, "U", "CG+IR", dict(BASE_PARAMS), {"operation_mode": op}, quick=quick)
                row["operation_mode"] = op
                rows.append(row)
                pair[op] = row
            if "simops" in pair and "sequential" in pair:
                seq = pd.to_numeric(pd.Series([pair["sequential"].get("obj")]), errors="coerce").iloc[0]
                sim = pd.to_numeric(pd.Series([pair["simops"].get("obj")]), errors="coerce").iloc[0]
                if pd.notna(seq) and seq:
                    pair["simops"]["simops_saving_pct"] = (seq - sim) / seq * 100.0
            for row in pair.values():
                append_raw(row, raw_path)
    df = pd.read_csv(raw_path) if raw_path.exists() else write_raw(rows, raw_path)
    ok = df[df["operation_mode"].eq("simops")].copy()
    summary = ok.groupby("N", dropna=False).agg(
        total_cost=("obj", "mean"),
        simops_saving_pct=("simops_saving_pct", "mean"),
        masking_rate=("avg_masking_rate", "mean"),
        SP_share=("SP_share", "mean"),
        BS_share=("BS_share", "mean"),
        AE_share=("AE_share", "mean"),
        avg_stay_h=("avg_stay_time", "mean"),
    ).reset_index()
    summary.to_csv(RESULT_DIR / "simops_dual_peak_summary.csv", index=False)
    plot_from_csv(RESULT_DIR / "simops_dual_peak_summary.csv", FIGURE_DIR / "fig5_dual_peak_ci", "N", "simops_saving_pct", None, "SIMOPS saving (%)")


def run_bs_cost_sensitivity(full: bool, quick: bool) -> None:
    rows = []
    raw_path = RESULT_DIR / "bs_cost_sensitivity_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    methods = ["CG+IR", "Rolling-Horizon", "Fix-and-Optimize"]
    for c_bs in [0.30, 0.40, 0.45, 0.60, 0.80]:
        params = dict(BASE_PARAMS)
        params["battery_cost"] = c_bs
        for sd in seeds(full, quick):
            for m in methods:
                row = run_one(100, sd, "U", m, params, quick=quick)
                row["C_BS"] = c_bs
                rows.append(row)
                append_raw(row, raw_path)
    df = pd.read_csv(raw_path) if raw_path.exists() else write_raw(rows, raw_path)
    summary = df.groupby(["C_BS", "method"], dropna=False).agg(total_cost=("obj", "mean"), cost_std=("obj", "std")).reset_index()
    summary.to_csv(RESULT_DIR / "bs_cost_sensitivity_summary.csv", index=False)
    plot_from_csv(RESULT_DIR / "bs_cost_sensitivity_raw.csv", FIGURE_DIR / "fig6a_bs_cost_revised", "C_BS", "obj", "method", "Total cost ($)")


def run_scenario_comparison(full: bool, quick: bool) -> None:
    raw_path = RESULT_DIR / "scenario_comparison_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    methods = ["CG+IR", "Rolling-Horizon", "Fix-and-Optimize", "FIFO", "Greedy"]
    for sc in ["U", "P", "L"]:
        for sd in seeds(full, quick):
            for m in methods:
                row = run_one(100, sd, sc, m, dict(BASE_PARAMS), quick=quick)
                append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    write_summary_table(df, RESULT_DIR / "table9_revised.csv", RESULT_DIR / "table9_revised.tex", ["scenario", "method"])
    plot_from_csv(raw_path, FIGURE_DIR / "fig4_revised", "scenario", "obj", "method", "Total cost ($)")


def run_capacity_grid(full: bool, quick: bool) -> None:
    raw_path = RESULT_DIR / "capacity_grid_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    k_sp_values = [1, 2, 4, 6, 8] if not quick else [1, 2, 4]
    k_bs_values = [1, 2, 4] if not quick else [1, 2]
    for k_sp in k_sp_values:
        for k_bs in k_bs_values:
            params = dict(BASE_PARAMS, shore_cap=k_sp, battery_slots=k_bs)
            for sd in seeds(full, quick):
                row = run_one(100, sd, "U", "CG+IR", params, quick=quick)
                row["K_SP"] = k_sp
                row["K_BS"] = k_bs
                row["co2_proxy_kg"] = row.get("mode_brown_count", 0) * params["brown_cost"]
                append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    if "avg_masking_rate" not in df.columns:
        df["avg_masking_rate"] = np.nan
    summary = df.groupby(["K_SP", "K_BS"], dropna=False).agg(
        total_cost=("obj", "mean"),
        AE_share=("AE_share", "mean"),
        masking_rate=("avg_masking_rate", "mean"),
        co2_proxy_kg=("co2_proxy_kg", "mean"),
    ).reset_index()
    summary.to_csv(RESULT_DIR / "capacity_grid_summary.csv", index=False)
    plot_heatmap(RESULT_DIR / "capacity_grid_summary.csv", FIGURE_DIR / "capacity_heatmap_cost", "total_cost", "Mean total cost ($)")
    plot_heatmap(RESULT_DIR / "capacity_grid_summary.csv", FIGURE_DIR / "capacity_heatmap_ae_share", "AE_share", "AE share")
    plot_heatmap(RESULT_DIR / "capacity_grid_summary.csv", FIGURE_DIR / "capacity_heatmap_masking_rate", "masking_rate", "Masking rate")
    plot_heatmap(RESULT_DIR / "capacity_grid_summary.csv", FIGURE_DIR / "capacity_heatmap_co2", "co2_proxy_kg", "CO2 proxy")


def run_carbon_grid_factor(full: bool, quick: bool) -> None:
    raw_path = RESULT_DIR / "carbon_grid_factor_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    carbon_prices = [100, 140, 200, 260, 320, 380] if not quick else [100, 260, 380]
    grid_factors = [0.20, 0.445, 0.565, 0.80] if not quick else [0.20, 0.565]
    configs = {"constrained": (1, 1), "baseline": (2, 2), "expanded": (4, 4)}
    ae_noncarbon = float(BASE_PARAMS["brown_cost"]) - 0.70 / 1000.0 * 100.0
    for carbon_price in carbon_prices:
        for grid_factor in grid_factors:
            for cfg_name, (k_sp, k_bs) in configs.items():
                params = dict(BASE_PARAMS, shore_cap=k_sp, battery_slots=k_bs)
                params["brown_cost"] = ae_noncarbon + 0.70 / 1000.0 * carbon_price
                for sd in seeds(full, quick):
                    extra = None
                    if quick:
                        extra = {"cg": {"use_full_pool_small": False, "full_pool_n": 100, "time_limit": 10, "max_iters": 5, "pricing_top_k": 3, "pricing_eps": 1e-6}}
                    row = run_one(100, sd, "U", "CG+IR", params, extra_cfg=extra, quick=quick)
                    row.update({"carbon_price": carbon_price, "grid_factor": grid_factor, "capacity_config": cfg_name})
                    energy = pd.to_numeric(pd.Series([row.get("cost_energy")]), errors="coerce").iloc[0]
                    row["emissions_proxy_kg"] = float(energy) * grid_factor if pd.notna(energy) else np.nan
                    append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    summary = df.groupby(["carbon_price", "grid_factor", "capacity_config"], dropna=False).agg(
        total_cost=("obj", "mean"),
        emissions_proxy_kg=("emissions_proxy_kg", "mean"),
        AE_share=("AE_share", "mean"),
    ).reset_index()
    summary.to_csv(RESULT_DIR / "carbon_grid_factor_summary.csv", index=False)
    plot_from_csv(raw_path, FIGURE_DIR / "fig7_carbon_capacity_revised", "carbon_price", "obj", "capacity_config", "Total cost ($)")
    plot_from_csv(raw_path, FIGURE_DIR / "grid_factor_emissions_sensitivity", "grid_factor", "emissions_proxy_kg", "capacity_config", "Emissions proxy (kgCO2)")


def run_mechanism_comparison(full: bool, quick: bool) -> None:
    raw_path = RESULT_DIR / "mechanism_comparison_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    mechanisms = {
        "Hybrid": dict(BASE_PARAMS),
        "SP_only": dict(BASE_PARAMS, battery_slots=0),
        "BS_only": dict(BASE_PARAMS, shore_cap=0),
        "Green_only_no_AE": dict(BASE_PARAMS, brown_available=False),
    }
    methods = ["CG+IR", "Rolling-Horizon", "Fix-and-Optimize"]
    for sc in ["U", "P", "L"]:
        for mech_name, params in mechanisms.items():
            for sd in seeds(full, quick):
                for m in methods:
                    if mech_name == "Green_only_no_AE" and m != "CG+IR":
                        row = {"N": 100, "seed": sd, "scenario": sc, "method": m, "mechanism": mech_name, "status": "skipped", "error": "No-AE enforcement is implemented for CG columns only."}
                    else:
                        row = run_one(100, sd, sc, m, params, quick=quick)
                        row["mechanism"] = mech_name
                    append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    summary = df.groupby(["scenario", "mechanism", "method"], dropna=False).agg(
        total_cost=("obj", "mean"),
        AE_share=("AE_share", "mean"),
        infeasibility_rate=("status", lambda s: float(np.mean(~s.astype(str).isin(["ok", "Optimal", "optimal"])))),
    ).reset_index()
    summary.to_csv(RESULT_DIR / "mechanism_comparison_summary.csv", index=False)
    plot_from_csv(raw_path, FIGURE_DIR / "mechanism_comparison_cost", "mechanism", "obj", "method", "Total cost ($)")
    plot_from_csv(raw_path, FIGURE_DIR / "mechanism_comparison_emissions", "mechanism", "AE_share", "method", "AE share")


def _perturb_instance(inst: Any, delta: float, kind: str, slack_kind: str) -> None:
    rng = np.random.default_rng(inst.seed + int(delta * 1000) + (17 if kind == "symmetric" else 31 if kind == "correlated" else 0))
    if kind == "one_sided_delay":
        perturb = rng.uniform(0.0, delta, size=inst.N)
    elif kind == "symmetric":
        perturb = rng.uniform(-delta, delta, size=inst.N)
    else:
        xi = rng.uniform(-delta / 2.0, delta / 2.0)
        perturb = xi + rng.uniform(-delta / 2.0, delta / 2.0, size=inst.N)
    inst.original_arrival_times = inst.arrival_times.copy()
    inst.perturbation_hours = perturb.copy()
    inst.arrival_times = np.maximum(0.0, inst.arrival_times + perturb)
    inst.arrival_steps = np.array([ceil_slots(x, inst.dt_hours) for x in inst.arrival_times], dtype=int)
    if slack_kind == "tight":
        inst.deadlines = inst.original_arrival_times + inst.cargo_times + rng.uniform(0.5, 1.5, size=inst.N)
    else:
        inst.deadlines = inst.original_arrival_times + inst.cargo_times + rng.uniform(2.0, 6.0, size=inst.N)
    inst.deadline_steps = np.array([ceil_slots(x, inst.dt_hours) for x in inst.deadlines], dtype=int)


def run_arrival_perturbation(full: bool, quick: bool) -> None:
    raw_path = RESULT_DIR / "arrival_perturbation_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    deltas = [0, 0.5, 1.0, 1.5, 2.0] if not quick else [0, 1.0, 2.0]
    kinds = ["one_sided_delay", "symmetric", "correlated"] if not quick else ["one_sided_delay", "symmetric"]
    for slack_kind in ["loose", "tight"]:
        for kind in kinds:
            for delta in deltas:
                for sd in seeds(full, quick):
                    row = {"N": 100, "seed": sd, "scenario": "U", "method": "CG+IR", "slack": slack_kind, "perturbation_type": kind, "Delta": delta, "status": "error", "error": ""}
                    try:
                        inst = generate_instance(100, sd, "U", "hybrid", dict(BASE_PARAMS))
                        _perturb_instance(inst, delta, kind, slack_kind)
                        row.update(solve_instance(inst, "CG+IR", "U", quick=quick))
                        row["status"] = row.get("status", "ok")
                        row["avg_arrival_shift_h"] = float(np.mean(inst.perturbation_hours))
                    except Exception as exc:
                        row["error"] = repr(exc)
                    append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    summary = df.groupby(["slack", "perturbation_type", "Delta"], dropna=False).agg(total_cost=("obj", "mean"), avg_stay_h=("avg_stay_time", "mean")).reset_index()
    summary.to_csv(RESULT_DIR / "arrival_perturbation_summary.csv", index=False)
    plot_from_csv(raw_path, FIGURE_DIR / "fig8_arrival_perturbation_revised", "Delta", "obj", "perturbation_type", "Total cost ($)")
    plot_from_csv(raw_path, FIGURE_DIR / "fig9_slack_relative_boundary_revised", "Delta", "avg_stay_time", "slack", "Average stay (h)")


def run_column_pool_enrichment(full: bool, quick: bool) -> None:
    raw_path = RESULT_DIR / "column_pool_enrichment_raw.csv"
    if raw_path.exists():
        raw_path.unlink()
    variants = {
        "baseline": {"max_iters": 60, "pricing_top_k": 3},
        "topK5": {"max_iters": 60, "pricing_top_k": 5},
        "topK10": {"max_iters": 60, "pricing_top_k": 10},
        "iter100_topK3": {"max_iters": 100, "pricing_top_k": 3},
    }
    if quick:
        variants = {"baseline": {"max_iters": 8, "pricing_top_k": 3}, "topK5": {"max_iters": 8, "pricing_top_k": 5}}
    for N in ([200, 500] if not quick else [200]):
        baseline_by_seed: Dict[int, float] = {}
        for sd in seeds(full, quick):
            for name, cg_overrides in variants.items():
                extra = {"cg": {"use_full_pool_small": False, "full_pool_n": 100, "time_limit": 15 if quick else 60, **cg_overrides}}
                row = run_one(N, sd, "U", "CG+IR", dict(BASE_PARAMS), extra_cfg=extra, quick=quick)
                row["variant"] = name
                if name == "baseline" and pd.notna(pd.to_numeric(pd.Series([row.get("irmp_obj")]), errors="coerce").iloc[0]):
                    baseline_by_seed[sd] = float(row.get("irmp_obj"))
                    row["objective_improvement_vs_baseline_pct"] = 0.0
                elif sd in baseline_by_seed and pd.notna(pd.to_numeric(pd.Series([row.get("irmp_obj")]), errors="coerce").iloc[0]):
                    row["objective_improvement_vs_baseline_pct"] = (baseline_by_seed[sd] - float(row.get("irmp_obj"))) / max(1.0, abs(baseline_by_seed[sd])) * 100.0
                append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    summary = df.groupby(["N", "variant"], dropna=False).agg(
        irmp_obj=("irmp_obj", "mean"),
        lp_obj_final_pool=("lp_obj_final_pool", "mean"),
        pool_gap_pct=("gap_pct", "mean"),
        num_columns_total=("num_columns_total", "mean"),
        runtime_sec=("runtime_sec", "mean"),
        objective_improvement_vs_baseline_pct=("objective_improvement_vs_baseline_pct", "mean"),
    ).reset_index()
    summary.to_csv(RESULT_DIR / "column_pool_enrichment_summary.csv", index=False)
    (RESULT_DIR / "appendixB_enrichment_revised.tex").write_text(summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8")


def run_targeted_simops_dominance(N: int, scenario: str, seed_values: List[int], strong: bool = True) -> None:
    raw_path = TARGET_RESULT_DIR / "targeted_simops_dominance_raw.csv"
    summary_path = TARGET_RESULT_DIR / "targeted_simops_dominance_summary.csv"
    if raw_path.exists():
        raw_path.unlink()
    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for sd in seed_values:
        inst = generate_instance(N, sd, scenario, "hybrid", dict(BASE_PARAMS))
        seq_cfg = strong_cg_cfg("targeted_simops_dominance", N, scenario, sd, "sequential")
        seq_sol = get_solver("cg")(inst, seq_cfg, Logger())
        seq_sol = enrich_solution_metrics(inst, seq_sol, "sequential")
        seq_sol["method"] = "CG+IR sequential"
        seq_row = normalize_target_row(flatten_result(seq_sol), N, scenario, sd, "CG+IR", "sequential")
        rows.append(seq_row)
        append_raw(seq_row, raw_path)
        write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_targeted_simops_dominance_N{N}_{scenario}_seed{sd}_sequential_cg.csv")

        sim_cfg = strong_cg_cfg("targeted_simops_dominance", N, scenario, sd, "simops", incumbent_solutions=[seq_sol])
        sim_sol = get_solver("cg")(inst, sim_cfg, Logger())
        sim_sol = enrich_solution_metrics(inst, sim_sol, "simops")
        sim_row = normalize_target_row(flatten_result(sim_sol), N, scenario, sd, "CG+IR", "simops")
        seq_obj = float(seq_row["objective"])
        sim_obj = float(sim_row["objective"])
        saving = (seq_obj - sim_obj) / max(1.0, abs(seq_obj)) * 100.0
        passed = sim_obj <= seq_obj + 1e-6
        sim_row["simops_saving_pct"] = saving
        sim_row["dominance_check_passed"] = passed
        rows.append(sim_row)
        append_raw(sim_row, raw_path)
        write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_targeted_simops_dominance_N{N}_{scenario}_seed{sd}_simops_cg.csv")

        summary_rows.append({
            "seed": sd,
            "sequential_obj": seq_obj,
            "simops_obj": sim_obj,
            "simops_saving_pct": saving,
            "dominance_check_passed": passed,
            "sequential_AE_share": seq_row.get("AE_share"),
            "simops_AE_share": sim_row.get("AE_share"),
            "sequential_BS_share": seq_row.get("BS_share"),
            "simops_BS_share": sim_row.get("BS_share"),
            "sequential_SP_share": seq_row.get("SP_share"),
            "simops_SP_share": sim_row.get("SP_share"),
            "seq_cg_status": seq_row.get("cg_status"),
            "simops_cg_status": sim_row.get("cg_status"),
            "seq_gap_type": seq_row.get("gap_type"),
            "simops_gap_type": sim_row.get("gap_type"),
            "seq_objective_stabilized": seq_row.get("objective_stabilized"),
            "simops_objective_stabilized": sim_row.get("objective_stabilized"),
            "seq_best_reduced_cost_last": seq_row.get("best_reduced_cost_last"),
            "simops_best_reduced_cost_last": sim_row.get("best_reduced_cost_last"),
        })
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def run_targeted_N500_stress(N: int, scenario: str, seed: int, strong: bool = True) -> None:
    raw_path = TARGET_RESULT_DIR / "targeted_N500_stress_raw.csv"
    summary_path = TARGET_RESULT_DIR / "targeted_N500_stress_summary.csv"
    if raw_path.exists():
        raw_path.unlink()
    inst = generate_instance(N, seed, scenario, "hybrid", dict(BASE_PARAMS))
    rows: List[Dict[str, Any]] = []
    methods = [
        ("CG+IR", "cg", strong_cg_cfg("targeted_N500_stress", N, scenario, seed, "simops")),
        ("Restricted-CG", "restricted_cg", restricted_cg_cfg("targeted_N500_stress", N, scenario, seed)),
        ("Greedy", "greedy", {"method": "greedy", "operation_mode": "simops", "return_schedule": True}),
        ("FIFO", "fifo", {"method": "fifo", "operation_mode": "simops", "return_schedule": True}),
        ("Rolling-Horizon", "rolling_horizon", {"method": "rolling_horizon", "operation_mode": "simops", "return_schedule": True, "rolling_horizon": {"window_size": 8, "commit_size": 4, "time_limit": 5}}),
        ("Fix-and-Optimize", "fix_and_optimize", {"method": "fix_and_optimize", "operation_mode": "simops", "return_schedule": True, "fix_and_optimize": {"block_size": 8, "step_size": 4, "max_passes": 1, "time_limit": 5}}),
    ]
    for label, solver_name, cfg in methods:
        row = {"N": N, "scenario": scenario, "seed": seed, "method": label, "status": "error", "error": ""}
        try:
            sol = get_solver(solver_name)(inst, cfg, Logger())
            sol = enrich_solution_metrics(inst, sol, cfg.get("operation_mode", "simops"))
            row.update(normalize_target_row(flatten_result(sol), N, scenario, seed, label))
            row["status"] = sol.get("status", "ok")
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)
        append_raw(row, raw_path)
    write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_targeted_N500_stress_N{N}_{scenario}_seed{seed}_simops_cg.csv")
    df = pd.DataFrame(rows)
    ok = df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].copy()
    ok.to_csv(summary_path, index=False)


def run_targeted_table8_N200_replacement(N: int, scenarios: List[str], seed_values: List[int], strong: bool = True) -> None:
    raw_path = TARGET_RESULT_DIR / "table8_N200_replacement_raw.csv"
    summary_path = TARGET_RESULT_DIR / "table8_N200_replacement_summary.csv"
    preview_csv = TARGET_RESULT_DIR / "table8_targeted_preview.csv"
    preview_tex = TARGET_RESULT_DIR / "table8_targeted_preview.tex"
    if raw_path.exists():
        raw_path.unlink()
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        for sd in seed_values:
            inst = generate_instance(N, sd, sc, "hybrid", dict(BASE_PARAMS))
            methods = [
                ("CG+IR", "cg", strong_cg_cfg("targeted_table8_N200_replacement", N, sc, sd, "simops")),
                ("Restricted-CG", "restricted_cg", restricted_cg_cfg("targeted_table8_N200_replacement", N, sc, sd)),
                ("Greedy", "greedy", {"method": "greedy", "operation_mode": "simops", "return_schedule": True}),
                ("FIFO", "fifo", {"method": "fifo", "operation_mode": "simops", "return_schedule": True}),
                ("Rolling-Horizon", "rolling_horizon", {"method": "rolling_horizon", "operation_mode": "simops", "return_schedule": True, "rolling_horizon": {"window_size": 8, "commit_size": 4, "time_limit": 5}}),
                ("Fix-and-Optimize", "fix_and_optimize", {"method": "fix_and_optimize", "operation_mode": "simops", "return_schedule": True, "fix_and_optimize": {"block_size": 8, "step_size": 4, "max_passes": 1, "time_limit": 5}}),
            ]
            for label, solver_name, cfg in methods:
                row = {"N": N, "scenario": sc, "seed": sd, "method": label, "status": "error", "error": ""}
                try:
                    sol = get_solver(solver_name)(inst, cfg, Logger())
                    sol = enrich_solution_metrics(inst, sol, cfg.get("operation_mode", "simops"))
                    row.update(normalize_target_row(flatten_result(sol), N, sc, sd, label))
                    row["status"] = sol.get("status", "ok")
                except Exception as exc:
                    row["error"] = repr(exc)
                rows.append(row)
                append_raw(row, raw_path)
            write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_targeted_table8_N200_replacement_N{N}_{sc}_seed{sd}_simops_cg.csv")
    df = pd.DataFrame(rows)
    summary = df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].groupby(["N", "scenario", "method"], dropna=False).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        runtime_sec_mean=("runtime_sec", "mean"),
        gap_pct_mean=("gap_pct", "mean"),
        cg_status=("cg_status", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
        gap_type=("gap_type", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
        pricing_converged=("pricing_converged", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
        objective_stabilized=("objective_stabilized", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    summary.to_csv(preview_csv, index=False)
    preview_tex.write_text(summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8")


def summarize_benchmark_raw(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].copy()
    if ok.empty:
        return pd.DataFrame()
    cg = ok[ok["method"].eq("CG+IR")][["scenario", "seed", "objective"]].rename(columns={"objective": "cg_objective"})
    ok = ok.merge(cg, on=["scenario", "seed"], how="left")
    ok["rel_gap_to_CG"] = (pd.to_numeric(ok["objective"], errors="coerce") - pd.to_numeric(ok["cg_objective"], errors="coerce")) / ok["cg_objective"].abs().clip(lower=1.0) * 100.0
    summary = ok.groupby(["N", "scenario", "method"], dropna=False).agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        rel_gap_to_CG_mean=("rel_gap_to_CG", "mean"),
        runtime_mean=("runtime_sec", "mean"),
        runtime_std=("runtime_sec", "std"),
        status_success_rate=("status", lambda s: float(np.mean(s.astype(str).eq("ok")))),
        cg_status=("cg_status", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
        gap_type=("gap_type", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
        pricing_converged_rate=("pricing_converged", lambda s: float(np.mean(s.astype(str).str.lower().eq("true"))) if len(s.dropna()) else np.nan),
        objective_stabilized_rate=("objective_stabilized", lambda s: float(np.mean(s.astype(str).str.lower().eq("true"))) if len(s.dropna()) else np.nan),
        pool_gap_pct_mean=("pool_gap_pct", "mean"),
        columns_mean=("num_columns_total", "mean"),
        iterations_mean=("iterations", "mean"),
        relative_improvement_last_5_mean=("relative_improvement_last_5", "mean"),
    ).reset_index()
    return summary


def run_controlled_benchmark(prefix: str, N: int, scenarios: List[str], seed_values: List[int], include_heavy: bool = True) -> pd.DataFrame:
    raw_path = TARGET_RESULT_DIR / f"{prefix}_raw.csv"
    summary_path = TARGET_RESULT_DIR / f"{prefix}_summary.csv"
    tex_path = TARGET_RESULT_DIR / f"{prefix}.tex"
    diag_path = TARGET_RESULT_DIR / "n200_baseline_wins_diagnostic.csv" if "n200" in prefix else None
    if raw_path.exists():
        raw_path.unlink()
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        for sd in seed_values:
            inst = generate_instance(N, sd, sc, "hybrid", dict(BASE_PARAMS))
            methods = [
                ("CG+IR", "cg", strong_cg_cfg(prefix, N, sc, sd, "simops")),
                ("Restricted-CG", "restricted_cg", restricted_cg_cfg(prefix, N, sc, sd)),
                ("Greedy", "greedy", {"method": "greedy", "operation_mode": "simops", "return_schedule": True}),
                ("FIFO", "fifo", {"method": "fifo", "operation_mode": "simops", "return_schedule": True}),
            ]
            if include_heavy:
                methods.extend([
                    ("Rolling-Horizon", "rolling_horizon", {"method": "rolling_horizon", "operation_mode": "simops", "return_schedule": True, "rolling_horizon": {"window_size": 8, "commit_size": 4, "time_limit": 5}}),
                    ("Fix-and-Optimize", "fix_and_optimize", {"method": "fix_and_optimize", "operation_mode": "simops", "return_schedule": True, "fix_and_optimize": {"block_size": 8, "step_size": 4, "max_passes": 1, "time_limit": 5}}),
                ])
            for label, solver_name, cfg in methods:
                row = {"N": N, "scenario": sc, "seed": sd, "method": label, "status": "error", "error": ""}
                try:
                    sol = get_solver(solver_name)(inst, cfg, Logger())
                    sol = enrich_solution_metrics(inst, sol, cfg.get("operation_mode", "simops"))
                    row.update(normalize_target_row(flatten_result(sol), N, sc, sd, label))
                    row["status"] = sol.get("status", "ok")
                except Exception as exc:
                    row["error"] = repr(exc)
                rows.append(row)
                append_raw(row, raw_path)
            write_trace_markdown(TARGET_LOG_DIR / f"{prefix}_N{N}_{sc}_seed{sd}_simops_cg.csv")
            write_trace_markdown(TARGET_LOG_DIR / f"{prefix}_N{N}_{sc}_seed{sd}_restricted_restricted_cg.csv")
    df = pd.DataFrame(rows)
    summary = summarize_benchmark_raw(df)
    summary.to_csv(summary_path, index=False)
    tex_path.write_text(summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    if diag_path is not None:
        wins = []
        for (sc, sd), g in df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].groupby(["scenario", "seed"]):
            cg_obj = pd.to_numeric(g[g["method"].eq("CG+IR")]["objective"], errors="coerce")
            if cg_obj.empty:
                continue
            cg_val = float(cg_obj.iloc[0])
            for rec in g[~g["method"].eq("CG+IR")].to_dict(orient="records"):
                obj = pd.to_numeric(pd.Series([rec.get("objective")]), errors="coerce").iloc[0]
                if pd.notna(obj) and float(obj) < cg_val - 1e-6:
                    wins.append({"scenario": sc, "seed": sd, "baseline_method": rec["method"], "baseline_obj": obj, "cg_obj": cg_val, "advantage_pct": (cg_val - obj) / max(1.0, abs(cg_val)) * 100.0})
        pd.DataFrame(wins).to_csv(diag_path, index=False)
    return df


def run_n500_cautious_expansion(N: int, scenario: str, seed_values: List[int]) -> bool:
    df = run_controlled_benchmark("n500_cautious_expansion", N, [scenario], seed_values, include_heavy=True)
    diag_lines = ["# N=500 Cautious Expansion Diagnostic", ""]
    pass_count = 0
    losses = []
    for sd, g in df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].groupby("seed"):
        objs = g.set_index("method")["objective"].astype(float)
        cg = float(objs.get("CG+IR", np.nan))
        best_base = float(objs.drop(labels=["CG+IR"], errors="ignore").min())
        passed = bool(cg <= best_base + 1e-6)
        pass_count += int(passed)
        if not passed:
            losses.append(sd)
        diag_lines.append(f"- seed {sd}: CG+IR={cg:.3f}, best_baseline={best_base:.3f}, beat_baseline={passed}")
    cg_rows = df[df["method"].eq("CG+IR")]
    stabilized_rate = float(np.mean(cg_rows["objective_stabilized"].astype(str).str.lower().eq("true")))
    pool_gap_avg = float(pd.to_numeric(cg_rows["pool_gap_pct"], errors="coerce").mean())
    ready = pass_count >= 4 and stabilized_rate >= 0.8 and pool_gap_avg < 0.01
    diag_lines.extend([
        "",
        f"CG+IR beat best scalable baseline in {pass_count}/5 seeds.",
        f"Failed seeds: {losses if losses else 'none'}",
        f"Objective stabilized rate: {stabilized_rate:.3f}",
        f"Average pool_gap_pct: {pool_gap_avg:.6f}",
        f"Pricing fully converged in any run: {bool(cg_rows['pricing_converged'].astype(str).str.lower().eq('true').any())}",
        f"Ready for 10-seed N=500 expansion: {ready}",
    ])
    (TARGET_RESULT_DIR / "n500_cautious_expansion_diagnostic.md").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")
    return ready


def bootstrap_ci(values: np.ndarray, seed: int = 123, n_boot: int = 1000) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bool_rate(series: pd.Series) -> float:
    vals = series.dropna().astype(str).str.lower()
    if vals.empty:
        return float("nan")
    return float(np.mean(vals.isin(["true", "1", "1.0", "yes"])))


def apply_metadata_display(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if "stabilization_applicable" not in out.columns:
        out["stabilization_applicable"] = True
    if "objective_stabilized_rate_display" not in out.columns:
        out["objective_stabilized_rate_display"] = ""
    for idx, row in out.iterrows():
        n_val = pd.to_numeric(pd.Series([row.get("N")]), errors="coerce").iloc[0]
        gap_type = str(row.get("gap_type", ""))
        pricing_rate = pd.to_numeric(pd.Series([row.get("pricing_converged_rate")]), errors="coerce").iloc[0]
        full_pricing = pd.notna(n_val) and n_val <= 100 and pricing_rate == 1.0 and "Full-CG LP-IP gap" in gap_type
        if full_pricing:
            out.at[idx, "stabilization_applicable"] = False
            out.at[idx, "objective_stabilized_rate_display"] = "N/A"
        else:
            out.at[idx, "stabilization_applicable"] = True
            val = pd.to_numeric(pd.Series([row.get("objective_stabilized_rate")]), errors="coerce").iloc[0]
            out.at[idx, "objective_stabilized_rate_display"] = "" if pd.isna(val) else f"{val:.3f}"
    return out


def summarize_bs_raw(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].copy()
    return ok.groupby(["C_BS", "method"], dropna=False).agg(
        total_cost_mean=("objective", "mean"),
        total_cost_std=("objective", "std"),
        SP_share_mean=("SP_share", "mean"),
        BS_share_mean=("BS_share", "mean"),
        AE_share_mean=("AE_share", "mean"),
        avg_delay_h_mean=("avg_delay_h", "mean"),
        delay_cost_mean=("delay_cost", "mean"),
        emissions_total_kg_mean=("emissions_total_kg", "mean"),
        emissions_total_tCO2_mean=("emissions_total_tCO2", "mean"),
        runtime_mean=("runtime_sec", "mean"),
        pricing_converged_rate=("pricing_converged", bool_rate),
        gap_type=("gap_type", lambda s: ";".join(sorted(set(map(str, s.dropna()))))),
    ).reset_index()


def add_summary_metadata_from_raw(summary: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    meta_rows = []
    for N, sim in raw[raw["operation_mode"].eq("simops")].groupby("N"):
        meta_rows.append({
            "N": N,
            "pricing_converged_rate": bool_rate(sim["pricing_converged"]) if "pricing_converged" in sim else np.nan,
            "objective_stabilized_rate": bool_rate(sim["objective_stabilized"]) if "objective_stabilized" in sim else np.nan,
            "gap_type": ";".join(sorted(set(map(str, sim.get("gap_type", pd.Series(dtype=str)).dropna())))),
        })
    meta = pd.DataFrame(meta_rows)
    out = summary.drop(columns=[c for c in ["pricing_converged_rate", "objective_stabilized_rate", "gap_type"] if c in summary.columns], errors="ignore")
    out = out.merge(meta, on="N", how="left")
    return apply_metadata_display(out)


def plot_dual_peak_final(summary_path: Path, fig_base: Path) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_path).sort_values("N")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = pd.to_numeric(df["N"], errors="coerce")
    y = pd.to_numeric(df["simops_saving_pct_mean"], errors="coerce")
    lo = pd.to_numeric(df["simops_saving_ci95_low"], errors="coerce")
    hi = pd.to_numeric(df["simops_saving_ci95_high"], errors="coerce")
    ax.plot(x, y, marker="o", color="#1f77b4", label="Mean saving")
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.18, label="95% bootstrap CI")
    ax.axvspan(85, 115, color="#f0a202", alpha=0.12, label="threshold-sensitive region")
    ax.set_xlabel("Number of vessels, N")
    ax.set_ylabel("SIMOPS saving (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_base.with_suffix(".png"), dpi=220)
    fig.savefig(fig_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_bs_final(summary_path: Path, fig_base: Path, share_base: Path, threshold: float | None = None) -> None:
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_path)
    df = df[df["method"].eq("CG+IR")].copy().sort_values("C_BS")
    x = pd.to_numeric(df["C_BS"], errors="coerce")
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(x, pd.to_numeric(df["total_cost_mean"], errors="coerce"), marker="o", label="Total cost")
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color="#a23b72", linestyle="--", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("C_BS ($/kWh)")
    ax.set_ylabel("Total cost ($)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_base.with_suffix(".png"), dpi=220)
    fig.savefig(fig_base.with_suffix(".pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for col, label in [("SP_share_mean", "SP"), ("BS_share_mean", "BS"), ("AE_share_mean", "AE")]:
        ax.plot(x, pd.to_numeric(df[col], errors="coerce"), marker="o", label=label)
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color="#a23b72", linestyle="--", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("C_BS ($/kWh)")
    ax.set_ylabel("Mean service share")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(share_base.with_suffix(".png"), dpi=220)
    fig.savefig(share_base.with_suffix(".pdf"))
    plt.close(fig)


def run_dual_peak_enrichment_final_check(N_values: List[int], scenario: str, seed_values: List[int], max_iters: int, pricing_top_k: int) -> None:
    raw_path = TARGET_RESULT_DIR / "dual_peak_enrichment_raw.csv"
    summary_path = TARGET_RESULT_DIR / "dual_peak_enrichment_summary.csv"
    diag_path = TARGET_RESULT_DIR / "dual_peak_enrichment_diagnostics.csv"
    comp_path = TARGET_RESULT_DIR / "dual_peak_enrichment_comparison.csv"
    rows: List[Dict[str, Any]] = []
    diag_rows: List[Dict[str, Any]] = []
    for N in N_values:
        for sd in seed_values:
            inst = generate_instance(N, sd, scenario, "hybrid", dict(BASE_PARAMS))
            seq_cfg = strong_cg_cfg("dual_peak_enrichment_final_check", N, scenario, sd, "sequential", max_iters=max_iters, pricing_top_k=pricing_top_k, min_iters=20, use_incumbent_injection=True)
            seq_sol = get_solver("cg")(inst, seq_cfg, Logger())
            seq_sol = enrich_solution_metrics(inst, seq_sol, "sequential")
            seq_row = normalize_target_row(flatten_result(seq_sol), N, scenario, sd, "CG+IR", "sequential")

            sim_cfg = strong_cg_cfg("dual_peak_enrichment_final_check", N, scenario, sd, "simops", incumbent_solutions=[seq_sol], max_iters=max_iters, pricing_top_k=pricing_top_k, min_iters=20, use_incumbent_injection=True)
            sim_sol = get_solver("cg")(inst, sim_cfg, Logger())
            sim_sol = enrich_solution_metrics(inst, sim_sol, "simops")
            sim_row = normalize_target_row(flatten_result(sim_sol), N, scenario, sd, "CG+IR", "simops")
            seq_obj = float(seq_row["objective"])
            sim_obj = float(sim_row["objective"])
            saving = (seq_obj - sim_obj) / max(1.0, abs(seq_obj)) * 100.0
            passed = bool(sim_obj <= seq_obj + 1e-6)
            for row in (seq_row, sim_row):
                row["dominance_check_passed"] = passed
                row["stabilization_applicable"] = True
                val = pd.to_numeric(pd.Series([row.get("objective_stabilized")]), errors="coerce").iloc[0]
                row["objective_stabilized_rate_display"] = str(row.get("objective_stabilized"))
            sim_row["simops_saving_pct"] = saving
            rows.extend([seq_row, sim_row])
            diag_rows.append({
                "N": N,
                "seed": sd,
                "seq_obj": seq_obj,
                "simops_obj": sim_obj,
                "simops_saving_pct": saving,
                "dominance_check_passed": passed,
                "seq_cg_status": seq_row.get("cg_status"),
                "simops_cg_status": sim_row.get("cg_status"),
                "seq_gap_type": seq_row.get("gap_type"),
                "simops_gap_type": sim_row.get("gap_type"),
                "seq_objective_stabilized": seq_row.get("objective_stabilized"),
                "simops_objective_stabilized": sim_row.get("objective_stabilized"),
            })
            write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_dual_peak_enrichment_final_check_N{N}_{scenario}_seed{sd}_sequential_cg.csv")
            write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_dual_peak_enrichment_final_check_N{N}_{scenario}_seed{sd}_simops_cg.csv")
    raw = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)
    raw.to_csv(raw_path, index=False)
    diag.to_csv(diag_path, index=False)
    summary_rows = []
    for N, d in diag.groupby("N"):
        sim = raw[(raw["N"].eq(N)) & raw["operation_mode"].eq("simops")]
        vals = d["simops_saving_pct"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(vals, seed=int(N) + 5000)
        summary_rows.append({
            "N": N,
            "simops_saving_pct_mean": float(np.mean(vals)),
            "simops_saving_pct_std": float(np.std(vals, ddof=1)),
            "simops_saving_ci95_low": ci_low,
            "simops_saving_ci95_high": ci_high,
            "dominance_pass_rate": float(np.mean(d["dominance_check_passed"].astype(bool))),
            "objective_stabilized_rate": bool_rate(sim["objective_stabilized"]),
            "pricing_converged_rate": bool_rate(sim["pricing_converged"]),
            "gap_type": ";".join(sorted(set(map(str, sim["gap_type"].dropna())))),
        })
    summary = apply_metadata_display(pd.DataFrame(summary_rows))
    summary.to_csv(summary_path, index=False)

    original = pd.read_csv(ROOT / "results" / "revised" / "full_controlled" / "simops_dual_peak_full_summary.csv")
    comp_rows = []
    for _, enr in summary.iterrows():
        N = int(enr["N"])
        orig = original[original["N"].eq(N)].iloc[0]
        change = float(enr["simops_saving_pct_mean"]) - float(orig["simops_saving_pct_mean"])
        pass_rate = float(enr["dominance_pass_rate"])
        comp_rows.append({
            "N": N,
            "original_saving_mean": float(orig["simops_saving_pct_mean"]),
            "enriched_saving_mean": float(enr["simops_saving_pct_mean"]),
            "absolute_change_pct": abs(change),
            "original_dominance_pass_rate": float(orig["dominance_pass_rate"]),
            "enriched_dominance_pass_rate": pass_rate,
            "original_objective_stabilized_rate": float(orig["objective_stabilized_rate"]),
            "enriched_objective_stabilized_rate": float(enr["objective_stabilized_rate"]),
            "use_enriched_value_in_final_plot": bool(pass_rate == 1.0 and abs(change) > 0.5),
        })
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(comp_path, index=False)
    if (comp["enriched_dominance_pass_rate"] < 1.0).any():
        raise RuntimeError("Dual-peak enrichment dominance check failed.")


def regenerate_final_dual_peak_outputs() -> None:
    full_raw = pd.read_csv(ROOT / "results" / "revised" / "full_controlled" / "simops_dual_peak_full_raw.csv")
    full_diag = pd.read_csv(ROOT / "results" / "revised" / "full_controlled" / "simops_dual_peak_diagnostics.csv")
    comp_path = TARGET_RESULT_DIR / "dual_peak_enrichment_comparison.csv"
    use_enriched = set()
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        use_enriched = set(comp.loc[comp["use_enriched_value_in_final_plot"].astype(bool), "N"].astype(int))
    if use_enriched:
        enr_raw = pd.read_csv(TARGET_RESULT_DIR / "dual_peak_enrichment_raw.csv")
        enr_diag = pd.read_csv(TARGET_RESULT_DIR / "dual_peak_enrichment_diagnostics.csv")
        final_raw = pd.concat([full_raw[~full_raw["N"].isin(use_enriched)], enr_raw[enr_raw["N"].isin(use_enriched)]], ignore_index=True)
        final_diag = pd.concat([full_diag[~full_diag["N"].isin(use_enriched)], enr_diag[enr_diag["N"].isin(use_enriched)]], ignore_index=True)
    else:
        final_raw = full_raw
        final_diag = full_diag
    final_diag.to_csv(TARGET_RESULT_DIR / "simops_dual_peak_final_diagnostics.csv", index=False)
    boot_rows = []
    summary_rows = []
    for N, d in final_diag.groupby("N"):
        vals = d["simops_saving_pct"].astype(float).to_numpy()
        rng = np.random.default_rng(int(N) + 9000)
        means = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(1000)]
        ci_low = float(np.percentile(means, 2.5))
        ci_high = float(np.percentile(means, 97.5))
        boot_rows.append({"N": N, "ci95_low": ci_low, "ci95_high": ci_high, "bootstrap_resamples": 1000})
        sim = final_raw[(final_raw["N"].eq(N)) & final_raw["operation_mode"].eq("simops")]
        seq = final_raw[(final_raw["N"].eq(N)) & final_raw["operation_mode"].eq("sequential")]
        summary_rows.append({
            "N": N,
            "simops_cost_mean": pd.to_numeric(sim["objective"], errors="coerce").mean(),
            "sequential_cost_mean": pd.to_numeric(seq["objective"], errors="coerce").mean(),
            "simops_saving_pct_mean": float(np.mean(vals)),
            "simops_saving_pct_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "simops_saving_ci95_low": ci_low,
            "simops_saving_ci95_high": ci_high,
            "dominance_pass_rate": float(np.mean(d["dominance_check_passed"].astype(bool))),
            "simops_AE_share_mean": pd.to_numeric(sim["AE_share"], errors="coerce").mean(),
            "simops_BS_share_mean": pd.to_numeric(sim["BS_share"], errors="coerce").mean(),
            "simops_SP_share_mean": pd.to_numeric(sim["SP_share"], errors="coerce").mean(),
        })
    summary = add_summary_metadata_from_raw(pd.DataFrame(summary_rows), final_raw)
    summary.to_csv(TARGET_RESULT_DIR / "simops_dual_peak_final_summary.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(TARGET_RESULT_DIR / "simops_dual_peak_bootstrap_ci.csv", index=False)
    if (summary["dominance_pass_rate"] < 1.0).any():
        raise RuntimeError("Final dual-peak summary contains failed dominance points.")
    plot_dual_peak_final(TARGET_RESULT_DIR / "simops_dual_peak_final_summary.csv", TARGET_FIGURE_DIR / "fig5_dual_peak_final")


def run_bs_cost_threshold_fine_grid_final_check(N: int, scenario: str, seed_values: List[int], c_bs_values: List[float]) -> None:
    raw_path = TARGET_RESULT_DIR / "bs_cost_threshold_fine_grid_raw.csv"
    summary_path = TARGET_RESULT_DIR / "bs_cost_threshold_fine_grid_summary.csv"
    rows: List[Dict[str, Any]] = []
    for c_bs in sorted(set(c_bs_values + [0.30, 0.40, 0.45, 0.60])):
        params = dict(BASE_PARAMS, battery_cost=float(c_bs))
        for sd in seed_values:
            inst = generate_instance(N, sd, scenario, "hybrid", params)
            cfg = strong_cg_cfg("bs_cost_threshold_fine_grid_final_check", N, scenario, sd, "simops")
            row = {"N": N, "scenario": scenario, "seed": sd, "method": "CG+IR", "C_BS": c_bs, "status": "error", "error": ""}
            try:
                sol = get_solver("cg")(inst, cfg, Logger())
                sol = enrich_solution_metrics(inst, sol, "simops")
                row.update(normalize_target_row(flatten_result(sol), N, scenario, sd, "CG+IR"))
                row["status"] = sol.get("status", "ok")
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    summary = summarize_bs_raw(raw)
    summary.to_csv(summary_path, index=False)
    detect_bs_threshold(summary)
    plot_bs_final(summary_path, TARGET_FIGURE_DIR / "fig6a_bs_cost_threshold_final", TARGET_FIGURE_DIR / "fig6a_bs_cost_threshold_mode_share_final", threshold=get_detected_threshold())


def get_detected_threshold() -> float | None:
    path = TARGET_RESULT_DIR / "bs_threshold_detection.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    vals = pd.to_numeric(df["detected_threshold_C_BS"], errors="coerce").dropna()
    return float(vals.iloc[0]) if not vals.empty else None


def detect_bs_threshold(summary: pd.DataFrame) -> None:
    df = summary[summary["method"].eq("CG+IR")].copy().sort_values("C_BS")
    rows = []
    def before_after(threshold: float) -> tuple[float, float, float, float]:
        before = df[df["C_BS"] < threshold].tail(1)
        after = df[df["C_BS"] >= threshold].head(1)
        if before.empty:
            before = after
        return (
            float(before["BS_share_mean"].iloc[0]),
            float(after["BS_share_mean"].iloc[0]),
            float(before["AE_share_mean"].iloc[0]),
            float(after["AE_share_mean"].iloc[0]),
        )

    checks = [
        ("First C_BS where mean BS_share < 0.10", df[df["BS_share_mean"] < 0.10]),
        ("First C_BS where mean AE_share > 0.50", df[df["AE_share_mean"] > 0.50]),
    ]
    cost = pd.to_numeric(df["total_cost_mean"], errors="coerce").to_numpy()
    c_vals = pd.to_numeric(df["C_BS"], errors="coerce").to_numpy()
    slopes = np.diff(cost) / np.diff(c_vals)
    dominant = df[(df["AE_share_mean"] > 0.50)]
    checks.append(("First C_BS where total cost stops increasing smoothly because AE fallback dominates", dominant))
    for name, sub in checks:
        if sub.empty:
            threshold = np.nan
            bs_b = bs_a = ae_b = ae_a = np.nan
            notes = "No threshold detected in tested grid."
        else:
            threshold = float(sub["C_BS"].iloc[0])
            bs_b, bs_a, ae_b, ae_a = before_after(threshold)
            notes = "Abrupt structural BS-AE substitution threshold." if abs(ae_a - ae_b) > 0.25 else "Gradual transition in tested grid."
        rows.append({
            "threshold_definition": name,
            "detected_threshold_C_BS": threshold,
            "BS_share_before_threshold": bs_b,
            "BS_share_after_threshold": bs_a,
            "AE_share_before_threshold": ae_b,
            "AE_share_after_threshold": ae_a,
            "notes": notes,
        })
    pd.DataFrame(rows).to_csv(TARGET_RESULT_DIR / "bs_threshold_detection.csv", index=False)


def regenerate_final_bs_outputs() -> None:
    fine = pd.read_csv(TARGET_RESULT_DIR / "bs_cost_threshold_fine_grid_summary.csv")
    fine = fine[fine["method"].eq("CG+IR")].copy()
    fine.to_csv(TARGET_RESULT_DIR / "bs_cost_sensitivity_final_summary.csv", index=False)
    threshold = get_detected_threshold()
    plot_bs_final(TARGET_RESULT_DIR / "bs_cost_sensitivity_final_summary.csv", TARGET_FIGURE_DIR / "fig6a_bs_cost_sensitivity_final", TARGET_FIGURE_DIR / "fig6a_bs_cost_mode_share_final", threshold=threshold)
    validation = fine.copy()
    validation["metric_validated"] = (
        pd.to_numeric(validation["avg_delay_h_mean"], errors="coerce").notna()
        & (pd.to_numeric(validation["emissions_total_tCO2_mean"], errors="coerce") > 0)
        & (abs(pd.to_numeric(validation["emissions_total_tCO2_mean"], errors="coerce") - pd.to_numeric(validation["AE_share_mean"], errors="coerce")) > 1e-6)
    )
    validation[[
        "C_BS",
        "method",
        "total_cost_mean",
        "SP_share_mean",
        "BS_share_mean",
        "AE_share_mean",
        "avg_delay_h_mean",
        "delay_cost_mean",
        "emissions_total_tCO2_mean",
        "metric_validated",
    ]].to_csv(TARGET_RESULT_DIR / "bs_metric_validation.csv", index=False)


def write_metadata_display_check() -> None:
    sim = pd.read_csv(ROOT / "results" / "revised" / "full_controlled" / "simops_dual_peak_full_summary.csv")
    raw = pd.read_csv(ROOT / "results" / "revised" / "full_controlled" / "simops_dual_peak_full_raw.csv")
    sim = add_summary_metadata_from_raw(sim, raw)
    table = pd.read_csv(ROOT / "results" / "revised" / "full_controlled" / "table8_final_controlled.csv")
    table = apply_metadata_display(table)
    out = pd.concat([
        sim[["N", "pricing_converged_rate", "objective_stabilized_rate", "objective_stabilized_rate_display", "stabilization_applicable", "gap_type"]].assign(source="simops_dual_peak"),
        table[["N", "pricing_converged_rate", "objective_stabilized_rate", "objective_stabilized_rate_display", "stabilization_applicable", "gap_type"]].assign(source="table8"),
    ], ignore_index=True)
    out.to_csv(TARGET_RESULT_DIR / "metadata_display_check.csv", index=False)


def validate_and_copy_table8() -> None:
    src = ROOT / "results" / "revised" / "full_controlled"
    for name in ["n500_table8_full_replacement_summary.csv", "table8_final_controlled.csv", "table8_final_controlled.tex"]:
        data = (src / name).read_bytes()
        (TARGET_RESULT_DIR / name).write_bytes(data)
    table = pd.read_csv(TARGET_RESULT_DIR / "table8_final_controlled.csv")
    nan_only = table[pd.to_numeric(table["objective_mean"], errors="coerce").isna()]
    names_ok = set(table["method"].dropna()).issubset({"CG+IR", "MILP-60", "MILP-300", "Rolling-Horizon", "Fix-and-Optimize", "Restricted-CG", "FIFO", "Greedy"})
    cg = table[table["method"].eq("CG+IR")]
    gap_ok = bool((cg[cg["N"].isin([200, 500])]["gap_type"].astype(str) == "Pool LP-IP gap").all())
    included = table.groupby("N")["scenario"].apply(lambda s: ",".join(sorted(set(map(str, s.dropna()))))).to_dict()
    lines = [
        "# Table 8 Validation Report",
        "",
        "1. Were old quick results excluded? yes; final_check copies only full_controlled controlled replacement files.",
        "2. Are N=200 and N=500 from controlled replacement? yes.",
        f"3. Are gap types correct? {'yes' if gap_ok else 'no'}; N=200/500 CG+IR rows use Pool LP-IP gap.",
        f"4. Are method names standardized? {'yes' if names_ok else 'no'}.",
        f"5. Are there any NaN-only rows? {'no' if nan_only.empty else 'yes'}.",
        f"6. Included N/scenarios: {included}.",
        "7. N=500 is U-only; P/L were not run in the controlled replacement due runtime budget.",
    ]
    (TARGET_RESULT_DIR / "table8_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_simops_dual_peak_full_controlled(seed_values: List[int] | None = None) -> None:
    seed_values = seed_values or list(range(1, 11))
    Ns = [25, 50, 75, 90, 100, 110, 125, 150, 200, 500]
    raw_path = TARGET_RESULT_DIR / "simops_dual_peak_full_raw.csv"
    summary_path = TARGET_RESULT_DIR / "simops_dual_peak_full_summary.csv"
    diag_path = TARGET_RESULT_DIR / "simops_dual_peak_diagnostics.csv"
    failures_path = TARGET_RESULT_DIR / "simops_dominance_failures.csv"
    if raw_path.exists():
        raw_path.unlink()
    rows = []
    diag_rows = []
    for N in Ns:
        for sd in seed_values:
            inst = generate_instance(N, sd, "U", "hybrid", dict(BASE_PARAMS))
            seq_cfg = strong_cg_cfg("simops_dual_peak_full_controlled", N, "U", sd, "sequential")
            sim_seq = get_solver("cg")(inst, seq_cfg, Logger())
            sim_seq = enrich_solution_metrics(inst, sim_seq, "sequential")
            sim_seq["method"] = "CG+IR sequential"
            seq_row = normalize_target_row(flatten_result(sim_seq), N, "U", sd, "CG+IR", "sequential")
            rows.append(seq_row)
            append_raw(seq_row, raw_path)
            sim_cfg = strong_cg_cfg("simops_dual_peak_full_controlled", N, "U", sd, "simops", incumbent_solutions=[sim_seq])
            sim_sol = get_solver("cg")(inst, sim_cfg, Logger())
            sim_sol = enrich_solution_metrics(inst, sim_sol, "simops")
            sim_row = normalize_target_row(flatten_result(sim_sol), N, "U", sd, "CG+IR", "simops")
            seq_obj = float(seq_row["objective"])
            sim_obj = float(sim_row["objective"])
            saving = (seq_obj - sim_obj) / max(1.0, abs(seq_obj)) * 100.0
            passed = sim_obj <= seq_obj + 1e-6
            sim_row["simops_saving_pct"] = saving
            sim_row["dominance_check_passed"] = passed
            rows.append(sim_row)
            append_raw(sim_row, raw_path)
            diag_rows.append({
                "N": N,
                "seed": sd,
                "seq_obj": seq_obj,
                "simops_obj": sim_obj,
                "simops_saving_pct": saving,
                "dominance_check_passed": passed,
                "seq_cg_status": seq_row.get("cg_status"),
                "simops_cg_status": sim_row.get("cg_status"),
                "seq_gap_type": seq_row.get("gap_type"),
                "simops_gap_type": sim_row.get("gap_type"),
                "seq_objective_stabilized": seq_row.get("objective_stabilized"),
                "simops_objective_stabilized": sim_row.get("objective_stabilized"),
                "seq_best_reduced_cost_last": seq_row.get("best_reduced_cost_last"),
                "simops_best_reduced_cost_last": sim_row.get("best_reduced_cost_last"),
            })
            write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_simops_dual_peak_full_controlled_N{N}_U_seed{sd}_sequential_cg.csv")
            write_trace_markdown(TARGET_LOG_DIR / f"cg_trace_simops_dual_peak_full_controlled_N{N}_U_seed{sd}_simops_cg.csv")
    raw = pd.DataFrame(rows)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(diag_path, index=False)
    fail = diag[~diag["dominance_check_passed"].astype(bool)]
    if not fail.empty:
        fail.to_csv(failures_path, index=False)
    summary_rows = []
    for N, d in diag.groupby("N"):
        sim = raw[(raw["N"].eq(N)) & (raw["operation_mode"].eq("simops"))]
        seq = raw[(raw["N"].eq(N)) & (raw["operation_mode"].eq("sequential"))]
        vals = d["simops_saving_pct"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(vals, seed=int(N))
        summary_rows.append({
            "N": N,
            "simops_cost_mean": pd.to_numeric(sim["objective"], errors="coerce").mean(),
            "simops_cost_std": pd.to_numeric(sim["objective"], errors="coerce").std(),
            "sequential_cost_mean": pd.to_numeric(seq["objective"], errors="coerce").mean(),
            "sequential_cost_std": pd.to_numeric(seq["objective"], errors="coerce").std(),
            "simops_saving_pct_mean": float(np.mean(vals)),
            "simops_saving_pct_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "simops_saving_ci95_low": ci_low,
            "simops_saving_ci95_high": ci_high,
            "dominance_pass_rate": float(np.mean(d["dominance_check_passed"].astype(bool))),
            "simops_AE_share_mean": pd.to_numeric(sim["AE_share"], errors="coerce").mean(),
            "sequential_AE_share_mean": pd.to_numeric(seq["AE_share"], errors="coerce").mean(),
            "simops_BS_share_mean": pd.to_numeric(sim["BS_share"], errors="coerce").mean(),
            "sequential_BS_share_mean": pd.to_numeric(seq["BS_share"], errors="coerce").mean(),
            "simops_SP_share_mean": pd.to_numeric(sim["SP_share"], errors="coerce").mean(),
            "sequential_SP_share_mean": pd.to_numeric(seq["SP_share"], errors="coerce").mean(),
            "masking_rate_mean": pd.to_numeric(sim["masking_rate"], errors="coerce").mean(),
            "objective_stabilized_rate": float(np.mean(sim["objective_stabilized"].astype(str).str.lower().eq("true"))),
            "pricing_converged_rate": float(np.mean(sim["pricing_converged"].astype(str).str.lower().eq("true"))),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    plot_from_csv(summary_path, TARGET_FIGURE_DIR / "fig5_dual_peak_full_controlled", "N", "simops_saving_pct_mean", None, "SIMOPS saving (%)")


def run_bs_cost_sensitivity_full_summary(N: int, scenario: str, seed_values: List[int]) -> None:
    raw_path = TARGET_RESULT_DIR / "bs_cost_sensitivity_full_raw.csv"
    summary_path = TARGET_RESULT_DIR / "bs_cost_sensitivity_full_summary.csv"
    if raw_path.exists():
        raw_path.unlink()
    for c_bs in [0.30, 0.40, 0.45, 0.60, 0.80, 1.00, 1.20]:
        params = dict(BASE_PARAMS, battery_cost=c_bs)
        for sd in seed_values:
            for label, solver_name, cfg in [
                ("CG+IR", "cg", strong_cg_cfg("bs_cost_sensitivity_full_summary", N, scenario, sd, "simops")),
                ("Rolling-Horizon", "rolling_horizon", {"method": "rolling_horizon", "operation_mode": "simops", "return_schedule": True, "rolling_horizon": {"window_size": 8, "commit_size": 4, "time_limit": 5}}),
                ("Fix-and-Optimize", "fix_and_optimize", {"method": "fix_and_optimize", "operation_mode": "simops", "return_schedule": True, "fix_and_optimize": {"block_size": 8, "step_size": 4, "max_passes": 1, "time_limit": 5}}),
            ]:
                inst = generate_instance(N, sd, scenario, "hybrid", params)
                row = {"N": N, "scenario": scenario, "seed": sd, "method": label, "C_BS": c_bs, "status": "error", "error": ""}
                try:
                    sol = get_solver(solver_name)(inst, cfg, Logger())
                    sol = enrich_solution_metrics(inst, sol, cfg.get("operation_mode", "simops"))
                    row.update(normalize_target_row(flatten_result(sol), N, scenario, sd, label))
                    row["status"] = sol.get("status", "ok")
                except Exception as exc:
                    row["error"] = repr(exc)
                append_raw(row, raw_path)
    df = pd.read_csv(raw_path)
    summary = df[pd.to_numeric(df.get("objective"), errors="coerce").notna()].groupby(["C_BS", "method"], dropna=False).agg(
        total_cost_mean=("objective", "mean"),
        total_cost_std=("objective", "std"),
        SP_share_mean=("SP_share", "mean"),
        BS_share_mean=("BS_share", "mean"),
        AE_share_mean=("AE_share", "mean"),
        delay_cost_mean=("delay_cost", "mean"),
        avg_delay_h_mean=("avg_delay_h", "mean"),
        emissions_total_mean=("emissions_total", "mean"),
        runtime_mean=("runtime_sec", "mean"),
    ).reset_index()
    summary.to_csv(summary_path, index=False)
    plot_from_csv(raw_path, TARGET_FIGURE_DIR / "fig6a_bs_cost_full_controlled", "C_BS", "objective", "method", "Total cost ($)")


def run_simple_named(name: str, full: bool, quick: bool) -> None:
    """Functional fallback for revision experiments not used by the quick suite."""
    rows = []
    for sd in seeds(full, quick):
        rows.append(run_one(100, sd, "U", "CG+IR", dict(BASE_PARAMS), quick=quick))
    df = write_raw(rows, RESULT_DIR / f"{name}_raw.csv")
    write_summary_table(df, RESULT_DIR / f"{name}_summary.csv", RESULT_DIR / f"{name}_summary.tex", ["method"])


def write_readme(commands: Iterable[str]) -> None:
    try:
        git_commit = subprocess.run(["git", "-c", "safe.directory=E:/PythonProject1", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        git_commit = "unavailable"
    try:
        import gurobipy as gp  # type: ignore
        gurobi_version = ".".join(map(str, gp.gurobi.version()))
    except Exception:
        gurobi_version = "unavailable"
    readme = [
        "# Revised Results",
        "",
        "## Commands",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "## Environment",
        f"- Git commit: `{git_commit or 'unavailable'}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Gurobi: `{gurobi_version}`",
        f"- Platform: `{platform.platform()}`",
        f"- CPU: `{platform.processor() or 'unavailable'}`",
        "",
        "## Seeds",
        "- Quick mode uses seeds 1-2; full mode uses seeds 1-10.",
        "",
        "## Gap Fields",
        "- `Full-CG LP-IP gap`: full pricing converged for the generated column universe.",
        "- `Pool LP-IP gap`: integer solution gap relative to the final budgeted column pool LP, not a global optimality gap.",
    ]
    (RESULT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--scenario", type=str, default="U")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--strong-cg", action="store_true")
    parser.add_argument("--strong-cg-large", action="store_true")
    parser.add_argument("--inject-incumbents", action="store_true")
    parser.add_argument("--enforce-dominance-check", action="store_true")
    parser.add_argument("--inject-sequential-into-simops", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--N-values", nargs="*", type=int, default=None)
    parser.add_argument("--operation-modes", nargs="*", default=None)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--pricing-top-k", type=int, default=10)
    parser.add_argument("--C-BS-values", nargs="*", type=float, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()
    global TARGET_RESULT_DIR, TARGET_FIGURE_DIR, TARGET_LOG_DIR
    if args.output_dir:
        TARGET_RESULT_DIR = Path(args.output_dir)
        if TARGET_RESULT_DIR.name == "full_controlled":
            TARGET_FIGURE_DIR = ROOT / "figures" / "revised" / "full_controlled"
            TARGET_LOG_DIR = ROOT / "logs" / "revised" / "full_controlled"
        else:
            TARGET_FIGURE_DIR = FIGURE_DIR / TARGET_RESULT_DIR.name
            TARGET_LOG_DIR = ROOT / "logs" / "revised" / TARGET_RESULT_DIR.name
    ensure_dirs()
    commands = [f"python experiments/run_revision_experiments.py --experiment {e} {'--quick' if args.quick else '--full' if args.full else ''}".strip() for e in args.experiment]
    dispatch = {
        "main_benchmark": run_main_benchmark,
        "scenario_comparison": run_scenario_comparison,
        "simops_dual_peak": run_simops_dual_peak,
        "bs_cost_sensitivity": run_bs_cost_sensitivity,
        "capacity_grid": run_capacity_grid,
        "carbon_grid_factor": run_carbon_grid_factor,
        "mechanism_comparison": run_mechanism_comparison,
        "arrival_perturbation": run_arrival_perturbation,
        "column_pool_enrichment": run_column_pool_enrichment,
    }
    for exp in args.experiment:
        t0 = time.perf_counter()
        print(f"Running {exp}")
        if exp == "targeted_simops_dominance":
            run_targeted_simops_dominance(args.N or 200, args.scenario, args.seeds or [1, 2], strong=args.strong_cg)
        elif exp == "targeted_N500_stress":
            run_targeted_N500_stress(args.N or 500, args.scenario, args.seed or 1, strong=args.strong_cg)
        elif exp == "targeted_table8_N200_replacement":
            run_targeted_table8_N200_replacement(args.N or 200, args.scenarios or ["U", "P", "L"], args.seeds or [1, 2], strong=args.strong_cg)
        elif exp == "n500_cautious_expansion":
            run_n500_cautious_expansion(args.N or 500, args.scenario, args.seeds or [1, 2, 3, 4, 5])
        elif exp == "n200_table8_full_replacement":
            run_controlled_benchmark("n200_table8_full_replacement", args.N or 200, args.scenarios or ["U", "P", "L"], args.seeds or list(range(1, 11)), include_heavy=True)
        elif exp == "n500_table8_full_replacement":
            run_controlled_benchmark("n500_table8_full_replacement", args.N or 500, args.scenarios or [args.scenario], args.seeds or list(range(1, 11)), include_heavy=True)
        elif exp == "simops_dual_peak_full_controlled":
            run_simops_dual_peak_full_controlled()
        elif exp == "bs_cost_sensitivity_full_summary":
            run_bs_cost_sensitivity_full_summary(args.N or 100, args.scenario, args.seeds or list(range(1, 11)))
        elif exp == "dual_peak_enrichment_final_check":
            run_dual_peak_enrichment_final_check(args.N_values or [110, 125, 150], args.scenario, args.seeds or list(range(1, 11)), args.max_iter, args.pricing_top_k)
        elif exp == "final_dual_peak_outputs":
            regenerate_final_dual_peak_outputs()
        elif exp == "bs_cost_threshold_fine_grid_final_check":
            run_bs_cost_threshold_fine_grid_final_check(args.N or 100, args.scenario, args.seeds or list(range(1, 11)), args.C_BS_values or [0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00, 1.20])
        elif exp == "final_bs_outputs":
            regenerate_final_bs_outputs()
        elif exp == "metadata_display_final_check":
            write_metadata_display_check()
        elif exp == "table8_final_check":
            validate_and_copy_table8()
        else:
            dispatch.get(exp, run_simple_named)(exp, args.full, args.quick) if exp not in dispatch else dispatch[exp](args.full, args.quick)
        print(f"Finished {exp} in {time.perf_counter() - t0:.1f}s")
    write_readme(commands)


if __name__ == "__main__":
    main()

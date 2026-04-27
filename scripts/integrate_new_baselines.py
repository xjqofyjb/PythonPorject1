"""Run and integrate new baseline results into the main experiment pipeline.

This script:
1. Runs new baselines (rolling_horizon, fix_and_optimize, restricted_cg) on the
   same main-experiment instances as the current framework.
2. Appends normalized result rows into the existing main results CSV schema.
3. Builds grouped summary statistics by (N, method).
4. Generates a booktabs-style LaTeX table for the paper.
5. Writes a short note describing where to update the manuscript discussion.

Usage example:
    python -m scripts.integrate_new_baselines \
        --config configs/main.yaml \
        --output_csv results/results_main_rigorous.csv \
        --summary_csv results/main_summary_with_new_baselines.csv \
        --latex_out results/main_table_with_new_baselines.tex \
        --note_out deliverables/new_baselines_paper_note_cn.txt
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from src.instances import generate_instance
from src.io import append_results, get_logger
from src.metrics import compute_cost_components, compute_mechanism_metrics, compute_mode_ratios
from src.runner import build_cg_cfg, expand_n_list, expand_seeds, load_config
from src.solvers import fifo_solver, greedy_solver, milp_solver, rolling_horizon_solver


COMMON_FIELDS = [
    "experiment",
    "scenario",
    "mechanism",
    "N",
    "seed",
    "method",
    "param_name",
    "param_value",
    "obj",
    "runtime_total",
    "runtime_pricing",
    "gap",
    "gap_pct",
    "status",
    "error",
    "num_iters",
    "num_pricing_calls",
    "num_columns_added",
    "min_reduced_cost_last",
    "pricing_time_share",
    "cost_energy",
    "cost_delay",
    "shore_utilization",
    "mode_shore_count",
    "mode_battery_count",
    "mode_brown_count",
    "infeasible_jobs",
    "shore_ratio",
    "battery_ratio",
    "brown_ratio",
    "mechanism_counts",
]


def _base_row() -> Dict[str, Any]:
    row = {field: np.nan for field in COMMON_FIELDS}
    row["status"] = "error"
    row["error"] = ""
    row["infeasible_jobs"] = 0
    return row


def _maybe_import_fix_and_optimize():
    try:
        return importlib.import_module("src.solvers.fix_and_optimize_solver")
    except Exception:
        return None


def _normalize_status(sol: Dict[str, Any]) -> str:
    if "status" in sol and pd.notna(sol["status"]):
        return str(sol["status"])
    if "success" in sol:
        return "ok" if bool(sol["success"]) else "error"
    return "ok"


def _logical_to_internal_method(logical_method: str, restricted_variant: str) -> str:
    if logical_method == "restricted_cg":
        return restricted_variant
    return logical_method


def _method_solver_registry(restricted_variant: str) -> Dict[str, Dict[str, Any]]:
    reg: Dict[str, Dict[str, Any]] = {
        "rolling_horizon": {
            "solver": rolling_horizon_solver.solve,
            "internal_method": "rolling_horizon",
        },
        "restricted_cg": {
            "solver": importlib.import_module("src.solvers.cg_solver").solve,
            "internal_method": restricted_variant,
        },
        "fifo": {"solver": fifo_solver.solve, "internal_method": "fifo"},
        "greedy": {"solver": greedy_solver.solve, "internal_method": "greedy"},
        "milp300": {"solver": milp_solver.solve, "internal_method": "milp300"},
        "cg": {"solver": importlib.import_module("src.solvers.cg_solver").solve, "internal_method": "cg"},
    }
    fao_mod = _maybe_import_fix_and_optimize()
    if fao_mod is not None and hasattr(fao_mod, "solve"):
        reg["fix_and_optimize"] = {"solver": fao_mod.solve, "internal_method": "fix_and_optimize"}
    return reg


def _build_method_cfg(config: Dict[str, Any], logical_method: str, internal_method: str, n_value: int) -> Dict[str, Any]:
    method_cfg: Dict[str, Any] = {"method": internal_method}
    if logical_method == "restricted_cg":
        method_cfg.update(build_cg_cfg(config, internal_method))
    elif internal_method.startswith("cg"):
        method_cfg.update(build_cg_cfg(config, internal_method))
    elif logical_method == "rolling_horizon":
        rh_cfg = dict(config.get("rolling_horizon", {}))
        rh_cfg.setdefault("window_size", 10)
        rh_cfg.setdefault("commit_size", max(1, rh_cfg["window_size"] // 2))
        rh_cfg.setdefault("time_limit", 30)
        method_cfg["rolling_horizon"] = rh_cfg
    elif logical_method == "fix_and_optimize":
        fao_cfg = dict(config.get("fix_and_optimize", {}))
        if n_value > 100:
            fao_cfg.setdefault("time_limit", 30)
        method_cfg["fix_and_optimize"] = fao_cfg
    return method_cfg


def _ensure_mode_fields(row: Dict[str, Any], sol: Dict[str, Any], instance) -> None:
    if "mechanism_counts" in sol:
        counts = sol["mechanism_counts"]
        row["mode_shore_count"] = counts.get("shore", 0)
        row["mode_battery_count"] = counts.get("battery", 0)
        row["mode_brown_count"] = counts.get("brown", 0)
        row.update(compute_mode_ratios(counts, instance.N))


def _ensure_cost_fields(row: Dict[str, Any], sol: Dict[str, Any], instance) -> None:
    if "cost_energy" in sol:
        row["cost_energy"] = sol.get("cost_energy", np.nan)
        row["cost_delay"] = sol.get("cost_delay", np.nan)
    else:
        row.update(compute_cost_components(instance, row["obj"]))


def _normalize_result_row(
    instance,
    experiment: str,
    scenario: str,
    mechanism: str,
    n_value: int,
    seed: int,
    logical_method: str,
    sol: Dict[str, Any],
) -> Dict[str, Any]:
    row = _base_row()
    row.update(
        {
            "experiment": experiment,
            "scenario": scenario,
            "mechanism": mechanism,
            "N": n_value,
            "seed": seed,
            "method": logical_method,
            "param_name": "",
            "param_value": np.nan,
        }
    )
    row.update(sol)
    row["status"] = _normalize_status(sol)
    _ensure_mode_fields(row, sol, instance)
    _ensure_cost_fields(row, sol, instance)
    row.update(compute_mechanism_metrics(instance))
    return row


def _remove_existing_method_rows(csv_path: str, methods: Iterable[str]) -> None:
    path = Path(csv_path)
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[~df["method"].isin(list(methods))].copy()
    df.to_csv(path, index=False)


def run_new_baselines(
    config: Dict[str, Any],
    output_csv: str,
    logical_methods: List[str],
    restricted_variant: str,
    overwrite: bool = True,
) -> Dict[str, Any]:
    logger = get_logger(config.get("log_dir", "results/logs"), name="integrate_new_baselines")
    registry = _method_solver_registry(restricted_variant)
    unavailable = [m for m in logical_methods if m not in registry]
    available = [m for m in logical_methods if m in registry]

    if overwrite and available:
        _remove_existing_method_rows(output_csv, available)

    seeds = expand_seeds(config.get("seeds", 10))
    n_list = expand_n_list(config.get("N_list", [20, 50, 100, 200, 500]))
    scenarios = config.get("scenarios", ["U"])
    mechanism = config.get("mechanism", "hybrid")
    experiment = config.get("experiment", "main")
    params = config.get("params", {})

    rows: List[Dict[str, Any]] = []
    for n_value in n_list:
        for seed in seeds:
            for scenario in scenarios:
                instance = generate_instance(n_value, seed, scenario, mechanism, params)
                for logical_method in available:
                    try:
                        solver_fn = registry[logical_method]["solver"]
                        internal_method = registry[logical_method]["internal_method"]
                        method_cfg = _build_method_cfg(config, logical_method, internal_method, n_value)
                        method_cfg.setdefault("operation_mode", "simops")
                        sol = solver_fn(instance, method_cfg, logger)
                        row = _normalize_result_row(
                            instance,
                            experiment,
                            scenario,
                            mechanism,
                            n_value,
                            seed,
                            logical_method,
                            sol,
                        )
                    except Exception as exc:  # pragma: no cover - keep run robust
                        row = _base_row()
                        row.update(
                            {
                                "experiment": experiment,
                                "scenario": scenario,
                                "mechanism": mechanism,
                                "N": n_value,
                                "seed": seed,
                                "method": logical_method,
                                "param_name": "",
                                "param_value": np.nan,
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                    rows.append(row)

    append_results(output_csv, rows)
    return {
        "available_methods": available,
        "unavailable_methods": unavailable,
        "rows_written": len(rows),
    }


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["N", "method"], dropna=False)
    rows = []
    for (n_value, method), g in grouped:
        success_rate = (g["status"] == "ok").mean() if len(g) else 0.0
        rows.append(
            {
                "N": int(n_value),
                "method": method,
                "obj_mean": float(g["obj"].mean()) if "obj" in g else np.nan,
                "obj_std": float(g["obj"].std(ddof=0)) if "obj" in g else np.nan,
                "runtime_mean": float(g["runtime_total"].mean()) if "runtime_total" in g else np.nan,
                "runtime_std": float(g["runtime_total"].std(ddof=0)) if "runtime_total" in g else np.nan,
                "success_rate": float(success_rate),
                "gap_pct_mean": float(g["gap_pct"].mean()) if "gap_pct" in g else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["N", "method"]).reset_index(drop=True)


def _format_cell(obj_mean: float, obj_std: float, runtime_mean: float) -> str:
    if any(pd.isna(x) for x in [obj_mean, obj_std, runtime_mean]):
        return "--"
    return f"{obj_mean:.2f} $\\pm$ {obj_std:.2f} / {runtime_mean:.2f}s"


def _format_gap(gap_pct_mean: float) -> str:
    if pd.isna(gap_pct_mean):
        return "--"
    return f"{gap_pct_mean:.2f}"


def generate_latex_table(summary: pd.DataFrame, latex_out: str) -> None:
    row_order = [20, 50, 100, 200, 500]
    column_methods = [
        ("cg", "CG"),
        ("milp300", "MILP300"),
        ("rolling_horizon", "Rolling-H(W=10)"),
        ("fix_and_optimize", "F\\&O"),
        ("restricted_cg", "Restricted-CG"),
        ("fifo", "FIFO"),
        ("greedy", "Greedy"),
    ]

    pivot = {(int(row["N"]), str(row["method"])): row for _, row in summary.iterrows()}

    lines: List[str] = []
    lines.append("\\begin{tabular}{rllllllll}")
    lines.append("\\toprule")
    lines.append("N & CG & Gap(\\%) & MILP300 & Rolling-H(W=10) & F\\&O & Restricted-CG & FIFO & Greedy \\\\")
    lines.append("\\midrule")

    for n_value in row_order:
        cg_row = pivot.get((n_value, "cg"))
        milp_row = pivot.get((n_value, "milp300"))
        rh_row = pivot.get((n_value, "rolling_horizon"))
        fao_row = pivot.get((n_value, "fix_and_optimize"))
        rcg_row = pivot.get((n_value, "restricted_cg"))
        fifo_row = pivot.get((n_value, "fifo"))
        greedy_row = pivot.get((n_value, "greedy"))

        milp_cell = "--" if n_value > 100 else _format_cell(
            milp_row["obj_mean"], milp_row["obj_std"], milp_row["runtime_mean"]
        ) if milp_row is not None else "--"

        row_cells = [
            str(n_value),
            _format_cell(cg_row["obj_mean"], cg_row["obj_std"], cg_row["runtime_mean"]) if cg_row is not None else "--",
            _format_gap(cg_row["gap_pct_mean"]) if cg_row is not None else "--",
            milp_cell,
            _format_cell(rh_row["obj_mean"], rh_row["obj_std"], rh_row["runtime_mean"]) if rh_row is not None else "--",
            _format_cell(fao_row["obj_mean"], fao_row["obj_std"], fao_row["runtime_mean"]) if fao_row is not None else "--",
            _format_cell(rcg_row["obj_mean"], rcg_row["obj_std"], rcg_row["runtime_mean"]) if rcg_row is not None else "--",
            _format_cell(fifo_row["obj_mean"], fifo_row["obj_std"], fifo_row["runtime_mean"]) if fifo_row is not None else "--",
            _format_cell(greedy_row["obj_mean"], greedy_row["obj_std"], greedy_row["runtime_mean"]) if greedy_row is not None else "--",
        ]
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    Path(latex_out).parent.mkdir(parents=True, exist_ok=True)
    Path(latex_out).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_update_note(note_out: str, available_methods: List[str], unavailable_methods: List[str]) -> None:
    note = f"""实验框架更新提示

建议在论文以下位置补入新基线的描述与结果讨论：

1. 第4章 数值实验 / 实验设置 / 对比算法与实现细节
位置：现有“小节 4.1 或 4.1.x 对比算法与实现细节”
建议新增内容：
- Rolling-Horizon MILP：作为“分窗口重优化”的近似精确基线，代表工程上常见的滚动求解策略。
- Fix-and-Optimize：作为“大邻域局部精确改进”基线，代表基于局部重优化的混合方法。
- Restricted-CG：作为“快速近似优化”基线，代表不完全定价和受限列池规模下的列生成近似法。

2. 第4章 数值实验 / 总体结果与对比分析
位置：现有“总体结果与对比分析”段落之后
建议新增一段比较：
- Rolling-Horizon 与完整 MILP/CG 的关系：速度更稳，但因窗口提交导致全局协调能力不足。
- Fix-and-Optimize 与 CG 的关系：优于简单启发式，但通常仍弱于完整 CG。
- Restricted-CG 与完整 CG 的关系：保留列生成思想，但通过限制定价范围换取速度，适合作为近似列生成基线。

3. 第4章 数值实验 / 可扩展性与算法讨论
位置：可放在“可扩展性与消融实验”前或后
建议新增一段“算法家族对比”讨论：
- 精确基准：MILP300
- 完整分解法：CG
- 近似分解法：Restricted-CG
- 滚动优化法：Rolling-Horizon
- 局部重优化法：Fix-and-Optimize
- 启发式规则：FIFO / Greedy

4. 第5章 结论 / 方法贡献与管理含义
位置：现有“方法贡献与管理含义”中算法贡献段落
建议补一句：
- 与 Rolling-Horizon、Fix-and-Optimize 和 Restricted-CG 等代表性近似优化框架相比，完整 CG 在解质量与规模鲁棒性之间提供了更强的整体平衡。

本次脚本已识别到的可运行新基线：
- {", ".join(available_methods) if available_methods else "无"}

当前仓库中尚未检测到的基线模块：
- {", ".join(unavailable_methods) if unavailable_methods else "无"}
"""
    Path(note_out).parent.mkdir(parents=True, exist_ok=True)
    Path(note_out).write_text(note, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate new baselines into the main experiment results.")
    parser.add_argument("--config", default="configs/main.yaml", help="Path to the main experiment YAML config.")
    parser.add_argument("--output_csv", default="results/results_main_rigorous.csv", help="Existing/main results CSV.")
    parser.add_argument("--summary_csv", default="results/main_summary_with_new_baselines.csv", help="Output grouped summary CSV.")
    parser.add_argument("--latex_out", default="results/main_table_with_new_baselines.tex", help="Output LaTeX table path.")
    parser.add_argument("--note_out", default="deliverables/new_baselines_paper_note_cn.txt", help="Output note for paper updates.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["rolling_horizon", "fix_and_optimize", "restricted_cg"],
        help="Logical baseline methods to run/integrate.",
    )
    parser.add_argument(
        "--restricted_variant",
        default="rcg_random",
        choices=["rcg_random", "rcg_arrival"],
        help="Internal restricted-CG implementation used for the logical method 'restricted_cg'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing rows of the new baseline methods before appending fresh results.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_info = run_new_baselines(
        config=config,
        output_csv=args.output_csv,
        logical_methods=list(args.methods),
        restricted_variant=args.restricted_variant,
        overwrite=args.overwrite,
    )

    df = pd.read_csv(args.output_csv)
    summary = build_summary_table(df)
    Path(args.summary_csv).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    generate_latex_table(summary, args.latex_out)
    write_paper_update_note(args.note_out, run_info["available_methods"], run_info["unavailable_methods"])

    print("Integrated methods:", run_info["available_methods"])
    if run_info["unavailable_methods"]:
        print("Unavailable methods skipped:", run_info["unavailable_methods"])
    print("Rows written:", run_info["rows_written"])
    print("Summary CSV:", args.summary_csv)
    print("LaTeX table:", args.latex_out)
    print("Paper note:", args.note_out)


if __name__ == "__main__":
    main()

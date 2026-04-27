from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tr_figures.config import METHOD_COLORS, METHOD_LABELS, METHOD_LINESTYLES, METHOD_MARKERS, REFERENCE_COLOR, REFERENCE_LINESTYLE, SINGLE_COLUMN
from analysis.tr_figures.utils import apply_common_axis_format, set_x_axis_label, start_figure


BASELINE_NS = [25, 50, 100, 200, 500]
TARGET_NS = [75, 125, 150]
FULL_NS = [25, 50, 75, 100, 125, 150, 200, 500]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 1 fill-point outputs.")
    parser.add_argument("--baseline", default="results/results_simops_rigorous.csv", help="Existing SIMOPS results CSV.")
    parser.add_argument("--results-dir", default="results/experiment_1_fill_points", help="Experiment 1 output directory.")
    return parser.parse_args()


def format_mean_std(series: pd.Series, digits: int = 2) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "-"
    mean = clean.mean()
    std = clean.std(ddof=0)
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def load_raw_json_rows(raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        raise FileNotFoundError(f"No raw result JSON files found in {raw_dir}")
    return pd.DataFrame(rows)


def normalize_baseline(df: pd.DataFrame) -> pd.DataFrame:
    cg = df[(df["method"] == "cg") & (df["scenario"] == "U")].copy()
    cg["objective"] = pd.to_numeric(cg["obj"], errors="coerce")
    cg["runtime"] = pd.to_numeric(cg["runtime_total"], errors="coerce")
    cg["internal_gap"] = pd.to_numeric(cg["gap_pct"], errors="coerce")
    cg["lp_lower_bound"] = cg["objective"] / (1.0 + cg["internal_gap"] / 100.0)
    cg["n_columns_generated"] = pd.to_numeric(cg.get("n_columns_generated"), errors="coerce")
    cg["sp_share"] = pd.to_numeric(cg["shore_ratio"], errors="coerce")
    cg["bs_share"] = pd.to_numeric(cg["battery_ratio"], errors="coerce")
    cg["ae_share"] = pd.to_numeric(cg["brown_ratio"], errors="coerce")
    cg["sp_utilization"] = pd.to_numeric(cg["shore_utilization"], errors="coerce")
    cg["masking_rate"] = pd.to_numeric(cg["avg_masking_rate"], errors="coerce")
    cg["delay_cost"] = pd.to_numeric(cg["cost_delay"], errors="coerce")
    cg["energy_cost"] = pd.to_numeric(cg["cost_energy"], errors="coerce")
    cg["success"] = cg["status"].eq("ok")
    keep = [
        "N",
        "seed",
        "operation_mode",
        "method",
        "objective",
        "runtime",
        "internal_gap",
        "lp_lower_bound",
        "n_columns_generated",
        "sp_share",
        "bs_share",
        "ae_share",
        "sp_utilization",
        "avg_stay_time",
        "masking_rate",
        "delay_cost",
        "energy_cost",
        "success",
    ]
    return cg[keep].copy()


def normalize_new(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "runtime": "runtime",
            "avg_stay_time": "avg_stay_time",
            "masking_rate": "masking_rate",
        }
    ).copy()
    keep = [
        "N",
        "seed",
        "operation_mode",
        "method",
        "objective",
        "runtime",
        "internal_gap",
        "lp_lower_bound",
        "n_columns_generated",
        "sp_share",
        "bs_share",
        "ae_share",
        "sp_utilization",
        "avg_stay_time",
        "masking_rate",
        "delay_cost",
        "energy_cost",
        "success",
    ]
    for column in keep:
        if column not in renamed.columns:
            renamed[column] = np.nan
    return renamed[keep].copy()


def aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["N", "operation_mode", "method"], dropna=False)
    rows: list[dict[str, Any]] = []
    metrics = [
        "objective",
        "runtime",
        "internal_gap",
        "lp_lower_bound",
        "n_columns_generated",
        "sp_share",
        "bs_share",
        "ae_share",
        "sp_utilization",
        "avg_stay_time",
        "masking_rate",
        "delay_cost",
        "energy_cost",
    ]
    for (N, operation_mode, method), block in grouped:
        row: dict[str, Any] = {"N": int(N), "operation_mode": operation_mode, "method": method}
        for metric in metrics:
            clean = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(clean.mean()) if not clean.empty else np.nan
            row[f"{metric}_std"] = float(clean.std(ddof=0)) if len(clean) else np.nan
        row["success_rate"] = float(pd.to_numeric(block["success"], errors="coerce").fillna(0).mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["N", "operation_mode", "method"]).reset_index(drop=True)


def compute_savings_curve(agg: pd.DataFrame) -> pd.DataFrame:
    sim = agg[agg["operation_mode"] == "simops"][["N", "objective_mean"]].rename(columns={"objective_mean": "sim_obj"})
    seq = agg[agg["operation_mode"] == "sequential"][["N", "objective_mean"]].rename(columns={"objective_mean": "seq_obj"})
    merged = sim.merge(seq, on="N", how="inner").sort_values("N")
    merged["saving_pct"] = (merged["seq_obj"] - merged["sim_obj"]) / merged["seq_obj"] * 100.0
    return merged


def compute_seed_level_savings(df: pd.DataFrame) -> pd.DataFrame:
    sim = df[df["operation_mode"] == "simops"][["N", "seed", "objective"]].rename(columns={"objective": "sim_obj"})
    seq = df[df["operation_mode"] == "sequential"][["N", "seed", "objective"]].rename(columns={"objective": "seq_obj"})
    merged = sim.merge(seq, on=["N", "seed"], how="inner").sort_values(["N", "seed"])
    merged["saving_pct"] = (merged["seq_obj"] - merged["sim_obj"]) / merged["seq_obj"] * 100.0
    return merged


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_table10_tex(df: pd.DataFrame, output_path: Path) -> None:
    grouped = df.groupby(["N", "operation_mode", "method"], dropna=False)
    rows: list[dict[str, Any]] = []
    for (N, operation_mode, method), block in grouped:
        rows.append(
            {
                "N": int(N),
                "operation_mode": operation_mode,
                "method": METHOD_LABELS.get(method, method),
                "obj": format_mean_std(block["objective"], digits=2),
                "runtime": format_mean_std(block["runtime"], digits=3),
                "avg_masking_rate": format_mean_std(block["masking_rate"], digits=2),
                "avg_stay_time": format_mean_std(block["avg_stay_time"], digits=2),
                "success": f"{100.0 * pd.to_numeric(block['success'], errors='coerce').fillna(0).mean():.1f}\\%",
            }
        )
    table = pd.DataFrame(rows).sort_values(["N", "operation_mode"])
    output_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def save_fig5a_updated(savings: pd.DataFrame, output_dir: Path) -> None:
    start_figure()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN.width, SINGLE_COLUMN.height))
    ax.plot(
        savings["N"],
        savings["saving_pct"],
        color=METHOD_COLORS["cg"],
        marker=METHOD_MARKERS["cg"],
        linestyle=METHOD_LINESTYLES["cg"],
        label=METHOD_LABELS["cg"],
    )
    peak = savings.loc[savings["saving_pct"].idxmax()]
    peak_label = "largest gain near\ncongestion transition" if int(peak["N"]) == 100 else f"local peak around N={int(peak['N'])}"
    ax.annotate(
        peak_label,
        xy=(peak["N"], peak["saving_pct"]),
        xytext=(peak["N"] + 10, float(peak["saving_pct"]) + 1.8),
        fontsize=8.0,
        color=METHOD_COLORS["cg"],
        ha="left",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": METHOD_COLORS["cg"]},
    )
    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    set_x_axis_label(ax, "Number of ships $N$")
    ax.set_ylabel("Savings vs sequential (%)")
    ax.set_xticks(FULL_NS)
    apply_common_axis_format(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5a_updated.pdf")
    fig.savefig(output_dir / "fig5a_updated.png", dpi=300)
    plt.close(fig)


def save_fig5a_prime(seed_savings: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(20260417)
    rows: list[dict[str, Any]] = []
    for N in FULL_NS:
        values = seed_savings.loc[seed_savings["N"] == N, "saving_pct"].to_numpy(dtype=float)
        mean, low, high = bootstrap_mean_ci(values, rng)
        rows.append({"N": N, "mean": mean, "low": low, "high": high})
    boot = pd.DataFrame(rows)

    start_figure()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN.width, SINGLE_COLUMN.height))
    ax.plot(
        boot["N"],
        boot["mean"],
        color=METHOD_COLORS["cg"],
        marker=METHOD_MARKERS["cg"],
        linestyle=METHOD_LINESTYLES["cg"],
        label=METHOD_LABELS["cg"],
    )
    ax.fill_between(boot["N"], boot["low"], boot["high"], color=METHOD_COLORS["cg"], alpha=0.16, label="95% bootstrap CI")
    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    set_x_axis_label(ax, "Number of ships $N$")
    ax.set_ylabel("Savings vs sequential (%)")
    ax.set_xticks(FULL_NS)
    apply_common_axis_format(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig5a_prime.pdf")
    fig.savefig(output_dir / "fig5a_prime.png", dpi=300)
    plt.close(fig)


def build_summary_markdown(agg: pd.DataFrame, savings: pd.DataFrame, output_path: Path) -> None:
    sim = agg[agg["operation_mode"] == "simops"].set_index("N")
    seq = agg[agg["operation_mode"] == "sequential"].set_index("N")
    savings = savings.set_index("N")

    def row_line(N: int) -> str:
        sim_row = sim.loc[N]
        seq_row = seq.loc[N]
        save_val = savings.loc[N, "saving_pct"]
        gap_val = sim_row["internal_gap_mean"]
        return (
            f"| {N} | {sim_row['objective_mean']:.2f} ± {sim_row['objective_std']:.2f} | "
            f"{seq_row['objective_mean']:.2f} ± {seq_row['objective_std']:.2f} | "
            f"{save_val:.2f} | {sim_row['masking_rate_mean']:.3f} | {gap_val:.4f} |"
        )

    seq_growth_intervals = [(50, 75), (75, 100), (100, 125), (125, 150)]
    growth_lines = []
    for left, right in seq_growth_intervals:
        growth = seq.loc[right, "objective_mean"] / seq.loc[left, "objective_mean"]
        growth_lines.append(f"- N={left}→{right}: ×{growth:.3f}")

    local_peak = savings.loc[100, "saving_pct"] > savings.loc[75, "saving_pct"] and savings.loc[100, "saving_pct"] > savings.loc[125, "saving_pct"]
    global_peak_n = int(savings["saving_pct"].idxmax())

    if savings.loc[75, "saving_pct"] > savings.loc[100, "saving_pct"] and savings.loc[125, "saving_pct"] > savings.loc[100, "saving_pct"]:
        scenario_label = "情景 C"
        shape_text = "新增点在 N=100 两侧同时高于 N=100，呈明显 U 形，不支持把 N=100 作为第二峰中心。"
        wording_text = "建议暂停强化原 claim，先把表述降级为“收益随规模变化呈显著异质性”，并人工复核 N=100 原始样本。"
    elif savings.loc[75, "saving_pct"] > savings.loc[100, "saving_pct"]:
        scenario_label = "情景 B"
        shape_text = "曲线仍然非单调，但局部峰值更接近 N=75-100 区间，而不是精确落在 N=100。"
        wording_text = "建议把“第二峰在 N=100 附近”改成“第二峰位于 N=75-100 区间”。"
    else:
        scenario_label = "情景 A"
        shape_text = "曲线在新增中间点后仍保持 N=75→100 上升、100→125→150 下降的局部峰结构。"
        wording_text = "可以保留“第二峰在 N=100 附近”的主张，但建议改成“在 N=75-125 区间形成局部峰，峰值位于 N=100”。"

    lines = [
        "# 实验一结果摘要",
        "",
        "## 1. 完整结果表",
        "",
        "| N | SIMOPS obj | Seq obj | Savings(%) | Masking rate | Internal gap(%) |",
        "|---|-----------|---------|------------|--------------|-----------------|",
    ]
    lines.extend(row_line(N) for N in FULL_NS)
    lines.extend(
        [
            "",
            "## 2. 曲线形状判定",
            "",
            f"- 情景判定: {scenario_label}",
            f"- N=100 是否仍是局部最大: {'是' if local_peak else '否'}。全局最大节约率出现在 N={global_peak_n}，值为 {savings.loc[global_peak_n, 'saving_pct']:.2f}%。",
            f"- 曲线整体形状: {shape_text}",
            f"- N=75→100→125 三点是否支撑“第二峰在 N=100 附近”: {'支撑' if local_peak else '不完全支撑'}。",
            f"- 建议措辞: {wording_text}",
            "",
            "## 3. 机制解释的一致性检查",
            "",
            f"- 论文原文的 N=50→100 Sequential 成本增长率约为 ×{(seq.loc[100, 'objective_mean'] / seq.loc[50, 'objective_mean']):.3f}。",
        ]
    )
    lines.extend(growth_lines)
    lines.extend(
        [
            "",
            "## 4. 对论文的修改建议",
            "",
            "- §5.3 建议替换当前基于五点曲线的段落，直接引用八点曲线和新增的 N=75/125/150 结果。",
            "- Table 10 建议在 50 与 100 之间插入 N=75，在 100 与 200 之间插入 N=125 和 N=150。",
            "- Fig. 5(a) 建议用新的八点 CG+IR 曲线替换原五点版本；若峰值不再位于 N=100，则同步调整箭头标注位置与文字。",
            f"- Finding 3 的陈述强度建议: {wording_text}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"

    baseline = pd.read_csv(args.baseline)
    new_raw = load_raw_json_rows(raw_dir)

    combined = pd.concat(
        [
            normalize_baseline(baseline),
            normalize_new(new_raw),
        ],
        ignore_index=True,
    )
    combined = combined[combined["N"].isin(FULL_NS)].copy()
    combined["method"] = "cg"

    aggregated = aggregate_rows(combined)
    aggregated.to_csv(results_dir / "aggregated.csv", index=False)

    build_table10_tex(combined, results_dir / "table10_extended.tex")
    savings = compute_savings_curve(aggregated)
    seed_savings = compute_seed_level_savings(combined)
    save_fig5a_updated(savings, results_dir)
    save_fig5a_prime(seed_savings, results_dir)
    build_summary_markdown(aggregated, savings, results_dir / "summary.md")


if __name__ == "__main__":
    main()

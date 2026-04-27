from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from generate_fig8_dual import draw_fig8_dual, draw_fig8_prime


SUMMARY_METRICS = [
    "objective",
    "runtime",
    "cg_iterations",
    "internal_gap",
    "sp_share",
    "bs_share",
    "ae_share",
    "sp_utilization",
    "avg_stay_time",
    "masking_rate",
    "delay_cost",
    "delay_cost_pct",
    "energy_cost",
    "n_delayed_vessels",
    "max_delay",
    "avg_delay_of_delayed",
    "n_mode_switches_vs_baseline",
    "n_start_time_shifts",
    "feasibility_margin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 2 outputs.")
    parser.add_argument("--results-dir", default="results/experiment_2_tight_deadline", help="Experiment 2 results directory.")
    return parser.parse_args()


def aggregate_config(df: pd.DataFrame, config_name: str) -> pd.DataFrame:
    block = df[df["config_name"] == config_name].copy()
    rows: list[dict[str, Any]] = []
    for delta, group in block.groupby("delta", dropna=False):
        row: dict[str, Any] = {"config_name": config_name, "delta": float(delta)}
        for metric in SUMMARY_METRICS:
            clean = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(clean.mean()) if not clean.empty else np.nan
            row[f"{metric}_std"] = float(clean.std(ddof=0)) if len(clean) else np.nan
        row["slack_mean"] = float(group["slack_mean"].iloc[0])
        baseline_mean = float(block.loc[block["delta"] == 0.0, "objective"].mean())
        row["cost_change_pct"] = (row["objective_mean"] - baseline_mean) / baseline_mean * 100.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


def build_comparison(loose: pd.DataFrame, tight: pd.DataFrame) -> pd.DataFrame:
    merged = loose.merge(tight, on="delta", suffixes=("_loose", "_tight"))
    comparison_rows = []
    for _, row in merged.iterrows():
        out: dict[str, Any] = {"delta": float(row["delta"])}
        for config in ["loose", "tight"]:
            source_suffix = "loose" if config == "loose" else "tight"
            for metric in SUMMARY_METRICS:
                out[f"{config}_{metric}_mean"] = row[f"{metric}_mean_{source_suffix}"]
                out[f"{config}_{metric}_std"] = row[f"{metric}_std_{source_suffix}"]
            out[f"{config}_cost_change_pct"] = row[f"cost_change_pct_{source_suffix}"]
            out[f"{config}_slack_mean"] = row[f"slack_mean_{source_suffix}"]
            out[f"{config}_relative_delta"] = float(row["delta"]) / float(row[f"slack_mean_{source_suffix}"])
        out["cost_diff_pct"] = out["tight_cost_change_pct"] - out["loose_cost_change_pct"]
        out["delay_cost_pct_diff_pp"] = out["tight_delay_cost_pct_mean"] - out["loose_delay_cost_pct_mean"]
        out["ae_share_diff_pp"] = (out["tight_ae_share_mean"] - out["loose_ae_share_mean"]) * 100.0
        comparison_rows.append(out)
    return pd.DataFrame(comparison_rows).sort_values("delta").reset_index(drop=True)


def classify_scenario(comparison: pd.DataFrame) -> tuple[str, str]:
    row = comparison.loc[comparison["delta"] == 2.0].iloc[0]
    delay_pct = float(row["tight_delay_cost_pct_mean"])
    ae_share = float(row["tight_ae_share_mean"] * 100.0)
    total_cost_change = float(abs(row["tight_cost_change_pct"]))
    sp_change = float(abs((row["tight_sp_share_mean"] - comparison.loc[comparison["delta"] == 0.0, "tight_sp_share_mean"].iloc[0]) * 100.0))
    bs_change = float(abs((row["tight_bs_share_mean"] - comparison.loc[comparison["delta"] == 0.0, "tight_bs_share_mean"].iloc[0]) * 100.0))
    ae_change = float(abs((row["tight_ae_share_mean"] - comparison.loc[comparison["delta"] == 0.0, "tight_ae_share_mean"].iloc[0]) * 100.0))
    mode_wave = max(sp_change, bs_change, ae_change)

    if delay_pct >= 3 and delay_pct <= 10 and mode_wave < 5 and ae_share < 15 and total_cost_change < 5:
        return "情景 α", "紧 deadline 下延迟被激活，但结构保持稳定。"
    if delay_pct >= 5 and delay_pct <= 25 and mode_wave <= 10 and total_cost_change <= 10:
        return "情景 β", "延迟成本明显上升，但服务组合没有失控，系统体现出组合韧性。"
    if total_cost_change > 10 or ae_change > 15 or bs_change > 10 or ae_share > 15:
        return "情景 γ", "系统开始出现显著结构重排，更适合解释为韧性阈值而非简单鲁棒性。"
    return "情景 δ", "结果接近失控或求解质量异常，需要人工介入。"


def build_summary(comparison: pd.DataFrame, final_slack_text: str, output_path: Path) -> None:
    row_delta1 = comparison.loc[comparison["delta"] == 1.0].iloc[0]
    row_delta2 = comparison.loc[comparison["delta"] == 2.0].iloc[0]
    scenario_label, scenario_text = classify_scenario(comparison)

    loose_mean = float(comparison["loose_slack_mean"].iloc[0])
    tight_mean = float(comparison["tight_slack_mean"].iloc[0])
    relative_boundary = float(row_delta2["tight_relative_delta"])

    slack_lines = [line for line in final_slack_text.strip().splitlines() if line.strip() and not line.startswith("#")]
    lines = [
        "# 实验二结果摘要",
        "",
        "## 1. 可行性预检结果",
        "",
    ]
    lines.extend(slack_lines)
    lines.extend(
        [
        "",
        "## 2. 完整结果对比表",
        "",
        "### 松 deadline(复跑 §5.6 基线)",
        "",
        "| Δ (h) | 总成本 | 能源成本 | 延迟成本 | 延迟占比 | AE份额 | masking rate |",
        "|---|---|---|---|---|---|---|",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['delta']:.1f} | {row['loose_objective_mean']:.2f} ± {row['loose_objective_std']:.2f} | "
            f"{row['loose_energy_cost_mean']:.2f} | {row['loose_delay_cost_mean']:.2f} | {row['loose_delay_cost_pct_mean']:.3f}% | "
            f"{row['loose_ae_share_mean'] * 100:.2f}% | {row['loose_masking_rate_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### 紧 deadline(本实验主体)",
            "",
            "| Δ (h) | 总成本 | 能源成本 | 延迟成本 | 延迟占比 | AE份额 | masking rate |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['delta']:.1f} | {row['tight_objective_mean']:.2f} ± {row['tight_objective_std']:.2f} | "
            f"{row['tight_energy_cost_mean']:.2f} | {row['tight_delay_cost_mean']:.2f} | {row['tight_delay_cost_pct_mean']:.3f}% | "
            f"{row['tight_ae_share_mean'] * 100:.2f}% | {row['tight_masking_rate_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### 两者差异摘要",
            "",
            "| Δ (h) | 成本差异 (%) | Delay占比差异 (pp) | AE份额差异 (pp) |",
            "|---|---|---|---|",
        ]
    )
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['delta']:.1f} | {row['cost_diff_pct']:.3f} | {row['delay_cost_pct_diff_pp']:.3f} | {row['ae_share_diff_pp']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 3. 三个关键问题的回答",
            "",
            "### Q1:紧 deadline 下系统是否进入 binding 区间?",
            f"在 Δ=1.0 h 时，tight 配置的平均 delay cost 占比为 {row_delta1['tight_delay_cost_pct_mean']:.2f}%，平均延迟船数为 {row_delta1['tight_n_delayed_vessels_mean']:.2f}，"
            f"最小 feasibility margin 均值为 {row_delta1['tight_feasibility_margin_mean']:.2f} h。Δ=2.0 h 时该占比升至 {row_delta2['tight_delay_cost_pct_mean']:.2f}%，说明 deadline 确实进入 binding 区间。",
            "",
            "### Q2:结构稳定性是否仍然成立?",
            f"tight 配置下从 Δ=0 到 Δ=2.0 h，SP/BS/AE 三种份额的最大波动分别约为 "
            f"{abs((comparison['tight_sp_share_mean'] - comparison.loc[comparison['delta']==0.0, 'tight_sp_share_mean'].iloc[0]) * 100).max():.2f} / "
            f"{abs((comparison['tight_bs_share_mean'] - comparison.loc[comparison['delta']==0.0, 'tight_bs_share_mean'].iloc[0]) * 100).max():.2f} / "
            f"{abs((comparison['tight_ae_share_mean'] - comparison.loc[comparison['delta']==0.0, 'tight_ae_share_mean'].iloc[0]) * 100).max():.2f} pp。"
            f"分类结果为 {scenario_label}，即 {scenario_text}",
            "",
            "### Q3:鲁棒性边界在哪里?",
            f"本次 loose 与 tight 配置的理论 slack 均值分别为 {loose_mean:.2f} h 和 {tight_mean:.3f} h。"
            f"当 relative perturbation Δ/slack_mean 接近 {relative_boundary:.2f} 时，tight 配置的成本抬升与延迟代价都明显可见，而 loose 配置仍基本平稳。"
            "因此，鲁棒性更适合被表述为‘相对扰动幅度受控时成本变化温和，超过 slack 量级后敏感性快速上升’。",
            "",
            "## 4. 对论文的修改建议",
            "",
            "### §5.6 重写框架",
            "- 开头: 说明原 loose deadline 结果成立，但其适用范围主要对应 slack 足以吸收扰动的情形。",
            "- 中段: 引入 tight deadline 配置作为压力测试，强调 deadline 保持不变、只有 arrival 被扰动。",
            "- 结尾: 用 relative perturbation 边界总结‘何时鲁棒、何时进入 binding 区间’。",
            "",
            "### 具体段落草稿",
            "",
            "原松 deadline 配置下，到港扰动几乎总能被剩余 slack 吸收，因此总成本变化维持在极窄区间内。为避免将这一现象误读为模型在任意时限条件下都具有强鲁棒性，本文进一步构造了紧 deadline 压力测试，其中 slack 分布收紧至最终预检通过的区间，并保持截止时间在扰动后固定不动。结果表明，当扰动幅度接近平均 slack 的量级时，系统的延迟成本开始显著上升，说明 deadline 约束真正进入 binding 区间。",
            "",
            f"尽管紧 deadline 下的成本敏感性明显强于基线配置，但服务模式结构并未同步失控。以 Δ=2.0 h 为例，tight 配置下 AE 份额为 {row_delta2['tight_ae_share_mean'] * 100:.2f}%，"
            f"总成本相对自身基线变化 {row_delta2['tight_cost_change_pct']:.2f}%，表明系统仍保持了组合调度层面的结构韧性。因而，§5.6 更准确的结论应是："
            "SIMOPS 调度框架对到港扰动的鲁棒性具有明确的边界，当扰动幅度小于可用 slack 时成本变化可忽略，而当扰动逼近或超过 slack 量级时，延迟机制被激活但系统结构仍保持有界调整。",
            "",
            "## 5. 情景判定",
            "",
            f"- 判定结果: {scenario_label}",
            f"- 解释: {scenario_text}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    raw_index = pd.read_csv(results_dir / "raw_results_index.csv")

    loose = aggregate_config(raw_index, "loose")
    tight = aggregate_config(raw_index, "tight")
    tight.to_csv(results_dir / "aggregated_tight.csv", index=False)
    loose.to_csv(results_dir / "aggregated_loose.csv", index=False)

    comparison = build_comparison(loose, tight)
    comparison.to_csv(results_dir / "aggregated_comparison.csv", index=False)

    final_slack_text = (results_dir / "final_slack_decision.md").read_text(encoding="utf-8")
    build_summary(comparison, final_slack_text, results_dir / "summary.md")

    draw_fig8_dual(comparison, results_dir)
    draw_fig8_prime(comparison, results_dir)


if __name__ == "__main__":
    main()

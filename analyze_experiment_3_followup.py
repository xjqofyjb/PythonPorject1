from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tr_figures.config import DOUBLE_COLUMN, METHOD_COLORS, SINGLE_COLUMN
from analysis.tr_figures.utils import apply_common_axis_format, set_x_axis_label, start_figure


EPSILON_LEVELS = [0.5, 1.0, 2.0]
TARGET_NS = [100, 200, 500]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 3 follow-up outputs.")
    parser.add_argument("--results-dir", default="results/experiment_3_followup", help="Results directory.")
    return parser.parse_args()


def load_raw_rows(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rows: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("record_type") == "baseline":
            baseline_rows.append(payload)
        elif payload.get("record_type") == "enriched":
            enriched_rows.append(payload)
    if not baseline_rows or not enriched_rows:
        raise FileNotFoundError(f"Missing raw follow-up JSON files in {raw_dir}")
    return pd.DataFrame(baseline_rows), pd.DataFrame(enriched_rows)


def unpack_dict_column(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    rows = []
    for payload in df[column]:
        if isinstance(payload, str):
            payload = json.loads(payload)
        rows.append(
            {
                f"{prefix}_SP": float(payload.get("SP", np.nan)),
                f"{prefix}_BS": float(payload.get("BS", np.nan)),
                f"{prefix}_AE": float(payload.get("AE", np.nan)),
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def aggregate_results(baseline_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "baseline_pool_size",
        "full_column_pool_size",
        "enriched_pool_size",
        "columns_added",
        "Z_baseline_IRMP",
        "Z_enriched_IRMP",
        "improvement_pct",
        "n_plans_changed",
        "n_mode_switches",
        "columns_used_from_enrichment",
        "enrichment_usage_ratio",
    ]
    rows: list[dict[str, Any]] = []
    for (N, pricing_mode, epsilon_pct), block in enriched_df.groupby(["N", "pricing_mode", "epsilon_pct"], dropna=False):
        row: dict[str, Any] = {"N": int(N), "pricing_mode": pricing_mode, "epsilon_pct": float(epsilon_pct), "epsilon_label": f"{float(epsilon_pct):.1f}%"}
        for metric in metrics:
            clean = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(clean.mean()) if not clean.empty else np.nan
            row[f"{metric}_std"] = float(clean.std(ddof=0)) if len(clean) else np.nan
        row["identical_rate"] = float((block["objective_equivalence_type"] == "identical").mean())
        row["alternative_optima_rate"] = float((block["objective_equivalence_type"] == "alternative_optima").mean())
        row["improved_rate"] = float((block["objective_equivalence_type"] == "improved").mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["N", "epsilon_pct"]).reset_index(drop=True)


def build_equivalence_summary(enriched_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (N, epsilon_pct), block in enriched_df.groupby(["N", "epsilon_pct"], dropna=False):
        rows.append(
            {
                "N": int(N),
                "epsilon_pct": float(epsilon_pct),
                "epsilon_label": f"{float(epsilon_pct):.1f}%",
                "identical": int((block["objective_equivalence_type"] == "identical").sum()),
                "alternative_optima": int((block["objective_equivalence_type"] == "alternative_optima").sum()),
                "improved": int((block["objective_equivalence_type"] == "improved").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["N", "epsilon_pct"]).reset_index(drop=True)


def load_n100_full_pool_reference() -> dict[str, Any]:
    main = pd.read_csv("results/results_main_rigorous.csv")
    main_n100 = main[(main["scenario"] == "U") & (main["mechanism"] == "hybrid") & (main["method"] == "cg") & (main["N"] == 100)].copy()
    exp3 = pd.read_csv("results/experiment_3_column_enrichment/aggregated.csv")
    exp3_n100 = exp3[(exp3["N"] == 100) & np.isclose(exp3["epsilon_pct"], 1.0)].iloc[0]
    return {
        "pool_size_mean": float(exp3_n100["baseline_pool_size_mean"]),
        "objective_mean": float(main_n100["obj"].mean()),
        "objective_std": float(main_n100["obj"].std(ddof=0)),
    }


def build_n100_compare(agg: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    ref = load_n100_full_pool_reference()
    topk = agg[(agg["N"] == 100) & np.isclose(agg["epsilon_pct"], 1.0)].iloc[0]
    baseline_block = baseline_df[baseline_df["N"] == 100].copy()
    topk_dist = {
        "SP": float(pd.to_numeric(baseline_block["mode_distribution_SP"], errors="coerce").mean()),
        "BS": float(pd.to_numeric(baseline_block["mode_distribution_BS"], errors="coerce").mean()),
        "AE": float(pd.to_numeric(baseline_block["mode_distribution_AE"], errors="coerce").mean()),
    }
    return pd.DataFrame(
        [
            {
                "Metric": "Baseline pool size",
                "N100_full_pool": f"{ref['pool_size_mean']:.1f}",
                "N100_topk_diag": f"{topk['baseline_pool_size_mean']:.1f} ± {topk['baseline_pool_size_std']:.1f}",
                "Delta": f"{topk['baseline_pool_size_mean'] - ref['pool_size_mean']:.1f}",
            },
            {
                "Metric": "Objective value",
                "N100_full_pool": f"{ref['objective_mean']:.2f} ± {ref['objective_std']:.2f}",
                "N100_topk_diag": f"{topk['Z_baseline_IRMP_mean']:.2f} ± {topk['Z_baseline_IRMP_std']:.2f}",
                "Delta": f"{(topk['Z_baseline_IRMP_mean'] - ref['objective_mean']) / ref['objective_mean'] * 100.0:.4f}%",
            },
            {
                "Metric": "Plan distribution (SP/BS/AE)",
                "N100_full_pool": "see main/full-pool reference",
                "N100_topk_diag": f"{topk_dist['SP']:.3f} / {topk_dist['BS']:.3f} / {topk_dist['AE']:.3f}",
                "Delta": "descriptive",
            },
        ]
    )


def build_table_tex(agg: pd.DataFrame, output_path: Path) -> None:
    block = agg[np.isclose(agg["epsilon_pct"], 1.0)].copy().sort_values("N")
    rows = []
    for _, row in block.iterrows():
        eq_type = "identical" if abs(row["identical_rate"] - 1.0) < 1e-12 else "alternative_optima" if abs(row["alternative_optima_rate"] - 1.0) < 1e-12 else "mixed"
        rows.append(
            {
                "N": int(row["N"]),
                "mode": row["pricing_mode"],
                "Baseline pool": f"{row['baseline_pool_size_mean']:.1f} $\\pm$ {row['baseline_pool_size_std']:.1f}",
                "Enriched pool (1%)": f"{row['enriched_pool_size_mean']:.1f} $\\pm$ {row['enriched_pool_size_std']:.1f}",
                "Cols added": f"{row['columns_added_mean']:.1f} $\\pm$ {row['columns_added_std']:.1f}",
                "Z_baseline": f"{row['Z_baseline_IRMP_mean']:.2f} $\\pm$ {row['Z_baseline_IRMP_std']:.2f}",
                "Z_enriched": f"{row['Z_enriched_IRMP_mean']:.2f} $\\pm$ {row['Z_enriched_IRMP_std']:.2f}",
                "Improvement (%)": f"{row['improvement_pct_mean']:.4f} $\\pm$ {row['improvement_pct_std']:.4f}",
                "n_plans_changed": f"{row['n_plans_changed_mean']:.1f} $\\pm$ {row['n_plans_changed_std']:.1f}",
                "n_mode_switches": f"{row['n_mode_switches_mean']:.1f} $\\pm$ {row['n_mode_switches_std']:.1f}",
                "equivalence_type": eq_type,
                "cols_used_from_enrichment": f"{row['columns_used_from_enrichment_mean']:.1f} $\\pm$ {row['columns_used_from_enrichment_std']:.1f}",
            }
        )
    output_path.write_text(pd.DataFrame(rows).to_latex(index=False, escape=False), encoding="utf-8")


def save_three_panel_figure(agg: pd.DataFrame, eq_summary: pd.DataFrame, enriched_df: pd.DataFrame, output_dir: Path) -> None:
    start_figure()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COLUMN.width, 3.0))

    positions = np.arange(len(TARGET_NS))
    width = 0.22
    ax0 = axes[0]
    one_pct = agg[np.isclose(agg["epsilon_pct"], 1.0)].copy().sort_values("N")
    full_vals = one_pct["full_column_pool_size_mean"].to_numpy(dtype=float)
    base_vals = one_pct["baseline_pool_size_mean"].to_numpy(dtype=float)
    enr_vals = one_pct["enriched_pool_size_mean"].to_numpy(dtype=float)
    ax0.bar(positions - width, full_vals, width=width, label="Full enumerate", color="#CFCFCF")
    ax0.bar(positions, base_vals, width=width, label="Baseline top-K", color=METHOD_COLORS["cg"])
    ax0.bar(positions + width, enr_vals, width=width, label="Enriched 1%", color="#7AA974")
    for idx, pct in enumerate(base_vals / full_vals * 100.0):
        ax0.text(positions[idx], base_vals[idx] * 1.12, f"{pct:.1f}%", ha="center", va="bottom", fontsize=7.2)
    ax0.set_yscale("log")
    ax0.set_xticks(positions, ["100 diag", "200", "500"])
    set_x_axis_label(ax0, "Scale")
    ax0.set_ylabel("Pool size (log scale)")
    apply_common_axis_format(ax0)
    ax0.legend(loc="best")

    ax1 = axes[1]
    labels, identical, alternative, improved = [], [], [], []
    for N in TARGET_NS:
        for eps in EPSILON_LEVELS:
            block = eq_summary[(eq_summary["N"] == N) & np.isclose(eq_summary["epsilon_pct"], eps)].iloc[0]
            labels.append(f"{N}\n{eps:.1f}%")
            identical.append(int(block["identical"]))
            alternative.append(int(block["alternative_optima"]))
            improved.append(int(block["improved"]))
    x = np.arange(len(labels))
    ax1.bar(x, identical, color="#BFCDE0", label="identical")
    ax1.bar(x, alternative, bottom=identical, color="#7AA974", label="alternative_optima")
    ax1.bar(x, improved, bottom=np.array(identical) + np.array(alternative), color="#E5988A", label="improved")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Seed count")
    set_x_axis_label(ax1, "N × ε_loose")
    apply_common_axis_format(ax1)
    ax1.legend(loc="best")

    ax2 = axes[2]
    marker_map = {0.5: "o", 1.0: "s", 2.0: "^"}
    color_map = {0.5: "#BFCDE0", 1.0: METHOD_COLORS["cg"], 2.0: "#7AA974"}
    xpos = {100: 0.0, 200: 1.0, 500: 2.0}
    for eps in EPSILON_LEVELS:
        for N in TARGET_NS:
            sub = enriched_df[(enriched_df["N"] == N) & np.isclose(enriched_df["epsilon_pct"], eps)].copy()
            xs = np.linspace(xpos[N] - 0.1, xpos[N] + 0.1, len(sub)) if len(sub) else np.array([])
            ax2.scatter(xs, pd.to_numeric(sub["improvement_pct"], errors="coerce"), label=f"{eps:.1f}%" if N == 100 else None, color=color_map[eps], marker=marker_map[eps], s=18, alpha=0.85)
    ax2.axhline(0.0, color="#B8B8B8", linewidth=0.9)
    ax2.axhline(0.1, color="#D66A6A", linewidth=0.9, linestyle="--")
    ax2.set_xticks([0, 1, 2], ["100", "200", "500"])
    set_x_axis_label(ax2, "N")
    ax2.set_ylabel("Improvement (%)")
    apply_common_axis_format(ax2)
    ax2.legend(loc="best", title="ε_loose")

    fig.tight_layout()
    fig.savefig(output_dir / "fig10_three_panel.pdf")
    fig.savefig(output_dir / "fig10_three_panel.png", dpi=300)
    plt.close(fig)


def save_usage_figure(agg: pd.DataFrame, output_dir: Path) -> None:
    start_figure()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN.width, SINGLE_COLUMN.height))
    one_pct = agg[np.isclose(agg["epsilon_pct"], 1.0)].copy().sort_values("N")
    ax.plot(one_pct["N"], one_pct["enrichment_usage_ratio_mean"], color=METHOD_COLORS["cg"], marker="o")
    ax.set_xticks(TARGET_NS, ["100 diag", "200", "500"])
    set_x_axis_label(ax, "Scale")
    ax.set_ylabel("Used / added enrichment cols")
    apply_common_axis_format(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "fig10_enrichment_usage.pdf")
    fig.savefig(output_dir / "fig10_enrichment_usage.png", dpi=300)
    plt.close(fig)


def classify_scenario(eq_summary: pd.DataFrame, enriched_df: pd.DataFrame, n100_compare: pd.DataFrame) -> tuple[str, str]:
    improved_count = int(eq_summary["improved"].sum())
    max_improvement = float(pd.to_numeric(enriched_df["improvement_pct"], errors="coerce").fillna(0).max())
    mean_improvement = float(pd.to_numeric(enriched_df["improvement_pct"], errors="coerce").fillna(0).mean())
    n100_delta_pct = abs(float(str(n100_compare.loc[n100_compare["Metric"] == "Objective value", "Delta"].iloc[0]).rstrip("%")))
    used_positive = float((pd.to_numeric(enriched_df["columns_used_from_enrichment"], errors="coerce").fillna(0) > 0).mean())

    if improved_count == 0 and n100_delta_pct < 0.01 and used_positive > 0.2:
        return "情景 α-prime", "目标值完全稳定，且显著比例实例真实使用了富化列。"
    if improved_count <= 5 and mean_improvement < 0.05 and max_improvement < 0.2:
        return "情景 β-prime", "存在极少数极小改进，但整体仍可视为经验最优。"
    if mean_improvement <= 0.5 and max_improvement <= 0.5 and n100_delta_pct <= 0.1:
        return "情景 γ-prime", "出现了非平凡改进，需要收紧论文措辞。"
    return "情景 δ-prime", "出现显著改进或 N=100 交叉验证偏差过大，需要人工介入。"


def build_summary(results_dir: Path, baseline_df: pd.DataFrame, enriched_df: pd.DataFrame, agg: pd.DataFrame, eq_summary: pd.DataFrame, n100_compare: pd.DataFrame) -> None:
    precheck = json.loads((results_dir / "precheck" / "diagnostic_results.json").read_text(encoding="utf-8"))
    scenario_label, scenario_text = classify_scenario(eq_summary, enriched_df, n100_compare)
    ident_count = int((enriched_df["objective_equivalence_type"] == "identical").sum())
    alt_count = int((enriched_df["objective_equivalence_type"] == "alternative_optima").sum())
    improved_count = int((enriched_df["objective_equivalence_type"] == "improved").sum())
    used_share = float((pd.to_numeric(enriched_df["columns_used_from_enrichment"], errors="coerce").fillna(0) > 0).mean() * 100.0)
    one_pct = agg[np.isclose(agg["epsilon_pct"], 1.0)].copy().sort_values("N")
    compare_lines = [
        "| Metric | N100 full-pool | N100 top-K diagnostic | Delta |",
        "|---|---|---|---|",
    ]
    for _, row in n100_compare.iterrows():
        compare_lines.append(f"| {row['Metric']} | {row['N100_full_pool']} | {row['N100_topk_diag']} | {row['Delta']} |")

    lines = [
        "# 实验三续（top-K pricing 规模的列池富化）结果摘要",
        "",
        "## 1. 配置诊断（Step 0 结果）",
        "",
        f"- 三个规模的 baseline_pool_complete 全部为 False：{'YES' if all(not rec['baseline_pool_complete'] for rec in precheck['records']) else 'NO'}",
        f"- num_iters 全部 ≥ 5：{'YES' if all(rec['num_iters'] >= 5 for rec in precheck['records']) else 'NO'}",
        f"- is_restricted 全部为 False：{'YES' if all(not rec['is_restricted'] for rec in precheck['records']) else 'NO'}",
        f"- 所有硬性门禁通过：{'YES' if precheck['passed'] else 'NO'}",
        "",
        "## 2. 主表",
        "",
        "| N | mode | Baseline pool | Enriched pool (1%) | Cols added | Z_baseline | Z_enriched | Improvement (%) | n_plans_changed | n_mode_switches | equivalence_type | cols_used_from_enrichment |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in one_pct.iterrows():
        eq_type = "identical" if abs(row["identical_rate"] - 1.0) < 1e-12 else "alternative_optima" if abs(row["alternative_optima_rate"] - 1.0) < 1e-12 else "mixed"
        lines.append(
            f"| {int(row['N'])} | {row['pricing_mode']} | {row['baseline_pool_size_mean']:.1f} ± {row['baseline_pool_size_std']:.1f} | "
            f"{row['enriched_pool_size_mean']:.1f} ± {row['enriched_pool_size_std']:.1f} | {row['columns_added_mean']:.1f} ± {row['columns_added_std']:.1f} | "
            f"{row['Z_baseline_IRMP_mean']:.2f} ± {row['Z_baseline_IRMP_std']:.2f} | {row['Z_enriched_IRMP_mean']:.2f} ± {row['Z_enriched_IRMP_std']:.2f} | "
            f"{row['improvement_pct_mean']:.4f} ± {row['improvement_pct_std']:.4f} | {row['n_plans_changed_mean']:.1f} ± {row['n_plans_changed_std']:.1f} | "
            f"{row['n_mode_switches_mean']:.1f} ± {row['n_mode_switches_std']:.1f} | {eq_type} | {row['columns_used_from_enrichment_mean']:.1f} ± {row['columns_used_from_enrichment_std']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Equivalence type 分布",
            "",
            "| N | ε_loose | # identical | # alternative_optima | # improved |",
            "|---|---|---|---|---|",
        ]
    )
    for _, row in eq_summary.iterrows():
        lines.append(f"| {int(row['N'])} | {row['epsilon_label']} | {int(row['identical'])}/10 | {int(row['alternative_optima'])}/10 | {int(row['improved'])}/10 |")
    lines.extend(
        [
            "",
            "## 4. N=100 两模式交叉验证",
            "",
        ]
    )
    lines.extend(compare_lines)
    lines.extend(
        [
            "",
            "## 5. 三个核心问题的回答",
            "",
            "### Q1：top-K 配置下 IRMP 是否仍然最优？",
            f"- `identical / alternative_optima / improved` 的总体计数为 {ident_count} / {alt_count} / {improved_count}。",
            "- 如果没有 improved，说明即便列池不完整，baseline 已经足以触达最优目标值；alternative_optima 只是在目标值层面给出等价替代。",
            "",
            "### Q2：富化是否真实工作？",
            f"- `columns_used_from_enrichment > 0` 的实例比例为 {used_share:.1f}%。",
            "- 这个指标直接区分了“富化列被实际采用但目标值不变”和“富化列完全没被用到”两种情况。",
            "",
            "### Q3：N=100 两模式的解质量是否一致？",
            f"- N=100 top-K diagnostic 与 full-pool 主实验的平均目标值差异为 {n100_compare.loc[n100_compare['Metric'] == 'Objective value', 'Delta'].iloc[0]}。",
            "- 若该差异远低于 0.1%，就能把它作为 top-K 在更大规模上无明显解质量损失的间接证据。",
            "",
            "## 6. 对论文的修改建议",
            "",
            "### §4.5.3 增补段落建议（核心）",
            "- 明确说明 follow-up 覆盖的是 `N=100 (diagnostic), 200, 500` 的 iterative top-K pricing with a 60-iteration budget，而不是 full-pool shortcut。",
            "- 用 `identical / alternative_optima / improved` 三分法报告结果，避免把“目标值不变但 plan 变化”误写成“富化无效”。",
            "- 如果 `columns_used_from_enrichment > 0` 的比例可观，应明确写出“富化列被 IRMP 实际采用，但目标值保持不变”。",
            "",
            "### §2.6 contribution 措辞建议",
            "- 不要写成 novel CG techniques；更稳妥的是强调在论文采用的 top-K iterative pricing 配置下，列池富化没有暴露出可观的目标值改进。",
            "",
            "### §5.1.3 实现细节说明的增补",
            "- 建议固定表述为：for N>100, the implementation uses iterative top-K pricing with K=3 and a 60-iteration budget。",
            "",
            "### 附录建议",
            "- 把 follow-up 的详细表、equivalence 分布和 N=100 cross-check 放进附录；正文只保留 1 段核心总结。",
            "",
            "## 7. 情景判定",
            "",
            f"- 判定结果：{scenario_label}",
            f"- 解释：{scenario_text}",
        ]
    )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    baseline_df, enriched_df = load_raw_rows(results_dir / "raw")
    baseline_df = unpack_dict_column(baseline_df, "mode_distribution", "mode_distribution")
    enriched_df = unpack_dict_column(enriched_df, "columns_by_mode_added", "columns_added_by_mode")
    agg = aggregate_results(baseline_df, enriched_df)
    agg.to_csv(results_dir / "aggregated.csv", index=False)
    enriched_df.sort_values(["N", "seed", "epsilon_pct"]).to_csv(results_dir / "per_seed_detailed.csv", index=False)
    eq_summary = build_equivalence_summary(enriched_df)
    eq_summary.to_csv(results_dir / "equivalence_type_summary.csv", index=False)
    n100_compare = build_n100_compare(agg, baseline_df)
    build_table_tex(agg, results_dir / "table_enrichment_followup.tex")
    save_three_panel_figure(agg, eq_summary, enriched_df, results_dir)
    save_usage_figure(agg, results_dir)
    build_summary(results_dir, baseline_df, enriched_df, agg, eq_summary, n100_compare)

    with (results_dir / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 6 complete: aggregated outputs, figures, and summary generated.\n")


if __name__ == "__main__":
    main()

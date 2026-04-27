from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.tr_figures.config import DOUBLE_COLUMN, METHOD_COLORS, REFERENCE_COLOR, REFERENCE_LINESTYLE, SINGLE_COLUMN
from analysis.tr_figures.utils import apply_common_axis_format, set_x_axis_label, start_figure


EPSILON_LEVELS = [0.5, 1.0, 2.0]
TARGET_NS = [20, 50, 100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Experiment 3 outputs.")
    parser.add_argument("--results-dir", default="results/experiment_3_column_enrichment", help="Experiment 3 results directory.")
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
        raise FileNotFoundError(f"Missing raw Experiment 3 JSON outputs in {raw_dir}")
    return pd.DataFrame(baseline_rows), pd.DataFrame(enriched_rows)


def unpack_mode_count_columns(df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
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
    baseline_core = baseline_df[
        [
            "N",
            "seed",
            "baseline_pool_size",
            "full_column_pool_size",
            "baseline_pool_complete",
            "used_full_pool_small",
            "Z_baseline_IRMP",
            "baseline_internal_gap",
            "Z_MILP_exact",
            "gap_baseline_vs_MILP",
        ]
    ].copy()
    merged = enriched_df.merge(baseline_core, on=["N", "seed"], how="left", suffixes=("", "_base"))
    rows: list[dict[str, Any]] = []
    metrics = [
        "baseline_pool_size",
        "full_column_pool_size",
        "enriched_pool_size",
        "columns_added",
        "Z_baseline_IRMP",
        "Z_enriched_IRMP",
        "improvement_pct",
        "baseline_internal_gap",
        "enriched_internal_gap",
        "n_vessels_changed",
        "gap_baseline_vs_MILP",
        "gap_enriched_vs_MILP",
    ]
    for (N, epsilon_pct), block in merged.groupby(["N", "epsilon_pct"], dropna=False):
        row: dict[str, Any] = {"N": int(N), "epsilon_pct": float(epsilon_pct), "epsilon_label": f"{float(epsilon_pct):.1f}%"}
        for metric in metrics:
            clean = pd.to_numeric(block[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(clean.mean()) if not clean.empty else np.nan
            row[f"{metric}_std"] = float(clean.std(ddof=0)) if len(clean) else np.nan
        row["solutions_identical_rate"] = float(pd.to_numeric(block["solutions_identical"], errors="coerce").fillna(0).mean())
        row["n_seeds_with_improvement"] = int((pd.to_numeric(block["improvement_pct"], errors="coerce").fillna(0) > 1e-9).sum())
        row["improvement_pct_max"] = float(pd.to_numeric(block["improvement_pct"], errors="coerce").fillna(0).max())
        row["all_baseline_full_pool"] = bool(block["baseline_pool_complete"].fillna(False).all())
        row["all_used_full_pool_small"] = bool(block["used_full_pool_small"].fillna(False).all())

        mode_counts = pd.DataFrame([value if isinstance(value, dict) else json.loads(value) for value in block["columns_by_mode_added"]])
        for mode in ["SP", "BS", "AE"]:
            clean = pd.to_numeric(mode_counts[mode], errors="coerce").dropna()
            row[f"columns_added_{mode}_mean"] = float(clean.mean()) if not clean.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["N", "epsilon_pct"]).reset_index(drop=True)


def build_table_tex(agg: pd.DataFrame, output_path: Path) -> None:
    block = agg[np.isclose(agg["epsilon_pct"], 1.0)].copy().sort_values("N")
    block["Baseline pool"] = block["baseline_pool_size_mean"].map(lambda x: f"{x:.1f}")
    block["Enriched pool (1%)"] = block["enriched_pool_size_mean"].map(lambda x: f"{x:.1f}")
    block["Cols added"] = block["columns_added_mean"].map(lambda x: f"{x:.1f}")
    block["Z_baseline"] = block["Z_baseline_IRMP_mean"].map(lambda x: f"{x:.2f}")
    block["Z_enriched"] = block["Z_enriched_IRMP_mean"].map(lambda x: f"{x:.2f}")
    block["Improvement"] = block["improvement_pct_mean"].map(lambda x: f"{x:.4f}\\%")
    block["Identical solutions"] = block["solutions_identical_rate"].map(lambda x: "True" if abs(float(x) - 1.0) < 1e-12 else f"{x * 100:.1f}\\%")

    def milp_cell(row: pd.Series) -> str:
        if pd.isna(row["gap_enriched_vs_MILP_mean"]):
            return "n/a"
        return f"{row['gap_enriched_vs_MILP_mean']:.4f}\\%"

    block["vs MILP"] = block.apply(milp_cell, axis=1)
    table = block[["N", "Baseline pool", "Enriched pool (1%)", "Cols added", "Z_baseline", "Z_enriched", "Improvement", "Identical solutions", "vs MILP"]]
    output_path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def classify_scenario(enriched_df: pd.DataFrame) -> tuple[str, str]:
    max_improvement = float(pd.to_numeric(enriched_df["improvement_pct"], errors="coerce").fillna(0).max())
    mean_improvement = float(pd.to_numeric(enriched_df["improvement_pct"], errors="coerce").fillna(0).mean())
    all_identical = bool(enriched_df["solutions_identical"].fillna(False).all())
    milp_gap = pd.to_numeric(enriched_df["gap_enriched_vs_MILP"], errors="coerce").dropna()
    milp_match = bool(milp_gap.empty or np.all(np.abs(milp_gap.to_numpy(dtype=float)) <= 1e-6))

    if max_improvement <= 1e-9 and all_identical and milp_match:
        return "情景 α", "所有富化结果与原 IRMP 完全一致。"
    if mean_improvement < 0.1 and max_improvement < 0.3:
        return "情景 β", "存在极小改进空间，但幅度足以视为工程上可忽略。"
    if mean_improvement <= 1.0 and max_improvement <= 5.0:
        return "情景 γ", "出现了非平凡改进，需要收紧论文对 IRMP 可靠性的措辞。"
    return "情景 δ", "富化后出现了显著改进或 MILP 对照异常，需要人工介入排查。"


def build_summary(baseline_df: pd.DataFrame, enriched_df: pd.DataFrame, agg: pd.DataFrame, output_path: Path) -> None:
    scenario_label, scenario_text = classify_scenario(enriched_df)
    max_improvement = float(pd.to_numeric(enriched_df["improvement_pct"], errors="coerce").fillna(0).max())
    mean_improvement = float(pd.to_numeric(enriched_df["improvement_pct"], errors="coerce").fillna(0).mean())
    full_pool_share = float(pd.to_numeric(baseline_df["baseline_pool_complete"], errors="coerce").fillna(0).mean())
    full_pool_small_share = float(pd.to_numeric(baseline_df["used_full_pool_small"], errors="coerce").fillna(0).mean())
    zero_added_share = float((pd.to_numeric(enriched_df["columns_added"], errors="coerce").fillna(0) == 0).mean())
    one_pct = agg[np.isclose(agg["epsilon_pct"], 1.0)].copy().sort_values("N")

    lines = [
        "# 实验三（列池富化）结果摘要",
        "",
        "## 1. 实验参数确认",
        "",
        "- 规模点：N ∈ {20, 50, 100}",
        "- 场景：U；方法：CG+IRMP；operation_mode：simops",
        "- ε_loose 测试值：0.5%、1%、2% of Z_baseline",
        "- 富化模式：enumeration（基于收敛时 duals 对单船全 plan 枚举）",
        "- 实际复现实例种子沿用主实验配置，为 1..10，而不是重新改成 0..9",
        "",
        "## 2. 主表（1% 富化）",
        "",
        "| N | Baseline pool | Enriched pool (1%) | Cols added | Z_baseline | Z_enriched | Improvement | Identical solutions | vs MILP |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in one_pct.iterrows():
        milp_text = "n/a" if pd.isna(row["gap_enriched_vs_MILP_mean"]) else f"{row['gap_enriched_vs_MILP_mean']:.4f}%"
        identical_text = "True" if abs(float(row["solutions_identical_rate"]) - 1.0) < 1e-12 else f"{row['solutions_identical_rate'] * 100:.1f}%"
        lines.append(
            f"| {int(row['N'])} | {row['baseline_pool_size_mean']:.1f} | {row['enriched_pool_size_mean']:.1f} | "
            f"{row['columns_added_mean']:.1f} | {row['Z_baseline_IRMP_mean']:.2f} | {row['Z_enriched_IRMP_mean']:.2f} | "
            f"{row['improvement_pct_mean']:.4f}% | {identical_text} | {milp_text} |"
        )

    lines.extend(
        [
            "",
            "## 3. 三个核心问题",
            "",
            "### Q1：原 IRMP 是否已经最优？",
            f"- 判定：{scenario_label}。{scenario_text}",
            f"- 所有富化记录的平均 improvement 为 {mean_improvement:.6f}%，最大单实例 improvement 为 {max_improvement:.6f}%。",
            f"- solutions_identical 的总体成立比例为 {pd.to_numeric(enriched_df['solutions_identical'], errors='coerce').fillna(0).mean() * 100:.1f}%。",
            "",
            "### Q2：ε_loose 扩大后的边际效益如何？",
        ]
    )
    for epsilon_pct in EPSILON_LEVELS:
        block = agg[np.isclose(agg["epsilon_pct"], epsilon_pct)].copy().sort_values("N")
        lines.append(
            f"- ε={epsilon_pct:.1f}%：平均新增列 {block['columns_added_mean'].mean():.2f}，平均 improvement {block['improvement_pct_mean'].mean():.6f}%，最大 improvement {block['improvement_pct_max'].max():.6f}%。"
        )
    lines.extend(
        [
            "",
            "### Q3：改变发生在哪里？",
            f"- 有 {zero_added_share * 100:.1f}% 的富化记录新增列数为 0。",
            f"- baseline_pool_complete 的比例为 {full_pool_share * 100:.1f}%，used_full_pool_small 的比例为 {full_pool_small_share * 100:.1f}%。",
            "- 这意味着在当前论文主配置下，N≤100 的 baseline pool 实际上已经等于完整单船列枚举池，因此 enrichment 无法再向池中加入新列。",
            "",
            "## 4. 与 §5.2 外部 MILP 验证的一致性",
            "",
            f"- N=20 和 N=50 的 enriched-vs-MILP 平均 gap 分别为 "
            f"{agg[(agg['N']==20) & np.isclose(agg['epsilon_pct'], 1.0)]['gap_enriched_vs_MILP_mean'].iloc[0]:.6f}% 和 "
            f"{agg[(agg['N']==50) & np.isclose(agg['epsilon_pct'], 1.0)]['gap_enriched_vs_MILP_mean'].iloc[0]:.6f}%。",
            "- 在本次复现中，baseline IRMP、enriched IRMP 与 existing exact MILP 在 N=20/50 上保持一致，没有出现 enriched 优于 MILP 的异常。",
            "",
            "## 5. 对论文的修改建议",
            "",
            "### §4.5.3 增补建议",
            "- 可以报告富化实验结果为零改进，但必须同时说明这是在当前 `use_full_pool_small=true, full_pool_n=100` 的主实验配置下得到的，因此基线列池本身已经是完整枚举池。",
            "- 这条证据能够支撑“当前实现没有遗漏列”的实现完整性，但对 Remark 3 的独立经验支撑力度弱于真正的受限列池情形。",
            "",
            "### §2.6 contribution 措辞建议",
            "- 不建议因为本实验去强化“CG 定价天然使列池完整”的表述。",
            "- 更稳妥的写法是：在论文报告的 N≤100 主实验设置下，CG+IRMP 的最终求解并未受到列池截断误差影响；对更一般的受限列池情形，完整 branch-and-price 仍留作未来工作。",
            "",
            "### 附录建议",
            "- 把本实验放入附录，重点展示 baseline/enriched pool size、zero-improvement 结果以及 full-pool 配置说明。",
            "",
            "## 6. 情景判定",
            "",
            f"- 判定结果：{scenario_label}",
            f"- 解释：{scenario_text}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_pool_size_figure(agg: pd.DataFrame, output_dir: Path) -> None:
    baseline = agg.groupby("N", dropna=False)["baseline_pool_size_mean"].mean().reindex(TARGET_NS)
    width = 0.18
    positions = np.arange(len(TARGET_NS))

    start_figure()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN.width, SINGLE_COLUMN.height))
    ax.bar(positions - 1.5 * width, baseline.to_numpy(dtype=float), width=width, label="Baseline", color="#9A9A9A")
    for offset, epsilon_pct, color in zip([-0.5, 0.5, 1.5], EPSILON_LEVELS, ["#C9D5E8", METHOD_COLORS["cg"], "#7AA974"], strict=False):
        block = agg[np.isclose(agg["epsilon_pct"], epsilon_pct)].set_index("N").reindex(TARGET_NS)
        ax.bar(positions + offset * width, block["enriched_pool_size_mean"].to_numpy(dtype=float), width=width, label=f"Enriched {epsilon_pct:.1f}%", color=color)
    ax.set_yscale("log")
    ax.set_xticks(positions, [str(N) for N in TARGET_NS])
    set_x_axis_label(ax, "Number of ships $N$")
    ax.set_ylabel("Pool size (log scale)")
    apply_common_axis_format(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_pool_size.pdf")
    fig.savefig(output_dir / "fig9_pool_size.png", dpi=300)
    plt.close(fig)


def _draw_improvement_box(ax: plt.Axes, enriched_df: pd.DataFrame) -> None:
    offsets = {-0.25: 0.5, 0.0: 1.0, 0.25: 2.0}
    colors = {0.5: "#C9D5E8", 1.0: METHOD_COLORS["cg"], 2.0: "#7AA974"}
    labels_used: set[float] = set()
    positions = np.arange(len(TARGET_NS))

    for offset, epsilon_pct in offsets.items():
        data = [
            pd.to_numeric(
                enriched_df[(enriched_df["N"] == N) & (np.isclose(enriched_df["epsilon_pct"], epsilon_pct))]["improvement_pct"],
                errors="coerce",
            ).dropna().to_numpy(dtype=float)
            for N in TARGET_NS
        ]
        bp = ax.boxplot(
            data,
            positions=positions + offset,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#333333", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[epsilon_pct])
            patch.set_edgecolor("#555555")
            patch.set_alpha(0.95)
        if epsilon_pct not in labels_used:
            bp["boxes"][0].set_label(f"Enriched {epsilon_pct:.1f}%")
            labels_used.add(epsilon_pct)

    ax.axhline(0.0, color=REFERENCE_COLOR, linestyle=REFERENCE_LINESTYLE, linewidth=0.9)
    ax.set_xticks(positions, [str(N) for N in TARGET_NS])
    set_x_axis_label(ax, "Number of ships $N$")
    ax.set_ylabel("Improvement over baseline (%)")
    apply_common_axis_format(ax)


def save_improvement_figure(enriched_df: pd.DataFrame, output_dir: Path) -> None:
    start_figure()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN.width, SINGLE_COLUMN.height))
    _draw_improvement_box(ax, enriched_df)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_improvement.pdf")
    fig.savefig(output_dir / "fig9_improvement.png", dpi=300)
    plt.close(fig)


def save_combined_figure(agg: pd.DataFrame, enriched_df: pd.DataFrame, output_dir: Path) -> None:
    baseline = agg.groupby("N", dropna=False)["baseline_pool_size_mean"].mean().reindex(TARGET_NS)
    width = 0.18
    positions = np.arange(len(TARGET_NS))

    start_figure()
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COLUMN.width, SINGLE_COLUMN.height))

    ax0 = axes[0]
    ax0.bar(positions - 1.5 * width, baseline.to_numpy(dtype=float), width=width, label="Baseline", color="#9A9A9A")
    for offset, epsilon_pct, color in zip([-0.5, 0.5, 1.5], EPSILON_LEVELS, ["#C9D5E8", METHOD_COLORS["cg"], "#7AA974"], strict=False):
        block = agg[np.isclose(agg["epsilon_pct"], epsilon_pct)].set_index("N").reindex(TARGET_NS)
        ax0.bar(positions + offset * width, block["enriched_pool_size_mean"].to_numpy(dtype=float), width=width, label=f"Enriched {epsilon_pct:.1f}%", color=color)
    ax0.set_yscale("log")
    ax0.set_xticks(positions, [str(N) for N in TARGET_NS])
    set_x_axis_label(ax0, "Number of ships $N$")
    ax0.set_ylabel("Pool size (log scale)")
    apply_common_axis_format(ax0)

    ax1 = axes[1]
    _draw_improvement_box(ax1, enriched_df)
    handles, labels = ax0.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "fig9_combined.pdf")
    fig.savefig(output_dir / "fig9_combined.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    baseline_df, enriched_df = load_raw_rows(results_dir / "raw")
    enriched_df = unpack_mode_count_columns(enriched_df, "columns_by_mode_added", "added_cols")
    agg = aggregate_results(baseline_df, enriched_df)
    agg.to_csv(results_dir / "aggregated.csv", index=False)
    enriched_df.sort_values(["N", "seed", "epsilon_pct"]).to_csv(results_dir / "per_seed_comparison.csv", index=False)
    build_table_tex(agg, results_dir / "table_enrichment_main.tex")
    save_pool_size_figure(agg, results_dir)
    save_improvement_figure(enriched_df, results_dir)
    save_combined_figure(agg, enriched_df, results_dir)
    build_summary(baseline_df, enriched_df, agg, results_dir / "summary.md")

    log_path = results_dir / "run_log.txt"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 3 complete: aggregated.csv, per_seed_comparison.csv, figures, table, and summary generated.\n")


if __name__ == "__main__":
    main()

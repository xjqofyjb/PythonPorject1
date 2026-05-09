"""Generate manuscript-ready Figure 3 from corrected CSV outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_OUT = ROOT / "results" / "revised" / "manuscript"
FIG_OUT = ROOT / "figures" / "revised" / "manuscript"

SMALL_SOURCE = ROOT / "results" / "revised" / "table8_revised.csv"
CONTROLLED_SOURCE = ROOT / "results" / "revised" / "final_check" / "table8_final_controlled.csv"
VALIDATION_SOURCE = ROOT / "results" / "revised" / "final_check" / "table8_validation_report.md"
DIAGNOSTIC_SOURCE = ROOT / "results" / "revised" / "final_check" / "final_check_diagnostic_report.md"

METHOD_ORDER = [
    "CG+IR",
    "Rolling-Horizon",
    "Fix-and-Optimize",
    "Restricted-CG",
    "FIFO",
    "Greedy",
]

METHOD_LABELS = {
    "CG+IR": "CG+IR",
    "rolling_horizon": "Rolling-Horizon",
    "Rolling-Horizon": "Rolling-Horizon",
    "fix_and_optimize": "Fix-and-Optimize",
    "Fix-and-Optimize": "Fix-and-Optimize",
    "Restricted-CG": "Restricted-CG",
    "FIFO": "FIFO",
    "Greedy": "Greedy",
}


def fail_missing(missing: list[Path]) -> None:
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    report = RESULT_OUT / "fig3_missing_data_report.md"
    lines = [
        "# Figure 3 Missing Data Report",
        "",
        "Figure 3 was not generated because required source files are missing.",
        "",
        *[f"- `{path.relative_to(ROOT)}`" for path in missing],
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    raise SystemExit(f"Missing required data. See {report}")


def standardize_small(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["N"].isin([20, 50, 100])].copy()
    out["method"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    out = out[out["method"].isin(METHOD_ORDER)].copy()
    out = out.rename(
        columns={
            "obj_mean": "objective_mean",
            "obj_std": "objective_std",
            "gap_pct_mean": "pool_gap_pct_mean",
        }
    )
    out["runtime_std"] = np.nan
    out["rel_gap_to_CG_mean"] = np.nan
    out["status_success_rate"] = out.get("success_rate", np.nan)
    out["pricing_converged_rate"] = np.where(
        out["method"].eq("CG+IR") & out["gap_type"].eq("Full-CG LP-IP gap"),
        1.0,
        np.nan,
    )
    out["objective_stabilized_rate"] = np.nan
    out["source_class"] = "corrected_small_full_pricing"
    return out


def standardize_controlled(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["N"].isin([200, 500])].copy()
    out["method"] = out["method"].map(METHOD_LABELS).fillna(out["method"])
    out = out[out["method"].isin(METHOD_ORDER)].copy()
    out["source_class"] = "final_controlled_replacement"
    return out


def add_relative_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key = ["N", "scenario"]
    cg = out[out["method"].eq("CG+IR")][key + ["objective_mean"]].rename(columns={"objective_mean": "cg_objective_mean"})
    out = out.merge(cg, on=key, how="left")
    out["rel_gap_to_CG_mean"] = (
        (pd.to_numeric(out["objective_mean"], errors="coerce") - pd.to_numeric(out["cg_objective_mean"], errors="coerce"))
        / pd.to_numeric(out["cg_objective_mean"], errors="coerce").abs().clip(lower=1.0)
        * 100.0
    )
    return out


def aggregate_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (N, method), g in df.groupby(["N", "method"], dropna=False):
        scenarios = ",".join(sorted(set(map(str, g["scenario"].dropna()))))
        rows.append(
            {
                "N": int(N),
                "scale_label": f"{int(N)} (U only)" if int(N) == 500 else str(int(N)),
                "method": method,
                "objective_mean": pd.to_numeric(g["objective_mean"], errors="coerce").mean(),
                "objective_std": pd.to_numeric(g["objective_std"], errors="coerce").mean(),
                "rel_gap_to_CG_mean": pd.to_numeric(g["rel_gap_to_CG_mean"], errors="coerce").mean(),
                "runtime_mean": pd.to_numeric(g["runtime_mean"], errors="coerce").mean(),
                "runtime_std": pd.to_numeric(g.get("runtime_std", np.nan), errors="coerce").mean(),
                "scenarios": scenarios,
                "cg_status": ";".join(sorted(set(map(str, g.get("cg_status", pd.Series(dtype=str)).dropna())))),
                "gap_type": ";".join(sorted(set(map(str, g.get("gap_type", pd.Series(dtype=str)).dropna())))),
                "pricing_converged_rate": pd.to_numeric(g.get("pricing_converged_rate", np.nan), errors="coerce").mean(),
                "source_class": ";".join(sorted(set(map(str, g["source_class"].dropna())))),
            }
        )
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    out = out.sort_values(["N", "method"])
    return out


def plot(agg: pd.DataFrame, combined: pd.DataFrame) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "CG+IR": "#1b4d89",
        "Rolling-Horizon": "#2a9d8f",
        "Fix-and-Optimize": "#e76f51",
        "Restricted-CG": "#7b2cbf",
        "FIFO": "#6c757d",
        "Greedy": "#f4a261",
    }
    scales = [20, 50, 100, 200, 500]
    labels = ["20", "50", "100", "200", "500\n(U only)"]
    x = np.arange(len(scales))
    width = 0.12

    fig, axs = plt.subplots(2, 2, figsize=(7.6, 5.7), constrained_layout=True)
    ax_obj, ax_gap, ax_rt, ax_diag = axs.ravel()

    for idx, method in enumerate(METHOD_ORDER):
        sub = agg[agg["method"].eq(method)].set_index("N")
        offset = (idx - (len(METHOD_ORDER) - 1) / 2) * width
        vals = [sub.loc[N, "objective_mean"] / 1000.0 if N in sub.index else np.nan for N in scales]
        gaps = [sub.loc[N, "rel_gap_to_CG_mean"] if N in sub.index else np.nan for N in scales]
        rts = [sub.loc[N, "runtime_mean"] if N in sub.index else np.nan for N in scales]
        ax_obj.bar(x + offset, vals, width=width, color=colors[method], label=method)
        ax_gap.bar(x + offset, gaps, width=width, color=colors[method])
        ax_rt.bar(x + offset, rts, width=width, color=colors[method])

    for ax in [ax_obj, ax_gap, ax_rt]:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        ax.axvline(2.5, color="#333333", linestyle=":", linewidth=1.0, alpha=0.7)

    ax_obj.set_title("(a) Objective by scale")
    ax_obj.set_ylabel("Mean objective (thousand $, log scale)")
    ax_obj.set_yscale("log")
    ax_obj.legend(ncol=2, frameon=False, loc="upper left")

    ax_gap.set_title("(b) Relative gap to CG+IR")
    ax_gap.set_ylabel("Mean gap to CG+IR (%)")
    ax_gap.axhline(0, color="#222222", linewidth=0.8)

    ax_rt.set_title("(c) Runtime")
    ax_rt.set_ylabel("Mean runtime (s)")
    ax_rt.set_yscale("log")
    ax_rt.set_ylim(bottom=0.002)

    ax_diag.axis("off")
    cg = combined[combined["method"].eq("CG+IR")].copy()
    small_ok = cg[cg["N"].isin([20, 50, 100])]["gap_type"].eq("Full-CG LP-IP gap").all()
    large_ok = cg[cg["N"].isin([200, 500])]["gap_type"].eq("Pool LP-IP gap").all()
    diag_lines = [
        "(d) Gap interpretation and data scope",
        "",
        f"N=20/50/100: {'Full-CG LP-IP gap' if small_ok else 'check source'}",
        "  Pricing converged for CG+IR small-scale rows.",
        "",
        f"N=200/500: {'Pool LP-IP gap' if large_ok else 'check source'}",
        "  Budgeted generated-pool evidence, not",
        "  complete-column global optimality.",
        "",
        "N=500: scenario U only.",
        "Old weak quick-CG outputs excluded.",
    ]
    ax_diag.text(
        0.02,
        0.98,
        "\n".join(diag_lines),
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#cfcfcf"),
    )

    fig.savefig(FIG_OUT / "fig3_benchmark_final.png")
    fig.savefig(FIG_OUT / "fig3_benchmark_final.pdf")
    plt.close(fig)


def write_docs(combined: pd.DataFrame, agg: pd.DataFrame) -> None:
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    caption = (
        "Figure 3. Benchmark performance across instance scales using corrected and final-controlled outputs. "
        "Panels show (a) mean objective on a log scale, (b) mean relative gap to CG+IR, (c) mean runtime on a log scale, "
        "and (d) the gap interpretation used for each scale. N=20, 50, and 100 use corrected full-pricing "
        "CG+IR rows and report Full-CG LP-IP gaps. N=200 and N=500 use final controlled replacement results "
        "and report Pool LP-IP gaps; these are generated-pool LP-IP gaps and should not be interpreted as "
        "complete-column global optimality certificates. N=500 is scenario U only."
    )
    (RESULT_OUT / "fig3_benchmark_final_caption.txt").write_text(caption + "\n", encoding="utf-8")

    sources = [
        "# Figure 3 Data Sources",
        "",
        "## CSV inputs",
        f"- `{SMALL_SOURCE.relative_to(ROOT)}`: corrected N=20/50/100 benchmark rows.",
        f"- `{CONTROLLED_SOURCE.relative_to(ROOT)}`: final controlled N=200 U/P/L and N=500 U-only replacement rows.",
        "",
        "## Diagnostic inputs",
        f"- `{VALIDATION_SOURCE.relative_to(ROOT)}`",
        f"- `{DIAGNOSTIC_SOURCE.relative_to(ROOT)}`",
        "",
        "## Inclusion rules",
        "- Old weak quick-CG outputs were not read.",
        "- N=20/50/100 rows are taken only from corrected small-scale benchmark summaries.",
        "- N=200 and N=500 rows are taken only from final controlled replacement outputs.",
        "- N=500 is labeled U-only.",
        "- CG+IR gap labels are preserved: Full-CG LP-IP gap for N<=100, Pool LP-IP gap for N=200/500.",
        "- The small-scale corrected summary available in this repository uses seeds 1-2; the final controlled replacement uses seeds 1-10 for N=200/500.",
        "",
        "## Aggregation",
        "- For N=20/50/100 and N=200, U/P/L scenario summaries are averaged equally by method and scale.",
        "- For N=500, only scenario U is available and shown.",
        "",
        "## Methods included",
        *[f"- {m}" for m in METHOD_ORDER if m in set(agg["method"].astype(str))],
    ]
    (RESULT_OUT / "fig3_data_sources.md").write_text("\n".join(sources) + "\n", encoding="utf-8")


def main() -> None:
    required = [SMALL_SOURCE, CONTROLLED_SOURCE, VALIDATION_SOURCE, DIAGNOSTIC_SOURCE]
    missing = [path for path in required if not path.exists()]
    if missing:
        fail_missing(missing)

    small = standardize_small(pd.read_csv(SMALL_SOURCE))
    controlled = standardize_controlled(pd.read_csv(CONTROLLED_SOURCE))
    combined = pd.concat([small, controlled], ignore_index=True, sort=False)
    combined = add_relative_gaps(combined)

    cg = combined[combined["method"].eq("CG+IR")]
    required_small = {20, 50, 100}
    if set(cg[cg["N"].isin(required_small)]["N"]) != required_small:
        fail_missing([SMALL_SOURCE])
    if not cg[cg["N"].isin([20, 50, 100])]["gap_type"].eq("Full-CG LP-IP gap").all():
        raise SystemExit("Small-scale CG+IR rows are not consistently labeled Full-CG LP-IP gap.")
    if not cg[cg["N"].isin([200, 500])]["gap_type"].eq("Pool LP-IP gap").all():
        raise SystemExit("N=200/N=500 CG+IR rows are not consistently labeled Pool LP-IP gap.")

    agg = aggregate_for_plot(combined)
    RESULT_OUT.mkdir(parents=True, exist_ok=True)
    agg.to_csv(RESULT_OUT / "fig3_benchmark_final_plot_data.csv", index=False)
    plot(agg, combined)
    write_docs(combined, agg)


if __name__ == "__main__":
    main()

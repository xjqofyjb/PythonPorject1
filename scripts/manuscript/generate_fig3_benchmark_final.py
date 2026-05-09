"""
Regenerate manuscript Figure 3 (benchmark across scales) from corrected and
final-controlled CSV outputs only. No hard-coded benchmark numbers; old quick-CG
results are excluded.

Inputs (read-only):
  - results/revised/table8_revised.csv                          (N=20/50/100, full_pricing_converged CG+IR)
  - results/revised/final_check/table8_final_controlled.csv     (N=200 U/P/L, N=500 U-only)
  - results/revised/final_check/table8_validation_report.md     (validation evidence)
  - results/revised/final_check/final_check_diagnostic_report.md (final caveats)

Outputs:
  - figures/revised/manuscript/fig3_benchmark_final.png
  - figures/revised/manuscript/fig3_benchmark_final.pdf
  - results/revised/manuscript/fig3_benchmark_final_plot_data.csv
  - results/revised/manuscript/fig3_benchmark_per_scenario.csv
  - results/revised/manuscript/fig3_benchmark_final_caption.txt
  - results/revised/manuscript/fig3_data_sources.md

Visual rules:
  - LaTeX-ready: no figure suptitle, no bottom note (caption is provided in
    LaTeX). The gap-type legend lives inside panel (d).
  - N = 500 is labeled "500 (U-only)".
  - Palette is matched to the earlier manuscript figures so this figure
    reads as part of the same family.
"""

from __future__ import annotations

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC_SMALL = os.path.join(ROOT, "results", "revised", "table8_revised.csv")
SRC_LARGE = os.path.join(ROOT, "results", "revised", "final_check",
                         "table8_final_controlled.csv")
SRC_VAL   = os.path.join(ROOT, "results", "revised", "final_check",
                         "table8_validation_report.md")
SRC_DIAG  = os.path.join(ROOT, "results", "revised", "final_check",
                         "final_check_diagnostic_report.md")

OUT_FIG_DIR = os.path.join(ROOT, "figures", "revised", "manuscript")
OUT_DAT_DIR = os.path.join(ROOT, "results", "revised", "manuscript")

METHOD_ORDER = ["CG+IR", "Restricted-CG", "Rolling-Horizon",
                "Fix-and-Optimize", "FIFO", "Greedy"]

# Palette matched to the earlier manuscript figures (panels a/b in the
# reference). Blues for the CG family, greens for horizon /
# fix-and-optimize, gray for FIFO, coral/orange for Greedy.
METHOD_COLOR = {
    "CG+IR":            "#4F8FC0",   # medium blue
    "Restricted-CG":    "#9DC3E6",   # light blue
    "Rolling-Horizon":  "#70AD47",   # medium green
    "Fix-and-Optimize": "#A9D18E",   # light green
    "FIFO":             "#7F7F7F",   # neutral gray
    "Greedy":           "#ED7D31",   # coral / orange
}

METHOD_MARKER = {
    "CG+IR":            "o",
    "Restricted-CG":    "s",
    "Rolling-Horizon":  "^",
    "Fix-and-Optimize": "D",
    "FIFO":             "v",
    "Greedy":           "P",
}

RENAME_METHOD = {
    "fix_and_optimize": "Fix-and-Optimize",
    "rolling_horizon":  "Rolling-Horizon",
}


def stop_with_missing_data_report(missing):
    os.makedirs(OUT_DAT_DIR, exist_ok=True)
    report = [
        "# Figure 3 missing-data report",
        "",
        "Figure 3 was NOT regenerated because required source files are missing.",
        "",
        "## Missing files",
    ] + [f"- `{p}`" for p in missing] + [
        "",
        "## Action",
        "Re-run the controlled benchmark / corrected small-scale benchmark and",
        "place the CSV outputs at the expected paths before retrying.",
        "",
    ]
    out = os.path.join(OUT_DAT_DIR, "fig3_missing_data_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[STOP] wrote missing-data report -> {out}", file=sys.stderr)
    sys.exit(1)


def load_and_filter():
    missing = [p for p in (SRC_SMALL, SRC_LARGE, SRC_VAL, SRC_DIAG)
               if not os.path.exists(p)]
    if missing:
        stop_with_missing_data_report(missing)

    small = pd.read_csv(SRC_SMALL)
    small = small[small["N"].isin([20, 50, 100])].copy()
    small["method"] = small["method"].replace(RENAME_METHOD)

    cg_small = small[small["method"] == "CG+IR"]
    if not (cg_small["gap_type"] == "Full-CG LP-IP gap").all():
        raise RuntimeError("Small-scale CG+IR rows must use 'Full-CG LP-IP gap'.")
    if not (cg_small["cg_status"] == "full_pricing_converged").all():
        raise RuntimeError("Small-scale CG+IR rows must be 'full_pricing_converged'.")

    large = pd.read_csv(SRC_LARGE)
    allow = (((large["N"] == 200) & large["scenario"].isin(["U", "P", "L"]))
             | ((large["N"] == 500) & (large["scenario"] == "U")))
    large = large[allow].copy()
    cg_large = large[large["method"] == "CG+IR"]
    if not (cg_large["gap_type"] == "Pool LP-IP gap").all():
        raise RuntimeError("Large-scale CG+IR rows must use 'Pool LP-IP gap'.")

    return small, large


def _add_rel_gap(df: pd.DataFrame, obj_col: str) -> pd.DataFrame:
    df = df.copy()
    base = df[df["method"] == "CG+IR"].set_index(["N", "scenario"])[obj_col]
    df["rel_gap_to_CG"] = df.apply(
        lambda r: 100.0 * (r[obj_col] - base.loc[(r["N"], r["scenario"])])
                  / base.loc[(r["N"], r["scenario"])], axis=1)
    return df


def build_combined_and_aggregate(small, large):
    small = _add_rel_gap(small, "obj_mean")
    large = _add_rel_gap(large, "objective_mean")

    small_std = small.rename(columns={
        "obj_mean":     "objective_mean",
        "obj_std":      "objective_std",
        "runtime_mean": "runtime_mean",
    })
    small_std["runtime_std"] = np.nan
    small_std["source_class"] = "corrected_small_full_pricing"

    large_std = large.copy()
    large_std["source_class"] = "final_controlled_replacement"

    keep_cols = ["N", "scenario", "method", "objective_mean", "objective_std",
                 "runtime_mean", "runtime_std", "rel_gap_to_CG",
                 "cg_status", "gap_type", "source_class"]
    combined = pd.concat([small_std[keep_cols], large_std[keep_cols]],
                         ignore_index=True)
    combined = combined[combined["method"].isin(METHOD_ORDER)].copy()

    rows = []
    for (N, method), g in combined.groupby(["N", "method"]):
        scale_label = f"{N}" if N != 500 else "500 (U-only)"
        rows.append({
            "N": int(N),
            "scale_label": scale_label,
            "method": method,
            "objective_mean": g["objective_mean"].mean(),
            "objective_std":  g["objective_std"].mean(),
            "rel_gap_to_CG_mean": g["rel_gap_to_CG"].mean(),
            "runtime_mean": g["runtime_mean"].mean(),
            "runtime_std":  g["runtime_std"].mean()
                            if g["runtime_std"].notna().any() else np.nan,
            "scenarios": ",".join(sorted(g["scenario"].unique())),
            "cg_status": ";".join(sorted({s for s in g["cg_status"].dropna().unique()}))
                         if g["cg_status"].notna().any() else "",
            "gap_type":  ";".join(sorted({s for s in g["gap_type"].dropna().unique()}))
                         if g["gap_type"].notna().any() else "",
            "source_class": ";".join(sorted(g["source_class"].unique())),
        })
    agg = pd.DataFrame(rows)
    rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    agg["_mr"] = agg["method"].map(rank)
    agg = agg.sort_values(["N", "_mr"]).drop(columns="_mr").reset_index(drop=True)
    return combined, agg


def style_axes(ax, *, log_y=False):
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, color="0.85")
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, color="0.92")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.7)
    if log_y:
        ax.set_yscale("log")


def panel_a_objective(ax, agg):
    Ns_ordered = sorted(agg["N"].unique())
    x = np.arange(len(Ns_ordered))
    n_methods = len(METHOD_ORDER)
    width = 0.78 / n_methods
    for i, m in enumerate(METHOD_ORDER):
        sub = agg[agg["method"] == m].set_index("N").reindex(Ns_ordered)
        offsets = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offsets, sub["objective_mean"].values, width,
               color=METHOD_COLOR[m], edgecolor="white", linewidth=0.4,
               label=m, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N}" if N != 500 else "500 (U-only)" for N in Ns_ordered])
    ax.set_xlabel("Vessel count N")
    ax.set_ylabel("Mean objective ($)")
    ax.set_title("(a) Mean objective by method and scale", loc="left", fontsize=10)
    style_axes(ax, log_y=True)


def panel_b_relgap(ax, agg):
    Ns_ordered = sorted(agg["N"].unique())
    x = np.arange(len(Ns_ordered))
    methods_no_cg = [m for m in METHOD_ORDER if m != "CG+IR"]
    width = 0.78 / len(methods_no_cg)
    for i, m in enumerate(methods_no_cg):
        sub = agg[agg["method"] == m].set_index("N").reindex(Ns_ordered)
        offsets = (i - (len(methods_no_cg) - 1) / 2) * width
        ax.bar(x + offsets, sub["rel_gap_to_CG_mean"].values, width,
               color=METHOD_COLOR[m], edgecolor="white", linewidth=0.4,
               label=m, zorder=3)
    ax.axhline(0.0, color="#444", linewidth=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N}" if N != 500 else "500 (U-only)" for N in Ns_ordered])
    ax.set_xlabel("Vessel count N")
    ax.set_ylabel("Relative gap to CG+IR (%)")
    ax.set_title("(b) Relative gap to CG+IR baseline", loc="left", fontsize=10)
    style_axes(ax)


def panel_c_runtime(ax, agg):
    Ns_ordered = sorted(agg["N"].unique())
    x = np.arange(len(Ns_ordered))
    for m in METHOD_ORDER:
        sub = agg[agg["method"] == m].set_index("N").reindex(Ns_ordered)
        ax.plot(x, sub["runtime_mean"].values, marker=METHOD_MARKER[m],
                color=METHOD_COLOR[m], linewidth=1.4, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6,
                label=m, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N}" if N != 500 else "500 (U-only)" for N in Ns_ordered])
    ax.set_xlabel("Vessel count N")
    ax.set_ylabel("Mean runtime (s)")
    ax.set_title("(c) Mean runtime by method and scale", loc="left", fontsize=10)
    style_axes(ax, log_y=True)


def panel_d_gap_summary(ax, agg):
    """Panel (d): gap-type summary — CG+IR margin over best non-CG baseline,
    annotated with the gap type used at each scale."""
    Ns_ordered = sorted(agg["N"].unique())
    base = agg[agg["method"] != "CG+IR"]
    best = (base.sort_values(["N", "rel_gap_to_CG_mean"])
                .groupby("N").first().reindex(Ns_ordered))

    x = np.arange(len(Ns_ordered))
    bar_colors = ["#4F8FC0" if N <= 100 else "#ED7D31" for N in Ns_ordered]
    bars = ax.bar(x, best["rel_gap_to_CG_mean"].values, 0.55,
                  color=bar_colors, edgecolor="0.25", linewidth=0.6,
                  zorder=3)
    for i, (rect, N) in enumerate(zip(bars, Ns_ordered)):
        gtype = "Full-CG LP-IP" if N <= 100 else "Pool LP-IP"
        margin = best["rel_gap_to_CG_mean"].iloc[i]
        winner = best["method"].iloc[i]
        label = f"{margin:.2f}%\n{winner}\n[{gtype}]"
        ax.annotate(label,
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.4, color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N}" if N != 500 else "500 (U-only)" for N in Ns_ordered])
    ax.set_xlabel("Vessel count N")
    ax.set_ylabel("CG+IR margin over best baseline (%)")
    ax.set_title("(d) Gap-type summary and CG+IR margin", loc="left", fontsize=10)
    style_axes(ax)
    ax.set_ylim(0, max(best["rel_gap_to_CG_mean"].max() * 1.55, 5))

    legend_elems = [
        Patch(facecolor="#4F8FC0", edgecolor="0.25",
              label="Full-CG LP-IP gap (full pricing converged)"),
        Patch(facecolor="#ED7D31", edgecolor="0.25",
              label="Pool LP-IP gap (generated-pool, NOT complete-column)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=7.6,
              frameon=True, framealpha=0.9, edgecolor="0.7")


def make_figure(agg):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    # No suptitle, no bottom note: caption is provided by LaTeX \caption{}.
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6), constrained_layout=False)
    panel_a_objective(axes[0, 0], agg)
    panel_b_relgap(axes[0, 1], agg)
    panel_c_runtime(axes[1, 0], agg)
    panel_d_gap_summary(axes[1, 1], agg)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
               linewidth=1.4, markersize=6, markeredgecolor="white",
               markeredgewidth=0.6, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=method_handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.985), ncol=len(METHOD_ORDER),
               frameon=False, fontsize=9.5, handletextpad=0.5,
               columnspacing=1.8)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.93, bottom=0.07,
                        wspace=0.24, hspace=0.36)

    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    png = os.path.join(OUT_FIG_DIR, "fig3_benchmark_final.png")
    pdf = os.path.join(OUT_FIG_DIR, "fig3_benchmark_final.pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


CAPTION_TEXT = (
    "Figure 3. Benchmark performance across instance scales using corrected "
    "and final-controlled outputs. Panels: (a) mean objective by method and "
    "scale on a log scale; (b) mean relative gap to the CG+IR baseline; (c) "
    "mean runtime by method on a log scale; (d) gap-type summary, showing "
    "the CG+IR optimality margin over the best non-CG baseline at each "
    "scale. N = 20, 50, 100 use corrected small-scale benchmark rows in "
    "which CG+IR fully converged, and the reported quantity is the Full-CG "
    "LP-IP gap. N = 200 (scenarios U/P/L) and N = 500 (scenario U only) use "
    "the final controlled budgeted-stabilized replacement runs, and the "
    "reported quantity is the Pool LP-IP gap, i.e. an LP-to-IP gap measured "
    "over the generated column pool. The Pool LP-IP gap is NOT a "
    "complete-column global optimality certificate and the N = 200 / N = 500 "
    "results should not be interpreted as full-pricing global optima. Old "
    "weak quick-CG runs are excluded by construction; method names are "
    "standardized; runtimes are end-to-end seconds.\n"
)


SOURCES_MD = """\
# Figure 3 Data Sources

## CSV inputs
- `results/revised/table8_revised.csv` - corrected small-scale benchmark
  summary used for N = 20, 50, 100. CG+IR rows have
  `cg_status = full_pricing_converged` and `gap_type = Full-CG LP-IP gap`.
- `results/revised/final_check/table8_final_controlled.csv` - final
  controlled replacement summary used for N = 200 (scenarios U/P/L) and
  N = 500 (scenario U only). CG+IR rows have
  `gap_type = Pool LP-IP gap` (budgeted-stabilized pricing).

## Diagnostic / validation inputs
- `results/revised/final_check/table8_validation_report.md`
- `results/revised/final_check/final_check_diagnostic_report.md`

## Inclusion rules
- Old weak quick-CG outputs (e.g. `cg_status = budgeted_topK` CG+IR rows
  for N = 200 / N = 500 in `table8_revised.csv`) are NOT read.
- N = 20 / 50 / 100 rows are taken only from the corrected small-scale
  summary and are validated to be `full_pricing_converged`.
- N = 200 and N = 500 rows are taken only from the final controlled
  replacement summary and are validated to be `Pool LP-IP gap`.
- N = 500 includes scenario U only; P/L were not run in the controlled
  replacement due to the runtime budget.
- CG+IR gap labels are preserved in the figure: Full-CG LP-IP gap for
  N <= 100, Pool LP-IP gap for N = 200 and N = 500.

## Aggregation
- Per row, the relative gap to CG+IR is recomputed from `objective_mean`
  against the CG+IR objective at the same (N, scenario) - we do not rely
  on stale `gap_pct` columns.
- For N = 20, 50, 100 and N = 200, U/P/L are averaged equally per method
  to produce a per-(N, method) summary.
- For N = 500, only scenario U is available and is shown without averaging.

## Methods included
- CG+IR (the column-generation upper bound)
- Restricted-CG
- Rolling-Horizon
- Fix-and-Optimize
- FIFO
- Greedy

## Outputs
- `figures/revised/manuscript/fig3_benchmark_final.png`
- `figures/revised/manuscript/fig3_benchmark_final.pdf`
- `results/revised/manuscript/fig3_benchmark_final_plot_data.csv`
- `results/revised/manuscript/fig3_benchmark_per_scenario.csv`
- `results/revised/manuscript/fig3_benchmark_final_caption.txt`
- `results/revised/manuscript/fig3_data_sources.md`
"""


def main():
    small, large = load_and_filter()
    combined, agg = build_combined_and_aggregate(small, large)

    os.makedirs(OUT_DAT_DIR, exist_ok=True)
    combined.to_csv(os.path.join(OUT_DAT_DIR, "fig3_benchmark_per_scenario.csv"),
                    index=False)
    agg.to_csv(os.path.join(OUT_DAT_DIR, "fig3_benchmark_final_plot_data.csv"),
               index=False)

    png, pdf = make_figure(agg)

    with open(os.path.join(OUT_DAT_DIR, "fig3_benchmark_final_caption.txt"),
              "w", encoding="utf-8") as f:
        f.write(CAPTION_TEXT)
    with open(os.path.join(OUT_DAT_DIR, "fig3_data_sources.md"),
              "w", encoding="utf-8") as f:
        f.write(SOURCES_MD)

    print(f"OK\n  fig: {png}\n       {pdf}")
    print(f"  data: {OUT_DAT_DIR}")


if __name__ == "__main__":
    main()

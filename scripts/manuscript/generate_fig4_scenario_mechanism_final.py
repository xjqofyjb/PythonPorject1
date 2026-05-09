"""
Regenerate manuscript Figure 4 (scenario + mechanism comparison) from
corrected CSV outputs only. Same visual family as Figure 3:
    - LaTeX-ready: no suptitle, no bottom note (caption goes in \\caption{}).
    - Palette matches the earlier manuscript figures.
    - Serif fonts, thin grid, clean academic style.

Inputs (read-only):
  - results/revised/scenario_comparison_raw.csv     (N=100, U/P/L, all methods)
  - results/revised/mechanism_comparison_raw.csv    (N=100, 4 mechanisms)
  - results/revised/mechanism_comparison_summary.csv (already-aggregated mechanisms)

Outputs:
  - figures/revised/manuscript/fig4_scenario_mechanism_final.png
  - figures/revised/manuscript/fig4_scenario_mechanism_final.pdf
  - results/revised/manuscript/fig4_scenario_panel_data.csv
  - results/revised/manuscript/fig4_mechanism_panel_data.csv
  - results/revised/manuscript/fig4_caption.txt
  - results/revised/manuscript/fig4_data_sources.md

Acceptance:
  - Corrected data only (CG+IR rows must be full_pricing_converged).
  - Method names standardized; legend and axis labels are clear.
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

SRC_SCEN = os.path.join(ROOT, "results", "revised", "scenario_comparison_raw.csv")
SRC_MECH = os.path.join(ROOT, "results", "revised", "mechanism_comparison_raw.csv")
SRC_MECH_SUMMARY = os.path.join(
    ROOT, "results", "revised", "mechanism_comparison_summary.csv")

OUT_FIG_DIR = os.path.join(ROOT, "figures", "revised", "manuscript")
OUT_DAT_DIR = os.path.join(ROOT, "results", "revised", "manuscript")

METHOD_ORDER_FULL = ["CG+IR", "Rolling-Horizon", "Fix-and-Optimize",
                     "FIFO", "Greedy"]
# For panel (b) we keep only methods that exercise the mechanism switch
METHOD_ORDER_MECH = ["CG+IR", "Rolling-Horizon", "Fix-and-Optimize"]

# Palette matched to Figure 3
METHOD_COLOR = {
    "CG+IR":            "#4F8FC0",
    "Restricted-CG":    "#9DC3E6",
    "Rolling-Horizon":  "#70AD47",
    "Fix-and-Optimize": "#A9D18E",
    "FIFO":             "#7F7F7F",
    "Greedy":           "#ED7D31",
}

RENAME_METHOD = {
    "fix_and_optimize": "Fix-and-Optimize",
    "rolling_horizon":  "Rolling-Horizon",
}

SCENARIO_LABEL = {
    "U": "Uniform (U)",
    "P": "Peaked (P)",
    "L": "Long service (L)",
}

MECHANISM_LABEL = {
    "Hybrid":           "Hybrid",
    "BS_only":          "Battery-only",
    "SP_only":          "Shore-power only",
    "Green_only_no_AE": "Green-only (no AE)",
}
MECHANISM_ORDER = ["Hybrid", "BS_only", "SP_only", "Green_only_no_AE"]


def stop_with_missing(missing):
    os.makedirs(OUT_DAT_DIR, exist_ok=True)
    out = os.path.join(OUT_DAT_DIR, "fig4_missing_data_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Figure 4 missing-data report\n\n"
                "Figure 4 was NOT regenerated because required source files are missing.\n\n"
                "## Missing files\n" + "\n".join(f"- `{p}`" for p in missing) + "\n")
    print(f"[STOP] wrote missing-data report -> {out}", file=sys.stderr)
    sys.exit(1)


def load_scenario():
    df = pd.read_csv(SRC_SCEN)
    df = df[df["N"] == 100].copy()
    df["method"] = df["method"].replace(RENAME_METHOD)
    df = df[df["status"] == "ok"]
    cg = df[df["method"] == "CG+IR"]
    if not (cg["cg_status"] == "full_pricing_converged").all():
        raise RuntimeError("Scenario-comparison CG+IR must be full_pricing_converged.")
    if not (cg["gap_type"] == "Full-CG LP-IP gap").all():
        raise RuntimeError("Scenario-comparison CG+IR must use Full-CG LP-IP gap.")
    return df


def aggregate_scenario(df):
    """Per (scenario, method): mean and std of objective across seeds."""
    rows = []
    for (sc, m), g in df.groupby(["scenario", "method"]):
        rows.append({
            "scenario": sc,
            "method": m,
            "obj_mean": g["obj"].mean(),
            "obj_std":  g["obj"].std(ddof=0),
            "n_seeds":  int(g.shape[0]),
        })
    out = pd.DataFrame(rows)
    # Per-(scenario) CG+IR baseline for relative gap
    base = out[out["method"] == "CG+IR"].set_index("scenario")["obj_mean"]
    out["rel_gap_to_CG_pct"] = out.apply(
        lambda r: 100.0 * (r["obj_mean"] - base.loc[r["scenario"]])
                  / base.loc[r["scenario"]], axis=1)
    return out


def load_mechanism():
    df = pd.read_csv(SRC_MECH)
    df = df[df["N"] == 100].copy()
    df["method"] = df["method"].replace(RENAME_METHOD)
    cg = df[df["method"] == "CG+IR"]
    if not (cg["cg_status"] == "full_pricing_converged").all():
        raise RuntimeError("Mechanism-comparison CG+IR must be full_pricing_converged.")
    return df


def aggregate_mechanism(df):
    """Compute per-(mechanism, method) cost penalty vs Hybrid (%) averaged
    across scenarios. Each row also carries an infeasibility rate so we can
    grey-out infeasible bars in the figure."""
    df = df.copy()
    df["infeasible"] = (df["status"] != "ok") | df["obj"].isna()

    # Per (scenario, method, mechanism): mean obj across seeds + infeasible rate
    mean_rows = []
    for (sc, m, mech), g in df.groupby(["scenario", "method", "mechanism"]):
        feas = g[~g["infeasible"]]
        mean_rows.append({
            "scenario": sc, "method": m, "mechanism": mech,
            "obj_mean": feas["obj"].mean() if len(feas) else np.nan,
            "infeasibility_rate": g["infeasible"].mean(),
            "n_seeds": int(g.shape[0]),
        })
    per_scen = pd.DataFrame(mean_rows)

    # Per-(scenario, method) Hybrid baseline -> per-row penalty (%)
    base = (per_scen[per_scen["mechanism"] == "Hybrid"]
            .set_index(["scenario", "method"])["obj_mean"])
    def _penalty(r):
        b = base.get((r["scenario"], r["method"]), np.nan)
        if np.isnan(r["obj_mean"]) or np.isnan(b):
            return np.nan
        return 100.0 * (r["obj_mean"] - b) / b
    per_scen["penalty_vs_hybrid_pct"] = per_scen.apply(_penalty, axis=1)

    # Average across scenarios per (method, mechanism)
    rows = []
    for (m, mech), g in per_scen.groupby(["method", "mechanism"]):
        rows.append({
            "method": m, "mechanism": mech,
            "penalty_vs_hybrid_pct_mean": g["penalty_vs_hybrid_pct"].mean(),
            "penalty_vs_hybrid_pct_std":  g["penalty_vs_hybrid_pct"].std(ddof=0),
            "infeasibility_rate_mean": g["infeasibility_rate"].mean(),
            "scenarios": ",".join(sorted(g["scenario"].unique())),
        })
    out = pd.DataFrame(rows)
    return per_scen, out


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


def panel_a_scenario(ax, agg_scen):
    scen_order = ["U", "P", "L"]
    x = np.arange(len(scen_order))
    n = len(METHOD_ORDER_FULL)
    width = 0.78 / n
    for i, m in enumerate(METHOD_ORDER_FULL):
        sub = agg_scen[agg_scen["method"] == m].set_index("scenario").reindex(scen_order)
        offsets = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offsets, sub["obj_mean"].values, width,
                      yerr=sub["obj_std"].values, capsize=2.0,
                      color=METHOD_COLOR[m], edgecolor="white", linewidth=0.4,
                      error_kw={"elinewidth": 0.7, "ecolor": "0.35"},
                      label=m, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABEL[s] for s in scen_order])
    ax.set_xlabel("Arrival scenario")
    ax.set_ylabel("Total cost ($)")
    ax.set_title("(a) Scenario comparison at $N = 100$", loc="left", fontsize=10)
    style_axes(ax)
    # Format y-axis with thousands separator
    ax.ticklabel_format(axis="y", style="plain")
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def panel_b_mechanism(ax, agg_mech):
    mech_order = MECHANISM_ORDER
    x = np.arange(len(mech_order))
    n = len(METHOD_ORDER_MECH)
    width = 0.78 / n

    for i, m in enumerate(METHOD_ORDER_MECH):
        sub = agg_mech[agg_mech["method"] == m].set_index("mechanism").reindex(mech_order)
        offsets = (i - (n - 1) / 2) * width
        vals = sub["penalty_vs_hybrid_pct_mean"].values.copy()
        infeas = sub["infeasibility_rate_mean"].values
        # Replace fully infeasible with 0 height; we'll annotate "infeasible"
        plot_vals = np.where(np.isnan(vals), 0.0, vals)
        bars = ax.bar(x + offsets, plot_vals, width,
                      color=METHOD_COLOR[m], edgecolor="white", linewidth=0.4,
                      label=m, zorder=3)
        # Mark infeasible bars
        for rect, v, inf in zip(bars, vals, infeas):
            if np.isnan(v) and inf > 0:
                rect.set_facecolor("#E8E8E8")
                rect.set_edgecolor("0.55")
                rect.set_hatch("///")
                ax.text(rect.get_x() + rect.get_width() / 2, 1.0,
                        "infeasible", rotation=90, ha="center", va="bottom",
                        fontsize=6.5, color="0.35")

    ax.axhline(0.0, color="#444", linewidth=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MECHANISM_LABEL[m] for m in mech_order])
    ax.set_xlabel("Service mechanism")
    ax.set_ylabel("Cost penalty vs Hybrid (%)")
    ax.set_title("(b) Mechanism comparison at $N = 100$ (avg. over U / P / L)",
                 loc="left", fontsize=10)
    style_axes(ax)


def make_figure(agg_scen, agg_mech):
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif":  ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), constrained_layout=False)
    panel_a_scenario(axes[0], agg_scen)
    panel_b_mechanism(axes[1], agg_mech)

    # Combined legend at top: methods + a hatch-patch to explain infeasible bars
    method_handles = [
        Patch(facecolor=METHOD_COLOR[m], edgecolor="white", label=m)
        for m in METHOD_ORDER_FULL
    ]
    infeas_patch = Patch(facecolor="#E8E8E8", edgecolor="0.55", hatch="///",
                         label="Infeasible (pure-mode constraint)")
    fig.legend(handles=method_handles + [infeas_patch],
               loc="upper center", bbox_to_anchor=(0.5, 1.005),
               ncol=len(METHOD_ORDER_FULL) + 1, frameon=False,
               fontsize=8.8, handletextpad=0.5, columnspacing=1.5)

    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.13,
                        wspace=0.24)

    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    png = os.path.join(OUT_FIG_DIR, "fig4_scenario_mechanism_final.png")
    pdf = os.path.join(OUT_FIG_DIR, "fig4_scenario_mechanism_final.pdf")
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png, pdf


CAPTION_TEXT = (
    "Figure 4. Scenario and mechanism comparison at $N = 100$ using corrected "
    "data only. (a) Total cost across the three arrival scenarios -- Uniform "
    "(U), Peaked (P), and Long service (L) -- for CG+IR (full-pricing "
    "converged) and the scalable baselines Rolling-Horizon, Fix-and-Optimize, "
    "FIFO, and Greedy; error bars show the per-seed standard deviation. (b) "
    "Service-mechanism comparison: cost penalty relative to the Hybrid "
    "reference (CG+IR / Rolling-Horizon / Fix-and-Optimize), averaged "
    "equally across the U, P, and L scenarios. Pure-mode mechanisms that "
    "violate the green-only constraint without auxiliary engines are shown "
    "as `infeasible'. Method names are standardized to the manuscript "
    "convention; CG+IR rows are validated as full-pricing converged with "
    "Full-CG LP-IP gap, so no old quick-CG outputs are mixed in.\n"
)


SOURCES_MD = """\
# Figure 4 Data Sources

## CSV inputs
- `results/revised/scenario_comparison_raw.csv` -- corrected scenario
  benchmark at N = 100. CG+IR rows are validated as
  `cg_status = full_pricing_converged` and `gap_type = Full-CG LP-IP gap`.
- `results/revised/mechanism_comparison_raw.csv` -- corrected mechanism
  comparison at N = 100. CG+IR rows are validated as
  `cg_status = full_pricing_converged`. Cross-checked against
  `results/revised/mechanism_comparison_summary.csv`.

## Inclusion rules
- Old weak quick-CG outputs are NOT read; both source CSVs only contain
  full-pricing CG+IR rows after explicit validation.
- Method names are standardized: `fix_and_optimize` -> `Fix-and-Optimize`,
  `rolling_horizon` -> `Rolling-Horizon`.
- Panel (a) shows U / P / L scenarios for the five comparison methods
  (CG+IR, Rolling-Horizon, Fix-and-Optimize, FIFO, Greedy).
- Panel (b) shows the four service mechanisms (Hybrid, Battery-only,
  Shore-power only, Green-only no-AE) for the three methods that exercise
  the mechanism switch (CG+IR, Rolling-Horizon, Fix-and-Optimize).
- For each (mechanism, method) bar in panel (b), the cost penalty vs the
  Hybrid reference is computed per scenario and then averaged across U/P/L.
- Bars where every seed was infeasible (pure-mode green-only constraint)
  are drawn as a hatched grey patch and labelled `infeasible'.

## Aggregation
- Panel (a): per (scenario, method) mean and std of objective across seeds.
- Panel (b): per (scenario, method, mechanism) seed-mean of objective,
  divided by the same-(scenario, method) Hybrid mean to get a per-scenario
  cost-penalty percentage; then averaged across scenarios.

## Outputs
- `figures/revised/manuscript/fig4_scenario_mechanism_final.png`
- `figures/revised/manuscript/fig4_scenario_mechanism_final.pdf`
- `results/revised/manuscript/fig4_scenario_panel_data.csv`
- `results/revised/manuscript/fig4_mechanism_panel_data.csv`
- `results/revised/manuscript/fig4_caption.txt`
- `results/revised/manuscript/fig4_data_sources.md`
"""


def main():
    missing = [p for p in (SRC_SCEN, SRC_MECH) if not os.path.exists(p)]
    if missing:
        stop_with_missing(missing)

    scen = load_scenario()
    mech = load_mechanism()

    agg_scen = aggregate_scenario(scen)
    per_scen_mech, agg_mech = aggregate_mechanism(mech)

    os.makedirs(OUT_DAT_DIR, exist_ok=True)
    agg_scen.to_csv(os.path.join(OUT_DAT_DIR, "fig4_scenario_panel_data.csv"),
                    index=False)
    agg_mech.to_csv(os.path.join(OUT_DAT_DIR, "fig4_mechanism_panel_data.csv"),
                    index=False)
    per_scen_mech.to_csv(
        os.path.join(OUT_DAT_DIR, "fig4_mechanism_per_scenario.csv"),
        index=False)

    png, pdf = make_figure(agg_scen, agg_mech)

    with open(os.path.join(OUT_DAT_DIR, "fig4_caption.txt"),
              "w", encoding="utf-8") as f:
        f.write(CAPTION_TEXT)
    with open(os.path.join(OUT_DAT_DIR, "fig4_data_sources.md"),
              "w", encoding="utf-8") as f:
        f.write(SOURCES_MD)

    print(f"OK\n  fig: {png}\n       {pdf}")
    print(f"  data: {OUT_DAT_DIR}")


if __name__ == "__main__":
    main()

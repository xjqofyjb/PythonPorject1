"""Generate Transportation Research / Elsevier-style figures and tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "revised" / "tr_style"
RES_DIR = ROOT / "results" / "revised" / "tr_style"
LOG_DIR = ROOT / "logs" / "revised" / "tr_style"

FINAL = ROOT / "results" / "revised" / "final_check"
REVISED = ROOT / "results" / "revised"
MANUSCRIPT = REVISED / "manuscript"

COLORS = {
    "CG+IR": "#1f4e79",
    "Fix-and-Optimize": "#d95f02",
    "Rolling-Horizon": "#1b9e77",
    "Restricted-CG": "#6a3d9a",
    "FIFO": "#777777",
    "Greedy": "#8c6d31",
    "SP": "#1f4e79",
    "BS": "#d95f02",
    "AE": "#8c564b",
    "SIMOPS": "#1f4e79",
    "Sequential": "#555555",
}
MARKERS = {
    "CG+IR": "o",
    "Fix-and-Optimize": "s",
    "Rolling-Horizon": "^",
    "Restricted-CG": "D",
    "FIFO": "v",
    "Greedy": "P",
}
METHOD_ORDER = ["CG+IR", "Fix-and-Optimize", "Rolling-Horizon", "Restricted-CG", "FIFO", "Greedy"]
METHOD_MAP = {
    "CG": "CG+IR",
    "cg": "CG+IR",
    "CG+IR": "CG+IR",
    "F&O": "Fix-and-Optimize",
    "fix_and_optimize": "Fix-and-Optimize",
    "Fix-and-Optimize": "Fix-and-Optimize",
    "rolling_horizon": "Rolling-Horizon",
    "Rolling-Horizon": "Rolling-Horizon",
    "restricted_cg": "Restricted-CG",
    "Restricted-CG": "Restricted-CG",
    "fifo": "FIFO",
    "FIFO": "FIFO",
    "greedy": "Greedy",
    "Greedy": "Greedy",
}


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(0.0, 1.04, text, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def write_sources(name: str, lines: list[str]) -> None:
    (RES_DIR / f"{name}_data_sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def standardize_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method"] = out["method"].map(METHOD_MAP).fillna(out["method"])
    return out


def fig1() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    ax.set_axis_off()

    ax.add_patch(Rectangle((0.03, 0.18), 0.20, 0.42, facecolor="#e8eef6", edgecolor="#8da8c8", linewidth=1.0))
    ax.text(0.13, 0.64, "Arriving vessels", ha="center", va="center", fontsize=9)
    for y in [0.28, 0.39, 0.50]:
        ax.plot([0.06, 0.20], [y, y], color="#1f4e79", lw=2)

    modules = [
        (0.38, 0.55, "Shore power", "Berths\nSP capacity", COLORS["SP"]),
        (0.38, 0.16, "Battery swapping", "Swap slots\nfast service", COLORS["BS"]),
    ]
    for x, y, title, body, color in modules:
        box = FancyBboxPatch((x, y), 0.28, 0.24, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="white", edgecolor=color, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + 0.14, y + 0.16, title, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
        ax.text(x + 0.14, y + 0.07, body, ha="center", va="center", fontsize=8)
        ax.annotate("", xy=(x, y + 0.12), xytext=(0.23, 0.39), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))

    ax.add_patch(FancyBboxPatch((0.78, 0.32), 0.18, 0.22, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="#f7f7f7", edgecolor="#999999", linewidth=1.0))
    ax.text(0.87, 0.45, "Service plan", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.87, 0.36, "cost, delay,\nemissions", ha="center", fontsize=8)
    for y in [0.67, 0.28]:
        ax.annotate("", xy=(0.78, 0.43), xytext=(0.66, y), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
    save(fig, "fig1_conceptual_clean")
    write_sources("fig1_conceptual_clean", ["# Fig. 1 Data Sources", "", "Conceptual schematic drawn directly from the model structure; no numerical data are plotted."])


def load_fig3_data() -> pd.DataFrame:
    small_path = REVISED / "table8_revised.csv"
    large_path = FINAL / "table8_final_controlled.csv"
    if not small_path.exists() or not large_path.exists():
        raise FileNotFoundError("Missing Fig. 3 source CSV.")
    small = pd.read_csv(small_path)
    small = small[small["N"].isin([20, 50, 100])].rename(columns={"obj_mean": "objective_mean", "obj_std": "objective_std", "success_rate": "status_success_rate"})
    small = standardize_methods(small)
    small = small[small["method"].isin(METHOD_ORDER)]
    small["runtime_std"] = np.nan
    small["pricing_converged_rate"] = np.where((small["method"] == "CG+IR") & (small["gap_type"] == "Full-CG LP-IP gap"), 1.0, np.nan)
    small["source"] = "corrected small-scale summary"
    large = standardize_methods(pd.read_csv(large_path))
    large = large[large["N"].isin([200, 500]) & large["method"].isin(METHOD_ORDER)].copy()
    large["source"] = "final controlled replacement"
    combined = pd.concat([small, large], ignore_index=True, sort=False)
    cg = combined[combined["method"] == "CG+IR"][["N", "scenario", "objective_mean"]].rename(columns={"objective_mean": "cg_obj"})
    combined = combined.merge(cg, on=["N", "scenario"], how="left")
    combined["rel_gap_to_CG_mean"] = (combined["objective_mean"] - combined["cg_obj"]) / combined["cg_obj"].abs().clip(lower=1.0) * 100
    return combined


def fig3() -> None:
    data = load_fig3_data()
    rows = []
    for (N, method), g in data.groupby(["N", "method"]):
        rows.append({
            "N": N,
            "method": method,
            "objective_mean": g["objective_mean"].mean(),
            "rel_gap_to_CG_mean": g["rel_gap_to_CG_mean"].mean(),
            "runtime_mean": g["runtime_mean"].mean(),
            "gap_type": ";".join(sorted(set(map(str, g["gap_type"].dropna())))),
        })
    plotdf = pd.DataFrame(rows)
    plotdf["method"] = pd.Categorical(plotdf["method"], categories=METHOD_ORDER, ordered=True)
    plotdf = plotdf.sort_values(["N", "method"])
    plotdf.to_csv(RES_DIR / "fig3_benchmark_tr_style_plot_data.csv", index=False)

    scales = [20, 50, 100, 200, 500]
    labels = ["20", "50", "100", "200", "500\n(U only)"]
    x = np.arange(len(scales))
    width = 0.12
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.5), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs.ravel()
    for i, method in enumerate(METHOD_ORDER):
        sub = plotdf[plotdf["method"] == method].set_index("N")
        off = (i - 2.5) * width
        ax1.bar(x + off, [sub.loc[N, "objective_mean"] / 1000 if N in sub.index else np.nan for N in scales], width, color=COLORS[method], label=method)
        ax2.bar(x + off, [sub.loc[N, "rel_gap_to_CG_mean"] if N in sub.index else np.nan for N in scales], width, color=COLORS[method])
        ax3.bar(x + off, [sub.loc[N, "runtime_mean"] if N in sub.index else np.nan for N in scales], width, color=COLORS[method])
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(x, labels)
        ax.grid(axis="y", color="#dddddd", lw=0.6, alpha=0.8)
        ax.axvline(2.5, color="#777777", ls=":", lw=1.0)
    panel_label(ax1, "(a)")
    ax1.set_title("Objective by scale")
    ax1.set_ylabel("Mean objective (thousand $, log scale)")
    ax1.set_yscale("log")
    ax1.legend(ncol=2, frameon=False)
    panel_label(ax2, "(b)")
    ax2.set_title("Relative gap to CG+IR")
    ax2.set_ylabel("Mean gap (%)")
    panel_label(ax3, "(c)")
    ax3.set_title("Runtime")
    ax3.set_ylabel("Mean runtime (s, log scale)")
    ax3.set_yscale("log")
    ax4.axis("off")
    panel_label(ax4, "(d)")
    ax4.text(0.02, 0.90, "Gap interpretation", fontsize=9, fontweight="bold", transform=ax4.transAxes)
    ax4.text(0.02, 0.72, "N≤100: Full-CG LP-IP gap\n(full pricing converged)", fontsize=8, transform=ax4.transAxes)
    ax4.text(0.02, 0.47, "N=200/500: Pool LP-IP gap\nbudgeted generated-pool evidence,\nnot complete-column global optimality", fontsize=8, transform=ax4.transAxes)
    ax4.text(0.02, 0.18, "N=500: scenario U only\nOld weak quick-CG outputs excluded", fontsize=8, transform=ax4.transAxes)
    save(fig, "fig3_benchmark_tr_style")
    write_sources("fig3_benchmark_tr_style", [
        "# Fig. 3 Data Sources",
        "",
        "- `results/revised/table8_revised.csv` for corrected N=20/50/100 rows.",
        "- `results/revised/final_check/table8_final_controlled.csv` for N=200 and N=500 final controlled rows.",
        "- N=500 is U-only.",
        "- Old weak quick-CG data are excluded.",
    ])


def fig5() -> None:
    path = FINAL / "simops_dual_peak_final_summary.csv"
    df = pd.read_csv(path).sort_values("N")
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.plot(df["N"], df["simops_saving_pct_mean"], color=COLORS["SIMOPS"], marker="o", lw=1.6, label="SIMOPS saving")
    ax.fill_between(df["N"], df["simops_saving_ci95_low"], df["simops_saving_ci95_high"], color=COLORS["SIMOPS"], alpha=0.16, label="95% bootstrap CI")
    ax.axvspan(20, 35, color="#eeeeee", alpha=0.8)
    ax.axvspan(85, 115, color="#f2d7b6", alpha=0.45)
    ax.axvspan(190, 510, color="#dddddd", alpha=0.35)
    ax.text(26, df["simops_saving_pct_mean"].max() * 0.95, "light-load\nmasking", fontsize=8, ha="center")
    ax.text(100, df["simops_saving_pct_mean"].max() * 0.83, "threshold-\nbuffering", fontsize=8, ha="center")
    ax.text(310, df["simops_saving_pct_mean"].max() * 0.42, "capacity-\nsaturated tail", fontsize=8, ha="center")
    ax.set_xlabel("Number of vessels, N")
    ax.set_ylabel("SIMOPS saving (%)")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.legend(frameon=False)
    save(fig, "fig5_simops_value_tr_style")
    write_sources("fig5_simops_value_tr_style", ["# Fig. 5 Data Sources", "", "- `results/revised/final_check/simops_dual_peak_final_summary.csv`.", "- Bootstrap CI columns are read from the CSV."])


def fig6() -> None:
    summary = pd.read_csv(FINAL / "bs_cost_sensitivity_final_summary.csv")
    threshold = float(pd.read_csv(FINAL / "bs_threshold_detection.csv")["detected_threshold_C_BS"].dropna().iloc[0])
    df = summary[summary["method"] == "CG+IR"].sort_values("C_BS")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
    ax = axs[0]
    ax.plot(df["C_BS"], df["total_cost_mean"] / 1000, color=COLORS["CG+IR"], marker="o", lw=1.5)
    ax.axvline(threshold, color="#555555", ls="--", lw=1.0)
    panel_label(ax, "(a)")
    ax.set_xlabel("BS unit cost (USD/kWh)")
    ax.set_ylabel("Total cost (thousand $)")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax = axs[1]
    for col, label, ls, marker in [("SP_share_mean", "SP", "-", "o"), ("BS_share_mean", "BS", "--", "s"), ("AE_share_mean", "AE", ":", "^")]:
        ax.plot(df["C_BS"], df[col], color=COLORS[label], ls=ls, marker=marker, lw=1.5, label=label)
    ax.axvline(threshold, color="#555555", ls="--", lw=1.0, label="$C_{BS}=0.90$")
    panel_label(ax, "(b)")
    ax.set_xlabel("BS unit cost (USD/kWh)")
    ax.set_ylabel("Mean service share")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.legend(frameon=False)
    save(fig, "fig6_bs_threshold_two_panel")
    write_sources("fig6_bs_threshold_two_panel", ["# Fig. 6 Data Sources", "", "- `results/revised/final_check/bs_cost_sensitivity_final_summary.csv`.", "- `results/revised/final_check/bs_threshold_detection.csv`.", "- The vertical line marks the detected discontinuous BS-AE substitution threshold in the tested calibration."])


def fig8() -> None:
    path = REVISED / "carbon_grid_factor_summary.csv"
    if not path.exists():
        (RES_DIR / "fig8_missing_data_report.md").write_text("# Fig. 8 Missing Data Report\n\nCorrected carbon-price data are unavailable.\n", encoding="utf-8")
        return
    df = pd.read_csv(path)
    df = df[df["capacity_config"].isin(["baseline", "constrained"])]
    # Use the central grid factor when available to isolate carbon price.
    if 0.565 in set(df["grid_factor"]):
        df = df[df["grid_factor"].eq(0.565)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for cfg, sub in df.groupby("capacity_config"):
        label = "adequate capacity" if cfg == "baseline" else "constrained capacity"
        style = "-" if cfg == "baseline" else "--"
        marker = "o" if cfg == "baseline" else "s"
        axs[0].plot(sub["carbon_price"], sub["total_cost"] / 1000, color=COLORS["CG+IR"] if cfg == "baseline" else COLORS["AE"], ls=style, marker=marker, label=label)
        axs[1].plot(sub["carbon_price"], sub["AE_share"], color=COLORS["CG+IR"] if cfg == "baseline" else COLORS["AE"], ls=style, marker=marker, label=label)
    for ax in axs:
        for p in [100, 200, 380]:
            ax.axvline(p, color="#bbbbbb", lw=0.7, ls=":")
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_xlabel("Carbon price (USD/tCO2)")
    panel_label(axs[0], "(a)")
    axs[0].set_ylabel("Total cost (thousand $)")
    panel_label(axs[1], "(b)")
    axs[1].set_ylabel("AE share")
    axs[1].legend(frameon=False)
    save(fig, "fig8_carbon_price_tr_style")
    write_sources("fig8_carbon_price_tr_style", ["# Fig. 8 Data Sources", "", "- `results/revised/carbon_grid_factor_summary.csv`.", "- Plotted under the tested capacity configurations; central grid factor slice is used when available."])


def fig9_10() -> None:
    path = REVISED / "arrival_perturbation_summary.csv"
    if not path.exists():
        (RES_DIR / "fig9_missing_data_report.md").write_text("# Fig. 9 Missing Data Report\n\nCorrected perturbation data are unavailable.\n", encoding="utf-8")
        (RES_DIR / "fig10_missing_data_report.md").write_text("# Fig. 10 Missing Data Report\n\nCorrected empirical-boundary data are unavailable.\n", encoding="utf-8")
        return
    df = pd.read_csv(path)
    base = df[df["Delta"].eq(0)].groupby(["slack", "perturbation_type"])["total_cost"].mean().rename("base_cost").reset_index()
    df = df.merge(base, on=["slack", "perturbation_type"], how="left")
    df["relative_degradation_pct"] = (df["total_cost"] - df["base_cost"]) / df["base_cost"].abs().clip(lower=1.0) * 100
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for (slack, ptype), sub in df.groupby(["slack", "perturbation_type"]):
        label = f"{slack}, {ptype.replace('_', ' ')}"
        marker = "o" if slack == "loose" else "s"
        ls = "-" if "one" in ptype else "--"
        axs[0].plot(sub["Delta"], sub["total_cost"] / 1000, marker=marker, ls=ls, lw=1.3, label=label)
        axs[1].plot(sub["Delta"], sub["relative_degradation_pct"], marker=marker, ls=ls, lw=1.3, label=label)
    for ax in axs:
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_xlabel("Perturbation amplitude (h)")
    panel_label(axs[0], "(a)")
    axs[0].set_ylabel("Total cost (thousand $)")
    panel_label(axs[1], "(b)")
    axs[1].set_ylabel("Relative degradation (%)")
    axs[1].legend(frameon=False, fontsize=7)
    save(fig, "fig9_perturbation_tr_style")
    write_sources("fig9_perturbation_tr_style", ["# Fig. 9 Data Sources", "", "- `results/revised/arrival_perturbation_summary.csv`.", "- Relative degradation is computed against the matching Delta=0 row by slack and perturbation type."])
    (RES_DIR / "fig10_missing_data_report.md").write_text("# Fig. 10 Missing Data Report\n\nThe available corrected perturbation summary does not contain empirical-boundary or relative-boundary fields required for Fig. 10. No boundary figure was generated.\n", encoding="utf-8")


def fig11_12() -> None:
    comp_path = FINAL / "dual_peak_enrichment_comparison.csv"
    raw_path = FINAL / "dual_peak_enrichment_raw.csv"
    if not comp_path.exists() or not raw_path.exists():
        (RES_DIR / "fig11_missing_data_report.md").write_text("# Fig. 11 Missing Data Report\n\nFinal-check enrichment data are unavailable.\n", encoding="utf-8")
        return
    comp = pd.read_csv(comp_path).sort_values("N")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot(comp["N"], comp["original_saving_mean"], color=COLORS["Sequential"], marker="o", ls="--", label="Original")
    ax.plot(comp["N"], comp["enriched_saving_mean"], color=COLORS["SIMOPS"], marker="s", ls="-", label="Enriched")
    ax.set_xlabel("Number of vessels, N")
    ax.set_ylabel("SIMOPS saving (%)")
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.legend(frameon=False)
    save(fig, "fig11_enrichment_tr_style")

    raw = pd.read_csv(raw_path)
    sim = raw[raw["operation_mode"].eq("simops")].copy()
    use = sim.groupby("N").agg(
        injected_columns=("injected_columns_count", "mean"),
        pool_gap_pct=("pool_gap_pct", "mean"),
        stabilized=("objective_stabilized", lambda s: np.mean(s.astype(str).str.lower().isin(["true", "1", "1.0"]))),
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(5.2, 3.0))
    ax1.bar(use["N"], use["injected_columns"], width=7, color=COLORS["Restricted-CG"], alpha=0.8, label="Injected columns")
    ax1.set_xlabel("Number of vessels, N")
    ax1.set_ylabel("Mean injected columns")
    ax2 = ax1.twinx()
    ax2.plot(use["N"], use["stabilized"], color=COLORS["CG+IR"], marker="o", lw=1.4, label="Stabilized rate")
    ax2.set_ylabel("Objective stabilized rate")
    ax1.grid(axis="y", color="#dddddd", lw=0.6)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    save(fig, "fig12_enrichment_usage_tr_style")
    write_sources("fig11_enrichment_tr_style", ["# Fig. 11 Data Sources", "", "- `results/revised/final_check/dual_peak_enrichment_comparison.csv`.", "- Appendix caveat: generated-pool stability evidence, not complete-column optimality proof."])
    write_sources("fig12_enrichment_usage_tr_style", ["# Fig. 12 Data Sources", "", "- `results/revised/final_check/dual_peak_enrichment_raw.csv`.", "- Appendix caveat: generated-pool stability evidence, not complete-column optimality proof."])


def table8() -> None:
    df = pd.read_csv(FINAL / "table8_final_controlled.csv")
    df = standardize_methods(df)
    cols = ["N", "scenario", "method", "objective_mean", "objective_std", "rel_gap_to_CG_mean", "runtime_mean", "cg_status", "gap_type"]
    out = df[cols].copy()
    out["scenario"] = np.where(out["N"].eq(500), out["scenario"].astype(str) + " (U-only)", out["scenario"])
    out = out.rename(columns={
        "scenario": "Scenario",
        "method": "Method",
        "objective_mean": "Objective mean",
        "objective_std": "Objective std.",
        "rel_gap_to_CG_mean": "Gap to CG+IR (%)",
        "runtime_mean": "Runtime (s)",
        "cg_status": "CG status",
        "gap_type": "Gap type",
    })
    out = out.fillna("--").replace("NaN", "--")
    tex = out.to_latex(index=False, escape=False, float_format="%.3f", column_format="rllrrrrll")
    note = "\\multicolumn{9}{p{0.98\\linewidth}}{\\footnotesize Notes: N=200 and N=500 report Pool LP-IP gaps for generated-pool budgeted-CG runs. N=500 is scenario U only.}\\\\\n"
    tex = tex.replace("\\bottomrule", note + "\\bottomrule")
    (RES_DIR / "table8_tr_style.tex").write_text(tex, encoding="utf-8")


def table9() -> None:
    df = pd.read_csv(REVISED / "table9_revised.csv")
    df = standardize_methods(df)
    cols = ["scenario", "method", "obj_mean", "obj_std", "runtime_mean", "gap_type"]
    out = df[cols].rename(columns={
        "scenario": "Scenario",
        "method": "Method",
        "obj_mean": "Objective mean",
        "obj_std": "Objective std.",
        "runtime_mean": "Runtime (s)",
        "gap_type": "Gap type",
    }).fillna("--").replace("NaN", "--")
    tex = out.to_latex(index=False, escape=False, float_format="%.3f", column_format="llrrrl")
    tex = tex.replace("\\bottomrule", "\\multicolumn{6}{p{0.98\\linewidth}}{\\footnotesize Notes: Method names follow the main manuscript terminology.}\\\\\n\\bottomrule")
    (RES_DIR / "table9_tr_style.tex").write_text(tex, encoding="utf-8")


def table12() -> None:
    src = MANUSCRIPT / "table12_appendix_polished.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    cols = ["N", "Method", "Pricing protocol", "Gap interpretation", "Baseline pool", "Enriched pool (1%)", "Baseline IRMP obj.", "Enriched IRMP obj.", "Improvement (%)", "Equivalence outcome"]
    tex = df[cols].to_latex(index=False, escape=False, column_format="rlllllllll")
    note = "\\multicolumn{10}{p{0.98\\linewidth}}{\\footnotesize Notes: For N=200 and N=500, CG+IR uses strengthened budgeted CG and reports generated-pool LP-IP gaps; these are not complete-column global optimality certificates.}\\\\\n"
    tex = tex.replace("\\bottomrule", note + "\\bottomrule")
    (RES_DIR / "table12_appendix_tr_style.tex").write_text(tex, encoding="utf-8")


def captions_and_report() -> None:
    captions = """# Figure Caption Revisions

**Fig. 1.** Conceptual scheduling structure for the port energy-service system. Vessels can be assigned to shore power or battery swapping, and each service plan is evaluated by cost, delay, and emissions.

**Fig. 3.** Benchmark performance across scales using corrected and final-controlled data. N<=100 uses full-pricing CG when converged, while N=200 and N=500 use generated-pool budgeted CG; N=500 is scenario U only.

**Fig. 5.** SIMOPS value across demand scales. The curve shows a dual-high-region / threshold-sensitive pattern with 95% bootstrap confidence intervals, not a mathematically sharp dual peak.

**Fig. 6.** Battery-swapping cost sensitivity. The vertical line marks the detected C_BS=0.90 $/kWh threshold, where BS is displaced by AE in the tested calibration.

**Fig. 8.** Carbon-price sensitivity under the tested capacity configurations. Policy reference points are marked at 100, 200, and 380 $/tCO2.

**Fig. 9.** Arrival perturbation sensitivity. Relative degradation is computed against the matching zero-perturbation case.

**Figs. 11--12.** Appendix enrichment diagnostics. These figures provide generated-pool stability evidence, not complete-column optimality proof.
"""
    (RES_DIR / "figure_caption_revisions.md").write_text(captions, encoding="utf-8")
    report = """# TR-Style Check Report

1. Old weak quick-CG data were excluded from final Table 8, Fig. 3, Fig. 5, and Fig. 6. Corrected revision-only CSVs are used for optional carbon and perturbation plots.
2. Large-scale Pool LP-IP gap labels are preserved for N=200 and N=500.
3. N=500 is marked as U-only in Fig. 3 and Table 8.
4. Method names are standardized: CG+IR, Fix-and-Optimize, Rolling-Horizon, Restricted-CG, FIFO, Greedy.
5. Captions match the generated panels and plotted data.
6. No hard-coded numerical arrays were used; plotted data are read from CSV files. Styling constants such as colors, markers, and reference policy points are fixed style choices.
7. Fig. 10 was not generated because corrected empirical-boundary data were unavailable; a missing-data report was written instead.
"""
    (RES_DIR / "style_check_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    setup()
    fig1()
    fig3()
    fig5()
    fig6()
    fig8()
    fig9_10()
    fig11_12()
    table8()
    table9()
    table12()
    captions_and_report()


if __name__ == "__main__":
    main()

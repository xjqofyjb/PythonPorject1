"""Polish appendix Table 12 from existing enrichment CSV data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "experiment_3_followup" / "aggregated.csv"
OUT_DIR = ROOT / "results" / "revised" / "manuscript"


def mean_pm(row: pd.Series, mean_col: str, std_col: str, digits: int = 1) -> str:
    return f"{float(row[mean_col]):.{digits}f} $\\pm$ {float(row[std_col]):.{digits}f}"


def objective_pm(row: pd.Series, mean_col: str, std_col: str) -> str:
    return f"{float(row[mean_col]):,.2f} $\\pm$ {float(row[std_col]):,.2f}"


def build_table() -> pd.DataFrame:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source CSV: {SOURCE}")
    raw = pd.read_csv(SOURCE)
    df = raw[pd.to_numeric(raw["epsilon_pct"], errors="coerce").eq(1.0)].copy()
    if df.empty:
        raise ValueError("No epsilon_pct == 1.0 rows found for Table 12.")

    rows = []
    for _, row in df.sort_values("N").iterrows():
        N = int(row["N"])
        protocol = "Full-pool diagnostic" if N == 100 else "Budgeted top-K pricing"
        gap_scope = "Full-CG LP-IP gap" if N == 100 else "Pool LP-IP gap"
        rows.append(
            {
                "N": N,
                "Method": "CG+IR",
                "Pricing protocol": protocol,
                "Gap interpretation": gap_scope,
                "Baseline pool": mean_pm(row, "baseline_pool_size_mean", "baseline_pool_size_std"),
                "Enriched pool (1%)": mean_pm(row, "enriched_pool_size_mean", "enriched_pool_size_std"),
                "Columns added": mean_pm(row, "columns_added_mean", "columns_added_std"),
                "Baseline IRMP obj.": objective_pm(row, "Z_baseline_IRMP_mean", "Z_baseline_IRMP_std"),
                "Enriched IRMP obj.": objective_pm(row, "Z_enriched_IRMP_mean", "Z_enriched_IRMP_std"),
                "Improvement (%)": f"{float(row['improvement_pct_mean']):.4f} $\\pm$ {float(row['improvement_pct_std']):.4f}",
                "Plans changed": mean_pm(row, "n_plans_changed_mean", "n_plans_changed_std"),
                "Mode switches": mean_pm(row, "n_mode_switches_mean", "n_mode_switches_std"),
                "Enriched cols used": mean_pm(row, "columns_used_from_enrichment_mean", "columns_used_from_enrichment_std"),
                "Equivalence outcome": "Alternative optima" if float(row["alternative_optima_rate"]) == 1.0 else "Mixed",
            }
        )
    return pd.DataFrame(rows)


def write_latex(table: pd.DataFrame, path: Path) -> None:
    display_cols = [
        "N",
        "Method",
        "Pricing protocol",
        "Gap interpretation",
        "Baseline pool",
        "Enriched pool (1%)",
        "Baseline IRMP obj.",
        "Enriched IRMP obj.",
        "Improvement (%)",
        "Enriched cols used",
        "Equivalence outcome",
    ]
    latex = table[display_cols].to_latex(index=False, escape=False, column_format="rllllllllll")
    note = (
        "\\\\[-0.25em]\n"
        "\\multicolumn{11}{p{0.98\\linewidth}}{\\footnotesize Notes: Values are mean $\\pm$ standard deviation over the retained seeds. "
        "Method names follow the main text. For $N=200$ and $N=500$, the CG+IR rows are generated-pool budgeted-CG evidence; "
        "the reported Pool LP-IP gaps indicate pool integrality tightness and incumbent stability, not complete-column global optimality certificates.}\\\\\n"
    )
    latex = latex.replace("\\bottomrule", note + "\\bottomrule")
    path.write_text(latex, encoding="utf-8")


def write_notes(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Table 12 Notes",
        "",
        f"Source CSV: `{SOURCE.relative_to(ROOT)}`.",
        "",
        "Polishing actions:",
        "- Kept the corrected numerical values from the existing enrichment follow-up output.",
        "- Standardized the method name to `CG+IR`.",
        "- Replaced informal pricing labels with manuscript terminology: `Full-pool diagnostic` and `Budgeted top-K pricing`.",
        "- Added explicit gap-interpretation labels.",
        "- Added the large-scale generated-pool caveat for N=200 and N=500.",
        "",
        "Large-scale caveat:",
        "For N=200 and N=500, the CG+IR rows are generated-pool budgeted-CG evidence rather than complete-column global optimality certificates.",
        "",
        "Rows included:",
        *[f"- N={int(row.N)}: {row['Method']}, {row['Pricing protocol']}, {row['Gap interpretation']}" for _, row in table.iterrows()],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    table.to_csv(OUT_DIR / "table12_appendix_polished.csv", index=False)
    write_latex(table, OUT_DIR / "table12_appendix_polished.tex")
    write_notes(table, OUT_DIR / "table12_notes.md")


if __name__ == "__main__":
    main()

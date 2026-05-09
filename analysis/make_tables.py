"""Generate LaTeX tables from results CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ORDER = [
    "cg",
    "milp300",
    "milp60",
    "rolling_horizon",
    "fix_and_optimize",
    "restricted_cg",
    "fifo",
    "greedy",
]
METHOD_LABELS = {
    "cg": "CG+IR",
    "CG+IR": "CG+IR",
    "milp300": "MILP-300",
    "MILP300": "MILP-300",
    "milp60": "MILP-60",
    "MILP60": "MILP-60",
    "rolling_horizon": "Rolling-H",
    "Rolling-Horizon": "Rolling-Horizon",
    "fix_and_optimize": "F\\&O",
    "Fix-and-Optimize": "Fix-and-Optimize",
    "restricted_cg": "Restricted-CG",
    "Restricted-CG": "Restricted-CG",
    "fifo": "FIFO",
    "FIFO": "FIFO",
    "greedy": "Greedy",
    "Greedy": "Greedy",
}


def format_mean_std(series: pd.Series, digits: int = 2) -> str:
    mean = series.mean()
    std = series.std(ddof=0)
    if np.isnan(mean):
        return "-"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    obj_col = "obj" if "obj" in df.columns else "objective"
    runtime_col = "runtime_total" if "runtime_total" in df.columns else "runtime_sec"
    df = df[pd.to_numeric(df.get(obj_col), errors="coerce").notna()].copy()
    grouped = df.groupby(["N", "method"], dropna=False)
    has_gap_pct = "gap_pct" in df.columns and df["gap_pct"].notna().any()
    has_gap = "gap" in df.columns and df["gap"].notna().any()
    rows = []
    for (N, method), g in grouped:
        success_rate = (g["status"] == "ok").mean() if len(g) else 0.0
        row = {
            "N": int(N),
            "method": method,
            "obj": format_mean_std(pd.to_numeric(g[obj_col], errors="coerce")),
            "runtime": format_mean_std(pd.to_numeric(g[runtime_col], errors="coerce"), digits=3),
            "success": f"{success_rate * 100:.1f}\\%",
        }
        if has_gap_pct:
            row["Gap(\\%)"] = format_mean_std(g["gap_pct"])
        elif has_gap:
            row["Gap(\\%)"] = format_mean_std(g["gap"] * 100.0)
        for meta_col in ["cg_status", "gap_type", "pricing_converged", "objective_stabilized"]:
            if meta_col in g.columns:
                vals = [str(v) for v in g[meta_col].dropna().unique() if str(v) != "nan"]
                row[meta_col] = ";".join(vals)
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["method"] = table["method"].map(METHOD_LABELS).fillna(table["method"])
        order = [METHOD_LABELS.get(m, m) for m in METHOD_ORDER if METHOD_LABELS.get(m, m) in table["method"].unique()]
        if order:
            table["method"] = pd.Categorical(table["method"], categories=order, ordered=True)
    return table.sort_values(["N", "method"])


def make_table_simops(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["N", "operation_mode", "method"], dropna=False)
    rows = []
    for (N, op_mode, method), g in grouped:
        success_rate = (g["status"] == "ok").mean() if len(g) else 0.0
        avg_mask = g["avg_masking_rate"] if "avg_masking_rate" in g.columns else pd.Series(dtype=float)
        avg_stay = g["avg_stay_time"] if "avg_stay_time" in g.columns else pd.Series(dtype=float)
        row = {
            "N": int(N),
            "operation_mode": op_mode,
            "method": method,
            "obj": format_mean_std(g["obj"]),
            "runtime": format_mean_std(g["runtime_total"], digits=3),
            "avg_masking_rate": format_mean_std(avg_mask),
            "avg_stay_time": format_mean_std(avg_stay),
            "success": f"{success_rate * 100:.1f}\\%",
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["method"] = table["method"].map(METHOD_LABELS).fillna(table["method"])
        order = [METHOD_LABELS.get(m, m) for m in METHOD_ORDER if METHOD_LABELS.get(m, m) in table["method"].unique()]
        if order:
            table["method"] = pd.Categorical(table["method"], categories=order, ordered=True)
    return table.sort_values(["N", "operation_mode", "method"])


def make_table_scenario(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["scenario", "method"], dropna=False)
    rows = []
    for (scenario, method), g in grouped:
        success_rate = (g["status"] == "ok").mean() if len(g) else 0.0
        n_value = int(g["N"].iloc[0]) if "N" in g.columns and len(g) else 0
        row = {
            "N": n_value,
            "scenario": scenario,
            "method": method,
            "obj": format_mean_std(g["obj"]),
            "runtime": format_mean_std(g["runtime_total"], digits=3),
            "success": f"{success_rate * 100:.1f}\\%",
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        table["method"] = table["method"].map(METHOD_LABELS).fillna(table["method"])
        order = [s for s in ["U", "P", "L"] if s in table["scenario"].unique()]
        if order:
            table["scenario"] = pd.Categorical(table["scenario"], categories=order, ordered=True)
        method_order = [METHOD_LABELS.get(m, m) for m in METHOD_ORDER if METHOD_LABELS.get(m, m) in table["method"].unique()]
        if method_order:
            table["method"] = pd.Categorical(table["method"], categories=method_order, ordered=True)
    return table.sort_values(["scenario", "method"])


def make_table8_final_controlled(source: str | Path) -> pd.DataFrame:
    source = Path(source)
    inputs = [
        source / "n200_table8_full_replacement_summary.csv",
        source / "n500_table8_full_replacement_summary.csv",
    ]
    frames = []
    for path in inputs:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No controlled Table 8 summary CSVs found in {source}")

    table = pd.concat(frames, ignore_index=True)
    table = table[pd.to_numeric(table["objective_mean"], errors="coerce").notna()].copy()
    keep_cols = [
        "N",
        "scenario",
        "method",
        "objective_mean",
        "objective_std",
        "rel_gap_to_CG_mean",
        "runtime_mean",
        "runtime_std",
        "status_success_rate",
        "cg_status",
        "gap_type",
        "pricing_converged_rate",
        "objective_stabilized_rate",
        "pool_gap_pct_mean",
        "columns_mean",
        "iterations_mean",
    ]
    for col in keep_cols:
        if col not in table.columns:
            table[col] = np.nan
    table = table[keep_cols]
    table["method"] = table["method"].map(METHOD_LABELS).fillna(table["method"])
    scenario_order = [s for s in ["U", "P", "L"] if s in table["scenario"].dropna().unique()]
    if scenario_order:
        table["scenario"] = pd.Categorical(table["scenario"], categories=scenario_order, ordered=True)
    method_order = [
        "CG+IR",
        "MILP-60",
        "MILP-300",
        "Rolling-Horizon",
        "Fix-and-Optimize",
        "Restricted-CG",
        "FIFO",
        "Greedy",
    ]
    present = [m for m in method_order if m in table["method"].dropna().unique()]
    if present:
        table["method"] = pd.Categorical(table["method"], categories=present, ordered=True)
    return table.sort_values(["N", "scenario", "method"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Make LaTeX tables from results.")
    parser.add_argument("--in", dest="inp", help="Input CSV path")
    parser.add_argument("--out", help="Output LaTeX file path")
    parser.add_argument("--experiment", default="", help="main/mechanism/sensitivity/simops; inferred if empty")
    parser.add_argument("--source", help="Directory containing controlled result CSVs")
    parser.add_argument("--table", help="Named table to generate")
    parser.add_argument("--output", help="Output directory for named tables")
    args = parser.parse_args()

    if args.table:
        if args.table != "table8_final_controlled":
            raise ValueError(f"Unsupported named table: {args.table}")
        if not args.source or not args.output:
            raise ValueError("--source and --output are required with --table")
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        table = make_table8_final_controlled(args.source)
        csv_path = out_dir / "table8_final_controlled.csv"
        tex_path = out_dir / "table8_final_controlled.tex"
        table.to_csv(csv_path, index=False)
        latex = table.to_latex(index=False, escape=False)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        return

    if not args.inp or not args.out:
        parser.error("--in and --out are required unless --table is used")

    df = pd.read_csv(args.inp)

    experiment = args.experiment
    if not experiment:
        base = args.inp.lower()
        if "simops" in base and "operation_mode" in df.columns:
            experiment = "simops"
        elif "scenario" in base and "scenario" in df.columns:
            experiment = "scenario"
        else:
            experiment = "main"

    if experiment == "simops":
        table = make_table_simops(df)
    elif experiment == "scenario":
        table = make_table_scenario(df)
    else:
        table = make_table(df)

    latex = table.to_latex(index=False, escape=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(latex)


if __name__ == "__main__":
    main()

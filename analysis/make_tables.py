"""Generate LaTeX tables from results CSV."""
from __future__ import annotations

import argparse

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
    "cg": "CG",
    "milp300": "MILP300",
    "milp60": "MILP60",
    "rolling_horizon": "Rolling-H",
    "fix_and_optimize": "F\\&O",
    "restricted_cg": "Restricted-CG",
    "fifo": "FIFO",
    "greedy": "Greedy",
}


def format_mean_std(series: pd.Series, digits: int = 2) -> str:
    mean = series.mean()
    std = series.std(ddof=0)
    if np.isnan(mean):
        return "-"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def make_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["N", "method"], dropna=False)
    has_gap_pct = "gap_pct" in df.columns and df["gap_pct"].notna().any()
    has_gap = "gap" in df.columns and df["gap"].notna().any()
    rows = []
    for (N, method), g in grouped:
        success_rate = (g["status"] == "ok").mean() if len(g) else 0.0
        row = {
            "N": int(N),
            "method": method,
            "obj": format_mean_std(g["obj"]),
            "runtime": format_mean_std(g["runtime_total"], digits=3),
            "success": f"{success_rate * 100:.1f}\\%",
        }
        if has_gap_pct:
            row["Gap(\\%)"] = format_mean_std(g["gap_pct"])
        elif has_gap:
            row["Gap(\\%)"] = format_mean_std(g["gap"] * 100.0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Make LaTeX tables from results.")
    parser.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output LaTeX file path")
    parser.add_argument("--experiment", default="", help="main/mechanism/sensitivity/simops; inferred if empty")
    args = parser.parse_args()

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

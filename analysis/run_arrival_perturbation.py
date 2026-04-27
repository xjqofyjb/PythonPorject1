"""Run fixed-deadline arrival-perturbation robustness experiments for CG+IR."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.style import set_style
from src.instances import Instance, generate_instance
from src.metrics import compute_mode_ratios
from src.runner import build_cg_cfg, expand_seeds, load_config
from src.solvers import cg_solver


PERTURBATION_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_method_cfg(config: dict[str, Any], instance_id: str) -> dict[str, Any]:
    cfg = {
        "method": "cg",
        "operation_mode": "simops",
        "trace_dir": "results/traces",
        "instance_id": instance_id,
        "return_schedule": True,
    }
    cfg.update(build_cg_cfg(config, "cg"))
    return cfg


def perturb_instance_fixed_deadline(instance: Instance, delta_hours: float) -> Instance:
    rng = np.random.default_rng(int(instance.seed) + 1000)
    perturb = rng.uniform(-delta_hours, delta_hours, size=instance.N)
    arrival_times = np.maximum(0.0, instance.arrival_times + perturb)
    deadlines = instance.deadlines.copy()

    dt = float(instance.dt_hours)
    arrival_steps = np.ceil(arrival_times / dt).astype(int)
    deadline_steps = np.ceil(deadlines / dt).astype(int)

    return replace(
        instance,
        arrival_times=arrival_times,
        deadlines=deadlines,
        arrival_steps=arrival_steps,
        deadline_steps=deadline_steps,
    )


def compute_delay_stats(instance: Instance, result: dict[str, Any]) -> tuple[int, float]:
    schedule = result.get("schedule", {})
    starts = np.asarray(schedule.get("service_start_times", []), dtype=float)
    durations = np.asarray(schedule.get("service_durations", []), dtype=float)
    if starts.size != instance.N or durations.size != instance.N:
        return 0, 0.0

    dt = float(instance.dt_hours)
    start_steps = np.rint(starts / dt).astype(int)
    duration_steps = np.rint(durations / dt).astype(int)
    completion_steps = np.maximum(instance.arrival_steps + instance.cargo_steps, start_steps + duration_steps)
    tardy_steps = np.maximum(0, completion_steps - instance.deadline_steps)
    delayed = tardy_steps > 0
    num_delayed = int(np.sum(delayed))
    avg_delay = float(np.mean(tardy_steps[delayed]) * dt) if num_delayed > 0 else 0.0
    return num_delayed, avg_delay


def run_experiment(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    params = dict(config.get("params", {}))
    mechanism = str(config.get("mechanism", "hybrid"))
    seeds = [int(s) for s in expand_seeds(config.get("seeds", 10))]
    scenario = "U"
    N = 100

    rows: list[dict[str, Any]] = []
    for delta in PERTURBATION_LEVELS:
        for seed in seeds:
            base_instance = generate_instance(N, int(seed), scenario, mechanism, params)
            instance = perturb_instance_fixed_deadline(base_instance, float(delta))
            method_cfg = make_method_cfg(config, f"robustfixed_N{N}_seed{seed}_delta{delta:.1f}")
            result = cg_solver.solve(instance, method_cfg, logger=None)
            shares = compute_mode_ratios(result.get("mechanism_counts", {}), instance.N)
            num_delayed, avg_delay_hours = compute_delay_stats(instance, result)

            rows.append(
                {
                    "delta": float(delta),
                    "seed": int(seed),
                    "total_cost": float(result["obj"]),
                    "energy_cost": float(result.get("cost_energy", np.nan)),
                    "delay_cost": float(result.get("cost_delay", np.nan)),
                    "sp_share": float(shares.get("shore_ratio", np.nan)),
                    "bs_share": float(shares.get("battery_ratio", np.nan)),
                    "ae_share": float(shares.get("brown_ratio", np.nan)),
                    "avg_stay_time": float(result.get("avg_stay_time", np.nan)),
                    "masking_rate": float(result.get("avg_masking_rate", np.nan)),
                    "num_delayed_vessels": int(num_delayed),
                    "avg_delay_hours": float(avg_delay_hours),
                    "runtime": float(result.get("runtime_total", np.nan)),
                }
            )
    return pd.DataFrame(rows).sort_values(["delta", "seed"]).reset_index(drop=True)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("delta", as_index=False)
        .agg(
            mean_cost=("total_cost", "mean"),
            std_cost=("total_cost", lambda s: float(np.std(s, ddof=0))),
            mean_delay_cost=("delay_cost", "mean"),
            std_delay_cost=("delay_cost", lambda s: float(np.std(s, ddof=0))),
            mean_energy_cost=("energy_cost", "mean"),
            mean_ae_share=("ae_share", "mean"),
            mean_masking_rate=("masking_rate", "mean"),
            mean_num_delayed=("num_delayed_vessels", "mean"),
            mean_avg_delay_hours=("avg_delay_hours", "mean"),
        )
        .sort_values("delta")
        .reset_index(drop=True)
    )
    baseline = summary.loc[summary["delta"] == 0.0].iloc[0]
    summary["cost_change_pct"] = (summary["mean_cost"] - float(baseline["mean_cost"])) / float(baseline["mean_cost"]) * 100.0
    summary["delay_cost_change_pct"] = (
        (summary["mean_delay_cost"] - float(baseline["mean_delay_cost"])) / max(float(baseline["mean_delay_cost"]), 1e-9) * 100.0
    )
    summary["energy_cost_change_pct"] = (
        (summary["mean_energy_cost"] - float(baseline["mean_energy_cost"])) / float(baseline["mean_energy_cost"]) * 100.0
    )
    return summary


def draw_figure(summary: pd.DataFrame, detail: pd.DataFrame, outdir: Path, stem: str = "Fig_Robustness_FixedDeadline") -> Path:
    set_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.6), constrained_layout=True)
    x = summary["delta"].to_numpy(dtype=float)

    cost_series = [
        ("mean_cost", "std_cost", "Total cost", "#0F4C81", "o"),
        ("mean_energy_cost", None, "Energy cost", "#2F6B5F", "s"),
        ("mean_delay_cost", "std_delay_cost", "Delay cost", "#B85C38", "^"),
    ]
    for mean_col, std_col, label, color, marker in cost_series:
        y = summary[mean_col].to_numpy(dtype=float)
        yerr = summary[std_col].to_numpy(dtype=float) if std_col is not None else None
        ax1.errorbar(x, y, yerr=yerr, color=color, marker=marker, linewidth=2.0, capsize=3, label=label)

    for _, row in summary.iterrows():
        ax1.annotate(
            f"{row['delay_cost_change_pct']:.0f}%",
            xy=(float(row["delta"]), float(row["mean_delay_cost"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8.0,
            color="#B85C38",
        )

    ax1.set_xlabel("Perturbation amplitude $\\Delta$ (hours)")
    ax1.set_ylabel("Cost")
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper left", ncol=1, frameon=False)

    mode_cols = [
        ("sp_share", "SP share", "#0F4C81"),
        ("bs_share", "BS share", "#2F6B5F"),
        ("ae_share", "AE share", "#B85C38"),
    ]
    grouped = detail.groupby("delta", as_index=False).agg(
        sp_share=("sp_share", "mean"),
        bs_share=("bs_share", "mean"),
        ae_share=("ae_share", "mean"),
    )
    for col, label, color in mode_cols:
        ax2.plot(
            grouped["delta"].to_numpy(dtype=float),
            grouped[col].to_numpy(dtype=float) * 100.0,
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )

    ax2.set_xlabel("Perturbation amplitude $\\Delta$ (hours)")
    ax2.set_ylabel("Mode share (%)")
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", ncol=1, frameon=False)

    ensure_dir(outdir)
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-deadline arrival-perturbation robustness analysis for CG+IR.")
    parser.add_argument("--config", default="configs/sensitivity.yaml", help="Base configuration YAML")
    parser.add_argument("--results_dir", default="results", help="Directory for CSV outputs")
    parser.add_argument("--fig_dir", default="figs/paper", help="Directory for figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    ensure_dir(results_dir)
    ensure_dir(fig_dir)

    df = run_experiment(args.config)
    summary = summarize_results(df)

    detail_path = results_dir / "robustness_fixed_deadline.csv"
    summary_path = results_dir / "robustness_fixed_deadline_summary.csv"
    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pdf_path = draw_figure(summary, df, fig_dir)

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

"""Run a dual-layer carbon-price sensitivity experiment for CG+IR."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.style import set_style
from src.instances import generate_instance
from src.metrics import compute_mode_ratios
from src.runner import build_cg_cfg, expand_seeds, load_config
from src.solvers import cg_solver


CARBON_SCENARIOS = [
    (100, 0.65),
    (140, 0.75),
    (200, 0.90),
    (260, 1.05),
    (320, 1.20),
    (380, 1.35),
]

ADEQUATE = {"capacity_config": "adequate", "KSP": 2, "KBS": 2}
CONSTRAINED_PRIMARY = {"capacity_config": "constrained", "KSP": 1, "KBS": 1}
CONSTRAINED_FALLBACK = {"capacity_config": "constrained", "KSP": 1, "KBS": 0}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_method_cfg(config: dict[str, Any], instance_id: str) -> dict[str, Any]:
    cfg = {
        "method": "cg",
        "operation_mode": "simops",
        "trace_dir": "results/traces",
        "instance_id": instance_id,
    }
    cfg.update(build_cg_cfg(config, "cg"))
    return cfg


def _run_layer(
    config: dict[str, Any],
    params: dict[str, Any],
    seeds: list[int],
    layer: dict[str, Any],
    scenario: str,
    mechanism: str,
    N: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    layer_params = dict(params)
    layer_params["shore_cap"] = int(layer["KSP"])
    layer_params["battery_slots"] = int(layer["KBS"])

    for carbon_price, c_ae in CARBON_SCENARIOS:
        local_params = dict(layer_params)
        local_params["brown_cost"] = float(c_ae)
        for seed in seeds:
            instance = generate_instance(N, int(seed), scenario, mechanism, local_params)
            instance_id = (
                f"carbondual_{layer['capacity_config']}_N{N}_seed{seed}_"
                f"ksp{layer['KSP']}_kbs{layer['KBS']}_cae{c_ae:.2f}"
            )
            method_cfg = make_method_cfg(config, instance_id)
            result = cg_solver.solve(instance, method_cfg, logger=None)
            shares = compute_mode_ratios(result.get("mechanism_counts", {}), instance.N)

            rows.append(
                {
                    "capacity_config": str(layer["capacity_config"]),
                    "carbon_price": int(carbon_price),
                    "C_AE": float(c_ae),
                    "KSP": int(layer["KSP"]),
                    "KBS": int(layer["KBS"]),
                    "seed": int(seed),
                    "total_cost": float(result["obj"]),
                    "energy_cost": float(result.get("cost_energy", np.nan)),
                    "delay_cost": float(result.get("cost_delay", np.nan)),
                    "sp_share": float(shares.get("shore_ratio", np.nan)),
                    "bs_share": float(shares.get("battery_ratio", np.nan)),
                    "ae_share": float(shares.get("brown_ratio", np.nan)),
                    "avg_stay_time": float(result.get("avg_stay_time", np.nan)),
                    "masking_rate": float(result.get("avg_masking_rate", np.nan)),
                    "runtime": float(result.get("runtime_total", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _needs_fallback(summary: pd.DataFrame) -> bool:
    constrained = summary[
        (summary["capacity_config"] == "constrained") & (summary["carbon_price"] == 100)
    ]
    if constrained.empty:
        return True
    return bool(np.allclose(constrained["mean_ae_share"].to_numpy(dtype=float), 0.0))


def run_experiment(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    params = dict(config.get("params", {}))
    mechanism = str(config.get("mechanism", "hybrid"))
    seeds = [int(s) for s in expand_seeds(config.get("seeds", 10))]
    scenario = "U"
    N = 100

    frames = [
        _run_layer(config, params, seeds, ADEQUATE, scenario, mechanism, N),
        _run_layer(config, params, seeds, CONSTRAINED_PRIMARY, scenario, mechanism, N),
    ]
    df = pd.concat(frames, ignore_index=True)
    summary = summarize_results(df)

    if _needs_fallback(summary):
        df = df[df["capacity_config"] != "constrained"].copy()
        fallback_df = _run_layer(config, params, seeds, CONSTRAINED_FALLBACK, scenario, mechanism, N)
        df = pd.concat([df, fallback_df], ignore_index=True)

    return df.sort_values(["capacity_config", "carbon_price", "seed"]).reset_index(drop=True)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["capacity_config", "carbon_price", "C_AE"], as_index=False)
        .agg(
            mean_cost=("total_cost", "mean"),
            std_cost=("total_cost", lambda s: float(np.std(s, ddof=0))),
            mean_ae_share=("ae_share", "mean"),
            std_ae_share=("ae_share", lambda s: float(np.std(s, ddof=0))),
            mean_sp_share=("sp_share", "mean"),
            mean_bs_share=("bs_share", "mean"),
            mean_masking_rate=("masking_rate", "mean"),
        )
        .sort_values(["capacity_config", "carbon_price"])
        .reset_index(drop=True)
    )
    return summary


def draw_figure(summary: pd.DataFrame, outdir: Path, stem: str = "Fig_Carbon_Price_DualLayer") -> Path:
    set_style()

    fig, ax1 = plt.subplots(figsize=(6.2, 3.9))
    ax2 = ax1.twinx()

    layers = [
        ("adequate", "Adequate capacity (2SP+2BS)", "-", "#0F4C81", "#B85C38"),
        ("constrained", "Constrained capacity", "--", "#0F4C81", "#B85C38"),
    ]

    handles: list[Any] = []
    labels: list[str] = []

    for capacity_config, label_base, linestyle, cost_color, share_color in layers:
        sub = summary[summary["capacity_config"] == capacity_config].sort_values("carbon_price")
        if sub.empty:
            continue
        x = sub["carbon_price"].to_numpy(dtype=float)
        mean_cost = sub["mean_cost"].to_numpy(dtype=float)
        std_cost = sub["std_cost"].to_numpy(dtype=float)
        mean_ae_share = sub["mean_ae_share"].to_numpy(dtype=float)

        # Reflect the actual constrained setup in the label if fallback was triggered.
        if capacity_config == "constrained":
            ksp = 1
            kbs = 1 if np.any(mean_ae_share > 0) else 0
            label_base = f"Constrained capacity ({ksp}SP+{kbs}BS)"

        h1 = ax1.plot(
            x,
            mean_cost,
            color=cost_color,
            linestyle=linestyle,
            marker="o",
            linewidth=2.2,
        )[0]
        ax1.fill_between(x, mean_cost - std_cost, mean_cost + std_cost, color=cost_color, alpha=0.10)

        h2 = ax2.plot(
            x,
            mean_ae_share * 100.0,
            color=share_color,
            linestyle=linestyle,
            marker="s",
            linewidth=2.0,
        )[0]

        handles.extend([h1, h2])
        labels.extend([f"{label_base}: total cost", f"{label_base}: AE share"])

    anchors = [
        (100, "EU ETS ($100)"),
        (200, "Baseline ($200)"),
        (380, "IMO 2027 RU ($380)"),
    ]
    ymax = float(summary["mean_cost"].max()) if not summary.empty else 0.0
    for x_anchor, text in anchors:
        ax1.axvline(x_anchor, color="#7A7A7A", linewidth=0.9, linestyle=":", alpha=0.9)
        ax1.text(x_anchor, ymax * 1.02, text, fontsize=8.2, color="#555555", ha="center", va="bottom")

    ax1.set_xlabel("Carbon price ($/tonne CO$_2$)")
    ax1.set_ylabel("Total cost")
    ax2.set_ylabel("AE mode share (%)", color="#B85C38")
    ax2.tick_params(axis="y", colors="#B85C38")
    ax2.spines["right"].set_alpha(0.35)
    ax1.grid(True, axis="y")
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    ensure_dir(outdir)
    pdf_path = outdir / f"{stem}.pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dual-layer carbon-price sensitivity analysis for CG+IR.")
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

    detail_path = results_dir / "carbon_price_dual_layer.csv"
    summary_path = results_dir / "carbon_price_dual_summary.csv"
    df.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pdf_path = draw_figure(summary, fig_dir)

    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

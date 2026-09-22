"""
plot_e_variants_comparison.py
Visualizes comparison of the four variants of E:
  1. E (1-step E(0))
  2. E_lambda_fixed (Method 1: FVI Stop-Gradient Form)
  3. E_lambda_differentiable (Method 2: Full Autodiff Moment Scan)
  4. E_lambda_geometric (Method 3: Geometric Jump Sampling)

Produces a 2-panel figure:
  - Left: Learning curves of all 4 variants on the task (best configurations +/- std).
  - Right: Performance scaling as a function of E_LAMBDA in [0.0, 0.5, 0.9].

Usage:
    python notebooks/plot_e_variants_comparison.py --sweep-dir results/ppo/sweeps/suite_12345/CartPole-v1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from notebooks.analyze_sweeps import (
    load_sweep_data,
    extract_best_configuration,
)


VARIANTS_CONFIG = {
    "E": {
        "display_name": "E(0) (1-step)",
        "color": "#7f7f7f",
        "linestyle": ":",
        "marker": "x",
        "zorder": 5,
    },
    "E_lambda_fixed": {
        "display_name": "E(λ) Fixed Trace",
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "^",
        "zorder": 6,
    },
    "E_lambda_fixed_recomputed": {
        "display_name": "E(λ) Refresh Epoch",
        "color": "#d62728",
        "linestyle": "-.",
        "marker": "D",
        "zorder": 7,
    },
    "E_lambda_differentiable": {
        "display_name": "E(λ) Differentiable",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "s",
        "zorder": 8,
    },
    "E_lambda_geometric": {
        "display_name": "E(λ) Geometric Jump",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "o",
        "zorder": 9,
    },
}


def load_variant_data(sweep_dir, algo_name):
    """Loads sweep data for an algorithm directory."""
    algo_path = os.path.join(sweep_dir, algo_name, "tuning")
    if not os.path.exists(algo_path):
        algo_path = os.path.join(sweep_dir, algo_name)
    if not os.path.exists(algo_path):
        return None
    try:
        return load_sweep_data(algo_path)
    except Exception as e:
        print(f"Warning: Could not load data for {algo_name}: {e}")
        return None


def plot_e_variants_comparison(
    sweep_dir: str,
    metric_key: str = "returned_discounted_episode_returns",
    rank_by: str = "final_window",
    window_size: int = 100,
    out_dir: str = None,
    log_scale: bool = False,
    title: str = None,
):
    """
    Plots learning curves for all 4 variants alongside their scaling with E_LAMBDA.
    """
    env_name = os.path.basename(os.path.normpath(sweep_dir))
    data_dict = {}

    target_algos = ["E", "E_lambda_fixed", "E_lambda_differentiable", "E_lambda_geometric"]
    for algo_name in target_algos:
        d = load_variant_data(sweep_dir, algo_name)
        if d is not None:
            data_dict[algo_name] = d

    if not data_dict:
        print(f"Error: No algorithm sweep data found in {sweep_dir}")
        return None

    fig, (ax_curves, ax_scaling) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.8, 1.2]}
    )

    # -------------------------------------------------------------
    # 1. Left Panel: Learning Curves of All Variants
    # -------------------------------------------------------------
    max_steps = 0
    all_trajs = {}

    # Standard E (1-step)
    if "E" in data_dict and metric_key in data_dict["E"].get("metrics", {}):
        try:
            traj, label, _, _ = extract_best_configuration(
                data_dict["E"], metric_key=metric_key, rank_by=rank_by, window_size=window_size
            )
            if traj is not None:
                all_trajs["E"] = (traj, label)
                max_steps = max(max_steps, traj.shape[-1])
        except Exception as ex:
            print(f"Warning: Could not extract E: {ex}")

    # Method 1: E_lambda_fixed (split into fixed vs recomputed if present)
    if "E_lambda_fixed" in data_dict and metric_key in data_dict["E_lambda_fixed"].get("metrics", {}):
        m1_data = data_dict["E_lambda_fixed"]
        summary_df = m1_data.get("summary_df")
        if summary_df is not None and "RECOMPUTE_TARGETS_EACH_EPOCH" in summary_df.columns:
            # Sub-split into False and True
            for flag, var_name in [
                (False, "E_lambda_fixed"),
                (True, "E_lambda_fixed_recomputed"),
            ]:
                sub_df = summary_df[summary_df["RECOMPUTE_TARGETS_EACH_EPOCH"] == flag]
                if not sub_df.empty:
                    best_row = sub_df.iloc[0]
                    c_idx = int(best_row["config_idx"])
                    c_traj = np.asarray(m1_data["metrics"][metric_key][c_idx])
                    hparams = f"λ={best_row.get('E_LAMBDA', '?')}, lr={best_row.get('LR', '?')}"
                    all_trajs[var_name] = (c_traj, hparams)
                    max_steps = max(max_steps, c_traj.shape[-1])
        else:
            try:
                traj, label, _, _ = extract_best_configuration(
                    m1_data, metric_key=metric_key, rank_by=rank_by, window_size=window_size
                )
                if traj is not None:
                    all_trajs["E_lambda_fixed"] = (traj, label)
                    max_steps = max(max_steps, traj.shape[-1])
            except Exception as ex:
                print(f"Warning: Could not extract E_lambda_fixed: {ex}")

    # Method 2: Differentiable E_lambda
    if "E_lambda_differentiable" in data_dict and metric_key in data_dict["E_lambda_differentiable"].get("metrics", {}):
        try:
            traj, label, _, _ = extract_best_configuration(
                data_dict["E_lambda_differentiable"], metric_key=metric_key, rank_by=rank_by, window_size=window_size
            )
            if traj is not None:
                all_trajs["E_lambda_differentiable"] = (traj, label)
                max_steps = max(max_steps, traj.shape[-1])
        except Exception as ex:
            print(f"Warning: Could not extract E_lambda_differentiable: {ex}")

    # Method 3: Geometric Jump Sampled E_lambda
    if "E_lambda_geometric" in data_dict and metric_key in data_dict["E_lambda_geometric"].get("metrics", {}):
        try:
            traj, label, _, _ = extract_best_configuration(
                data_dict["E_lambda_geometric"], metric_key=metric_key, rank_by=rank_by, window_size=window_size
            )
            if traj is not None:
                all_trajs["E_lambda_geometric"] = (traj, label)
                max_steps = max(max_steps, traj.shape[-1])
        except Exception as ex:
            print(f"Warning: Could not extract E_lambda_geometric: {ex}")

    x = np.arange(max_steps)

    for var_key, (traj, label_str) in all_trajs.items():
        style = VARIANTS_CONFIG.get(var_key, {})
        color = style.get("color", "#333333")
        linestyle = style.get("linestyle", "-")
        disp_name = style.get("display_name", var_key)
        zorder = style.get("zorder", 5)

        y_mean = traj.mean(axis=0)
        y_std = traj.std(axis=0)

        curve_len = len(y_mean)
        ax_curves.plot(
            x[:curve_len],
            y_mean,
            label=f"{disp_name} ({label_str})",
            color=color,
            linestyle=linestyle,
            linewidth=2.4 if "geometric" in var_key or "differentiable" in var_key else 1.8,
            zorder=zorder,
        )
        if traj.shape[0] > 1:
            ax_curves.fill_between(
                x[:curve_len],
                y_mean - y_std,
                y_mean + y_std,
                color=color,
                alpha=0.18,
                zorder=zorder - 1,
            )

    if log_scale:
        ax_curves.set_yscale("log")
    ax_curves.set_xlabel("Training Update Steps", fontsize=12)
    ax_curves.set_ylabel(metric_key.replace("_", " ").title(), fontsize=12)
    ax_curves.set_title(f"Learning Curves: 4 Variants of E ({env_name})", fontsize=13, fontweight="bold")
    ax_curves.grid(True, linestyle="--", alpha=0.5)
    ax_curves.legend(loc="best", fontsize=9, frameon=True)

    # -------------------------------------------------------------
    # 2. Right Panel: Scaling with E_LAMBDA
    # -------------------------------------------------------------
    e0_baseline_val = None
    if "E" in all_trajs:
        e0_baseline_val = float(all_trajs["E"][0].mean(axis=0)[-window_size:].mean())

    def extract_lambda_curve(algo_data, filter_col=None, filter_val=None):
        summary_df = algo_data.get("summary_df")
        metrics = algo_data.get("metrics", {}).get(metric_key)
        if summary_df is None or metrics is None:
            return None, None, None

        lam_col = "E_LAMBDA" if "E_LAMBDA" in summary_df.columns else "VALUE_LAMBDA"
        if lam_col not in summary_df.columns:
            return None, None, None

        sub_df = summary_df
        if filter_col and filter_col in summary_df.columns:
            sub_df = summary_df[summary_df[filter_col] == filter_val]

        if sub_df.empty:
            return None, None, None

        lambdas = sorted(sub_df[lam_col].unique())
        means = []
        stds = []
        valid_lambdas = []

        for lam in lambdas:
            lam_df = sub_df[sub_df[lam_col] == lam]
            if lam_df.empty:
                continue
            best_row = lam_df.iloc[0]
            c_idx = int(best_row["config_idx"])
            c_traj = np.asarray(metrics[c_idx])
            final_window = c_traj[:, -window_size:]  # (n_seeds, window)
            seed_means = final_window.mean(axis=1)  # (n_seeds,)
            means.append(float(seed_means.mean()))
            stds.append(float(seed_means.std()))
            valid_lambdas.append(float(lam))

        return np.array(valid_lambdas), np.array(means), np.array(stds)

    # Plot lines for methods across E_LAMBDA
    # 1. Method 3: Geometric Jump
    if "E_lambda_geometric" in data_dict:
        lams, means, stds = extract_lambda_curve(data_dict["E_lambda_geometric"])
        if lams is not None and len(lams) > 0:
            cfg = VARIANTS_CONFIG["E_lambda_geometric"]
            ax_scaling.plot(lams, means, label=cfg["display_name"], color=cfg["color"],
                            marker=cfg["marker"], linewidth=2.5, markersize=8, zorder=9)
            ax_scaling.fill_between(lams, means - stds, means + stds, color=cfg["color"], alpha=0.15)

    # 2. Method 2: Differentiable
    if "E_lambda_differentiable" in data_dict:
        lams, means, stds = extract_lambda_curve(data_dict["E_lambda_differentiable"])
        if lams is not None and len(lams) > 0:
            cfg = VARIANTS_CONFIG["E_lambda_differentiable"]
            ax_scaling.plot(lams, means, label=cfg["display_name"], color=cfg["color"],
                            marker=cfg["marker"], linewidth=2.5, markersize=8, zorder=8)
            ax_scaling.fill_between(lams, means - stds, means + stds, color=cfg["color"], alpha=0.15)

    # 3. Method 1: E_lambda_fixed (Fixed trace)
    if "E_lambda_fixed" in data_dict:
        lams, means, stds = extract_lambda_curve(
            data_dict["E_lambda_fixed"],
            filter_col="RECOMPUTE_TARGETS_EACH_EPOCH",
            filter_val=False,
        )
        if lams is not None and len(lams) > 0:
            cfg = VARIANTS_CONFIG["E_lambda_fixed"]
            ax_scaling.plot(lams, means, label=cfg["display_name"], color=cfg["color"],
                            marker=cfg["marker"], linestyle=cfg["linestyle"], linewidth=2.0, markersize=7, zorder=7)
            ax_scaling.fill_between(lams, means - stds, means + stds, color=cfg["color"], alpha=0.15)

    # 4. Method 1: E_lambda_fixed (Recomputed trace)
    if "E_lambda_fixed" in data_dict:
        lams, means, stds = extract_lambda_curve(
            data_dict["E_lambda_fixed"],
            filter_col="RECOMPUTE_TARGETS_EACH_EPOCH",
            filter_val=True,
        )
        if lams is not None and len(lams) > 0:
            cfg = VARIANTS_CONFIG["E_lambda_fixed_recomputed"]
            ax_scaling.plot(lams, means, label=cfg["display_name"], color=cfg["color"],
                            marker=cfg["marker"], linestyle=cfg["linestyle"], linewidth=2.0, markersize=7, zorder=6)
            ax_scaling.fill_between(lams, means - stds, means + stds, color=cfg["color"], alpha=0.15)

    # Baseline: E(0) (1-step)
    if e0_baseline_val is not None:
        cfg = VARIANTS_CONFIG["E"]
        ax_scaling.axhline(
            y=e0_baseline_val,
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=2.0,
            label=f"{cfg['display_name']} Baseline",
            zorder=4,
        )

    ax_scaling.set_xlabel("Dirichlet Parameter $E(\\lambda)$", fontsize=12)
    ax_scaling.set_ylabel(f"Final Return (Window={window_size})", fontsize=12)
    ax_scaling.set_title(f"Scaling with $E(\\lambda)$ ({env_name})", fontsize=13, fontweight="bold")
    ax_scaling.grid(True, linestyle="--", alpha=0.5)
    ax_scaling.set_xticks([0.0, 0.5, 0.9])
    ax_scaling.legend(loc="best", fontsize=9, frameon=True)

    fig.tight_layout()

    out_dir = out_dir or os.path.join(sweep_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, "comparison_e_variants.png")
    pdf_path = os.path.join(out_dir, "comparison_e_variants.pdf")

    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved E variants comparison PNG: {png_path}")
    print(f"Saved E variants comparison PDF: {pdf_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot comparison of 4 E variants and E(lambda) scaling")
    parser.add_argument("--sweep-dir", type=str, required=True, help="Sweep environment directory containing variants")
    parser.add_argument("--metric", type=str, default="returned_discounted_episode_returns", help="Metric to plot")
    parser.add_argument("--rank-by", type=str, default="final_window", help="Ranking criterion")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for final score")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log-scale", action="store_true", help="Log scale for learning curves")

    args = parser.parse_args()
    plot_e_variants_comparison(
        sweep_dir=args.sweep_dir,
        metric_key=args.metric,
        rank_by=args.rank_by,
        window_size=args.window_size,
        out_dir=args.out_dir,
        log_scale=args.log_scale,
    )


if __name__ == "__main__":
    main()

"""
analyze_sweeps.py
Modular analysis, extraction, and visualization tools for hyperparameter sweeps
and cross-algorithm comparisons (including comparing E against the full spectrum of lambda).
"""

import sys
import os
import json
import cloudpickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Force JAX to CPU to prevent GPU VRAM allocation during analysis
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from core.utils import load_run_data, load_run_data_from_path


def find_latest_run_dir(base_dir):
    """
    Finds the latest timestamp directory and environment subfolder under a tuning/results base directory.
    Returns: (timestamp, env_name, full_path) or (None, None, None)
    """
    if not os.path.exists(base_dir):
        return None, None, None

    # Check if base_dir itself directly contains config.json / out.pkl
    if os.path.exists(os.path.join(base_dir, "config.json")) and (
        os.path.exists(os.path.join(base_dir, "out.pkl")) or os.path.exists(os.path.join(base_dir, "best_config.json"))
    ):
        env_name = os.path.basename(base_dir)
        parent_name = os.path.basename(os.path.dirname(base_dir))
        return parent_name, env_name, base_dir

    subdirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")])
    if not subdirs:
        return None, None, None

    latest_sub = subdirs[-1]
    sub_path = os.path.join(base_dir, latest_sub)

    envs = [e for e in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, e)) and not e.startswith(".")]
    if envs:
        return latest_sub, envs[0], os.path.join(sub_path, envs[0])

    if os.path.exists(os.path.join(sub_path, "config.json")):
        return latest_sub, "default", sub_path

    return None, None, None


def load_sweep_data(path_or_base_dir, env_name=None):
    """
    Loads all data associated with a sweep run: config, out.pkl metrics,
    tuning_summary.csv, and best_config.json.
    """
    if os.path.isdir(path_or_base_dir) and (
        os.path.exists(os.path.join(path_or_base_dir, "config.json"))
        or os.path.exists(os.path.join(path_or_base_dir, "best_config.json"))
    ):
        run_dir = path_or_base_dir
        timestamp = os.path.basename(os.path.dirname(run_dir))
        env = os.path.basename(run_dir)
    else:
        timestamp, env, run_dir = find_latest_run_dir(path_or_base_dir)
        if not run_dir:
            raise FileNotFoundError(f"Could not find valid run directory in: {path_or_base_dir}")

    config_path = os.path.join(run_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    pkl_path = os.path.join(run_dir, "out.pkl")
    metrics = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            raw = cloudpickle.load(f)
        if isinstance(raw, dict) and "metrics" in raw:
            metrics = raw["metrics"]
        elif isinstance(raw, dict):
            metrics = raw

    csv_path = os.path.join(run_dir, "tuning_summary.csv")
    summary_df = None
    if os.path.exists(csv_path):
        summary_df = pd.read_csv(csv_path)

    best_config_path = os.path.join(run_dir, "best_config.json")
    best_config = None
    if os.path.exists(best_config_path):
        with open(best_config_path, "r") as f:
            best_config = json.load(f)

    return {
        "config": config,
        "metrics": metrics,
        "summary_df": summary_df,
        "best_config": best_config,
        "run_dir": run_dir,
        "env_name": env,
        "timestamp": timestamp,
    }


def extract_best_configuration(
    sweep_data,
    metric_key="returned_episode_returns",
    rank_by="auc",
    rank_order="higher",
    config_idx=None,
    window_size=20,
):
    """
    Extracts the 2D seed trajectories (n_seeds, time_steps) and hyperparameter label
    for the best performing configuration in a sweep.
    """
    metrics = sweep_data["metrics"]
    summary_df = sweep_data.get("summary_df")
    best_config_meta = sweep_data.get("best_config")

    if metric_key not in metrics:
        lower_keys = {k.lower(): k for k in metrics.keys()}
        if metric_key.lower() in lower_keys:
            metric_key = lower_keys[metric_key.lower()]
        else:
            raise KeyError(f"Metric '{metric_key}' not found in metrics. Available: {list(metrics.keys())}")

    arr = np.asarray(metrics[metric_key])

    if arr.ndim == 2:
        return arr, "Default", 0, {}
    elif arr.ndim == 3:
        n_combos, n_seeds, time_steps = arr.shape

        if config_idx is not None:
            best_idx = int(config_idx)
        else:
            rank_by_lower = rank_by.lower()
            if rank_by_lower in ["auc", "mean", "time_mean", "auc_mean"]:
                scores = arr.mean(axis=(1, 2))
            elif rank_by_lower in ["final_window", "window", "final_window_mean"]:
                win = max(1, min(time_steps, window_size))
                scores = arr[:, :, -win:].mean(axis=(1, 2))
            elif rank_by_lower in ["final", "final_step", "final_mean", "last"]:
                scores = arr[:, :, -1].mean(axis=1)
            elif rank_by_lower in ["min", "minimum"]:
                scores = arr.min(axis=-1).mean(axis=1)
            elif rank_by_lower in ["max", "maximum"]:
                scores = arr.max(axis=-1).mean(axis=1)
            else:
                scores = arr.mean(axis=(1, 2))

            is_lower = rank_order.lower() in ["lower", "min", "asc", "ascending"]
            best_idx = int(np.argmin(scores)) if is_lower else int(np.argmax(scores))

        best_hparams = {}
        if summary_df is not None and not summary_df.empty:
            match = summary_df[summary_df["config_idx"] == best_idx]
            if not match.empty:
                row = match.iloc[0]
                def is_metric_col(col):
                    c = col.lower()
                    if c in ["rank", "config_idx", "timestamp", "env_name", "env"]:
                        return True
                    for prefix in ["auc", "final", "final_window", "mean", "min", "max", "std"]:
                        if c.startswith(prefix + "_") or c == prefix:
                            return True
                    return False

                hparam_cols = [c for c in summary_df.columns if not is_metric_col(c)]
                best_hparams = {c: row[c] for c in hparam_cols}

        if not best_hparams and best_config_meta is not None and best_config_meta.get("best_config_idx") == best_idx:
            best_hparams = best_config_meta.get("best_hyperparameters", {})

        best_label = ", ".join([f"{k}={v}" for k, v in best_hparams.items()]) if best_hparams else f"Config #{best_idx}"
        seed_trajectories = arr[best_idx]
        return seed_trajectories, best_label, best_idx, best_hparams
    else:
        raise ValueError(f"Unexpected shape for metric array: {arr.shape}")


def plot_algorithm_comparison(
    algorithms_dict,
    metric_key="returned_episode_returns",
    ylabel=None,
    title=None,
    log_scale=False,
    use_geom_mean=False,
    rank_by="auc",
    rank_order="higher",
    window_size=20,
    save_path=None,
    env_name=None,
):
    """
    Plots best learning curves for each algorithm with error bands.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(algorithms_dict), 10)))

    color_map = {
        "sampled_E": "#2ca02c",  # Vibrant green
        "E": "#2ca02c",
        "E_min": "#2ca02c",
        "sampled_td_lambda": "#1f77b4",  # Blue
        "td_lambda": "#1f77b4",
        "sampled_mc": "#ff7f0e",  # Orange
        "mc": "#ff7f0e",
    }

    for idx, (algo_name, sweep_data) in enumerate(algorithms_dict.items()):
        try:
            seed_trajectories, best_label, _, _ = extract_best_configuration(
                sweep_data,
                metric_key=metric_key,
                rank_by=rank_by,
                rank_order=rank_order,
                window_size=window_size,
            )
        except Exception as e:
            print(f"Skipping {algo_name}: {e}")
            continue

        n_seeds, time_steps = seed_trajectories.shape
        x = list(range(time_steps))
        mean_curve = seed_trajectories.mean(axis=0)
        std_curve = seed_trajectories.std(axis=0)

        color = color_map.get(algo_name, colors[idx % len(colors)])
        display_label = f"{algo_name} ({best_label})" if best_label else algo_name
        ax.plot(x, mean_curve, label=display_label, color=color, linewidth=2.4)
        if n_seeds > 1:
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.18)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Update Steps", fontsize=12)
    ax.set_ylabel(ylabel or metric_key, fontsize=12)
    ax.set_title(title or f"Algorithm Comparison - {env_name or ''}", fontsize=13, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Comparison plot saved to {save_path}")

    return fig


def summarize_algorithm_comparison(
    algorithms_dict,
    metric_key="returned_episode_returns",
    rank_by="auc",
    rank_order="higher",
    window_size=20,
    save_path=None,
):
    """
    Generates a DataFrame summarizing the best configurations for each algorithm.
    """
    rows = []
    for algo_name, sweep_data in algorithms_dict.items():
        try:
            seed_trajectories, best_label, best_idx, best_hparams = extract_best_configuration(
                sweep_data,
                metric_key=metric_key,
                rank_by=rank_by,
                rank_order=rank_order,
                window_size=window_size,
            )
        except Exception:
            continue

        n_seeds, time_steps = seed_trajectories.shape
        win = max(1, min(time_steps, window_size))
        mean_curve = seed_trajectories.mean(axis=0)

        auc_mean = float(mean_curve.mean())
        auc_std = float(seed_trajectories.mean(axis=-1).std()) if n_seeds > 1 else 0.0
        window_mean = float(mean_curve[-win:].mean())
        window_std = float(seed_trajectories[:, -win:].mean(axis=-1).std()) if n_seeds > 1 else 0.0
        final_mean = float(mean_curve[-1])
        final_std = float(seed_trajectories[:, -1].std()) if n_seeds > 1 else 0.0

        rows.append({
            "algorithm": algo_name,
            "best_config_idx": best_idx,
            "best_hyperparameters": str(best_hparams),
            f"auc_{metric_key}": auc_mean,
            f"auc_{metric_key}_std": auc_std,
            f"final_window_{metric_key}": window_mean,
            f"final_window_{metric_key}_std": window_std,
            f"final_{metric_key}_mean": final_mean,
            f"final_{metric_key}_std": final_std,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        sort_col = f"final_window_{metric_key}" if rank_by == "final_window" else f"auc_{metric_key}"
        is_ascending = rank_order.lower() in ["lower", "min", "asc", "ascending"]
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=is_ascending).reset_index(drop=True)
            df["rank"] = df.index + 1

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        df.to_csv(save_path, index=False)

    return df


def plot_lambda_spectrum_vs_E(
    td_sweep_data,
    e_sweep_data,
    metric_key="returned_episode_returns",
    lambda_param="VALUE_LAMBDA",
    title=None,
    ylabel=None,
    save_path=None,
    log_scale=False,
):
    """
    Plots learning curves for the full spectrum of lambda alongside E-minimization.
    """
    fig, (ax_curves, ax_sensitivity) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1]})

    # Extract E curve
    e_traj, e_label, _, _ = extract_best_configuration(e_sweep_data, metric_key=metric_key)
    e_mean = e_traj.mean(axis=0)
    e_std = e_traj.std(axis=0)
    x = list(range(len(e_mean)))

    # Plot E in distinct bold green
    ax_curves.plot(x, e_mean, label=f"★ E-Minimization ({e_label})", color="#2ca02c", linewidth=3.2, zorder=10)
    if e_traj.shape[0] > 1:
        ax_curves.fill_between(x, e_mean - e_std, e_mean + e_std, color="#2ca02c", alpha=0.25, zorder=9)

    # Extract all lambda configurations from TD sweep
    td_metrics = td_sweep_data["metrics"][metric_key]  # shape: (n_combos, n_seeds, time_steps)
    summary_df = td_sweep_data["summary_df"]

    lambda_vals = []
    lambda_final_scores = []

    if summary_df is not None and lambda_param in summary_df.columns:
        # Group by lambda, find best LR for each lambda
        unique_lambdas = sorted(summary_df[lambda_param].unique())
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(unique_lambdas)))

        for idx, lam in enumerate(unique_lambdas):
            lam_df = summary_df[summary_df[lambda_param] == lam]
            best_row = lam_df.iloc[0]
            best_c_idx = int(best_row["config_idx"])

            c_traj = np.asarray(td_metrics[best_c_idx])
            c_mean = c_traj.mean(axis=0)

            lam_label = f"TD(λ={lam})" if lam < 1.0 else "MC (λ=1.0)"
            ax_curves.plot(x, c_mean, label=lam_label, color=colors[idx], linewidth=1.8, linestyle="--", alpha=0.85)

            lambda_vals.append(float(lam))
            lambda_final_scores.append(float(c_mean[-20:].mean()))

    # Styling for left curve plot
    if log_scale:
        ax_curves.set_yscale("log")
    ax_curves.set_xlabel("Update Steps", fontsize=12)
    ax_curves.set_ylabel(ylabel or metric_key, fontsize=12)
    ax_curves.set_title("Learning Curves: E vs. TD(λ) Spectrum", fontsize=13, fontweight="bold")
    ax_curves.grid(True, linestyle="--", alpha=0.5)
    ax_curves.legend(loc="best", fontsize=9, frameon=True)

    # Right panel: Sensitivity plot (Final Performance vs Lambda)
    if lambda_vals:
        ax_sensitivity.plot(lambda_vals, lambda_final_scores, marker="o", color="#1f77b4", linewidth=2.0, label="TD(λ) Best-LR")
        e_final = float(e_mean[-20:].mean())
        ax_sensitivity.axhline(e_final, color="#2ca02c", linestyle="-", linewidth=2.5, label=f"E-Min (Score={e_final:.2f})")
        ax_sensitivity.set_xlabel("Lambda (λ)", fontsize=12)
        ax_sensitivity.set_ylabel(f"Final Window ({metric_key})", fontsize=12)
        ax_sensitivity.set_title("Performance vs. λ", fontsize=13, fontweight="bold")
        ax_sensitivity.grid(True, linestyle="--", alpha=0.5)
        ax_sensitivity.legend(loc="best", fontsize=9, frameon=True)

    fig.suptitle(title or "E-Minimization vs. TD(λ) Spectrum Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Lambda spectrum comparison plot saved to {save_path}")

    return fig

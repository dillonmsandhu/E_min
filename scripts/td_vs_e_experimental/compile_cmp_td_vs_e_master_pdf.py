#!/usr/bin/env python3
"""
scripts/compile_cmp_td_vs_e_master_pdf.py

Compiles all environment runs from a Classic TD vs. E_experimental comparison sweep
(e.g., results/ppo/sweeps/cmp_td_vs_e_12724069/ or results/sweeps/cmp_td_vs_e_12724069/)
into a comprehensive, publication-grade master report:

Outputs:
1. <suite_dir>/master_summary_table.csv: Complete suite-wide comparison metrics, win rates, and deltas
2. <suite_dir>/master_td_vs_e_report.pdf: Multi-page publication-grade vector PDF:
   - Page 1: Executive Dashboard (Summary stats, win rate breakdown, performance delta bar chart)
   - Page 2: Cross-Environment Comparison Grid (Best E vs Best TD learning curves, with in-progress indicators)
   - Pages 3+: Per-environment 4-panel detailed comparison posters (embedded from disk or dynamically generated)

Usage:
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py --suite-dir results/ppo/sweeps/cmp_td_vs_e_12724069
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py cmp_td_vs_e_12724069 --rank-by auc
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py 12724069 --window-size 50
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py --suite-dir results/ppo/sweeps/cmp_td_vs_e_12724069 --rank-by auc --window-size 100
"""

import os
import sys

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import glob
import math
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Vector publication settings
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10


def resolve_suite_directory(raw_path: str) -> str:
    """Resolves raw path, suite name, or job ID into a valid directory path."""
    candidates = [
        raw_path,
        os.path.join("results/ppo/sweeps", raw_path),
        os.path.join("results/ppo/sweeps", f"cmp_td_vs_e_{raw_path}"),
        os.path.join("results/sweeps", raw_path),
        os.path.join("results/sweeps", f"cmp_td_vs_e_{raw_path}"),
        os.path.join("results", raw_path),
        os.path.join("results", f"cmp_td_vs_e_{raw_path}"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    # Try glob matching
    patterns = [
        f"results/ppo/sweeps/*{raw_path}*",
        f"results/sweeps/*{raw_path}*",
        f"results/*{raw_path}*",
        f"*{raw_path}*",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        for m in matches:
            if os.path.isdir(m):
                return os.path.abspath(m)

    raise FileNotFoundError(f"Could not find sweep directory for '{raw_path}'. Checked candidates: {candidates}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile TD vs E_experimental sweep into a master PDF report with customizable ranking"
    )
    parser.add_argument("sweep_target", nargs="?", default=None,
                        help="Suite path, suite directory name, or SLURM Job ID (e.g., cmp_td_vs_e_12724069 or 12724069)")
    parser.add_argument("--suite-dir", type=str, default=None,
                        help="Explicit path to suite directory")
    parser.add_argument("--output-pdf", type=str, default=None,
                        help="Custom path for generated master PDF (default: <suite_dir>/master_td_vs_e_report.pdf)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Custom path for generated summary CSV (default: <suite_dir>/master_summary_table.csv)")
    parser.add_argument("--rank-by", type=str, default="final_window",
                        choices=["final_window", "auc", "max"],
                        help="Metric to rank best configuration and head-to-head win rate: 'final_window', 'auc', or 'max' (default: final_window)")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size in updates for calculating final evaluation return (default: 100)")
    parser.add_argument("--metric", type=str, default=None,
                        help="Name of metric array inside metrics.pkl (default: auto-detect episode returns)")
    return parser.parse_args()


def extract_curves_and_metrics(pkl_path: str, preferred_metric: str = None):
    """
    Extracts {config_label: (mean_curve, sem_curve, raw_array)} from metrics.pkl.
    Handles multiple dict layouts:
      1. {config_label: {metric_name: ndarray(n_seeds, updates)}}
      2. {config_label: (mean, sem)}
      3. {"curves": {config_label: (mean, sem)}}
    """
    if not os.path.isfile(pkl_path):
        return {}

    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as ex:
        print(f"  Warning: Failed to load {pkl_path}: {ex}")
        return {}

    if not isinstance(data, dict):
        return {}

    if "curves" in data and isinstance(data["curves"], dict):
        out = {}
        for k, v in data["curves"].items():
            if isinstance(v, (tuple, list)) and len(v) >= 2:
                out[k] = (np.asarray(v[0]), np.asarray(v[1]), None)
            else:
                out[k] = (np.asarray(v), np.zeros_like(np.asarray(v)), None)
        return out

    results = {}
    for cfg_lbl, entry in data.items():
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            results[cfg_lbl] = (np.asarray(entry[0]), np.asarray(entry[1]), None)
        elif isinstance(entry, dict):
            candidates = [preferred_metric] if preferred_metric else []
            candidates += [
                "returned_episode_returns",
                "returned_discounted_episode_returns",
                "returns",
                "mean_rew",
            ]
            arr = None
            for c in candidates:
                if c and c in entry:
                    arr = np.asarray(entry[c])
                    break
            if arr is None and len(entry) > 0:
                first_v = list(entry.values())[0]
                if isinstance(first_v, (np.ndarray, list)):
                    arr = np.asarray(first_v)

            if arr is not None:
                if arr.ndim > 1:
                    mean_c = arr.mean(axis=0)
                    sem_c = arr.std(axis=0) / np.sqrt(max(1, arr.shape[0]))
                else:
                    mean_c = arr
                    sem_c = np.zeros_like(mean_c)
                results[cfg_lbl] = (mean_c, sem_c, arr)
    return results


def detect_algo_subdirs(env_dir: str):
    """Finds E and TD algorithm subdirectories inside an environment directory."""
    e_candidates = ["E_experimental", "summary_e_experimental.csv", "E_lambda_experimental", "e_0", "E", "E0"]
    td_candidates = ["td", "summary_td.csv", "baseline_ppo", "fitted", "td_0", "TD"]

    e_dir = None
    td_dir = None

    for cand in e_candidates:
        p = os.path.join(env_dir, cand)
        if os.path.isdir(p):
            e_dir = p
            break

    for cand in td_candidates:
        p = os.path.join(env_dir, cand)
        if os.path.isdir(p):
            td_dir = p
            break

    # Fallback: scan any subdirectories with summary_*.csv
    if e_dir is None or td_dir is None:
        for sub in sorted(os.listdir(env_dir)):
            sub_p = os.path.join(env_dir, sub)
            if not os.path.isdir(sub_p):
                continue
            sub_lower = sub.lower()
            if e_dir is None and any(tag in sub_lower for tag in ["e_exp", "e_lambda", "e0", "e_0", "sampled_e"]):
                e_dir = sub_p
            elif td_dir is None and any(tag in sub_lower for tag in ["td", "fitted", "baseline"]):
                td_dir = sub_p

    return e_dir, td_dir


def load_algo_data(algo_dir: str, window_size: int = 100, preferred_metric: str = None):
    """Loads CSV and pickles for an algorithm and computes window/AUC/max metrics."""
    if not algo_dir or not os.path.isdir(algo_dir):
        return None, {}

    # Find summary csv
    summary_files = glob.glob(os.path.join(algo_dir, "summary_*.csv"))
    df = None
    if summary_files:
        try:
            df = pd.read_csv(summary_files[0])
        except Exception:
            df = None

    # Load curves
    pkl_path = os.path.join(algo_dir, "metrics.pkl")
    curves = extract_curves_and_metrics(pkl_path, preferred_metric)

    # If df is None but we have curves, build df
    if (df is None or df.empty) and curves:
        rows = []
        for cfg_lbl, (m_c, s_c, _) in curves.items():
            win = min(len(m_c), window_size)
            rows.append({
                "config": cfg_lbl,
                "final_window_mean": float(np.mean(m_c[-win:])),
                "final_window_sem": float(np.mean(s_c[-win:])),
                "auc": float(np.mean(m_c)),
                "max_return": float(np.max(m_c)),
            })
        df = pd.DataFrame(rows)
    elif df is not None and not df.empty and curves:
        # Recompute final_window_mean, auc, max_return dynamically for accuracy with window_size
        win_means = []
        win_sems = []
        aucs = []
        max_rets = []
        for _, row in df.iterrows():
            cfg = str(row["config"])
            if cfg in curves:
                m_c, s_c, _ = curves[cfg]
                win = min(len(m_c), window_size)
                win_means.append(float(np.mean(m_c[-win:])))
                win_sems.append(float(np.mean(s_c[-win:])))
                aucs.append(float(np.mean(m_c)))
                max_rets.append(float(np.max(m_c)))
            else:
                win_means.append(row.get("final_window_mean", np.nan))
                win_sems.append(row.get("final_window_sem", 0.0))
                aucs.append(row.get("auc", np.nan))
                max_rets.append(row.get("max_return", row.get("final_window_mean", np.nan)))
        df["final_window_mean"] = win_means
        df["final_window_sem"] = win_sems
        df["auc"] = aucs
        df["max_return"] = max_rets

    return df, curves


def load_env_results(env_dir: str, rank_by: str = "final_window", window_size: int = 100, preferred_metric: str = None):
    """Extracts summary and curve data for a single environment in the suite."""
    env_name = os.path.basename(env_dir)
    poster_pdf = os.path.join(env_dir, "head_to_head_poster.pdf")
    poster_png = os.path.join(env_dir, "head_to_head_poster.png")

    e_dir, td_dir = detect_algo_subdirs(env_dir)

    df_e, curves_e = load_algo_data(e_dir, window_size, preferred_metric)
    df_td, curves_td = load_algo_data(td_dir, window_size, preferred_metric)

    # Determine status
    has_e = df_e is not None and not df_e.empty
    has_td = df_td is not None and not df_td.empty

    if has_e and has_td:
        status = "Complete"
    elif has_e and not has_td:
        status = "Incomplete (E Only)"
    elif not has_e and has_td:
        status = "Incomplete (TD Only)"
    else:
        status = "In Progress / No Data"

    # Map rank_by to column name
    metric_col_map = {
        "final_window": "final_window_mean",
        "auc": "auc",
        "max": "max_return",
    }
    target_metric = metric_col_map.get(rank_by, "final_window_mean")

    # Extract best configs
    if has_e:
        sort_col = target_metric if target_metric in df_e.columns else "final_window_mean"
        best_e_row = df_e.sort_values(sort_col, ascending=False).iloc[0]
        best_e_score = float(best_e_row[sort_col])
        best_e_final_win = float(best_e_row.get("final_window_mean", np.nan))
        best_e_auc = float(best_e_row.get("auc", np.nan))
        best_e_sem = float(best_e_row.get("final_window_sem", 0.0))
        best_e_cfg = str(best_e_row["config"])
    else:
        best_e_score = best_e_final_win = best_e_auc = best_e_sem = np.nan
        best_e_cfg = "N/A"

    if has_td:
        sort_col = target_metric if target_metric in df_td.columns else "final_window_mean"
        best_td_row = df_td.sort_values(sort_col, ascending=False).iloc[0]
        best_td_score = float(best_td_row[sort_col])
        best_td_final_win = float(best_td_row.get("final_window_mean", np.nan))
        best_td_auc = float(best_td_row.get("auc", np.nan))
        best_td_sem = float(best_td_row.get("final_window_sem", 0.0))
        best_td_cfg = str(best_td_row["config"])
    else:
        best_td_score = best_td_final_win = best_td_auc = best_td_sem = np.nan
        best_td_cfg = "N/A"

    delta_best = best_e_score - best_td_score if not (np.isnan(best_e_score) or np.isnan(best_td_score)) else np.nan
    delta_best_auc = best_e_auc - best_td_auc if not (np.isnan(best_e_auc) or np.isnan(best_td_auc)) else np.nan
    delta_best_win = best_e_final_win - best_td_final_win if not (np.isnan(best_e_final_win) or np.isnan(best_td_final_win)) else np.nan

    # Matched Pair Merging
    df_h2h = None
    e_wins = td_wins = ties = pairs = 0
    p_val = mean_delta_grid = np.nan

    if has_e and has_td:
        # Determine merge keys
        common_cols = [c for c in ["critic_lr", "critic_epochs", "weight_decay", "num_value_heads"]
                       if c in df_e.columns and c in df_td.columns]
        if not common_cols and "config" in df_e.columns and "config" in df_td.columns:
            common_cols = ["config"]

        if common_cols:
            df_h2h = pd.merge(df_e, df_td, on=common_cols, suffixes=("_e", "_td"))

        # Fallback to existing head_to_head_summary.csv if merge failed
        if (df_h2h is None or df_h2h.empty):
            existing_h2h = os.path.join(env_dir, "head_to_head_summary.csv")
            if os.path.isfile(existing_h2h):
                try:
                    df_h2h = pd.read_csv(existing_h2h)
                except Exception:
                    df_h2h = None

        if df_h2h is not None and not df_h2h.empty:
            m_col_e = f"{target_metric}_e" if f"{target_metric}_e" in df_h2h.columns else "final_window_mean_e"
            m_col_td = f"{target_metric}_td" if f"{target_metric}_td" in df_h2h.columns else "final_window_mean_td"

            if m_col_e in df_h2h.columns and m_col_td in df_h2h.columns:
                deltas = df_h2h[m_col_e] - df_h2h[m_col_td]
                df_h2h["delta_ranked"] = deltas
                e_wins = int(np.sum(deltas > 0))
                td_wins = int(np.sum(deltas < 0))
                ties = int(np.sum(deltas == 0))
                pairs = len(deltas)
                mean_delta_grid = float(deltas.mean())

                if pairs > 1 and np.std(deltas) > 1e-9:
                    try:
                        t_stat, p_val = stats.ttest_rel(df_h2h[m_col_e], df_h2h[m_col_td])
                    except Exception:
                        p_val = np.nan

    win_rate_pct = (100.0 * e_wins / pairs) if pairs > 0 else np.nan

    return {
        "env_name": env_name,
        "env_dir": env_dir,
        "status": status,
        "df_e": df_e,
        "df_td": df_td,
        "df_h2h": df_h2h,
        "curves_e": curves_e,
        "curves_td": curves_td,
        "best_e_score": best_e_score,
        "best_e_final_win": best_e_final_win,
        "best_e_auc": best_e_auc,
        "best_e_sem": best_e_sem,
        "best_e_cfg": best_e_cfg,
        "best_td_score": best_td_score,
        "best_td_final_win": best_td_final_win,
        "best_td_auc": best_td_auc,
        "best_td_sem": best_td_sem,
        "best_td_cfg": best_td_cfg,
        "delta_best": delta_best,
        "delta_best_auc": delta_best_auc,
        "delta_best_win": delta_best_win,
        "mean_delta_grid": mean_delta_grid,
        "e_wins": e_wins,
        "td_wins": td_wins,
        "ties": ties,
        "total_pairs": pairs,
        "win_rate_pct": win_rate_pct,
        "p_value": p_val,
        "poster_pdf": poster_pdf if os.path.isfile(poster_pdf) else None,
        "poster_png": poster_png if os.path.isfile(poster_png) else None,
    }


def render_dynamic_env_poster(env_info: dict, rank_by: str = "final_window"):
    """
    Renders a publication-grade 4-panel comparison poster figure for an environment
    if the PNG poster is missing or needs dynamic re-ranking.
    """
    ename = env_info["env_name"]
    df_e = env_info["df_e"]
    df_td = env_info["df_td"]
    curves_e = env_info["curves_e"]
    curves_td = env_info["curves_td"]
    df_h2h = env_info["df_h2h"]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # --------------------------------------------------------------------------
    # Panel 1: Best Learning Curves
    # --------------------------------------------------------------------------
    ax_lc = fig.add_subplot(gs[0, 0])
    has_e = False
    has_td = False

    if curves_e and env_info["best_e_cfg"] in curves_e:
        m_e, s_e, _ = curves_e[env_info["best_e_cfg"]]
        x_e = np.arange(len(m_e))
        ax_lc.plot(x_e, m_e, color="#1b7837", lw=2.2,
                   label=f"Best E ({env_info['best_e_score']:.1f})\n[{env_info['best_e_cfg']}]")
        ax_lc.fill_between(x_e, m_e - s_e, m_e + s_e, color="#1b7837", alpha=0.18)
        has_e = True

    if curves_td and env_info["best_td_cfg"] in curves_td:
        m_td, s_td, _ = curves_td[env_info["best_td_cfg"]]
        x_td = np.arange(len(m_td))
        ax_lc.plot(x_td, m_td, color="#762a83", lw=2.2, linestyle="--",
                   label=f"Best TD ({env_info['best_td_score']:.1f})\n[{env_info['best_td_cfg']}]")
        ax_lc.fill_between(x_td, m_td - s_td, m_td + s_td, color="#762a83", alpha=0.18)
        has_td = True

    ax_lc.set_title(f"A. Learning Curves: Best E vs Best TD ({ename})", fontweight="bold")
    ax_lc.set_xlabel("Updates", fontweight="bold")
    ax_lc.set_ylabel("Episode Return", fontweight="bold")
    if has_e or has_td:
        ax_lc.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
    else:
        ax_lc.text(0.5, 0.5, "No Learning Curves Available", ha="center", va="center")
    ax_lc.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel 2: Critic Epochs Scaling (or LR Scaling)
    # --------------------------------------------------------------------------
    ax_ep = fig.add_subplot(gs[0, 1])
    target_metric = "auc" if rank_by == "auc" else "final_window_mean"

    epoch_col = "critic_epochs"
    if df_e is not None and epoch_col in df_e.columns and df_e[epoch_col].nunique() > 1:
        ep_vals = sorted(df_e[epoch_col].unique())
        mean_e = [df_e[df_e[epoch_col] == ep][target_metric].mean() for ep in ep_vals]
        ax_ep.plot(ep_vals, mean_e, "o-", color="#1b7837", lw=2.0, label="E_experimental", ms=7)
        if df_td is not None and epoch_col in df_td.columns:
            mean_td = [df_td[df_td[epoch_col] == ep][target_metric].mean() for ep in ep_vals if ep in df_td[epoch_col].values]
            ax_ep.plot(ep_vals[:len(mean_td)], mean_td, "s--", color="#762a83", lw=2.0, label="Classic TD", ms=7)
        ax_ep.set_xlabel("Critic Epochs", fontweight="bold")
        ax_ep.set_ylabel(f"Mean {rank_by.upper()}", fontweight="bold")
        ax_ep.set_title("B. Scaling with Critic Epochs", fontweight="bold")
        ax_ep.legend(loc="best", framealpha=0.9)
    else:
        ax_ep.text(0.5, 0.5, "Parameter Sensitivity\n(Single epoch setting)", ha="center", va="center")
        ax_ep.set_title("B. Parameter Sensitivity", fontweight="bold")
    ax_ep.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel 3: Value Heads Scaling (or Weight Decay)
    # --------------------------------------------------------------------------
    ax_hd = fig.add_subplot(gs[1, 0])
    head_col = "num_value_heads"
    if df_e is not None and head_col in df_e.columns and df_e[head_col].nunique() > 1:
        hd_vals = sorted(df_e[head_col].unique())
        mean_e_hd = [df_e[df_e[head_col] == h][target_metric].mean() for h in hd_vals]
        ax_hd.plot(hd_vals, mean_e_hd, "o-", color="#1b7837", lw=2.0, label="E_experimental", ms=7)
        if df_td is not None and head_col in df_td.columns:
            mean_td_hd = [df_td[df_td[head_col] == h][target_metric].mean() for h in hd_vals if h in df_td[head_col].values]
            ax_hd.plot(hd_vals[:len(mean_td_hd)], mean_td_hd, "s--", color="#762a83", lw=2.0, label="Classic TD", ms=7)
        ax_hd.set_xlabel("Value Heads", fontweight="bold")
        ax_hd.set_ylabel(f"Mean {rank_by.upper()}", fontweight="bold")
        ax_hd.set_title("C. Scaling with Value Heads", fontweight="bold")
        ax_hd.legend(loc="best", framealpha=0.9)
    else:
        ax_hd.text(0.5, 0.5, "Value Heads Sensitivity\n(Single head setting)", ha="center", va="center")
        ax_hd.set_title("C. Value Heads Sensitivity", fontweight="bold")
    ax_hd.grid(True, linestyle=":", alpha=0.6)

    # --------------------------------------------------------------------------
    # Panel 4: Matched-Pair Scatter (E vs TD)
    # --------------------------------------------------------------------------
    ax_sc = fig.add_subplot(gs[1, 1])
    if df_h2h is not None and not df_h2h.empty and env_info["total_pairs"] > 0:
        m_e = f"{target_metric}_e" if f"{target_metric}_e" in df_h2h.columns else "final_window_mean_e"
        m_td = f"{target_metric}_td" if f"{target_metric}_td" in df_h2h.columns else "final_window_mean_td"
        y_e = df_h2h[m_e].values
        x_td = df_h2h[m_td].values

        mn = min(y_e.min(), x_td.min()) - 1.0
        mx = max(y_e.max(), x_td.max()) + 1.0

        ax_sc.plot([mn, mx], [mn, mx], "k--", alpha=0.6, label="y = x (Parity)")
        ax_sc.scatter(x_td, y_e, c="#1a73e8", edgecolors="#174ea6", s=50, alpha=0.85, zorder=4)

        diffs = y_e - x_td
        p_val_str = f"{env_info['p_value']:.4f}" if not np.isnan(env_info["p_value"]) else "N/A"
        win_text = (f"E Wins: {env_info['e_wins']}/{env_info['total_pairs']} ({env_info['win_rate_pct']:.1f}%)\n"
                    f"TD Wins: {env_info['td_wins']}/{env_info['total_pairs']}\n"
                    f"Ties: {env_info['ties']}\n"
                    f"Mean Δ(E - TD) = {diffs.mean():+.2f}\n"
                    f"p = {p_val_str}")
        ax_sc.text(0.05, 0.95, win_text, transform=ax_sc.transAxes, va="top", ha="left",
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dadce0"))
        ax_sc.set_title(f"D. Matched-Pair Scatter ({rank_by.upper()})", fontweight="bold")
        ax_sc.set_xlabel(f"Classic TD {rank_by.upper()}", fontweight="bold")
        ax_sc.set_ylabel(f"E_experimental {rank_by.upper()}", fontweight="bold")
        ax_sc.set_xlim(mn, mx)
        ax_sc.set_ylim(mn, mx)
        ax_sc.legend(loc="lower right", framealpha=0.92)
    else:
        ax_sc.text(0.5, 0.5, "Matched Comparison Pending\n(Requires both E and TD completed)", ha="center", va="center")
        ax_sc.set_title("D. Matched-Pair Grid Scatter", fontweight="bold")
    ax_sc.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"Head-to-Head Detailed Analysis: {ename} (Ranked by {rank_by.upper()})",
                 fontsize=15, fontweight="bold", y=0.985)
    return fig


def main():
    args = parse_args()
    raw_target = args.suite_dir or args.sweep_target
    if not raw_target:
        print("Error: Please provide a suite directory path or Job ID (e.g. cmp_td_vs_e_12724069 or 12724069).")
        sys.exit(1)

    suite_dir = resolve_suite_directory(raw_target)
    suite_name = os.path.basename(suite_dir)

    rank_by = args.rank_by
    window_size = args.window_size
    metric_label = "AUC" if rank_by == "auc" else f"Final Window (last {window_size} updates)" if rank_by == "final_window" else "Max Return"

    print("=" * 80)
    print(f"COMPILING MASTER TD vs E REPORT")
    print(f"  Suite Directory: {suite_dir}")
    print(f"  Ranking Metric:  {rank_by.upper()} ({metric_label})")
    print(f"  Window Size:     {window_size} updates")
    print("=" * 80)

    # Discover environment directories
    env_dirs = [
        os.path.join(suite_dir, d) for d in sorted(os.listdir(suite_dir))
        if os.path.isdir(os.path.join(suite_dir, d)) and not d.startswith(".") and (
            os.path.isfile(os.path.join(suite_dir, d, "head_to_head_summary.csv")) or
            os.path.isdir(os.path.join(suite_dir, d, "E_experimental")) or
            os.path.isdir(os.path.join(suite_dir, d, "td")) or
            any(tag in d.lower() for tag in ["minatar", "cartpole", "pendulum", "mountaincar", "acrobot", "hopper", "walker", "cheetah", "ant"])
        )
    ]

    if not env_dirs:
        print(f"Error: No environment subdirectories found in '{suite_dir}'.")
        sys.exit(1)

    print(f"Discovered {len(env_dirs)} environments in suite:")
    for ed in env_dirs:
        print(f"  - {os.path.basename(ed)}")

    # Load results
    env_data_list = [load_env_results(ed, rank_by=rank_by, window_size=window_size, preferred_metric=args.metric)
                     for ed in env_dirs]

    # =========================================================================
    # 1. COMPILE MASTER SUMMARY METRICS
    # =========================================================================
    summary_rows = []
    total_e_wins = 0
    total_td_wins = 0
    total_ties = 0
    total_pairs = 0
    complete_envs = 0

    for d in env_data_list:
        ename = d["env_name"]
        status = d["status"]
        if status == "Complete":
            complete_envs += 1

        total_e_wins += d["e_wins"]
        total_td_wins += d["td_wins"]
        total_ties += d["ties"]
        total_pairs += d["total_pairs"]

        summary_rows.append({
            "environment": ename,
            "status": status,
            "rank_metric": rank_by,
            "best_e_score": d["best_e_score"],
            "best_e_sem": d["best_e_sem"],
            "best_e_final_win": d["best_e_final_win"],
            "best_e_auc": d["best_e_auc"],
            "best_e_config": d["best_e_cfg"],
            "best_td_score": d["best_td_score"],
            "best_td_sem": d["best_td_sem"],
            "best_td_final_win": d["best_td_final_win"],
            "best_td_auc": d["best_td_auc"],
            "best_td_config": d["best_td_cfg"],
            "delta_best": d["delta_best"],
            "delta_best_auc": d["delta_best_auc"],
            "delta_best_final_win": d["delta_best_win"],
            "mean_delta_grid": d["mean_delta_grid"],
            "e_wins": d["e_wins"],
            "td_wins": d["td_wins"],
            "ties": d["ties"],
            "total_pairs": d["total_pairs"],
            "win_rate_pct": d["win_rate_pct"],
            "p_value": d["p_value"],
            "significant_05": (d["p_value"] < 0.05) if not np.isnan(d["p_value"]) else False,
        })

    df_master = pd.DataFrame(summary_rows)
    master_csv = args.output_csv if args.output_csv else os.path.join(suite_dir, "master_summary_table.csv")
    df_master.to_csv(master_csv, index=False)

    global_win_rate = (100.0 * total_e_wins / total_pairs) if total_pairs > 0 else 0.0

    print("\n" + "=" * 90)
    print(f"MASTER COMPARISON SUMMARY: CLASSIC TD vs E_EXPERIMENTAL ({suite_name})")
    print(f"Ranking Metric: {rank_by.upper()} | Window: {window_size} updates")
    print(f"Environments:   {len(df_master)} total ({complete_envs} complete, {len(df_master) - complete_envs} in progress / partial)")
    print(f"Global Wins:    E={total_e_wins} | TD={total_td_wins} | Ties={total_ties} across {total_pairs} matched pairs ({global_win_rate:.1f}% E Win Rate)")
    print("=" * 90)
    print_cols = ["environment", "status", "best_e_score", "best_td_score", "delta_best", "win_rate_pct", "p_value"]
    print(df_master[print_cols].to_string(index=False))
    print("=" * 90)
    print(f"Saved master summary CSV to:\n  {master_csv}")

    # =========================================================================
    # 2. GENERATE MASTER MULTI-PAGE PDF REPORT
    # =========================================================================
    out_pdf = args.output_pdf if args.output_pdf else os.path.join(suite_dir, "master_td_vs_e_report.pdf")
    print(f"\nGenerating master PDF report: {out_pdf} ...")

    with PdfPages(out_pdf) as pdf:
        # ---------------------------------------------------------------------
        # Page 1: Executive Dashboard & Performance Delta
        # ---------------------------------------------------------------------
        fig = plt.figure(figsize=(16, 11))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        # Panel (0, 0): Win Rate Breakdown Donut / Pie
        ax_pie = fig.add_subplot(gs[0, 0])
        if total_pairs > 0 and (total_e_wins + total_td_wins + total_ties) > 0:
            pie_slices = []
            pie_labels = []
            pie_colors = []
            if total_e_wins > 0:
                pie_slices.append(total_e_wins)
                pie_labels.append(f"E_experimental ({total_e_wins})")
                pie_colors.append("#1b7837")
            if total_td_wins > 0:
                pie_slices.append(total_td_wins)
                pie_labels.append(f"Classic TD ({total_td_wins})")
                pie_colors.append("#762a83")
            if total_ties > 0:
                pie_slices.append(total_ties)
                pie_labels.append(f"Ties ({total_ties})")
                pie_colors.append("#9e9e9e")

            ax_pie.pie(pie_slices, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%",
                       startangle=140, textprops={"fontsize": 11, "fontweight": "bold"},
                       wedgeprops={"edgecolor": "white", "linewidth": 1.5})
            ax_pie.set_title(f"Global Head-to-Head Win Rate ({total_pairs} Matched Pairs)\nRanked by {rank_by.upper()}",
                             fontweight="bold")
        else:
            ax_pie.text(0.5, 0.5, "No Matched Pairs Completed Yet\n(Runs may still be in progress)",
                        ha="center", va="center", fontsize=11, fontweight="bold", color="#757575")
            ax_pie.set_title(f"Global Head-to-Head Win Rate (Pending)", fontweight="bold")
            ax_pie.axis("off")

        # Panel (0, 1): Per-Environment Win Rate Bar Chart
        ax_win = fig.add_subplot(gs[0, 1])
        x_idx = np.arange(len(df_master))
        win_rates = df_master["win_rate_pct"].values
        bar_colors = ["#1b7837" if (not np.isnan(w) and w >= 50.0) else ("#762a83" if not np.isnan(w) else "#bdbdbd")
                      for w in win_rates]
        plot_wins = [w if not np.isnan(w) else 0.0 for w in win_rates]
        bars_w = ax_win.bar(x_idx, plot_wins, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.8)
        ax_win.axhline(50.0, color="gray", linestyle="--", lw=1.2, label="Parity (50%)")
        ax_win.set_xticks(x_idx)
        ax_win.set_xticklabels(df_master["environment"], rotation=30, ha="right", fontsize=8.5, fontweight="bold")
        ax_win.set_ylabel(f"E Win Rate (%) [{rank_by.upper()}]", fontweight="bold")
        ax_win.set_ylim(0, 105)
        ax_win.set_title(f"E_experimental Win Rate by Environment ({rank_by.upper()})", fontweight="bold")
        ax_win.grid(axis="y", linestyle=":", alpha=0.6)
        ax_win.legend(loc="upper right")

        for idx_b, (bar, w) in enumerate(zip(bars_w, win_rates)):
            if np.isnan(w):
                status_short = df_master.iloc[idx_b]["status"]
                lbl = "E Only" if "E Only" in status_short else ("TD Only" if "TD Only" in status_short else "Pending")
                ax_win.text(bar.get_x() + bar.get_width() / 2, 8, lbl,
                            ha="center", va="bottom", fontsize=7.5, color="#616161", rotation=90, fontweight="bold")
            else:
                ax_win.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{w:.1f}%",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Panel (1, :): Performance Delta (Best E - Best TD) Bar Chart
        ax_bar = fig.add_subplot(gs[1, :])
        deltas = df_master["delta_best"].values
        d_colors = ["#1b7837" if (not np.isnan(d) and d >= 0) else ("#762a83" if not np.isnan(d) else "#bdbdbd")
                    for d in deltas]
        plot_deltas = [d if not np.isnan(d) else 0.0 for d in deltas]
        bars_d = ax_bar.bar(x_idx, plot_deltas, color=d_colors, alpha=0.85, edgecolor="black", lw=0.8)
        ax_bar.axhline(0.0, color="black", lw=1.2)
        ax_bar.set_xticks(x_idx)
        ax_bar.set_xticklabels(df_master["environment"], rotation=25, ha="right", fontsize=9, fontweight="bold")
        ax_bar.set_ylabel(f"Delta: Best E - Best TD ({rank_by.upper()})", fontweight="bold")
        ax_bar.set_title(f"Performance Advantage: Best E_experimental vs. Best Classic TD ({metric_label})",
                         fontweight="bold")
        ax_bar.grid(axis="y", linestyle=":", alpha=0.6)

        for idx_b, (bar, d) in enumerate(zip(bars_d, deltas)):
            if np.isnan(d):
                status_short = df_master.iloc[idx_b]["status"]
                lbl = "E Only" if "E Only" in status_short else ("TD Only" if "TD Only" in status_short else "Pending")
                ax_bar.text(bar.get_x() + bar.get_width() / 2, 0, f"[{lbl}]",
                            ha="center", va="bottom", fontsize=8, color="#757575", fontweight="bold")
            else:
                h = bar.get_height()
                offset = 4 if h >= 0 else -12
                ax_bar.annotate(f"{h:+.1f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, offset), textcoords="offset points",
                                ha="center", va="bottom" if h >= 0 else "top",
                                fontsize=8.5, fontweight="bold")

        fig.suptitle(f"Master Head-to-Head Report: Classic TD vs E_experimental\nSuite: {suite_name} | Ranking Metric: {rank_by.upper()} ({metric_label})",
                     fontsize=15, fontweight="bold", y=0.985)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---------------------------------------------------------------------
        # Page 2: Cross-Environment Comparison Grid (Learning Curves)
        # ---------------------------------------------------------------------
        n_envs = len(env_data_list)
        n_cols = 4
        n_rows = math.ceil(n_envs / n_cols)
        fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
        plt.subplots_adjust(hspace=0.42, wspace=0.28)

        for idx, d in enumerate(env_data_list):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes_grid[r, c]
            ename = d["env_name"]
            c_e = d["curves_e"]
            c_td = d["curves_td"]
            best_e_lbl = d["best_e_cfg"]
            best_td_lbl = d["best_td_cfg"]
            status = d["status"]

            has_e_curve = False
            has_td_curve = False

            if c_e and best_e_lbl in c_e:
                m_e, s_e, _ = c_e[best_e_lbl]
                x_e = np.arange(len(m_e))
                ax.plot(x_e, m_e, color="#1b7837", lw=1.8, label=f"Best E ({d['best_e_score']:.1f})")
                ax.fill_between(x_e, m_e - s_e, m_e + s_e, color="#1b7837", alpha=0.18)
                has_e_curve = True

            if c_td and best_td_lbl in c_td:
                m_td, s_td, _ = c_td[best_td_lbl]
                x_td = np.arange(len(m_td))
                ax.plot(x_td, m_td, color="#762a83", lw=1.8, linestyle="--", label=f"Best TD ({d['best_td_score']:.1f})")
                ax.fill_between(x_td, m_td - s_td, m_td + s_td, color="#762a83", alpha=0.18)
                has_td_curve = True

            # Titles and status annotations
            if has_e_curve and has_td_curve:
                delta_str = f"Δ={d['delta_best']:+.1f}" if not np.isnan(d['delta_best']) else ""
                ax.set_title(f"{ename}\n({delta_str})", fontweight="bold", fontsize=9.5)
            elif has_e_curve and not has_td_curve:
                ax.set_title(f"{ename}\n[Classic TD: In Progress]", fontweight="bold", fontsize=9, color="#b2182b")
            elif not has_e_curve and has_td_curve:
                ax.set_title(f"{ename}\n[E_exp: In Progress]", fontweight="bold", fontsize=9, color="#b2182b")
            else:
                ax.set_title(f"{ename}\n[In Progress / Pending]", fontweight="bold", fontsize=9, color="#757575")
                ax.text(0.5, 0.5, "Data In Progress", ha="center", va="center", color="#757575", fontsize=9)

            ax.set_xlabel("Updates", fontsize=8)
            ax.set_ylabel("Return", fontsize=8)
            if has_e_curve or has_td_curve:
                ax.legend(loc="lower right", fontsize=7.2, framealpha=0.9)
            ax.grid(True, linestyle=":", alpha=0.5)

        for i in range(n_envs, n_rows * n_cols):
            r = i // n_cols
            c = i % n_cols
            axes_grid[r, c].axis("off")

        fig_grid.suptitle(f"Cross-Environment Learning Curves: Best E vs. Best Classic TD (Ranked by {rank_by.upper()})",
                          fontsize=15, fontweight="bold", y=0.995)
        pdf.savefig(fig_grid, bbox_inches="tight")
        plt.close(fig_grid)

        # ---------------------------------------------------------------------
        # Pages 3+: Per-Environment Detailed Posters
        # ---------------------------------------------------------------------
        print("Embedding per-environment comparison posters...")
        for idx, d in enumerate(env_data_list, 1):
            ename = d["env_name"]
            png_path = d["poster_png"]

            # If poster PNG already exists and ranking matches default final_window, embed it
            if png_path and os.path.isfile(png_path) and rank_by == "final_window":
                try:
                    img = plt.imread(png_path)
                    fig_p, ax_p = plt.subplots(figsize=(16, 11))
                    ax_p.imshow(img)
                    ax_p.axis("off")
                    fig_p.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    pdf.savefig(fig_p, bbox_inches="tight", dpi=300)
                    plt.close(fig_p)
                    print(f"  [{idx:02d}/{len(env_data_list)}] Embedded disk poster for {ename}")
                    continue
                except Exception as ex:
                    print(f"  Warning: Failed to embed PNG poster for {ename}: {ex}")

            # Otherwise, render dynamically to respect rank_by (e.g. AUC) and ensure 100% completion
            try:
                fig_dyn = render_dynamic_env_poster(d, rank_by=rank_by)
                pdf.savefig(fig_dyn, bbox_inches="tight")
                plt.close(fig_dyn)
                print(f"  [{idx:02d}/{len(env_data_list)}] Dynamically generated poster for {ename} ({rank_by.upper()})")
            except Exception as ex:
                print(f"  Warning: Failed to render dynamic poster for {ename}: {ex}")

    print("\n" + "=" * 80)
    print("MASTER REPORT COMPILATION COMPLETE:")
    print(f"  - Master PDF:   {out_pdf}")
    print(f"  - Summary CSV:  {master_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

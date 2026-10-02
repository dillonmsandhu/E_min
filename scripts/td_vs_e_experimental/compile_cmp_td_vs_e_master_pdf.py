#!/usr/bin/env python3
"""
scripts/compile_cmp_td_vs_e_master_pdf.py

Compiles all environment runs from a Comprehensive Critic comparison sweep:
    - E(0) vs. TD(0)  (1-step comparison)
    - E(lambda) vs. TD(lambda)  (multi-step / eligibility trace comparison)
or backwards-compatible 2-way comparison sweeps (e.g., E_experimental vs td)
into a comprehensive, publication-grade master report:

Outputs:
1. <suite_dir>/suite_auc_summary.md: Markdown table of AUC for each task + dual pairwise comparisons
2. <suite_dir>/master_summary_table.csv: Complete suite-wide comparison metrics, win rates, and deltas
3. <suite_dir>/master_td_vs_e_report.pdf: Multi-page publication-grade vector PDF:
   - Page 1: Executive Dashboard (Summary stats, pairwise win rates, performance delta charts)
   - Page 2: Cross-Environment Comparison Grid (Best learning curves across tasks)
   - Pages 3+: Per-environment detailed 4-panel comparison posters

Usage:
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py --suite-dir results/ppo/sweeps/cmp_td_vs_e_12724069
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py cmp_td_vs_e_12724069 --rank-by auc
    python scripts/td_vs_e_experimental/compile_cmp_td_vs_e_master_pdf.py 12724069 --window-size 100
"""

import os
import sys
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

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Vector publication settings
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10

ALGO_PALETTE = {
    "E_0": "#1b7837",        # Dark Forest Green
    "TD_0": "#762a83",       # Dark Purple
    "E_lambda": "#1a73e8",   # Vibrant Blue
    "TD_lambda": "#d95f02",  # Vivid Orange
    "E_experimental": "#1b7837",
    "td": "#762a83",
}

ALGO_PRETTY_NAMES = {
    "E_0": "E(0)",
    "TD_0": "TD(0)",
    "E_lambda": "E(lambda)",
    "TD_lambda": "TD(lambda)",
    "E_experimental": "E(0)",
    "td": "TD(0)",
}


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
        description="Compile Critic Comparison sweep into master Markdown tables and PDF report"
    )
    parser.add_argument("sweep_target", nargs="?", default=None,
                        help="Suite path, suite directory name, or SLURM Job ID (e.g., cmp_td_vs_e_12724069 or 12724069)")
    parser.add_argument("--suite-dir", type=str, default=None,
                        help="Explicit path to suite directory")
    parser.add_argument("--output-pdf", type=str, default=None,
                        help="Custom path for generated master PDF (default: <suite_dir>/master_td_vs_e_report.pdf)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Custom path for generated summary CSV (default: <suite_dir>/master_summary_table.csv)")
    parser.add_argument("--output-md", type=str, default=None,
                        help="Custom path for generated markdown summary (default: <suite_dir>/suite_auc_summary.md)")
    parser.add_argument("--rank-by", type=str, default="auc",
                        choices=["final_window", "auc", "max"],
                        help="Metric to rank best configuration and head-to-head win rate: 'final_window', 'auc', or 'max' (default: auc)")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size in updates for calculating final evaluation return (default: 100)")
    parser.add_argument("--metric", type=str, default=None,
                        help="Name of metric array inside metrics.pkl (default: auto-detect episode returns)")
    return parser.parse_args()


def extract_curves_and_metrics(pkl_path: str, preferred_metric: str = None):
    """
    Extracts {config_label: (mean_curve, sem_curve, raw_array)} from metrics.pkl.
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
    """
    Finds all available algorithm subdirectories inside an environment directory:
    E_0, TD_0, E_lambda, TD_lambda, plus backwards-compatible names.
    """
    algo_candidates = {
        "E_0": ["E_0", "E_experimental", "e_0", "e0", "E"],
        "TD_0": ["TD_0", "td_0", "td0", "td", "TD"],
        "E_lambda": ["E_lambda", "E_lambda_experimental", "e_lambda", "elambda", "e_lambda_geometric"],
        "TD_lambda": ["TD_lambda", "td_lambda", "tdlambda", "ppo_fitted", "baseline_ppo", "fitted"],
    }

    found_dirs = {}
    used_paths = set()

    for algo_key, cands in algo_candidates.items():
        for cand in cands:
            p = os.path.join(env_dir, cand)
            if os.path.isdir(p) and p not in used_paths:
                found_dirs[algo_key] = p
                used_paths.add(p)
                break

    # Fallback scan for any subdirectories with summary_*.csv
    for sub in sorted(os.listdir(env_dir)):
        sub_p = os.path.join(env_dir, sub)
        if not os.path.isdir(sub_p) or sub_p in used_paths:
            continue
        sub_l = sub.lower()
        if "e_lambda" in sub_l and "E_lambda" not in found_dirs:
            found_dirs["E_lambda"] = sub_p
            used_paths.add(sub_p)
        elif "td_lambda" in sub_l and "TD_lambda" not in found_dirs:
            found_dirs["TD_lambda"] = sub_p
            used_paths.add(sub_p)
        elif any(k in sub_l for k in ["e_0", "e0", "e_exp"]) and "E_0" not in found_dirs:
            found_dirs["E_0"] = sub_p
            used_paths.add(sub_p)
        elif any(k in sub_l for k in ["td_0", "td0", "td"]) and "TD_0" not in found_dirs:
            found_dirs["TD_0"] = sub_p
            used_paths.add(sub_p)

    return found_dirs


def load_algo_data(algo_dir: str, window_size: int = 100, preferred_metric: str = None):
    """Loads CSV and pickles for an algorithm and computes window/AUC/max metrics."""
    if not algo_dir or not os.path.isdir(algo_dir):
        return None, {}

    summary_files = glob.glob(os.path.join(algo_dir, "summary_*.csv"))
    df = None
    if summary_files:
        try:
            df = pd.read_csv(summary_files[0])
        except Exception:
            df = None

    pkl_path = os.path.join(algo_dir, "metrics.pkl")
    curves = extract_curves_and_metrics(pkl_path, preferred_metric)

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
        win_means, win_sems, aucs, max_rets = [], [], [], []
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


def load_env_results(env_dir: str, rank_by: str = "auc", window_size: int = 100, preferred_metric: str = None):
    """Loads all algorithm results and computes pairwise matchups for a single environment."""
    env_name = os.path.basename(env_dir)
    poster_pdf = os.path.join(env_dir, "head_to_head_poster.pdf")
    poster_png = os.path.join(env_dir, "head_to_head_poster.png")

    algo_dirs = detect_algo_subdirs(env_dir)
    algo_data = {}

    target_metric_col = "auc" if rank_by == "auc" else ("final_window_mean" if rank_by == "final_window" else "max_return")

    for k in ["E_0", "TD_0", "E_lambda", "TD_lambda"]:
        if k in algo_dirs:
            df, curves = load_algo_data(algo_dirs[k], window_size, preferred_metric)
            if df is not None and not df.empty:
                sort_col = target_metric_col if target_metric_col in df.columns else "auc"
                best_row = df.sort_values(sort_col, ascending=False).iloc[0]
                algo_data[k] = {
                    "dir": algo_dirs[k],
                    "df": df,
                    "curves": curves,
                    "best_score": float(best_row[sort_col]),
                    "best_auc": float(best_row.get("auc", np.nan)),
                    "best_final": float(best_row.get("final_window_mean", np.nan)),
                    "best_sem": float(best_row.get("final_window_sem", 0.0)),
                    "best_config": str(best_row["config"]),
                    "mean_auc": float(df["auc"].mean()),
                    "mean_final": float(df["final_window_mean"].mean()),
                }

    # Matchup 1: E(0) vs TD(0)
    match_0 = {"valid": False, "delta_auc": np.nan, "delta_final": np.nan, "winner": "N/A", "p_val": np.nan}
    if "E_0" in algo_data and "TD_0" in algo_data:
        df_e0 = algo_data["E_0"]["df"]
        df_td0 = algo_data["TD_0"]["df"]
        cols = [c for c in ["critic_lr", "critic_epochs", "weight_decay", "num_value_heads"] if c in df_e0.columns and c in df_td0.columns]
        if not cols and "config" in df_e0.columns and "config" in df_td0.columns:
            cols = ["config"]
        if cols:
            merged0 = pd.merge(df_e0, df_td0, on=cols, suffixes=("_e", "_td"))
            if not merged0.empty:
                d_auc = merged0["auc_e"] - merged0["auc_td"]
                d_fin = merged0["final_window_mean_e"] - merged0["final_window_mean_td"]
                _, p = stats.ttest_rel(merged0["auc_e"], merged0["auc_td"]) if len(merged0) > 1 else (0.0, 1.0)
                mean_d = float(d_auc.mean())
                match_0 = {
                    "valid": True,
                    "n_pairs": len(merged0),
                    "e_wins_auc": int(np.sum(d_auc > 0)),
                    "td_wins_auc": int(np.sum(d_auc < 0)),
                    "ties_auc": int(np.sum(d_auc == 0)),
                    "delta_auc": mean_d,
                    "delta_final": float(d_fin.mean()),
                    "winner": "E(0)" if mean_d > 0 else ("TD(0)" if mean_d < 0 else "Tie"),
                    "p_val": float(p) if not np.isnan(p) else 1.0,
                }

    # Matchup 2: E(lambda) vs TD(lambda)
    match_l = {"valid": False, "delta_auc": np.nan, "delta_final": np.nan, "winner": "N/A", "p_val": np.nan}
    if "E_lambda" in algo_data and "TD_lambda" in algo_data:
        df_el = algo_data["E_lambda"]["df"]
        df_tdl = algo_data["TD_lambda"]["df"]
        cols = [c for c in ["critic_lr", "critic_epochs", "weight_decay", "num_value_heads"] if c in df_el.columns and c in df_tdl.columns]
        if not cols and "config" in df_el.columns and "config" in df_tdl.columns:
            cols = ["config"]
        if cols:
            mergedl = pd.merge(df_el, df_tdl, on=cols, suffixes=("_e", "_td"))
            if not mergedl.empty:
                d_auc = mergedl["auc_e"] - mergedl["auc_td"]
                d_fin = mergedl["final_window_mean_e"] - mergedl["final_window_mean_td"]
                _, p = stats.ttest_rel(mergedl["auc_e"], mergedl["auc_td"]) if len(mergedl) > 1 else (0.0, 1.0)
                mean_d = float(d_auc.mean())
                match_l = {
                    "valid": True,
                    "n_pairs": len(mergedl),
                    "e_wins_auc": int(np.sum(d_auc > 0)),
                    "td_wins_auc": int(np.sum(d_auc < 0)),
                    "ties_auc": int(np.sum(d_auc == 0)),
                    "delta_auc": mean_d,
                    "delta_final": float(d_fin.mean()),
                    "winner": "E(λ)" if mean_d > 0 else ("TD(λ)" if mean_d < 0 else "Tie"),
                    "p_val": float(p) if not np.isnan(p) else 1.0,
                }

    # Overall Best algorithm for this task
    best_overall_algo = "N/A"
    best_overall_auc = -np.inf
    for k, d in algo_data.items():
        if not np.isnan(d["best_auc"]) and d["best_auc"] > best_overall_auc:
            best_overall_auc = d["best_auc"]
            best_overall_algo = ALGO_PRETTY_NAMES.get(k, k)

    return {
        "env_name": env_name,
        "env_dir": env_dir,
        "algo_data": algo_data,
        "match_0": match_0,
        "match_lambda": match_l,
        "best_overall_algo": best_overall_algo,
        "best_overall_auc": best_overall_auc,
        "poster_pdf": poster_pdf if os.path.isfile(poster_pdf) else None,
        "poster_png": poster_png if os.path.isfile(poster_png) else None,
    }


def generate_suite_markdown_table(env_results_list, rank_by="auc"):
    """
    Constructs a comprehensive Markdown Table of AUC across all tasks in the suite,
    plus pairwise comparisons for E(0) vs TD(0) and E(lambda) vs TD(lambda).
    """
    md = []
    md.append("# Multi-Task Critic Benchmark Suite: AUC Comparison Table\n")
    md.append(f"Ranked by: **{rank_by.upper()}** across all swept learning hyperparameters.\n")
    md.append("| Task / Environment | E(0) AUC | TD(0) AUC | Δ AUC (E0 - TD0) | Winner (0) | E(λ) AUC | TD(λ) AUC | Δ AUC (Eλ - TDλ) | Winner (λ) | Best Overall |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    e0_aucs, td0_aucs, el_aucs, tdl_aucs = [], [], [], []
    e0_wins = td0_wins = el_wins = tdl_wins = 0

    for d in env_results_list:
        ename = d["env_name"]
        adata = d["algo_data"]
        m0 = d["match_0"]
        ml = d["match_lambda"]

        e0_str = f"{adata['E_0']['best_auc']:.1f}" if "E_0" in adata else "—"
        td0_str = f"{adata['TD_0']['best_auc']:.1f}" if "TD_0" in adata else "—"
        if "E_0" in adata and not np.isnan(adata['E_0']['best_auc']):
            e0_aucs.append(adata['E_0']['best_auc'])
        if "TD_0" in adata and not np.isnan(adata['TD_0']['best_auc']):
            td0_aucs.append(adata['TD_0']['best_auc'])

        if m0["valid"]:
            d0_str = f"{m0['delta_auc']:+.2f}"
            w0_str = f"**{m0['winner']}**"
            if m0["winner"] == "E(0)":
                e0_wins += 1
            elif m0["winner"] == "TD(0)":
                td0_wins += 1
        else:
            d0_str = "—"
            w0_str = "—"

        el_str = f"{adata['E_lambda']['best_auc']:.1f}" if "E_lambda" in adata else "—"
        tdl_str = f"{adata['TD_lambda']['best_auc']:.1f}" if "TD_lambda" in adata else "—"
        if "E_lambda" in adata and not np.isnan(adata['E_lambda']['best_auc']):
            el_aucs.append(adata['E_lambda']['best_auc'])
        if "TD_lambda" in adata and not np.isnan(adata['TD_lambda']['best_auc']):
            tdl_aucs.append(adata['TD_lambda']['best_auc'])

        if ml["valid"]:
            dl_str = f"{ml['delta_auc']:+.2f}"
            wl_str = f"**{ml['winner']}**"
            if "E" in ml["winner"]:
                el_wins += 1
            elif "TD" in ml["winner"]:
                tdl_wins += 1
        else:
            dl_str = "—"
            wl_str = "—"

        b_overall = f"**{d['best_overall_algo']}**" if d["best_overall_algo"] != "N/A" else "—"

        md.append(f"| `{ename}` | {e0_str} | {td0_str} | {d0_str} | {w0_str} | {el_str} | {tdl_str} | {dl_str} | {wl_str} | {b_overall} |")

    # Summary row
    mean_e0_str = f"{np.mean(e0_aucs):.1f}" if e0_aucs else "—"
    mean_td0_str = f"{np.mean(td0_aucs):.1f}" if td0_aucs else "—"
    mean_el_str = f"{np.mean(el_aucs):.1f}" if el_aucs else "—"
    mean_tdl_str = f"{np.mean(tdl_aucs):.1f}" if tdl_aucs else "—"

    md.append(f"| **SUITE MEAN / WINS** | **{mean_e0_str}** | **{mean_td0_str}** | **{e0_wins}W - {td0_wins}L** | — | **{mean_el_str}** | **{mean_tdl_str}** | **{el_wins}W - {tdl_wins}L** | — | — |")

    md.append("\n---\n")
    md.append("## Pairwise Statistical Summary Across Suite Tasks\n")

    # 1. 1-step Matchup
    valid_m0 = [d["match_0"] for d in env_results_list if d["match_0"]["valid"]]
    if valid_m0:
        d0_vals = [m["delta_auc"] for m in valid_m0]
        _, p0 = stats.ttest_1samp(d0_vals, 0.0) if len(d0_vals) > 1 else (0.0, 1.0)
        md.append("### 1. E(0) vs. TD(0) (1-step Methods)")
        md.append(f"- **Tasks Evaluated:** {len(valid_m0)}")
        md.append(f"- **Task Win Counts:** **E(0)** won {e0_wins} tasks | **TD(0)** won {td0_wins} tasks")
        md.append(f"- **Suite-wide Mean Δ AUC (E(0) - TD(0)):** **`{np.mean(d0_vals):+.2f}`** ± {np.std(d0_vals) / np.sqrt(len(d0_vals)):.2f} (*p* = {p0:.4e})")
        w0 = "E(0)" if np.mean(d0_vals) > 0 else ("TD(0)" if np.mean(d0_vals) < 0 else "Tie")
        md.append(f"- **Suite 1-Step Champion:** **{w0}**\n")

    # 2. Multi-step Matchup
    valid_ml = [d["match_lambda"] for d in env_results_list if d["match_lambda"]["valid"]]
    if valid_ml:
        dl_vals = [m["delta_auc"] for m in valid_ml]
        _, pl = stats.ttest_1samp(dl_vals, 0.0) if len(dl_vals) > 1 else (0.0, 1.0)
        md.append("### 2. E(λ) vs. TD(λ) (λ = 0.9 Eligibility Trace Methods)")
        md.append(f"- **Tasks Evaluated:** {len(valid_ml)}")
        md.append(f"- **Task Win Counts:** **E(λ)** won {el_wins} tasks | **TD(λ)** won {tdl_wins} tasks")
        md.append(f"- **Suite-wide Mean Δ AUC (E(λ) - TD(λ)):** **`{np.mean(dl_vals):+.2f}`** ± {np.std(dl_vals) / np.sqrt(len(dl_vals)):.2f} (*p* = {pl:.4e})")
        wl = "E(λ)" if np.mean(dl_vals) > 0 else ("TD(λ)" if np.mean(dl_vals) < 0 else "Tie")
        md.append(f"- **Suite λ-Trace Champion:** **{wl}**\n")

    return "\n".join(md)


def main():
    args = parse_args()
    raw_target = args.suite_dir or args.sweep_target
    if not raw_target:
        print("Error: Please provide a suite directory path or Job ID.")
        sys.exit(1)

    suite_dir = resolve_suite_directory(raw_target)
    suite_name = os.path.basename(suite_dir)
    rank_by = args.rank_by
    window_size = args.window_size

    print("=" * 85)
    print(f"COMPILING MASTER 4-WAY CRITIC REPORT: {suite_name}")
    print(f"  Suite Directory: {suite_dir}")
    print(f"  Ranking Metric:  {rank_by.upper()}")
    print("=" * 85)

    # Check if suite_dir itself is a single environment run directory
    is_self_env = (
        os.path.isfile(os.path.join(suite_dir, "auc_summary.md")) or
        os.path.isfile(os.path.join(suite_dir, "head_to_head_summary.csv")) or
        any(os.path.isdir(os.path.join(suite_dir, sub)) for sub in ["E_0", "TD_0", "E_lambda", "TD_lambda", "E_experimental", "td"])
    )

    if is_self_env:
        env_dirs = [suite_dir]
    else:
        # Discover environment directories within suite_dir
        env_dirs = [
            os.path.join(suite_dir, d) for d in sorted(os.listdir(suite_dir))
            if os.path.isdir(os.path.join(suite_dir, d)) and not d.startswith(".") and (
                os.path.isfile(os.path.join(suite_dir, d, "auc_summary.md")) or
                os.path.isfile(os.path.join(suite_dir, d, "head_to_head_summary.csv")) or
                os.path.isdir(os.path.join(suite_dir, d, "E_0")) or
                os.path.isdir(os.path.join(suite_dir, d, "E_experimental")) or
                os.path.isdir(os.path.join(suite_dir, d, "td")) or
                os.path.isdir(os.path.join(suite_dir, d, "TD_0"))
            )
        ]

    if not env_dirs:
        print(f"Error: No environment subdirectories found in '{suite_dir}'.")
        sys.exit(1)

    print(f"Discovered {len(env_dirs)} environments in suite:")
    for ed in env_dirs:
        print(f"  - {os.path.basename(ed)}")

    env_data_list = [load_env_results(ed, rank_by=rank_by, window_size=window_size, preferred_metric=args.metric)
                     for ed in env_dirs]

    # Generate Markdown Summary
    suite_md = generate_suite_markdown_table(env_data_list, rank_by=rank_by)
    md_path = args.output_md or os.path.join(suite_dir, "suite_auc_summary.md")
    with open(md_path, "w") as f:
        f.write(suite_md)

    print("\n" + "=" * 85)
    print("SUITE-WIDE AUC COMPARISON TABLE")
    print("=" * 85)
    print(suite_md)
    print("=" * 85)

    # Compile CSV table
    csv_rows = []
    for d in env_data_list:
        row = {"environment": d["env_name"], "best_overall_algo": d["best_overall_algo"], "best_overall_auc": d["best_overall_auc"]}
        for a_k, a_d in d["algo_data"].items():
            row[f"{a_k}_best_auc"] = a_d["best_auc"]
            row[f"{a_k}_best_final"] = a_d["best_final"]
            row[f"{a_k}_mean_auc"] = a_d["mean_auc"]
            row[f"{a_k}_best_config"] = a_d["best_config"]
        if d["match_0"]["valid"]:
            row["delta_auc_0"] = d["match_0"]["delta_auc"]
            row["winner_0"] = d["match_0"]["winner"]
        if d["match_lambda"]["valid"]:
            row["delta_auc_lambda"] = d["match_lambda"]["delta_auc"]
            row["winner_lambda"] = d["match_lambda"]["winner"]
        csv_rows.append(row)

    master_df = pd.DataFrame(csv_rows)
    csv_path = args.output_csv or os.path.join(suite_dir, "master_summary_table.csv")
    master_df.to_csv(csv_path, index=False)
    print(f"\nSaved master summary CSV to:\n  {csv_path}")
    print(f"Saved suite markdown summary to:\n  {md_path}")

    # Generate master PDF
    out_pdf = args.output_pdf or os.path.join(suite_dir, "master_td_vs_e_report.pdf")
    print(f"\nGenerating master PDF report: {out_pdf} ...")

    with PdfPages(out_pdf) as pdf:
        # Page 1: Suite Executive Dashboard
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        # Left: Matchup 0
        valid_m0 = [d for d in env_data_list if d["match_0"]["valid"]]
        if valid_m0:
            e_names0 = [d["env_name"] for d in valid_m0]
            d0_vals = [d["match_0"]["delta_auc"] for d in valid_m0]
            colors0 = [ALGO_PALETTE["E_0"] if v >= 0 else ALGO_PALETTE["TD_0"] for v in d0_vals]
            axes[0].barh(e_names0, d0_vals, color=colors0, edgecolor="#222", alpha=0.88)
            axes[0].axvline(0, color="k", linewidth=1.2)
            axes[0].set_title("A. 1-Step Advantage: Δ AUC (E(0) - TD(0))", fontweight="bold")
            axes[0].set_xlabel("Δ AUC", fontweight="bold")
            axes[0].grid(True, linestyle=":", alpha=0.6, axis="x")

        # Right: Matchup Lambda
        valid_ml = [d for d in env_data_list if d["match_lambda"]["valid"]]
        if valid_ml:
            e_namesl = [d["env_name"] for d in valid_ml]
            dl_vals = [d["match_lambda"]["delta_auc"] for d in valid_ml]
            colorsl = [ALGO_PALETTE["E_lambda"] if v >= 0 else ALGO_PALETTE["TD_lambda"] for v in dl_vals]
            axes[1].barh(e_namesl, dl_vals, color=colorsl, edgecolor="#222", alpha=0.88)
            axes[1].axvline(0, color="k", linewidth=1.2)
            axes[1].set_title("B. λ-Trace Advantage: Δ AUC (E(λ) - TD(λ))", fontweight="bold")
            axes[1].set_xlabel("Δ AUC", fontweight="bold")
            axes[1].grid(True, linestyle=":", alpha=0.6, axis="x")

        fig.suptitle(f"Executive Critic Benchmark Dashboard: {suite_name}", fontsize=14, fontweight="bold", y=0.98)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Cross-Environment Comparison Grid
        n_envs = len(env_data_list)
        n_cols = 4
        n_rows = math.ceil(n_envs / n_cols)
        fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows), squeeze=False)
        plt.subplots_adjust(hspace=0.45, wspace=0.3)

        algos_order = ["E_0", "TD_0", "E_lambda", "TD_lambda"]
        ls_map = {"E_0": "-", "TD_0": "--", "E_lambda": "-", "TD_lambda": "--"}

        for idx, d in enumerate(env_data_list):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes_grid[r, c]
            ename = d["env_name"]
            adata = d["algo_data"]

            for a_k in algos_order:
                if a_k in adata:
                    best_lbl = adata[a_k]["best_config"]
                    curves = adata[a_k]["curves"]
                    if best_lbl in curves:
                        m_c, s_c, _ = curves[best_lbl]
                        x_arr = np.arange(len(m_c))
                        ax.plot(x_arr, m_c, color=ALGO_PALETTE.get(a_k, "#333"),
                                linestyle=ls_map.get(a_k, "-"), lw=1.8,
                                label=ALGO_PRETTY_NAMES.get(a_k, a_k))

            ax.set_title(f"{ename}\n(Best: {d['best_overall_algo']})", fontweight="bold", fontsize=9)
            ax.set_xlabel("Updates", fontsize=7.5)
            ax.set_ylabel("Return", fontsize=7.5)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

        for i in range(n_envs, n_rows * n_cols):
            r = i // n_cols
            c = i % n_cols
            axes_grid[r, c].axis("off")

        fig_grid.suptitle(f"Cross-Environment Learning Curves: Best of Each Algorithm ({suite_name})",
                          fontsize=14, fontweight="bold", y=0.995)
        pdf.savefig(fig_grid, bbox_inches="tight")
        plt.close(fig_grid)

        # Pages 3+: Embed per-environment posters
        print("Embedding per-environment comparison posters...")
        for idx, d in enumerate(env_data_list, 1):
            ename = d["env_name"]
            png_path = d["poster_png"]
            if png_path and os.path.isfile(png_path):
                try:
                    img = plt.imread(png_path)
                    fig_p, ax_p = plt.subplots(figsize=(16, 11))
                    ax_p.imshow(img)
                    ax_p.axis("off")
                    fig_p.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    pdf.savefig(fig_p, bbox_inches="tight", dpi=300)
                    plt.close(fig_p)
                    print(f"  [{idx:02d}/{len(env_data_list)}] Embedded poster for {ename}")
                except Exception as ex:
                    print(f"  Warning: Failed to embed PNG for {ename}: {ex}")

    print("\n" + "=" * 85)
    print("MASTER REPORT COMPILATION COMPLETE:")
    print(f"  - Master PDF:     {out_pdf}")
    print(f"  - Summary CSV:    {csv_path}")
    print(f"  - Markdown Table: {md_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()

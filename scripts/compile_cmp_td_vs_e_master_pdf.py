#!/usr/bin/env python3
"""
scripts/compile_cmp_td_vs_e_master_pdf.py

Compiles all environment runs from a Classic TD vs. E_experimental comparison sweep
(e.g., results/sweeps/cmp_td_vs_e_12724069/) into a comprehensive master report:

Outputs:
1. <suite_dir>/master_summary_table.csv: Complete suite-wide comparison metrics, win rates, and deltas
2. <suite_dir>/master_td_vs_e_report.pdf: Multi-page publication-grade PDF containing:
   - Page 1: Executive Dashboard (Summary stats, win rate breakdown, performance delta bar chart)
   - Page 2: Cross-Environment Comparison Grid (Best E vs Best TD learning curves for all environments)
   - Pages 3+: Per-environment 4-panel detailed comparison posters

Usage:
    python scripts/compile_cmp_td_vs_e_master_pdf.py --suite-dir results/sweeps/cmp_td_vs_e_12724069
    python scripts/compile_cmp_td_vs_e_master_pdf.py cmp_td_vs_e_12724069
    python scripts/compile_cmp_td_vs_e_master_pdf.py 12724069
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
        os.path.join("results/sweeps", raw_path),
        os.path.join("results/sweeps", f"cmp_td_vs_e_{raw_path}"),
        os.path.join("results", raw_path),
        os.path.join("results", f"cmp_td_vs_e_{raw_path}"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    
    # Try glob matching
    for pattern in [f"results/sweeps/*{raw_path}*", f"results/*{raw_path}*"]:
        matches = glob.glob(pattern)
        if matches and os.path.isdir(matches[0]):
            return os.path.abspath(matches[0])
            
    raise FileNotFoundError(f"Could not find sweep directory for '{raw_path}'. Checked candidates: {candidates}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compile TD vs E_experimental sweep into a master PDF report")
    parser.add_argument("sweep_target", nargs="?", default=None,
                        help="Suite path, suite directory name, or SLURM Job ID (e.g., cmp_td_vs_e_12724069 or 12724069)")
    parser.add_argument("--suite-dir", type=str, default=None,
                        help="Explicit path to suite directory")
    parser.add_argument("--output-pdf", type=str, default=None,
                        help="Custom path for generated master PDF (default: <suite_dir>/master_td_vs_e_report.pdf)")
    return parser.parse_args()


def load_env_results(env_dir: str):
    """Extracts summary and curve data for a single environment in the suite."""
    env_name = os.path.basename(env_dir)
    h2h_csv = os.path.join(env_dir, "head_to_head_summary.csv")
    poster_pdf = os.path.join(env_dir, "head_to_head_poster.pdf")
    poster_png = os.path.join(env_dir, "head_to_head_poster.png")

    e_dir = os.path.join(env_dir, "E_experimental")
    td_dir = os.path.join(env_dir, "td")

    e_csv = os.path.join(e_dir, "summary_e_experimental.csv") if os.path.exists(os.path.join(e_dir, "summary_e_experimental.csv")) else os.path.join(e_dir, "summary_E_experimental.csv")
    td_csv = os.path.join(td_dir, "summary_td.csv")

    e_pkl = os.path.join(e_dir, "metrics.pkl")
    td_pkl = os.path.join(td_dir, "metrics.pkl")

    df_h2h = pd.read_csv(h2h_csv) if os.path.isfile(h2h_csv) else None
    df_e = pd.read_csv(e_csv) if os.path.isfile(e_csv) else None
    df_td = pd.read_csv(td_csv) if os.path.isfile(td_csv) else None

    # Load curves if available
    curves_e = None
    curves_td = None
    if os.path.isfile(e_pkl):
        try:
            with open(e_pkl, "rb") as f:
                d = pickle.load(f)
                curves_e = d.get("curves", d.get("curves_e", {}))
        except Exception:
            pass

    if os.path.isfile(td_pkl):
        try:
            with open(td_pkl, "rb") as f:
                d = pickle.load(f)
                curves_td = d.get("curves", d.get("curves_td", {}))
        except Exception:
            pass

    return {
        "env_name": env_name,
        "env_dir": env_dir,
        "df_h2h": df_h2h,
        "df_e": df_e,
        "df_td": df_td,
        "poster_pdf": poster_pdf if os.path.isfile(poster_pdf) else None,
        "poster_png": poster_png if os.path.isfile(poster_png) else None,
        "curves_e": curves_e,
        "curves_td": curves_td,
    }


def main():
    args = parse_args()
    raw_target = args.suite_dir or args.sweep_target
    if not raw_target:
        print("Error: Please provide a suite directory path or Job ID (e.g. 12724069).")
        sys.exit(1)

    suite_dir = resolve_suite_directory(raw_target)
    suite_name = os.path.basename(suite_dir)
    print(f"Loading suite from: {suite_dir}")

    # Discover environment directories
    env_dirs = [
        os.path.join(suite_dir, d) for d in sorted(os.listdir(suite_dir))
        if os.path.isdir(os.path.join(suite_dir, d)) and not d.startswith(".") and (
            os.path.isfile(os.path.join(suite_dir, d, "head_to_head_summary.csv")) or
            os.path.isdir(os.path.join(suite_dir, d, "E_experimental"))
        )
    ]

    if not env_dirs:
        print(f"Error: No completed environment directories found in '{suite_dir}'.")
        sys.exit(1)

    print(f"Discovered {len(env_dirs)} environments in suite:")
    for ed in env_dirs:
        print(f"  - {os.path.basename(ed)}")

    env_data_list = [load_env_results(ed) for ed in env_dirs]

    # =========================================================================
    # 1. COMPILE MASTER SUMMARY METRICS
    # =========================================================================
    summary_rows = []
    total_e_wins = 0
    total_td_wins = 0
    total_pairs = 0

    for d in env_data_list:
        ename = d["env_name"]
        df_h2h = d["df_h2h"]
        df_e = d["df_e"]
        df_td = d["df_td"]

        if df_e is not None and not df_e.empty:
            best_e = df_e.sort_values("final_window_mean", ascending=False).iloc[0]
            best_e_ret = best_e["final_window_mean"]
            best_e_sem = best_e.get("final_window_sem", 0.0)
            best_e_cfg = best_e["config"]
        else:
            best_e_ret = best_e_sem = np.nan
            best_e_cfg = "N/A"

        if df_td is not None and not df_td.empty:
            best_td = df_td.sort_values("final_window_mean", ascending=False).iloc[0]
            best_td_ret = best_td["final_window_mean"]
            best_td_sem = best_td.get("final_window_sem", 0.0)
            best_td_cfg = best_td["config"]
        else:
            best_td_ret = best_td_sem = np.nan
            best_td_cfg = "N/A"

        delta_best = best_e_ret - best_td_ret if not (np.isnan(best_e_ret) or np.isnan(best_td_ret)) else np.nan

        # Pairwise win statistics
        if df_h2h is not None and not df_h2h.empty:
            e_wins = int(np.sum(df_h2h["delta_e_minus_td"] > 0))
            td_wins = int(np.sum(df_h2h["delta_e_minus_td"] < 0))
            pairs = len(df_h2h)
            mean_delta_grid = float(df_h2h["delta_e_minus_td"].mean())

            # Paired t-test across matched configurations
            try:
                t_stat, p_val = stats.ttest_rel(df_h2h["final_window_mean_e"], df_h2h["final_window_mean_td"])
            except Exception:
                p_val = np.nan

            total_e_wins += e_wins
            total_td_wins += td_wins
            total_pairs += pairs
        else:
            e_wins = td_wins = pairs = 0
            mean_delta_grid = p_val = np.nan

        summary_rows.append({
            "environment": ename,
            "best_e_return": best_e_ret,
            "best_e_sem": best_e_sem,
            "best_e_config": best_e_cfg,
            "best_td_return": best_td_ret,
            "best_td_sem": best_td_sem,
            "best_td_config": best_td_cfg,
            "delta_best": delta_best,
            "mean_delta_grid": mean_delta_grid,
            "e_wins": e_wins,
            "td_wins": td_wins,
            "total_pairs": pairs,
            "win_rate_pct": (100.0 * e_wins / pairs) if pairs > 0 else np.nan,
            "p_value": p_val,
            "significant_05": (p_val < 0.05) if not np.isnan(p_val) else False,
        })

    df_master = pd.DataFrame(summary_rows)
    master_csv = os.path.join(suite_dir, "master_summary_table.csv")
    df_master.to_csv(master_csv, index=False)

    print("\n" + "=" * 80)
    print(f"MASTER COMPARISON SUMMARY: CLASSIC TD vs E_EXPERIMENTAL ({suite_name})")
    print(f"Total Environments Evaluated: {len(df_master)}")
    print(f"Global Head-to-Head Win Rate: {total_e_wins}/{total_pairs} ({100.0 * total_e_wins / max(1, total_pairs):.1f}% E Wins)")
    print("=" * 80)
    print(df_master[["environment", "best_e_return", "best_td_return", "delta_best", "win_rate_pct", "p_value"]].to_string(index=False))
    print("=" * 80)
    print(f"Saved master summary table to:\n  {master_csv}")

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

        # Panel (0, 0): Win Rate Breakdown
        ax_pie = fig.add_subplot(gs[0, 0])
        labels = [f"E_experimental ({total_e_wins})", f"Classic TD ({total_td_wins})"]
        colors = ["#1b7837", "#762a83"]
        ax_pie.pie([total_e_wins, total_td_wins], labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=140, textprops={"fontsize": 11, "fontweight": "bold"})
        ax_pie.set_title(f"Global Head-to-Head Win Rate across {total_pairs} Matched Pairs", fontweight="bold")

        # Panel (0, 1): Per-Environment Win Rate Bar Chart
        ax_win = fig.add_subplot(gs[0, 1])
        x_idx = np.arange(len(df_master))
        win_rates = df_master["win_rate_pct"].values
        bar_colors = ["#1b7837" if w >= 50.0 else "#762a83" for w in win_rates]
        ax_win.bar(x_idx, win_rates, color=bar_colors, alpha=0.85, edgecolor="black", lw=0.8)
        ax_win.axhline(50.0, color="gray", linestyle="--", lw=1.2, label="Parity (50%)")
        ax_win.set_xticks(x_idx)
        ax_win.set_xticklabels(df_master["environment"], rotation=30, ha="right", fontsize=8.5, fontweight="bold")
        ax_win.set_ylabel("E_experimental Win Rate (%)", fontweight="bold")
        ax_win.set_ylim(0, 105)
        ax_win.set_title("E_experimental Win Rate by Environment", fontweight="bold")
        ax_win.grid(axis="y", linestyle=":", alpha=0.6)
        ax_win.legend(loc="upper right")

        # Panel (1, :): Performance Delta (Best E - Best TD) Bar Chart
        ax_bar = fig.add_subplot(gs[1, :])
        deltas = df_master["delta_best"].values
        d_colors = ["#1b7837" if d >= 0 else "#762a83" for d in deltas]
        bars = ax_bar.bar(x_idx, deltas, color=d_colors, alpha=0.85, edgecolor="black", lw=0.8)
        ax_bar.axhline(0.0, color="black", lw=1.2)
        ax_bar.set_xticks(x_idx)
        ax_bar.set_xticklabels(df_master["environment"], rotation=25, ha="right", fontsize=9, fontweight="bold")
        ax_bar.set_ylabel("Score Delta (Best E - Best TD)", fontweight="bold")
        ax_bar.set_title("Performance Advantage: Best E_experimental vs. Best Classic TD", fontweight="bold")
        ax_bar.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                offset = 4 if h >= 0 else -12
                ax_bar.annotate(f"{h:+.1f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, offset), textcoords="offset points",
                                ha="center", va="bottom" if h >= 0 else "top",
                                fontsize=8.5, fontweight="bold")

        fig.suptitle(f"Master Head-to-Head Report: Classic TD vs E_experimental\nSuite: {suite_name}",
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
        plt.subplots_adjust(hspace=0.38, wspace=0.28)

        for idx, d in enumerate(env_data_list):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes_grid[r, c]
            ename = d["env_name"]
            c_e = d["curves_e"]
            c_td = d["curves_td"]
            df_e = d["df_e"]
            df_td = d["df_td"]

            has_curve = False
            if df_e is not None and c_e is not None and not df_e.empty:
                best_e_lbl = df_e.sort_values("final_window_mean", ascending=False).iloc[0]["config"]
                if best_e_lbl in c_e:
                    m_e, s_e = c_e[best_e_lbl]
                    x_e = np.arange(len(m_e))
                    ax.plot(x_e, m_e, color="#1b7837", lw=1.8, label="Best E")
                    ax.fill_between(x_e, m_e - s_e, m_e + s_e, color="#1b7837", alpha=0.18)
                    has_curve = True

            if df_td is not None and c_td is not None and not df_td.empty:
                best_td_lbl = df_td.sort_values("final_window_mean", ascending=False).iloc[0]["config"]
                if best_td_lbl in c_td:
                    m_td, s_td = c_td[best_td_lbl]
                    x_td = np.arange(len(m_td))
                    ax.plot(x_td, m_td, color="#762a83", lw=1.8, linestyle="--", label="Best TD")
                    ax.fill_between(x_td, m_td - s_td, m_td + s_td, color="#762a83", alpha=0.18)
                    has_curve = True

            ax.set_title(ename, fontweight="bold", fontsize=10)
            ax.set_xlabel("Steps (Updates)", fontsize=8)
            ax.set_ylabel("Return", fontsize=8)
            if has_curve:
                ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
            ax.grid(True, linestyle=":", alpha=0.5)

        for i in range(n_envs, n_rows * n_cols):
            r = i // n_cols
            c = i % n_cols
            axes_grid[r, c].axis("off")

        fig_grid.suptitle(f"Cross-Environment Learning Curves: Best E_experimental vs. Best Classic TD",
                          fontsize=15, fontweight="bold", y=0.995)
        pdf.savefig(fig_grid, bbox_inches="tight")
        plt.close(fig_grid)

        # ---------------------------------------------------------------------
        # Pages 3+: Per-Environment 4-Panel Detailed Posters
        # ---------------------------------------------------------------------
        print("Embedding detailed per-environment comparison posters...")
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
                    print(f"  Warning: Failed to embed PNG poster for {ename}: {ex}")

    print("\n" + "=" * 80)
    print("MASTER REPORT COMPILATION COMPLETE:")
    print(f"  - Master PDF:   {out_pdf}")
    print(f"  - Summary CSV:  {master_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()

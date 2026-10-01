#!/usr/bin/env python3
"""
scripts/brax_e_lambda/generate_brax_e_lambda_suite_pdf.py

Aggregates all environment runs in a Brax E(lambda) sweep suite into:
1. A suite-wide comparison summary table: <suite_dir>/suite_summary.csv
2. A publication-ready multi-page PDF report: <suite_dir>/brax_e_lambda_suite_report.pdf
"""

import os
import sys
import glob
import math
import argparse
import pickle
import numpy as np
import pandas as pd
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compile Brax E(lambda) sweep results into a publication PDF")
    parser.add_argument("--suite-dir", type=str, required=True,
                        help="Path to suite directory (e.g., results/sweeps/brax_e_lambda_123456)")
    parser.add_argument("--output-pdf", type=str, default=None,
                        help="Path for generated PDF report (default: <suite_dir>/brax_e_lambda_suite_report.pdf)")
    return parser.parse_args()


def load_env_data(env_dir):
    """Loads CSV summaries and pickle metrics for a given environment."""
    metrics_path = os.path.join(env_dir, "metrics.pkl")
    comp_path = os.path.join(env_dir, "summary_comparison.csv")
    base_path = os.path.join(env_dir, "summary_baseline.csv")
    e_path = os.path.join(env_dir, "summary_e_lambda.csv")

    data = {
        "env_name": os.path.basename(env_dir),
        "env_dir": env_dir,
        "metrics": None,
        "df_comp": None,
        "df_base": None,
        "df_e": None,
    }

    if os.path.isfile(metrics_path):
        with open(metrics_path, "rb") as f:
            data["metrics"] = pickle.load(f)
    if os.path.isfile(comp_path):
        data["df_comp"] = pd.read_csv(comp_path)
    if os.path.isfile(base_path):
        data["df_base"] = pd.read_csv(base_path)
    if os.path.isfile(e_path):
        data["df_e"] = pd.read_csv(e_path)

    return data


def main():
    args = parse_args()
    suite_dir = os.path.abspath(args.suite_dir)
    if not os.path.isdir(suite_dir):
        print(f"Error: suite directory '{suite_dir}' does not exist.")
        sys.exit(1)

    # Discover environment directories
    env_dirs = [
        os.path.join(suite_dir, d) for d in sorted(os.listdir(suite_dir))
        if os.path.isdir(os.path.join(suite_dir, d)) and not d.startswith(".")
        and os.path.isfile(os.path.join(suite_dir, d, "metrics.pkl"))
    ]

    if not env_dirs:
        print(f"No environment results found in '{suite_dir}'.")
        sys.exit(1)

    print(f"Found {len(env_dirs)} environment directories in {suite_dir}:")
    for ed in env_dirs:
        print(f"  - {os.path.basename(ed)}")

    all_env_data = [load_env_data(ed) for ed in env_dirs]

    # =========================================================================
    # 1. COMPILE GLOBAL SUITE SUMMARY TABLE
    # =========================================================================
    suite_rows = []
    for d in all_env_data:
        env_name = d["env_name"]
        df_comp = d["df_comp"]
        df_base = d["df_base"]
        df_e = d["df_e"]

        if df_comp is not None and not df_comp.empty:
            best_row = df_comp.iloc[0]
            suite_rows.append({
                "environment": env_name,
                "baseline_return": best_row["baseline_return"],
                "baseline_sem": best_row["baseline_sem"],
                "best_e_lambda_return": best_row["e_lambda_return"],
                "best_e_lambda_sem": best_row["e_lambda_sem"],
                "best_critic_lr": best_row["critic_lr"],
                "best_critic_epochs": best_row["critic_epochs"],
                "e_lambda": best_row.get("e_lambda", 0.9),
                "return_lambda": best_row.get("return_lambda", 0.9),
                "delta": best_row["delta"],
                "pct_gain": best_row["pct_gain"],
                "p_value": best_row["p_value"],
                "significant": best_row["p_value"] < 0.05,
            })
        elif df_e is not None and not df_e.empty:
            best_e = df_e.iloc[0]
            b_ret = df_base.iloc[0]["final_mean"] if df_base is not None and not df_base.empty else np.nan
            b_sem = df_base.iloc[0]["final_sem"] if df_base is not None and not df_base.empty else np.nan
            delta = best_e["final_mean"] - b_ret if not np.isnan(b_ret) else np.nan
            pct = (delta / abs(b_ret)) * 100.0 if not np.isnan(delta) and b_ret != 0 else np.nan
            suite_rows.append({
                "environment": env_name,
                "baseline_return": b_ret,
                "baseline_sem": b_sem,
                "best_e_lambda_return": best_e["final_mean"],
                "best_e_lambda_sem": best_e["final_sem"],
                "best_critic_lr": best_e["critic_lr"],
                "best_critic_epochs": best_e["critic_epochs"],
                "e_lambda": best_e.get("e_lambda", 0.9),
                "return_lambda": best_e.get("return_lambda", 0.9),
                "delta": delta,
                "pct_gain": pct,
                "p_value": np.nan,
                "significant": False,
            })

    df_suite = pd.DataFrame(suite_rows)
    suite_csv = os.path.join(suite_dir, "suite_summary.csv")
    df_suite.to_csv(suite_csv, index=False)
    print(f"\nSaved suite summary table to: {suite_csv}")
    print(df_suite.to_string(index=False))

    # =========================================================================
    # 2. GENERATE MULTI-PAGE PDF REPORT
    # =========================================================================
    out_pdf = args.output_pdf if args.output_pdf else os.path.join(suite_dir, "brax_e_lambda_suite_report.pdf")

    with PdfPages(out_pdf) as pdf:
        # Page 1: Overview Grid of Learning Curves
        n_envs = len(all_env_data)
        cols = 3
        rows = math.ceil(n_envs / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        for i, d in enumerate(all_env_data):
            r = i // cols
            c = i % cols
            ax = axes[r, c]
            m = d["metrics"]
            env_name = d["env_name"]

            if m is not None:
                step_axis = m.get("step_axis", np.arange(100))
                # Plot Baseline
                b_curves = m.get("baseline_curves", {})
                if b_curves and b_curves.get("mean") is not None:
                    ax.plot(step_axis, b_curves["mean"], color="#d62728", lw=1.8, linestyle="--", label="Baseline PPO")
                    if b_curves.get("sem") is not None:
                        ax.fill_between(step_axis, b_curves["mean"] - b_curves["sem"],
                                        b_curves["mean"] + b_curves["sem"], color="#d62728", alpha=0.15)

                # Plot Best E(lambda)
                e_results = m.get("e_lambda_results", [])
                e_curves = m.get("e_lambda_curves", {})
                if e_results and e_curves:
                    best_r = sorted(e_results, key=lambda x: x["final_mean"], reverse=True)[0]
                    best_c = e_curves.get(best_r["label"])
                    if best_c and "mean" in best_c:
                        elam = best_r.get("e_lambda", 0.9)
                        ax.plot(step_axis, best_c["mean"], color="#1f77b4", lw=2.0,
                                label=f"Best E(λ={elam}) (lr={best_r['critic_lr']}, ep={best_r['critic_epochs']})")
                        if best_c.get("sem") is not None:
                            ax.fill_between(step_axis, best_c["mean"] - best_c["sem"],
                                            best_c["mean"] + best_c["sem"], color="#1f77b4", alpha=0.2)

            ax.set_title(env_name.upper(), fontweight="bold")
            ax.set_xlabel("Steps (M)")
            ax.set_ylabel("Return")
            ax.legend(loc="lower right", fontsize=8, frameon=True)
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for i in range(n_envs, rows * cols):
            r = i // cols
            c = i % cols
            axes[r, c].axis("off")

        fig.suptitle("Brax Continuous Control Suite: Symmetrized E(λ=0.9) vs Baseline PPO", fontsize=15, fontweight="bold", y=0.995)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2: Summary Delta / Gain Bar Chart
        if not df_suite.empty and "delta" in df_suite.columns and not df_suite["delta"].isna().all():
            fig, ax = plt.subplots(figsize=(12, 6))
            x_pos = np.arange(len(df_suite))
            deltas = df_suite["delta"].values
            colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]

            bars = ax.bar(x_pos, deltas, color=colors, alpha=0.85, edgecolor="black", lw=0.8)
            ax.axhline(0, color="black", lw=1.0)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df_suite["environment"], rotation=25, ha="right", fontweight="bold")
            ax.set_ylabel("Score Delta (Best E(λ=0.9) - Baseline PPO)")
            ax.set_title("Performance Advantage of Symmetrized E(λ=0.9) over Baseline PPO", fontweight="bold", fontsize=13)
            ax.grid(axis="y", alpha=0.3)

            for bar in bars:
                h = bar.get_height()
                offset = 5 if h >= 0 else -15
                ax.annotate(f"{h:+.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, offset), textcoords="offset points",
                            ha="center", va="bottom" if h >= 0 else "top",
                            fontsize=9, fontweight="bold")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    print(f"Generated publication PDF report: {out_pdf}")


if __name__ == "__main__":
    main()

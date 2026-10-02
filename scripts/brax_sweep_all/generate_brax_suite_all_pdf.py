#!/usr/bin/env python3
"""
scripts/brax_sweep_all/generate_brax_suite_all_pdf.py

Aggregates all environment runs in a Brax 4-way critic sweep suite into:
1. Suite-wide summary CSV: <suite_dir>/suite_summary_best.csv
2. A publication-ready multi-page PDF report: <suite_dir>/brax_sweep_all_suite_report.pdf
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

ALGO_COLORS = {
    "E_0": "#1f78b4",         # Ocean Blue
    "TD_0": "#762a83",        # Dark Purple
    "E_lambda": "#33a02c",    # Forest Green
    "TD_lambda": "#e31a1c",   # Crimson Red
}

ALGO_PRETTY_NAMES = {
    "TD_0": "TD(0)",
    "E_0": "E(0)",
    "TD_lambda": r"TD($\lambda=0.9$)",
    "E_lambda": r"E($\lambda=0.9$)",
}

ALGO_LINE_STYLES = {
    "E_0": "-",
    "TD_0": "--",
    "E_lambda": "-",
    "TD_lambda": "--",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compile Brax 4-way sweep results into a publication PDF")
    parser.add_argument("--suite-dir", type=str, required=True,
                        help="Path to suite directory (e.g., results/sweeps/brax_sweep_all_123456)")
    parser.add_argument("--output-pdf", type=str, default=None,
                        help="Path for generated PDF report (default: <suite_dir>/brax_sweep_all_suite_report.pdf)")
    return parser.parse_args()


def load_env_data(env_dir):
    """Loads CSV summaries and pickle metrics for a given environment."""
    metrics_path = os.path.join(env_dir, "metrics.pkl")
    best_path = os.path.join(env_dir, "summary_best_per_algo.csv")
    cmp_0_path = os.path.join(env_dir, "summary_comparison_0.csv")
    cmp_l_path = os.path.join(env_dir, "summary_comparison_lambda.csv")

    data = {
        "env_name": os.path.basename(env_dir),
        "env_dir": env_dir,
        "metrics": None,
        "df_best": None,
        "df_cmp_0": None,
        "df_cmp_l": None,
    }

    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "rb") as f:
                data["metrics"] = pickle.load(f)
        except Exception as e:
            print(f"Warning: could not load {metrics_path}: {e}")
    if os.path.isfile(best_path):
        data["df_best"] = pd.read_csv(best_path)
    if os.path.isfile(cmp_0_path):
        data["df_cmp_0"] = pd.read_csv(cmp_0_path)
    if os.path.isfile(cmp_l_path):
        data["df_cmp_l"] = pd.read_csv(cmp_l_path)

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
        df_best = d["df_best"]
        df_c0 = d["df_cmp_0"]
        df_cl = d["df_cmp_l"]

        row = {"environment": env_name}
        if df_best is not None and not df_best.empty:
            for _, r in df_best.iterrows():
                algo = r["algorithm"]
                row[f"{algo}_return"] = r["final_mean"]
                row[f"{algo}_sem"] = r["final_sem"]
                row[f"{algo}_lr"] = r["critic_lr"]
                row[f"{algo}_ep"] = r["critic_epochs"]

            # Best overall
            best_overall = df_best.sort_values(by="final_mean", ascending=False).iloc[0]
            row["best_overall_algo"] = best_overall["algorithm"]
            row["best_overall_return"] = best_overall["final_mean"]

        if df_c0 is not None and not df_c0.empty:
            best_c0 = df_c0.iloc[0]
            row["delta_E0_vs_TD0"] = best_c0["delta"]
            row["p_val_0"] = best_c0["p_value"]

        if df_cl is not None and not df_cl.empty:
            best_cl = df_cl.iloc[0]
            row["delta_Elam_vs_TDlam"] = best_cl["delta"]
            row["p_val_lambda"] = best_cl["p_value"]

        # Also grab AUC if available from df_best
        if df_best is not None and not df_best.empty and "auc" in df_best.columns:
            for _, r in df_best.iterrows():
                algo = r["algorithm"]
                row[f"{algo}_auc"] = r["auc"]

        suite_rows.append(row)

    df_suite = pd.DataFrame(suite_rows)
    suite_csv = os.path.join(suite_dir, "suite_summary_best.csv")
    df_suite.to_csv(suite_csv, index=False)
    print(f"\nSaved suite summary table to: {suite_csv}")

    # =========================================================================
    # GENERATE MARKDOWN SUMMARY TABLE (like compile_cmp_td_vs_e_master_pdf.py)
    # =========================================================================
    md_lines = []
    md_lines.append("# Brax Continuous Control Benchmark Suite: 4-Way Critic Comparison\n")
    md_lines.append("Benchmark evaluating **TD(0)**, **E(0)**, **TD(λ=0.9)**, and **E(λ=0.9)** across all Brax continuous control environments.\n")
    md_lines.append("| Environment | E(0) Return | TD(0) Return | Δ (E0 - TD0) | Winner (0) | E(λ=0.9) Return | TD(λ=0.9) Return | Δ (Eλ - TDλ) | Winner (λ) | Best Overall |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    e0_wins, td0_wins, el_wins, tdl_wins = 0, 0, 0, 0
    e0_returns, td0_returns, el_returns, tdl_returns = [], [], [], []

    for _, r in df_suite.iterrows():
        env_name = r["environment"]
        e0_ret = r.get("E_0_return", np.nan)
        e0_sem = r.get("E_0_sem", np.nan)
        td0_ret = r.get("TD_0_return", np.nan)
        td0_sem = r.get("TD_0_sem", np.nan)

        el_ret = r.get("E_lambda_return", np.nan)
        el_sem = r.get("E_lambda_sem", np.nan)
        tdl_ret = r.get("TD_lambda_return", np.nan)
        tdl_sem = r.get("TD_lambda_sem", np.nan)

        if not np.isnan(e0_ret):
            e0_returns.append(e0_ret)
        if not np.isnan(td0_ret):
            td0_returns.append(td0_ret)
        if not np.isnan(el_ret):
            el_returns.append(el_ret)
        if not np.isnan(tdl_ret):
            tdl_returns.append(tdl_ret)

        e0_str = f"{e0_ret:.1f} ± {e0_sem:.1f}" if not np.isnan(e0_ret) else "—"
        td0_str = f"{td0_ret:.1f} ± {td0_sem:.1f}" if not np.isnan(td0_ret) else "—"

        if not np.isnan(e0_ret) and not np.isnan(td0_ret):
            d0 = e0_ret - td0_ret
            d0_str = f"{d0:+.1f}"
            if d0 > 0:
                w0_str = "**E(0)**"
                e0_wins += 1
            elif d0 < 0:
                w0_str = "**TD(0)**"
                td0_wins += 1
            else:
                w0_str = "Tie"
        else:
            d0_str = "—"
            w0_str = "—"

        el_str = f"{el_ret:.1f} ± {el_sem:.1f}" if not np.isnan(el_ret) else "—"
        tdl_str = f"{tdl_ret:.1f} ± {tdl_sem:.1f}" if not np.isnan(tdl_ret) else "—"

        if not np.isnan(el_ret) and not np.isnan(tdl_ret):
            dl = el_ret - tdl_ret
            dl_str = f"{dl:+.1f}"
            if dl > 0:
                wl_str = "**E(λ)**"
                el_wins += 1
            elif dl < 0:
                wl_str = "**TD(λ)**"
                tdl_wins += 1
            else:
                wl_str = "Tie"
        else:
            dl_str = "—"
            wl_str = "—"

        best_algo = r.get("best_overall_algo", "—")
        best_str = f"**{ALGO_PRETTY_NAMES.get(best_algo, best_algo)}**" if best_algo != "—" else "—"

        md_lines.append(f"| `{env_name}` | {e0_str} | {td0_str} | {d0_str} | {w0_str} | {el_str} | {tdl_str} | {dl_str} | {wl_str} | {best_str} |")

    # Summary Row
    mean_e0_str = f"{np.mean(e0_returns):.1f}" if e0_returns else "—"
    mean_td0_str = f"{np.mean(td0_returns):.1f}" if td0_returns else "—"
    mean_d0_str = f"{np.mean(e0_returns) - np.mean(td0_returns):+.1f}" if (e0_returns and td0_returns) else "—"

    mean_el_str = f"{np.mean(el_returns):.1f}" if el_returns else "—"
    mean_tdl_str = f"{np.mean(tdl_returns):.1f}" if tdl_returns else "—"
    mean_dl_str = f"{np.mean(el_returns) - np.mean(tdl_returns):+.1f}" if (el_returns and tdl_returns) else "—"

    md_lines.append(f"| **SUITE MEAN** | **{mean_e0_str}** | **{mean_td0_str}** | **{mean_d0_str}** | — | **{mean_el_str}** | **{mean_tdl_str}** | **{mean_dl_str}** | — | — |")
    md_lines.append(f"| **WIN TALLY** | **{e0_wins} Wins** | **{td0_wins} Wins** | — | — | **{el_wins} Wins** | **{tdl_wins} Wins** | — | — | — |\n")

    md_content = "\n".join(md_lines)
    suite_md = os.path.join(suite_dir, "suite_summary.md")
    with open(suite_md, "w") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print(md_content)
    print("=" * 80)
    print(f"Saved suite markdown summary to: {suite_md}")

    # =========================================================================
    # 2. GENERATE MULTI-PAGE PUBLICATION PDF REPORT
    # =========================================================================
    out_pdf = args.output_pdf if args.output_pdf else os.path.join(suite_dir, "brax_sweep_all_suite_report.pdf")

    with PdfPages(out_pdf) as pdf:
        # Page 1: Overview Grid of Learning Curves
        n_envs = len(all_env_data)
        cols = 3
        rows = math.ceil(n_envs / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.2 * rows), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        for i, d in enumerate(all_env_data):
            r = i // cols
            c = i % cols
            ax = axes[r, c]
            m = d["metrics"]
            env_name = d["env_name"]

            if m is not None:
                step_axis = m.get("step_axis", np.arange(100))
                best_res = m.get("best_results", {})
                curves_dict = m.get("curves_dict", {})

                for algo in ["TD_0", "E_0", "TD_lambda", "E_lambda"]:
                    if algo in best_res and algo in curves_dict:
                        best_label = best_res[algo]["label"]
                        curve = curves_dict[algo].get(best_label)
                        if curve is not None and "mean" in curve:
                            color = ALGO_COLORS.get(algo, "gray")
                            ls = ALGO_LINE_STYLES.get(algo, "-")
                            disp = ALGO_PRETTY_NAMES.get(algo, algo)
                            ax.plot(step_axis, curve["mean"], color=color, linestyle=ls, lw=1.8, label=disp)
                            if "sem" in curve:
                                ax.fill_between(step_axis, curve["mean"] - curve["sem"],
                                                curve["mean"] + curve["sem"], color=color, alpha=0.15)

            ax.set_title(env_name.upper(), fontweight="bold")
            ax.set_xlabel("Steps (M)")
            ax.set_ylabel("Return")
            ax.legend(loc="lower right", fontsize=8, frameon=True)
            ax.grid(True, alpha=0.3)

        for i in range(n_envs, rows * cols):
            axes[i // cols, i % cols].axis("off")

        fig.suptitle("Brax Continuous Control Suite: 4-Way Critic Benchmark", fontsize=15, fontweight="bold", y=0.995)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2: Pairwise Advantage Bar Charts across Suite
        if not df_suite.empty:
            fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10))
            plt.subplots_adjust(hspace=0.35)

            x_pos = np.arange(len(df_suite))
            env_labels = df_suite["environment"]

            # Subplot 1: E(0) vs TD(0)
            if "delta_E0_vs_TD0" in df_suite.columns:
                deltas_0 = df_suite["delta_E0_vs_TD0"].fillna(0).values
                colors_0 = [ALGO_COLORS["E_0"] if v >= 0 else ALGO_COLORS["TD_0"] for v in deltas_0]
                bars0 = ax_top.bar(x_pos, deltas_0, color=colors_0, edgecolor="black", alpha=0.85, width=0.55)
                ax_top.axhline(0, color="black", lw=1.0)
                ax_top.set_xticks(x_pos)
                ax_top.set_xticklabels(env_labels, rotation=20, ha="right", fontweight="bold")
                ax_top.set_ylabel(r"$\Delta$ Final Return (E(0) - TD(0))", fontweight="bold")
                ax_top.set_title("A. Advantage of E(0) over TD(0) across Environments", fontweight="bold", fontsize=12)
                ax_top.grid(axis="y", alpha=0.3)

                for bar in bars0:
                    h = bar.get_height()
                    offset = 4 if h >= 0 else -14
                    ax_top.annotate(f"{h:+.1f}",
                                    xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, offset), textcoords="offset points",
                                    ha="center", va="bottom" if h >= 0 else "top",
                                    fontsize=9, fontweight="bold")

            # Subplot 2: E(lambda) vs TD(lambda)
            if "delta_Elam_vs_TDlam" in df_suite.columns:
                deltas_l = df_suite["delta_Elam_vs_TDlam"].fillna(0).values
                colors_l = [ALGO_COLORS["E_lambda"] if v >= 0 else ALGO_COLORS["TD_lambda"] for v in deltas_l]
                barsl = ax_bot.bar(x_pos, deltas_l, color=colors_l, edgecolor="black", alpha=0.85, width=0.55)
                ax_bot.axhline(0, color="black", lw=1.0)
                ax_bot.set_xticks(x_pos)
                ax_bot.set_xticklabels(env_labels, rotation=20, ha="right", fontweight="bold")
                ax_bot.set_ylabel(r"$\Delta$ Final Return (E($\lambda$) - TD($\lambda$))", fontweight="bold")
                ax_bot.set_title(r"B. Advantage of E($\lambda=0.9$) over TD($\lambda=0.9$) across Environments", fontweight="bold", fontsize=12)
                ax_bot.grid(axis="y", alpha=0.3)

                for bar in barsl:
                    h = bar.get_height()
                    offset = 4 if h >= 0 else -14
                    ax_bot.annotate(f"{h:+.1f}",
                                    xy=(bar.get_x() + bar.get_width() / 2, h),
                                    xytext=(0, offset), textcoords="offset points",
                                    ha="center", va="bottom" if h >= 0 else "top",
                                    fontsize=9, fontweight="bold")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # Page 3: Grouped Bar Chart of Final Returns
        algos_present = [a for a in ["TD_0", "E_0", "TD_lambda", "E_lambda"] if f"{a}_return" in df_suite.columns]
        if algos_present:
            fig, ax = plt.subplots(figsize=(15, 7))
            n_envs = len(df_suite)
            n_algos = len(algos_present)
            width = 0.8 / n_algos
            x_indices = np.arange(n_envs)

            for idx, algo in enumerate(algos_present):
                offset = (idx - n_algos / 2 + 0.5) * width
                ret_vals = df_suite[f"{algo}_return"].fillna(0).values
                sem_vals = df_suite[f"{algo}_sem"].fillna(0).values if f"{algo}_sem" in df_suite.columns else None
                ax.bar(x_indices + offset, ret_vals, width=width, yerr=sem_vals,
                       capsize=3, label=ALGO_PRETTY_NAMES.get(algo, algo),
                       color=ALGO_COLORS.get(algo, "gray"), edgecolor="black", alpha=0.85)

            ax.set_xticks(x_indices)
            ax.set_xticklabels(df_suite["environment"], rotation=20, ha="right", fontweight="bold")
            ax.set_ylabel("Final Mean Return", fontweight="bold")
            ax.set_title("Suite-Wide Final Performance Comparison by Environment", fontweight="bold", fontsize=13)
            ax.legend(loc="upper left", frameon=True, fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    print(f"Generated publication PDF report: {out_pdf}")


if __name__ == "__main__":
    main()

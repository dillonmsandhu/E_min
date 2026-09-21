"""
scripts/generate_suite_pdf.py
Compiles learning curves for all environments in a sweep suite into a single publication-ready PDF.

Features:
1. Grid Overview Page:
   - Displays all environments in a clean multi-panel grid comparing Sampled E vs TD(lambda).
2. Detailed Per-Environment Profile Pages:
   - Side-by-side: E vs full lambda spectrum learning curves on the left, performance vs lambda sensitivity on the right.
3. Robust Fallbacks:
   - Automatically handles missing algorithms or partial runs gracefully.

Usage:
    python scripts/generate_suite_pdf.py --suite-dir results/ppo/sweeps/suite_1234567
    python scripts/generate_suite_pdf.py --suite-dir results/ppo/sweeps/suite_1234567 --metric returned_discounted_episode_returns
"""

import os
import sys
import argparse
import glob
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from notebooks.analyze_sweeps import (
    load_sweep_data,
    extract_best_configuration,
    plot_lambda_spectrum_vs_E,
)


def find_env_dirs(suite_dir: str):
    """Finds all environment directories within a suite directory."""
    if not os.path.isdir(suite_dir):
        return []
    
    env_dirs = []
    for item in sorted(os.listdir(suite_dir)):
        item_path = os.path.join(suite_dir, item)
        if not os.path.isdir(item_path) or item.startswith(".") or item == "comparison":
            continue
        # Check if it contains algorithm runs (e.g. sampled_E, sampled_td_lambda)
        sub_items = os.listdir(item_path)
        if any(algo in sub_items for algo in ["sampled_E", "sampled_td_lambda", "E", "td_lambda", "ppo"]):
            env_dirs.append((item, item_path))
    return env_dirs


def load_env_sweep_data(env_path: str):
    """Loads E and TD sweep data from an environment directory."""
    e_data = None
    td_data = None

    # Check for E
    for e_name in ["sampled_E", "E", "E_min"]:
        e_dir = os.path.join(env_path, e_name, "tuning")
        if not os.path.exists(e_dir):
            e_dir = os.path.join(env_path, e_name)
        if os.path.exists(e_dir):
            try:
                e_data = load_sweep_data(e_dir)
                break
            except Exception as ex:
                print(f"Warning: Failed to load E data from {e_dir}: {ex}")

    # Check for TD
    for td_name in ["sampled_td_lambda", "td_lambda", "td"]:
        td_dir = os.path.join(env_path, td_name, "tuning")
        if not os.path.exists(td_dir):
            td_dir = os.path.join(env_path, td_name)
        if os.path.exists(td_dir):
            try:
                td_data = load_sweep_data(td_dir)
                break
            except Exception as ex:
                print(f"Warning: Failed to load TD data from {td_dir}: {ex}")

    return e_data, td_data


def generate_suite_pdf(
    suite_dir: str,
    output_pdf: str = None,
    metric_key: str = "returned_discounted_episode_returns",
    rank_by: str = "final_window",
    window_size: int = 100,
    email: str = None,
):
    """Generates the multi-page PDF for the entire suite."""
    suite_name = os.path.basename(os.path.normpath(suite_dir))
    env_dirs = find_env_dirs(suite_dir)
    if not env_dirs:
        print(f"No environment directories with sweep data found in {suite_dir}")
        return None

    if output_pdf is None:
        output_pdf = os.path.join(suite_dir, "all_environments_learning_curves.pdf")

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    print(f"Found {len(env_dirs)} environments in suite: {[name for name, _ in env_dirs]}")
    print(f"Generating PDF: {output_pdf} ...")

    loaded_envs = []
    for env_name, env_path in env_dirs:
        e_data, td_data = load_env_sweep_data(env_path)
        if e_data is not None or td_data is not None:
            loaded_envs.append((env_name, env_path, e_data, td_data))

    if not loaded_envs:
        print("No valid sweep data could be loaded.")
        return None

    with PdfPages(output_pdf) as pdf:
        # ======================================================================
        # 1. OVERVIEW GRID (Poster View across all environments)
        # ======================================================================
        n_envs = len(loaded_envs)
        n_cols = min(4, n_envs)
        n_rows = math.ceil(n_envs / n_cols)

        fig_grid, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5.5 * n_cols, 4.0 * n_rows),
            squeeze=False,
        )

        suite_name = os.path.basename(os.path.normpath(suite_dir))
        fig_grid.suptitle(
            f"Gymnax Benchmark Suite: E-Minimization vs. TD(λ)\nSuite: {suite_name} | Metric: {metric_key}",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )

        for idx, (env_name, env_path, e_data, td_data) in enumerate(loaded_envs):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r][c]

            has_plotted = False

            # Plot E
            if e_data is not None and metric_key in e_data.get("metrics", {}):
                try:
                    e_traj, e_label, _, _ = extract_best_configuration(
                        e_data, metric_key=metric_key, rank_by=rank_by, window_size=window_size
                    )
                    x = np.arange(e_traj.shape[-1])
                    mean_e = e_traj.mean(axis=0)
                    std_e = e_traj.std(axis=0)
                    ax.plot(x, mean_e, label="Sampled E", color="#2ca02c", linewidth=2.4)
                    if e_traj.shape[0] > 1:
                        ax.fill_between(x, mean_e - std_e, mean_e + std_e, color="#2ca02c", alpha=0.2)
                    has_plotted = True
                except Exception as ex:
                    print(f"Could not extract E curve for {env_name}: {ex}")

            # Plot TD(lambda)
            if td_data is not None and metric_key in td_data.get("metrics", {}):
                try:
                    td_traj, td_label, _, _ = extract_best_configuration(
                        td_data, metric_key=metric_key, rank_by=rank_by, window_size=window_size
                    )
                    x = np.arange(td_traj.shape[-1])
                    mean_td = td_traj.mean(axis=0)
                    std_td = td_traj.std(axis=0)
                    ax.plot(x, mean_td, label="Best TD(λ)", color="#1f77b4", linewidth=2.0, linestyle="--")
                    if td_traj.shape[0] > 1:
                        ax.fill_between(x, mean_td - std_td, mean_td + std_td, color="#1f77b4", alpha=0.18)
                    has_plotted = True
                except Exception as ex:
                    print(f"Could not extract TD curve for {env_name}: {ex}")

            ax.set_title(env_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Update Steps", fontsize=10)
            ax.set_ylabel(metric_key if c == 0 else "", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            if has_plotted:
                ax.legend(loc="best", fontsize=8, framealpha=0.75)

        # Hide empty remaining grid subplots
        for idx in range(n_envs, n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r][c].axis("off")

        fig_grid.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig_grid, dpi=200)
        plt.close(fig_grid)

        # ======================================================================
        # 2. INDIVIDUAL ENVIRONMENT PROFILE PAGES (Detailed Spectrum vs E)
        # ======================================================================
        for env_name, env_path, e_data, td_data in loaded_envs:
            if td_data is not None and e_data is not None:
                try:
                    fig = plot_lambda_spectrum_vs_E(
                        td_sweep_data=td_data,
                        e_sweep_data=e_data,
                        metric_key=metric_key,
                        lambda_param="VALUE_LAMBDA",
                        title=f"{env_name}: E-Minimization vs. TD(λ) Spectrum",
                    )
                    pdf.savefig(fig, dpi=200)
                    plt.close(fig)
                except Exception as ex:
                    print(f"Warning: Failed to render detailed page for {env_name}: {ex}")

    print(f"Successfully generated full suite PDF: {output_pdf}")

    recipient = email or os.environ.get("EMAIL_RECIPIENT")
    if recipient and os.path.exists(output_pdf):
        from core.mail import email_pdf
        email_pdf(
            output_pdf,
            recipient=recipient,
            subject=f"Gymnax Suite Results: {suite_name}",
            body=f"Sweep suite completed!\nDirectory: {suite_dir}\nEnvironments: {len(loaded_envs)}\nMetric: {metric_key}",
        )

    return output_pdf


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive PDF of learning curves for all environments")
    parser.add_argument("--suite-dir", type=str, required=True, help="Path to suite directory (or root containing env dirs)")
    parser.add_argument("--output-pdf", type=str, default=None, help="Output path for PDF file")
    parser.add_argument("--metric", type=str, default="returned_discounted_episode_returns", help="Metric to plot")
    parser.add_argument("--rank-by", type=str, default="final_window", help="Ranking method")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for final window ranking")
    parser.add_argument("--email", type=str, default=None, help="Email address to send PDF upon generation")

    args = parser.parse_args()

    # If suite-dir is a job ID or partial name, search results
    suite_dir = args.suite_dir
    if not os.path.exists(suite_dir):
        matches = glob.glob(f"results/**/suite_{suite_dir}", recursive=True)
        if not matches:
            matches = glob.glob(f"results/**/*{suite_dir}*", recursive=True)
        if matches:
            suite_dir = matches[0]

    generate_suite_pdf(
        suite_dir=suite_dir,
        output_pdf=args.output_pdf,
        metric_key=args.metric,
        rank_by=args.rank_by,
        window_size=args.window_size,
        email=args.email,
    )


if __name__ == "__main__":
    main()

"""
generate_e_variants_suite_pdf.py
Compiles comprehensive comparison of the 4 variants of E across all environments in a sweep suite
into a single publication-ready PDF.

Pages:
  1. Poster Grid Overview:
     - Displays all environments in a clean multi-panel grid comparing the 4 variants of E:
       E, E_lambda_fixed, E_lambda_differentiable, and E_lambda_geometric.
  2. Per-Environment Profile Pages:
     - Side-by-side 2-panel figure:
       * Left: Learning curves of all 4 variants with error bands.
       * Right: Performance scaling as a function of E(λ) in [0.0, 0.5, 0.9].

Usage:
    python scripts/generate_e_variants_suite_pdf.py --suite-dir results/ppo/sweeps/suite_e_variants_12345
"""

import os
import sys
import argparse
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

from notebooks.analyze_sweeps import extract_best_configuration
from notebooks.plot_e_variants_comparison import (
    load_variant_data,
    plot_e_variants_comparison,
    VARIANTS_CONFIG,
)

TARGET_ALGOS = [
    "E",
    "E_lambda_fixed",
    "E_lambda_differentiable",
    "E_lambda_geometric",
    "ppo",
]


def find_env_dirs(suite_dir: str):
    """Finds all environment directories within a suite directory."""
    if not os.path.isdir(suite_dir):
        return []
    
    env_dirs = []
    for item in sorted(os.listdir(suite_dir)):
        item_path = os.path.join(suite_dir, item)
        if not os.path.isdir(item_path) or item.startswith(".") or item == "comparison":
            continue
        sub_items = os.listdir(item_path)
        if any(algo in sub_items for algo in TARGET_ALGOS):
            env_dirs.append((item, item_path))
    return env_dirs


def load_env_variants(env_path: str):
    """Loads all 4 variants for a given environment directory."""
    data = {}
    for algo in TARGET_ALGOS:
        d = load_variant_data(env_path, algo)
        if d is not None:
            data[algo] = d
    return data


def generate_e_variants_suite_pdf(
    suite_dir: str,
    output_pdf: str = None,
    metric_key: str = "returned_discounted_episode_returns",
    rank_by: str = "final_window",
    window_size: int = 100,
    email: str = None,
):
    """Generates the multi-page PDF for the entire 4 E-variants suite."""
    suite_name = os.path.basename(os.path.normpath(suite_dir))
    env_dirs = find_env_dirs(suite_dir)
    if not env_dirs:
        print(f"No environment directories with sweep data found in {suite_dir}")
        return None

    if output_pdf is None:
        output_pdf = os.path.join(suite_dir, "e_variants_suite_comparison.pdf")

    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)
    print(f"Found {len(env_dirs)} environments in suite: {[name for name, _ in env_dirs]}")
    print(f"Generating PDF: {output_pdf} ...")

    loaded_envs = []
    for env_name, env_path in env_dirs:
        v_data = load_env_variants(env_path)
        if v_data:
            loaded_envs.append((env_name, env_path, v_data))

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

        fig_grid.suptitle(
            f"Comparison of Four Variants of E across Tasks\nSuite: {suite_name} | Metric: {metric_key}",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )

        for idx, (env_name, env_path, v_data) in enumerate(loaded_envs):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r][c]

            has_plotted = False

            for algo_key in TARGET_ALGOS:
                if algo_key in v_data and metric_key in v_data[algo_key].get("metrics", {}):
                    try:
                        traj, label, _, _ = extract_best_configuration(
                            v_data[algo_key], metric_key=metric_key, rank_by=rank_by, window_size=window_size
                        )
                        x = np.arange(traj.shape[-1])
                        mean_y = traj.mean(axis=0)
                        std_y = traj.std(axis=0)

                        cfg = VARIANTS_CONFIG.get(algo_key, {})
                        color = cfg.get("color", "#333333")
                        disp_name = cfg.get("display_name", algo_key)

                        ax.plot(x, mean_y, label=disp_name, color=color, linewidth=2.0)
                        if traj.shape[0] > 1:
                            ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.15)
                        has_plotted = True
                    except Exception as ex:
                        print(f"Could not extract curve for {algo_key} in {env_name}: {ex}")

            ax.set_title(env_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Update Steps", fontsize=10)
            if c == 0:
                ax.set_ylabel(metric_key, fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            if has_plotted:
                ax.legend(loc="best", fontsize=7, framealpha=0.8)

        # Hide any unused subplots
        for extra_idx in range(n_envs, n_rows * n_cols):
            r = extra_idx // n_cols
            c = extra_idx % n_cols
            fig_grid.delaxes(axes[r][c])

        fig_grid.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(fig_grid, dpi=200)
        plt.close(fig_grid)

        # ======================================================================
        # 2. INDIVIDUAL ENVIRONMENT PROFILE PAGES (Learning Curves + E(λ) Scaling)
        # ======================================================================
        for env_name, env_path, v_data in loaded_envs:
            try:
                fig = plot_e_variants_comparison(
                    sweep_dir=env_path,
                    metric_key=metric_key,
                    rank_by=rank_by,
                    window_size=window_size,
                )
                if fig is not None:
                    pdf.savefig(fig, dpi=200)
                    plt.close(fig)
            except Exception as ex:
                print(f"Warning: Failed to render detailed profile for {env_name}: {ex}")

    print(f"Successfully generated E variants suite PDF: {output_pdf}")

    # Optional email notification
    if email:
        try:
            import subprocess
            cmd = f'echo "Sweep suite completed for {suite_name}. See attached comparison PDF." | mail -s "E-Variants Sweep Complete: {suite_name}" -a "{output_pdf}" "{email}"'
            subprocess.run(cmd, shell=True, check=True)
            print(f"Sent completion email with PDF attachment to: {email}")
        except Exception as e:
            print(f"Notice: Mail command could not be sent: {e}")

    return output_pdf


def main():
    parser = argparse.ArgumentParser(description="Generate 4 E-variants suite comparison PDF")
    parser.add_argument("--suite-dir", type=str, required=True, help="Suite root directory")
    parser.add_argument("--output-pdf", type=str, default=None, help="Explicit PDF output filepath")
    parser.add_argument("--metric", type=str, default="returned_discounted_episode_returns", help="Metric to rank and plot")
    parser.add_argument("--rank-by", type=str, default="final_window", help="Rank criteria (auc or final_window)")
    parser.add_argument("--window-size", type=int, default=100, help="Window size for final score")
    parser.add_argument("--email", type=str, default=None, help="Email recipient for PDF")

    args = parser.parse_args()
    generate_e_variants_suite_pdf(
        suite_dir=args.suite_dir,
        output_pdf=args.output_pdf,
        metric_key=args.metric,
        rank_by=args.rank_by,
        window_size=args.window_size,
        email=args.email,
    )


if __name__ == "__main__":
    main()

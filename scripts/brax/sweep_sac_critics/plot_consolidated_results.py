#!/usr/bin/env python3
"""
plot_consolidated_results.py

Consolidates per-environment runs in <out_dir>/*/data/sweep_metrics_all.pkl,
generates per-environment & multi-panel vector PDF learning curves, summary CSV,
best_configs.json, and Markdown/LaTeX tables.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (
        os.path.exists(os.path.join(p, "fixed_policy"))
        or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy")))
    )
]

from scripts.brax.sweep_sac_critics.sweep_sac_critics import (
    ENV_TITLES,
    METHOD_COLORS,
    METHODS,
    generate_outputs,
    score_candidates,
)


def plot_grid(consolidated_runs: dict, timesteps: dict, out_dir: str) -> None:
    envs = sorted(consolidated_runs.keys())
    if not envs:
        return
    ncols = min(4, len(envs))
    nrows = math.ceil(len(envs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)

    for idx, env_name in enumerate(envs):
        ax = axes[idx // ncols][idx % ncols]
        total_steps = timesteps[env_name]
        title = ENV_TITLES[env_name] if env_name in ENV_TITLES else env_name
        for method_key, method_label in METHODS:
            cand_dict = consolidated_runs[env_name].get(method_key, {})
            if not cand_dict:
                continue
            best_tag = score_candidates(cand_dict)[0][0]
            best_res = cand_dict[best_tag]
            curves = best_res["curves"]
            n_seeds, n_updates = curves.shape
            x = np.linspace(0, total_steps, n_updates)
            mean_c = np.mean(curves, axis=0)
            sem_c = np.std(curves, axis=0, ddof=0) / np.sqrt(n_seeds)
            color = METHOD_COLORS[method_key]
            ax.plot(x, mean_c, label=method_label, color=color, linewidth=1.6)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.18)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Env Steps", fontsize=9)
        ax.set_ylabel("Return", fontsize=9)
        ax.grid(True, alpha=0.25, linestyle="--")
        if idx == 0:
            ax.legend(frameon=False, fontsize=8)

    for idx in range(len(envs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Brax SAC Critic Comparison (Mean ± SEM)", fontsize=12.5, y=1.01)
    fig.tight_layout()
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    fig.savefig(os.path.join(plots_dir, "learning_curves_grid.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate Brax SAC Critics Sweep Results")
    parser.add_argument("--out_dir", type=str, default="results/sweep_sac_critics_brax")
    args = parser.parse_args()

    out_dir = args.out_dir
    env_pickles = glob.glob(os.path.join(out_dir, "*", "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        env_pickles = glob.glob(os.path.join(out_dir, "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        print(f"[Warning] No sweep_metrics_all.pkl found under {out_dir}")
        return

    consolidated_runs = {}
    timesteps = {}
    for pkl_path in sorted(env_pickles):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        for env_name, env_data in d.get("runs", {}).items():
            print(f"Loaded {env_name} from {pkl_path}")
            consolidated_runs[env_name] = env_data
            if "timesteps" in d and env_name in d["timesteps"]:
                timesteps[env_name] = d["timesteps"][env_name]

    generate_outputs(consolidated_runs, timesteps, out_dir)
    plot_grid(consolidated_runs, timesteps, out_dir)
    print(f"Consolidated outputs saved under {out_dir}/")


if __name__ == "__main__":
    main()

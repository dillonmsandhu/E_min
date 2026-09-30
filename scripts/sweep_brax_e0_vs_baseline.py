#!/usr/bin/env python3
"""
scripts/sweep_brax_e0_vs_baseline.py

Executes a hyperparameter sweep on Brax continuous control environments comparing:
1. Symmetrized E0 (adjacent-state Dirichlet error minimization) over:
   - Critic Learning Rate: [1e-4, 3e-4, 1e-3]
   - Critic Epochs: [4, 8, 16]
   (9 configurations total)
2. Standard PPO (Fitted Value Iteration MSE) as a single un-swept reference baseline:
   - Critic Learning Rate: 3e-4
   - Critic Epochs: 4

Evaluated across independent random seeds using JAX vmap.
Outputs:
- <out_dir>/summary_baseline.csv
- <out_dir>/summary_e0.csv
- <out_dir>/summary_comparison.csv
- <out_dir>/metrics.pkl
- <out_dir>/comparison_poster.pdf (and .png)
"""

import os
import sys
import time
import argparse
import datetime
import itertools
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from brax_rl.core.brax_config import config as default_config
from brax_rl.algos.ppo import make_train

# Set matplotlib parameters for publication-quality vector figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Brax E0 Sweep vs. Baseline PPO Reference")
    parser.add_argument("--env-name", type=str, default="hopper",
                        help="Brax environment name (default: hopper)")
    parser.add_argument("--total-timesteps", type=int, default=50_000_000,
                        help="Total environment timesteps (default: 50,000,000)")
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of independent random seeds (default: 5)")
    parser.add_argument("--num-envs", type=int, default=1024,
                        help="Number of parallel environments (default: 1024)")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="Number of rollout steps per env (default: 128)")
    parser.add_argument("--actor-lr", type=float, default=3e-4,
                        help="Actor learning rate (default: 3e-4)")
    parser.add_argument("--actor-epochs", type=int, default=4,
                        help="Actor update epochs (default: 4)")
    parser.add_argument("--actor-minibatch-size", type=int, default=1024,
                        help="Actor minibatch size (default: 1024)")
    parser.add_argument("--critic-minibatch-size", type=int, default=1024,
                        help="Critic minibatch size (default: 1024)")

    # E0 Sweep Grid
    parser.add_argument("--critic-lr-grid", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3],
                        help="Critic learning rate grid for E0 (default: 1e-4 3e-4 1e-3)")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[4, 8, 16],
                        help="Critic epochs grid for E0 (default: 4 8 16)")

    # Baseline PPO Settings
    parser.add_argument("--baseline-critic-lr", type=float, default=3e-4,
                        help="Critic learning rate for reference PPO (default: 3e-4)")
    parser.add_argument("--baseline-critic-epochs", type=int, default=4,
                        help="Critic epochs for reference PPO (default: 4)")

    # Execution controls
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Unified sweep identifier across SLURM tasks (default: SLURM_ARRAY_JOB_ID or timestamp)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Window size in policy iterations for final evaluation (default: 10)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip reference baseline PPO run")
    parser.add_argument("--skip-e0", action="store_true",
                        help="Skip E0 sweep")

    return parser.parse_args()


def run_configuration(cfg, n_seeds, base_seed=42):
    """Executes a single configuration vectorized across seeds using JAX vmap."""
    train_fn = make_train(cfg)
    train_vjit = jax.jit(jax.vmap(train_fn))
    rngs = jax.random.split(jax.random.PRNGKey(base_seed), n_seeds)
    outs = jax.block_until_ready(train_vjit(rngs))
    return outs["metrics"]


def extract_scalar_summary(returns_matrix, window_size=10):
    """
    returns_matrix: (n_seeds, num_updates)
    Returns: mean, sem, std, max, auc, seed_finals
    """
    n_seeds, num_updates = returns_matrix.shape
    w = min(window_size, num_updates)
    final_per_seed = np.mean(returns_matrix[:, -w:], axis=1)

    mean_ret = float(np.mean(final_per_seed))
    std_ret = float(np.std(final_per_seed, ddof=1)) if n_seeds > 1 else 0.0
    sem_ret = float(std_ret / np.sqrt(n_seeds)) if n_seeds > 1 else 0.0
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    auc_ret = float(np.mean(trapz_fn(returns_matrix, axis=1)))

    return {
        "final_mean": mean_ret,
        "final_sem": sem_ret,
        "final_std": std_ret,
        "max_return": max_ret,
        "auc": auc_ret,
        "seed_finals": final_per_seed,
    }


def main():
    args = parse_args()

    # Determine unified sweep directory
    sweep_id = args.sweep_id
    if sweep_id is None:
        array_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_id:
            sweep_id = f"brax_e0_{array_id}"
        else:
            job_id = os.environ.get("SLURM_JOB_ID")
            sweep_id = f"brax_e0_{job_id}" if job_id else f"brax_e0_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = args.output_dir if args.output_dir else os.path.join("results/sweeps", sweep_id, args.env_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print(f"BRAX EXPERIMENT: E0 CRITIC SWEEP VS BASELINE PPO")
    print(f"  Suite / Sweep ID: {sweep_id}")
    print(f"  Environment:      {args.env_name}")
    print(f"  Total Timesteps:  {args.total_timesteps:,}")
    print(f"  Seeds:            {args.n_seeds}")
    print(f"  Rollout:          {args.num_envs} envs x {args.num_steps} steps (batch = {args.num_envs * args.num_steps:,})")
    print(f"  E0 LR Grid:       {args.critic_lr_grid}")
    print(f"  E0 Epochs Grid:   {args.epochs_grid}")
    print(f"  Output Directory: {out_dir}")
    print("=" * 75)

    base_cfg = default_config.copy()
    base_cfg["ENV_NAME"] = args.env_name
    base_cfg["TOTAL_TIMESTEPS"] = args.total_timesteps
    base_cfg["NUM_ENVS"] = args.num_envs
    base_cfg["NUM_STEPS"] = args.num_steps
    base_cfg["ACTOR_LR"] = args.actor_lr
    base_cfg["ACTOR_EPOCHS"] = args.actor_epochs
    base_cfg["ACTOR_MINIBATCH_SIZE"] = args.actor_minibatch_size
    base_cfg["CRITIC_MINIBATCH_SIZE"] = args.critic_minibatch_size

    timesteps_per_update = args.num_envs * args.num_steps
    num_updates = args.total_timesteps // timesteps_per_update
    step_axis = np.arange(1, num_updates + 1) * timesteps_per_update / 1_000_000.0  # in Millions

    baseline_metrics = None
    baseline_summary = None
    baseline_curve_mean = None
    baseline_curve_sem = None

    # =========================================================================
    # 1. RUN BASELINE PPO (fitted)
    # =========================================================================
    if not args.skip_baseline:
        print("\n>>> [1/2] RUNNING BASELINE PPO (FITTED VALUE ITERATION)...")
        b_cfg = base_cfg.copy()
        b_cfg["CRITIC_TYPE"] = "fitted"
        b_cfg["CRITIC_LR"] = args.baseline_critic_lr
        b_cfg["CRITIC_EPOCHS"] = args.baseline_critic_epochs
        b_cfg["VALUE_LAMBDA"] = 0.95
        b_cfg["GAE_LAMBDA"] = 0.95

        t0 = time.time()
        raw_b_metrics = run_configuration(b_cfg, args.n_seeds)
        b_time = time.time() - t0

        b_returns = np.array(raw_b_metrics["returned_episode_returns"])  # (n_seeds, updates)
        baseline_curve_mean = np.mean(b_returns, axis=0)
        baseline_curve_sem = np.std(b_returns, axis=0, ddof=1) / np.sqrt(args.n_seeds) if args.n_seeds > 1 else np.zeros_like(baseline_curve_mean)

        b_stats = extract_scalar_summary(b_returns, args.window_size)
        baseline_summary = {
            "algorithm": "PPO_Baseline",
            "critic_lr": args.baseline_critic_lr,
            "critic_epochs": args.baseline_critic_epochs,
            "final_mean": b_stats["final_mean"],
            "final_sem": b_stats["final_sem"],
            "final_std": b_stats["final_std"],
            "max_return": b_stats["max_return"],
            "auc": b_stats["auc"],
            "walltime_sec": b_time,
        }
        df_base = pd.DataFrame([baseline_summary])
        df_base.to_csv(os.path.join(out_dir, "summary_baseline.csv"), index=False)
        print(f"Baseline PPO Finished in {b_time:.1f}s | Final Return: {b_stats['final_mean']:.2f} ± {b_stats['final_sem']:.2f}")

    # =========================================================================
    # 2. RUN E0 SWEEP GRID
    # =========================================================================
    e0_results = []
    e0_curves = {}
    e0_all_metrics = {}

    if not args.skip_e0:
        grid = list(itertools.product(args.critic_lr_grid, args.epochs_grid))
        print(f"\n>>> [2/2] RUNNING E0 CRITIC SWEEP ({len(grid)} Configurations)...")

        for idx, (clr, ep) in enumerate(grid, 1):
            label = f"E0_clr{clr}_ep{ep}"
            print(f"[{idx:02d}/{len(grid)}] Running E0 with CRITIC_LR={clr}, CRITIC_EPOCHS={ep} ...")

            cfg = base_cfg.copy()
            cfg["CRITIC_TYPE"] = "e_0"
            cfg["CRITIC_LR"] = clr
            cfg["CRITIC_EPOCHS"] = ep
            cfg["RETURN_LAMBDA"] = 1.0
            cfg["GAE_LAMBDA"] = 0.95

            t0 = time.time()
            raw_metrics = run_configuration(cfg, args.n_seeds)
            elapsed = time.time() - t0

            returns = np.array(raw_metrics["returned_episode_returns"])  # (n_seeds, updates)
            e0_all_metrics[label] = {
                "returns": returns,
                "delta_mag": np.array(raw_metrics.get("delta_mag", [])),
                "magnitude_loss": np.array(raw_metrics.get("magnitude_loss", [])),
                "dirichlet_loss": np.array(raw_metrics.get("dirichlet_loss", [])),
            }
            e0_curves[label] = {
                "mean": np.mean(returns, axis=0),
                "sem": np.std(returns, axis=0, ddof=1) / np.sqrt(args.n_seeds) if args.n_seeds > 1 else np.zeros(returns.shape[1]),
                "critic_lr": clr,
                "critic_epochs": ep,
            }

            stats_dict = extract_scalar_summary(returns, args.window_size)
            row = {
                "algorithm": "E_0",
                "label": label,
                "critic_lr": clr,
                "critic_epochs": ep,
                "final_mean": stats_dict["final_mean"],
                "final_sem": stats_dict["final_sem"],
                "final_std": stats_dict["final_std"],
                "max_return": stats_dict["max_return"],
                "auc": stats_dict["auc"],
                "walltime_sec": elapsed,
                "_seed_finals": stats_dict["seed_finals"],
            }
            e0_results.append(row)
            print(f"    Finished in {elapsed:.1f}s | Return: {stats_dict['final_mean']:.2f} ± {stats_dict['final_sem']:.2f} | Max: {stats_dict['max_return']:.2f}")

        df_e0 = pd.DataFrame([{k: v for k, v in r.items() if k != "_seed_finals"} for r in e0_results])
        df_e0.sort_values(by="final_mean", ascending=False, inplace=True)
        df_e0.to_csv(os.path.join(out_dir, "summary_e0.csv"), index=False)

    # Save full metrics pickle
    with open(os.path.join(out_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({
            "baseline": baseline_summary,
            "baseline_curves": {"mean": baseline_curve_mean, "sem": baseline_curve_sem},
            "e0_results": e0_results,
            "e0_curves": e0_curves,
            "e0_all_metrics": e0_all_metrics,
            "step_axis": step_axis,
            "config": base_cfg,
        }, f)

    # =========================================================================
    # 3. STATISTICAL COMPARISON & PAIRWISE SUMMARY
    # =========================================================================
    if baseline_summary is not None and len(e0_results) > 0:
        base_seeds = extract_scalar_summary(b_returns, args.window_size)["seed_finals"]
        comparison_rows = []

        for r in e0_results:
            e0_seeds = r["_seed_finals"]
            delta = float(np.mean(e0_seeds) - np.mean(base_seeds))
            pct_gain = float(delta / (abs(np.mean(base_seeds)) + 1e-8) * 100.0)

            # Statistical significance via paired t-test
            if len(e0_seeds) == len(base_seeds) and len(e0_seeds) > 1:
                try:
                    t_stat, p_val = stats.ttest_rel(e0_seeds, base_seeds)
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0

            comparison_rows.append({
                "env_name": args.env_name,
                "label": r["label"],
                "critic_lr": r["critic_lr"],
                "critic_epochs": r["critic_epochs"],
                "e0_return": r["final_mean"],
                "e0_sem": r["final_sem"],
                "baseline_return": baseline_summary["final_mean"],
                "baseline_sem": baseline_summary["final_sem"],
                "delta": delta,
                "pct_gain": pct_gain,
                "p_value": p_val,
                "significant_05": p_val < 0.05,
            })

        df_cmp = pd.DataFrame(comparison_rows)
        df_cmp.sort_values(by="e0_return", ascending=False, inplace=True)
        df_cmp.to_csv(os.path.join(out_dir, "summary_comparison.csv"), index=False)

        best_e0 = df_cmp.iloc[0]
        print("\n" + "=" * 75)
        print("HEAD-TO-HEAD COMPARISON SUMMARY:")
        print(f"  Baseline PPO:  {baseline_summary['final_mean']:.2f} ± {baseline_summary['final_sem']:.2f}")
        print(f"  Best E0:       {best_e0['e0_return']:.2f} ± {best_e0['e0_sem']:.2f} ({best_e0['label']})")
        print(f"  Delta:         {best_e0['delta']:+.2f} ({best_e0['pct_gain']:+.2f}%) | p-value: {best_e0['p_value']:.4f}")
        print("=" * 75)

    # =========================================================================
    # 4. PUBLICATION QUALITY 4-PANEL POSTER FIGURE
    # =========================================================================
    if len(e0_results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.32, wspace=0.25)

        # Panel 1: Learning Curves of Best E0 vs Baseline PPO
        ax1 = axes[0, 0]
        best_r = sorted(e0_results, key=lambda x: x["final_mean"], reverse=True)[0]
        best_label = best_r["label"]
        best_curve = e0_curves[best_label]

        ax1.plot(step_axis, best_curve["mean"], color="#1f77b4", lw=2.2,
                 label=f"Best E0 ({best_r['critic_lr']}, ep={best_r['critic_epochs']})")
        ax1.fill_between(step_axis, best_curve["mean"] - best_curve["sem"],
                         best_curve["mean"] + best_curve["sem"], color="#1f77b4", alpha=0.2)

        if baseline_curve_mean is not None:
            ax1.plot(step_axis, baseline_curve_mean, color="#d62728", lw=2.0, linestyle="--",
                     label=f"Baseline PPO (3e-4, ep=4)")
            ax1.fill_between(step_axis, baseline_curve_mean - baseline_curve_sem,
                             baseline_curve_mean + baseline_curve_sem, color="#d62728", alpha=0.15)

        ax1.set_title(f"A. Learning Curves: Best E0 vs Baseline ({args.env_name})", fontweight="bold")
        ax1.set_xlabel("Environment Steps (Millions)")
        ax1.set_ylabel("Episode Return")
        ax1.legend(loc="lower right", frameon=True)
        ax1.grid(True, alpha=0.3)

        # Panel 2: E0 Full Grid Learning Curves
        ax2 = axes[0, 1]
        colors = {1e-4: "#2ca02c", 3e-4: "#1f77b4", 1e-3: "#ff7f0e"}
        linestyles = {4: "-", 8: "--", 16: ":"}

        for label, cinfo in e0_curves.items():
            clr = cinfo["critic_lr"]
            ep = cinfo["critic_epochs"]
            color = colors.get(clr, "gray")
            ls = linestyles.get(ep, "-")
            ax2.plot(step_axis, cinfo["mean"], color=color, linestyle=ls, lw=1.3,
                     label=f"lr={clr} ep={ep}")

        if baseline_curve_mean is not None:
            ax2.plot(step_axis, baseline_curve_mean, color="#d62728", lw=2.0, linestyle="-.", label="Baseline PPO")

        ax2.set_title("B. E0 Learning Dynamics Across Grid", fontweight="bold")
        ax2.set_xlabel("Environment Steps (Millions)")
        ax2.set_ylabel("Episode Return")
        ax2.legend(loc="lower right", fontsize=8, ncol=2, frameon=True)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Performance vs Critic Epochs stratified by Critic LR
        ax3 = axes[1, 0]
        for clr in sorted(args.critic_lr_grid):
            sub = [r for r in e0_results if r["critic_lr"] == clr]
            sub.sort(key=lambda x: x["critic_epochs"])
            x_eps = [s["critic_epochs"] for s in sub]
            y_means = [s["final_mean"] for s in sub]
            y_sems = [s["final_sem"] for s in sub]
            ax3.errorbar(x_eps, y_means, yerr=y_sems, marker="o", lw=1.8, capsize=4,
                         color=colors.get(clr, "#1f77b4"), label=f"Critic LR = {clr}")

        if baseline_summary is not None:
            ax3.axhline(baseline_summary["final_mean"], color="#d62728", linestyle="--", lw=1.8,
                        label=f"Baseline PPO ({baseline_summary['final_mean']:.1f})")

        ax3.set_title("C. Critic Epochs Scaling (4 -> 8 -> 16)", fontweight="bold")
        ax3.set_xlabel("Critic Epochs per Update")
        ax3.set_ylabel("Final Mean Return")
        ax3.set_xticks(sorted(args.epochs_grid))
        ax3.legend(loc="best", frameon=True)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Heatmap of Performance (LR x Epochs)
        ax4 = axes[1, 1]
        lrs = sorted(args.critic_lr_grid)
        epochs = sorted(args.epochs_grid)
        grid_matrix = np.zeros((len(lrs), len(epochs)))

        for i, lr_val in enumerate(lrs):
            for j, ep_val in enumerate(epochs):
                match = [r for r in e0_results if r["critic_lr"] == lr_val and r["critic_epochs"] == ep_val]
                grid_matrix[i, j] = match[0]["final_mean"] if match else np.nan

        im = ax4.imshow(grid_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax4, label="Final Mean Return")

        ax4.set_xticks(np.arange(len(epochs)))
        ax4.set_xticklabels(epochs)
        ax4.set_yticks(np.arange(len(lrs)))
        ax4.set_yticklabels([str(x) for x in lrs])
        ax4.set_xlabel("Critic Epochs")
        ax4.set_ylabel("Critic Learning Rate")
        ax4.set_title("D. E0 Hyperparameter Sensitivity Heatmap", fontweight="bold")

        for i in range(len(lrs)):
            for j in range(len(epochs)):
                val = grid_matrix[i, j]
                ax4.text(j, i, f"{val:.1f}", ha="center", va="center",
                         color="white" if val < np.nanmean(grid_matrix) else "black",
                         fontweight="bold")

        pdf_path = os.path.join(out_dir, "comparison_poster.pdf")
        png_path = os.path.join(out_dir, "comparison_poster.png")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Generated comparison figures:\n  {pdf_path}\n  {png_path}")


if __name__ == "__main__":
    main()

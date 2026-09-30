#!/usr/bin/env python3
"""
sweep_td_vs_e_experimental.py

Standard Head-to-Head Comparison Sweep:
Classic TD Learning (algos/td.py) vs. Symmetrized E-Minimization (algos/E_experimental.py)
across the multi-dimensional grid:
    - Critic LR: [0.0003, 0.001, 0.003]  (3 values)
    - Critic Epochs: [4, 16, 32]          (3 values)
    - Weight Decay: [0.001, 0.01]         (2 values)
    - Value Heads: [1, 4]                 (2 values)

Total Configurations per algorithm: 3 x 3 x 2 x 2 = 36 configurations.
Total Evaluated across both: 72 configurations (vmapped over seeds in JAX).

Outputs:
    - <out_dir>/td/: summary_td.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/E_experimental/: summary_e_experimental.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/head_to_head_summary.csv: pairwise matching table (E vs TD) with delta and p-values
    - <out_dir>/head_to_head_poster.pdf & .png: 4-panel comparison poster:
        1. Learning Curves of Best E vs. Best TD (Mean ± 1 SEM)
        2. Epochs Scaling: Performance vs Critic Epochs (4 -> 16 -> 32) for E vs TD
        3. Value Heads Scaling: Performance for 1 Head vs 4 Heads for E vs TD
        4. Matched-Pair Scatter: E vs TD across all 36 configurations with win rates

Usage:
    python scripts/sweep_td_vs_e_experimental.py --env-name MountainCarContinuous-v0 --n-seeds 8
    python scripts/sweep_td_vs_e_experimental.py --env-name CartPole-v1 --n-seeds 8
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
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from core.config import config as default_config
from algos.E_experimental import make_train as make_train_e
from algos.td import make_train as make_train_td
from scripts.e_optimization.visualize_multidim_sweep import generate_multidim_analysis_pdf

# TrueType font embedding for vector publication quality
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head comparison sweep: TD vs E_experimental")
    parser.add_argument("--env-name", type=str, default="MountainCarContinuous-v0",
                        help="Gymnax environment name (default: MountainCarContinuous-v0)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment timesteps (default: 2,048,000 for classic, 10M for MinAtar)")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Number of independent random seeds (default: 8)")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments (default: 64)")
    parser.add_argument("--num-steps", type=int, default=64,
                        help="Number of rollout steps per env (default: 64)")
    parser.add_argument("--minibatch-size", type=int, default=1024,
                        help="Minibatch size for SGD updates (default: 1024)")
    parser.add_argument("--actor-lr", type=float, default=0.0003,
                        help="Actor learning rate held constant (default: 0.0003)")
    parser.add_argument("--actor-epochs", type=int, default=4,
                        help="Actor update epochs held constant (default: 4)")
    
    # Grids for both algorithms
    parser.add_argument("--critic-lr-grid", type=float, nargs="+", default=[0.0003, 0.001, 0.003],
                        help="Critic learning rate grid (default: 0.0003 0.001 0.003)")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[4, 16, 32],
                        help="Critic epochs grid (default: 4 16 32)")
    parser.add_argument("--wd-grid", type=float, nargs="+", default=[0.001, 0.01],
                        help="Critic weight decay grid (default: 0.001 0.01)")
    parser.add_argument("--heads-grid", type=int, nargs="+", default=[1, 4],
                        help="Value heads grid (default: 1 4)")
    
    parser.add_argument("--critic-loss-type", type=str, default="mse",
                        help="Critic loss type for both algos (default: mse)")
    parser.add_argument("--td-lambda", type=float, default=0.0,
                        help="TD lambda parameter for algos.td (default: 0.0 for live 1-step TD)")
    parser.add_argument("--return-lambda-grid", type=float, nargs="+", default=[0.9, 0.99],
                        help="Return anchor lambda grid for E_experimental (default: 0.9 0.99)")
    parser.add_argument("--return-lambda", type=float, default=None,
                        help="Single return anchor lambda for backwards compatibility")
    
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Unified sweep identifier across SLURM array tasks (default: SLURM_ARRAY_JOB_ID or timestamp)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for final mean evaluation (default: 100)")
    parser.add_argument("--metric", type=str, default="returned_episode_returns",
                        help="Primary evaluation metric (default: returned_episode_returns)")
    return parser.parse_args()


def run_single_algo_sweep(algo_name, make_train_fn, grid, args, out_algo_dir):
    """Executes the full grid for a single algorithm across all random seeds."""
    os.makedirs(out_algo_dir, exist_ok=True)
    summary_rows = []
    curves_dict = {}
    all_metrics = {}

    print(f"\n{'='*70}")
    print(f"RUNNING SWEEP FOR ALGORITHM: {algo_name}")
    print(f"Total Configurations: {len(grid)} | Seeds: {args.n_seeds}")
    print(f"Output Directory: {out_algo_dir}")
    print(f"{'='*70}")

    for idx, item in enumerate(grid, 1):
        label = item["label"]
        clr = item["critic_lr"]
        ep = item["critic_epochs"]
        wd = item["weight_decay"]
        heads = item["num_value_heads"]

        print(f"[{idx:02d}/{len(grid)}] {algo_name} | {label} ...")

        cfg = default_config.copy()
        cfg["ENV_NAME"] = args.env_name
        cfg["TOTAL_TIMESTEPS"] = args.total_timesteps
        cfg["NUM_ENVS"] = args.num_envs
        cfg["NUM_STEPS"] = args.num_steps
        cfg["MINIBATCH_SIZE"] = min(args.minibatch_size, args.num_envs * args.num_steps)
        cfg["NUM_EPOCHS"] = ep
        cfg["CRITIC_EPOCHS"] = ep
        cfg["ACTOR_EPOCHS"] = args.actor_epochs
        cfg["WEIGHT_DECAY"] = wd
        cfg["CRITIC_WEIGHT_DECAY"] = wd
        cfg["ACTOR_WEIGHT_DECAY"] = wd
        cfg["CRITIC_LOSS_TYPE"] = args.critic_loss_type
        cfg["NUM_VALUE_HEADS"] = heads
        cfg["VALUE_HEAD_AGG"] = "sum"
        cfg["ACTOR_LR"] = args.actor_lr
        cfg["LR"] = clr
        cfg["CRITIC_LR"] = clr
        cfg["TD_LAMBDA"] = item.get("td_lambda", args.td_lambda)
        cfg["RETURN_LAMBDA"] = item.get("return_lambda", args.return_lambda if args.return_lambda is not None else 0.99)

        start_time = time.time()
        train_fn = make_train_fn(cfg)
        train_vjit = jax.jit(jax.vmap(train_fn))

        rng = jax.random.PRNGKey(42)
        rngs = jax.random.split(rng, args.n_seeds)

        out = train_vjit(rngs)
        # Block until JAX execution completes
        _ = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")

        # Extract metric
        if args.metric in out["metrics"]:
            m_arr = out["metrics"][args.metric]
        elif "returned_discounted_episode_returns" in out["metrics"]:
            m_arr = out["metrics"]["returned_discounted_episode_returns"]
        else:
            m_arr = out["metrics"]["returned_episode_returns"]

        tensor = np.asarray(m_arr)
        mean_c = tensor.mean(axis=0)
        sem_c = tensor.std(axis=0) / np.sqrt(args.n_seeds)

        curves_dict[label] = (mean_c, sem_c)
        all_metrics[label] = {k: np.asarray(v) for k, v in out["metrics"].items()}

        win = min(len(mean_c), args.window_size)
        final_mean = float(np.mean(mean_c[-win:]))
        final_sem = float(np.mean(sem_c[-win:]))
        auc = float(np.mean(mean_c))

        row = {
            "algo": algo_name,
            "config": label,
            "critic_lr": clr,
            "critic_epochs": ep,
            "weight_decay": wd,
            "num_value_heads": heads,
            "final_window_mean": final_mean,
            "final_window_sem": final_sem,
            "auc": auc,
            "elapsed_seconds": elapsed,
        }
        if "return_lambda" in item:
            row["return_lambda"] = item["return_lambda"]
        if "td_lambda" in item:
            row["td_lambda"] = item["td_lambda"]
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.sort_values(by="final_window_mean", ascending=False, inplace=True)
    csv_name = f"summary_{algo_name.lower()}.csv"
    df.to_csv(os.path.join(out_algo_dir, csv_name), index=False)

    # Save metrics pickle
    with open(os.path.join(out_algo_dir, "metrics.pkl"), "wb") as f:
        pickle.dump(all_metrics, f)

    # Generate multi-dimensional poster for this algorithm
    try:
        generate_multidim_analysis_pdf(out_algo_dir)
    except Exception as ex:
        print(f"Warning: Failed to generate multidim analysis for {algo_name}: {ex}")

    return df, curves_dict, all_metrics


def plot_head_to_head_poster(df_e, curves_e, df_td, curves_td, out_dir, env_name, window_size=100):
    """
    Generates a 4-panel publication-grade comparison poster between E_experimental and TD:
        1. Best Learning Curves with Mean ± 1 SEM
        2. Epochs Scaling: Performance vs Critic Epochs (4 -> 16 -> 32)
        3. Value Heads Scaling: Performance for 1 Head vs 4 Heads
        4. Matched-Pair Scatter: E vs TD across all identical hyperparameter configurations
    """
    pdf_path = os.path.join(out_dir, "head_to_head_poster.pdf")
    png_path = os.path.join(out_dir, "head_to_head_poster.png")

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    # --------------------------------------------------------------------------
    # Panel 1: Best Learning Curves (E vs TD)
    # --------------------------------------------------------------------------
    ax_lc = fig.add_subplot(gs[0, 0])
    best_e_row = df_e.sort_values("final_window_mean", ascending=False).iloc[0]
    best_td_row = df_td.sort_values("final_window_mean", ascending=False).iloc[0]

    best_e_lbl = best_e_row["config"]
    best_td_lbl = best_td_row["config"]

    if best_e_lbl in curves_e:
        m_e, s_e = curves_e[best_e_lbl]
        x_e = np.arange(len(m_e))
        ax_lc.plot(x_e, m_e, label=f"Best E ({best_e_row['final_window_mean']:.2f})\n[{best_e_lbl}]",
                   color="#1b7837", linewidth=2.4)
        ax_lc.fill_between(x_e, m_e - s_e, m_e + s_e, color="#1b7837", alpha=0.18)

    if best_td_lbl in curves_td:
        m_td, s_td = curves_td[best_td_lbl]
        x_td = np.arange(len(m_td))
        ax_lc.plot(x_td, m_td, label=f"Best TD ({best_td_row['final_window_mean']:.2f})\n[{best_td_lbl}]",
                   color="#762a83", linewidth=2.4, linestyle="--")
        ax_lc.fill_between(x_td, m_td - s_td, m_td + s_td, color="#762a83", alpha=0.18)

    ax_lc.set_title(f"A. Best-in-Class Learning Curves: {env_name} (Mean ± 1 SEM)", fontweight="bold")
    ax_lc.set_xlabel("Environment Steps (Updates)", fontweight="bold")
    ax_lc.set_ylabel("Episode Return", fontweight="bold")
    ax_lc.grid(True, linestyle=":", alpha=0.6)
    ax_lc.legend(loc="lower right", framealpha=0.95, fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel 2: Critic Epochs Scaling (4 -> 16 -> 32)
    # --------------------------------------------------------------------------
    ax_ep = fig.add_subplot(gs[0, 1])
    ep_vals = sorted(df_e["critic_epochs"].unique())

    e_ep_means = [df_e[df_e["critic_epochs"] == ep]["final_window_mean"].mean() for ep in ep_vals]
    e_ep_sems = [df_e[df_e["critic_epochs"] == ep]["final_window_mean"].std() / np.sqrt(max(1, len(df_e[df_e["critic_epochs"] == ep]))) for ep in ep_vals]

    td_ep_means = [df_td[df_td["critic_epochs"] == ep]["final_window_mean"].mean() for ep in ep_vals]
    td_ep_sems = [df_td[df_td["critic_epochs"] == ep]["final_window_mean"].std() / np.sqrt(max(1, len(df_td[df_td["critic_epochs"] == ep]))) for ep in ep_vals]

    ax_ep.errorbar(ep_vals, e_ep_means, yerr=e_ep_sems, label="E_experimental",
                   color="#1b7837", marker="o", linewidth=2.2, capsize=4)
    ax_ep.errorbar(ep_vals, td_ep_means, yerr=td_ep_sems, label="Classic TD",
                   color="#762a83", marker="s", linewidth=2.2, linestyle="--", capsize=4)

    ax_ep.set_title("B. Critic Epochs Scaling (4 → 16 → 32)", fontweight="bold")
    ax_ep.set_xlabel("Critic Epochs per Policy Update", fontweight="bold")
    ax_ep.set_ylabel("Marginal Mean Return (across LR/WD/Heads)", fontweight="bold")
    ax_ep.set_xticks(ep_vals)
    ax_ep.grid(True, linestyle=":", alpha=0.6)
    ax_ep.legend(loc="best", framealpha=0.95)

    # --------------------------------------------------------------------------
    # Panel 3: Value Heads Scaling (1 vs 4)
    # --------------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[1, 0])
    h_vals = sorted(df_e["num_value_heads"].unique())
    x_pos = np.arange(len(h_vals))
    width = 0.35

    e_h_means = [df_e[df_e["num_value_heads"] == h]["final_window_mean"].mean() for h in h_vals]
    e_h_sems = [df_e[df_e["num_value_heads"] == h]["final_window_mean"].std() / np.sqrt(max(1, len(df_e[df_e["num_value_heads"] == h]))) for h in h_vals]

    td_h_means = [df_td[df_td["num_value_heads"] == h]["final_window_mean"].mean() for h in h_vals]
    td_h_sems = [df_td[df_td["num_value_heads"] == h]["final_window_mean"].std() / np.sqrt(max(1, len(df_td[df_td["num_value_heads"] == h]))) for h in h_vals]

    ax_h.bar(x_pos - width / 2, e_h_means, width, yerr=e_h_sems, label="E_experimental",
             color="#1b7837", edgecolor="#333", capsize=4)
    ax_h.bar(x_pos + width / 2, td_h_means, width, yerr=td_h_sems, label="Classic TD",
             color="#762a83", edgecolor="#333", capsize=4)

    ax_h.set_title("C. Value Heads Scaling (1 vs 4 Heads)", fontweight="bold")
    ax_h.set_xticks(x_pos)
    ax_h.set_xticklabels([f"{h} Head{'s' if h > 1 else ''}" for h in h_vals], fontweight="bold")
    ax_h.set_ylabel("Marginal Mean Return", fontweight="bold")
    ax_h.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax_h.legend(loc="best", framealpha=0.95)

    # --------------------------------------------------------------------------
    # Panel 4: Matched-Pair Scatter (E vs TD across all 36 configurations)
    # --------------------------------------------------------------------------
    ax_sc = fig.add_subplot(gs[1, 1])

    # Merge on the matching configuration parameters
    merge_cols = ["critic_lr", "critic_epochs", "weight_decay", "num_value_heads"]
    merged = pd.merge(df_e, df_td, on=merge_cols, suffixes=("_e", "_td"))

    y_e = merged["final_window_mean_e"].values
    x_td = merged["final_window_mean_td"].values

    mn = min(y_e.min(), x_td.min()) - 1.0
    mx = max(y_e.max(), x_td.max()) + 1.0

    ax_sc.plot([mn, mx], [mn, mx], "k--", alpha=0.6, label="y = x (Equality)")
    if "return_lambda" in merged.columns and merged["return_lambda"].nunique() > 1:
        u_lams = sorted(merged["return_lambda"].unique())
        colors = ["#1b7837", "#1a73e8", "#ea4335", "#fbbc04"]
        for i_lam, lam in enumerate(u_lams):
            sub = merged[merged["return_lambda"] == lam]
            ax_sc.scatter(sub["final_window_mean_td"], sub["final_window_mean_e"],
                          color=colors[i_lam % len(colors)], edgecolors="#333", s=45, alpha=0.85, zorder=4,
                          label=f"E (λ_ret={lam})")
    else:
        ax_sc.scatter(x_td, y_e, c="#1a73e8", edgecolors="#174ea6", s=50, alpha=0.85, zorder=4)

    e_wins = np.sum(y_e > x_td)
    td_wins = np.sum(x_td > y_e)
    ties = np.sum(y_e == x_td)
    total_pairs = len(y_e)

    diffs = y_e - x_td
    t_stat, p_val = stats.ttest_rel(y_e, x_td) if total_pairs > 1 else (0.0, 1.0)

    win_text = f"E Wins: {e_wins}/{total_pairs} ({100.0*e_wins/max(1, total_pairs):.1f}%)\nTD Wins: {td_wins}/{total_pairs}\nMean Δ(E - TD) = {diffs.mean():+.2f}\np = {p_val:.4f}"
    ax_sc.text(0.05, 0.95, win_text, transform=ax_sc.transAxes, va="top", ha="left",
               fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#dadce0"))

    ax_sc.set_title("D. Matched-Pair Grid Scatter (E vs TD)", fontweight="bold")
    ax_sc.set_xlabel("Classic TD Return", fontweight="bold")
    ax_sc.set_ylabel("E_experimental Return", fontweight="bold")
    ax_sc.set_xlim(mn, mx)
    ax_sc.set_ylim(mn, mx)
    ax_sc.grid(True, linestyle=":", alpha=0.6)
    ax_sc.legend(loc="lower right", framealpha=0.95)

    fig.suptitle(f"Head-to-Head Benchmark: E_experimental vs. Classic TD ({env_name})",
                 fontsize=15, fontweight="bold", y=0.98)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # Save head-to-head summary CSV
    merged["delta_e_minus_td"] = merged["final_window_mean_e"] - merged["final_window_mean_td"]
    merged["winner"] = np.where(merged["delta_e_minus_td"] > 0, "E_experimental", "Classic TD")
    merged_summary_path = os.path.join(out_dir, "head_to_head_summary.csv")
    merged.to_csv(merged_summary_path, index=False)

    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD COMPARISON COMPLETE:")
    print(f"  - E_experimental Best: {best_e_row['final_window_mean']:.2f} ({best_e_lbl})")
    print(f"  - Classic TD Best:     {best_td_row['final_window_mean']:.2f} ({best_td_lbl})")
    print(f"  - E Win Rate:          {e_wins}/{total_pairs} ({100.0*e_wins/max(1, total_pairs):.1f}%)")
    print(f"  - Mean Delta (E - TD): {diffs.mean():+.2f}")
    print(f"  - Poster PDF:          {pdf_path}")
    print(f"  - Summary Table:       {merged_summary_path}")
    print(f"{'='*70}\n")


def build_e_grid(args):
    """Builds the Cartesian product of (critic_lr, epochs, weight_decay, num_value_heads, return_lambda)."""
    ret_lams = [args.return_lambda] if args.return_lambda is not None else args.return_lambda_grid
    grid = []
    for clr, ep, wd, heads, ret_lam in itertools.product(
        args.critic_lr_grid, args.epochs_grid, args.wd_grid, args.heads_grid, ret_lams
    ):
        if len(ret_lams) > 1:
            label = f"lr={clr}_ep={ep}_wd={wd}_heads={heads}_retlam={ret_lam}"
        else:
            label = f"lr={clr}_ep={ep}_wd={wd}_heads={heads}"
        grid.append({
            "label": label,
            "critic_lr": clr,
            "critic_epochs": ep,
            "weight_decay": wd,
            "num_value_heads": heads,
            "return_lambda": ret_lam,
        })
    return grid


def build_td_grid(args):
    """Builds the Cartesian product for Classic TD (which does not depend on return_lambda)."""
    grid = []
    for clr, ep, wd, heads in itertools.product(
        args.critic_lr_grid, args.epochs_grid, args.wd_grid, args.heads_grid
    ):
        label = f"lr={clr}_ep={ep}_wd={wd}_heads={heads}"
        grid.append({
            "label": label,
            "critic_lr": clr,
            "critic_epochs": ep,
            "weight_decay": wd,
            "num_value_heads": heads,
            "td_lambda": args.td_lambda,
        })
    return grid


def main():
    args = parse_args()

    # Determine default timesteps if not provided
    if args.total_timesteps is None:
        if "MinAtar" in args.env_name:
            args.total_timesteps = 10_000_000
        else:
            args.total_timesteps = 2_048_000

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        suite_id = (
            args.sweep_id
            or os.environ.get("SWEEP_ID")
            or (f"cmp_td_vs_e_{os.environ['SLURM_ARRAY_JOB_ID']}" if "SLURM_ARRAY_JOB_ID" in os.environ else None)
            or (f"cmp_td_vs_e_{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else None)
            or f"cmp_td_vs_e_{timestamp}"
        )
        out_dir = os.path.join(_repo_root, "results", "ppo", "sweeps", suite_id, args.env_name)

    os.makedirs(out_dir, exist_ok=True)
    grid_e = build_e_grid(args)
    grid_td = build_td_grid(args)

    print("=" * 70)
    print("STANDARD COMPARISON SWEEP: CLASSIC TD vs E_EXPERIMENTAL")
    print(f"Environment:       {args.env_name}")
    print(f"Timesteps:         {args.total_timesteps:,} | Seeds: {args.n_seeds}")
    print(f"Critic LR Grid:    {args.critic_lr_grid}")
    print(f"Critic Epochs:     {args.epochs_grid}")
    print(f"Weight Decay:      {args.wd_grid}")
    print(f"Value Heads:       {args.heads_grid}")
    ret_lams_info = [args.return_lambda] if args.return_lambda is not None else args.return_lambda_grid
    print(f"Return Lambdas:    {ret_lams_info} (E_experimental)")
    print(f"Configs:           E_experimental: {len(grid_e)} | Classic TD: {len(grid_td)}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 70)

    # 1. Sweep E_experimental
    e_dir = os.path.join(out_dir, "E_experimental")
    df_e, curves_e, _ = run_single_algo_sweep(
        algo_name="E_experimental",
        make_train_fn=make_train_e,
        grid=grid_e,
        args=args,
        out_algo_dir=e_dir,
    )

    # 2. Sweep Classic TD
    td_dir = os.path.join(out_dir, "td")
    df_td, curves_td, _ = run_single_algo_sweep(
        algo_name="td",
        make_train_fn=make_train_td,
        grid=grid_td,
        args=args,
        out_algo_dir=td_dir,
    )

    # 3. Generate Head-to-Head Comparison Poster and Table
    plot_head_to_head_poster(
        df_e=df_e,
        curves_e=curves_e,
        df_td=df_td,
        curves_td=curves_td,
        out_dir=out_dir,
        env_name=args.env_name,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()

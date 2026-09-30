#!/usr/bin/env python3
"""
sweep_td_vs_e_lambda.py

Standard Head-to-Head Comparison Sweep:
E(lambda) (algos/E_lambda_experimental.py) vs. TD(lambda) (algos/td.py)
across the multi-dimensional grid:
    - Lambda: [0.0, 0.8, 0.95]           (3 values)
    - Critic Epochs: [4, 16, 32]          (3 values)
    - Critic LR: [0.0003, 0.001]          (2 values)
    - Value Heads: [1, 4]                 (2 values)
    - Weight Decay: [0.001, 0.01]         (2 values, configurable via --wd-grid)

Total Configurations per algorithm: 3 x 3 x 2 x 2 x 2 = 72 configurations (or 36 if single WD).
Total Evaluated across both: 144 configurations (or 72).

Environment Scaling:
    - MinAtar: 10M timesteps, 256 parallel envs, k=64
    - Classic Control / Misc: 2,048,000 timesteps, 64 parallel envs, k=32

Outputs:
    - <out_dir>/td/: summary_td_lambda.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/E_lambda_experimental/: summary_e_lambda.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/head_to_head_lambda_summary.csv: pairwise matching table (E vs TD)
    - <out_dir>/head_to_head_lambda_poster.pdf & .png: 4-panel comparison poster:
        1. Learning Curves of Best E(lambda) vs. Best TD(lambda) (Mean ± 1 SEM)
        2. Lambda Scaling: Performance vs Lambda [0.0, 0.8, 0.95] for E vs TD
        3. Epochs Scaling: Performance vs Critic Epochs [4, 16, 32] for E vs TD
        4. Matched-Pair Scatter: E(lambda) vs TD(lambda) with overall win rates

Usage:
    python scripts/sweep_td_vs_e_lambda.py --env-name CartPole-v1 --n-seeds 8
    python scripts/sweep_td_vs_e_lambda.py --env-name Asterix-MinAtar --n-seeds 8
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
from algos.E_lambda_experimental import make_train as make_train_e_lambda
from algos.td import make_train as make_train_td
from scripts.e_optimization.visualize_multidim_sweep import generate_multidim_analysis_pdf

# TrueType font embedding for vector publication quality
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Head-to-head comparison sweep: E(lambda) vs TD(lambda)")
    parser.add_argument("--env-name", type=str, default="MountainCarContinuous-v0",
                        help="Gymnax environment name")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total timesteps (default: 2,048,000 for classic, 10M for MinAtar)")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Number of independent random seeds (default: 8)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (default: 256 for MinAtar, 64 for classic)")
    parser.add_argument("--num-steps", type=int, default=64,
                        help="Number of rollout steps per env (default: 64)")
    parser.add_argument("--minibatch-size", type=int, default=1024,
                        help="Minibatch size for SGD updates (default: 1024)")
    parser.add_argument("--k-dim", type=int, default=None,
                        help="Feature/hidden dimension k (default: 64 for MinAtar, 32 for classic)")
    parser.add_argument("--actor-lr", type=float, default=0.0003,
                        help="Actor learning rate held constant (default: 0.0003)")
    parser.add_argument("--actor-epochs", type=int, default=4,
                        help="Actor update epochs held constant (default: 4)")

    # Grids
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.8, 0.95],
                        help="Lambda grid values (default: 0.0 0.8 0.95)")
    parser.add_argument("--critic-lr-grid", type=float, nargs="+", default=[0.0003, 0.001],
                        help="Critic learning rate grid (default: 0.0003 0.001)")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[4, 16, 32],
                        help="Critic epochs grid (default: 4 16 32)")
    parser.add_argument("--wd-grid", type=float, nargs="+", default=[0.001, 0.01],
                        help="Critic weight decay grid (default: 0.001 0.01)")
    parser.add_argument("--heads-grid", type=int, nargs="+", default=[1, 4],
                        help="Value heads grid (default: 1 4)")

    parser.add_argument("--critic-loss-type", type=str, default="mse",
                        help="Critic loss type for both algos (default: mse)")
    parser.add_argument("--return-lambda", type=float, default=0.99,
                        help="Return anchor lambda for E(lambda) (default: 0.99)")

    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Unified sweep identifier across SLURM array tasks")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for final mean evaluation (default: 100)")
    parser.add_argument("--metric", type=str, default="returned_episode_returns",
                        help="Primary evaluation metric (default: returned_episode_returns)")
    return parser.parse_args()


def build_comparison_grid(args):
    """Builds the full Cartesian product grid for lambda comparison."""
    grid = []
    for lmbda, ep, clr, heads, wd in itertools.product(
        args.lambda_grid,
        args.epochs_grid,
        args.critic_lr_grid,
        args.heads_grid,
        args.wd_grid,
    ):
        label = f"lmbda_{lmbda}_ep_{ep}_clr_{clr}_heads_{heads}_wd_{wd}"
        grid.append({
            "label": label,
            "lambda": float(lmbda),
            "critic_epochs": int(ep),
            "critic_lr": float(clr),
            "num_value_heads": int(heads),
            "weight_decay": float(wd),
        })
    return grid


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
        lmbda = item["lambda"]
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
        cfg["k"] = args.k_dim
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

        # Set algorithmic lambda parameters
        cfg["TD_LAMBDA"] = lmbda
        cfg["E_LAMBDA"] = lmbda
        cfg["VALUE_LAMBDA"] = lmbda
        cfg["RETURN_LAMBDA"] = args.return_lambda
        cfg["GAE_LAMBDA"] = 0.8
        cfg["CLIP_EPS"] = 0.2

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

        w = min(args.window_size, len(mean_c))
        final_mean = float(np.mean(mean_c[-w:]))
        final_sem = float(np.mean(sem_c[-w:]))
        auc = float(np.sum(mean_c))

        curves_dict[label] = {
            "mean": mean_c,
            "sem": sem_c,
            "raw": tensor,
            "final_mean": final_mean,
            "final_sem": final_sem,
            "auc": auc,
        }

        # Light metrics collection for pkl
        scalar_metrics = {}
        for k_m, v_m in out["metrics"].items():
            arr_m = np.asarray(v_m)
            if arr_m.ndim >= 2:
                scalar_metrics[k_m] = arr_m.mean(axis=0)
        all_metrics[label] = scalar_metrics

        summary_rows.append({
            "algo": algo_name,
            "label": label,
            "lambda": lmbda,
            "critic_epochs": ep,
            "critic_lr": clr,
            "num_value_heads": heads,
            "weight_decay": wd,
            "final_window_mean": final_mean,
            "final_window_sem": final_sem,
            "auc": auc,
        })

    df = pd.DataFrame(summary_rows)
    df.sort_values(by="final_window_mean", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save summary CSV
    if "E_lambda" in algo_name:
        csv_filename = "summary_e_lambda.csv"
    elif "td" in algo_name.lower():
        csv_filename = "summary_td_lambda.csv"
    else:
        csv_filename = f"summary_{algo_name.lower()}.csv"

    csv_path = os.path.join(out_algo_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"Saved {algo_name} summary: {csv_path}")

    # Save metrics pkl
    pkl_path = os.path.join(out_algo_dir, "metrics.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"curves": curves_dict, "summary_df": df, "metrics": all_metrics}, f)

    # Automatically generate multi-dimensional diagnostic report
    try:
        generate_multidim_analysis_pdf(out_algo_dir)
    except Exception as ex:
        print(f"Warning: Could not auto-generate multidim pdf for {algo_name}: {ex}")

    return df, curves_dict, all_metrics


def build_head_to_head_poster(df_e, curves_e, df_td, curves_td, env_name, out_dir):
    """Constructs a 4-panel comparison poster comparing E(lambda) and TD(lambda)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    colors = {"E": "#1f77b4", "TD": "#d62728"}

    # ----------------------------------------------------
    # Panel (0, 0): Best Learning Curves (Mean ± 1 SEM)
    # ----------------------------------------------------
    ax0 = axes[0, 0]
    best_e_lbl = df_e.iloc[0]["label"]
    best_td_lbl = df_td.iloc[0]["label"]

    c_e = curves_e[best_e_lbl]
    c_td = curves_td[best_td_lbl]

    xs_e = np.arange(len(c_e["mean"]))
    xs_td = np.arange(len(c_td["mean"]))

    ax0.plot(xs_e, c_e["mean"], label=f"Best E(λ): {best_e_lbl}", color=colors["E"], lw=2.5)
    ax0.fill_between(xs_e, c_e["mean"] - c_e["sem"], c_e["mean"] + c_e["sem"], color=colors["E"], alpha=0.2)

    ax0.plot(xs_td, c_td["mean"], label=f"Best TD(λ): {best_td_lbl}", color=colors["TD"], lw=2.5, linestyle="--")
    ax0.fill_between(xs_td, c_td["mean"] - c_td["sem"], c_td["mean"] + c_td["sem"], color=colors["TD"], alpha=0.2)

    ax0.set_title(f"Best Learning Curves: {env_name}", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Updates", fontweight="bold")
    ax0.set_ylabel("Episode Return", fontweight="bold")
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    # ----------------------------------------------------
    # Panel (0, 1): Lambda Scaling Comparison [0.0, 0.8, 0.95]
    # ----------------------------------------------------
    ax1 = axes[0, 1]
    lmbda_vals = sorted(df_e["lambda"].unique())
    x_pos = np.arange(len(lmbda_vals))
    width = 0.35

    e_lmbda_means = [df_e[df_e["lambda"] == lv]["final_window_mean"].mean() for lv in lmbda_vals]
    e_lmbda_sems = [df_e[df_e["lambda"] == lv]["final_window_mean"].sem() for lv in lmbda_vals]

    td_lmbda_means = [df_td[df_td["lambda"] == lv]["final_window_mean"].mean() for lv in lmbda_vals]
    td_lmbda_sems = [df_td[df_td["lambda"] == lv]["final_window_mean"].sem() for lv in lmbda_vals]

    ax1.bar(x_pos - width / 2, e_lmbda_means, width, yerr=e_lmbda_sems, capsize=4,
            label="E(λ)", color=colors["E"], alpha=0.85)
    ax1.bar(x_pos + width / 2, td_lmbda_means, width, yerr=td_lmbda_sems, capsize=4,
            label="TD(λ)", color=colors["TD"], alpha=0.85)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"λ={lv}" for lv in lmbda_vals], fontweight="bold")
    ax1.set_title("Lambda Scaling: Marginal Performance by λ", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Final Window Mean Return", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax1.legend(loc="upper left", fontsize=10)

    # ----------------------------------------------------
    # Panel (1, 0): Epochs Scaling (4 -> 16 -> 32)
    # ----------------------------------------------------
    ax2 = axes[1, 0]
    epoch_vals = sorted(df_e["critic_epochs"].unique())
    e_ep_means = [df_e[df_e["critic_epochs"] == ep]["final_window_mean"].mean() for ep in epoch_vals]
    e_ep_sems = [df_e[df_e["critic_epochs"] == ep]["final_window_mean"].sem() for ep in epoch_vals]

    td_ep_means = [df_td[df_td["critic_epochs"] == ep]["final_window_mean"].mean() for ep in epoch_vals]
    td_ep_sems = [df_td[df_td["critic_epochs"] == ep]["final_window_mean"].sem() for ep in epoch_vals]

    ax2.errorbar(epoch_vals, e_ep_means, yerr=e_ep_sems, fmt="-o", color=colors["E"],
                 lw=2.5, capsize=5, label="E(λ)")
    ax2.errorbar(epoch_vals, td_ep_means, yerr=td_ep_sems, fmt="--s", color=colors["TD"],
                 lw=2.5, capsize=5, label="TD(λ)")

    ax2.set_xticks(epoch_vals)
    ax2.set_title("Epochs Scaling (Replay Ratio): Critic Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Critic Epochs per Rollout", fontweight="bold")
    ax2.set_ylabel("Final Window Mean Return", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=10)

    # ----------------------------------------------------
    # Panel (1, 1): Matched-Pair Scatter (E vs TD)
    # ----------------------------------------------------
    ax3 = axes[1, 1]
    # Merge on configuration parameters
    merge_cols = ["lambda", "critic_epochs", "critic_lr", "num_value_heads", "weight_decay"]
    matched = pd.merge(df_e, df_td, on=merge_cols, suffixes=("_E", "_TD"))

    e_scores = matched["final_window_mean_E"].values
    td_scores = matched["final_window_mean_TD"].values

    mn = min(np.min(e_scores), np.min(td_scores))
    mx = max(np.max(e_scores), np.max(td_scores))
    pad = (mx - mn) * 0.05
    line_x = np.linspace(mn - pad, mx + pad, 100)

    ax3.plot(line_x, line_x, "k--", alpha=0.6, label="Parity Line (y = x)")
    ax3.scatter(td_scores, e_scores, color="#2ca02c", edgecolors="k", s=60, alpha=0.85, zorder=4)

    e_wins = np.sum(e_scores > td_scores)
    td_wins = np.sum(td_scores > e_scores)
    ties = np.sum(e_scores == td_scores)
    total_pairs = len(matched)
    win_pct = (e_wins / total_pairs) * 100.0 if total_pairs > 0 else 0.0

    ax3.set_title(f"Matched Configurations: E(λ) vs. TD(λ) ({total_pairs} pairs)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("TD(λ) Return", fontweight="bold")
    ax3.set_ylabel("E(λ) Return", fontweight="bold")
    ax3.set_xlim(mn - pad, mx + pad)
    ax3.set_ylim(mn - pad, mx + pad)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # Annotation box
    text_str = f"E(λ) Wins: {e_wins} ({win_pct:.1f}%)\nTD(λ) Wins: {td_wins}\nTies: {ties}"
    ax3.text(0.05, 0.95, text_str, transform=ax3.transAxes, fontsize=10.5,
             verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"))

    fig.suptitle(f"Head-to-Head Comparison: E(λ) vs. TD(λ) — {env_name}",
                 fontsize=15, fontweight="bold", y=0.98)

    pdf_path = os.path.join(out_dir, "head_to_head_lambda_poster.pdf")
    png_path = os.path.join(out_dir, "head_to_head_lambda_poster.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated comparison poster:\n  - {pdf_path}\n  - {png_path}")

    # Save matched pairwise CSV table
    matched["delta_E_minus_TD"] = matched["final_window_mean_E"] - matched["final_window_mean_TD"]
    matched_csv = os.path.join(out_dir, "head_to_head_lambda_summary.csv")
    matched.sort_values(by="delta_E_minus_TD", ascending=False, inplace=True)
    matched.to_csv(matched_csv, index=False)
    print(f"Saved matched-pair summary: {matched_csv}")


def main():
    args = parse_args()

    # Dynamic environment-specific scaling
    if "MinAtar" in args.env_name:
        if args.total_timesteps is None:
            args.total_timesteps = 10_000_000
        if args.num_envs is None:
            args.num_envs = 256
        if args.k_dim is None:
            args.k_dim = 64
    else:
        if args.total_timesteps is None:
            args.total_timesteps = 2_048_000
        if args.num_envs is None:
            args.num_envs = 64
        if args.k_dim is None:
            args.k_dim = 32

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        suite_id = (
            args.sweep_id
            or os.environ.get("SWEEP_ID")
            or (f"cmp_td_vs_e_lambda_{os.environ['SLURM_ARRAY_JOB_ID']}" if "SLURM_ARRAY_JOB_ID" in os.environ else None)
            or (f"cmp_td_vs_e_lambda_{os.environ['SLURM_JOB_ID']}" if "SLURM_JOB_ID" in os.environ else None)
            or f"cmp_td_vs_e_lambda_{timestamp}"
        )
        out_dir = os.path.join(_repo_root, "results", "ppo", "sweeps", suite_id, args.env_name)

    os.makedirs(out_dir, exist_ok=True)
    grid = build_comparison_grid(args)

    print("=" * 70)
    print("STANDARD COMPARISON SWEEP: E(lambda) vs TD(lambda)")
    print(f"Environment:       {args.env_name}")
    print(f"Timesteps:         {args.total_timesteps:,} | Parallel Envs: {args.num_envs} | k: {args.k_dim}")
    print(f"Seeds:             {args.n_seeds}")
    print(f"Lambda Grid:       {args.lambda_grid}")
    print(f"Critic Epochs:     {args.epochs_grid}")
    print(f"Critic LR Grid:    {args.critic_lr_grid}")
    print(f"Weight Decay:      {args.wd_grid}")
    print(f"Value Heads:       {args.heads_grid}")
    print(f"Configs per Algo:  {len(grid)} (Total evaluations: {2 * len(grid)})")
    print(f"Output Directory:  {out_dir}")
    print("=" * 70)

    # 1. Sweep E_lambda_experimental
    e_dir = os.path.join(out_dir, "E_lambda_experimental")
    df_e, curves_e, _ = run_single_algo_sweep(
        algo_name="E_lambda_experimental",
        make_train_fn=make_train_e_lambda,
        grid=grid,
        args=args,
        out_algo_dir=e_dir,
    )

    # 2. Sweep TD(lambda)
    td_dir = os.path.join(out_dir, "td")
    df_td, curves_td, _ = run_single_algo_sweep(
        algo_name="td",
        make_train_fn=make_train_td,
        grid=grid,
        args=args,
        out_algo_dir=td_dir,
    )

    # 3. Build Head-to-Head Comparison Poster and Matching Table
    build_head_to_head_poster(df_e, curves_e, df_td, curves_td, args.env_name, out_dir)


if __name__ == "__main__":
    main()

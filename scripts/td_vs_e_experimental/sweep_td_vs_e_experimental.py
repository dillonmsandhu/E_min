#!/usr/bin/env python3
"""
sweep_td_vs_e_experimental.py

Comprehensive 4-Way Critic Algorithm Comparison Sweep:
1. E(0): Symmetrized 1-step E-minimization (algos/E_experimental.py)
2. TD(0): Classic 1-step online TD learning (algos/td.py with td_lambda=0.0)
3. E(lambda): Geometric jump sampled E(lambda) (algos/E_lambda_experimental.py, lambda=0.9)
4. TD(lambda): Standard PPO fitted T^lambda regression (algos/td.py with td_lambda=0.9, fitted)

Swept across the identical multi-dimensional critic grid:
    - Critic LR: [0.0003, 0.001, 0.003]  (3 values)
    - Critic Epochs: [4, 16, 32]          (3 values)
    - Weight Decay: [0.001, 0.01]         (2 values)
    - Value Heads: [1, 4]                 (2 values)

Total Configurations per algorithm: 3 x 3 x 2 x 2 = 36 configurations.
Total Evaluated across all 4 algorithms: 144 configurations (vmapped over seeds in JAX).

Outputs:
    - <out_dir>/E_0/: summary_e_0.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/TD_0/: summary_td_0.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/E_lambda/: summary_e_lambda.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/TD_lambda/: summary_td_lambda.csv, metrics.pkl, multidim_analysis.pdf
    - <out_dir>/all_algos_summary.csv: All configurations across all 4 algorithms
    - <out_dir>/auc_summary.md: Markdown table of AUC & Final Return + Pairwise matchups
    - <out_dir>/matched_pairs_e0_vs_td0.csv: Pairwise matching table (E(0) vs TD(0))
    - <out_dir>/matched_pairs_elambda_vs_tdlambda.csv: Pairwise matching table (E(lambda) vs TD(lambda))
    - <out_dir>/head_to_head_summary.csv: Combined pairwise matching table
    - <out_dir>/head_to_head_poster.pdf & .png: Publication-grade comparison poster:
        1. Learning Curves of Best-in-Class for all 4 algorithms (Mean +/- 1 SEM)
        2. Critic Epochs Scaling (4 -> 16 -> 32) comparing all 4 algorithms
        3. Value Heads Scaling (1 vs 4 heads) comparing all 4 algorithms
        4. Matched-Pair Scatters: E(0) vs TD(0) and E(lambda) vs TD(lambda)

Usage:
    python scripts/td_vs_e_experimental/sweep_td_vs_e_experimental.py --env-name CartPole-v1 --n-seeds 8
    python scripts/td_vs_e_experimental/sweep_td_vs_e_experimental.py --env-name MountainCarContinuous-v0 --n-seeds 8
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
from algos.E_experimental import make_train as make_train_e_0
from algos.td import make_train as make_train_td
from algos.E_lambda_experimental import make_train as make_train_e_lambda
from algos.ppo import make_train as make_train_ppo
from scripts.e_optimization.visualize_multidim_sweep import generate_multidim_analysis_pdf

# TrueType font embedding for vector publication quality
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10

# Color palette for 4 algorithms
ALGO_PALETTE = {
    "E_0": "#1b7837",        # Dark Forest Green
    "TD_0": "#762a83",       # Dark Purple
    "E_lambda": "#1a73e8",   # Vibrant Blue
    "TD_lambda": "#d95f02",  # Vivid Orange / Ochre
}

ALGO_PRETTY_NAMES = {
    "E_0": "E(0)",
    "TD_0": "TD(0)",
    "E_lambda": "E(lambda)",
    "TD_lambda": "TD(lambda)",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive 4-way comparison sweep: E(0), TD(0), E(lambda), and TD(lambda)"
    )
    parser.add_argument("--env-name", type=str, default="MountainCarContinuous-v0",
                        help="Gymnax environment name (default: MountainCarContinuous-v0)")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total environment timesteps (default: 2,048,000 for classic, 10M for MinAtar)")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Number of independent random seeds (default: 8)")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments (default: 64)")
    parser.add_argument("--num-steps", type=int, default=256,
                        help="Number of rollout steps per env (default: 256)")
    parser.add_argument("--minibatch-size", type=int, default=1024,
                        help="Minibatch size for SGD updates (default: 1024)")
    parser.add_argument("--actor-lr", type=float, default=0.0003,
                        help="Actor learning rate held constant (default: 0.0003)")
    parser.add_argument("--actor-epochs", type=int, default=4,
                        help="Actor update epochs held constant (default: 4)")

    # Grids evaluated identically across all algorithms
    parser.add_argument("--critic-lr-grid", type=float, nargs="+", default=[0.0003, 0.001, 0.003],
                        help="Critic learning rate grid (default: 0.0003 0.001 0.003)")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[4, 16, 32],
                        help="Critic epochs grid (default: 4 16 32)")
    parser.add_argument("--wd-grid", type=float, nargs="+", default=[0.001, 0.01],
                        help="Critic weight decay grid (default: 0.001 0.01)")
    parser.add_argument("--heads-grid", type=int, nargs="+", default=[1, 4],
                        help="Value heads grid (default: 1 4)")
    parser.add_argument("--layer-norm-grid", type=str2bool, nargs="+", default=[False, True],
                        help="Layer norm grid for policy and value networks (default: False True)")
    parser.add_argument("--layer-norm", type=str2bool, default=None,
                        help="Single layer norm boolean (overrides --layer-norm-grid if specified)")

    # Algorithm hyperparameters
    parser.add_argument("--lambda-val", type=float, default=0.9,
                        help="Fixed lambda parameter for E(lambda) and TD(lambda) (default: 0.9)")
    parser.add_argument("--return-lambda-grid", type=float, nargs="+", default=[0.95, 1.0],
                        help="Return anchor lambda grid for E(0) and E(lambda) (default: 0.95 1.0)")
    parser.add_argument("--return-lambda", type=float, default=None,
                        help="Single return anchor lambda parameter (overrides --return-lambda-grid if specified)")
    parser.add_argument("--critic-loss-type", type=str, default="mse",
                        help="Critic loss type for all algorithms (default: mse)")

    # Algorithm selection
    parser.add_argument("--algos", type=str, nargs="+", default=["E_0", "TD_0", "E_lambda", "TD_lambda"],
                        choices=["E_0", "TD_0", "E_lambda", "TD_lambda"],
                        help="Algorithms to evaluate (default: E_0 TD_0 E_lambda TD_lambda)")

    # Run metadata & logging
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Unified sweep identifier across SLURM array tasks")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for final mean evaluation (default: 100)")
    parser.add_argument("--metric", type=str, default="returned_episode_returns",
                        help="Primary evaluation metric (default: returned_episode_returns)")
    return parser.parse_args()


def build_algo_grid(algo_name, args):
    """
    Builds the hyperparameter grid for an algorithm.
    - For E(0) and E(lambda): sweeps over (critic_lr, critic_epochs, weight_decay, num_value_heads, layer_norm, return_lambda).
    - For TD(0) and TD(lambda): sweeps over (critic_lr, critic_epochs, weight_decay, num_value_heads, layer_norm).
    """
    grid = []
    ret_lambda_list = [args.return_lambda] if args.return_lambda is not None else args.return_lambda_grid
    ln_list = [args.layer_norm] if args.layer_norm is not None else args.layer_norm_grid

    if algo_name in ["E_0", "E_lambda"]:
        for clr, ep, wd, heads, ln, ret_lam in itertools.product(
            args.critic_lr_grid, args.epochs_grid, args.wd_grid, args.heads_grid, ln_list, ret_lambda_list
        ):
            label = f"lr={clr}_ep={ep}_wd={wd}_heads={heads}_ln={ln}_rlam={ret_lam}"
            item = {
                "label": label,
                "critic_lr": clr,
                "critic_epochs": ep,
                "weight_decay": wd,
                "num_value_heads": heads,
                "layer_norm": ln,
                "return_lambda": ret_lam,
            }
            if algo_name == "E_lambda":
                item["e_lambda"] = args.lambda_val
                item["value_lambda"] = args.lambda_val
            grid.append(item)
    else:
        for clr, ep, wd, heads, ln in itertools.product(
            args.critic_lr_grid, args.epochs_grid, args.wd_grid, args.heads_grid, ln_list
        ):
            label = f"lr={clr}_ep={ep}_wd={wd}_heads={heads}_ln={ln}"
            item = {
                "label": label,
                "critic_lr": clr,
                "critic_epochs": ep,
                "weight_decay": wd,
                "num_value_heads": heads,
                "layer_norm": ln,
            }
            if algo_name == "TD_0":
                item["td_lambda"] = 0.0
            elif algo_name == "TD_lambda":
                item["td_lambda"] = args.lambda_val
                item["value_lambda"] = args.lambda_val
            grid.append(item)
    return grid


def run_single_algo_sweep(algo_name, make_train_fn, grid, args, out_algo_dir):
    """Executes the full grid for a single algorithm across all random seeds."""
    os.makedirs(out_algo_dir, exist_ok=True)
    summary_rows = []
    curves_dict = {}
    all_metrics = {}

    display_name = ALGO_PRETTY_NAMES.get(algo_name, algo_name)
    print(f"\n{'='*70}")
    print(f"RUNNING SWEEP FOR ALGORITHM: {display_name} ({algo_name})")
    print(f"Total Configurations: {len(grid)} | Seeds: {args.n_seeds}")
    print(f"Output Directory: {out_algo_dir}")
    print(f"{'='*70}")

    for idx, item in enumerate(grid, 1):
        label = item["label"]
        clr = item["critic_lr"]
        ep = item["critic_epochs"]
        wd = item["weight_decay"]
        heads = item["num_value_heads"]

        print(f"[{idx:02d}/{len(grid)}] {display_name} | {label} ...")

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
        cfg["LAYER_NORM"] = item.get("layer_norm", False)

        if "td_lambda" in item:
            cfg["TD_LAMBDA"] = item["td_lambda"]
        if "e_lambda" in item:
            cfg["E_LAMBDA"] = item["e_lambda"]
        if "value_lambda" in item:
            cfg["VALUE_LAMBDA"] = item["value_lambda"]
        if "return_lambda" in item:
            cfg["RETURN_LAMBDA"] = item["return_lambda"]
        if "recompute_targets_each_epoch" in item:
            cfg["RECOMPUTE_TARGETS_EACH_EPOCH"] = item["recompute_targets_each_epoch"]

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
            "display_name": display_name,
            "config": label,
            "critic_lr": clr,
            "critic_epochs": ep,
            "weight_decay": wd,
            "num_value_heads": heads,
            "layer_norm": item.get("layer_norm", False),
            "auc": auc,
            "final_window_mean": final_mean,
            "final_window_sem": final_sem,
            "elapsed_seconds": elapsed,
        }
        if "return_lambda" in item:
            row["return_lambda"] = item["return_lambda"]
        if "td_lambda" in item:
            row["td_lambda"] = item["td_lambda"]
        if "e_lambda" in item:
            row["e_lambda"] = item["e_lambda"]
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


def compute_pairwise_comparison(df_e, df_td, label_e="E", label_td="TD"):
    """
    Computes matched-pair comparison between E and TD across identical configuration labels.
    Returns merged DataFrame and summary stats dictionary.
    """
    merge_cols = ["critic_lr", "critic_epochs", "weight_decay", "num_value_heads", "layer_norm"]
    merged = pd.merge(df_e, df_td, on=merge_cols, suffixes=("_e", "_td"))

    y_e_final = merged["final_window_mean_e"].values
    x_td_final = merged["final_window_mean_td"].values

    y_e_auc = merged["auc_e"].values
    x_td_auc = merged["auc_td"].values

    n_pairs = len(merged)
    if n_pairs == 0:
        return merged, {
            "n_pairs": 0,
            "e_wins_final": 0, "td_wins_final": 0, "ties_final": 0,
            "e_win_rate_final": 0.0, "mean_delta_final": 0.0, "p_val_final": 1.0,
            "e_wins_auc": 0, "td_wins_auc": 0, "ties_auc": 0,
            "e_win_rate_auc": 0.0, "mean_delta_auc": 0.0, "p_val_auc": 1.0,
        }

    # Final return comparison
    e_wins_final = int(np.sum(y_e_final > x_td_final))
    td_wins_final = int(np.sum(x_td_final > y_e_final))
    ties_final = int(np.sum(y_e_final == x_td_final))
    diff_final = y_e_final - x_td_final
    _, p_final = stats.ttest_rel(y_e_final, x_td_final) if n_pairs > 1 else (0.0, 1.0)

    # AUC comparison
    e_wins_auc = int(np.sum(y_e_auc > x_td_auc))
    td_wins_auc = int(np.sum(x_td_auc > y_e_auc))
    ties_auc = int(np.sum(y_e_auc == x_td_auc))
    diff_auc = y_e_auc - x_td_auc
    _, p_auc = stats.ttest_rel(y_e_auc, x_td_auc) if n_pairs > 1 else (0.0, 1.0)

    merged["delta_auc"] = diff_auc
    merged["delta_final_mean"] = diff_final
    merged["winner_auc"] = np.where(diff_auc > 0, label_e, np.where(diff_auc < 0, label_td, "Tie"))
    merged["winner_final"] = np.where(diff_final > 0, label_e, np.where(diff_final < 0, label_td, "Tie"))

    stats_dict = {
        "n_pairs": n_pairs,
        "e_wins_final": e_wins_final,
        "td_wins_final": td_wins_final,
        "ties_final": ties_final,
        "e_win_rate_final": 100.0 * e_wins_final / max(1, n_pairs),
        "mean_delta_final": float(np.mean(diff_final)),
        "sem_delta_final": float(np.std(diff_final) / np.sqrt(max(1, n_pairs))),
        "p_val_final": float(p_final) if not np.isnan(p_final) else 1.0,
        "e_wins_auc": e_wins_auc,
        "td_wins_auc": td_wins_auc,
        "ties_auc": ties_auc,
        "e_win_rate_auc": 100.0 * e_wins_auc / max(1, n_pairs),
        "mean_delta_auc": float(np.mean(diff_auc)),
        "sem_delta_auc": float(np.std(diff_auc) / np.sqrt(max(1, n_pairs))),
        "p_val_auc": float(p_auc) if not np.isnan(p_auc) else 1.0,
    }
    return merged, stats_dict


def generate_markdown_summary(env_name, results_dict, pairwise_0, pairwise_lambda, args):
    """
    Constructs a comprehensive, publication-grade markdown table showing AUC and Final Return
    for each algorithm, followed by pairwise comparisons for E(0) vs TD(0) and E(lambda) vs TD(lambda).
    """
    md = []
    md.append(f"# 4-Way Critic Algorithm Benchmark: `{env_name}`\n")
    md.append(f"**Environment:** `{env_name}` | **Seeds:** {args.n_seeds} | **Timesteps:** {args.total_timesteps:,}")
    md.append(f"**Critic Lambda:** `{args.lambda_val}` | **Return Lambda (E):** `{args.return_lambda}` | **Loss:** `{args.critic_loss_type.upper()}`\n")

    md.append("## 1. Algorithm Performance Summary across All Swept Configurations\n")
    md.append("| Algorithm | Method Description | Best Config (by AUC) | Best AUC | Mean AUC (± SEM) | Best Final Return | Mean Final Return (± SEM) |")
    md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")

    method_descriptions = {
        "E_0": "Symmetrized 1-step E-Minimization",
        "TD_0": "Classic 1-step Online TD (Minibatch Delta)",
        "E_lambda": f"Geometric Jump Sampled E(λ={args.lambda_val})",
        "TD_lambda": f"Standard PPO Fitted T^λ (λ={args.lambda_val})",
    }

    for algo_key in ["E_0", "TD_0", "E_lambda", "TD_lambda"]:
        if algo_key not in results_dict or results_dict[algo_key]["df"] is None:
            continue
        df = results_dict[algo_key]["df"]
        desc = method_descriptions.get(algo_key, "")
        pname = ALGO_PRETTY_NAMES.get(algo_key, algo_key)

        best_auc_row = df.sort_values("auc", ascending=False).iloc[0]
        best_cfg = f"`{best_auc_row['config']}`"
        best_auc = f"{best_auc_row['auc']:.2f}"
        mean_auc = f"{df['auc'].mean():.2f} ± {df['auc'].std() / np.sqrt(len(df)):.2f}"

        best_final_row = df.sort_values("final_window_mean", ascending=False).iloc[0]
        best_final = f"{best_final_row['final_window_mean']:.2f}"
        mean_final = f"{df['final_window_mean'].mean():.2f} ± {df['final_window_mean'].std() / np.sqrt(len(df)):.2f}"

        md.append(f"| **{pname}** | {desc} | {best_cfg} | **{best_auc}** | {mean_auc} | **{best_final}** | {mean_final} |")

    md.append("\n---\n")
    md.append("## 2. Matched-Pair Head-to-Head Comparisons\n")
    md.append("Configurations matched 1-to-1 on `[critic_lr, critic_epochs, weight_decay, num_value_heads]`:\n")

    # Matchup 1: E(0) vs TD(0)
    if pairwise_0 is not None:
        p0_stats = pairwise_0[1]
        md.append("### A. 1-Step Methods: **E(0)** vs. **TD(0)**\n")
        md.append(f"- **Configurations Matched:** {p0_stats['n_pairs']}")
        md.append(f"- **AUC Win Rate:** **E(0)** won {p0_stats['e_wins_auc']}/{p0_stats['n_pairs']} ({p0_stats['e_win_rate_auc']:.1f}%) | **TD(0)** won {p0_stats['td_wins_auc']}/{p0_stats['n_pairs']} | Ties: {p0_stats['ties_auc']}")
        md.append(f"- **Mean Δ AUC (E(0) - TD(0)):** **`{p0_stats['mean_delta_auc']:+.2f}`** ± {p0_stats['sem_delta_auc']:.2f} (paired t-test *p* = {p0_stats['p_val_auc']:.4e})")
        md.append(f"- **Final Return Win Rate:** **E(0)** won {p0_stats['e_wins_final']}/{p0_stats['n_pairs']} ({p0_stats['e_win_rate_final']:.1f}%) | **TD(0)** won {p0_stats['td_wins_final']}/{p0_stats['n_pairs']} | Ties: {p0_stats['ties_final']}")
        md.append(f"- **Mean Δ Final Return:** **`{p0_stats['mean_delta_final']:+.2f}`** ± {p0_stats['sem_delta_final']:.2f} (paired t-test *p* = {p0_stats['p_val_final']:.4e})")
        win_label_0 = "E(0)" if p0_stats['mean_delta_auc'] > 0 else ("TD(0)" if p0_stats['mean_delta_auc'] < 0 else "Tie")
        md.append(f"- **Overall 1-Step Advantage:** **{win_label_0}**\n")

    # Matchup 2: E(lambda) vs TD(lambda)
    if pairwise_lambda is not None:
        pl_stats = pairwise_lambda[1]
        md.append(f"### B. Multi-Step / Eligibility Trace Methods: **E(λ={args.lambda_val})** vs. **TD(λ={args.lambda_val})**\n")
        md.append(f"- **Configurations Matched:** {pl_stats['n_pairs']}")
        md.append(f"- **AUC Win Rate:** **E(λ)** won {pl_stats['e_wins_auc']}/{pl_stats['n_pairs']} ({pl_stats['e_win_rate_auc']:.1f}%) | **TD(λ)** won {pl_stats['td_wins_auc']}/{pl_stats['n_pairs']} | Ties: {pl_stats['ties_auc']}")
        md.append(f"- **Mean Δ AUC (E(λ) - TD(λ)):** **`{pl_stats['mean_delta_auc']:+.2f}`** ± {pl_stats['sem_delta_auc']:.2f} (paired t-test *p* = {pl_stats['p_val_auc']:.4e})")
        md.append(f"- **Final Return Win Rate:** **E(λ)** won {pl_stats['e_wins_final']}/{pl_stats['n_pairs']} ({pl_stats['e_win_rate_final']:.1f}%) | **TD(λ)** won {pl_stats['td_wins_final']}/{pl_stats['n_pairs']} | Ties: {pl_stats['ties_final']}")
        md.append(f"- **Mean Δ Final Return:** **`{pl_stats['mean_delta_final']:+.2f}`** ± {pl_stats['sem_delta_final']:.2f} (paired t-test *p* = {pl_stats['p_val_final']:.4e})")
        win_label_l = f"E(λ={args.lambda_val})" if pl_stats['mean_delta_auc'] > 0 else (f"TD(λ={args.lambda_val})" if pl_stats['mean_delta_auc'] < 0 else "Tie")
        md.append(f"- **Overall λ-Trace Advantage:** **{win_label_l}**\n")

    return "\n".join(md)


def plot_4way_comparison_poster(results_dict, pairwise_0, pairwise_lambda, out_dir, env_name, lambda_val):
    """
    Generates a publication-grade 4-panel poster:
        1. Best-in-Class Learning Curves (All 4 algorithms, Mean +/- 1 SEM)
        2. Critic Epochs Scaling (4 -> 16 -> 32) for all 4 algorithms
        3. Value Heads Scaling (1 vs 4 heads) for all 4 algorithms
        4. Matched-Pair Scatters: Dual side-by-side subplots for E(0) vs TD(0) and E(lambda) vs TD(lambda)
    """
    pdf_path = os.path.join(out_dir, "head_to_head_poster.pdf")
    png_path = os.path.join(out_dir, "head_to_head_poster.png")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

    algos_order = ["E_0", "TD_0", "E_lambda", "TD_lambda"]
    line_styles = {"E_0": "-", "TD_0": "--", "E_lambda": "-", "TD_lambda": "--"}
    markers = {"E_0": "o", "TD_0": "s", "E_lambda": "^", "TD_lambda": "D"}

    # --------------------------------------------------------------------------
    # Panel A: Best Learning Curves (All 4 algorithms)
    # --------------------------------------------------------------------------
    ax_lc = fig.add_subplot(gs[0, 0])
    for a_key in algos_order:
        if a_key not in results_dict or results_dict[a_key]["df"] is None:
            continue
        df_a = results_dict[a_key]["df"]
        curves_a = results_dict[a_key]["curves"]
        best_row = df_a.sort_values("auc", ascending=False).iloc[0]
        lbl = best_row["config"]
        if lbl in curves_a:
            m_c, s_c = curves_a[lbl]
            x_arr = np.arange(len(m_c))
            pname = ALGO_PRETTY_NAMES.get(a_key, a_key)
            color = ALGO_PALETTE.get(a_key, "#333")
            ls = line_styles.get(a_key, "-")
            ax_lc.plot(
                x_arr, m_c,
                label=f"{pname} (AUC={best_row['auc']:.1f}, Final={best_row['final_window_mean']:.1f})",
                color=color, linewidth=2.2, linestyle=ls,
            )
            ax_lc.fill_between(x_arr, m_c - s_c, m_c + s_c, color=color, alpha=0.15)

    ax_lc.set_title(f"A. Best-in-Class Learning Curves: {env_name} (Mean ± 1 SEM)", fontweight="bold")
    ax_lc.set_xlabel("Environment Steps (Updates)", fontweight="bold")
    ax_lc.set_ylabel("Episode Return", fontweight="bold")
    ax_lc.grid(True, linestyle=":", alpha=0.6)
    ax_lc.legend(loc="lower right", framealpha=0.95, fontsize=8.5)

    # --------------------------------------------------------------------------
    # Panel B: Critic Epochs Scaling (4 -> 16 -> 32)
    # --------------------------------------------------------------------------
    ax_ep = fig.add_subplot(gs[0, 1])
    all_ep_vals = set()
    for a_key in algos_order:
        if a_key in results_dict and results_dict[a_key]["df"] is not None:
            all_ep_vals.update(results_dict[a_key]["df"]["critic_epochs"].unique())
    ep_vals = sorted(all_ep_vals)

    for a_key in algos_order:
        if a_key not in results_dict or results_dict[a_key]["df"] is None:
            continue
        df_a = results_dict[a_key]["df"]
        pname = ALGO_PRETTY_NAMES.get(a_key, a_key)
        color = ALGO_PALETTE.get(a_key, "#333")
        ls = line_styles.get(a_key, "-")
        marker = markers.get(a_key, "o")

        means = [df_a[df_a["critic_epochs"] == ep]["final_window_mean"].mean() for ep in ep_vals]
        sems = [
            df_a[df_a["critic_epochs"] == ep]["final_window_mean"].std() / np.sqrt(max(1, len(df_a[df_a["critic_epochs"] == ep])))
            for ep in ep_vals
        ]
        ax_ep.errorbar(
            ep_vals, means, yerr=sems, label=pname,
            color=color, marker=marker, linewidth=2.0, linestyle=ls, capsize=4
        )

    ax_ep.set_title("B. Critic Epochs Scaling (4 → 16 → 32)", fontweight="bold")
    ax_ep.set_xlabel("Critic Epochs per Policy Update", fontweight="bold")
    ax_ep.set_ylabel("Marginal Mean Return (across LR/WD/Heads)", fontweight="bold")
    if ep_vals:
        ax_ep.set_xticks(ep_vals)
    ax_ep.grid(True, linestyle=":", alpha=0.6)
    ax_ep.legend(loc="best", framealpha=0.95)

    # --------------------------------------------------------------------------
    # Panel C: Value Heads Scaling (1 Head vs 4 Heads)
    # --------------------------------------------------------------------------
    ax_h = fig.add_subplot(gs[1, 0])
    all_heads = set()
    for a_key in algos_order:
        if a_key in results_dict and results_dict[a_key]["df"] is not None:
            all_heads.update(results_dict[a_key]["df"]["num_value_heads"].unique())
    h_vals = sorted(all_heads)

    n_algos = len([k for k in algos_order if k in results_dict and results_dict[k]["df"] is not None])
    x_indices = np.arange(len(h_vals))
    bar_width = 0.8 / max(1, n_algos)

    algo_idx = 0
    for a_key in algos_order:
        if a_key not in results_dict or results_dict[a_key]["df"] is None:
            continue
        df_a = results_dict[a_key]["df"]
        pname = ALGO_PRETTY_NAMES.get(a_key, a_key)
        color = ALGO_PALETTE.get(a_key, "#333")

        means = [df_a[df_a["num_value_heads"] == h]["final_window_mean"].mean() for h in h_vals]
        sems = [
            df_a[df_a["num_value_heads"] == h]["final_window_mean"].std() / np.sqrt(max(1, len(df_a[df_a["num_value_heads"] == h])))
            for h in h_vals
        ]

        pos = x_indices - 0.4 + (algo_idx + 0.5) * bar_width
        ax_h.bar(pos, means, bar_width, yerr=sems, label=pname, color=color, edgecolor="#222", alpha=0.88, capsize=3)
        algo_idx += 1

    ax_h.set_title("C. Value Heads Scaling (1 vs 4 Heads)", fontweight="bold")
    ax_h.set_xticks(x_indices)
    ax_h.set_xticklabels([f"{h} Head{'s' if h > 1 else ''}" for h in h_vals], fontweight="bold")
    ax_h.set_ylabel("Marginal Mean Return", fontweight="bold")
    ax_h.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax_h.legend(loc="best", framealpha=0.95)

    # --------------------------------------------------------------------------
    # Panel D: Dual Matched-Pair Scatters (D1: E(0) vs TD(0), D2: E(lambda) vs TD(lambda))
    # --------------------------------------------------------------------------
    sub_gs = gs[1, 1].subgridspec(1, 2, wspace=0.3)
    ax_sc1 = fig.add_subplot(sub_gs[0, 0])
    ax_sc2 = fig.add_subplot(sub_gs[0, 1])

    # Subplot D1: E(0) vs TD(0)
    if pairwise_0 is not None and not pairwise_0[0].empty:
        merged_0, st0 = pairwise_0
        y_e = merged_0["auc_e"].values
        x_td = merged_0["auc_td"].values
        mn = min(y_e.min(), x_td.min()) - 1.0
        mx = max(y_e.max(), x_td.max()) + 1.0
        ax_sc1.plot([mn, mx], [mn, mx], "k--", alpha=0.55, linewidth=1.2)
        ax_sc1.scatter(x_td, y_e, color=ALGO_PALETTE["E_0"], edgecolors="#222", s=45, alpha=0.85, zorder=4)
        ax_sc1.set_xlim(mn, mx)
        ax_sc1.set_ylim(mn, mx)
        info1 = f"E(0) Wins: {st0['e_wins_auc']}/{st0['n_pairs']} ({st0['e_win_rate_auc']:.0f}%)\nTD(0) Wins: {st0['td_wins_auc']}/{st0['n_pairs']}\nMean Δ: {st0['mean_delta_auc']:+.2f}\np = {st0['p_val_auc']:.3f}"
        ax_sc1.text(0.04, 0.96, info1, transform=ax_sc1.transAxes, va="top", ha="left",
                    fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#ccc"))
        ax_sc1.set_title("D1. E(0) vs. TD(0) (AUC)", fontweight="bold", fontsize=9.5)
        ax_sc1.set_xlabel("TD(0) AUC", fontweight="bold", fontsize=8.5)
        ax_sc1.set_ylabel("E(0) AUC", fontweight="bold", fontsize=8.5)
        ax_sc1.grid(True, linestyle=":", alpha=0.6)

    # Subplot D2: E(lambda) vs TD(lambda)
    if pairwise_lambda is not None and not pairwise_lambda[0].empty:
        merged_l, stl = pairwise_lambda
        y_el = merged_l["auc_e"].values
        x_tdl = merged_l["auc_td"].values
        mn = min(y_el.min(), x_tdl.min()) - 1.0
        mx = max(y_el.max(), x_tdl.max()) + 1.0
        ax_sc2.plot([mn, mx], [mn, mx], "k--", alpha=0.55, linewidth=1.2)
        ax_sc2.scatter(x_tdl, y_el, color=ALGO_PALETTE["E_lambda"], edgecolors="#222", s=45, alpha=0.85, zorder=4)
        ax_sc2.set_xlim(mn, mx)
        ax_sc2.set_ylim(mn, mx)
        infol = f"E(λ) Wins: {stl['e_wins_auc']}/{stl['n_pairs']} ({stl['e_win_rate_auc']:.0f}%)\nTD(λ) Wins: {stl['td_wins_auc']}/{stl['n_pairs']}\nMean Δ: {stl['mean_delta_auc']:+.2f}\np = {stl['p_val_auc']:.3f}"
        ax_sc2.text(0.04, 0.96, infol, transform=ax_sc2.transAxes, va="top", ha="left",
                    fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#ccc"))
        ax_sc2.set_title(f"D2. E(λ) vs. TD(λ) (λ={lambda_val})", fontweight="bold", fontsize=9.5)
        ax_sc2.set_xlabel("TD(λ) AUC", fontweight="bold", fontsize=8.5)
        ax_sc2.set_ylabel("E(λ) AUC", fontweight="bold", fontsize=8.5)
        ax_sc2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"4-Way Critic Benchmark: E(0), TD(0), E(λ={lambda_val}), TD(λ={lambda_val}) — {env_name}",
                 fontsize=14, fontweight="bold", y=0.98)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


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

    print("=" * 75)
    print("COMPREHENSIVE 4-WAY CRITIC COMPARISON SWEEP")
    print("=" * 75)
    print(f"Environment:       {args.env_name}")
    print(f"Timesteps:         {args.total_timesteps:,} | Seeds: {args.n_seeds}")
    print(f"Critic LR Grid:    {args.critic_lr_grid}")
    print(f"Critic Epochs:     {args.epochs_grid}")
    print(f"Weight Decay:      {args.wd_grid}")
    print(f"Value Heads:       {args.heads_grid}")
    print(f"Lambda Parameter:  {args.lambda_val} (for E_lambda and TD_lambda)")
    print(f"Return Lambda:     {args.return_lambda} (for E_0 and E_lambda)")
    print(f"Algorithms:        {args.algos}")
    print(f"Output Directory:  {out_dir}")
    print("=" * 75)

    algo_runner_map = {
        "E_0": (make_train_e_0, "E_0"),
        "TD_0": (make_train_td, "TD_0"),
        "E_lambda": (make_train_e_lambda, "E_lambda"),
        "TD_lambda": (make_train_td, "TD_lambda"),
    }

    results_dict = {}
    all_summary_dfs = []

    for algo_name in args.algos:
        if algo_name not in algo_runner_map:
            continue
        make_fn, subdir = algo_runner_map[algo_name]
        grid = build_algo_grid(algo_name, args)
        algo_out_dir = os.path.join(out_dir, subdir)

        df, curves, metrics = run_single_algo_sweep(
            algo_name=algo_name,
            make_train_fn=make_fn,
            grid=grid,
            args=args,
            out_algo_dir=algo_out_dir,
        )
        results_dict[algo_name] = {"df": df, "curves": curves, "metrics": metrics}
        all_summary_dfs.append(df)

    # Combine all algorithms into single summary dataframe
    if all_summary_dfs:
        combined_df = pd.concat(all_summary_dfs, ignore_index=True)
        combined_df.to_csv(os.path.join(out_dir, "all_algos_summary.csv"), index=False)

    # 1. Pairwise match: E(0) vs TD(0)
    pairwise_0 = None
    if "E_0" in results_dict and "TD_0" in results_dict:
        merged_0, stats_0 = compute_pairwise_comparison(
            results_dict["E_0"]["df"], results_dict["TD_0"]["df"],
            label_e="E(0)", label_td="TD(0)"
        )
        pairwise_0 = (merged_0, stats_0)
        merged_0.to_csv(os.path.join(out_dir, "matched_pairs_e0_vs_td0.csv"), index=False)

    # 2. Pairwise match: E(lambda) vs TD(lambda)
    pairwise_lambda = None
    if "E_lambda" in results_dict and "TD_lambda" in results_dict:
        merged_l, stats_l = compute_pairwise_comparison(
            results_dict["E_lambda"]["df"], results_dict["TD_lambda"]["df"],
            label_e="E(lambda)", label_td="TD(lambda)"
        )
        pairwise_lambda = (merged_l, stats_l)
        merged_l.to_csv(os.path.join(out_dir, "matched_pairs_elambda_vs_tdlambda.csv"), index=False)

    # Backward compatible head_to_head_summary.csv
    if pairwise_0 is not None:
        pairwise_0[0].to_csv(os.path.join(out_dir, "head_to_head_summary.csv"), index=False)

    # Generate Markdown Summary
    md_summary = generate_markdown_summary(args.env_name, results_dict, pairwise_0, pairwise_lambda, args)
    md_path = os.path.join(out_dir, "auc_summary.md")
    with open(md_path, "w") as f:
        f.write(md_summary)

    print("\n" + "=" * 75)
    print("SWEEP COMPLETE — PERFORMANCE & PAIRWISE SUMMARY")
    print("=" * 75)
    print(md_summary)
    print("=" * 75)

    # Generate 4-panel vector comparison poster
    try:
        plot_4way_comparison_poster(
            results_dict=results_dict,
            pairwise_0=pairwise_0,
            pairwise_lambda=pairwise_lambda,
            out_dir=out_dir,
            env_name=args.env_name,
            lambda_val=args.lambda_val,
        )
        print(f"\nGenerated poster: {os.path.join(out_dir, 'head_to_head_poster.pdf')}")
    except Exception as ex:
        print(f"Warning: Failed to generate 4-way poster: {ex}")


if __name__ == "__main__":
    main()

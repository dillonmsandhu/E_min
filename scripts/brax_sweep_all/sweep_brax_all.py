#!/usr/bin/env python3
"""
scripts/brax_sweep_all/sweep_brax_all.py

Comprehensive 4-way continuous control critic benchmark on Brax:
1. TD(0)      - Classic 1-step online TD MSE critic
2. E(0)       - Adjacent-state Dirichlet smoothness error minimization
3. TD(lambda) - Standard PPO Fitted Value Iteration MSE (lambda = 0.9)
4. E(lambda)  - Geometric jumping Dirichlet error minimization (lambda = 0.9)

Grid:
  - Critic Learning Rate: [1e-4, 3e-4, 1e-3]
  - Critic Epochs:        [4, 8, 16]
Evaluated across all 11 Brax environments (excluding 'fast') with independent random seeds.
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
from brax_rl.core.brax_config import config as default_config
from brax_rl.algos.ppo import make_train

# Set matplotlib parameters for publication-quality vector figures
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
    parser = argparse.ArgumentParser(description="Brax 4-Way Critic Sweep: TD(0), E(0), TD(lambda), E(lambda)")
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

    # Hyperparameter Sweep Grid
    parser.add_argument("--critic-lr-grid", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3],
                        help="Critic learning rate grid (default: 1e-4 3e-4 1e-3)")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[4, 8, 16],
                        help="Critic epochs grid (default: 4 8 16)")
    parser.add_argument("--layer-norm-grid", type=str2bool, nargs="+", default=[False, True],
                        help="Layer norm grid for policy and value networks (default: False True)")
    parser.add_argument("--layer-norm", type=str2bool, default=None,
                        help="Single layer norm boolean (overrides --layer-norm-grid if specified)")

    # Lambda parameters
    parser.add_argument("--lambda-val", type=float, default=0.9,
                        help="Fixed lambda for TD(lambda) and E(lambda) (default: 0.9)")
    parser.add_argument("--return-lambda-grid", type=float, nargs="+", default=[0.95, 1.0],
                        help="Return target lambda grid for E(0) and E(lambda) (default: 0.95 1.0)")
    parser.add_argument("--return-lambda", type=float, default=None,
                        help="Single return target lambda (overrides --return-lambda-grid if specified)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda for policy advantage estimation (default: 0.95)")

    # Algorithms to evaluate
    parser.add_argument("--algos", type=str, nargs="+",
                        default=["TD_0", "E_0", "TD_lambda", "E_lambda"],
                        choices=["TD_0", "E_0", "TD_lambda", "E_lambda"],
                        help="Algorithms to evaluate (default: TD_0 E_0 TD_lambda E_lambda)")

    # Execution controls
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Unified sweep identifier across SLURM tasks (default: SLURM_ARRAY_JOB_ID or timestamp)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=10,
                        help="Window size in policy iterations for final evaluation (default: 10)")

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
    max_ret = float(np.max(returns_matrix))
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


def configure_algorithm(base_cfg, algo_name, clr, ep, ln, lambda_val, return_lambda, gae_lambda):
    """Configures the specific algorithm parameters into a fresh config copy."""
    cfg = base_cfg.copy()
    cfg["CRITIC_LR"] = clr
    cfg["CRITIC_EPOCHS"] = ep
    cfg["LAYER_NORM"] = ln
    cfg["GAE_LAMBDA"] = gae_lambda

    if algo_name == "TD_0":
        cfg["CRITIC_TYPE"] = "td_0"
        cfg["TD_LAMBDA"] = 0.0
    elif algo_name == "E_0":
        cfg["CRITIC_TYPE"] = "e_0"
        cfg["E_LAMBDA"] = 0.0
        cfg["RETURN_LAMBDA"] = return_lambda
    elif algo_name == "TD_lambda":
        cfg["CRITIC_TYPE"] = "fitted"
        cfg["VALUE_LAMBDA"] = lambda_val
    elif algo_name == "E_lambda":
        cfg["CRITIC_TYPE"] = "e_lambda"
        cfg["E_LAMBDA"] = lambda_val
        cfg["RETURN_LAMBDA"] = return_lambda
    else:
        raise ValueError(f"Unknown algorithm '{algo_name}'")

    return cfg


def plot_4way_poster(best_results, curves_dict, all_results, args, out_dir, step_axis):
    """
    Generates a publication-quality 4-panel poster comparison figure:
    Panel A: Learning Curves of Best Configuration for each of the 4 algorithms.
    Panel B: Pairwise Head-to-Head Deltas (E0 vs TD0, and E(lambda) vs TD(lambda)).
    Panel C: Scaling across Critic Epochs (4 -> 8 -> 16).
    Panel D: Hyperparameter Sensitivity Heatmaps (Critic LR x Epochs).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.32, wspace=0.25)

    # -------------------------------------------------------------------------
    # Panel A: Learning Curves of Best of each Algorithm
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    for algo in args.algos:
        if algo not in best_results:
            continue
        best_r = best_results[algo]
        lbl = best_r["label"]
        curve = curves_dict[algo][lbl]
        color = ALGO_COLORS.get(algo, "black")
        ls = ALGO_LINE_STYLES.get(algo, "-")
        disp_name = ALGO_PRETTY_NAMES.get(algo, algo)

        ax1.plot(step_axis, curve["mean"], color=color, linestyle=ls, lw=2.0,
                 label=f"{disp_name} (lr={best_r['critic_lr']}, ep={best_r['critic_epochs']})")
        ax1.fill_between(step_axis, curve["mean"] - curve["sem"],
                         curve["mean"] + curve["sem"], color=color, alpha=0.18)

    ax1.set_title(f"A. Learning Curves: Best of Each Algorithm ({args.env_name})", fontweight="bold")
    ax1.set_xlabel("Environment Steps (Millions)")
    ax1.set_ylabel("Episode Return")
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel B: Head-to-Head Comparisons (E0 vs TD0 & E(lam) vs TD(lam))
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    comparisons = []
    labels_bar = []
    colors_bar = []

    # E(0) vs TD(0)
    if "E_0" in best_results and "TD_0" in best_results:
        e0_mean = best_results["E_0"]["final_mean"]
        td0_mean = best_results["TD_0"]["final_mean"]
        delta_0 = e0_mean - td0_mean
        comparisons.append(delta_0)
        labels_bar.append("E(0) vs TD(0)")
        colors_bar.append(ALGO_COLORS["E_0"] if delta_0 >= 0 else ALGO_COLORS["TD_0"])

    # E(lambda) vs TD(lambda)
    if "E_lambda" in best_results and "TD_lambda" in best_results:
        el_mean = best_results["E_lambda"]["final_mean"]
        tdl_mean = best_results["TD_lambda"]["final_mean"]
        delta_l = el_mean - tdl_mean
        comparisons.append(delta_l)
        labels_bar.append(r"E($\lambda$) vs TD($\lambda$)")
        colors_bar.append(ALGO_COLORS["E_lambda"] if delta_l >= 0 else ALGO_COLORS["TD_lambda"])

    if comparisons:
        x_pos = np.arange(len(comparisons))
        bars = ax2.bar(x_pos, comparisons, color=colors_bar, width=0.45, edgecolor="black", alpha=0.85)
        ax2.axhline(0, color="gray", linestyle="--", lw=1.0)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels_bar, fontweight="bold", fontsize=10)
        ax2.set_ylabel("Difference in Final Return ($\Delta$)")
        for bar in bars:
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            ax2.annotate(f"{height:+.1f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3 if height >= 0 else -10),
                         textcoords="offset points",
                         ha="center", va=va, fontweight="bold", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "Insufficient algorithms for pairwise comparison",
                 ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("B. Head-to-Head Advantage ($\Delta$ Best Return)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # -------------------------------------------------------------------------
    # Panel C: Performance across Critic Epochs Scaling (4 -> 8 -> 16)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    epochs_sorted = sorted(args.epochs_grid)
    for algo in args.algos:
        if algo not in all_results:
            continue
        best_lr = best_results[algo]["critic_lr"]
        best_rlam = best_results[algo].get("return_lambda")
        best_ln = best_results[algo].get("layer_norm")
        sub = [
            r for r in all_results[algo]
            if r["critic_lr"] == best_lr
            and r.get("return_lambda") == best_rlam
            and r.get("layer_norm") == best_ln
        ]
        sub.sort(key=lambda x: x["critic_epochs"])

        x_eps = [r["critic_epochs"] for r in sub]
        y_means = [r["final_mean"] for r in sub]
        y_sems = [r["final_sem"] for r in sub]

        color = ALGO_COLORS.get(algo, "black")
        ls = ALGO_LINE_STYLES.get(algo, "-")
        disp_name = ALGO_PRETTY_NAMES.get(algo, algo)

        ax3.errorbar(x_eps, y_means, yerr=y_sems, marker="o", lw=1.8, capsize=4,
                     linestyle=ls, color=color, label=f"{disp_name} (best lr={best_lr})")

    ax3.set_title("C. Critic Epochs Scaling (4 -> 8 -> 16)", fontweight="bold")
    ax3.set_xlabel("Critic Epochs per Policy Update")
    ax3.set_ylabel("Final Mean Return")
    ax3.set_xticks(epochs_sorted)
    ax3.legend(loc="best", frameon=True, fontsize=9)
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Panel D: Multi-Algorithm Return Summary Bar Chart
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    algo_names = [a for a in args.algos if a in best_results]
    x_coords = np.arange(len(algo_names))
    means = [best_results[a]["final_mean"] for a in algo_names]
    sems = [best_results[a]["final_sem"] for a in algo_names]
    colors = [ALGO_COLORS.get(a, "gray") for a in algo_names]

    bars = ax4.bar(x_coords, means, yerr=sems, capsize=5, color=colors, edgecolor="black", alpha=0.85, width=0.5)
    ax4.set_xticks(x_coords)
    ax4.set_xticklabels([ALGO_PRETTY_NAMES.get(a, a) for a in algo_names], fontweight="bold", fontsize=10)
    ax4.set_ylabel("Final Mean Return")
    ax4.set_title("D. Best Final Return by Algorithm", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    for bar, m, s in zip(bars, means, sems):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax4.annotate(f"{m:.1f}\n±{s:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4 if height >= 0 else -16),
                     textcoords="offset points",
                     ha="center", va=va, fontweight="bold", fontsize=9)

    fig.suptitle(f"Brax 4-Way Critic Comparison — {args.env_name}", fontsize=13, fontweight="bold", y=0.995)

    pdf_path = os.path.join(out_dir, "comparison_poster.pdf")
    png_path = os.path.join(out_dir, "comparison_poster.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nGenerated comparison figures:\n  {pdf_path}\n  {png_path}")


def main():
    args = parse_args()

    # Determine unified sweep directory
    sweep_id = args.sweep_id
    if sweep_id is None:
        array_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_id:
            sweep_id = f"brax_sweep_all_{array_id}"
        else:
            job_id = os.environ.get("SLURM_JOB_ID")
            sweep_id = f"brax_sweep_all_{job_id}" if job_id else f"brax_sweep_all_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = args.output_dir if args.output_dir else os.path.join("results/sweeps", sweep_id, args.env_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 75)
    print(f"BRAX 4-WAY CRITIC SWEEP: TD(0), E(0), TD(lambda), E(lambda)")
    print(f"  Suite / Sweep ID: {sweep_id}")
    print(f"  Environment:      {args.env_name}")
    print(f"  Total Timesteps:  {args.total_timesteps:,}")
    print(f"  Seeds:            {args.n_seeds}")
    print(f"  Rollout:          {args.num_envs} envs x {args.num_steps} steps (batch = {args.num_envs * args.num_steps:,})")
    print(f"  Critic LR Grid:   {args.critic_lr_grid}")
    print(f"  Critic Ep Grid:   {args.epochs_grid}")
    print(f"  Layer Norm Grid:  {args.layer_norm_grid if args.layer_norm is None else [args.layer_norm]}")
    print(f"  Lambda Val:       {args.lambda_val} (for TD_lambda & E_lambda)")
    print(f"  Return Lambda Grid: {args.return_lambda_grid if args.return_lambda is None else [args.return_lambda]} (for E_0 & E_lambda)")
    print(f"  GAE Lambda:       {args.gae_lambda}")
    print(f"  Algorithms:       {args.algos}")
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
    step_axis = np.arange(1, num_updates + 1) * timesteps_per_update / 1_000_000.0

    all_results = {}
    curves_dict = {}
    best_results = {}
    all_metrics = {}
    ret_lambda_list = [args.return_lambda] if args.return_lambda is not None else args.return_lambda_grid
    ln_list = [args.layer_norm] if args.layer_norm is not None else args.layer_norm_grid

    for algo_idx, algo_name in enumerate(args.algos, 1):
        disp_name = ALGO_PRETTY_NAMES.get(algo_name, algo_name)
        if algo_name in ["E_0", "E_lambda"]:
            algo_grid = [
                (clr, ep, ln, rlam)
                for (clr, ep, ln), rlam in itertools.product(
                    itertools.product(args.critic_lr_grid, args.epochs_grid, ln_list), ret_lambda_list
                )
            ]
        else:
            algo_grid = [
                (clr, ep, ln, None)
                for clr, ep, ln in itertools.product(args.critic_lr_grid, args.epochs_grid, ln_list)
            ]

        print(f"\n>>> [{algo_idx}/{len(args.algos)}] RUNNING SWEEP FOR {disp_name} ({len(algo_grid)} Configurations)...")

        algo_rows = []
        curves_dict[algo_name] = {}
        all_metrics[algo_name] = {}

        for cfg_idx, (clr, ep, ln, rlam) in enumerate(algo_grid, 1):
            if rlam is not None:
                label = f"{algo_name}_clr{clr}_ep{ep}_ln{ln}_rlam{rlam}"
                print(f"  [{cfg_idx:02d}/{len(algo_grid)}] {algo_name} | LR={clr}, Epochs={ep}, LN={ln}, ReturnLambda={rlam} ...")
            else:
                label = f"{algo_name}_clr{clr}_ep{ep}_ln{ln}"
                print(f"  [{cfg_idx:02d}/{len(algo_grid)}] {algo_name} | LR={clr}, Epochs={ep}, LN={ln} ...")

            cfg = configure_algorithm(
                base_cfg=base_cfg,
                algo_name=algo_name,
                clr=clr,
                ep=ep,
                ln=ln,
                lambda_val=args.lambda_val,
                return_lambda=rlam if rlam is not None else 1.0,
                gae_lambda=args.gae_lambda,
            )

            t0 = time.time()
            raw_metrics = run_configuration(cfg, args.n_seeds)
            elapsed = time.time() - t0

            returns = np.array(raw_metrics["returned_episode_returns"])  # (n_seeds, updates)
            curves_dict[algo_name][label] = {
                "mean": np.mean(returns, axis=0),
                "sem": np.std(returns, axis=0, ddof=1) / np.sqrt(args.n_seeds) if args.n_seeds > 1 else np.zeros(returns.shape[1]),
                "critic_lr": clr,
                "critic_epochs": ep,
                "layer_norm": ln,
            }
            all_metrics[algo_name][label] = {
                "returns": returns,
                "delta_mag": np.array(raw_metrics.get("delta_mag", [])),
                "magnitude_loss": np.array(raw_metrics.get("magnitude_loss", [])),
                "dirichlet_loss": np.array(raw_metrics.get("dirichlet_loss", [])),
            }

            stats_dict = extract_scalar_summary(returns, args.window_size)
            row = {
                "algorithm": algo_name,
                "label": label,
                "critic_lr": clr,
                "critic_epochs": ep,
                "layer_norm": ln,
                "lambda_val": args.lambda_val if "lambda" in algo_name else None,
                "return_lambda": rlam,
                "final_mean": stats_dict["final_mean"],
                "final_sem": stats_dict["final_sem"],
                "final_std": stats_dict["final_std"],
                "max_return": stats_dict["max_return"],
                "auc": stats_dict["auc"],
                "walltime_sec": elapsed,
                "_seed_finals": stats_dict["seed_finals"],
            }
            algo_rows.append(row)
            print(f"      Finished in {elapsed:.1f}s | Return: {stats_dict['final_mean']:.2f} ± {stats_dict['final_sem']:.2f} | Max: {stats_dict['max_return']:.2f}")

        all_results[algo_name] = algo_rows

        # Save per-algo CSV
        df_algo = pd.DataFrame([{k: v for k, v in r.items() if k != "_seed_finals"} for r in algo_rows])
        df_algo.sort_values(by="final_mean", ascending=False, inplace=True)
        df_algo.to_csv(os.path.join(out_dir, f"summary_{algo_name.lower()}.csv"), index=False)

        # Record best configuration for this algo
        best_results[algo_name] = sorted(algo_rows, key=lambda x: x["final_mean"], reverse=True)[0]

    # Save overall summary CSV containing all configurations
    all_rows = []
    for algo, rows in all_results.items():
        all_rows.extend(rows)
    df_all = pd.DataFrame([{k: v for k, v in r.items() if k != "_seed_finals"} for r in all_rows])
    df_all.sort_values(by=["algorithm", "final_mean"], ascending=[True, False], inplace=True)
    df_all.to_csv(os.path.join(out_dir, "summary_all_algos.csv"), index=False)

    # Save best per algo CSV
    df_best = pd.DataFrame([{k: v for k, v in r.items() if k != "_seed_finals"} for r in best_results.values()])
    df_best.sort_values(by="final_mean", ascending=False, inplace=True)
    df_best.to_csv(os.path.join(out_dir, "summary_best_per_algo.csv"), index=False)

    # Head-to-Head comparisons
    def perform_pairwise_comparison(algo_a, algo_b, comp_filename):
        if algo_a in best_results and algo_b in best_results:
            rows = []
            for r_a in all_results[algo_a]:
                matching_b = [
                    r for r in all_results[algo_b]
                    if r["critic_lr"] == r_a["critic_lr"]
                    and r["critic_epochs"] == r_a["critic_epochs"]
                    and r["layer_norm"] == r_a["layer_norm"]
                ]
                if matching_b:
                    r_b = matching_b[0]
                    seeds_a = r_a["_seed_finals"]
                    seeds_b = r_b["_seed_finals"]
                    delta = float(np.mean(seeds_a) - np.mean(seeds_b))
                    pct_gain = float(delta / (abs(np.mean(seeds_b)) + 1e-8) * 100.0)

                    if len(seeds_a) == len(seeds_b) and len(seeds_a) > 1:
                        try:
                            _, p_val = stats.ttest_rel(seeds_a, seeds_b)
                        except Exception:
                            p_val = 1.0
                    else:
                        p_val = 1.0

                    rows.append({
                        "env_name": args.env_name,
                        "critic_lr": r_a["critic_lr"],
                        "critic_epochs": r_a["critic_epochs"],
                        "layer_norm": r_a["layer_norm"],
                        f"{algo_a}_return": r_a["final_mean"],
                        f"{algo_a}_sem": r_a["final_sem"],
                        f"{algo_b}_return": r_b["final_mean"],
                        f"{algo_b}_sem": r_b["final_sem"],
                        "delta": delta,
                        "pct_gain": pct_gain,
                        "p_value": p_val,
                        "significant_05": p_val < 0.05,
                    })
            if rows:
                df_cmp = pd.DataFrame(rows)
                df_cmp.sort_values(by="delta", ascending=False, inplace=True)
                df_cmp.to_csv(os.path.join(out_dir, comp_filename), index=False)

    perform_pairwise_comparison("E_0", "TD_0", "summary_comparison_0.csv")
    perform_pairwise_comparison("E_lambda", "TD_lambda", "summary_comparison_lambda.csv")

    # Save full metrics pickle
    with open(os.path.join(out_dir, "metrics.pkl"), "wb") as f:
        pickle.dump({
            "all_results": all_results,
            "best_results": best_results,
            "curves_dict": curves_dict,
            "all_metrics": all_metrics,
            "step_axis": step_axis,
            "config": base_cfg,
            "env_name": args.env_name,
            "lambda_val": args.lambda_val,
            "return_lambda": args.return_lambda,
        }, f)

    # Generate 4-panel poster
    plot_4way_poster(best_results, curves_dict, all_results, args, out_dir, step_axis)

    # Print Best Summary
    print("\n" + "=" * 75)
    print("BEST CONFIGURATION SUMMARY PER ALGORITHM:")
    for algo in args.algos:
        if algo in best_results:
            b = best_results[algo]
            disp = ALGO_PRETTY_NAMES.get(algo, algo)
            print(f"  {disp:15s}: {b['final_mean']:.2f} ± {b['final_sem']:.2f} | Max: {b['max_return']:.2f} | AUC: {b['auc']:.2f} ({b['label']})")
    print("=" * 75)


if __name__ == "__main__":
    main()

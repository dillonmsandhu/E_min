#!/usr/bin/env python3
"""
sweep_minatar_e_opt.py

Comprehensive 8-condition hyperparameter sweep for E-minimization on MinAtar:
    - WEIGHT_DECAY: [0.0, 0.01]
    - epochs (NUM_EPOCHS / CRITIC_EPOCHS): [4, 16]
    - num_envs (NUM_ENVS): [64, 512]

Evaluates all 2 x 2 x 2 = 8 configurations on a given MinAtar game,
ranks configurations, plots comparison learning curves (Mean ± SEM) as a publication-grade PDF,
and exports summary CSV tables and metric pickles.
"""

import os
import sys
import time
import argparse
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from core.config import config as default_config
from core.utils import merge_hparams, save_results
from algos.E_experimental import make_train

# TrueType font embedding for vector publication quality
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Tune E optimization on MinAtar")
    parser.add_argument("--env-name", type=str, default="Asterix-MinAtar",
                        help="MinAtar environment name (default: Asterix-MinAtar)")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="Total environment timesteps (default: 10,000,000)")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Number of independent random seeds (default: 8)")
    parser.add_argument("--actor-lr", type=float, default=0.003,
                        help="Actor learning rate (default: 0.003)")
    parser.add_argument("--critic-lr", type=float, default=0.001,
                        help="Critic learning rate (default: 0.001)")
    parser.add_argument("--critic-loss-type", type=str, default="mse", choices=["mse", "huber"],
                        help="Critic loss type: 'mse' or 'huber' (default: mse)")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber loss transition delta (default: 1.0)")
    parser.add_argument("--num-value-heads", type=int, default=1,
                        help="Number of value heads in critic ensemble (default: 1)")
    parser.add_argument("--value-head-agg", type=str, default="sum", choices=["sum", "mean"],
                        help="Aggregation of value head losses: 'sum' or 'mean' (default: sum)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for final mean evaluation (default: 100)")
    parser.add_argument("--metric", type=str, default="returned_discounted_episode_returns",
                        help="Primary evaluation metric")
    return parser.parse_args()


def plot_comparison_curves(curves_dict, out_path, env_name, metric_name, window_size=100):
    """Plots all 8 configurations on a single graph with Mean ± SEM and ranked legend."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Sort curves by final window performance (higher is better)
    ranked = []
    for label, (mean_c, sem_c) in curves_dict.items():
        win = min(len(mean_c), window_size)
        score = float(np.mean(mean_c[-win:]))
        ranked.append((score, label, mean_c, sem_c))

    ranked.sort(key=lambda x: x[0], reverse=True)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(ranked), 10)))

    for idx, (score, label, mean_c, sem_c) in enumerate(ranked):
        x = np.arange(len(mean_c))
        c = colors[idx % 10]
        is_best = (idx == 0)
        display_label = f"[{idx+1}] {label} (final={score:.2f})"
        if is_best:
            display_label = f"★ {display_label}"
            lw = 2.6
            alpha_line = 1.0
        else:
            lw = 1.6
            alpha_line = 0.85

        ax.plot(x, mean_c, label=display_label, color=c, linewidth=lw, alpha=alpha_line)
        ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color=c, alpha=0.15)

    ax.set_title(f"E-Minimization Optimization Tuning: {env_name}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Update Step", fontsize=12)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved: {out_path}")


def main():
    args = parse_args()

    # Parameter grid to sweep: 2 x 2 x 2 = 8 configs
    NUM_ENVS_LIST = [64, 512]
    EPOCHS_LIST = [4, 16]
    WEIGHT_DECAY_LIST = [0.0, 0.01]

    grid_combos = list(itertools.product(NUM_ENVS_LIST, EPOCHS_LIST, WEIGHT_DECAY_LIST))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_id = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", f"local_{timestamp}"))

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join("results", "ppo", "sweeps", f"e_opt_{suite_id}", args.env_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STARTING E OPTIMIZATION SWEEP ON MINATAR")
    print(f"Environment: {args.env_name}")
    print(f"Timesteps: {args.total_timesteps:,} | Seeds: {args.n_seeds}")
    print(f"Grid: NUM_ENVS={NUM_ENVS_LIST} x EPOCHS={EPOCHS_LIST} x WEIGHT_DECAY={WEIGHT_DECAY_LIST}")
    print(f"Total Configurations: {len(grid_combos)}")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)

    curves_dict = {}
    summary_rows = []
    all_metrics = {}

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, args.n_seeds)

    for c_idx, (num_envs, epochs, wd) in enumerate(grid_combos):
        cfg = default_config.copy()
        cfg.update({
            "ENV_NAME": args.env_name,
            "TOTAL_TIMESTEPS": args.total_timesteps,
            "N_SEEDS": args.n_seeds,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": 64,
            "MINIBATCH_SIZE": 1024,
            "NUM_EPOCHS": epochs,
            "CRITIC_EPOCHS": epochs,
            "ACTOR_EPOCHS": 4,  # Keep actor epochs controlled at 4 to test critic-specific decoupling
            "LR": args.critic_lr,
            "ACTOR_LR": args.actor_lr,
            "WEIGHT_DECAY": wd,
            "CRITIC_WEIGHT_DECAY": wd,
            "ACTOR_WEIGHT_DECAY": 0.01,
            "GAE_LAMBDA": 0.8,
            "RETURN_LAMBDA": 1.0,
            "k": 64,
            "LAYER_NORM": "True",
            "VF_CLIP": 1e6,
            "ENT_COEF": 0.001,
            "CRITIC_LOSS_TYPE": args.critic_loss_type,
            "HUBER_DELTA": args.huber_delta,
            "NUM_VALUE_HEADS": args.num_value_heads,
            "VALUE_HEAD_AGG": args.value_head_agg,
        })

        label = f"envs={num_envs}_ep={epochs}_wd={wd}"
        print(f"\n[{c_idx + 1}/{len(grid_combos)}] Running configuration: {label} ...", flush=True)

        t0 = time.time()
        train_fn = make_train(cfg)
        vmapped_train = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

        out = vmapped_train(rngs, {})
        metric_arr = out["metrics"][args.metric].block_until_ready()
        elapsed = time.time() - t0
        print(f"   Completed in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")

        tensor = np.asarray(metric_arr)
        mean_c = tensor.mean(axis=0)
        sem_c = tensor.std(axis=0) / np.sqrt(args.n_seeds)

        curves_dict[label] = (mean_c, sem_c)
        all_metrics[label] = {k: np.asarray(v) for k, v in out["metrics"].items()}

        win = min(len(mean_c), args.window_size)
        final_mean = float(np.mean(mean_c[-win:]))
        final_sem = float(np.mean(sem_c[-win:]))
        auc = float(np.mean(mean_c))

        summary_rows.append({
            "config": label,
            "num_envs": num_envs,
            "epochs": epochs,
            "weight_decay": wd,
            "final_window_mean": final_mean,
            "final_window_sem": final_sem,
            "auc": auc,
            "elapsed_seconds": elapsed,
        })

    # Save summary dataframe
    df = pd.DataFrame(summary_rows)
    df.sort_values(by="final_window_mean", ascending=False, inplace=True)
    csv_path = os.path.join(out_dir, "summary_e_opt.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY RANKINGS:")
    print(df.to_string(index=False))
    print("=" * 70)

    # Plot comparison curves (both PDF and PNG)
    pdf_path = os.path.join(out_dir, "comparison_e_opt.pdf")
    png_path = os.path.join(out_dir, "comparison_e_opt.png")
    plot_comparison_curves(curves_dict, pdf_path, args.env_name, args.metric, args.window_size)
    plot_comparison_curves(curves_dict, png_path, args.env_name, args.metric, args.window_size)

    import cloudpickle
    metrics_path = os.path.join(out_dir, "metrics.pkl")
    with open(metrics_path, "wb") as f:
        cloudpickle.dump(all_metrics, f)
    print(f"\nAll artifacts written to {out_dir}")


if __name__ == "__main__":
    main()

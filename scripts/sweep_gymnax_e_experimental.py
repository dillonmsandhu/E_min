#!/usr/bin/env python3
"""
sweep_gymnax_e_experimental.py

Multi-Dimensional Hyperparameter Sweep for E_experimental across Gymnax Environments:
Swept Dimensions:
    - CRITIC_EPOCHS: [4, 16]
    - WEIGHT_DECAY (CRITIC_WEIGHT_DECAY): [0.0, 0.01]
    - CRITIC_LOSS_TYPE: ["mse", "huber"]
    - NUM_VALUE_HEADS: [1, 4]
    - (Optionally NUM_ENVS if swept)

Total default combinations: 2 x 2 x 2 x 2 = 16 configurations.
Evaluates all configurations across independent random seeds using JAX vmap,
exports summary CSV rankings, metric pickles, and invokes the multi-dimensional
visualization engine to produce publication-ready PDF & PNG posters.

Usage:
    python scripts/sweep_gymnax_e_experimental.py --env-name CartPole-v1 --n-seeds 8
    python scripts/sweep_gymnax_e_experimental.py --env-name MountainCar-v0 --n-seeds 8 --total-timesteps 2048000
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

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from core.config import config as default_config
from core.utils import merge_hparams, save_results
from algos.E_experimental import make_train
from scripts.visualize_multidim_sweep import generate_multidim_analysis_pdf

# TrueType font embedding for vector publication quality
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Tune E_experimental on Gymnax Environments")
    parser.add_argument("--env-name", type=str, default="CartPole-v1",
                        help="Gymnax environment name (default: CartPole-v1)")
    parser.add_argument("--total-timesteps", type=int, default=2_048_000,
                        help="Total environment timesteps (default: 2,048,000)")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Number of independent random seeds (default: 8)")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel environments (default: 64)")
    parser.add_argument("--num-steps", type=int, default=64,
                        help="Number of rollout steps per env (default: 64)")
    parser.add_argument("--minibatch-size", type=int, default=1024,
                        help="Minibatch size for SGD updates (default: 1024)")
    parser.add_argument("--actor-lr", type=float, default=0.0003,
                        help="Actor learning rate (default: 0.0003)")
    parser.add_argument("--critic-lr", type=float, default=0.001,
                        help="Critic learning rate (default: 0.001)")
    parser.add_argument("--actor-epochs", type=int, default=4,
                        help="Fixed actor update epochs (default: 4)")
    parser.add_argument("--huber-delta", type=float, default=1.0,
                        help="Huber loss transition delta threshold (default: 1.0)")
    parser.add_argument("--value-head-agg", type=str, default="sum", choices=["sum", "mean"],
                        help="Multi-head loss aggregation (default: sum)")
    
    # Grid options
    parser.add_argument("--grid-mode", type=str, default="default",
                        choices=["default", "opt8", "epochs16", "heads16", "custom"],
                        help="Grid mode: 'default'/'opt8' (8 configs: epochs x wd x heads with MSE), 'epochs16' (16 configs: [4,8,16,32] epochs), 'heads16' (16 configs: [1,2,4,8] heads), or 'custom'")
    parser.add_argument("--epochs-grid", type=int, nargs="+", default=[4, 16],
                        help="Critic epochs grid (default: 4 16)")
    parser.add_argument("--wd-grid", type=float, nargs="+", default=[0.001, 0.01],
                        help="Critic weight decay grid (default: 0.001 0.01)")
    parser.add_argument("--loss-grid", type=str, nargs="+", default=["mse"],
                        help="Critic loss types grid (default: mse)")
    parser.add_argument("--heads-grid", type=int, nargs="+", default=[1, 4],
                        help="Value heads grid (default: 1 4)")
    parser.add_argument("--num-envs-grid", type=int, nargs="+", default=None,
                        help="Optional num_envs grid (default: None, uses --num-envs)")

    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for final mean evaluation (default: 100)")
    parser.add_argument("--metric", type=str, default="returned_episode_returns",
                        help="Evaluation metric (default: returned_episode_returns)")
    return parser.parse_args()


def build_grid(args):
    """Constructs the list of configurations based on the chosen grid mode."""
    if args.grid_mode in ["default", "opt8"]:
        epochs_list = [4, 16]
        wd_list = [0.001, 0.01]
        loss_list = ["mse"]
        heads_list = [1, 4]
        envs_list = [args.num_envs] if args.num_envs_grid is None else args.num_envs_grid
    elif args.grid_mode == "epochs16":
        epochs_list = [4, 8, 16, 32]
        wd_list = [0.001, 0.01]
        loss_list = ["mse"]
        heads_list = [1, 4]
        envs_list = [args.num_envs] if args.num_envs_grid is None else args.num_envs_grid
    elif args.grid_mode == "heads16":
        epochs_list = [4, 16]
        wd_list = [0.001, 0.01]
        loss_list = ["mse"]
        heads_list = [1, 2, 4, 8]
        envs_list = [args.num_envs] if args.num_envs_grid is None else args.num_envs_grid
    else:  # custom
        epochs_list = args.epochs_grid
        wd_list = args.wd_grid
        loss_list = args.loss_grid
        heads_list = args.heads_grid
        envs_list = [args.num_envs] if args.num_envs_grid is None else args.num_envs_grid

    grid = []
    for envs, ep, wd, loss_t, heads in itertools.product(envs_list, epochs_list, wd_list, loss_list, heads_list):
        label_parts = []
        if len(envs_list) > 1:
            label_parts.append(f"envs={envs}")
        label_parts.append(f"ep={ep}")
        label_parts.append(f"wd={wd}")
        if len(loss_list) > 1 or args.grid_mode == "full":
            label_parts.append(f"loss={loss_t}")
        if len(heads_list) > 1 or args.grid_mode == "full":
            label_parts.append(f"heads={heads}")
        
        label = "_".join(label_parts)
        grid.append({
            "label": label,
            "num_envs": envs,
            "critic_epochs": ep,
            "weight_decay": wd,
            "critic_loss_type": loss_t,
            "num_value_heads": heads,
        })
    return grid


def plot_comparison_curves(curves_dict, out_path, env_name, metric_name, window_size=100):
    """Plots all configurations on a single graph with Mean ± SEM and ranked legend."""
    fig, ax = plt.subplots(figsize=(12, 7))

    ranked = []
    for label, (mean_c, sem_c) in curves_dict.items():
        win = min(len(mean_c), window_size)
        score = float(np.mean(mean_c[-win:]))
        ranked.append((score, label, mean_c, sem_c))

    ranked.sort(key=lambda x: x[0], reverse=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(ranked), 20)))

    for idx, (score, label, mean_c, sem_c) in enumerate(ranked):
        x = np.arange(len(mean_c))
        c = colors[idx % 20]
        is_best = (idx == 0)
        display_label = f"[{idx+1:02d}] {label} (final={score:.2f})"
        if is_best:
            display_label = f"★ {display_label}"
            lw = 2.5
            alpha_line = 1.0
        else:
            lw = 1.4
            alpha_line = 0.8

        ax.plot(x, mean_c, label=display_label, color=c, linewidth=lw, alpha=alpha_line)
        ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color=c, alpha=0.10)

    ax.set_title(f"E_experimental Optimization Sweep: {env_name} (Mean ± 1 SEM)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Environment Steps (Updates)", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    ncol = 2 if len(ranked) > 8 else 1
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0,
              fontsize=8.5, framealpha=0.95, ncol=ncol)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        sweep_id = f"e_exp_gymnax_{timestamp}"
        out_dir = os.path.join(_repo_root, "results", "ppo", "sweeps", sweep_id, args.env_name)

    os.makedirs(out_dir, exist_ok=True)
    grid = build_grid(args)

    print("=" * 70)
    print(f"E_EXPERIMENTAL MULTI-DIMENSIONAL SWEEP: {args.env_name}")
    print(f"Total Configurations: {len(grid)}")
    print(f"Seeds: {args.n_seeds} | Timesteps: {args.total_timesteps:,}")
    print(f"Output Directory: {out_dir}")
    print("=" * 70)

    curves_dict = {}
    all_metrics = {}
    summary_rows = []

    for idx, cfg_item in enumerate(grid, 1):
        label = cfg_item["label"]
        num_envs = cfg_item["num_envs"]
        critic_epochs = cfg_item["critic_epochs"]
        wd = cfg_item["weight_decay"]
        loss_type = cfg_item["critic_loss_type"]
        heads = cfg_item["num_value_heads"]

        print(f"\n[{idx}/{len(grid)}] Running configuration: {label} ...")

        # Compile fresh base config per static shape combination
        cfg = default_config.copy()
        cfg["ENV_NAME"] = args.env_name
        cfg["TOTAL_TIMESTEPS"] = args.total_timesteps
        cfg["NUM_ENVS"] = num_envs
        cfg["NUM_STEPS"] = args.num_steps
        cfg["MINIBATCH_SIZE"] = min(args.minibatch_size, num_envs * args.num_steps)
        cfg["NUM_EPOCHS"] = critic_epochs
        cfg["CRITIC_EPOCHS"] = critic_epochs
        cfg["ACTOR_EPOCHS"] = args.actor_epochs
        cfg["WEIGHT_DECAY"] = wd
        cfg["CRITIC_WEIGHT_DECAY"] = wd
        cfg["ACTOR_WEIGHT_DECAY"] = wd
        cfg["CRITIC_LOSS_TYPE"] = loss_type
        cfg["HUBER_DELTA"] = args.huber_delta
        cfg["NUM_VALUE_HEADS"] = heads
        cfg["VALUE_HEAD_AGG"] = args.value_head_agg
        cfg["ACTOR_LR"] = args.actor_lr
        cfg["LR"] = args.critic_lr
        cfg["CRITIC_LR"] = args.critic_lr

        start_time = time.time()
        train_fn = make_train(cfg)
        train_vjit = jax.jit(jax.vmap(train_fn))

        rng = jax.random.PRNGKey(42)
        rngs = jax.random.split(rng, args.n_seeds)

        out = train_vjit(rngs)
        # Block until JAX execution completes
        _ = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.1f}s ({elapsed / 60.0:.1f} min)")

        # Extract primary metric
        if args.metric in out["metrics"]:
            metric_arr = out["metrics"][args.metric]
        elif "returned_discounted_episode_returns" in out["metrics"]:
            metric_arr = out["metrics"]["returned_discounted_episode_returns"]
        else:
            metric_arr = out["metrics"]["returned_episode_returns"]

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
            "critic_epochs": critic_epochs,
            "weight_decay": wd,
            "critic_loss_type": loss_type,
            "num_value_heads": heads,
            "final_window_mean": final_mean,
            "final_window_sem": final_sem,
            "auc": auc,
            "elapsed_seconds": elapsed,
        })

    # Save summary DataFrame
    df = pd.DataFrame(summary_rows)
    df.sort_values(by="final_window_mean", ascending=False, inplace=True)
    csv_path = os.path.join(out_dir, "summary_e_experimental.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY RANKINGS:")
    print(df.to_string(index=False))
    print("=" * 70)

    # Plot comparison curves (PDF & PNG)
    pdf_path = os.path.join(out_dir, "comparison_curves.pdf")
    png_path = os.path.join(out_dir, "comparison_curves.png")
    plot_comparison_curves(curves_dict, pdf_path, args.env_name, args.metric, args.window_size)
    plot_comparison_curves(curves_dict, png_path, args.env_name, args.metric, args.window_size)

    # Save metrics pickle
    import cloudpickle
    metrics_path = os.path.join(out_dir, "metrics.pkl")
    with open(metrics_path, "wb") as f:
        cloudpickle.dump(all_metrics, f)

    # Generate rich multi-dimensional analysis poster (Parallel coordinates, ANOVA, etc.)
    try:
        generate_multidim_analysis_pdf(out_dir)
    except Exception as ex:
        print(f"Warning: Failed to generate multidimensional analysis poster: {ex}")

    print(f"\nAll artifacts successfully saved to {out_dir}")
    print(f"Sweep completed for {args.env_name}")


if __name__ == "__main__":
    main()

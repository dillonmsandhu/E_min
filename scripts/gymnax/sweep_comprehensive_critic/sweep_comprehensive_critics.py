#!/usr/bin/env python3
"""
sweep_comprehensive_critics.py

Comprehensive Critic Comparison Sweep & Diagnostic Suite:
1. E(lambda) with geometric lookahead jumps (sweeping lambda, selecting best)
2. TD(lambda) fitted value iteration (sweeping lambda, selecting best)
3. E(0) via algos/E.py (with V_aux)
4. MC via algos/mc.py (Monte Carlo rollout return regression)
5. V_aux Regularization Ablations for a single lambda:
   - Baseline (MSE)
   - + Huber Loss (delta=10.0)
   - + Trust Region Regularization (beta=1.0)
   - + Both Combined (Huber + Trust)

Visualizations (Modeled after sampled_lambda_sweep):
- main_comparison_learning_curves: Clean multi-panel grid comparing the key algorithms.
- v_aux_ablations_learning_curves: Multi-panel grid comparing V_aux regularizations against TD(lambda) converged performance.
- main_comparison_auc_table: Publication-grade table with heatmap coloring for AUC.
- main_comparison_final_return_table: Publication-grade table with heatmap coloring for final returns.
- v_aux_ablations_table: Heatmap table for V_aux ablations.
"""

import os
import sys
import argparse
import time
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Publication font formatting
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["axes.linewidth"] = 0.8

# Ensure repository root is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))  # E_min root
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))   # E_min/scripts

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

import jax
import jax.numpy as jnp

from gymnax_rl.algos.ppo import make_train as make_ppo_train

ENV_TITLES = {
    # Classic Control
    "CartPole-v1": "CartPole-v1",
    "Pendulum-v1": "Pendulum-v1",
    "Acrobot-v1": "Acrobot-v1",
    "MountainCar-v0": "Mountain Car (Sparse)",
    "MountainCarContinuous-v0": "Mountain Car (Continuous)",
    # BSuite
    "DeepSea-bsuite": "DeepSea (bsuite)",
    "DiscountingChain-bsuite": "Discounting Chain (bsuite)",
    # Misc / Navigation / Continuous
    "FourRooms-misc": "FourRooms (Sparse)",
    "PointRobot-misc": "PointRobot (Continuous)",
    "Reacher-misc": "Reacher (Continuous)",
    "Swimmer-misc": "Swimmer (Continuous)",
    "Pong-misc": "Pong",
}


def compute_auc(returns_curve):
    """Computes normalized Area Under Curve (AUC) across update timesteps."""
    return float(np.mean(returns_curve))


def run_experiment(config, seed_keys):
    """JIT-compiles and runs vmapped training across random seeds."""
    train_fn = make_ppo_train(config)
    vmapped_fn = jax.jit(jax.vmap(train_fn))
    start_t = time.time()
    out = vmapped_fn(seed_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - start_t

    metrics = out["metrics"]
    res = {
        "returned_episode_returns": np.array(metrics["returned_episode_returns"]),
        "returned_episode_lengths": np.array(metrics["returned_episode_lengths"]),
        "critic_loss": np.array(metrics["value_loss"] if "value_loss" in metrics else metrics.get("val_loss", metrics.get("critic_loss", 0.0))),
        "actor_loss": np.array(metrics["actor_loss"]),
        "entropy": np.array(metrics["entropy"]),
        "elapsed_time": elapsed,
    }
    if "n_terminations" in metrics:
        res["n_terminations"] = np.array(metrics["n_terminations"])
    if "n_truncations" in metrics:
        res["n_truncations"] = np.array(metrics["n_truncations"])
    if "critic_v_mse" in metrics:
        res["critic_v_mse"] = np.array(metrics["critic_v_mse"])
    if "critic_v_aux_mse" in metrics:
        res["critic_v_aux_mse"] = np.array(metrics["critic_v_aux_mse"])
    if "critic_similarity" in metrics:
        res["critic_similarity"] = np.array(metrics["critic_similarity"])
    for k_m, v_m in metrics.items():
        if k_m not in res:
            res[k_m] = np.array(v_m)
    return res


def run_sweep(args):
    """Main sweep execution function."""
    envs = args.envs
    lambdas = args.lambdas
    critic_lrs = args.critic_lrs
    actor_lr = args.actor_lr
    total_timesteps = args.total_timesteps
    num_envs = args.num_envs
    num_steps = args.num_steps
    minibatch_size = args.minibatch_size
    n_seeds = args.n_seeds
    master_seed = args.master_seed
    out_dir = args.out_dir
    ablation_lambda = args.ablation_lambda
    huber_delta = args.huber_delta
    trust_coef = args.trust_coef

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "benchmarks"), exist_ok=True)

    master_key = jax.random.PRNGKey(master_seed)
    seed_keys = jax.random.split(master_key, n_seeds)

    all_runs = {}

    if getattr(args, "load_existing", None) and os.path.exists(args.load_existing):
        print(f"Loading existing metrics from: {args.load_existing}")
        with open(args.load_existing, "rb") as f:
            loaded = pickle.load(f)
            all_runs = loaded.get("runs", {})

    batch_size = args.num_envs * args.num_steps
    num_updates = max(1, args.total_timesteps // batch_size)
    timesteps = np.arange(1, num_updates + 1) * batch_size

    if all_runs:
        for env in envs:
            if env in all_runs:
                for k in all_runs[env]:
                    sub = all_runs[env][k]
                    first_val = list(sub.values())[0]
                    if isinstance(first_val, dict):
                        first_val = list(first_val.values())[0]
                    if "returned_episode_returns" in first_val:
                        curve_len = first_val["returned_episode_returns"].shape[-1]
                        timesteps = np.arange(1, curve_len + 1) * batch_size
                        break

    print(f"=================================================================")
    print(f"STARTING COMPREHENSIVE CRITIC SWEEP & V_AUX ABLATIONS")
    print(f"Envs: {envs}")
    print(f"Rollouts: {num_envs} envs x {num_steps} steps (batch {batch_size})")
    print(f"Updates: {num_updates} | Total timesteps: {total_timesteps}")
    print(f"Critic LRs: {critic_lrs} | Lambdas: {lambdas} | Seeds: {n_seeds}")
    print(f"V_aux Ablation Lambda: {ablation_lambda} (Huber δ={huber_delta}, Trust β={trust_coef})")
    print(f"=================================================================\n")

    for env_name in envs:
        if env_name not in all_runs:
            all_runs[env_name] = {
                "MC": {},
                "E(0)_v_aux": {},
                "E(0)_no_v_aux": {},
                "TD(lambda)": {lr: {} for lr in critic_lrs},
                "E(lambda)": {lr: {} for lr in critic_lrs},
                "E(lambda)_huber": {},
                "E(lambda)_trust": {},
                "E(lambda)_both": {},
            }

        print(f"\n>>> Running Environment: {env_name} <<<")

        # 1. MC Baseline
        for lr in critic_lrs:
            if lr not in all_runs[env_name]["MC"]:
                print(f"[MC] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "mc",
                    "VALUE_LAMBDA": 1.0,
                    "RETURN_LAMBDA": 1.0,
                    "GAE_LAMBDA": 0.8,
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_timesteps,
                    "MINIBATCH_SIZE": minibatch_size,
                    "ACTOR_MINIBATCH_SIZE": minibatch_size,
                    "CRITIC_MINIBATCH_SIZE": minibatch_size,
                    "NUM_EPOCHS": 4,
                    "ACTOR_EPOCHS": 4,
                    "CRITIC_EPOCHS": 4,
                    "DEBUG": False,
                }
                res = run_experiment(cfg, seed_keys)
                all_runs[env_name]["MC"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 2. E(0) with V_aux
        for lr in critic_lrs:
            if lr not in all_runs[env_name]["E(0)_v_aux"]:
                print(f"[E(0) with V_aux] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0",
                    "GAE_LAMBDA": 0.8,
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_timesteps,
                    "MINIBATCH_SIZE": minibatch_size,
                    "ACTOR_MINIBATCH_SIZE": minibatch_size,
                    "CRITIC_MINIBATCH_SIZE": minibatch_size,
                    "NUM_EPOCHS": 4,
                    "ACTOR_EPOCHS": 4,
                    "CRITIC_EPOCHS": 4,
                    "DEBUG": False,
                }
                res = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(0)_v_aux"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 3. E(0) without V_aux
        for lr in critic_lrs:
            if lr not in all_runs[env_name]["E(0)_no_v_aux"]:
                print(f"[E(0) Basic No V_aux] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0_basic",
                    "GAE_LAMBDA": 0.8,
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_timesteps,
                    "MINIBATCH_SIZE": minibatch_size,
                    "ACTOR_MINIBATCH_SIZE": minibatch_size,
                    "CRITIC_MINIBATCH_SIZE": minibatch_size,
                    "NUM_EPOCHS": 4,
                    "ACTOR_EPOCHS": 4,
                    "CRITIC_EPOCHS": 4,
                    "DEBUG": False,
                }
                res = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(0)_no_v_aux"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 4. TD(lambda) Grid
        for lr in critic_lrs:
            for lmb in lambdas:
                if lmb not in all_runs[env_name]["TD(lambda)"][lr]:
                    print(f"[TD(lambda)] env={env_name}, lambda={lmb}, critic_lr={lr}...")
                    cfg = {
                        "ENV_NAME": env_name,
                        "LR": actor_lr,
                        "ACTOR_LR": actor_lr,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "td_lambda",
                        "VALUE_LAMBDA": lmb,
                        "GAE_LAMBDA": 0.8,
                        "NUM_ENVS": num_envs,
                        "NUM_STEPS": num_steps,
                        "TOTAL_TIMESTEPS": total_timesteps,
                        "MINIBATCH_SIZE": minibatch_size,
                        "ACTOR_MINIBATCH_SIZE": minibatch_size,
                        "CRITIC_MINIBATCH_SIZE": minibatch_size,
                        "NUM_EPOCHS": 4,
                        "ACTOR_EPOCHS": 4,
                        "CRITIC_EPOCHS": 4,
                        "DEBUG": False,
                    }
                    res = run_experiment(cfg, seed_keys)
                    all_runs[env_name]["TD(lambda)"][lr][lmb] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 5. E(lambda) Grid
        for lr in critic_lrs:
            for lmb in lambdas:
                if lmb not in all_runs[env_name]["E(lambda)"][lr]:
                    print(f"[E(lambda)] env={env_name}, lambda={lmb}, critic_lr={lr}...")
                    cfg = {
                        "ENV_NAME": env_name,
                        "LR": actor_lr,
                        "ACTOR_LR": actor_lr,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_lambda",
                        "E_LAMBDA": lmb,
                        "JUMP_MODE": "full_mc_invalid_jump",
                        "GAE_LAMBDA": 0.8,
                        "NUM_ENVS": num_envs,
                        "NUM_STEPS": num_steps,
                        "TOTAL_TIMESTEPS": total_timesteps,
                        "MINIBATCH_SIZE": minibatch_size,
                        "ACTOR_MINIBATCH_SIZE": minibatch_size,
                        "CRITIC_MINIBATCH_SIZE": minibatch_size,
                        "NUM_EPOCHS": 4,
                        "ACTOR_EPOCHS": 4,
                        "CRITIC_EPOCHS": 4,
                        "DEBUG": False,
                    }
                    res = run_experiment(cfg, seed_keys)
                    all_runs[env_name]["E(lambda)"][lr][lmb] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 6. V_aux Regularization Ablations (at ablation_lambda)
        print(f"\n--- V_aux Regularization Ablations for {env_name} at λ={ablation_lambda} ---")
        for lr in critic_lrs:
            # A. Huber Loss
            if lr not in all_runs[env_name]["E(lambda)_huber"]:
                print(f"[E(λ={ablation_lambda}) + Huber] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_lambda",
                    "E_LAMBDA": ablation_lambda,
                    "JUMP_MODE": "full_mc_invalid_jump",
                    "GAE_LAMBDA": 0.8,
                    "V_AUX_LOSS_TYPE": "huber",
                    "V_AUX_HUBER_DELTA": huber_delta,
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_timesteps,
                    "MINIBATCH_SIZE": minibatch_size,
                    "ACTOR_MINIBATCH_SIZE": minibatch_size,
                    "CRITIC_MINIBATCH_SIZE": minibatch_size,
                    "NUM_EPOCHS": 4,
                    "ACTOR_EPOCHS": 4,
                    "CRITIC_EPOCHS": 4,
                    "DEBUG": False,
                }
                res = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(lambda)_huber"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

            # B. Trust Region Regularization
            if lr not in all_runs[env_name]["E(lambda)_trust"]:
                print(f"[E(λ={ablation_lambda}) + Trust Region] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_lambda",
                    "E_LAMBDA": ablation_lambda,
                    "JUMP_MODE": "full_mc_invalid_jump",
                    "GAE_LAMBDA": 0.8,
                    "V_AUX_TRUST_COEF": trust_coef,
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_timesteps,
                    "MINIBATCH_SIZE": minibatch_size,
                    "ACTOR_MINIBATCH_SIZE": minibatch_size,
                    "CRITIC_MINIBATCH_SIZE": minibatch_size,
                    "NUM_EPOCHS": 4,
                    "ACTOR_EPOCHS": 4,
                    "CRITIC_EPOCHS": 4,
                    "DEBUG": False,
                }
                res = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(lambda)_trust"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

            # C. Both Combined
            if lr not in all_runs[env_name]["E(lambda)_both"]:
                print(f"[E(λ={ablation_lambda}) + Both (Huber+Trust)] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_lambda",
                    "E_LAMBDA": ablation_lambda,
                    "JUMP_MODE": "full_mc_invalid_jump",
                    "GAE_LAMBDA": 0.8,
                    "V_AUX_LOSS_TYPE": "huber",
                    "V_AUX_HUBER_DELTA": huber_delta,
                    "V_AUX_TRUST_COEF": trust_coef,
                    "NUM_ENVS": num_envs,
                    "NUM_STEPS": num_steps,
                    "TOTAL_TIMESTEPS": total_timesteps,
                    "MINIBATCH_SIZE": minibatch_size,
                    "ACTOR_MINIBATCH_SIZE": minibatch_size,
                    "CRITIC_MINIBATCH_SIZE": minibatch_size,
                    "NUM_EPOCHS": 4,
                    "ACTOR_EPOCHS": 4,
                    "CRITIC_EPOCHS": 4,
                    "DEBUG": False,
                }
                res = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(lambda)_both"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

    # Save raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate Diagnostics & Benchmark Plots
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args)


def plot_curve_with_std(ax, curve, std, label, color, batch_size, lw=2.0, alpha=0.15, linestyle="-"):
    t = np.arange(1, len(curve) + 1) * batch_size
    ax.plot(t, curve, label=label, color=color, lw=lw, linestyle=linestyle)
    if std is not None:
        ax.fill_between(t, curve - std, curve + std, color=color, alpha=alpha)


def render_heatmap_table(df_mean, df_std, title, cbar_label, output_base, cmap="Blues"):
    """
    Renders a publication-grade table with heatmap-colored cells:
    Rows = Environments, Columns = Conditions.
    Each cell contains the formatted numerical text (Mean ± Std),
    with the background color showing relative performance, and
    the best condition highlighted with a bold green border.
    Saves both PNG (300 DPI) and vector PDF.
    """
    n_rows, n_cols = df_mean.shape
    fig_w = max(9.0, n_cols * 1.85 + 2.0)
    fig_h = max(3.5, n_rows * 0.70 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    mat_mean = df_mean.values
    mat_std = df_std.values if df_std is not None else np.zeros_like(mat_mean)

    # Row-wise normalization so relative rank per environment is clear
    row_mins = np.nanmin(mat_mean, axis=1, keepdims=True)
    row_maxs = np.nanmax(mat_mean, axis=1, keepdims=True)
    row_ranges = np.maximum(row_maxs - row_mins, 1e-8)
    mat_norm = (mat_mean - row_mins) / row_ranges

    color_map = getattr(plt.cm, cmap)
    im = ax.imshow(mat_norm, cmap=color_map, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(df_mean.columns, fontsize=10.5, fontweight="bold", rotation=25, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_mean.index, fontsize=10.5, fontweight="bold")

    # Clean cell grid lines
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d7de", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n_rows):
        best_c = np.nanargmax(mat_mean[r])
        for c in range(n_cols):
            val = mat_mean[r, c]
            std = mat_std[r, c]
            norm_val = mat_norm[r, c]

            is_best = (c == best_c)
            txt = f"{val:.1f}\n±{std:.1f}" if abs(val) >= 10 else f"{val:.2f}\n±{std:.2f}"
            text_color = "white" if norm_val > 0.60 else "#111111"
            weight = "bold" if is_best else "normal"

            if is_best:
                rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor="#0b6623", linewidth=2.5)
                ax.add_patch(rect)

            ax.text(c, r, txt, ha="center", va="center", color=text_color, fontsize=9.5, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=16)

    plt.tight_layout()
    png_path = f"{output_base}.png"
    pdf_path = f"{output_base}.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Heatmap Table to {png_path} and {pdf_path}")


def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    """Generates the streamlined publication-quality figures and heatmap tables."""
    batch_size = args.num_envs * args.num_steps
    for env in envs:
        for k in all_runs[env]:
            sub = all_runs[env][k]
            first_val = list(sub.values())[0]
            if isinstance(first_val, dict):
                first_val = list(first_val.values())[0]
            if "returned_episode_returns" in first_val:
                curve_len = first_val["returned_episode_returns"].shape[-1]
                timesteps = np.arange(1, curve_len + 1) * batch_size
                break

    print("\n================ GENERATING ANALYSES & FIGURES ================")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    n_envs = len(envs)

    # Resolve ablation lambda
    abl_lmb = getattr(args, "ablation_lambda", 0.8)
    if abl_lmb not in lambdas:
        abl_lmb = 0.8 if 0.8 in lambdas else lambdas[0]

    # Grid dimensions for learning curves (up to 4 columns per row)
    n_cols = min(n_envs, 4)
    n_rows = math.ceil(n_envs / n_cols)

    # -------------------------------------------------------------
    # 1. MAIN COMPARISON: 5 Core Curves Per Environment
    # -------------------------------------------------------------
    # TD(lambda*), E(lambda*) Baseline, E(lambda*) + Huber + Trust, E(0), MC
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    axes_flat = axes.flatten()

    alg_names = ["MC", "E(0)", "TD(λ*)", "E(λ*) Baseline", "E(λ*) + Huber + Trust"]
    colors = {
        "MC": "#7f7f7f",
        "E(0)": "#1f77b4",
        "TD(λ*)": "#e6550d",
        "E(λ*) Baseline": "#74c476",
        "E(λ*) + Huber + Trust": "#005a32",
    }
    linestyles = {
        "MC": "-",
        "E(0)": "-",
        "TD(λ*)": "--",
        "E(λ*) Baseline": "--",
        "E(λ*) + Huber + Trust": "-",
    }

    auc_means_dict = {alg: [] for alg in alg_names}
    auc_stds_dict = {alg: [] for alg in alg_names}
    final_means_dict = {alg: [] for alg in alg_names}
    final_stds_dict = {alg: [] for alg in alg_names}
    summary_rows = []

    for idx, env in enumerate(envs):
        ax = axes_flat[idx]
        env_title = ENV_TITLES.get(env, env)

        env_curves = {}
        env_stds = {}

        # 1. MC (best lr)
        best_lr_mc = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["MC"][lr]["returned_episode_returns"]))
        mc_data = all_runs[env]["MC"][best_lr_mc]["returned_episode_returns"]
        env_curves["MC"] = mc_data.mean(axis=0)
        env_stds["MC"] = mc_data.std(axis=0)
        auc_means_dict["MC"].append(compute_auc(env_curves["MC"]))
        auc_stds_dict["MC"].append(np.mean(env_stds["MC"]))
        final_means_dict["MC"].append(env_curves["MC"][-1])
        final_stds_dict["MC"].append(env_stds["MC"][-1])

        # 2. E(0) (best lr)
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"]))
        e0_data = all_runs[env]["E(0)_v_aux"][best_lr_e0]["returned_episode_returns"]
        env_curves["E(0)"] = e0_data.mean(axis=0)
        env_stds["E(0)"] = e0_data.std(axis=0)
        auc_means_dict["E(0)"].append(compute_auc(env_curves["E(0)"]))
        auc_stds_dict["E(0)"].append(np.mean(env_stds["E(0)"]))
        final_means_dict["E(0)"].append(env_curves["E(0)"][-1])
        final_stds_dict["E(0)"].append(env_stds["E(0)"][-1])

        # 3. TD(lambda*) (best lr and lambda)
        best_lr_td, best_lmb_td, best_auc_td = None, None, -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_auc = compute_auc(all_runs[env]["TD(lambda)"][clr][lmb]["returned_episode_returns"].mean(axis=0))
                if c_auc > best_auc_td:
                    best_auc_td = c_auc
                    best_lr_td = clr
                    best_lmb_td = lmb
        td_data = all_runs[env]["TD(lambda)"][best_lr_td][best_lmb_td]["returned_episode_returns"]
        env_curves["TD(λ*)"] = td_data.mean(axis=0)
        env_stds["TD(λ*)"] = td_data.std(axis=0)
        auc_means_dict["TD(λ*)"].append(best_auc_td)
        auc_stds_dict["TD(λ*)"].append(np.mean(env_stds["TD(λ*)"]))
        final_means_dict["TD(λ*)"].append(env_curves["TD(λ*)"][-1])
        final_stds_dict["TD(λ*)"].append(env_stds["TD(λ*)"][-1])

        # 4. E(lambda*) Baseline (best lr and lambda)
        best_lr_el, best_lmb_el, best_auc_el = None, None, -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_auc = compute_auc(all_runs[env]["E(lambda)"][clr][lmb]["returned_episode_returns"].mean(axis=0))
                if c_auc > best_auc_el:
                    best_auc_el = c_auc
                    best_lr_el = clr
                    best_lmb_el = lmb
        el_data = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]["returned_episode_returns"]
        env_curves["E(λ*) Baseline"] = el_data.mean(axis=0)
        env_stds["E(λ*) Baseline"] = el_data.std(axis=0)
        auc_means_dict["E(λ*) Baseline"].append(best_auc_el)
        auc_stds_dict["E(λ*) Baseline"].append(np.mean(env_stds["E(λ*) Baseline"]))
        final_means_dict["E(λ*) Baseline"].append(env_curves["E(λ*) Baseline"][-1])
        final_stds_dict["E(λ*) Baseline"].append(env_stds["E(λ*) Baseline"][-1])

        # 5. E(lambda) + Huber + Trust Region (best lr at ablation_lambda)
        best_lr_both = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(lambda)_both"][lr]["returned_episode_returns"]))
        both_data = all_runs[env]["E(lambda)_both"][best_lr_both]["returned_episode_returns"]
        env_curves["E(λ*) + Huber + Trust"] = both_data.mean(axis=0)
        env_stds["E(λ*) + Huber + Trust"] = both_data.std(axis=0)
        auc_means_dict["E(λ*) + Huber + Trust"].append(compute_auc(env_curves["E(λ*) + Huber + Trust"]))
        auc_stds_dict["E(λ*) + Huber + Trust"].append(np.mean(env_stds["E(λ*) + Huber + Trust"]))
        final_means_dict["E(λ*) + Huber + Trust"].append(env_curves["E(λ*) + Huber + Trust"][-1])
        final_stds_dict["E(λ*) + Huber + Trust"].append(env_stds["E(λ*) + Huber + Trust"][-1])

        # Plot curves
        for alg in alg_names:
            c = env_curves[alg]
            s = env_stds[alg]
            lbl = f"{alg} (λ={best_lmb_td})" if alg == "TD(λ*)" else (f"{alg} (λ={best_lmb_el})" if alg == "E(λ*) Baseline" else alg)
            plot_curve_with_std(ax, c, s, lbl, colors[alg], batch_size, lw=2.2 if "Both" in alg or "TD" in alg else 1.8, linestyle=linestyles[alg])

        ax.set_title(env_title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Timesteps", fontsize=9.5)
        ax.set_ylabel("Episodic Return", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="lower right")

        summary_rows.append({
            "Environment": env_title,
            "Best TD(λ) λ": best_lmb_td,
            "Best TD(λ) Return": env_curves["TD(λ*)"][-1],
            "Best TD(λ) AUC": best_auc_td,
            "Best E(λ) Baseline λ": best_lmb_el,
            "Best E(λ) Baseline Return": env_curves["E(λ*) Baseline"][-1],
            "Best E(λ) Baseline AUC": best_auc_el,
            "E(λ) + Huber + Trust Return": env_curves["E(λ*) + Huber + Trust"][-1],
            "E(λ) + Huber + Trust AUC": auc_means_dict["E(λ*) + Huber + Trust"][-1],
            "E(0) Return": env_curves["E(0)"][-1],
            "E(0) AUC": auc_means_dict["E(0)"][-1],
            "MC Return": env_curves["MC"][-1],
            "MC AUC": auc_means_dict["MC"][-1],
        })

    # Hide unused subplots if total envs < n_rows * n_cols
    for extra_idx in range(n_envs, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    plt.tight_layout()
    bench_curves_base = os.path.join(out_dir, "benchmarks", "main_comparison_learning_curves")
    plt.savefig(f"{bench_curves_base}.png", bbox_inches="tight")
    plt.savefig(f"{bench_curves_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved Main Comparison Learning Curves to {bench_curves_base}.png and .pdf")

    # -------------------------------------------------------------
    # 2. V_AUX ABLATIONS LEARNING CURVES (with TD Converged Dotted Line)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    axes_flat = axes.flatten()

    abl_names = ["Baseline (MSE)", "+ Huber", "+ Trust Region", "+ Both"]
    abl_colors = {
        "Baseline (MSE)": "#6baed6",
        "+ Huber": "#2ca02c",
        "+ Trust Region": "#756bb1",
        "+ Both": "#00441b",
    }

    abl_auc_means = {name: [] for name in abl_names}
    abl_auc_stds = {name: [] for name in abl_names}
    abl_final_means = {name: [] for name in abl_names}
    abl_final_stds = {name: [] for name in abl_names}

    for idx, env in enumerate(envs):
        ax = axes_flat[idx]
        env_title = ENV_TITLES.get(env, env)

        best_clr = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(lambda)"][lr][abl_lmb]["returned_episode_returns"]))

        # Converged TD line
        td_curve = all_runs[env]["TD(lambda)"][best_clr][abl_lmb]["returned_episode_returns"].mean(axis=0)
        td_converged = np.mean(td_curve[-max(1, int(len(td_curve)*0.1)):])
        ax.axhline(td_converged, color="#e6550d", linestyle=":", linewidth=2.0, label="TD(λ) Converged Ref")

        # 1. Baseline
        b_data = all_runs[env]["E(lambda)"][best_clr][abl_lmb]["returned_episode_returns"]
        b_curve = b_data.mean(axis=0)
        b_std = b_data.std(axis=0)
        plot_curve_with_std(ax, b_curve, b_std, "Baseline (MSE)", abl_colors["Baseline (MSE)"], batch_size, lw=1.8, linestyle="--")
        abl_auc_means["Baseline (MSE)"].append(compute_auc(b_curve))
        abl_auc_stds["Baseline (MSE)"].append(np.mean(b_std))
        abl_final_means["Baseline (MSE)"].append(b_curve[-1])
        abl_final_stds["Baseline (MSE)"].append(b_std[-1])

        # 2. + Huber
        h_data = all_runs[env]["E(lambda)_huber"][best_clr]["returned_episode_returns"]
        h_curve = h_data.mean(axis=0)
        h_std = h_data.std(axis=0)
        plot_curve_with_std(ax, h_curve, h_std, "+ Huber", abl_colors["+ Huber"], batch_size, lw=2.0)
        abl_auc_means["+ Huber"].append(compute_auc(h_curve))
        abl_auc_stds["+ Huber"].append(np.mean(h_std))
        abl_final_means["+ Huber"].append(h_curve[-1])
        abl_final_stds["+ Huber"].append(h_std[-1])

        # 3. + Trust
        t_data = all_runs[env]["E(lambda)_trust"][best_clr]["returned_episode_returns"]
        t_curve = t_data.mean(axis=0)
        t_std = t_data.std(axis=0)
        plot_curve_with_std(ax, t_curve, t_std, "+ Trust Region", abl_colors["+ Trust Region"], batch_size, lw=2.0)
        abl_auc_means["+ Trust Region"].append(compute_auc(t_curve))
        abl_auc_stds["+ Trust Region"].append(np.mean(t_std))
        abl_final_means["+ Trust Region"].append(t_curve[-1])
        abl_final_stds["+ Trust Region"].append(t_std[-1])

        # 4. + Both
        bo_data = all_runs[env]["E(lambda)_both"][best_clr]["returned_episode_returns"]
        bo_curve = bo_data.mean(axis=0)
        bo_std = bo_data.std(axis=0)
        plot_curve_with_std(ax, bo_curve, bo_std, "+ Both (Huber+Trust)", abl_colors["+ Both"], batch_size, lw=2.4)
        abl_auc_means["+ Both"].append(compute_auc(bo_curve))
        abl_auc_stds["+ Both"].append(np.mean(bo_std))
        abl_final_means["+ Both"].append(bo_curve[-1])
        abl_final_stds["+ Both"].append(bo_std[-1])

        ax.set_title(f"{env_title} (λ={abl_lmb})", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Timesteps", fontsize=9.5)
        ax.set_ylabel("Episodic Return", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0:
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.5, loc="lower right")

    for extra_idx in range(n_envs, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    plt.tight_layout()
    v_aux_curves_base = os.path.join(out_dir, "diagnostics", "v_aux_ablations_learning_curves")
    plt.savefig(f"{v_aux_curves_base}.png", bbox_inches="tight")
    plt.savefig(f"{v_aux_curves_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved V_aux Ablations Learning Curves to {v_aux_curves_base}.png and .pdf")

    # -------------------------------------------------------------
    # 3. REGULAR TABLES WITH HEATMAP COLORING
    # -------------------------------------------------------------
    env_row_labels = [ENV_TITLES.get(e, e) for e in envs]

    # Table 1: Main Comparison AUC Table
    df_auc_mean = pd.DataFrame(auc_means_dict, index=env_row_labels)
    df_auc_std = pd.DataFrame(auc_stds_dict, index=env_row_labels)
    render_heatmap_table(
        df_auc_mean,
        df_auc_std,
        title="AUC Performance Matrix Across Critic Architectures",
        cbar_label="Relative AUC (Row Normalized: 0=Min, 1=Max)",
        output_base=os.path.join(out_dir, "benchmarks", "main_comparison_auc_table"),
        cmap="Blues",
    )

    # Table 2: Main Comparison Final Return Table
    df_final_mean = pd.DataFrame(final_means_dict, index=env_row_labels)
    df_final_std = pd.DataFrame(final_stds_dict, index=env_row_labels)
    render_heatmap_table(
        df_final_mean,
        df_final_std,
        title="Final Episodic Return Matrix Across Critic Architectures",
        cbar_label="Relative Return (Row Normalized: 0=Min, 1=Max)",
        output_base=os.path.join(out_dir, "benchmarks", "main_comparison_final_return_table"),
        cmap="Blues",
    )

    # Table 3: V_aux Ablations Table
    df_abl_mean = pd.DataFrame(abl_auc_means, index=env_row_labels)
    df_abl_std = pd.DataFrame(abl_auc_stds, index=env_row_labels)
    render_heatmap_table(
        df_abl_mean,
        df_abl_std,
        title=f"V_aux Regularization AUC Matrix (λ={abl_lmb})",
        cbar_label="Relative AUC (Row Normalized)",
        output_base=os.path.join(out_dir, "diagnostics", "v_aux_ablations_table"),
        cmap="YlGn",
    )

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, "benchmarks", "main_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved Summary CSV to {csv_path}")
    print("\nAll figures and diagnostics generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Critic Comparison Sweep & Diagnostic Suite")
    parser.add_argument("--envs", nargs="+", default=["CartPole-v1"], help="List of environments")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.4, 0.8, 0.95], help="Lambdas to sweep")
    parser.add_argument("--critic-lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3], help="Critic learning rates")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--total-timesteps", type=int, default=300000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments (rollouts)")
    parser.add_argument("--num-steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--minibatch-size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n-seeds", type=int, default=3, help="Random seeds per configuration")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--out-dir", type=str, default="results/critic_sweep", help="Output directory")
    parser.add_argument("--ablation-lambda", type=float, default=0.8, help="Single lambda value to test V_aux regularizations")
    parser.add_argument("--huber-delta", type=float, default=10.0, help="Huber loss delta for V_aux ablation")
    parser.add_argument("--trust-coef", type=float, default=1.0, help="Trust-region L2 coefficient for V_aux ablation")
    parser.add_argument("--load-existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl to skip already completed runs")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

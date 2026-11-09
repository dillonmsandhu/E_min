#!/usr/bin/env python3
"""
sweep_comprehensive_critics.py

Comprehensive Critic Comparison Sweep & Automated Cross-Algorithm Analysis Suite:
Runs per environment across critic learning rates and lambdas:
1. MC (Monte Carlo rollout return regression, lambda=1.0)
2. TD(lambda) fitted value iteration grid (lambdas: [0.0, 0.4, 0.8, 0.95])
3. E(0) variants:
   - Naive (no v_aux, full symmetric (e_i - e_j)^2)
   - Forward-Only (no v_aux, semi-gradient (e_i - sg(e_j))^2)
   - Basic v_aux (surgical v_aux with AUX_LAMBDA=1.0)
   - v_aux across AUX_LAMBDA in [1.0, 0.9, 0.8]
4. E(lambda) grid (lambdas: [0.0, 0.4, 0.8, 0.95] with v_aux, AUX_LAMBDA=1.0)
5. E(lambda=0.8) variants:
   - Naive (no v_aux)
   - Forward-Only (no v_aux)
   - Basic v_aux (AUX_LAMBDA=1.0)
   - v_aux across AUX_LAMBDA in [1.0, 0.9, 0.8]

Automatically produces 4 cross-algorithm analyses per environment set, each with:
- Learning curve figures under <out_dir>/plots/ (.pdf and .png)
- Cross-algorithm tables (Envs = rows, Algos = columns) under <out_dir>/tables/ (.md, .tex, .pdf, .png):
  1. Analysis 1: E(0) comparison (Naive, Forward-Only, v_aux [lambda_aux=1.0])
  2. Analysis 2: E(lambda=0.8) comparison (Naive, Forward-Only, v_aux [lambda_aux=1.0])
  3. Analysis 3: v_aux across AUX_LAMBDA values (1.0, 0.9, 0.8)
  4. Analysis 4: Best E run vs Best TD run
"""

from __future__ import annotations
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

AUX_LAMBDAS_DEFAULT = [1.0, 0.9, 0.8]


def get_env_default_timesteps(env_name: str, fallback_timesteps: int | None = None) -> int:
    """Returns environment-specific timesteps: 1M on MountainCar envs, 300k on others."""
    if fallback_timesteps is not None:
        return fallback_timesteps
    if "MountainCar" in env_name:
        return 1_000_000
    else:
        return 300_000


def compute_auc(returns_curve):
    """Computes normalized Area Under Curve (AUC) across update timesteps."""
    return float(np.mean(returns_curve))


def get_metric_curve(run_dict, metric_key="returned_discounted_episode_returns"):
    """Extracts the specified metric curve of shape (n_seeds, n_updates)."""
    if metric_key in run_dict:
        return np.array(run_dict[metric_key])
    elif "returned_discounted_episode_returns" in run_dict:
        return np.array(run_dict["returned_discounted_episode_returns"])
    elif "returned_episode_returns" in run_dict:
        return np.array(run_dict["returned_episode_returns"])
    elif "metrics" in run_dict:
        m = run_dict["metrics"]
        if metric_key in m:
            return np.array(m[metric_key])
        elif "returned_discounted_episode_returns" in m:
            return np.array(m["returned_discounted_episode_returns"])
        elif "returned_episode_returns" in m:
            return np.array(m["returned_episode_returns"])
    raise KeyError(f"Metric {metric_key} not found in run data.")


def select_best_lr_run(lr_dict, critic_lrs, metric_key):
    """Selects the run with the highest mean AUC across available critic_lrs."""
    available_lrs = [lr for lr in critic_lrs if lr in lr_dict]
    if not available_lrs:
        available_lrs = list(lr_dict.keys())
    best_lr = max(available_lrs, key=lambda lr: compute_auc(get_metric_curve(lr_dict[lr], metric_key).mean(axis=0)))
    data = get_metric_curve(lr_dict[best_lr], metric_key)
    mean_curve = data.mean(axis=0)
    std_curve = data.std(axis=0)
    seed_aucs = data.mean(axis=-1)
    auc_mean = float(seed_aucs.mean())
    auc_std = float(seed_aucs.std())
    final_mean = float(mean_curve[-1])
    final_std = float(std_curve[-1])
    return {
        "lr": best_lr,
        "data": data,
        "curve": mean_curve,
        "std": std_curve,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "final_mean": final_mean,
        "final_std": final_std,
    }


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
        "returned_discounted_episode_returns": np.array(
            metrics.get("returned_discounted_episode_returns", metrics["returned_episode_returns"])
        ),
        "returned_episode_lengths": np.array(metrics["returned_episode_lengths"]),
        "critic_loss": np.array(
            metrics["value_loss"] if "value_loss" in metrics else metrics.get("val_loss", metrics.get("critic_loss", 0.0))
        ),
        "actor_loss": np.array(metrics["actor_loss"]),
        "entropy": np.array(metrics["entropy"]),
        "elapsed_time": elapsed,
    }
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
    num_envs = args.num_envs
    num_steps = args.num_steps
    minibatch_size = args.minibatch_size
    n_seeds = args.n_seeds
    master_seed = args.master_seed
    out_dir = args.out_dir
    ablation_lambda = getattr(args, "ablation_lambda", 0.8)
    aux_lambdas = getattr(args, "aux_lambdas", AUX_LAMBDAS_DEFAULT)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    master_key = jax.random.PRNGKey(master_seed)
    seed_keys = jax.random.split(master_key, n_seeds)

    all_runs = {}

    if getattr(args, "load_existing", None) and os.path.exists(args.load_existing):
        print(f"Loading existing metrics from: {args.load_existing}")
        with open(args.load_existing, "rb") as f:
            loaded = pickle.load(f)
            all_runs = loaded.get("runs", {})

    batch_size = args.num_envs * args.num_steps
    base_timesteps = args.total_timesteps or 300000
    num_updates = max(1, base_timesteps // batch_size)
    timesteps = np.arange(1, num_updates + 1) * batch_size

    print("=================================================================")
    print("STARTING COMPREHENSIVE CRITIC SWEEP")
    print(f"Envs: {envs}")
    print(f"Rollouts: {num_envs} envs x {num_steps} steps (batch {batch_size})")
    print(f"Critic LRs: {critic_lrs} | Lambdas: {lambdas} | Seeds: {n_seeds}")
    print(f"Ablation Lambda: {ablation_lambda} | Aux Lambdas: {aux_lambdas}")
    print("=================================================================\n")

    for env_name in envs:
        env_timesteps = get_env_default_timesteps(env_name, args.total_timesteps)
        if env_name not in all_runs:
            all_runs[env_name] = {}

        env_dict = all_runs[env_name]
        env_dict.setdefault("MC", {})
        env_dict.setdefault("E(0)_v_aux", {})
        env_dict.setdefault("E(0)_no_v_aux", {})
        env_dict.setdefault("E(0)_fwd", {})
        env_dict.setdefault("E(0)_aux_lam", {alam: {} for alam in aux_lambdas})
        env_dict.setdefault("TD(lambda)", {lr: {} for lr in critic_lrs})
        env_dict.setdefault("E(lambda)", {lr: {} for lr in critic_lrs})
        env_dict.setdefault("E(lambda)_naive", {})
        env_dict.setdefault("E(lambda)_fwd", {})
        env_dict.setdefault("E(lambda)_aux_lam", {alam: {} for alam in aux_lambdas})

        for lr in critic_lrs:
            env_dict["TD(lambda)"].setdefault(lr, {})
            env_dict["E(lambda)"].setdefault(lr, {})
        for alam in aux_lambdas:
            env_dict["E(0)_aux_lam"].setdefault(alam, {})
            env_dict["E(lambda)_aux_lam"].setdefault(alam, {})

        base_env_cfg = {
            "ENV_NAME": env_name,
            "LR": actor_lr,
            "ACTOR_LR": actor_lr,
            "GAE_LAMBDA": 0.8,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": env_timesteps,
            "MINIBATCH_SIZE": minibatch_size,
            "ACTOR_MINIBATCH_SIZE": minibatch_size,
            "CRITIC_MINIBATCH_SIZE": minibatch_size,
            "NUM_EPOCHS": 4,
            "ACTOR_EPOCHS": 4,
            "CRITIC_EPOCHS": 4,
            "DEBUG": False,
        }

        print(f"\n>>> Running Environment: {env_name} (Timesteps: {env_timesteps:,}) <<<")

        # 1. MC Baseline
        for lr in critic_lrs:
            if lr not in env_dict["MC"]:
                print(f"[MC] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "mc",
                    "VALUE_LAMBDA": 1.0,
                    "RETURN_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["MC"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 2. E(0) Variants: v_aux (lambda_aux=1.0), Naive, Forward-Only, and aux_lambda in {1.0, 0.9, 0.8}
        for lr in critic_lrs:
            if lr not in env_dict["E(0)_v_aux"]:
                print(f"[E(0) v_aux (lambda_aux=1.0)] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0",
                    "E_VARIANT": "aux",
                    "RETURN_LAMBDA": 1.0,
                    "AUX_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(0)_v_aux"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")
            env_dict["E(0)_aux_lam"][1.0][lr] = env_dict["E(0)_v_aux"][lr]

            if lr not in env_dict["E(0)_no_v_aux"]:
                print(f"[E(0) Naive (No v_aux)] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0",
                    "E_VARIANT": "naive",
                    "RETURN_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(0)_no_v_aux"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

            if lr not in env_dict["E(0)_fwd"]:
                print(f"[E(0) Forward-Only] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0",
                    "E_VARIANT": "fwd",
                    "RETURN_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(0)_fwd"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

            for alam in aux_lambdas:
                if alam == 1.0:
                    continue
                if lr not in env_dict["E(0)_aux_lam"][alam]:
                    print(f"[E(0) v_aux (lambda_aux={alam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_env_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_0",
                        "E_VARIANT": "aux",
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": alam,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["E(0)_aux_lam"][alam][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 3. TD(lambda) Grid
        for lr in critic_lrs:
            for lmb in lambdas:
                if lmb not in env_dict["TD(lambda)"][lr]:
                    print(f"[TD(lambda)] env={env_name}, lambda={lmb}, critic_lr={lr}...")
                    cfg = {
                        **base_env_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "td_lambda",
                        "VALUE_LAMBDA": lmb,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["TD(lambda)"][lr][lmb] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 4. E(lambda) Grid (with v_aux, AUX_LAMBDA=1.0)
        all_e_lambdas = list(lambdas)
        if ablation_lambda not in all_e_lambdas:
            all_e_lambdas.append(ablation_lambda)
        for lr in critic_lrs:
            for lmb in all_e_lambdas:
                if lmb not in env_dict["E(lambda)"][lr]:
                    print(f"[E(lambda) v_aux (lambda_aux=1.0)] env={env_name}, lambda={lmb}, critic_lr={lr}...")
                    cfg = {
                        **base_env_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_lambda",
                        "E_LAMBDA": lmb,
                        "E_VARIANT": "aux",
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": 1.0,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["E(lambda)"][lr][lmb] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")
            env_dict["E(lambda)_aux_lam"][1.0][lr] = env_dict["E(lambda)"][lr][ablation_lambda]

        # 5. E(lambda = ablation_lambda) Variants: Naive, Forward-Only, and AUX_LAMBDA in {1.0, 0.9, 0.8}
        print(f"\n--- E(lambda={ablation_lambda}) Variants & AUX_LAMBDA Sweep for {env_name} ---")
        for lr in critic_lrs:
            if lr not in env_dict["E(lambda)_naive"]:
                print(f"[E(lambda={ablation_lambda}) Naive] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_lambda",
                    "E_LAMBDA": ablation_lambda,
                    "E_VARIANT": "naive",
                    "RETURN_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(lambda)_naive"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

            if lr not in env_dict["E(lambda)_fwd"]:
                print(f"[E(lambda={ablation_lambda}) Forward-Only] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_lambda",
                    "E_LAMBDA": ablation_lambda,
                    "E_VARIANT": "fwd",
                    "RETURN_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(lambda)_fwd"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

            for alam in aux_lambdas:
                if alam == 1.0:
                    continue
                if lr not in env_dict["E(lambda)_aux_lam"][alam]:
                    print(f"[E(lambda={ablation_lambda}) v_aux (lambda_aux={alam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_env_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_lambda",
                        "E_LAMBDA": ablation_lambda,
                        "E_VARIANT": "aux",
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": alam,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["E(lambda)_aux_lam"][alam][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

    # Save raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate all 4 cross-algorithm analyses (plots + md/tex/pdf tables)
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args)


def plot_curve_with_std(ax, curve, std, label, color, batch_size, lw=2.0, alpha=0.15, linestyle="-"):
    t = np.arange(1, len(curve) + 1) * batch_size
    ax.plot(t, curve, label=label, color=color, lw=lw, linestyle=linestyle)
    if std is not None:
        ax.fill_between(t, curve - std, curve + std, color=color, alpha=alpha)


def _format_val_std(val, std):
    if np.isnan(val):
        return "N/A"
    if abs(val) >= 10:
        return f"{val:.1f} ± {std:.1f}"
    return f"{val:.2f} ± {std:.2f}"


def _format_val_std_tex(val, std, is_best=False):
    if np.isnan(val):
        return "N/A"
    if abs(val) >= 10:
        s = f"{val:.1f} \\pm {std:.1f}"
    else:
        s = f"{val:.2f} \\pm {std:.2f}"
    return f"$\\mathbf{{{s}}}$" if is_best else f"${s}$"


def _escape_tex(text: str) -> str:
    rep = (
        str(text)
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("λ", "$\\lambda$")
    )
    return rep


def render_heatmap_table(df_mean, df_std, title, cbar_label, output_base, cmap="Blues"):
    """
    Renders a publication-grade table with heatmap-colored cells:
    Rows = Environments, Columns = Algorithms.
    Saves both PDF and PNG (300 DPI).
    """
    n_rows, n_cols = df_mean.shape
    fig_w = max(8.5, n_cols * 2.1 + 2.2)
    fig_h = max(3.2, n_rows * 0.72 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    mat_mean = df_mean.values.astype(float)
    mat_std = df_std.values.astype(float) if df_std is not None else np.zeros_like(mat_mean)

    row_mins = np.nanmin(mat_mean, axis=1, keepdims=True)
    row_maxs = np.nanmax(mat_mean, axis=1, keepdims=True)
    row_ranges = np.maximum(row_maxs - row_mins, 1e-8)
    mat_norm = (mat_mean - row_mins) / row_ranges

    color_map = getattr(plt.cm, cmap)
    im = ax.imshow(mat_norm, cmap=color_map, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(df_mean.columns, fontsize=10.5, fontweight="bold", rotation=20, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_mean.index, fontsize=10.5, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d7de", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n_rows):
        best_c = int(np.nanargmax(mat_mean[r]))
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
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14)

    plt.tight_layout()
    png_path = f"{output_base}.png"
    pdf_path = f"{output_base}.pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def save_table_bundle(df_mean, df_std, title, cbar_label, output_base, cmap="Blues"):
    """
    Saves a cross-algorithm table (Envs = row headers, Algos = column headers) in:
    - Markdown (.md)
    - LaTeX (.tex)
    - Vector PDF (.pdf) & PNG (.png)
    """
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    cols = list(df_mean.columns)
    rows = list(df_mean.index)
    mat_mean = df_mean.values.astype(float)
    mat_std = df_std.values.astype(float) if df_std is not None else np.zeros_like(mat_mean)

    # 1. Markdown (.md)
    md_lines = [f"# {title}\n"]
    header = "| Environment | " + " | ".join(cols) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |"
    md_lines.append(header)
    md_lines.append(sep)

    for r_idx, env_label in enumerate(rows):
        best_c = int(np.nanargmax(mat_mean[r_idx]))
        cells = []
        for c_idx in range(len(cols)):
            cell_str = _format_val_std(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx])
            if c_idx == best_c:
                cell_str = f"**{cell_str}**"
            cells.append(cell_str)
        md_lines.append(f"| **{env_label}** | " + " | ".join(cells) + " |")

    if len(rows) > 1:
        col_means = np.nanmean(mat_mean, axis=0)
        col_stds = np.nanmean(mat_std, axis=0)
        best_mean_c = int(np.nanargmax(col_means))
        mean_cells = []
        for c_idx in range(len(cols)):
            cell_str = _format_val_std(col_means[c_idx], col_stds[c_idx])
            if c_idx == best_mean_c:
                cell_str = f"**{cell_str}**"
            mean_cells.append(cell_str)
        md_lines.append(f"| **Mean** | " + " | ".join(mean_cells) + " |")

    md_path = f"{output_base}.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # 2. LaTeX (.tex)
    col_spec = "l" + ("c" * len(cols))
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_escape_tex(title)}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "\\textbf{Environment} & " + " & ".join([f"\\textbf{{{_escape_tex(c)}}}" for c in cols]) + " \\\\",
        "\\midrule",
    ]
    for r_idx, env_label in enumerate(rows):
        best_c = int(np.nanargmax(mat_mean[r_idx]))
        cells = [
            _format_val_std_tex(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx], is_best=(c_idx == best_c))
            for c_idx in range(len(cols))
        ]
        tex_lines.append(f"{_escape_tex(env_label)} & " + " & ".join(cells) + " \\\\")

    if len(rows) > 1:
        tex_lines.append("\\midrule")
        col_means = np.nanmean(mat_mean, axis=0)
        col_stds = np.nanmean(mat_std, axis=0)
        best_mean_c = int(np.nanargmax(col_means))
        mean_cells = [
            _format_val_std_tex(col_means[c_idx], col_stds[c_idx], is_best=(c_idx == best_mean_c))
            for c_idx in range(len(cols))
        ]
        tex_lines.append("\\textbf{Mean} & " + " & ".join(mean_cells) + " \\\\")

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    tex_path = f"{output_base}.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    # 3. PDF & PNG Heatmap Table
    render_heatmap_table(df_mean, df_std, title, cbar_label, output_base, cmap=cmap)
    print(f"Saved table bundle (.md, .tex, .pdf, .png): {output_base}")


def render_analysis_plot_and_tables(
    envs,
    alg_names,
    per_env_results,
    colors,
    linestyles,
    batch_size,
    plot_title_suffix,
    plot_output_base,
    table_output_base,
    table_title_prefix,
    cmap="Blues",
):
    """
    Renders a multi-panel learning curve plot under plots/ (.pdf and .png)
    and saves AUC + Final Return table bundles (.md, .tex, .pdf, .png) under tables/.
    """
    n_envs = len(envs)
    n_cols = min(n_envs, 4)
    n_rows = math.ceil(n_envs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    axes_flat = axes.flatten()

    auc_means = {alg: [] for alg in alg_names}
    auc_stds = {alg: [] for alg in alg_names}
    final_means = {alg: [] for alg in alg_names}
    final_stds = {alg: [] for alg in alg_names}
    env_row_labels = []

    for idx, env in enumerate(envs):
        ax = axes_flat[idx]
        env_title = ENV_TITLES.get(env, env)
        env_row_labels.append(env_title)

        for alg in alg_names:
            info = per_env_results[env][alg]
            lbl = info.get("legend_label", alg)
            plot_curve_with_std(
                ax,
                info["curve"],
                info["std"],
                lbl,
                colors[alg],
                batch_size,
                lw=2.2,
                linestyle=linestyles.get(alg, "-"),
            )
            auc_means[alg].append(info["auc_mean"])
            auc_stds[alg].append(info["auc_std"])
            final_means[alg].append(info["final_mean"])
            final_stds[alg].append(info["final_std"])

        ax.set_title(f"{env_title}{plot_title_suffix}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Timesteps", fontsize=9.5)
        ax.set_ylabel("Discounted Return", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=7.8, loc="best")

    for extra_idx in range(n_envs, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_output_base), exist_ok=True)
    plt.savefig(f"{plot_output_base}.png", bbox_inches="tight")
    plt.savefig(f"{plot_output_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved learning curves plot (.pdf, .png): {plot_output_base}")

    df_auc_mean = pd.DataFrame(auc_means, index=env_row_labels)
    df_auc_std = pd.DataFrame(auc_stds, index=env_row_labels)
    df_final_mean = pd.DataFrame(final_means, index=env_row_labels)
    df_final_std = pd.DataFrame(final_stds, index=env_row_labels)

    save_table_bundle(
        df_auc_mean,
        df_auc_std,
        title=f"{table_title_prefix} — Area Under Curve (AUC)",
        cbar_label="Relative AUC (Row Normalized)",
        output_base=f"{table_output_base}_auc",
        cmap=cmap,
    )
    save_table_bundle(
        df_final_mean,
        df_final_std,
        title=f"{table_title_prefix} — Final Episodic Return",
        cbar_label="Relative Final Return (Row Normalized)",
        output_base=f"{table_output_base}_final",
        cmap=cmap,
    )
    # Also write the primary table name (Final Return) at table_output_base.{md,tex,pdf,png}
    save_table_bundle(
        df_final_mean,
        df_final_std,
        title=f"{table_title_prefix} (Final Return)",
        cbar_label="Relative Final Return (Row Normalized)",
        output_base=table_output_base,
        cmap=cmap,
    )


def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args):
    """
    Produces the 4 automated cross-algorithm analyses:
    1. E(0) variants: Naive, Forward-Only, and basic v_aux (lambda_aux = 1.0)
    2. E(lambda=0.8) variants: Naive, Forward-Only, and basic v_aux (lambda_aux = 1.0)
    3. v_aux across aux_lambda values (1.0, 0.9, 0.8)
    4. Best E run vs Best TD run
    Each analysis outputs learning curves under <out_dir>/plots/ and (.md, .tex, .pdf) tables under <out_dir>/tables/.
    """
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    metric_key = getattr(args, "metric_key", "returned_discounted_episode_returns")
    batch_size = args.num_envs * args.num_steps
    abl_lmb = getattr(args, "ablation_lambda", 0.8)
    aux_lambdas = getattr(args, "aux_lambdas", AUX_LAMBDAS_DEFAULT)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    print(f"\n================ GENERATING 4 AUTOMATED ANALYSES (Metric: {metric_key}) ================")

    # =========================================================================
    # ANALYSIS 1: E(0) with Naive, Forward-Only, and Basic v_aux (lambda_aux=1.0)
    # =========================================================================
    a1_algs = ["E(0) Naive", "E(0) Forward-Only", "E(0) v_aux (λ_aux=1.0)"]
    a1_colors = {
        "E(0) Naive": "#9467bd",
        "E(0) Forward-Only": "#ff7f0e",
        "E(0) v_aux (λ_aux=1.0)": "#1f77b4",
    }
    a1_styles = {
        "E(0) Naive": "--",
        "E(0) Forward-Only": "-.",
        "E(0) v_aux (λ_aux=1.0)": "-",
    }
    a1_results = {}
    for env in envs:
        env_d = all_runs[env]
        a1_results[env] = {
            "E(0) Naive": select_best_lr_run(env_d["E(0)_no_v_aux"], critic_lrs, metric_key),
            "E(0) Forward-Only": select_best_lr_run(env_d["E(0)_fwd"], critic_lrs, metric_key),
            "E(0) v_aux (λ_aux=1.0)": select_best_lr_run(env_d["E(0)_v_aux"], critic_lrs, metric_key),
        }

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a1_algs,
        per_env_results=a1_results,
        colors=a1_colors,
        linestyles=a1_styles,
        batch_size=batch_size,
        plot_title_suffix=" — E(0) Variants",
        plot_output_base=os.path.join(plots_dir, "e0_variants_learning_curves"),
        table_output_base=os.path.join(tables_dir, "e0_variants"),
        table_title_prefix="E(0) Critic Variants Comparison",
        cmap="Blues",
    )

    # =========================================================================
    # ANALYSIS 2: E(lambda=0.8) with Naive, Forward-Only, and Basic v_aux (lambda_aux=1.0)
    # =========================================================================
    a2_algs = [
        f"E({abl_lmb}) Naive",
        f"E({abl_lmb}) Forward-Only",
        f"E({abl_lmb}) v_aux (λ_aux=1.0)",
    ]
    a2_colors = {
        f"E({abl_lmb}) Naive": "#9467bd",
        f"E({abl_lmb}) Forward-Only": "#ff7f0e",
        f"E({abl_lmb}) v_aux (λ_aux=1.0)": "#2ca02c",
    }
    a2_styles = {
        f"E({abl_lmb}) Naive": "--",
        f"E({abl_lmb}) Forward-Only": "-.",
        f"E({abl_lmb}) v_aux (λ_aux=1.0)": "-",
    }
    a2_results = {}
    for env in envs:
        env_d = all_runs[env]
        el_lmb_lr_dict = {
            lr: env_d["E(lambda)"][lr][abl_lmb]
            for lr in env_d["E(lambda)"]
            if abl_lmb in env_d["E(lambda)"][lr]
        }
        a2_results[env] = {
            f"E({abl_lmb}) Naive": select_best_lr_run(env_d["E(lambda)_naive"], critic_lrs, metric_key),
            f"E({abl_lmb}) Forward-Only": select_best_lr_run(env_d["E(lambda)_fwd"], critic_lrs, metric_key),
            f"E({abl_lmb}) v_aux (λ_aux=1.0)": select_best_lr_run(el_lmb_lr_dict, critic_lrs, metric_key),
        }

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a2_algs,
        per_env_results=a2_results,
        colors=a2_colors,
        linestyles=a2_styles,
        batch_size=batch_size,
        plot_title_suffix=f" — E(λ={abl_lmb}) Variants",
        plot_output_base=os.path.join(plots_dir, "e_lambda08_variants_learning_curves"),
        table_output_base=os.path.join(tables_dir, "e_lambda08_variants"),
        table_title_prefix=f"E(λ={abl_lmb}) Critic Variants Comparison",
        cmap="Blues",
    )

    # =========================================================================
    # ANALYSIS 3: Comparing v_aux at Different AUX_LAMBDA Values (1.0, 0.9, 0.8)
    # =========================================================================
    a3_algs = [f"v_aux (λ_aux={alam})" for alam in aux_lambdas]
    palette_a3 = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    styles_a3 = ["--", "-.", "-", ":"]
    a3_colors = {alg: palette_a3[i % len(palette_a3)] for i, alg in enumerate(a3_algs)}
    a3_styles = {alg: styles_a3[i % len(styles_a3)] for i, alg in enumerate(a3_algs)}
    a3_results = {}
    for env in envs:
        env_d = all_runs[env]
        a3_results[env] = {}
        for alam in aux_lambdas:
            alg_label = f"v_aux (λ_aux={alam})"
            lr_dict = env_d["E(lambda)_aux_lam"].get(alam, {})
            if not lr_dict and alam == 1.0:
                lr_dict = {
                    lr: env_d["E(lambda)"][lr][abl_lmb]
                    for lr in env_d["E(lambda)"]
                    if abl_lmb in env_d["E(lambda)"][lr]
                }
            a3_results[env][alg_label] = select_best_lr_run(lr_dict, critic_lrs, metric_key)

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a3_algs,
        per_env_results=a3_results,
        colors=a3_colors,
        linestyles=a3_styles,
        batch_size=batch_size,
        plot_title_suffix=f" — v_aux λ_aux Sweep (E(λ={abl_lmb}))",
        plot_output_base=os.path.join(plots_dir, "v_aux_lambda_comparison_learning_curves"),
        table_output_base=os.path.join(tables_dir, "v_aux_lambda_comparison"),
        table_title_prefix=f"v_aux Auxiliary Return Lambda (λ_aux) Comparison at E(λ={abl_lmb})",
        cmap="YlGn",
    )

    # =========================================================================
    # ANALYSIS 4: Comparing Best E Run to Best TD Run
    # =========================================================================
    a4_algs = ["Best TD(λ*)", "Best E (v_aux)", "Best E (All)"]
    a4_colors = {
        "Best TD(λ*)": "#e6550d",
        "Best E (v_aux)": "#1f77b4",
        "Best E (All)": "#2ca02c",
    }
    a4_styles = {
        "Best TD(λ*)": "--",
        "Best E (v_aux)": "-",
        "Best E (All)": "-",
    }
    a4_results = {}
    for env in envs:
        env_d = all_runs[env]

        # 1. Best TD(lambda*) across all (lr, lmb)
        best_td_info = None
        for lr, lmb_dict in env_d["TD(lambda)"].items():
            for lmb in lmb_dict:
                res_info = select_best_lr_run({lr: lmb_dict[lmb]}, [lr], metric_key)
                res_info["legend_label"] = f"Best TD (λ={lmb})"
                if best_td_info is None or res_info["auc_mean"] > best_td_info["auc_mean"]:
                    best_td_info = res_info

        # 2. Best E (v_aux): best across E(0)_v_aux, E(0)_aux_lam, E(lambda), E(lambda)_aux_lam
        v_aux_candidates = []
        if env_d.get("E(0)_v_aux"):
            info = select_best_lr_run(env_d["E(0)_v_aux"], critic_lrs, metric_key)
            info["legend_label"] = "Best E v_aux (E(0), λ_aux=1.0)"
            v_aux_candidates.append(info)
        for alam, lr_d in env_d.get("E(0)_aux_lam", {}).items():
            if lr_d:
                info = select_best_lr_run(lr_d, critic_lrs, metric_key)
                info["legend_label"] = f"Best E v_aux (E(0), λ_aux={alam})"
                v_aux_candidates.append(info)
        for lr, lmb_dict in env_d.get("E(lambda)", {}).items():
            for lmb in lmb_dict:
                info = select_best_lr_run({lr: lmb_dict[lmb]}, [lr], metric_key)
                info["legend_label"] = f"Best E v_aux (λ={lmb}, λ_aux=1.0)"
                v_aux_candidates.append(info)
        for alam, lr_d in env_d.get("E(lambda)_aux_lam", {}).items():
            if lr_d:
                info = select_best_lr_run(lr_d, critic_lrs, metric_key)
                info["legend_label"] = f"Best E v_aux (λ={abl_lmb}, λ_aux={alam})"
                v_aux_candidates.append(info)

        best_e_v_aux_info = max(v_aux_candidates, key=lambda x: x["auc_mean"])

        # 3. Best E (All): best across v_aux + Forward-Only + Naive
        all_e_candidates = list(v_aux_candidates)
        if env_d.get("E(0)_no_v_aux"):
            info = select_best_lr_run(env_d["E(0)_no_v_aux"], critic_lrs, metric_key)
            info["legend_label"] = "Best E (E(0) Naive)"
            all_e_candidates.append(info)
        if env_d.get("E(0)_fwd"):
            info = select_best_lr_run(env_d["E(0)_fwd"], critic_lrs, metric_key)
            info["legend_label"] = "Best E (E(0) Fwd)"
            all_e_candidates.append(info)
        if env_d.get("E(lambda)_naive"):
            info = select_best_lr_run(env_d["E(lambda)_naive"], critic_lrs, metric_key)
            info["legend_label"] = f"Best E (E({abl_lmb}) Naive)"
            all_e_candidates.append(info)
        if env_d.get("E(lambda)_fwd"):
            info = select_best_lr_run(env_d["E(lambda)_fwd"], critic_lrs, metric_key)
            info["legend_label"] = f"Best E (E({abl_lmb}) Fwd)"
            all_e_candidates.append(info)

        best_e_all_info = max(all_e_candidates, key=lambda x: x["auc_mean"])

        a4_results[env] = {
            "Best TD(λ*)": best_td_info,
            "Best E (v_aux)": best_e_v_aux_info,
            "Best E (All)": best_e_all_info,
        }

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a4_algs,
        per_env_results=a4_results,
        colors=a4_colors,
        linestyles=a4_styles,
        batch_size=batch_size,
        plot_title_suffix=" — Best E vs Best TD",
        plot_output_base=os.path.join(plots_dir, "best_e_vs_best_td_learning_curves"),
        table_output_base=os.path.join(tables_dir, "best_e_vs_best_td"),
        table_title_prefix="Best E Run vs Best TD Run Comparison",
        cmap="Blues",
    )

    print("\nAll 4 cross-algorithm analyses (plots and .md/.tex/.pdf tables) generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Critic Comparison Sweep & Automated Analysis Suite")
    parser.add_argument("--envs", nargs="+", default=["CartPole-v1"], help="List of environments")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.4, 0.8, 0.95], help="Lambdas to sweep")
    parser.add_argument("--aux-lambdas", nargs="+", type=float, default=AUX_LAMBDAS_DEFAULT, help="Auxiliary critic lambdas to sweep")
    parser.add_argument("--critic-lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3], help="Critic learning rates")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps (default: auto per env: 1M for MountainCar, 300k for small envs)")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments (rollouts)")
    parser.add_argument("--num-steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--minibatch-size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n-seeds", type=int, default=3, help="Random seeds per configuration")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--out-dir", type=str, default="results/sweep_comprehensive_critic", help="Output directory")
    parser.add_argument("--ablation-lambda", type=float, default=0.8, help="Lambda for E(lambda) variants & v_aux sweep")
    parser.add_argument("--metric-key", type=str, default="returned_discounted_episode_returns", help="Metric key to plot")
    parser.add_argument("--load-existing", type=str, default=None, help="Load existing sweep_metrics_all.pkl and regenerate plots/tables only")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

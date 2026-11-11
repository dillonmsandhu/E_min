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
    # MinAtar
    "Asterix-MinAtar": "Asterix (MinAtar)",
    "Breakout-MinAtar": "Breakout (MinAtar)",
    "Freeway-MinAtar": "Freeway (MinAtar)",
    "SpaceInvaders-MinAtar": "Space Invaders (MinAtar)",
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
    if not lr_dict:
        return {
            "lr": None,
            "data": np.array([np.nan]),
            "curve": np.array([np.nan]),
            "std": np.array([np.nan]),
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "final_mean": float("nan"),
            "final_std": float("nan"),
            "legend_label": "N/A",
        }
    available_lrs = [lr for lr in critic_lrs if lr in lr_dict]
    if not available_lrs:
        available_lrs = list(lr_dict.keys())
    if not available_lrs:
        return {
            "lr": None,
            "data": np.array([np.nan]),
            "curve": np.array([np.nan]),
            "std": np.array([np.nan]),
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "final_mean": float("nan"),
            "final_std": float("nan"),
            "legend_label": "N/A",
        }
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
        env_dict.setdefault("E(0)_e_aux", {})
        env_dict.setdefault("E(0)_aux_lam", {alam: {} for alam in aux_lambdas})
        env_dict.setdefault("E(0)_e_aux_lam", {alam: {} for alam in aux_lambdas})
        env_dict.setdefault("TD(lambda)", {lr: {} for lr in critic_lrs})
        env_dict.setdefault("E(lambda)", {lr: {} for lr in critic_lrs})
        env_dict.setdefault("E(lambda)_naive", {})
        env_dict.setdefault("E(lambda)_fwd", {})
        env_dict.setdefault("E(lambda)_e_aux", {})
        env_dict.setdefault("E(lambda)_aux_lam", {alam: {} for alam in aux_lambdas})
        env_dict.setdefault("E(lambda)_e_aux_lam", {alam: {} for alam in aux_lambdas})

        for lr in critic_lrs:
            env_dict["TD(lambda)"].setdefault(lr, {})
            env_dict["E(lambda)"].setdefault(lr, {})
        for alam in aux_lambdas:
            env_dict["E(0)_aux_lam"].setdefault(alam, {})
            env_dict["E(0)_e_aux_lam"].setdefault(alam, {})
            env_dict["E(lambda)_aux_lam"].setdefault(alam, {})
            env_dict["E(lambda)_e_aux_lam"].setdefault(alam, {})

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

            if lr not in env_dict["E(0)_e_aux"]:
                print(f"[E(0) e_aux (lambda_aux=1.0)] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0",
                    "E_VARIANT": "e_aux",
                    "RETURN_LAMBDA": 1.0,
                    "AUX_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(0)_e_aux"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")
            env_dict["E(0)_e_aux_lam"][1.0][lr] = env_dict["E(0)_e_aux"][lr]

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

                if lr not in env_dict["E(0)_e_aux_lam"][alam]:
                    print(f"[E(0) e_aux (lambda_aux={alam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_env_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_0",
                        "E_VARIANT": "e_aux",
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": alam,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["E(0)_e_aux_lam"][alam][lr] = res
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

            if lr not in env_dict["E(lambda)_e_aux"]:
                print(f"[E(lambda={ablation_lambda}) e_aux (lambda_aux=1.0)] env={env_name}, critic_lr={lr}...")
                cfg = {
                    **base_env_cfg,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_lambda",
                    "E_LAMBDA": ablation_lambda,
                    "E_VARIANT": "e_aux",
                    "RETURN_LAMBDA": 1.0,
                    "AUX_LAMBDA": 1.0,
                }
                res = run_experiment(cfg, seed_keys)
                env_dict["E(lambda)_e_aux"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")
            env_dict["E(lambda)_e_aux_lam"][1.0][lr] = env_dict["E(lambda)_e_aux"][lr]

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

                if lr not in env_dict["E(lambda)_e_aux_lam"][alam]:
                    print(f"[E(lambda={ablation_lambda}) e_aux (lambda_aux={alam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_env_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_lambda",
                        "E_LAMBDA": ablation_lambda,
                        "E_VARIANT": "e_aux",
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": alam,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["E(lambda)_e_aux_lam"][alam][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

    # Save raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate all 4 cross-algorithm analyses (plots + md/tex/pdf tables)
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args)


def plot_curve_with_std(ax, curve, std, label, color, batch_size, lw=2.0, alpha=0.15, linestyle="-"):
    if np.isnan(curve).all():
        return
    t = np.arange(1, len(curve) + 1) * batch_size
    ax.plot(t, curve, label=label, color=color, lw=lw, linestyle=linestyle)
    if std is not None and not np.isnan(std).all():
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


def clean_variant_name(lbl: str) -> str:
    """Formats raw candidate legend labels into clean variant names for table annotations."""
    if not lbl:
        return ""
    if lbl.startswith("Best E (") and lbl.endswith(")"):
        return lbl[len("Best E ("):-1]
    if lbl.startswith("Best E v_aux "):
        return "v_aux " + lbl[len("Best E v_aux "):]
    if lbl.startswith("Best E e_aux "):
        return "e_aux " + lbl[len("Best E e_aux "):]
    if lbl.startswith("Best E "):
        return lbl[len("Best E "):]
    return lbl


REAL_WIN_REL_TOL = 0.01  # 1% of environment return magnitude


def _row_real_win_tol(vals, rel_tol=REAL_WIN_REL_TOL):
    """Computes a meaningful difference threshold relative to the row's return magnitude."""
    vals = np.asarray(vals, dtype=float)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return 1e-4
    scale = max(float(np.max(np.abs(valid))), 1e-8)
    return max(rel_tol * scale, 1e-6)


def _best_col_mask(vals, rel_tol=REAL_WIN_REL_TOL):
    """Returns boolean mask of columns within rel_tol of the row maximum (wins + ties for 1st)."""
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return np.zeros(len(vals), dtype=bool)
    mx = np.nanmax(vals)
    tol = _row_real_win_tol(vals, rel_tol=rel_tol)
    return valid & (vals >= mx - tol)


def _max_int_mask(vals):
    """Returns boolean mask of columns achieving the maximum integer/score value."""
    vals = np.asarray(vals, dtype=float)
    mx = np.max(vals)
    return vals >= mx - 1e-9


def _compute_wtl_stats(mat_mean, rel_tol=REAL_WIN_REL_TOL):
    """
    Computes:
    - Top-1 (W/T/L): outright env wins, ties for 1st (within rel_tol), and losses.
    - Pairwise (W/T/L): head-to-head wins, ties, losses across all opponent-env pairs.
    """
    n_rows, n_cols = mat_mean.shape
    top_w = np.zeros(n_cols, dtype=int)
    top_t = np.zeros(n_cols, dtype=int)
    top_l = np.zeros(n_cols, dtype=int)
    pw_w = np.zeros(n_cols, dtype=int)
    pw_t = np.zeros(n_cols, dtype=int)
    pw_l = np.zeros(n_cols, dtype=int)

    for r in range(n_rows):
        row = mat_mean[r]
        valid = ~np.isnan(row)
        if not np.any(valid):
            continue
        tol = _row_real_win_tol(row, rel_tol=rel_tol)
        mx = np.nanmax(row)
        is_top = valid & (row >= mx - tol)
        n_top = int(np.sum(is_top))

        for c in range(n_cols):
            if not valid[c]:
                continue
            if is_top[c]:
                if n_top == 1:
                    top_w[c] += 1
                else:
                    top_t[c] += 1
            else:
                top_l[c] += 1

            for c2 in range(n_cols):
                if c == c2 or not valid[c2]:
                    continue
                if row[c] > row[c2] + tol:
                    pw_w[c] += 1
                elif row[c2] > row[c] + tol:
                    pw_l[c] += 1
                else:
                    pw_t[c] += 1

    return top_w, top_t, top_l, pw_w, pw_t, pw_l


def render_clean_table(df_mean, df_std, title, output_base, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL, focus_cols=None):
    """
    Renders a clean, high-contrast academic table (no heatmap backgrounds).
    If focus_cols is provided (e.g. [2, 3] for v_aux vs e_aux with baselines on the left),
    draws a vertical separator before focus_cols[0], highlights 1v1 winners within focus_cols,
    and shows both Direct 1v1 (focus_cols) W/T/L and All-Column Top-1 W/T/L summary rows.
    Saves both vector PDF (.pdf) and 300 DPI PNG (.png).
    """
    cols = list(df_mean.columns)
    rows = list(df_mean.index)
    mat_mean = df_mean.values.astype(float)
    mat_std = df_std.values.astype(float) if df_std is not None else np.zeros_like(mat_mean)

    n_rows, n_cols = mat_mean.shape
    has_ann = bool(cell_annotations and any(v for v in cell_annotations.values()))

    col_labels = ["Environment"] + [str(c) for c in cols]
    table_data = []
    best_masks = []

    for r_idx, env_label in enumerate(rows):
        if focus_cols is not None:
            sub_mask = _best_col_mask(mat_mean[r_idx, focus_cols], rel_tol=rel_tol)
            mask = np.zeros(n_cols, dtype=bool)
            for idx_k, fc in enumerate(focus_cols):
                mask[fc] = sub_mask[idx_k]
        else:
            mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
        best_masks.append(mask)

        row_cells = [str(env_label)]
        for c_idx in range(n_cols):
            val_str = _format_val_std(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx])
            ann = cell_annotations.get((r_idx, c_idx), "") if cell_annotations else ""
            if ann:
                row_cells.append(f"{val_str}\n({ann})")
            else:
                row_cells.append(val_str)
        table_data.append(row_cells)

    if n_rows > 1:
        col_means = np.nanmean(mat_mean, axis=0)
        col_stds = np.nanmean(mat_std, axis=0)
        mean_mask = _best_col_mask(col_means, rel_tol=rel_tol)
        best_masks.append(mean_mask)
        mean_cells = ["Mean"] + [_format_val_std(col_means[c], col_stds[c]) for c in range(n_cols)]
        table_data.append(mean_cells)

        if focus_cols is not None:
            sub_mat = mat_mean[:, focus_cols]
            fw, ft, fl, _, _, _ = _compute_wtl_stats(sub_mat, rel_tol=rel_tol)
            f_scores = fw - fl
            f_win = _max_int_mask(f_scores)
            f_mask = np.zeros(n_cols, dtype=bool)
            f_cells = ["Direct 1v1 (v_aux vs e_aux, 1%)"] + ["Baseline" for _ in range(n_cols)]
            for idx_k, fc in enumerate(focus_cols):
                f_mask[fc] = f_win[idx_k]
                f_cells[fc + 1] = f"{fw[idx_k]}W / {ft[idx_k]}T / {fl[idx_k]}L"
            best_masks.append(f_mask)
            table_data.append(f_cells)

        top_w, top_t, top_l, pw_w, pw_t, pw_l = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)
        top_scores = top_w + 0.5 * top_t
        best_masks.append(_max_int_mask(top_scores))
        top_label = "Top-1 vs All (W / T / L, 1%)" if focus_cols is not None else "Top-1 (W / T / L, 1%)"
        top_cells = [top_label] + [
            f"{top_w[c]}W / {top_t[c]}T / {top_l[c]}L" for c in range(n_cols)
        ]
        table_data.append(top_cells)

        if n_cols > 2 and focus_cols is None:
            pw_scores = pw_w - pw_l
            best_masks.append(_max_int_mask(pw_scores))
            pw_cells = ["Pairwise (W / T / L, 1%)"] + [
                f"{pw_w[c]}W / {pw_t[c]}T / {pw_l[c]}L\n(net {pw_w[c] - pw_l[c]:+d})" for c in range(n_cols)
            ]
            table_data.append(pw_cells)

    total_rows = len(table_data)
    total_cols = len(col_labels)

    col_char_widths = []
    for c_i in range(total_cols):
        max_len = len(col_labels[c_i])
        for row_cells in table_data:
            for line in str(row_cells[c_i]).split("\n"):
                max_len = max(max_len, len(line))
        col_char_widths.append(max(max_len + 4, 15))

    total_chars = sum(col_char_widths)
    col_widths = [w / total_chars for w in col_char_widths]

    fig_w = max(8.5, total_chars * 0.115)
    row_unit_h = 0.56 if (has_ann or (n_cols > 2 and focus_cols is None)) else 0.44
    fig_h = max(3.4, (total_rows + 1) * row_unit_h + 1.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.92],
    )
    tbl.auto_set_font_size(False)

    focus_set = set(focus_cols) if focus_cols is not None else None

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.75)

        if r == 0:
            val_c = c - 1
            if focus_set is not None and c > 0 and val_c not in focus_set:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(weight="bold", color="#475569", fontsize=9.8)
            else:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(weight="bold", color="#0f172a", fontsize=10.0)
            cell.set_edgecolor("#64748b")
            cell.set_linewidth(1.1)
            if c == 0:
                cell.set_text_props(ha="left")
        else:
            data_r = r - 1
            is_summary = data_r >= n_rows
            if is_summary:
                bg = "#f1f5f9"
                cell.set_edgecolor("#94a3b8")
                cell.set_linewidth(0.9)
            else:
                bg = "#ffffff" if (data_r % 2 == 0) else "#f8fafc"

            if c == 0:
                cell.set_facecolor(bg)
                cell.set_text_props(weight="bold", color="#0f172a", fontsize=9.5, ha="left")
            else:
                val_c = c - 1
                is_baseline = focus_set is not None and val_c not in focus_set
                is_best = bool(best_masks[data_r][val_c])
                if is_best:
                    cell.set_facecolor("#eff6ff")
                    cell.set_text_props(weight="bold", color="#0f172a", fontsize=9.0 if has_ann else 9.5)
                elif is_baseline:
                    cell.set_facecolor("#fafafa" if not is_summary else bg)
                    cell.set_text_props(weight="normal", color="#475569", fontsize=9.0 if has_ann else 9.5)
                else:
                    cell.set_facecolor(bg)
                    cell.set_text_props(weight="normal", color="#1e293b", fontsize=9.0 if has_ann else 9.5)

    # Draw crisp vertical separator before focus_cols[0] to separate baselines from unbiased methods
    if focus_cols is not None and len(focus_cols) > 0:
        sep_col_idx = focus_cols[0] + 1
        x_sep = sum(col_widths[:sep_col_idx])
        ax.plot([x_sep, x_sep], [0.0, 0.92], transform=ax.transAxes, color="#334155", linewidth=2.0, zorder=10)

    ax.set_title(title, fontsize=12.0, fontweight="bold", color="#0f172a", pad=12, loc="center")

    plt.tight_layout()
    plt.savefig(f"{output_base}.png", bbox_inches="tight", facecolor="white")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def save_table_bundle(df_mean, df_std, title, cbar_label, output_base, cmap="Blues", cell_annotations=None, rel_tol=REAL_WIN_REL_TOL, focus_cols=None):
    """
    Saves a cross-algorithm table (Envs = row headers, Algos = column headers) in:
    - Markdown (.md)
    - CSV (.csv)
    - LaTeX (.tex)
    - Clean Normal Table Vector PDF (.pdf) & PNG (.png)
    Includes real-win (>=1%) W/T/L statistics.
    """
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    cols = list(df_mean.columns)
    rows = list(df_mean.index)
    mat_mean = df_mean.values.astype(float)
    mat_std = df_std.values.astype(float) if df_std is not None else np.zeros_like(mat_mean)
    n_cols = len(cols)

    # 1. Markdown (.md) & CSV assembly
    md_lines = [f"# {title}\n", f"*Real-win threshold: >= {rel_tol*100:.0f}% of environment return scale.*\n"]
    header = "| Environment | " + " | ".join(cols) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |"
    md_lines.append(header)
    md_lines.append(sep)

    csv_rows = []
    for r_idx, env_label in enumerate(rows):
        if focus_cols is not None:
            sub_mask = _best_col_mask(mat_mean[r_idx, focus_cols], rel_tol=rel_tol)
            best_mask = np.zeros(n_cols, dtype=bool)
            for idx_k, fc in enumerate(focus_cols):
                best_mask[fc] = sub_mask[idx_k]
        else:
            best_mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)

        md_cells = []
        csv_row = {"Environment": env_label}
        for c_idx, col_name in enumerate(cols):
            val_std_str = _format_val_std(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx])
            ann = cell_annotations.get((r_idx, c_idx), "") if cell_annotations else ""

            cell_md = f"**{val_std_str}**" if best_mask[c_idx] else val_std_str
            if ann:
                cell_md = f"{cell_md} *({ann})*"
            md_cells.append(cell_md)

            csv_row[col_name] = f"{val_std_str} ({ann})" if ann else val_std_str

        md_lines.append(f"| **{env_label}** | " + " | ".join(md_cells) + " |")
        csv_rows.append(csv_row)

    if len(rows) > 1:
        col_means = np.nanmean(mat_mean, axis=0)
        col_stds = np.nanmean(mat_std, axis=0)
        mean_mask = _best_col_mask(col_means, rel_tol=rel_tol)
        mean_cells_md = []
        csv_mean = {"Environment": "Mean"}
        for c_idx, col_name in enumerate(cols):
            val_std_str = _format_val_std(col_means[c_idx], col_stds[c_idx])
            mean_cells_md.append(f"**{val_std_str}**" if mean_mask[c_idx] else val_std_str)
            csv_mean[col_name] = val_std_str
        md_lines.append("| **Mean** | " + " | ".join(mean_cells_md) + " |")
        csv_rows.append(csv_mean)

        if focus_cols is not None:
            sub_mat = mat_mean[:, focus_cols]
            fw, ft, fl, _, _, _ = _compute_wtl_stats(sub_mat, rel_tol=rel_tol)
            f_win = _max_int_mask(fw - fl)
            f_cells_md = ["*Baseline*" for _ in range(n_cols)]
            csv_f = {"Environment": "Direct 1v1 (v_aux vs e_aux, 1%)"}
            for c_idx, col_name in enumerate(cols):
                csv_f[col_name] = "Baseline"
            for idx_k, fc in enumerate(focus_cols):
                s = f"{fw[idx_k]}W / {ft[idx_k]}T / {fl[idx_k]}L"
                f_cells_md[fc] = f"**{s}**" if f_win[idx_k] else s
                csv_f[cols[fc]] = s
            md_lines.append("| **Direct 1v1 (v_aux vs e_aux, 1%)** | " + " | ".join(f_cells_md) + " |")
            csv_rows.append(csv_f)

        top_w, top_t, top_l, pw_w, pw_t, pw_l = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)
        top_mask = _max_int_mask(top_w + 0.5 * top_t)
        top_cells_md = []
        top_lbl = "Top-1 vs All (W / T / L, 1%)" if focus_cols is not None else "Top-1 (W / T / L, 1%)"
        csv_top = {"Environment": top_lbl}
        for c_idx, col_name in enumerate(cols):
            s = f"{top_w[c_idx]}W / {top_t[c_idx]}T / {top_l[c_idx]}L"
            top_cells_md.append(f"**{s}**" if top_mask[c_idx] else s)
            csv_top[col_name] = s
        md_lines.append(f"| **{top_lbl}** | " + " | ".join(top_cells_md) + " |")
        csv_rows.append(csv_top)

        if n_cols > 2 and focus_cols is None:
            pw_mask = _max_int_mask(pw_w - pw_l)
            pw_cells_md = []
            csv_pw = {"Environment": "Pairwise (W / T / L, 1%)"}
            for c_idx, col_name in enumerate(cols):
                s = f"{pw_w[c_idx]}W / {pw_t[c_idx]}T / {pw_l[c_idx]}L (net {pw_w[c_idx] - pw_l[c_idx]:+d})"
                pw_cells_md.append(f"**{s}**" if pw_mask[c_idx] else s)
                csv_pw[col_name] = s
            md_lines.append("| **Pairwise (W / T / L, 1%)** | " + " | ".join(pw_cells_md) + " |")
            csv_rows.append(csv_pw)

    with open(f"{output_base}.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # 2. CSV (.csv)
    pd.DataFrame(csv_rows).to_csv(f"{output_base}.csv", index=False)

    # 3. LaTeX (.tex)
    if focus_cols is not None and len(focus_cols) > 0:
        sep_idx = focus_cols[0]
        col_spec = "l" + ("c" * sep_idx) + "|" + ("c" * (n_cols - sep_idx))
    else:
        col_spec = "l" + ("c" * n_cols)

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
        if focus_cols is not None:
            sub_mask = _best_col_mask(mat_mean[r_idx, focus_cols], rel_tol=rel_tol)
            best_mask = np.zeros(n_cols, dtype=bool)
            for idx_k, fc in enumerate(focus_cols):
                best_mask[fc] = sub_mask[idx_k]
        else:
            best_mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
        cells = []
        for c_idx in range(n_cols):
            base_tex = _format_val_std_tex(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx], is_best=bool(best_mask[c_idx]))
            ann = cell_annotations.get((r_idx, c_idx), "") if cell_annotations else ""
            if ann:
                base_tex = f"{base_tex} \\scriptsize{{({_escape_tex(ann)})}}"
            cells.append(base_tex)
        tex_lines.append(f"{_escape_tex(env_label)} & " + " & ".join(cells) + " \\\\")

    if len(rows) > 1:
        tex_lines.append("\\midrule")
        col_means = np.nanmean(mat_mean, axis=0)
        col_stds = np.nanmean(mat_std, axis=0)
        mean_mask = _best_col_mask(col_means, rel_tol=rel_tol)
        mean_cells = [
            _format_val_std_tex(col_means[c_idx], col_stds[c_idx], is_best=bool(mean_mask[c_idx]))
            for c_idx in range(n_cols)
        ]
        tex_lines.append("\\textbf{Mean} & " + " & ".join(mean_cells) + " \\\\")

        if focus_cols is not None:
            sub_mat = mat_mean[:, focus_cols]
            fw, ft, fl, _, _, _ = _compute_wtl_stats(sub_mat, rel_tol=rel_tol)
            f_win = _max_int_mask(fw - fl)
            f_cells = ["\\textit{Baseline}" for _ in range(n_cols)]
            for idx_k, fc in enumerate(focus_cols):
                s = f"{fw[idx_k]}W / {ft[idx_k]}T / {fl[idx_k]}L"
                f_cells[fc] = f"\\textbf{{{s}}}" if f_win[idx_k] else s
            tex_lines.append("\\textbf{Direct 1v1 (v\\_aux vs e\\_aux, 1\\%)} & " + " & ".join(f_cells) + " \\\\")

        top_w, top_t, top_l, pw_w, pw_t, pw_l = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)
        top_mask = _max_int_mask(top_w + 0.5 * top_t)
        top_cells = [
            f"\\textbf{{{top_w[c]}W / {top_t[c]}T / {top_l[c]}L}}" if top_mask[c] else f"{top_w[c]}W / {top_t[c]}T / {top_l[c]}L"
            for c in range(n_cols)
        ]
        top_tex_lbl = "Top-1 vs All (W/T/L, 1\\%)" if focus_cols is not None else "Top-1 (W/T/L, 1\\%)"
        tex_lines.append(f"\\textbf{{{top_tex_lbl}}} & " + " & ".join(top_cells) + " \\\\")

        if n_cols > 2 and focus_cols is None:
            pw_mask = _max_int_mask(pw_w - pw_l)
            pw_cells = [
                f"\\textbf{{{pw_w[c]}W / {pw_t[c]}T / {pw_l[c]}L}}" if pw_mask[c] else f"{pw_w[c]}W / {pw_t[c]}T / {pw_l[c]}L"
                for c in range(n_cols)
            ]
            tex_lines.append("\\textbf{Pairwise (W/T/L, 1\\%)} & " + " & ".join(pw_cells) + " \\\\")

    tex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    with open(f"{output_base}.tex", "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    # 4. Clean Normal Table Vector PDF (.pdf) & PNG (.png)
    render_clean_table(df_mean, df_std, title, output_base, cell_annotations=cell_annotations, rel_tol=rel_tol, focus_cols=focus_cols)
    print(f"Saved table bundle (.md, .csv, .tex, .pdf, .png): {output_base}")


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
    cell_annotations=None,
    per_env_results_final=None,
    focus_cols=None,
    y_label="Discounted Return",
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

    res_final = per_env_results_final if per_env_results_final is not None else per_env_results

    for idx, env in enumerate(envs):
        ax = axes_flat[idx]
        env_title = ENV_TITLES.get(env, env)
        env_row_labels.append(env_title)

        for alg in alg_names:
            info = per_env_results[env][alg]
            info_f = res_final[env][alg]
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
            final_means[alg].append(info_f["final_mean"])
            final_stds[alg].append(info_f["final_std"])

        ax.set_title(f"{env_title}{plot_title_suffix}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Timesteps", fontsize=9.5)
        ax.set_ylabel(y_label, fontsize=9.5)
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

    if isinstance(cell_annotations, dict) and ("final" in cell_annotations or "auc" in cell_annotations):
        ann_final = cell_annotations.get("final", {})
        ann_auc = cell_annotations.get("auc", {})
    else:
        ann_final = cell_annotations
        ann_auc = cell_annotations

    save_table_bundle(
        df_auc_mean,
        df_auc_std,
        title=f"{table_title_prefix} — Area Under Curve (AUC)",
        cbar_label="Relative AUC (Row Normalized)",
        output_base=f"{table_output_base}_auc",
        cmap=cmap,
        cell_annotations=ann_auc,
        focus_cols=focus_cols,
    )
    save_table_bundle(
        df_final_mean,
        df_final_std,
        title=f"{table_title_prefix} — Final Episodic Return",
        cbar_label="Relative Final Return (Row Normalized)",
        output_base=f"{table_output_base}_final",
        cmap=cmap,
        cell_annotations=ann_final,
        focus_cols=focus_cols,
    )
    # Also write the primary table name (Final Return) at table_output_base.{md,tex,pdf,png}
    save_table_bundle(
        df_final_mean,
        df_final_std,
        title=f"{table_title_prefix} (Final Return)",
        cbar_label="Relative Final Return (Row Normalized)",
        output_base=table_output_base,
        cmap=cmap,
        cell_annotations=ann_final,
        focus_cols=focus_cols,
    )


def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args):
    """
    Produces the comprehensive cross-algorithm analyses:
    1. E(0) variants: Forward-Only, e_aux, v_aux, Naive
    2. E(lambda=0.8) variants: Forward-Only, e_aux, v_aux, Naive
    3. Cross-paradigm E variants comparison (Forward-Only vs e_aux vs v_aux vs Naive)
    4. Auxiliary mechanism comparison (e_aux vs v_aux head-to-head)
    5. e_aux across aux_lambda values (1.0, 0.9, 0.8)
    6. v_aux across aux_lambda values (1.0, 0.9, 0.8)
    7. Best E run vs Best TD run (with Best E (All) annotated with winning variant)
    Each analysis outputs learning curves under <out_dir>/plots/ and (.md, .tex, .pdf, .png) tables under <out_dir>/tables/.
    """
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    metric_key = getattr(args, "metric_key", "returned_discounted_episode_returns")
    batch_size = args.num_envs * args.num_steps
    abl_lmb = getattr(args, "ablation_lambda", 0.8)
    aux_lambdas = getattr(args, "aux_lambdas", AUX_LAMBDAS_DEFAULT)

    # Check whether any env has e_aux runs
    has_e_aux = any(
        bool(all_runs[env].get("E(0)_e_aux") or all_runs[env].get("E(lambda)_e_aux"))
        for env in envs
    )

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    print(f"\n================ GENERATING AUTOMATED ANALYSES (Metric: {metric_key}, has_e_aux={has_e_aux}) ================")

    # =========================================================================
    # STORY 1A: E(0) Variants (Naive, Forward-Only | v_aux, e_aux) at λ_aux=1.0
    # =========================================================================
    if has_e_aux:
        a1_algs = ["E(0) Naive", "E(0) Forward-Only", "E(0) v_aux (λ_aux=1.0)", "E(0) e_aux (λ_aux=1.0)"]
        a1_colors = {
            "E(0) Naive": "#9467bd",
            "E(0) Forward-Only": "#ff7f0e",
            "E(0) v_aux (λ_aux=1.0)": "#1f77b4",
            "E(0) e_aux (λ_aux=1.0)": "#17becf",
        }
        a1_styles = {
            "E(0) Naive": "--",
            "E(0) Forward-Only": "-.",
            "E(0) v_aux (λ_aux=1.0)": "-",
            "E(0) e_aux (λ_aux=1.0)": "-",
        }
        a1_focus = [2, 3]
    else:
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
        a1_focus = None

    a1_results = {}
    for env in envs:
        env_d = all_runs[env]
        res_d = {
            "E(0) Naive": select_best_lr_run(env_d.get("E(0)_no_v_aux", {}), critic_lrs, metric_key),
            "E(0) Forward-Only": select_best_lr_run(env_d.get("E(0)_fwd", {}), critic_lrs, metric_key),
            "E(0) v_aux (λ_aux=1.0)": select_best_lr_run(env_d.get("E(0)_v_aux", {}), critic_lrs, metric_key),
        }
        if has_e_aux:
            res_d["E(0) e_aux (λ_aux=1.0)"] = select_best_lr_run(env_d.get("E(0)_e_aux", {}), critic_lrs, metric_key)
        a1_results[env] = res_d

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a1_algs,
        per_env_results=a1_results,
        colors=a1_colors,
        linestyles=a1_styles,
        batch_size=batch_size,
        plot_title_suffix=" — Story 1: E(0) Variants",
        plot_output_base=os.path.join(plots_dir, "story1_e0_variants_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story1_e0_variants"),
        table_title_prefix="Story 1: E(0) Critic Variants (Baselines vs v_aux vs e_aux)",
        cmap="Blues",
        focus_cols=a1_focus,
    )

    # =========================================================================
    # STORY 1B: E(lambda=0.8) Variants (Naive, Forward-Only | v_aux, e_aux) at λ_aux=1.0
    # =========================================================================
    if has_e_aux:
        a2_algs = [
            f"E({abl_lmb}) Naive",
            f"E({abl_lmb}) Forward-Only",
            f"E({abl_lmb}) v_aux (λ_aux=1.0)",
            f"E({abl_lmb}) e_aux (λ_aux=1.0)",
        ]
        a2_colors = {
            f"E({abl_lmb}) Naive": "#9467bd",
            f"E({abl_lmb}) Forward-Only": "#ff7f0e",
            f"E({abl_lmb}) v_aux (λ_aux=1.0)": "#2ca02c",
            f"E({abl_lmb}) e_aux (λ_aux=1.0)": "#17becf",
        }
        a2_styles = {
            f"E({abl_lmb}) Naive": "--",
            f"E({abl_lmb}) Forward-Only": "-.",
            f"E({abl_lmb}) v_aux (λ_aux=1.0)": "-",
            f"E({abl_lmb}) e_aux (λ_aux=1.0)": "-",
        }
        a2_focus = [2, 3]
    else:
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
        a2_focus = None

    a2_results = {}
    for env in envs:
        env_d = all_runs[env]
        el_lmb_lr_dict = {
            lr: env_d["E(lambda)"][lr][abl_lmb]
            for lr in env_d.get("E(lambda)", {})
            if abl_lmb in env_d["E(lambda)"][lr]
        }
        res_d = {
            f"E({abl_lmb}) Naive": select_best_lr_run(env_d.get("E(lambda)_naive", {}), critic_lrs, metric_key),
            f"E({abl_lmb}) Forward-Only": select_best_lr_run(env_d.get("E(lambda)_fwd", {}), critic_lrs, metric_key),
            f"E({abl_lmb}) v_aux (λ_aux=1.0)": select_best_lr_run(el_lmb_lr_dict, critic_lrs, metric_key),
        }
        if has_e_aux:
            res_d[f"E({abl_lmb}) e_aux (λ_aux=1.0)"] = select_best_lr_run(env_d.get("E(lambda)_e_aux", {}), critic_lrs, metric_key)
        a2_results[env] = res_d

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a2_algs,
        per_env_results=a2_results,
        colors=a2_colors,
        linestyles=a2_styles,
        batch_size=batch_size,
        plot_title_suffix=f" — Story 1: E(λ={abl_lmb}) Variants",
        plot_output_base=os.path.join(plots_dir, "story1_e_lambda08_variants_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story1_e_lambda08_variants"),
        table_title_prefix=f"Story 1: E(λ={abl_lmb}) Critic Variants (Baselines vs v_aux vs e_aux)",
        cmap="Blues",
        focus_cols=a2_focus,
    )

    # =========================================================================
    # STORY 1 (PRIMARY): E Variants Comparison (Naive, Forward-Only | v_aux, e_aux)
    # Apples-to-apples over shared lambda in {0.0, ablation_lambda}
    # =========================================================================
    if has_e_aux:
        a3_algs = ["Naive", "Forward-Only", "v_aux", "e_aux"]
        a3_colors = {
            "Naive": "#9467bd",
            "Forward-Only": "#ff7f0e",
            "v_aux": "#1f77b4",
            "e_aux": "#17becf",
        }
        a3_styles = {
            "Naive": "--",
            "Forward-Only": "-.",
            "v_aux": "-",
            "e_aux": "-",
        }
        a3_focus = [2, 3]
    else:
        a3_algs = ["Naive", "Forward-Only", "v_aux"]
        a3_colors = {
            "Naive": "#9467bd",
            "Forward-Only": "#ff7f0e",
            "v_aux": "#1f77b4",
        }
        a3_styles = {
            "Naive": "--",
            "Forward-Only": "-.",
            "v_aux": "-",
        }
        a3_focus = None

    a3_results_auc = {}
    a3_results_final = {}
    a3_ann_auc = {}
    a3_ann_final = {}

    for r_idx, env in enumerate(envs):
        env_d = all_runs[env]

        # 1. Forward-Only candidates (lambda in {0.0, ablation_lambda})
        fwd_cands = []
        if env_d.get("E(0)_fwd"):
            c = select_best_lr_run(env_d["E(0)_fwd"], critic_lrs, metric_key)
            c["legend_label"] = "E(0) Fwd"
            fwd_cands.append(c)
        if env_d.get("E(lambda)_fwd"):
            c = select_best_lr_run(env_d["E(lambda)_fwd"], critic_lrs, metric_key)
            c["legend_label"] = f"E({abl_lmb}) Fwd"
            fwd_cands.append(c)

        # 2. Naive candidates (lambda in {0.0, ablation_lambda})
        naive_cands = []
        if env_d.get("E(0)_no_v_aux"):
            c = select_best_lr_run(env_d["E(0)_no_v_aux"], critic_lrs, metric_key)
            c["legend_label"] = "E(0) Naive"
            naive_cands.append(c)
        if env_d.get("E(lambda)_naive"):
            c = select_best_lr_run(env_d["E(lambda)_naive"], critic_lrs, metric_key)
            c["legend_label"] = f"E({abl_lmb}) Naive"
            naive_cands.append(c)

        # 3. v_aux candidates (matching lambda in {0.0, ablation_lambda})
        v_aux_cands = []
        if env_d.get("E(0)_v_aux"):
            c = select_best_lr_run(env_d["E(0)_v_aux"], critic_lrs, metric_key)
            c["legend_label"] = "v_aux (E(0), λ_aux=1.0)"
            v_aux_cands.append(c)
        for alam, lr_d in env_d.get("E(0)_aux_lam", {}).items():
            if lr_d:
                c = select_best_lr_run(lr_d, critic_lrs, metric_key)
                c["legend_label"] = f"v_aux (E(0), λ_aux={alam})"
                v_aux_cands.append(c)
        el_lmb_d = {
            lr: env_d["E(lambda)"][lr][abl_lmb]
            for lr in env_d.get("E(lambda)", {})
            if abl_lmb in env_d["E(lambda)"][lr]
        }
        if el_lmb_d:
            c = select_best_lr_run(el_lmb_d, critic_lrs, metric_key)
            c["legend_label"] = f"v_aux (λ={abl_lmb}, λ_aux=1.0)"
            v_aux_cands.append(c)
        for alam, lr_d in env_d.get("E(lambda)_aux_lam", {}).items():
            if lr_d:
                c = select_best_lr_run(lr_d, critic_lrs, metric_key)
                c["legend_label"] = f"v_aux (λ={abl_lmb}, λ_aux={alam})"
                v_aux_cands.append(c)

        # 4. e_aux candidates (matching lambda in {0.0, ablation_lambda})
        e_aux_cands = []
        if env_d.get("E(0)_e_aux"):
            c = select_best_lr_run(env_d["E(0)_e_aux"], critic_lrs, metric_key)
            c["legend_label"] = "e_aux (E(0), λ_aux=1.0)"
            e_aux_cands.append(c)
        for alam, lr_d in env_d.get("E(0)_e_aux_lam", {}).items():
            if lr_d:
                c = select_best_lr_run(lr_d, critic_lrs, metric_key)
                c["legend_label"] = f"e_aux (E(0), λ_aux={alam})"
                e_aux_cands.append(c)
        if env_d.get("E(lambda)_e_aux"):
            c = select_best_lr_run(env_d["E(lambda)_e_aux"], critic_lrs, metric_key)
            c["legend_label"] = f"e_aux (λ={abl_lmb}, λ_aux=1.0)"
            e_aux_cands.append(c)
        for alam, lr_d in env_d.get("E(lambda)_e_aux_lam", {}).items():
            if lr_d:
                c = select_best_lr_run(lr_d, critic_lrs, metric_key)
                c["legend_label"] = f"e_aux (λ={abl_lmb}, λ_aux={alam})"
                e_aux_cands.append(c)

        best_fwd_auc = max(fwd_cands, key=lambda x: x["auc_mean"]) if fwd_cands else select_best_lr_run({}, critic_lrs, metric_key)
        best_fwd_final = max(fwd_cands, key=lambda x: x["final_mean"]) if fwd_cands else best_fwd_auc

        best_naive_auc = max(naive_cands, key=lambda x: x["auc_mean"]) if naive_cands else select_best_lr_run({}, critic_lrs, metric_key)
        best_naive_final = max(naive_cands, key=lambda x: x["final_mean"]) if naive_cands else best_naive_auc

        best_v_aux_auc = max(v_aux_cands, key=lambda x: x["auc_mean"]) if v_aux_cands else select_best_lr_run({}, critic_lrs, metric_key)
        best_v_aux_final = max(v_aux_cands, key=lambda x: x["final_mean"]) if v_aux_cands else best_v_aux_auc

        res_auc = {
            "Naive": best_naive_auc,
            "Forward-Only": best_fwd_auc,
            "v_aux": best_v_aux_auc,
        }
        res_final = {
            "Naive": best_naive_final,
            "Forward-Only": best_fwd_final,
            "v_aux": best_v_aux_final,
        }

        if has_e_aux:
            best_e_aux_auc = max(e_aux_cands, key=lambda x: x["auc_mean"]) if e_aux_cands else select_best_lr_run({}, critic_lrs, metric_key)
            best_e_aux_final = max(e_aux_cands, key=lambda x: x["final_mean"]) if e_aux_cands else best_e_aux_auc
            res_auc["e_aux"] = best_e_aux_auc
            res_final["e_aux"] = best_e_aux_final

            # Order: 0: Naive, 1: Forward-Only, 2: v_aux, 3: e_aux
            a3_ann_auc[(r_idx, 0)] = clean_variant_name(best_naive_auc.get("legend_label", ""))
            a3_ann_auc[(r_idx, 1)] = clean_variant_name(best_fwd_auc.get("legend_label", ""))
            a3_ann_auc[(r_idx, 2)] = clean_variant_name(best_v_aux_auc.get("legend_label", ""))
            a3_ann_auc[(r_idx, 3)] = clean_variant_name(best_e_aux_auc.get("legend_label", ""))

            a3_ann_final[(r_idx, 0)] = clean_variant_name(best_naive_final.get("legend_label", ""))
            a3_ann_final[(r_idx, 1)] = clean_variant_name(best_fwd_final.get("legend_label", ""))
            a3_ann_final[(r_idx, 2)] = clean_variant_name(best_v_aux_final.get("legend_label", ""))
            a3_ann_final[(r_idx, 3)] = clean_variant_name(best_e_aux_final.get("legend_label", ""))
        else:
            a3_ann_auc[(r_idx, 0)] = clean_variant_name(best_naive_auc.get("legend_label", ""))
            a3_ann_auc[(r_idx, 1)] = clean_variant_name(best_fwd_auc.get("legend_label", ""))
            a3_ann_auc[(r_idx, 2)] = clean_variant_name(best_v_aux_auc.get("legend_label", ""))

            a3_ann_final[(r_idx, 0)] = clean_variant_name(best_naive_final.get("legend_label", ""))
            a3_ann_final[(r_idx, 1)] = clean_variant_name(best_fwd_final.get("legend_label", ""))
            a3_ann_final[(r_idx, 2)] = clean_variant_name(best_v_aux_final.get("legend_label", ""))

        a3_results_auc[env] = res_auc
        a3_results_final[env] = res_final

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a3_algs,
        per_env_results=a3_results_auc,
        colors=a3_colors,
        linestyles=a3_styles,
        batch_size=batch_size,
        plot_title_suffix=f" — Story 1: E Variants (λ = 0 or {abl_lmb})",
        plot_output_base=os.path.join(plots_dir, "story1_e_variant_comparison_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story1_e_variant_comparison"),
        table_title_prefix=f"Story 1: E Variants (Baselines vs Unbiased v_aux vs e_aux, λ = 0 or {abl_lmb})",
        cmap="Blues",
        cell_annotations={"auc": a3_ann_auc, "final": a3_ann_final},
        per_env_results_final=a3_results_final,
        focus_cols=a3_focus,
    )

    # =========================================================================
    # STORY 2: The Use of λ_aux (v_aux, e_aux, and combined 1.0 vs 0.8)
    # =========================================================================
    if has_e_aux:
        a4_algs = ["v_aux (λ_aux=1.0)", "v_aux (λ_aux=0.8)", "e_aux (λ_aux=1.0)", "e_aux (λ_aux=0.8)"]
        a4_colors = {
            "v_aux (λ_aux=1.0)": "#1f77b4",
            "v_aux (λ_aux=0.8)": "#2ca02c",
            "e_aux (λ_aux=1.0)": "#17becf",
            "e_aux (λ_aux=0.8)": "#d62728",
        }
        a4_styles = {
            "v_aux (λ_aux=1.0)": "--",
            "v_aux (λ_aux=0.8)": "-",
            "e_aux (λ_aux=1.0)": "--",
            "e_aux (λ_aux=0.8)": "-",
        }
        a4_results = {}
        for env in envs:
            env_d = all_runs[env]
            el_lmb_lr_dict = {
                lr: env_d["E(lambda)"][lr][abl_lmb]
                for lr in env_d.get("E(lambda)", {})
                if abl_lmb in env_d["E(lambda)"][lr]
            }
            v_aux_08 = env_d.get("E(lambda)_aux_lam", {}).get(0.8, {})
            e_aux_08 = env_d.get("E(lambda)_e_aux_lam", {}).get(0.8, {})
            a4_results[env] = {
                "v_aux (λ_aux=1.0)": select_best_lr_run(el_lmb_lr_dict, critic_lrs, metric_key),
                "v_aux (λ_aux=0.8)": select_best_lr_run(v_aux_08, critic_lrs, metric_key),
                "e_aux (λ_aux=1.0)": select_best_lr_run(env_d.get("E(lambda)_e_aux", {}), critic_lrs, metric_key),
                "e_aux (λ_aux=0.8)": select_best_lr_run(e_aux_08, critic_lrs, metric_key),
            }

        render_analysis_plot_and_tables(
            envs=envs,
            alg_names=a4_algs,
            per_env_results=a4_results,
            colors=a4_colors,
            linestyles=a4_styles,
            batch_size=batch_size,
            plot_title_suffix=f" — Story 2: Effect of λ_aux (E(λ={abl_lmb}))",
            plot_output_base=os.path.join(plots_dir, "story2_aux_lambda_effect_learning_curves"),
            table_output_base=os.path.join(tables_dir, "story2_aux_lambda_effect"),
            table_title_prefix=f"Story 2: Effect of Auxiliary Return Lambda (λ_aux=1.0 vs 0.8 at E(λ={abl_lmb}))",
            cmap="PuBuGn",
        )

        # STORY 2B: e_aux λ_aux sweep (1.0, 0.9, 0.8)
        a5_algs = [f"e_aux (λ_aux={alam})" for alam in aux_lambdas]
        palette_a5 = ["#7f7f7f", "#17becf", "#2ca02c", "#d62728"]
        styles_a5 = ["--", "-.", "-", ":"]
        a5_colors = {alg: palette_a5[i % len(palette_a5)] for i, alg in enumerate(a5_algs)}
        a5_styles = {alg: styles_a5[i % len(styles_a5)] for i, alg in enumerate(a5_algs)}
        a5_results = {}
        for env in envs:
            env_d = all_runs[env]
            a5_results[env] = {}
            for alam in aux_lambdas:
                alg_label = f"e_aux (λ_aux={alam})"
                lr_dict = env_d.get("E(lambda)_e_aux_lam", {}).get(alam, {})
                if not lr_dict and alam == 1.0:
                    lr_dict = env_d.get("E(lambda)_e_aux", {})
                a5_results[env][alg_label] = select_best_lr_run(lr_dict, critic_lrs, metric_key)

        render_analysis_plot_and_tables(
            envs=envs,
            alg_names=a5_algs,
            per_env_results=a5_results,
            colors=a5_colors,
            linestyles=a5_styles,
            batch_size=batch_size,
            plot_title_suffix=f" — Story 2: e_aux λ_aux Sweep (E(λ={abl_lmb}))",
            plot_output_base=os.path.join(plots_dir, "story2_e_aux_lambda_comparison_learning_curves"),
            table_output_base=os.path.join(tables_dir, "story2_e_aux_lambda_comparison"),
            table_title_prefix=f"Story 2: e_aux λ_aux Comparison at E(λ={abl_lmb})",
            cmap="BuPu",
        )

    # STORY 2C: v_aux λ_aux sweep (1.0, 0.9, 0.8)
    a6_algs = [f"v_aux (λ_aux={alam})" for alam in aux_lambdas]
    palette_a6 = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]
    styles_a6 = ["--", "-.", "-", ":"]
    a6_colors = {alg: palette_a6[i % len(palette_a6)] for i, alg in enumerate(a6_algs)}
    a6_styles = {alg: styles_a6[i % len(styles_a6)] for i, alg in enumerate(a6_algs)}
    a6_results = {}
    for env in envs:
        env_d = all_runs[env]
        a6_results[env] = {}
        for alam in aux_lambdas:
            alg_label = f"v_aux (λ_aux={alam})"
            lr_dict = env_d.get("E(lambda)_aux_lam", {}).get(alam, {})
            if not lr_dict and alam == 1.0:
                lr_dict = {
                    lr: env_d["E(lambda)"][lr][abl_lmb]
                    for lr in env_d.get("E(lambda)", {})
                    if abl_lmb in env_d["E(lambda)"][lr]
                }
            a6_results[env][alg_label] = select_best_lr_run(lr_dict, critic_lrs, metric_key)

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a6_algs,
        per_env_results=a6_results,
        colors=a6_colors,
        linestyles=a6_styles,
        batch_size=batch_size,
        plot_title_suffix=f" — Story 2: v_aux λ_aux Sweep (E(λ={abl_lmb}))",
        plot_output_base=os.path.join(plots_dir, "story2_v_aux_lambda_comparison_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story2_v_aux_lambda_comparison"),
        table_title_prefix=f"Story 2: v_aux λ_aux Comparison at E(λ={abl_lmb})",
        cmap="YlGn",
    )

    # =========================================================================
    # STORY 3 (PRIMARY): Forward Winning-est Unbiased Variant (v_aux vs e_aux) vs Best TD(λ*)
    # Selects winning-est unbiased E variant directly from Direct 1v1 (v_aux vs e_aux)
    # =========================================================================
    unbiased_candidates = [a for a in a3_algs if a in ("v_aux", "e_aux")]
    if len(unbiased_candidates) == 2:
        mat_u_fin = np.array([[a3_results_final[e][a]["final_mean"] for a in unbiased_candidates] for e in envs])
        mat_u_auc = np.array([[a3_results_auc[e][a]["auc_mean"] for a in unbiased_candidates] for e in envs])
        uw_f, ut_f, ul_f, _, _, _ = _compute_wtl_stats(mat_u_fin, rel_tol=REAL_WIN_REL_TOL)
        uw_a, ut_a, ul_a, _, _, _ = _compute_wtl_stats(mat_u_auc, rel_tol=REAL_WIN_REL_TOL)
        u_net = {
            a: int((uw_f[i] - ul_f[i]) + (uw_a[i] - ul_a[i]))
            for i, a in enumerate(unbiased_candidates)
        }
        print("\n--- Story 1 Direct 1v1 Unbiased Real-Win (>=1%) W/T/L (v_aux vs e_aux) ---")
        for i, a in enumerate(unbiased_candidates):
            print(
                f"  {a:<8}: Final={uw_f[i]}W/{ut_f[i]}T/{ul_f[i]}L | "
                f"AUC={uw_a[i]}W/{ut_a[i]}T/{ul_a[i]}L | Combined Net={u_net[a]:+d}"
            )
        forwarded_e_name = max(unbiased_candidates, key=lambda a: u_net[a])
        print(f"--> Forwarding Winning-est Unbiased Variant '{forwarded_e_name}' to Story 3!\n")
    else:
        forwarded_e_name = "v_aux"

    s3v_algs = ["TD(λ*)", "Unbiased E (v_aux)"]
    s3v_colors = {"TD(λ*)": "#e6550d", "Unbiased E (v_aux)": "#1f77b4"}
    s3v_styles = {"TD(λ*)": "--", "Unbiased E (v_aux)": "-"}

    s3e_algs = ["TD(λ*)", "Unbiased E (e_aux)"]
    s3e_colors = {"TD(λ*)": "#e6550d", "Unbiased E (e_aux)": "#17becf"}
    s3e_styles = {"TD(λ*)": "--", "Unbiased E (e_aux)": "-"}

    s3v_results_auc, s3v_results_final = {}, {}
    s3v_ann_auc, s3v_ann_final = {}, {}

    s3e_results_auc, s3e_results_final = {}, {}
    s3e_ann_auc, s3e_ann_final = {}, {}

    # Also build comprehensive best_e_vs_best_td summary
    if has_e_aux:
        a7_algs = ["Best TD(λ*)", "Best E (v_aux)", "Best E (e_aux)", "Best E (All)"]
        a7_colors = {
            "Best TD(λ*)": "#e6550d",
            "Best E (v_aux)": "#1f77b4",
            "Best E (e_aux)": "#17becf",
            "Best E (All)": "#2ca02c",
        }
        a7_styles = {
            "Best TD(λ*)": "--",
            "Best E (v_aux)": "-",
            "Best E (e_aux)": "-",
            "Best E (All)": "-",
        }
        col_all_idx = 3
    else:
        a7_algs = ["Best TD(λ*)", "Best E (v_aux)", "Best E (All)"]
        a7_colors = {
            "Best TD(λ*)": "#e6550d",
            "Best E (v_aux)": "#1f77b4",
            "Best E (All)": "#2ca02c",
        }
        a7_styles = {
            "Best TD(λ*)": "--",
            "Best E (v_aux)": "-",
            "Best E (All)": "-",
        }
        col_all_idx = 2

    a7_results_auc = {}
    a7_results_final = {}
    a7_annotations_auc = {}
    a7_annotations_final = {}

    for r_idx, env in enumerate(envs):
        env_d = all_runs[env]

        # 1. Best TD(lambda*) across all (lr, lmb)
        td_candidates = []
        for lr, lmb_dict in env_d.get("TD(lambda)", {}).items():
            for lmb in lmb_dict:
                res_info = select_best_lr_run({lr: lmb_dict[lmb]}, [lr], metric_key)
                res_info["legend_label"] = f"Best TD (λ={lmb})"
                res_info["short_ann"] = f"λ={lmb}"
                td_candidates.append(res_info)

        best_td_auc = max(td_candidates, key=lambda x: x["auc_mean"]) if td_candidates else select_best_lr_run({}, critic_lrs, metric_key)
        best_td_final = max(td_candidates, key=lambda x: x["final_mean"]) if td_candidates else best_td_auc

        # 2. Best E (v_aux): best across E(0)_v_aux, E(0)_aux_lam, E(lambda), E(lambda)_aux_lam
        v_aux_candidates = []
        if env_d.get("E(0)_v_aux"):
            info = select_best_lr_run(env_d["E(0)_v_aux"], critic_lrs, metric_key)
            info["legend_label"] = "Best E v_aux (E(0), λ_aux=1.0)"
            info["short_ann"] = "λ=0, λ_aux=1.0"
            v_aux_candidates.append(info)
        for alam, lr_d in env_d.get("E(0)_aux_lam", {}).items():
            if lr_d:
                info = select_best_lr_run(lr_d, critic_lrs, metric_key)
                info["legend_label"] = f"Best E v_aux (E(0), λ_aux={alam})"
                info["short_ann"] = f"λ=0, λ_aux={alam}"
                v_aux_candidates.append(info)
        for lr, lmb_dict in env_d.get("E(lambda)", {}).items():
            for lmb in lmb_dict:
                info = select_best_lr_run({lr: lmb_dict[lmb]}, [lr], metric_key)
                info["legend_label"] = f"Best E v_aux (λ={lmb}, λ_aux=1.0)"
                info["short_ann"] = f"λ={lmb}, λ_aux=1.0"
                v_aux_candidates.append(info)
        for alam, lr_d in env_d.get("E(lambda)_aux_lam", {}).items():
            if lr_d:
                info = select_best_lr_run(lr_d, critic_lrs, metric_key)
                info["legend_label"] = f"Best E v_aux (λ={abl_lmb}, λ_aux={alam})"
                info["short_ann"] = f"λ={abl_lmb}, λ_aux={alam}"
                v_aux_candidates.append(info)

        best_e_v_aux_auc = max(v_aux_candidates, key=lambda x: x["auc_mean"]) if v_aux_candidates else select_best_lr_run({}, critic_lrs, metric_key)
        best_e_v_aux_final = max(v_aux_candidates, key=lambda x: x["final_mean"]) if v_aux_candidates else best_e_v_aux_auc

        # 3. Best E (e_aux)
        e_aux_candidates = []
        if env_d.get("E(0)_e_aux"):
            info = select_best_lr_run(env_d["E(0)_e_aux"], critic_lrs, metric_key)
            info["legend_label"] = "Best E e_aux (E(0), λ_aux=1.0)"
            info["short_ann"] = "λ=0, λ_aux=1.0"
            e_aux_candidates.append(info)
        for alam, lr_d in env_d.get("E(0)_e_aux_lam", {}).items():
            if lr_d:
                info = select_best_lr_run(lr_d, critic_lrs, metric_key)
                info["legend_label"] = f"Best E e_aux (E(0), λ_aux={alam})"
                info["short_ann"] = f"λ=0, λ_aux={alam}"
                e_aux_candidates.append(info)
        if env_d.get("E(lambda)_e_aux"):
            info = select_best_lr_run(env_d["E(lambda)_e_aux"], critic_lrs, metric_key)
            info["legend_label"] = f"Best E e_aux (λ={abl_lmb}, λ_aux=1.0)"
            info["short_ann"] = f"λ={abl_lmb}, λ_aux=1.0"
            e_aux_candidates.append(info)
        for alam, lr_d in env_d.get("E(lambda)_e_aux_lam", {}).items():
            if lr_d:
                info = select_best_lr_run(lr_d, critic_lrs, metric_key)
                info["legend_label"] = f"Best E e_aux (λ={abl_lmb}, λ_aux={alam})"
                info["short_ann"] = f"λ={abl_lmb}, λ_aux={alam}"
                e_aux_candidates.append(info)

        best_e_e_aux_auc = max(e_aux_candidates, key=lambda x: x["auc_mean"]) if e_aux_candidates else select_best_lr_run({}, critic_lrs, metric_key)
        best_e_e_aux_final = max(e_aux_candidates, key=lambda x: x["final_mean"]) if e_aux_candidates else best_e_e_aux_auc

        # Populate Story 3A: TD(λ*) vs Unbiased E (v_aux)
        s3v_results_auc[env] = {
            "TD(λ*)": best_td_auc,
            "Unbiased E (v_aux)": best_e_v_aux_auc,
        }
        s3v_results_final[env] = {
            "TD(λ*)": best_td_final,
            "Unbiased E (v_aux)": best_e_v_aux_final,
        }
        s3v_ann_auc[(r_idx, 0)] = best_td_auc.get("short_ann", "")
        s3v_ann_auc[(r_idx, 1)] = best_e_v_aux_auc.get("short_ann", "")
        s3v_ann_final[(r_idx, 0)] = best_td_final.get("short_ann", "")
        s3v_ann_final[(r_idx, 1)] = best_e_v_aux_final.get("short_ann", "")

        # Populate Story 3B: TD(λ*) vs Unbiased E (e_aux)
        s3e_results_auc[env] = {
            "TD(λ*)": best_td_auc,
            "Unbiased E (e_aux)": best_e_e_aux_auc,
        }
        s3e_results_final[env] = {
            "TD(λ*)": best_td_final,
            "Unbiased E (e_aux)": best_e_e_aux_final,
        }
        s3e_ann_auc[(r_idx, 0)] = best_td_auc.get("short_ann", "")
        s3e_ann_auc[(r_idx, 1)] = best_e_e_aux_auc.get("short_ann", "")
        s3e_ann_final[(r_idx, 0)] = best_td_final.get("short_ann", "")
        s3e_ann_final[(r_idx, 1)] = best_e_e_aux_final.get("short_ann", "")

        # 4. Best E (All)
        all_e_candidates = list(v_aux_candidates) + list(e_aux_candidates)
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

        best_e_all_auc = max(all_e_candidates, key=lambda x: x["auc_mean"]) if all_e_candidates else select_best_lr_run({}, critic_lrs, metric_key)
        best_e_all_final = max(all_e_candidates, key=lambda x: x["final_mean"]) if all_e_candidates else best_e_all_auc

        res_auc = {
            "Best TD(λ*)": best_td_auc,
            "Best E (v_aux)": best_e_v_aux_auc,
            "Best E (All)": best_e_all_auc,
        }
        res_final = {
            "Best TD(λ*)": best_td_final,
            "Best E (v_aux)": best_e_v_aux_final,
            "Best E (All)": best_e_all_final,
        }
        if has_e_aux:
            res_auc["Best E (e_aux)"] = best_e_e_aux_auc
            res_final["Best E (e_aux)"] = best_e_e_aux_final

        a7_results_auc[env] = res_auc
        a7_results_final[env] = res_final

        a7_annotations_auc[(r_idx, col_all_idx)] = clean_variant_name(best_e_all_auc.get("legend_label", ""))
        a7_annotations_final[(r_idx, col_all_idx)] = clean_variant_name(best_e_all_final.get("legend_label", ""))

    # Render Story 3A: Best TD(λ*) vs Unbiased E (v_aux)
    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=s3v_algs,
        per_env_results=s3v_results_auc,
        colors=s3v_colors,
        linestyles=s3v_styles,
        batch_size=batch_size,
        plot_title_suffix=" — Story 3A: TD(λ*) vs Unbiased E (v_aux)",
        plot_output_base=os.path.join(plots_dir, "story3_td_vs_v_aux_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story3_td_vs_v_aux"),
        table_title_prefix="Story 3A: Best TD(λ*) vs Best Unbiased E (v_aux) per Environment",
        cmap="Blues",
        cell_annotations={"auc": s3v_ann_auc, "final": s3v_ann_final},
        per_env_results_final=s3v_results_final,
    )

    # Render Story 3B: Best TD(λ*) vs Unbiased E (e_aux)
    if has_e_aux:
        render_analysis_plot_and_tables(
            envs=envs,
            alg_names=s3e_algs,
            per_env_results=s3e_results_auc,
            colors=s3e_colors,
            linestyles=s3e_styles,
            batch_size=batch_size,
            plot_title_suffix=" — Story 3B: TD(λ*) vs Unbiased E (e_aux)",
            plot_output_base=os.path.join(plots_dir, "story3_td_vs_e_aux_learning_curves"),
            table_output_base=os.path.join(tables_dir, "story3_td_vs_e_aux"),
            table_title_prefix="Story 3B: Best TD(λ*) vs Best Unbiased E (e_aux) per Environment",
            cmap="Blues",
            cell_annotations={"auc": s3e_ann_auc, "final": s3e_ann_final},
            per_env_results_final=s3e_results_final,
        )

    # Render full Best E summary
    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=a7_algs,
        per_env_results=a7_results_auc,
        colors=a7_colors,
        linestyles=a7_styles,
        batch_size=batch_size,
        plot_title_suffix=" — Best E Variants vs Best TD(λ*)",
        plot_output_base=os.path.join(plots_dir, "story3_best_e_vs_best_td_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story3_best_e_vs_best_td"),
        table_title_prefix="Best E Runs vs Best TD(λ*) Summary",
        cmap="Blues",
        cell_annotations={"auc": a7_annotations_auc, "final": a7_annotations_final},
        per_env_results_final=a7_results_final,
    )

    print("\nAll automated story analyses (learning curve plots and .md/.csv/.tex/.pdf tables) generated successfully!")


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

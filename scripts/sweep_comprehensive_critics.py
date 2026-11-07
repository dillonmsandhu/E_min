#!/usr/bin/env python3
"""
sweep_comprehensive_critics.py

Comprehensive Critic Comparison Sweep & Diagnostic Suite:
1. E(lambda) with geometric lookahead jumps (sweeping lambda, selecting best)
2. TD(lambda) fitted value iteration (sweeping lambda, selecting best)
3. E(0) via algos/E.py (evaluated both with and without V_aux for diagnostic 2)
4. MC via algos/mc.py (Monte Carlo rollout return regression)
5. V_aux Regularization Ablations for a single lambda:
   - Baseline (MSE)
   - + Huber Loss (delta=10.0)
   - + Trust Region Regularization (beta=1.0)
   - + Both Combined (Huber + Trust)

Settings:
- Actor & Critic epochs: 4
- Minibatch size: 512
- Rollouts: 32 envs x 256 steps (8,192 transitions per update)
- E(lambda) jump mode: 'full_mc_invalid_jump' (revert to full MC on invalid jump)
- Actor LR: fixed (default 3e-4)
- Critic LR: swept across 3 learning rates [1e-4, 3e-4, 1e-3]
- Tracked metrics: terminations, truncations, MSE(V, G), MSE(V_aux, G), similarity(V, V_aux)

Analysis Outputs:
1. Diagnostic 1: E(0) script vs. E(lambda=0) script comparison
2. Diagnostic 2: E(0) with vs. without V_aux performance across environments
3. Diagnostic 3: V_aux vs. V_main MSE learning curves with dotted similarity line
4. Diagnostic 4: V_aux Regularization Ablations with TD(lambda) converged performance dotted line
5. Main Comparison: E(lambda*), TD(lambda*), E(0), MC learning curves + AUC Heatmap
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)
if _script_dir not in sys.path:
    sys.path.insert(1, _script_dir)

import jax
import jax.numpy as jnp

from gymnax_rl.algos.ppo import make_train as make_ppo_train


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


def run_experiment(config, seed_keys):
    """JIT-compiles and runs vmapped training across random seeds."""
    train_fn = make_ppo_train(config)
    vmapped_fn = jax.jit(jax.vmap(train_fn))
    out = vmapped_fn(seed_keys)
    # Convert jax arrays in metrics to numpy
    metrics = {k: np.array(v) for k, v in out["metrics"].items()}
    return metrics


def run_sweep(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)

    master_rng = jax.random.PRNGKey(args.master_seed)
    seed_keys = jax.random.split(master_rng, args.n_seeds)

    lambdas = args.lambdas
    critic_lrs = args.critic_lrs
    envs = args.envs
    abl_lmb = args.ablation_lambda

    base_config = {
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "MINIBATCH_SIZE": args.minibatch_size,
        "ACTOR_EPOCHS": 4,
        "CRITIC_EPOCHS": 4,
        "ACTOR_LR": args.actor_lr,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.8,
        "CLIP_EPS": 0.1,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "SURGICAL_V_AUX": True,
        "AUX_VF_COEF": 1.0,
        "E_LAMBDA_JUMP_MODE": "full_mc_invalid_jump",
        "AUX_LAMBDA": 1.0,
    }

    # Store full results: all_runs[env][alg_name][critic_lr][optional_lambda] = metrics
    all_runs = {}

    # Check if user requested loading existing data
    if args.load_existing and os.path.exists(args.load_existing):
        print(f"Loading existing metrics from: {args.load_existing}")
        with open(args.load_existing, "rb") as f:
            loaded = pickle.load(f)
            all_runs = loaded.get("runs", {})

    batch_size = args.num_envs * args.num_steps
    base_timesteps = args.total_timesteps or 300000
    num_updates = max(1, base_timesteps // batch_size)
    timesteps = np.arange(1, num_updates + 1) * batch_size

    # Align timesteps to loaded trajectory length if loading existing data
    if all_runs:
        for env in envs:
            if env in all_runs:
                for k in all_runs[env]:
                    sub = all_runs[env][k]
                    first_val = list(sub.values())[0]
                    if isinstance(first_val, dict):
                        first_val = list(first_val.values())[0]
                    if 'returned_episode_returns' in first_val:
                        curve_len = first_val['returned_episode_returns'].shape[-1]
                        timesteps = np.arange(1, curve_len + 1) * batch_size
                        break

    print(f"=================================================================")
    print(f"STARTING COMPREHENSIVE CRITIC SWEEP & V_AUX ABLATIONS")
    print(f"Envs: {envs}")
    print(f"Rollouts: {args.num_envs} envs x {args.num_steps} steps (batch {batch_size})")
    print(f"Timesteps Policy: Auto (1M for MountainCar envs, 300k for small envs) | Override: {args.total_timesteps}")
    print(f"Critic LRs: {critic_lrs} | Lambdas: {lambdas} | Seeds: {args.n_seeds}")
    print(f"V_aux Ablation Lambda: {abl_lmb} (Huber δ={args.huber_delta}, Trust β={args.trust_coef})")
    print(f"=================================================================")

    for env_name in envs:
        env_timesteps = get_env_default_timesteps(env_name, args.total_timesteps)
        env_base_config = {**base_config, "ENV_NAME": env_name, "TOTAL_TIMESTEPS": env_timesteps}
        if env_name not in all_runs:
            all_runs[env_name] = {}
        print(f"\n>>> Running Environment: {env_name} (Timesteps: {env_timesteps:,}) <<<")

        # -------------------------------------------------------------
        # 1. MC (CRITIC_TYPE = 'mc')
        # -------------------------------------------------------------
        if "MC" not in all_runs[env_name]:
            all_runs[env_name]["MC"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["MC"]:
                print(f"[MC] env={env_name}, critic_lr={clr}...")
                cfg = {**env_base_config, "CRITIC_TYPE": "mc", "CRITIC_LR": clr, "USES_AUX": True}
                metrics = run_experiment(cfg, seed_keys)
                all_runs[env_name]["MC"][clr] = metrics
                mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                print(f"   -> Final Return: {mean_ret:.2f}")

        # -------------------------------------------------------------
        # 2. E(0) Standard with V_aux (CRITIC_TYPE = 'e_0', USES_AUX=True)
        # -------------------------------------------------------------
        if "E(0)_v_aux" not in all_runs[env_name]:
            all_runs[env_name]["E(0)_v_aux"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["E(0)_v_aux"]:
                print(f"[E(0) with V_aux] env={env_name}, critic_lr={clr}...")
                cfg = {**env_base_config, "CRITIC_TYPE": "e_0", "CRITIC_LR": clr, "USES_AUX": True, "RETURN_LAMBDA": 1.0}
                metrics = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(0)_v_aux"][clr] = metrics
                mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                print(f"   -> Final Return: {mean_ret:.2f}")

        # -------------------------------------------------------------
        # 3. E(0) Basic without V_aux (CRITIC_TYPE = 'e_0_basic', USES_AUX=False)
        # -------------------------------------------------------------
        if "E(0)_no_v_aux" not in all_runs[env_name]:
            all_runs[env_name]["E(0)_no_v_aux"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["E(0)_no_v_aux"]:
                print(f"[E(0) Basic No V_aux] env={env_name}, critic_lr={clr}...")
                cfg = {**env_base_config, "CRITIC_TYPE": "e_0_basic", "CRITIC_LR": clr, "USES_AUX": False, "RETURN_LAMBDA": 1.0}
                metrics = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(0)_no_v_aux"][clr] = metrics
                mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                print(f"   -> Final Return: {mean_ret:.2f}")

        # -------------------------------------------------------------
        # 4. TD(lambda) (CRITIC_TYPE = 'fitted', sweeping lambda)
        # -------------------------------------------------------------
        if "TD(lambda)" not in all_runs[env_name]:
            all_runs[env_name]["TD(lambda)"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["TD(lambda)"]:
                all_runs[env_name]["TD(lambda)"][clr] = {}
            for lmb in lambdas:
                if lmb not in all_runs[env_name]["TD(lambda)"][clr]:
                    print(f"[TD(lambda)] env={env_name}, lambda={lmb}, critic_lr={clr}...")
                    cfg = {**env_base_config, "CRITIC_TYPE": "fitted", "CRITIC_LR": clr, "VALUE_LAMBDA": lmb}
                    metrics = run_experiment(cfg, seed_keys)
                    all_runs[env_name]["TD(lambda)"][clr][lmb] = metrics
                    mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                    print(f"   -> Final Return: {mean_ret:.2f}")

        # -------------------------------------------------------------
        # 5. E(lambda) (CRITIC_TYPE = 'e_lambda', sweeping lambda)
        # -------------------------------------------------------------
        if "E(lambda)" not in all_runs[env_name]:
            all_runs[env_name]["E(lambda)"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["E(lambda)"]:
                all_runs[env_name]["E(lambda)"][clr] = {}
            for lmb in lambdas:
                if lmb not in all_runs[env_name]["E(lambda)"][clr]:
                    print(f"[E(lambda)] env={env_name}, lambda={lmb}, critic_lr={clr}...")
                    cfg = {
                        **env_base_config,
                        "CRITIC_TYPE": "e_lambda",
                        "CRITIC_LR": clr,
                        "E_LAMBDA": lmb,
                        "VALUE_LAMBDA": lmb,
                        "RETURN_LAMBDA": 1.0,
                        "USES_AUX": True,
                        "AUX_LOSS_TYPE": "mse",
                        "AUX_REG_TYPE": "none",
                        "AUX_TRUST_REGION_COEF": 0.0,
                    }
                    metrics = run_experiment(cfg, seed_keys)
                    all_runs[env_name]["E(lambda)"][clr][lmb] = metrics
                    mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                    print(f"   -> Final Return: {mean_ret:.2f}")

        # -------------------------------------------------------------
        # 6. V_AUX REGULARIZATION ABLATIONS (at ablation_lambda)
        # -------------------------------------------------------------
        print(f"\n--- V_aux Regularization Ablations for {env_name} at λ={abl_lmb} ---")

        # 6a. + Huber Loss (delta=10.0)
        if "E(lambda)_huber" not in all_runs[env_name]:
            all_runs[env_name]["E(lambda)_huber"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["E(lambda)_huber"]:
                print(f"[E(λ={abl_lmb}) + Huber] env={env_name}, critic_lr={clr}...")
                cfg = {
                    **env_base_config,
                    "CRITIC_TYPE": "e_lambda",
                    "CRITIC_LR": clr,
                    "E_LAMBDA": abl_lmb,
                    "VALUE_LAMBDA": abl_lmb,
                    "RETURN_LAMBDA": 1.0,
                    "USES_AUX": True,
                    "AUX_LOSS_TYPE": "huber",
                    "AUX_HUBER_DELTA": args.huber_delta,
                    "AUX_REG_TYPE": "none",
                    "AUX_TRUST_REGION_COEF": 0.0,
                }
                metrics = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(lambda)_huber"][clr] = metrics
                mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                print(f"   -> Final Return: {mean_ret:.2f}")

        # 6b. + Trust Region Regularization (beta=1.0)
        if "E(lambda)_trust" not in all_runs[env_name]:
            all_runs[env_name]["E(lambda)_trust"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["E(lambda)_trust"]:
                print(f"[E(λ={abl_lmb}) + Trust Region] env={env_name}, critic_lr={clr}...")
                cfg = {
                    **env_base_config,
                    "CRITIC_TYPE": "e_lambda",
                    "CRITIC_LR": clr,
                    "E_LAMBDA": abl_lmb,
                    "VALUE_LAMBDA": abl_lmb,
                    "RETURN_LAMBDA": 1.0,
                    "USES_AUX": True,
                    "AUX_LOSS_TYPE": "mse",
                    "AUX_REG_TYPE": "l2",
                    "AUX_TRUST_REGION_COEF": args.trust_coef,
                }
                metrics = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(lambda)_trust"][clr] = metrics
                mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                print(f"   -> Final Return: {mean_ret:.2f}")

        # 6c. + Both Combined (Huber + Trust Region)
        if "E(lambda)_both" not in all_runs[env_name]:
            all_runs[env_name]["E(lambda)_both"] = {}
        for clr in critic_lrs:
            if clr not in all_runs[env_name]["E(lambda)_both"]:
                print(f"[E(λ={abl_lmb}) + Both (Huber+Trust)] env={env_name}, critic_lr={clr}...")
                cfg = {
                    **env_base_config,
                    "CRITIC_TYPE": "e_lambda",
                    "CRITIC_LR": clr,
                    "E_LAMBDA": abl_lmb,
                    "VALUE_LAMBDA": abl_lmb,
                    "RETURN_LAMBDA": 1.0,
                    "USES_AUX": True,
                    "AUX_LOSS_TYPE": "huber",
                    "AUX_HUBER_DELTA": args.huber_delta,
                    "AUX_REG_TYPE": "l2",
                    "AUX_TRUST_REGION_COEF": args.trust_coef,
                }
                metrics = run_experiment(cfg, seed_keys)
                all_runs[env_name]["E(lambda)_both"][clr] = metrics
                mean_ret = metrics["returned_episode_returns"].mean(axis=0)[-1]
                print(f"   -> Final Return: {mean_ret:.2f}")

    # Save raw results
    raw_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(raw_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "config": base_config}, f)
    print(f"\nSaved raw metrics to {raw_path}")

    # Generate analyses
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args)



def plot_curve_with_std(ax, curve, std, label, color, batch_size, lw=2.2, alpha=0.12, linestyle="-"):
    t = np.arange(1, len(curve) + 1) * batch_size
    ax.plot(t, curve, label=label, color=color, lw=lw, linestyle=linestyle)
    if std is not None:
        ax.fill_between(t, curve - std, curve + std, color=color, alpha=alpha)

def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args):
    batch_size = args.num_envs * args.num_steps
    # Align timesteps to actual recorded trajectory length
    for env in envs:
        for k in all_runs[env]:
            sub = all_runs[env][k]
            first_val = list(sub.values())[0]
            if isinstance(first_val, dict):
                first_val = list(first_val.values())[0]
            if 'returned_episode_returns' in first_val:
                curve_len = first_val['returned_episode_returns'].shape[-1]
                timesteps = np.arange(1, curve_len + 1) * batch_size
                break
    """Generates the 5 diagnostic and benchmark figures and summary tables."""
    print("\n================ GENERATING ANALYSES & FIGURES ================")
    n_envs = len(envs)

    # -------------------------------------------------------------
    # DIAGNOSTIC 1: Does E(0) script match E(lambda = 0) script?
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, n_envs, figsize=(6.5 * n_envs, 4.5), dpi=150, squeeze=False)
    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        best_lr = max(critic_lrs, key=lambda lr: compute_auc(all_runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"].mean(axis=0)))
        e0_curve = all_runs[env]["E(0)_v_aux"][best_lr]["returned_episode_returns"].mean(axis=0)
        e0_std = all_runs[env]["E(0)_v_aux"][best_lr]["returned_episode_returns"].std(axis=0)
        elambda0_curve = all_runs[env]["E(lambda)"][best_lr][0.0]["returned_episode_returns"].mean(axis=0)
        elambda0_std = all_runs[env]["E(lambda)"][best_lr][0.0]["returned_episode_returns"].std(axis=0)

        diff = np.abs(e0_curve - elambda0_curve)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        plot_curve_with_std(ax, e0_curve, e0_std, f"E(0) Script (lr={best_lr})", "#1f77b4", batch_size, lw=2.2)
        plot_curve_with_std(ax, elambda0_curve, elambda0_std, f"E(λ=0) Script (lr={best_lr})", "#ff7f0e", batch_size, lw=2.0, linestyle="--")

        ax.set_title(f"{env}: E(0) vs E(λ=0)\nMean |Δ|: {mean_diff:.2f}, Max |Δ|: {max_diff:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    diag1_path = os.path.join(out_dir, "diagnostics", "diagnostic_1_e0_vs_elambda0.png")
    plt.savefig(diag1_path)
    plt.close()
    print(f"Saved Diagnostic 1 to {diag1_path}")

    # -------------------------------------------------------------
    # DIAGNOSTIC 2: E(0) With vs. Without V_aux Performance
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, n_envs, figsize=(6.5 * n_envs, 4.5), dpi=150, squeeze=False)
    diag2_data = []

    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        best_lr_aux = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"]))
        best_lr_no_aux = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(0)_no_v_aux"][lr]["returned_episode_returns"]))

        curve_aux = all_runs[env]["E(0)_v_aux"][best_lr_aux]["returned_episode_returns"].mean(axis=0)
        std_aux = all_runs[env]["E(0)_v_aux"][best_lr_aux]["returned_episode_returns"].std(axis=0)
        curve_no = all_runs[env]["E(0)_no_v_aux"][best_lr_no_aux]["returned_episode_returns"].mean(axis=0)
        std_no = all_runs[env]["E(0)_no_v_aux"][best_lr_no_aux]["returned_episode_returns"].std(axis=0)

        plot_curve_with_std(ax, curve_no, std_no, f"E(0) Basic (No V_aux, lr={best_lr_no_aux})", "#d62728", batch_size, lw=2.4)
        plot_curve_with_std(ax, curve_aux, std_aux, f"E(0) Surgical (V_aux, lr={best_lr_aux})", "#1f77b4", batch_size, lw=2.2, linestyle="--")

        ax.set_title(f"{env}: E(0) with vs without V_aux", fontsize=11, fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=9)

        diag2_data.append({
            "Environment": env,
            "E(0) No V_aux AUC": compute_auc(curve_no),
            "E(0) No V_aux Peak": np.max(curve_no),
            "E(0) With V_aux AUC": compute_auc(curve_aux),
            "E(0) With V_aux Peak": np.max(curve_aux),
        })

    plt.tight_layout()
    diag2_path = os.path.join(out_dir, "diagnostics", "diagnostic_2_e0_v_aux_effect.png")
    plt.savefig(diag2_path)
    plt.close()
    pd.DataFrame(diag2_data).to_csv(os.path.join(out_dir, "diagnostics", "diagnostic_2_summary.csv"), index=False)
    print(f"Saved Diagnostic 2 to {diag2_path}")

    # -------------------------------------------------------------
    # DIAGNOSTIC 3: V_aux vs. V_main MSE learning curves + Dotted Similarity
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, n_envs, figsize=(7.0 * n_envs, 4.8), dpi=150, squeeze=False)
    for idx, env in enumerate(envs):
        ax1 = axes[0, idx]
        ax2 = ax1.twinx()

        best_lr = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"]))
        metrics = all_runs[env]["E(0)_v_aux"][best_lr]

        if "mse_v_main" in metrics and "mse_v_aux" in metrics:
            mse_main = metrics["mse_v_main"].mean(axis=0)
            mse_aux = metrics["mse_v_aux"].mean(axis=0)
            similarity = metrics.get("v_similarity", np.zeros_like(mse_main)).mean(axis=0)

            t_d3 = np.arange(1, len(mse_main) + 1) * batch_size
            l1, = ax1.plot(t_d3, mse_main, label="MSE(V_main, Target)", color="#1f77b4", lw=2.2)
            l2, = ax1.plot(t_d3, mse_aux, label="MSE(V_aux, Target)", color="#ff7f0e", lw=2.0)
            ax1.set_xlabel("Timesteps")
            ax1.set_ylabel("MSE against Target (G)", color="#1f77b4")
            ax1.tick_params(axis='y', labelcolor="#1f77b4")
            ax1.grid(True, linestyle="--", alpha=0.4)

            l3, = ax2.plot(t_d3, similarity, label="Sim(V_main, V_aux)", color="#2ca02c", lw=2.2, linestyle=":")
            ax2.set_ylabel("Cosine Similarity (V_main, V_aux)", color="#2ca02c")
            ax2.tick_params(axis='y', labelcolor="#2ca02c")
            ax2.set_ylim(0.0, 1.05)

            lines = [l1, l2, l3]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc="center right", frameon=True, facecolor="white", fontsize=9)
            ax1.set_title(f"{env}: V_aux vs. V_main MSE & Similarity", fontsize=11, fontweight="bold")

    plt.tight_layout()
    diag3_path = os.path.join(out_dir, "diagnostics", "diagnostic_3_v_aux_vs_v_main.png")
    plt.savefig(diag3_path)
    plt.close()
    print(f"Saved Diagnostic 3 to {diag3_path}")

    # -------------------------------------------------------------
    # DIAGNOSTIC 4: V_aux Regularization Ablations vs TD(lambda)
    # (Huber, Trust Region, Both vs TD converged performance dotted line)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, n_envs, figsize=(7.5 * n_envs, 5.2), dpi=150, squeeze=False)
    diag4_data = []
    abl_lmb = args.ablation_lambda
    if abl_lmb not in lambdas:
        if 0.8 in lambdas:
            abl_lmb = 0.8
        else:
            abl_lmb = lambdas[0]
        print(f"[Diagnostic 4] Notice: requested ablation_lambda {args.ablation_lambda} not in {lambdas}. Defaulting to {abl_lmb}.")

    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        # Choose best critic LR for E(lambda) at ablation_lambda
        best_clr = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(lambda)"][lr][abl_lmb]["returned_episode_returns"]))

        # 1. TD(lambda) at ablation_lambda: get converged performance and trajectory
        td_metrics = all_runs[env]["TD(lambda)"][best_clr][abl_lmb]
        td_curve = td_metrics["returned_episode_returns"].mean(axis=0)
        td_std = td_metrics["returned_episode_returns"].std(axis=0)
        td_converged = np.mean(td_curve[-5:])

        # 2. V_aux Baseline (MSE, no trust region)
        base_curve = all_runs[env]["E(lambda)"][best_clr][abl_lmb]["returned_episode_returns"].mean(axis=0)
        base_std = all_runs[env]["E(lambda)"][best_clr][abl_lmb]["returned_episode_returns"].std(axis=0)

        # 3. + Huber
        huber_curve = all_runs[env]["E(lambda)_huber"][best_clr]["returned_episode_returns"].mean(axis=0)
        huber_std = all_runs[env]["E(lambda)_huber"][best_clr]["returned_episode_returns"].std(axis=0)

        # 4. + Trust Region
        trust_curve = all_runs[env]["E(lambda)_trust"][best_clr]["returned_episode_returns"].mean(axis=0)
        trust_std = all_runs[env]["E(lambda)_trust"][best_clr]["returned_episode_returns"].std(axis=0)

        # 5. + Both Combined
        both_curve = all_runs[env]["E(lambda)_both"][best_clr]["returned_episode_returns"].mean(axis=0)
        both_std = all_runs[env]["E(lambda)_both"][best_clr]["returned_episode_returns"].std(axis=0)

        # Plot curves
        plot_curve_with_std(ax, base_curve, base_std, f"V_aux Baseline (MSE, AUC={compute_auc(base_curve):.1f})", "#1f77b4", batch_size, lw=2.2)
        plot_curve_with_std(ax, huber_curve, huber_std, f"+ Huber Loss (δ={args.huber_delta}, AUC={compute_auc(huber_curve):.1f})", "#2ca02c", batch_size, lw=2.2)
        plot_curve_with_std(ax, trust_curve, trust_std, f"+ Trust Region (β={args.trust_coef}, AUC={compute_auc(trust_curve):.1f})", "#ff7f0e", batch_size, lw=2.2)
        plot_curve_with_std(ax, both_curve, both_std, f"+ Both (Huber+Trust, AUC={compute_auc(both_curve):.1f})", "#d62728", batch_size, lw=2.4)

        # Dotted line for TD(lambda) converged performance ceiling
        ax.axhline(td_converged, color="black", linestyle=":", lw=2.2, label=f"TD(λ={abl_lmb}) Converged: {td_converged:.1f}")

        ax.set_title(f"{env}: V_aux Regularization Ablations (λ={abl_lmb}, lr={best_clr})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Timesteps", fontsize=10)
        ax.set_ylabel("Episodic Return", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=9, loc="lower right")

        diag4_data.append({
            "Environment": env,
            "Ablation_Lambda": abl_lmb,
            "Best_Critic_LR": best_clr,
            "TD_Converged_Return": td_converged,
            "TD_AUC": compute_auc(td_curve),
            "Baseline_MSE_AUC": compute_auc(base_curve),
            "Baseline_MSE_Final": np.mean(base_curve[-5:]),
            "Huber_AUC": compute_auc(huber_curve),
            "Huber_Final": np.mean(huber_curve[-5:]),
            "Trust_Region_AUC": compute_auc(trust_curve),
            "Trust_Region_Final": np.mean(trust_curve[-5:]),
            "Both_AUC": compute_auc(both_curve),
            "Both_Final": np.mean(both_curve[-5:]),
        })

    plt.tight_layout()
    diag4_path = os.path.join(out_dir, "diagnostics", "diagnostic_4_v_aux_ablations.png")
    plt.savefig(diag4_path)
    plt.close()
    pd.DataFrame(diag4_data).to_csv(os.path.join(out_dir, "diagnostics", "diagnostic_4_v_aux_ablations_summary.csv"), index=False)
    print(f"Saved Diagnostic 4 to {diag4_path}")

    # -------------------------------------------------------------
    # 5. MAIN BENCHMARK: E(lambda*), TD(lambda*), E(0), MC
    # -------------------------------------------------------------
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, n_envs, figsize=(7.0 * n_envs, 5.0), dpi=150, squeeze=False)
    auc_matrix = []
    env_terminations_summary = []

    alg_names = ["E(λ*)", "TD(λ*)", "E(0)", "E(0) Naive Sampling", "MC"]
    colors = {"E(λ*)": "#2ca02c", "TD(λ*)": "#d62728", "E(0)": "#1f77b4", "E(0) Naive Sampling": "#9467bd", "MC": "#7f7f7f"}

    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        env_curves = {}
        env_aucs = {}

        # 1. Best MC (across critic_lrs)
        best_lr_mc = max(critic_lrs, key=lambda lr: compute_auc(all_runs[env]["MC"][lr]["returned_episode_returns"].mean(axis=0)))
        env_curves["MC"] = all_runs[env]["MC"][best_lr_mc]["returned_episode_returns"].mean(axis=0)
        env_curves["MC_std"] = all_runs[env]["MC"][best_lr_mc]["returned_episode_returns"].std(axis=0)
        env_aucs["MC"] = compute_auc(env_curves["MC"])

        # 2. Best E(0) with V_aux (across critic_lrs)
        best_lr_e0 = max(critic_lrs, key=lambda lr: compute_auc(all_runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"].mean(axis=0)))
        env_curves["E(0)"] = all_runs[env]["E(0)_v_aux"][best_lr_e0]["returned_episode_returns"].mean(axis=0)
        env_curves["E(0)_std"] = all_runs[env]["E(0)_v_aux"][best_lr_e0]["returned_episode_returns"].std(axis=0)
        env_aucs["E(0)"] = compute_auc(env_curves["E(0)"])

        # 3. Best E(0) Naive Sampling without V_aux (across critic_lrs)
        best_lr_e0_no = max(critic_lrs, key=lambda lr: compute_auc(all_runs[env]["E(0)_no_v_aux"][lr]["returned_episode_returns"].mean(axis=0)))
        env_curves["E(0) Naive Sampling"] = all_runs[env]["E(0)_no_v_aux"][best_lr_e0_no]["returned_episode_returns"].mean(axis=0)
        env_curves["E(0) Naive Sampling_std"] = all_runs[env]["E(0)_no_v_aux"][best_lr_e0_no]["returned_episode_returns"].std(axis=0)
        env_aucs["E(0) Naive Sampling"] = compute_auc(env_curves["E(0) Naive Sampling"])

        # 4. Best TD(lambda) (across critic_lrs and lambdas)
        best_lr_td = None
        best_lmb_td = None
        best_auc_td = -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_auc = compute_auc(all_runs[env]["TD(lambda)"][clr][lmb]["returned_episode_returns"].mean(axis=0))
                if c_auc > best_auc_td:
                    best_auc_td = c_auc
                    best_lr_td = clr
                    best_lmb_td = lmb
        env_curves["TD(λ*)"] = all_runs[env]["TD(lambda)"][best_lr_td][best_lmb_td]["returned_episode_returns"].mean(axis=0)
        env_curves["TD(λ*)_std"] = all_runs[env]["TD(lambda)"][best_lr_td][best_lmb_td]["returned_episode_returns"].std(axis=0)
        env_aucs["TD(λ*)"] = best_auc_td

        # 5. Best E(lambda) (across critic_lrs and lambdas)
        best_lr_el = None
        best_lmb_el = None
        best_auc_el = -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_auc = compute_auc(all_runs[env]["E(lambda)"][clr][lmb]["returned_episode_returns"].mean(axis=0))
                if c_auc > best_auc_el:
                    best_auc_el = c_auc
                    best_lr_el = clr
                    best_lmb_el = lmb
        env_curves["E(λ*)"] = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]["returned_episode_returns"].mean(axis=0)
        env_curves["E(λ*)_std"] = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]["returned_episode_returns"].std(axis=0)
        env_aucs["E(λ*)"] = best_auc_el

        # Plot curves
        for alg in alg_names:
            c = env_curves[alg]
            s = env_curves[f"{alg}_std"]
            label = f"TD(λ={best_lmb_td})" if alg == "TD(λ*)" else (f"E(λ={best_lmb_el})" if alg == "E(λ*)" else alg)
            plot_curve_with_std(ax, c, s, label, colors[alg], batch_size, lw=2.4, alpha=0.15)

        ax.set_title(f"{env}: Head-to-Head Learning Curves", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Timesteps", fontsize=10)
        ax.set_ylabel("Episodic Return", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=9.5, loc="upper left")

        # Track AUCs
        auc_matrix.append([env_aucs[alg] for alg in alg_names])

        # Track terminations & truncations
        m_best_el = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]
        n_term = float(m_best_el.get("n_terminations", np.zeros(1)).sum())
        n_trunc = float(m_best_el.get("n_truncations", np.zeros(1)).sum())
        env_terminations_summary.append({
            "Environment": env,
            "Total Terminations": n_term,
            "Total Truncations": n_trunc,
            "Best E(λ) λ": best_lmb_el,
            "Best TD(λ) λ": best_lmb_td,
            "Best E(λ) AUC": best_auc_el,
            "Best TD(λ) AUC": best_auc_td,
            "E(0) AUC": env_aucs["E(0)"],
            "E(0) Naive Sampling AUC": env_aucs["E(0) Naive Sampling"],
            "MC AUC": env_aucs["MC"],
        })

        # Save individual environment learning curve figure in plots/
        fig_single, ax_single = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        for alg in alg_names:
            c = env_curves[alg]
            s = env_curves[f"{alg}_std"]
            label = f"TD(λ={best_lmb_td})" if alg == "TD(λ*)" else (f"E(λ={best_lmb_el})" if alg == "E(λ*)" else alg)
            plot_curve_with_std(ax_single, c, s, label, colors[alg], batch_size, lw=2.4, alpha=0.15)
        ax_single.set_title(f"{env}: Critic Learning Curves", fontsize=12, fontweight="bold", pad=10)
        ax_single.set_xlabel("Timesteps", fontsize=10)
        ax_single.set_ylabel("Discounted Return", fontsize=10)
        ax_single.grid(True, linestyle="--", alpha=0.5)
        ax_single.legend(frameon=True, facecolor="white", fontsize=9.5, loc="lower right")
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(plots_dir, f"{env}.png"), bbox_inches="tight")
        fig_single.savefig(os.path.join(plots_dir, f"{env}.pdf"), bbox_inches="tight")
        plt.close(fig_single)

    plt.tight_layout()
    plots_curves_path = os.path.join(plots_dir, "all_envs_learning_curves.png")
    bench_curves_path = os.path.join(out_dir, "benchmarks", "main_comparison_learning_curves.png")
    plt.savefig(plots_curves_path)
    plt.savefig(os.path.join(plots_dir, "main_comparison_learning_curves.png"))
    plt.savefig(bench_curves_path)
    plt.close()
    print(f"Saved Learning Curves to {plots_curves_path} and {bench_curves_path}")

    # -------------------------------------------------------------
    # AUC Matrix Heatmap
    # -------------------------------------------------------------
    auc_arr = np.array(auc_matrix)  # shape (n_envs, 4)
    auc_min = auc_arr.min(axis=1, keepdims=True)
    auc_max = auc_arr.max(axis=1, keepdims=True)
    auc_norm = (auc_arr - auc_min) / np.maximum(auc_max - auc_min, 1e-8)

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 1.2 * n_envs)), dpi=150)
    cax = ax.matshow(auc_norm, cmap="viridis", aspect="auto")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Normalized AUC (Row-wise)")

    ax.set_xticks(range(len(alg_names)))
    ax.set_xticklabels(alg_names, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_envs))
    ax.set_yticklabels(envs, fontsize=11, fontweight="bold")

    for i in range(n_envs):
        for j in range(len(alg_names)):
            val = auc_arr[i, j]
            text_color = "white" if auc_norm[i, j] < 0.65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    ax.set_title("Normalized AUC Comparison Across Critic Architectures", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "benchmarks", "main_comparison_auc_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved AUC Heatmap to {heatmap_path}")

    # Save summary tables
    summary_df = pd.DataFrame(env_terminations_summary)
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
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps (default: auto per env: 1M for MountainCar, 300k for small envs)")
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

#!/usr/bin/env python3
"""
sweep_comprehensive_q_critics.py

Comprehensive Q-PPO Critic Comparison Sweep & Diagnostic Suite:
Applies to deterministic, discrete-action domains:
1. E(lambda)-Q with geometric lookahead jumps (sweeping lambda, selecting best)
2. SARSA(lambda) fitted Q-iteration (sweeping lambda, selecting best)
3. E(0)-Q via Dirichlet energy on state-action pairs
4. MC-Q via Monte Carlo rollout return regression

Notice: In deterministic MDPs with discrete actions, Q-PPO does not require
an auxiliary network V_aux because:
  R_{t:t+K} = G_t - gamma^K G_{t+K}
is exact with zero transition noise along the rollout trajectory.

Analysis Outputs:
1. Diagnostic 1: E(0)-Q vs. E(lambda=0)-Q verification
2. Diagnostic 2: SARSA(lambda) vs. E(lambda)-Q spectrum comparison across lambdas
3. Main Head-to-Head Comparison: E(lambda*-Q), SARSA(lambda*), E(0)-Q, MC-Q
4. Normalized AUC Heatmap across environments
5. Comprehensive Performance Summary CSV
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

# Ensure E_min root is strictly prioritized on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))  # E_min root
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))   # E_min/scripts

sys.path = [_repo_root, _scripts_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

import jax
import jax.numpy as jnp

from gymnax_rl.algos.q_ppo import make_train as make_q_ppo_train


def compute_auc(returns_curve):
    """Computes normalized Area Under Curve (AUC) across update timesteps."""
    return float(np.mean(returns_curve))


def run_experiment(config, seed_keys):
    """JIT-compiles and runs vmapped Q-PPO training across random seeds."""
    train_fn = make_q_ppo_train(config)
    vmapped_fn = jax.jit(jax.vmap(train_fn))
    start_t = time.time()
    out = vmapped_fn(seed_keys)
    # Block until execution finishes to track wall-clock time
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - start_t

    metrics = out["metrics"]
    res = {
        "returned_episode_returns": np.array(metrics["returned_episode_returns"]),
        "returned_episode_lengths": np.array(metrics["returned_episode_lengths"]),
        "q_loss": np.array(metrics.get("q_loss", metrics.get("critic_loss", np.zeros(1)))),
        "actor_loss": np.array(metrics.get("actor_loss", np.zeros(1))),
        "entropy": np.array(metrics.get("entropy", np.zeros(1))),
        "elapsed_time": elapsed,
    }
    if "n_terminations" in metrics:
        res["n_terminations"] = np.array(metrics["n_terminations"])
    if "n_truncations" in metrics:
        res["n_truncations"] = np.array(metrics["n_truncations"])
    return res


def run_sweep(args):
    """Executes the Q-PPO critic sweep across deterministic discrete environments."""
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

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "benchmarks"), exist_ok=True)

    master_key = jax.random.PRNGKey(master_seed)
    seed_keys = jax.random.split(master_key, n_seeds)

    all_runs = {}

    # Check if user requested loading existing data
    if getattr(args, "load_existing", None) and os.path.exists(args.load_existing):
        print(f"Loading existing metrics from: {args.load_existing}")
        with open(args.load_existing, "rb") as f:
            loaded = pickle.load(f)
            all_runs = loaded.get("runs", {})

    batch_size = args.num_envs * args.num_steps
    num_updates = max(1, args.total_timesteps // batch_size)
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
                    if "returned_episode_returns" in first_val:
                        curve_len = first_val["returned_episode_returns"].shape[-1]
                        timesteps = np.arange(1, curve_len + 1) * batch_size
                        break

    print(f"=================================================================")
    print(f"STARTING COMPREHENSIVE Q-PPO CRITIC SWEEP (DETERMINISTIC MDPs)")
    print(f"Envs: {envs}")
    print(f"Rollouts: {num_envs} envs x {num_steps} steps (batch {batch_size})")
    print(f"Updates: {num_updates} | Total timesteps: {total_timesteps}")
    print(f"Critic LRs: {critic_lrs} | Lambdas: {lambdas} | Seeds: {n_seeds}")
    print(f"=================================================================\n")

    for env_name in envs:
        if env_name not in all_runs:
            all_runs[env_name] = {
                "MC": {},
                "E(0)": {},
                "SARSA(lambda)": {lr: {} for lr in critic_lrs},
                "E(lambda)": {lr: {} for lr in critic_lrs},
            }

        print(f"\n>>> Running Environment: {env_name} <<<")

        # -------------------------------------------------------------
        # 1. MC-Q Baseline (swept across critic LRs)
        # -------------------------------------------------------------
        for lr in critic_lrs:
            if lr not in all_runs[env_name]["MC"]:
                print(f"[MC-Q] env={env_name}, critic_lr={lr}...")
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

        # -------------------------------------------------------------
        # 2. E(0)-Q (swept across critic LRs)
        # -------------------------------------------------------------
        for lr in critic_lrs:
            if lr not in all_runs[env_name]["E(0)"]:
                print(f"[E(0)-Q] env={env_name}, critic_lr={lr}...")
                cfg = {
                    "ENV_NAME": env_name,
                    "LR": actor_lr,
                    "ACTOR_LR": actor_lr,
                    "CRITIC_LR": lr,
                    "CRITIC_TYPE": "e_0_q",
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
                all_runs[env_name]["E(0)"][lr] = res
                print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # -------------------------------------------------------------
        # 3. SARSA(lambda) Grid (critic_lrs x lambdas)
        # -------------------------------------------------------------
        for lr in critic_lrs:
            for lmb in lambdas:
                if lmb not in all_runs[env_name]["SARSA(lambda)"][lr]:
                    print(f"[SARSA(λ)] env={env_name}, lambda={lmb}, critic_lr={lr}...")
                    cfg = {
                        "ENV_NAME": env_name,
                        "LR": actor_lr,
                        "ACTOR_LR": actor_lr,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "sarsa",
                        "VALUE_LAMBDA": lmb,
                        "RETURN_LAMBDA": lmb,
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
                    all_runs[env_name]["SARSA(lambda)"][lr][lmb] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # -------------------------------------------------------------
        # 4. E(lambda)-Q Grid (critic_lrs x lambdas)
        # -------------------------------------------------------------
        for lr in critic_lrs:
            for lmb in lambdas:
                if lmb not in all_runs[env_name]["E(lambda)"][lr]:
                    print(f"[E(λ)-Q] env={env_name}, lambda={lmb}, critic_lr={lr}...")
                    cfg = {
                        "ENV_NAME": env_name,
                        "LR": actor_lr,
                        "ACTOR_LR": actor_lr,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": "e_lambda_q",
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

    # Save raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate Diagnostics & Benchmark Plots
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args)


def plot_curve_with_std(ax, curve, std, label, color, batch_size, lw=2.2, alpha=0.12, linestyle="-"):
    t = np.arange(1, len(curve) + 1) * batch_size
    ax.plot(t, curve, label=label, color=color, lw=lw, linestyle=linestyle)
    if std is not None:
        ax.fill_between(t, curve - std, curve + std, color=color, alpha=alpha)


def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args):
    """Generates diagnostic and benchmark figures and summary tables for Q-PPO."""
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

    # -------------------------------------------------------------
    # Diagnostic 1: E(0)-Q vs. E(lambda=0)-Q
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, n_envs, figsize=(7.0 * n_envs, 5.0), dpi=150, squeeze=False)
    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        best_clr_e0 = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(0)"][lr]["returned_episode_returns"]))
        e0_curve = all_runs[env]["E(0)"][best_clr_e0]["returned_episode_returns"].mean(axis=0)
        e0_std = all_runs[env]["E(0)"][best_clr_e0]["returned_episode_returns"].std(axis=0)

        best_clr_el0 = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(lambda)"][lr][0.0]["returned_episode_returns"]))
        el0_curve = all_runs[env]["E(lambda)"][best_clr_el0][0.0]["returned_episode_returns"].mean(axis=0)
        el0_std = all_runs[env]["E(lambda)"][best_clr_el0][0.0]["returned_episode_returns"].std(axis=0)

        plot_curve_with_std(ax, e0_curve, e0_std, f"E(0)-Q (lr={best_clr_e0})", "#1f77b4", batch_size, lw=2.4)
        plot_curve_with_std(ax, el0_curve, el0_std, f"E(λ=0)-Q (lr={best_clr_el0})", "#ff7f0e", batch_size, lw=2.2, linestyle="--")

        ax.set_title(f"{env}: Diagnostic 1 (E(0)-Q vs E(λ=0)-Q)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Timesteps", fontsize=10)
        ax.set_ylabel("Episodic Return", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=9.5)

    plt.tight_layout()
    diag1_path = os.path.join(out_dir, "diagnostics", "diagnostic_1_e0_vs_elambda0.png")
    plt.savefig(diag1_path)
    plt.close()
    print(f"Saved Diagnostic 1 to {diag1_path}")

    # -------------------------------------------------------------
    # Diagnostic 2: SARSA(lambda) vs. E(lambda)-Q Across Lambda Spectrum
    # -------------------------------------------------------------
    fig, axes = plt.subplots(len(lambdas), n_envs, figsize=(7.0 * n_envs, 4.0 * len(lambdas)), dpi=150, squeeze=False)
    for col, env in enumerate(envs):
        for row, lmb in enumerate(lambdas):
            ax = axes[row, col]
            best_clr_sarsa = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["SARSA(lambda)"][lr][lmb]["returned_episode_returns"]))
            sarsa_curve = all_runs[env]["SARSA(lambda)"][best_clr_sarsa][lmb]["returned_episode_returns"].mean(axis=0)
            sarsa_std = all_runs[env]["SARSA(lambda)"][best_clr_sarsa][lmb]["returned_episode_returns"].std(axis=0)

            best_clr_el = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(lambda)"][lr][lmb]["returned_episode_returns"]))
            el_curve = all_runs[env]["E(lambda)"][best_clr_el][lmb]["returned_episode_returns"].mean(axis=0)
            el_std = all_runs[env]["E(lambda)"][best_clr_el][lmb]["returned_episode_returns"].std(axis=0)

            plot_curve_with_std(ax, sarsa_curve, sarsa_std, f"SARSA(λ={lmb})", "#4575b4", batch_size, lw=2.0)
            plot_curve_with_std(ax, el_curve, el_std, f"E(λ={lmb})-Q", "#d73027", batch_size, lw=2.2)

            ax.set_title(f"{env} (λ={lmb})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Timesteps", fontsize=9)
            ax.set_ylabel("Episodic Return", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(frameon=True, facecolor="white", fontsize=8.5, loc="lower right")

    plt.tight_layout()
    diag2_path = os.path.join(out_dir, "diagnostics", "diagnostic_2_sarsa_vs_elambda_spectrum.png")
    plt.savefig(diag2_path)
    plt.close()
    print(f"Saved Diagnostic 2 to {diag2_path}")

    # -------------------------------------------------------------
    # Main Head-to-Head Comparison: E(lambda*-Q), SARSA(lambda*), E(0)-Q, MC-Q
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, n_envs, figsize=(7.5 * n_envs, 5.5), dpi=150, squeeze=False)
    alg_names = ["MC-Q", "E(0)-Q", "SARSA(λ*)", "E(λ*-Q)"]
    colors = {
        "MC-Q": "#7f7f7f",
        "E(0)-Q": "#1f77b4",
        "SARSA(λ*)": "#ff7f0e",
        "E(λ*-Q)": "#2ca02c",
    }

    auc_matrix = []
    summary_rows = []

    for idx, env in enumerate(envs):
        ax = axes[0, idx]
        env_curves = {}
        env_aucs = {}

        # 1. Best MC-Q (across critic_lrs)
        best_lr_mc = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["MC"][lr]["returned_episode_returns"]))
        env_curves["MC-Q"] = all_runs[env]["MC"][best_lr_mc]["returned_episode_returns"].mean(axis=0)
        env_curves["MC-Q_std"] = all_runs[env]["MC"][best_lr_mc]["returned_episode_returns"].std(axis=0)
        env_aucs["MC-Q"] = compute_auc(env_curves["MC-Q"])

        # 2. Best E(0)-Q (across critic_lrs)
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(all_runs[env]["E(0)"][lr]["returned_episode_returns"]))
        env_curves["E(0)-Q"] = all_runs[env]["E(0)"][best_lr_e0]["returned_episode_returns"].mean(axis=0)
        env_curves["E(0)-Q_std"] = all_runs[env]["E(0)"][best_lr_e0]["returned_episode_returns"].std(axis=0)
        env_aucs["E(0)-Q"] = compute_auc(env_curves["E(0)-Q"])

        # 3. Best SARSA(lambda) (across critic_lrs and lambdas)
        best_lr_sarsa = None
        best_lmb_sarsa = None
        best_auc_sarsa = -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_auc = compute_auc(all_runs[env]["SARSA(lambda)"][clr][lmb]["returned_episode_returns"].mean(axis=0))
                if c_auc > best_auc_sarsa:
                    best_auc_sarsa = c_auc
                    best_lr_sarsa = clr
                    best_lmb_sarsa = lmb
        env_curves["SARSA(λ*)"] = all_runs[env]["SARSA(lambda)"][best_lr_sarsa][best_lmb_sarsa]["returned_episode_returns"].mean(axis=0)
        env_curves["SARSA(λ*)_std"] = all_runs[env]["SARSA(lambda)"][best_lr_sarsa][best_lmb_sarsa]["returned_episode_returns"].std(axis=0)
        env_aucs["SARSA(λ*)"] = best_auc_sarsa

        # 4. Best E(lambda)-Q (across critic_lrs and lambdas)
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
        env_curves["E(λ*-Q)"] = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]["returned_episode_returns"].mean(axis=0)
        env_curves["E(λ*-Q)_std"] = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]["returned_episode_returns"].std(axis=0)
        env_aucs["E(λ*-Q)"] = best_auc_el

        # Plot curves
        for alg in alg_names:
            c = env_curves[alg]
            s = env_curves[f"{alg}_std"]
            label = f"{alg} (λ={best_lmb_el})" if alg == "E(λ*-Q)" else (f"{alg} (λ={best_lmb_sarsa})" if alg == "SARSA(λ*)" else alg)
            plot_curve_with_std(ax, c, s, label, colors[alg], batch_size, lw=2.4, alpha=0.15)

        ax.set_title(f"{env}: Head-to-Head Q-PPO Comparison", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Timesteps", fontsize=10)
        ax.set_ylabel("Episodic Return", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", fontsize=9.5, loc="upper left")

        auc_matrix.append([env_aucs[alg] for alg in alg_names])

        m_best_el = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]
        n_term = float(m_best_el.get("n_terminations", np.zeros(1)).sum())
        n_trunc = float(m_best_el.get("n_truncations", np.zeros(1)).sum())
        summary_rows.append({
            "Environment": env,
            "Best E(λ)-Q λ": best_lmb_el,
            "Best E(λ)-Q Return": env_curves["E(λ*-Q)"][-1],
            "Best E(λ)-Q AUC": best_auc_el,
            "Best SARSA(λ) λ": best_lmb_sarsa,
            "Best SARSA(λ) Return": env_curves["SARSA(λ*)"][-1],
            "Best SARSA(λ) AUC": best_auc_sarsa,
            "E(0)-Q Return": env_curves["E(0)-Q"][-1],
            "E(0)-Q AUC": env_aucs["E(0)-Q"],
            "MC-Q Return": env_curves["MC-Q"][-1],
            "MC-Q AUC": env_aucs["MC-Q"],
            "Terminations": n_term,
            "Truncations": n_trunc,
        })

    plt.tight_layout()
    bench_curves_path = os.path.join(out_dir, "benchmarks", "main_comparison_learning_curves.png")
    plt.savefig(bench_curves_path)
    plt.close()
    print(f"Saved Main Comparison Learning Curves to {bench_curves_path}")

    # -------------------------------------------------------------
    # Normalized AUC Heatmap
    # -------------------------------------------------------------
    auc_arr = np.array(auc_matrix)  # shape (n_envs, 4)
    auc_min = auc_arr.min(axis=1, keepdims=True)
    auc_max = auc_arr.max(axis=1, keepdims=True)
    auc_norm = (auc_arr - auc_min) / np.maximum(auc_max - auc_min, 1e-8)

    fig, ax = plt.subplots(figsize=(8.0, max(3.5, 1.2 * n_envs)), dpi=150)
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

    ax.set_title("Normalized AUC Comparison Across Q-Critic Architectures", pad=20, fontsize=12, fontweight="bold")
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "benchmarks", "main_comparison_auc_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved AUC Heatmap to {heatmap_path}")

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, "benchmarks", "main_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved Summary CSV to {csv_path}")
    print("\nAll Q-PPO figures and diagnostics generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Q-PPO Critic Comparison Sweep")
    parser.add_argument("--envs", nargs="+", default=["CartPole-v1"], help="List of deterministic discrete environments")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.4, 0.8, 0.95], help="Lambdas to sweep")
    parser.add_argument("--critic-lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3], help="Critic learning rates")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--total-timesteps", type=int, default=300000, help="Total training timesteps")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments (rollouts)")
    parser.add_argument("--num-steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--minibatch-size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n-seeds", type=int, default=3, help="Random seeds per configuration")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--out-dir", type=str, default="results/sweep_comprehensive_q_critic", help="Output directory")
    parser.add_argument("--load-existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

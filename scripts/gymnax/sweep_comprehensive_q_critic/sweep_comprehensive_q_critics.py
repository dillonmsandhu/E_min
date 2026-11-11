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

Visualizations (Modeled after sampled_lambda_sweep):
- main_comparison_learning_curves: 2x3 grid comparing the 4 key Q-critic algorithms.
- lambda_sensitivity_curves: 2x3 grid plotting Performance vs. lambda for SARSA(lambda) vs. E(lambda)-Q.
- main_comparison_auc_table: Publication-grade regular table with heatmap coloring for AUC.
- main_comparison_final_return_table: Publication-grade regular table with heatmap coloring for final returns.
- main_comparison_summary.csv: Tabulating metrics across all environments.
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

# Ensure E_min root is strictly prioritized on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))  # E_min root
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))   # E_min/scripts

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

import jax
import jax.numpy as jnp

from gymnax_rl.algos.q_ppo import make_train as make_q_ppo_train

ENV_TITLES = {
    "CartPole-v1": "CartPole-v1",
    "Acrobot-v1": "Acrobot-v1",
    "MountainCar-v0": "Mountain Car (Sparse)",
    "DeepSea-bsuite": "DeepSea (bsuite)",
    "DiscountingChain-bsuite": "Discounting Chain (bsuite)",
    "Pong-misc": "Pong",
    "Asterix-MinAtar": "Asterix (MinAtar)",
    "Breakout-MinAtar": "Breakout (MinAtar)",
    "Freeway-MinAtar": "Freeway (MinAtar)",
    "SpaceInvaders-MinAtar": "Space Invaders (MinAtar)",
}


def get_env_default_timesteps(env_name: str, fallback_timesteps: int | None = None) -> int:
    """Returns environment-specific timesteps: 10M on MinAtar, 1M on MountainCar, 300k on others."""
    if fallback_timesteps is not None:
        return fallback_timesteps
    if "MinAtar" in env_name:
        return 10_000_000
    elif "MountainCar" in env_name:
        return 1_000_000
    else:
        return 300_000


def compute_auc(returns_curve):
    """Computes normalized Area Under Curve (AUC) across update timesteps."""
    return float(np.mean(returns_curve))


def get_metric_curve(run_dict, metric_key="returned_episode_returns"):
    """Safely extracts the specified metric curve, falling back to returned_episode_returns or returned_discounted_episode_returns."""
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


def run_experiment(config, seed_keys):
    """JIT-compiles and runs vmapped Q-PPO training across random seeds."""
    train_fn = make_q_ppo_train(config)
    vmapped_fn = jax.jit(jax.vmap(train_fn))
    start_t = time.time()
    out = vmapped_fn(seed_keys)
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

    if getattr(args, "load_existing", None) and os.path.exists(args.load_existing):
        print(f"Loading existing metrics from: {args.load_existing}")
        with open(args.load_existing, "rb") as f:
            loaded = pickle.load(f)
            all_runs = loaded.get("runs", {})

    batch_size = args.num_envs * args.num_steps
    base_timesteps = args.total_timesteps or 300000
    num_updates = max(1, base_timesteps // batch_size)
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
    print(f"STARTING COMPREHENSIVE Q-PPO CRITIC SWEEP (DETERMINISTIC MDPs & MINATAR)")
    print(f"Envs: {envs}")
    print(f"Rollouts: {num_envs} envs x {num_steps} steps (batch {batch_size})")
    print(f"Timesteps Policy: Auto (10M for MinAtar, 1M for MountainCar, 300k for small envs) | Override: {args.total_timesteps}")
    print(f"Critic LRs: {critic_lrs} | Lambdas: {lambdas} | Seeds: {n_seeds}")
    print(f"=================================================================\n")

    for env_name in envs:
        env_timesteps = get_env_default_timesteps(env_name, args.total_timesteps)
        if env_name not in all_runs:
            all_runs[env_name] = {
                "MC": {},
                "E(0)": {},
                "SARSA(lambda)": {lr: {} for lr in critic_lrs},
                "E(lambda)": {lr: {} for lr in critic_lrs},
            }

        print(f"\n>>> Running Environment: {env_name} (Timesteps: {env_timesteps:,}) <<<")

        # 1. MC-Q Baseline
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
                    "TOTAL_TIMESTEPS": env_timesteps,
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

        # 2. E(0)-Q
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
                    "TOTAL_TIMESTEPS": env_timesteps,
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

        # 3. SARSA(lambda) Grid
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
                        "TOTAL_TIMESTEPS": env_timesteps,
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

        # 4. E(lambda)-Q Grid
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
                        "TOTAL_TIMESTEPS": env_timesteps,
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

    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, lambdas, args)


def plot_curve_with_std(ax, curve, std, label, color, batch_size, lw=2.0, alpha=0.15, linestyle="-"):
    t = np.arange(1, len(curve) + 1) * batch_size
    ax.plot(t, curve, label=label, color=color, lw=lw, linestyle=linestyle)
    if std is not None:
        ax.fill_between(t, curve - std, curve + std, color=color, alpha=alpha)


def format_cell(mean_val, std_val):
    if abs(mean_val) >= 100:
        return f"{mean_val:.1f} ± {std_val:.1f}"
    elif abs(mean_val) >= 10:
        return f"{mean_val:.2f} ± {std_val:.2f}"
    else:
        return f"{mean_val:.3f} ± {std_val:.3f}"


def format_tex_cell(mean_val, std_val, is_best=False):
    if abs(mean_val) >= 100:
        s = f"{mean_val:.1f} \\pm {std_val:.1f}"
    elif abs(mean_val) >= 10:
        s = f"{mean_val:.2f} \\pm {std_val:.2f}"
    else:
        s = f"{mean_val:.3f} \\pm {std_val:.3f}"
    if is_best:
        return f"\\textbf{{${s}$}}"
    return f"${s}$"


def format_md_cell(mean_val, std_val, is_best=False):
    s = format_cell(mean_val, std_val)
    if is_best:
        return f"**{s}**"
    return s


def export_table_bundle(folder_path, title, subtitle, col_names, row_names, mean_mat, std_mat, higher_is_better=True, cmap="Blues"):
    """Exports structured table bundle: table.md, table.tex, and vector table.pdf / table.png."""
    os.makedirs(folder_path, exist_ok=True)
    n_rows, n_cols = len(row_names), len(col_names)

    # 1. table.md (GitHub Flavored Markdown)
    md_lines = [
        f"# {title}",
        f"*{subtitle}*",
        "",
        "| Environment | " + " | ".join(col_names) + " |",
        "| :--- | " + " | ".join([":---:"] * n_cols) + " |",
    ]
    for r in range(n_rows):
        row_vals = mean_mat[r]
        best_c = np.nanargmax(row_vals) if higher_is_better else np.nanargmin(row_vals)
        cells = [row_names[r]]
        for c in range(n_cols):
            cells.append(format_md_cell(mean_mat[r, c], std_mat[r, c], is_best=(c == best_c)))
        md_lines.append("| " + " | ".join(cells) + " |")

    with open(os.path.join(folder_path, "table.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # 2. table.tex (LaTeX with booktabs)
    col_align = "l" + "c" * n_cols
    tex_cols = [c.replace("λ*", r"$\lambda^*$").replace("λ", r"$\lambda$").replace("_", r"\_") for c in col_names]
    clean_title = title.replace("λ", r"$\lambda$").replace("_", r"\_")
    clean_sub = subtitle.replace("λ", r"$\lambda$").replace("∈", r"$\in$").replace("_", r"\_")
    tex_lines = [
        r"% Auto-generated publication table",
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{clean_title}: {clean_sub}}}",
        r"\label{tab:" + os.path.basename(folder_path) + r"}",
        r"\small",
        f"\\begin{{tabular}}{{{col_align}}}",
        r"\toprule",
        "Environment & " + " & ".join(tex_cols) + r" \\",
        r"\midrule",
    ]
    for r in range(n_rows):
        row_vals = mean_mat[r]
        best_c = np.nanargmax(row_vals) if higher_is_better else np.nanargmin(row_vals)
        cells = [row_names[r]]
        for c in range(n_cols):
            cells.append(format_tex_cell(mean_mat[r, c], std_mat[r, c], is_best=(c == best_c)))
        tex_lines.append(" & ".join(cells) + r" \\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    with open(os.path.join(folder_path, "table.tex"), "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    # 3. table.pdf & table.png
    fig_w = max(8.5, n_cols * 2.0 + 2.5)
    fig_h = max(3.8, n_rows * 0.65 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    row_mins = np.nanmin(mean_mat, axis=1, keepdims=True)
    row_maxs = np.nanmax(mean_mat, axis=1, keepdims=True)
    row_ranges = np.maximum(row_maxs - row_mins, 1e-8)
    mat_norm = (mean_mat - row_mins) / row_ranges if higher_is_better else (row_maxs - mean_mat) / row_ranges

    color_map = getattr(plt.cm, cmap)
    im = ax.imshow(mat_norm, cmap=color_map, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_names, fontsize=10.5, fontweight="bold", rotation=20, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_names, fontsize=10.5, fontweight="bold")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d7de", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for r in range(n_rows):
        best_c = np.nanargmax(mean_mat[r]) if higher_is_better else np.nanargmin(mean_mat[r])
        for c in range(n_cols):
            val = mean_mat[r, c]
            std = std_mat[r, c]
            norm_val = mat_norm[r, c]

            is_best = (c == best_c)
            txt = format_cell(val, std)
            text_color = "white" if norm_val > 0.60 else "#111111"
            weight = "bold" if is_best else "normal"

            if is_best:
                rect = plt.Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor="#0b6623", linewidth=2.5)
                ax.add_patch(rect)

            ax.text(c, r, txt, ha="center", va="center", color=text_color, fontsize=9.5, fontweight=weight)

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.04)
    cbar.set_label("Relative Score (0=Min, 1=Max)", fontsize=10, fontweight="bold")
    ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=14)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "table.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(folder_path, "table.png"), bbox_inches="tight")
    plt.close()


def render_heatmap_table(df_mean, df_std, title, cbar_label, output_base, cmap="Blues"):
    """
    Renders a publication-grade table with heatmap-colored cells:
    Rows = Environments, Columns = Conditions.
    Each cell contains formatted numerical text (Mean ± Std),
    background color shows relative rank, and best condition is bordered green.
    """
    n_rows, n_cols = df_mean.shape
    fig_w = max(8.5, n_cols * 1.85 + 2.0)
    fig_h = max(3.5, n_rows * 0.70 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    mat_mean = df_mean.values
    mat_std = df_std.values if df_std is not None else np.zeros_like(mat_mean)

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
    """Generates the streamlined publication-quality figures, plots directory, and heatmap tables for Q-PPO."""
    os.makedirs(out_dir, exist_ok=True)
    benchmarks_dir = os.path.join(out_dir, "benchmarks")
    diagnostics_dir = os.path.join(out_dir, "diagnostics")
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(benchmarks_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    metric_key = getattr(args, "metric_key", "returned_episode_returns")
    batch_size = args.num_envs * args.num_steps

    for env in envs:
        for k in all_runs[env]:
            sub = all_runs[env][k]
            first_val = list(sub.values())[0]
            if isinstance(first_val, dict):
                first_val = list(first_val.values())[0]
            if "returned_episode_returns" in first_val or "returned_discounted_episode_returns" in first_val:
                c_data = first_val.get("returned_episode_returns", first_val.get("returned_discounted_episode_returns"))
                curve_len = c_data.shape[-1]
                timesteps = np.arange(1, curve_len + 1) * batch_size
                break

    print(f"\n================ GENERATING ANALYSES & FIGURES (Q-PPO | Metric: {metric_key}) ================")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    n_envs = len(envs)

    n_cols = min(n_envs, 3)
    n_rows = math.ceil(n_envs / n_cols)

    # -------------------------------------------------------------
    # 1. MAIN COMPARISON: 4 Core Curves (Grid Across Environments)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    axes_flat = axes.flatten()

    alg_names = ["MC-Q", "E(0)-Q", "SARSA(λ*)", "E(λ*-Q)"]
    colors = {
        "MC-Q": "#7f7f7f",
        "E(0)-Q": "#1f77b4",
        "SARSA(λ*)": "#e6550d",
        "E(λ*-Q)": "#2ca02c",
    }
    linestyles = {
        "MC-Q": "-",
        "E(0)-Q": "-",
        "SARSA(λ*)": "--",
        "E(λ*-Q)": "-",
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

        # 1. MC-Q
        best_lr_mc = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["MC"][lr], metric_key)))
        mc_data = get_metric_curve(all_runs[env]["MC"][best_lr_mc], metric_key)
        env_curves["MC-Q"] = mc_data.mean(axis=0)
        env_stds["MC-Q"] = mc_data.std(axis=0)
        auc_means_dict["MC-Q"].append(compute_auc(env_curves["MC-Q"]))
        auc_stds_dict["MC-Q"].append(np.mean(env_stds["MC-Q"]))
        final_means_dict["MC-Q"].append(env_curves["MC-Q"][-1])
        final_stds_dict["MC-Q"].append(env_stds["MC-Q"][-1])

        # 2. E(0)-Q
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(0)"][lr], metric_key)))
        e0_data = get_metric_curve(all_runs[env]["E(0)"][best_lr_e0], metric_key)
        env_curves["E(0)-Q"] = e0_data.mean(axis=0)
        env_stds["E(0)-Q"] = e0_data.std(axis=0)
        auc_means_dict["E(0)-Q"].append(compute_auc(env_curves["E(0)-Q"]))
        auc_stds_dict["E(0)-Q"].append(np.mean(env_stds["E(0)-Q"]))
        final_means_dict["E(0)-Q"].append(env_curves["E(0)-Q"][-1])
        final_stds_dict["E(0)-Q"].append(env_stds["E(0)-Q"][-1])

        # 3. SARSA(lambda*)
        best_lr_sarsa, best_lmb_sarsa, best_auc_sarsa = None, None, -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_data = get_metric_curve(all_runs[env]["SARSA(lambda)"][clr][lmb], metric_key)
                c_auc = compute_auc(c_data.mean(axis=0))
                if c_auc > best_auc_sarsa:
                    best_auc_sarsa = c_auc
                    best_lr_sarsa = clr
                    best_lmb_sarsa = lmb
        sarsa_data = get_metric_curve(all_runs[env]["SARSA(lambda)"][best_lr_sarsa][best_lmb_sarsa], metric_key)
        env_curves["SARSA(λ*)"] = sarsa_data.mean(axis=0)
        env_stds["SARSA(λ*)"] = sarsa_data.std(axis=0)
        auc_means_dict["SARSA(λ*)"].append(best_auc_sarsa)
        auc_stds_dict["SARSA(λ*)"].append(np.mean(env_stds["SARSA(λ*)"]))
        final_means_dict["SARSA(λ*)"].append(env_curves["SARSA(λ*)"][-1])
        final_stds_dict["SARSA(λ*)"].append(env_stds["SARSA(λ*)"][-1])

        # 4. E(lambda*-Q)
        best_lr_el, best_lmb_el, best_auc_el = None, None, -1e9
        for clr in critic_lrs:
            for lmb in lambdas:
                c_data = get_metric_curve(all_runs[env]["E(lambda)"][clr][lmb], metric_key)
                c_auc = compute_auc(c_data.mean(axis=0))
                if c_auc > best_auc_el:
                    best_auc_el = c_auc
                    best_lr_el = clr
                    best_lmb_el = lmb
        el_data = get_metric_curve(all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el], metric_key)
        env_curves["E(λ*-Q)"] = el_data.mean(axis=0)
        env_stds["E(λ*-Q)"] = el_data.std(axis=0)
        auc_means_dict["E(λ*-Q)"].append(best_auc_el)
        auc_stds_dict["E(λ*-Q)"].append(np.mean(env_stds["E(λ*-Q)"]))
        final_means_dict["E(λ*-Q)"].append(env_curves["E(λ*-Q"][-1] if "E(λ*-Q" in env_curves else env_curves["E(λ*-Q)"][-1])
        final_stds_dict["E(λ*-Q)"].append(env_stds["E(λ*-Q)"][-1])

        # Plot curves on multi-panel grid
        for alg in alg_names:
            c = env_curves[alg]
            s = env_stds[alg]
            lbl = f"SARSA(λ={best_lmb_sarsa})" if alg == "SARSA(λ*)" else (f"E(λ={best_lmb_el})" if alg == "E(λ*-Q)" else alg)
            plot_curve_with_std(ax, c, s, lbl, colors[alg], batch_size, lw=2.2 if "E(λ" in alg or "SARSA" in alg else 1.8, linestyle=linestyles[alg])

        ax.set_title(env_title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Timesteps", fontsize=9.5)
        ax.set_ylabel("Episodic Return", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=7.2, loc="best")

        # Save individual per-environment learning curve figure in plots/
        fig_single, ax_single = plt.subplots(figsize=(6.5, 4.5), dpi=300)
        for alg in alg_names:
            c = env_curves[alg]
            s = env_stds[alg]
            lbl = f"SARSA(λ={best_lmb_sarsa})" if alg == "SARSA(λ*)" else (f"E(λ={best_lmb_el})" if alg == "E(λ*-Q)" else alg)
            plot_curve_with_std(ax_single, c, s, lbl, colors[alg], batch_size, lw=2.2 if "E(λ" in alg or "SARSA" in alg else 1.8, linestyle=linestyles[alg])
        ax_single.set_title(f"{env_title}: Q-Critic Learning Curves", fontsize=12, fontweight="bold", pad=10)
        ax_single.set_xlabel("Timesteps", fontsize=10)
        ax_single.set_ylabel("Episodic Return", fontsize=10)
        ax_single.grid(True, linestyle=":", alpha=0.5)
        ax_single.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="best")
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(plots_dir, f"{env}.png"), bbox_inches="tight")
        fig_single.savefig(os.path.join(plots_dir, f"{env}.pdf"), bbox_inches="tight")
        plt.close(fig_single)

        m_best_el = all_runs[env]["E(lambda)"][best_lr_el][best_lmb_el]
        n_term = float(m_best_el.get("n_terminations", np.zeros(1)).sum())
        n_trunc = float(m_best_el.get("n_truncations", np.zeros(1)).sum())
        summary_rows.append({
            "Environment": env_title,
            "Best E(λ)-Q λ": best_lmb_el,
            "Best E(λ)-Q Return": env_curves["E(λ*-Q)"][-1],
            "Best E(λ)-Q AUC": best_auc_el,
            "Best SARSA(λ) λ": best_lmb_sarsa,
            "Best SARSA(λ) Return": env_curves["SARSA(λ*)"][-1],
            "Best SARSA(λ) AUC": best_auc_sarsa,
            "E(0)-Q Return": env_curves["E(0)-Q"][-1],
            "E(0)-Q AUC": auc_means_dict["E(0)-Q"][-1],
            "MC-Q Return": env_curves["MC-Q"][-1],
            "MC-Q AUC": auc_means_dict["MC-Q"][-1],
            "Terminations": n_term,
            "Truncations": n_trunc,
        })

    for extra_idx in range(n_envs, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    plt.tight_layout()
    bench_curves_base = os.path.join(benchmarks_dir, "main_comparison_learning_curves")
    plots_grid_base = os.path.join(plots_dir, "all_envs_learning_curves")
    plots_main_base = os.path.join(plots_dir, "main_comparison_learning_curves")
    plt.savefig(f"{bench_curves_base}.png", bbox_inches="tight")
    plt.savefig(f"{bench_curves_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{plots_grid_base}.png", bbox_inches="tight")
    plt.savefig(f"{plots_grid_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{plots_main_base}.png", bbox_inches="tight")
    plt.savefig(f"{plots_main_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved Main Comparison Learning Curves to {bench_curves_base}.png, {plots_grid_base}.png, and {plots_main_base}.png")

    # -------------------------------------------------------------
    # 2. DIAGNOSTIC 1: Pure Dirichlet E(0)-Q vs 1-Step E(λ=0)-Q
    # -------------------------------------------------------------
    fig_d1, axes_d1 = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    axes_d1_flat = axes_d1.flatten()

    for idx, env in enumerate(envs):
        ax = axes_d1_flat[idx]
        env_title = ENV_TITLES.get(env, env)

        # Best E(0)-Q
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(0)"][lr], metric_key)))
        e0_c = get_metric_curve(all_runs[env]["E(0)"][best_lr_e0], metric_key)
        plot_curve_with_std(ax, e0_c.mean(axis=0), e0_c.std(axis=0), "E(0)-Q (Action Anchor)", "#1f77b4", batch_size, lw=2.2)

        # Best E(lambda=0)-Q
        best_lr_el0 = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(lambda)"][lr][0.0], metric_key)))
        el0_c = get_metric_curve(all_runs[env]["E(lambda)"][best_lr_el0][0.0], metric_key)
        plot_curve_with_std(ax, el0_c.mean(axis=0), el0_c.std(axis=0), "E(λ=0)-Q (1-Step Jump)", "#9467bd", batch_size, lw=2.0, linestyle="--")

        ax.set_title(f"{env_title}: E(0) vs E(λ=0)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Timesteps", fontsize=9.5)
        ax.set_ylabel("Episodic Return", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.0, loc="best")

    for extra_idx in range(n_envs, len(axes_d1_flat)):
        axes_d1_flat[extra_idx].set_visible(False)

    plt.tight_layout()
    diag1_base = os.path.join(diagnostics_dir, "diagnostic_1_e0_vs_elambda0")
    diag1_plots_base = os.path.join(plots_dir, "diagnostic_1_e0_vs_elambda0")
    plt.savefig(f"{diag1_base}.png", bbox_inches="tight")
    plt.savefig(f"{diag1_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{diag1_plots_base}.png", bbox_inches="tight")
    plt.savefig(f"{diag1_plots_base}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved Diagnostic 1 (E(0)-Q vs E(λ=0)-Q) to {diag1_base}.png and {diag1_plots_base}.png")

    # -------------------------------------------------------------
    # 3. DIAGNOSTIC 2: SARSA(λ) vs. E(λ)-Q Lookahead Spectrum
    # -------------------------------------------------------------
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), dpi=300, squeeze=False)
    axes_flat = axes.flatten()

    for idx, env in enumerate(envs):
        ax = axes_flat[idx]
        env_title = ENV_TITLES.get(env, env)

        sarsa_lam_aucs = []
        elam_lam_aucs = []

        for lmb in lambdas:
            best_lr_s = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["SARSA(lambda)"][lr][lmb], metric_key)))
            sarsa_lam_aucs.append(compute_auc(get_metric_curve(all_runs[env]["SARSA(lambda)"][best_lr_s][lmb], metric_key).mean(axis=0)))

            best_lr_e = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(lambda)"][lr][lmb], metric_key)))
            elam_lam_aucs.append(compute_auc(get_metric_curve(all_runs[env]["E(lambda)"][best_lr_e][lmb], metric_key).mean(axis=0)))

        ax.plot(lambdas, sarsa_lam_aucs, marker="o", markersize=6, color="#e6550d", linewidth=2.0, linestyle="--", label="SARSA(λ)")
        ax.plot(lambdas, elam_lam_aucs, marker="s", markersize=6, color="#2ca02c", linewidth=2.2, linestyle="-", label="E(λ)-Q")

        ax.set_title(f"{env_title}: λ-Sensitivity (AUC)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(r"Lookahead Parameter $\lambda$", fontsize=9.5)
        ax.set_ylabel("Normalized AUC", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.set_xticks(lambdas)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=8.0, loc="best")

    for extra_idx in range(n_envs, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    plt.tight_layout()
    sens_curves_base = os.path.join(diagnostics_dir, "lambda_sensitivity_curves")
    sens_curves_spec = os.path.join(diagnostics_dir, "diagnostic_2_sarsa_vs_elambda_spectrum")
    sens_curves_plot = os.path.join(plots_dir, "lambda_sensitivity_curves")
    plt.savefig(f"{sens_curves_base}.png", bbox_inches="tight")
    plt.savefig(f"{sens_curves_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{sens_curves_spec}.png", bbox_inches="tight")
    plt.savefig(f"{sens_curves_spec}.pdf", bbox_inches="tight")
    plt.savefig(f"{sens_curves_plot}.png", bbox_inches="tight")
    plt.savefig(f"{sens_curves_plot}.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved Lambda Sensitivity Curves to {sens_curves_base}.png, {sens_curves_spec}.png, and {sens_curves_plot}.png")

    # -------------------------------------------------------------
    # 4. CROSS-ENVIRONMENT HEATMAP TABLES (BENCHMARKS)
    # -------------------------------------------------------------
    env_row_labels = [ENV_TITLES.get(e, e) for e in envs]

    # Table 1: Q-PPO AUC Table
    df_auc_mean = pd.DataFrame(auc_means_dict, index=env_row_labels)
    df_auc_std = pd.DataFrame(auc_stds_dict, index=env_row_labels)
    render_heatmap_table(
        df_auc_mean,
        df_auc_std,
        title="Q-PPO AUC Performance Matrix Across Deterministic Domains",
        cbar_label="Relative AUC (Row Normalized: 0=Min, 1=Max)",
        output_base=os.path.join(benchmarks_dir, "main_comparison_auc_table"),
        cmap="Blues",
    )

    # Table 2: Q-PPO Final Return Table
    df_final_mean = pd.DataFrame(final_means_dict, index=env_row_labels)
    df_final_std = pd.DataFrame(final_stds_dict, index=env_row_labels)
    render_heatmap_table(
        df_final_mean,
        df_final_std,
        title="Q-PPO Final Episodic Return Matrix Across Deterministic Domains",
        cbar_label="Relative Return (Row Normalized: 0=Min, 1=Max)",
        output_base=os.path.join(benchmarks_dir, "main_comparison_final_return_table"),
        cmap="Blues",
    )

    # Summary CSV in benchmarks/ and plots/
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(benchmarks_dir, "main_comparison_summary.csv")
    csv_plots_path = os.path.join(plots_dir, "main_comparison_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_csv(csv_plots_path, index=False)
    print(f"Saved Summary CSV to {csv_path} and {csv_plots_path}")

    # -------------------------------------------------------------
    # 5. STRUCTURED TABLE BUNDLES (Markdown, LaTeX, and Vector PDF)
    # -------------------------------------------------------------
    # Bundle 1: Main Comparison
    export_table_bundle(
        folder_path=os.path.join(tables_dir, "main_comparison"),
        title="Comprehensive Q-Critic Comparison",
        subtitle="Normalized Area Under Curve (AUC) across Deterministic Domains (Mean ± Std)",
        col_names=alg_names,
        row_names=env_row_labels,
        mean_mat=df_auc_mean.values,
        std_mat=df_auc_std.values,
        cmap="Blues",
    )

    # Bundle 2: Diagnostic 1 (E(0)-Q vs E(lambda=0)-Q)
    col_names_d1 = ["E(0)-Q (Action Anchor)", "E(λ=0)-Q (1-Step Jump)"]
    mean_d1 = np.zeros((n_envs, len(col_names_d1)))
    std_d1 = np.zeros((n_envs, len(col_names_d1)))
    for i, env in enumerate(envs):
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(0)"][lr], metric_key)))
        e0_data = get_metric_curve(all_runs[env]["E(0)"][best_lr_e0], metric_key)
        mean_d1[i, 0] = float(np.mean(e0_data))
        std_d1[i, 0] = float(np.std(np.mean(e0_data, axis=1))) if e0_data.ndim > 1 else float(np.std(e0_data))

        best_lr_el0 = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(lambda)"][lr][0.0], metric_key)))
        el0_data = get_metric_curve(all_runs[env]["E(lambda)"][best_lr_el0][0.0], metric_key)
        mean_d1[i, 1] = float(np.mean(el0_data))
        std_d1[i, 1] = float(np.std(np.mean(el0_data, axis=1))) if el0_data.ndim > 1 else float(np.std(el0_data))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_1_e0_vs_elambda0"),
        title="Diagnostic 1: Pure Dirichlet E(0)-Q vs. 1-Step E(λ=0)-Q",
        subtitle="AUC Comparison of Action Anchor Dirichlet Formulation vs 1-Step Formulation",
        col_names=col_names_d1,
        row_names=env_row_labels,
        mean_mat=mean_d1,
        std_mat=std_d1,
        cmap="Purples",
    )

    # Bundle 3: Diagnostic 2 (SARSA vs E-Q across Lambda Spectrum)
    col_names_d2 = [f"SARSA(λ={l})" for l in lambdas] + [f"E(λ={l})-Q" for l in lambdas]
    mean_d2 = np.zeros((n_envs, len(col_names_d2)))
    std_d2 = np.zeros((n_envs, len(col_names_d2)))
    for i, env in enumerate(envs):
        for j, lmb in enumerate(lambdas):
            best_lr_s = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["SARSA(lambda)"][lr][lmb], metric_key)))
            s_data = get_metric_curve(all_runs[env]["SARSA(lambda)"][best_lr_s][lmb], metric_key)
            mean_d2[i, j] = float(np.mean(s_data))
            std_d2[i, j] = float(np.std(np.mean(s_data, axis=1))) if s_data.ndim > 1 else float(np.std(s_data))

            best_lr_e = max(critic_lrs, key=lambda lr: np.mean(get_metric_curve(all_runs[env]["E(lambda)"][lr][lmb], metric_key)))
            e_data = get_metric_curve(all_runs[env]["E(lambda)"][best_lr_e][lmb], metric_key)
            mean_d2[i, len(lambdas) + j] = float(np.mean(e_data))
            std_d2[i, len(lambdas) + j] = float(np.std(np.mean(e_data, axis=1))) if e_data.ndim > 1 else float(np.std(e_data))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_2_sarsa_vs_elambda_spectrum"),
        title="Diagnostic 2: SARSA(lambda) vs. E(lambda)-Q Lookahead Spectrum",
        subtitle=f"AUC Comparison across lambda in {lambdas}",
        col_names=col_names_d2,
        row_names=env_row_labels,
        mean_mat=mean_d2,
        std_mat=std_d2,
        cmap="Blues",
    )

    # -------------------------------------------------------------
    # 6. STORY 4: Clean SARSA(λ*) vs E_Q(λ*) Normal Tables & Curves (.md, .csv, .tex, .pdf)
    # -------------------------------------------------------------
    from scripts.gymnax.sweep_comprehensive_critic.sweep_comprehensive_critics import (
        render_analysis_plot_and_tables,
        select_best_lr_run,
    )

    q_s4_algs = ["SARSA(λ*)", "E_Q(λ*)"]
    q_s4_colors = {"SARSA(λ*)": "#e6550d", "E_Q(λ*)": "#2ca02c"}
    q_s4_styles = {"SARSA(λ*)": "--", "E_Q(λ*)": "-"}
    q_s4_res_auc, q_s4_res_final = {}, {}
    q_s4_ann_auc, q_s4_ann_final = {}, {}

    for r_idx, env in enumerate(envs):
        env_d = all_runs[env]

        # Best SARSA(lambda*) across (lr, lmb)
        sarsa_cands = []
        for lr, lmb_d in env_d.get("SARSA(lambda)", {}).items():
            for lmb in lmb_d:
                info = select_best_lr_run({lr: lmb_d[lmb]}, [lr], metric_key)
                info["legend_label"] = f"Best SARSA (λ={lmb})"
                info["short_ann"] = f"λ={lmb}"
                sarsa_cands.append(info)
        best_s_auc = max(sarsa_cands, key=lambda x: x["auc_mean"]) if sarsa_cands else select_best_lr_run({}, critic_lrs, metric_key)
        best_s_final = max(sarsa_cands, key=lambda x: x["final_mean"]) if sarsa_cands else best_s_auc

        # Best E_Q(lambda*) across (lr, lmb) + E(0)-Q
        eq_cands = []
        for lr, lmb_d in env_d.get("E(lambda)", {}).items():
            for lmb in lmb_d:
                info = select_best_lr_run({lr: lmb_d[lmb]}, [lr], metric_key)
                info["legend_label"] = f"Best E_Q (λ={lmb})"
                info["short_ann"] = f"λ={lmb}"
                eq_cands.append(info)
        if env_d.get("E(0)"):
            info = select_best_lr_run(env_d["E(0)"], critic_lrs, metric_key)
            info["legend_label"] = "Best E_Q (E(0))"
            info["short_ann"] = "E(0)"
            eq_cands.append(info)
        best_e_auc = max(eq_cands, key=lambda x: x["auc_mean"]) if eq_cands else select_best_lr_run({}, critic_lrs, metric_key)
        best_e_final = max(eq_cands, key=lambda x: x["final_mean"]) if eq_cands else best_e_auc

        q_s4_res_auc[env] = {"SARSA(λ*)": best_s_auc, "E_Q(λ*)": best_e_auc}
        q_s4_res_final[env] = {"SARSA(λ*)": best_s_final, "E_Q(λ*)": best_e_final}
        q_s4_ann_auc[(r_idx, 0)] = best_s_auc.get("short_ann", "")
        q_s4_ann_auc[(r_idx, 1)] = best_e_auc.get("short_ann", "")
        q_s4_ann_final[(r_idx, 0)] = best_s_final.get("short_ann", "")
        q_s4_ann_final[(r_idx, 1)] = best_e_final.get("short_ann", "")

    render_analysis_plot_and_tables(
        envs=envs,
        alg_names=q_s4_algs,
        per_env_results=q_s4_res_auc,
        colors=q_s4_colors,
        linestyles=q_s4_styles,
        batch_size=batch_size,
        plot_title_suffix=" — SARSA(λ*) vs E_Q(λ*)",
        plot_output_base=os.path.join(plots_dir, "story4_q_sarsa_vs_e_q_learning_curves"),
        table_output_base=os.path.join(tables_dir, "story4_q_sarsa_vs_e_q"),
        table_title_prefix="Story 4: Deterministic Q-Critic — Best SARSA(λ*) vs Best E_Q(λ*) per Environment",
        cmap="Blues",
        cell_annotations={"auc": q_s4_ann_auc, "final": q_s4_ann_final},
        per_env_results_final=q_s4_res_final,
        y_label="Episodic Return",
    )

    print("\nAll Q-PPO cross-environment figures, plots, and tables generated successfully!")



def main():
    parser = argparse.ArgumentParser(description="Comprehensive Q-PPO Critic Comparison Sweep")
    parser.add_argument("--envs", nargs="+", default=["CartPole-v1"], help="List of deterministic discrete environments")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.4, 0.8, 0.95], help="Lambdas to sweep")
    parser.add_argument("--critic-lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3], help="Critic learning rates")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps (default: auto per env: 1M for MountainCar, 10M for MinAtar, 300k for small envs)")
    parser.add_argument("--num-envs", type=int, default=32, help="Parallel environments (rollouts)")
    parser.add_argument("--num-steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--minibatch-size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n-seeds", type=int, default=3, help="Random seeds per configuration")
    parser.add_argument("--master-seed", type=int, default=42, help="Master random seed")
    parser.add_argument("--out-dir", type=str, default="results/sweep_comprehensive_q_critic", help="Output directory")
    parser.add_argument("--load-existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")
    parser.add_argument("--metric-key", type=str, default="returned_episode_returns", help="Metric key to analyze (default: returned_episode_returns)")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

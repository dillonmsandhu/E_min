#!/usr/bin/env python3
"""
sweep_continuous_q_critics.py

Comprehensive Continuous Q-Critic (DDPG) Comparison Sweep & Analysis Suite for Gymnax:
Applies to continuous control environments with deterministic physics:
1. E(0) Family:
   - Naive (no v_aux, symmetric Dirichlet (e_i - e_j)^2 -> mathematically unbiased for Q-critic!)
   - v_aux (surgical auxiliary Q_aux network)
   - Forward-Only (no v_aux, semi-gradient (e_i - sg(e_j))^2)
2. E(lambda=0.8) Family:
   - Naive (no v_aux, geometric lookahead jumps)
   - v_aux (surgical auxiliary Q_aux network)
   - Forward-Only (no v_aux, semi-gradient)
3. Expected SARSA(lambda) Grid:
   - SARSA(0.0) / 1-step online TD(0)
   - SARSA(0.4)
   - SARSA(0.8)
   - SARSA(0.95)
   - MC (lambda=1.0 rollout return regression)

Automatically generates 5 publication-grade analyses (Plots in .pdf/.png and Tables in .md/.tex/.pdf/.png):
- Table 1 & Figure 1: E(0) Variants (Naive vs v_aux vs Forward-Only)
- Table 2 & Figure 2: E(lambda=0.8) Variants (Naive vs v_aux vs Forward-Only)
- Table 3 & Figure 3: Naive Showdown (E(0) Naive [Unbiased] vs E(0.8) Naive [Multi-step Jump])
- Table 4 & Figure 4: Expected SARSA(lambda) Grid (Effect of lambda on TD)
- Table 5 & Figure 5: Master Championship (Overall Best E vs Overall Best TD)
"""

from __future__ import annotations
import os
import sys
import math
import argparse
import time
import pickle
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

from gymnax_rl.core.ddpg_config import config as gymnax_ddpg_config
from gymnax_rl.algos.ddpg import make_train as make_ddpg_train

ENV_TITLES = {
    "Pendulum-v1": "Pendulum-v1",
    "MountainCarContinuous-v0": "Mountain Car Continuous (Sparse)",
    "MountainCarDenseContinuous-v0": "Mountain Car Continuous (Dense)",
    "PointRobot-misc": "PointRobot (Continuous)",
}

DEFAULT_ENVS = ["Pendulum-v1", "MountainCarContinuous-v0", "MountainCarDenseContinuous-v0", "PointRobot-misc"]
DEFAULT_CRITIC_LRS = [3e-4, 1e-3, 3e-3]
DEFAULT_SARSA_LAMBDAS = [0.0, 0.4, 0.8, 0.95, 1.0]
REAL_WIN_REL_TOL = 0.01  # 1% of environment return magnitude


def get_env_default_timesteps(env_name: str, fallback_timesteps: int | None = None) -> int:
    """Returns environment-specific timesteps."""
    if fallback_timesteps is not None:
        return fallback_timesteps
    if "MountainCar" in env_name:
        return 500_000
    else:
        return 300_000


def compute_auc(returns_curve):
    """Computes normalized Area Under Curve (AUC) across update timesteps."""
    return float(np.mean(returns_curve))


def get_metric_curve(run_dict, metric_key="returned_episode_returns"):
    """Safely extracts metric curve of shape (n_seeds, n_updates)."""
    if metric_key in run_dict:
        return np.array(run_dict[metric_key])
    elif "returned_discounted_episode_returns" in run_dict:
        return np.array(run_dict["returned_discounted_episode_returns"])
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
    best_lr = max(available_lrs, key=lambda lr: compute_auc(get_metric_curve(lr_dict[lr], metric_key).mean(axis=0)))
    data = get_metric_curve(lr_dict[best_lr], metric_key)
    mean_curve = data.mean(axis=0)
    std_curve = data.std(axis=0)
    seed_aucs = data.mean(axis=-1)
    return {
        "lr": best_lr,
        "data": data,
        "curve": mean_curve,
        "std": std_curve,
        "auc_mean": float(seed_aucs.mean()),
        "auc_std": float(seed_aucs.std()),
        "final_mean": float(mean_curve[-1]),
        "final_std": float(std_curve[-1]),
    }


def run_experiment(config, seed_keys):
    """JIT-compiles and runs vmapped training across random seeds."""
    train_fn = make_ddpg_train(config)
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
        "returned_episode_lengths": np.array(metrics.get("returned_episode_lengths", np.zeros_like(metrics["returned_episode_returns"]))),
        "critic_loss": np.array(
            metrics["value_loss"] if "value_loss" in metrics else metrics.get("val_loss", 0.0)
        ),
        "actor_loss": np.array(metrics["actor_loss"]),
        "elapsed_time": elapsed,
    }
    for k_m, v_m in metrics.items():
        if k_m not in res:
            res[k_m] = np.array(v_m)
    return res


# ==============================================================================
# Table & Plot Formatting Utilities
# ==============================================================================

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


def _row_real_win_tol(vals, rel_tol=REAL_WIN_REL_TOL):
    vals = np.asarray(vals, dtype=float)
    valid = vals[~np.isnan(vals)]
    if len(valid) == 0:
        return 1e-4
    scale = max(float(np.max(np.abs(valid))), 1e-8)
    return max(rel_tol * scale, 1e-6)


def _best_col_mask(vals, rel_tol=REAL_WIN_REL_TOL):
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return np.zeros(len(vals), dtype=bool)
    mx = np.nanmax(vals)
    tol = _row_real_win_tol(vals, rel_tol=rel_tol)
    return valid & (vals >= mx - tol)


def _max_int_mask(vals):
    vals = np.asarray(vals, dtype=float)
    mx = np.max(vals)
    return vals >= mx - 1e-9


def _compute_wtl_stats(mat_mean, rel_tol=REAL_WIN_REL_TOL):
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


def render_clean_table(df_mean, df_std, title, output_base, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL):
    """Renders academic publication table saving as PDF and PNG."""
    cols = list(df_mean.columns)
    rows = list(df_mean.index)
    mat_mean = df_mean.values.astype(float)
    mat_std = df_std.values.astype(float) if df_std is not None else np.zeros_like(mat_mean)
    n_rows, n_cols = mat_mean.shape

    col_labels = ["Environment"] + [str(c) for c in cols]
    table_data = []
    best_masks = []

    for r_idx, env_label in enumerate(rows):
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

        top_w, top_t, top_l, pw_w, pw_t, pw_l = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)
        top_scores = top_w + 0.5 * top_t
        best_masks.append(_max_int_mask(top_scores))
        top_cells = ["Top-1 (W / T / L, 1%)"] + [f"{top_w[c]}W / {top_t[c]}T / {top_l[c]}L" for c in range(n_cols)]
        table_data.append(top_cells)

        if n_cols > 2:
            pw_scores = pw_w - pw_l
            best_masks.append(_max_int_mask(pw_scores))
            pw_cells = ["Pairwise (W / T / L)"] + [
                f"{pw_w[c]}W / {pw_t[c]}T / {pw_l[c]}L" for c in range(n_cols)
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

    fig_w = max(8.5, total_chars * 0.11)
    fig_h = max(3.2, (total_rows + 1) * 0.50 + 0.8)

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

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.75)
        if r == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a", fontsize=10.0)
            cell.set_edgecolor("#64748b")
            cell.set_linewidth(1.1)
            if c == 0:
                cell.set_text_props(ha="left")
        else:
            data_r = r - 1
            is_summary = data_r >= n_rows
            bg = "#f1f5f9" if is_summary else ("#ffffff" if (data_r % 2 == 0) else "#f8fafc")
            if c == 0:
                cell.set_facecolor(bg)
                cell.set_text_props(weight="bold", color="#0f172a", fontsize=9.5, ha="left")
            else:
                val_c = c - 1
                is_best = bool(best_masks[data_r][val_c])
                if is_best:
                    cell.set_facecolor("#eff6ff")
                    cell.set_text_props(weight="bold", color="#0f172a", fontsize=9.5)
                else:
                    cell.set_facecolor(bg)
                    cell.set_text_props(weight="normal", color="#1e293b", fontsize=9.5)

    ax.set_title(title, fontsize=12.0, fontweight="bold", color="#0f172a", pad=12, loc="center")
    plt.tight_layout()
    plt.savefig(f"{output_base}.png", bbox_inches="tight", facecolor="white")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight", facecolor="white")
    plt.close()


def save_table_bundle(df_mean, df_std, title, output_base, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL):
    """Saves table in Markdown (.md), CSV (.csv), LaTeX (.tex), PDF (.pdf), and PNG (.png)."""
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    cols = list(df_mean.columns)
    rows = list(df_mean.index)
    mat_mean = df_mean.values.astype(float)
    mat_std = df_std.values.astype(float) if df_std is not None else np.zeros_like(mat_mean)
    n_cols = len(cols)

    # 1. Markdown
    md_lines = [f"# {title}\n", f"*Real-win threshold: >= {rel_tol*100:.0f}% of environment return scale.*\n"]
    header = "| Environment | " + " | ".join(cols) + " |"
    sep = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |"
    md_lines.extend([header, sep])

    for r_idx, env_label in enumerate(rows):
        mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
        cells = [str(env_label)]
        for c_idx in range(n_cols):
            val_str = _format_val_std(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx])
            if mask[c_idx]:
                val_str = f"**{val_str}**"
            ann = cell_annotations.get((r_idx, c_idx), "") if cell_annotations else ""
            if ann:
                val_str += f" ({ann})"
            cells.append(val_str)
        md_lines.append("| " + " | ".join(cells) + " |")

    if len(rows) > 1:
        col_means = np.nanmean(mat_mean, axis=0)
        col_stds = np.nanmean(mat_std, axis=0)
        mean_mask = _best_col_mask(col_means, rel_tol=rel_tol)
        mean_cells = ["**Mean**"] + [
            f"**{_format_val_std(col_means[c], col_stds[c])}**" if mean_mask[c] else _format_val_std(col_means[c], col_stds[c])
            for c in range(n_cols)
        ]
        md_lines.append("| " + " | ".join(mean_cells) + " |")

        top_w, top_t, top_l, _, _, _ = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)
        top_mask = _max_int_mask(top_w + 0.5 * top_t)
        top_cells = ["**Top-1 (W / T / L)**"] + [
            f"**{top_w[c]}W / {top_t[c]}T / {top_l[c]}L**" if top_mask[c] else f"{top_w[c]}W / {top_t[c]}T / {top_l[c]}L"
            for c in range(n_cols)
        ]
        md_lines.append("| " + " | ".join(top_cells) + " |")

    with open(f"{output_base}.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # 2. CSV
    df_mean.to_csv(f"{output_base}.csv")

    # 3. LaTeX
    tex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        f"\\caption{{{title}}}",
        f"\\label{{tab:{os.path.basename(output_base)}}}",
        "\\begin{tabular}{l" + "c" * n_cols + "}",
        "\\toprule",
        "Environment & " + " & ".join([str(c).replace("_", "\\_") for c in cols]) + " \\\\",
        "\\midrule",
    ]
    for r_idx, env_label in enumerate(rows):
        mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
        cells = [str(env_label).replace("_", "\\_")]
        for c_idx in range(n_cols):
            cells.append(_format_val_std_tex(mat_mean[r_idx, c_idx], mat_std[r_idx, c_idx], is_best=mask[c_idx]))
        tex_lines.append(" & ".join(cells) + " \\\\")

    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    with open(f"{output_base}.tex", "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    # 4. Render clean table graphic (.pdf and .png)
    render_clean_table(df_mean, df_std, title, output_base, cell_annotations=cell_annotations, rel_tol=rel_tol)


# ==============================================================================
# Main Sweep Execution
# ==============================================================================

def run_sweep(args):
    envs = args.envs
    critic_lrs = args.critic_lrs
    actor_lr = args.actor_lr
    sarsa_lambdas = args.sarsa_lambdas
    num_envs = args.num_envs
    num_steps = args.num_steps
    minibatch_size = args.minibatch_size
    n_seeds = args.n_seeds
    master_seed = args.master_seed
    out_dir = args.out_dir
    metric_key = getattr(args, "metric_key", "returned_episode_returns")
    exploration_noise = getattr(args, "exploration_noise", 0.1)

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
            all_runs = pickle.load(f).get("runs", {})

    batch_size = num_envs * num_steps
    base_timesteps = args.total_timesteps or 300000
    num_updates = max(1, base_timesteps // batch_size)
    timesteps = np.arange(1, num_updates + 1) * batch_size

    print("=================================================================")
    print("STARTING CONTINUOUS Q-CRITIC (DDPG) COMPREHENSIVE SWEEP")
    print(f"Envs: {envs}")
    print(f"Rollouts: {num_envs} envs x {num_steps} steps (batch {batch_size})")
    print(f"Critic LRs: {critic_lrs} | SARSA Lambdas: {sarsa_lambdas} | Seeds: {n_seeds}")
    print(f"Exploration Noise: {exploration_noise}")
    print("=================================================================\n")

    for env_name in envs:
        env_timesteps = get_env_default_timesteps(env_name, args.total_timesteps)
        if env_name not in all_runs:
            all_runs[env_name] = {}

        env_dict = all_runs[env_name]
        env_dict.setdefault("E(0)_naive", {})
        env_dict.setdefault("E(0)_v_aux", {})
        env_dict.setdefault("E(0)_fwd", {})
        env_dict.setdefault("E(0.8)_naive", {})
        env_dict.setdefault("E(0.8)_v_aux", {})
        env_dict.setdefault("E(0.8)_fwd", {})
        env_dict.setdefault("SARSA", {lam: {} for lam in sarsa_lambdas})

        for lam in sarsa_lambdas:
            env_dict["SARSA"].setdefault(lam, {})

        base_cfg = {
            **gymnax_ddpg_config,
            "ENV_NAME": env_name,
            "LR": actor_lr,
            "ACTOR_LR": actor_lr,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": env_timesteps,
            "MINIBATCH_SIZE": minibatch_size,
            "ACTOR_MINIBATCH_SIZE": minibatch_size,
            "CRITIC_MINIBATCH_SIZE": minibatch_size,
            "EXPLORATION_NOISE": exploration_noise,
        }

        print(f"\n>>> Running Environment: {env_name} ({env_timesteps:,} steps) <<<")

        # 1. E(0) Variants
        for var_name, critic_type, e_var in [
            ("E(0)_naive", "e_0", "naive"),
            ("E(0)_v_aux", "e_0", "aux"),
            ("E(0)_fwd", "e_0", "fwd"),
        ]:
            for lr in critic_lrs:
                if lr not in env_dict[var_name]:
                    print(f"[{var_name}] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": 0.1 * lr,
                        "CRITIC_TYPE": critic_type,
                        "E_VARIANT": e_var,
                        "RETURN_LAMBDA": 1.0,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_name][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 2. E(lambda=0.8) Variants
        for var_name, critic_type, e_var in [
            ("E(0.8)_naive", "e_lambda", "naive"),
            ("E(0.8)_v_aux", "e_lambda", "aux"),
            ("E(0.8)_fwd", "e_lambda", "fwd"),
        ]:
            for lr in critic_lrs:
                if lr not in env_dict[var_name]:
                    print(f"[{var_name}] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": 0.1 * lr,
                        "CRITIC_TYPE": critic_type,
                        "E_VARIANT": e_var,
                        "E_LAMBDA": 0.8,
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": 1.0,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_name][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 3. Expected SARSA(lambda) Grid
        for lam in sarsa_lambdas:
            for lr in critic_lrs:
                if lr not in env_dict["SARSA"][lam]:
                    c_type = "td_0" if lam == 0.0 else "sarsa"
                    print(f"[SARSA(lambda={lam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": 0.1 * lr,
                        "CRITIC_TYPE": c_type,
                        "VALUE_LAMBDA": lam,
                        "TD_LAMBDA": 0.0,
                        "E_VARIANT": "naive",
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["SARSA"][lam][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

    # Save all raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate all analyses
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, sarsa_lambdas, metric_key, batch_size)


# ==============================================================================
# Automated Cross-Algorithm Diagnostics and Tables
# ==============================================================================

def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, sarsa_lambdas, metric_key, batch_size):
    """Generates the 5 cross-algorithm tables and 5 comparative learning curve figures."""
    print("\nGenerating comprehensive tables and comparative learning curves...")
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    n_envs = len(envs)
    n_cols_plot = min(4, n_envs)
    n_rows_plot = int(math.ceil(n_envs / n_cols_plot))

    # -------------------------------------------------------------
    # 1. ANALYSIS 1: E(0) Variants (Naive vs v_aux vs Forward-Only)
    # -------------------------------------------------------------
    e0_cols = ["E(0) Naive (Unbiased)", "E(0) v_aux", "E(0) Forward-Only"]
    e0_keys = ["E(0)_naive", "E(0)_v_aux", "E(0)_fwd"]
    e0_means, e0_stds, e0_annots = [], [], {}

    fig1, axes1 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    colors_e0 = ["#2563eb", "#059669", "#d97706"]

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes1[r][c]
        row_means, row_stds = [], []
        for c_idx, k in enumerate(e0_keys):
            best = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key)
            row_means.append(best["auc_mean"])
            row_stds.append(best["auc_std"])
            e0_annots[(idx, c_idx)] = f"lr={best['lr']}"
            t = np.arange(1, len(best["curve"]) + 1) * batch_size
            ax.plot(t, best["curve"], label=e0_cols[c_idx], color=colors_e0[c_idx], lw=2.0)
            ax.fill_between(t, best["curve"] - best["std"], best["curve"] + best["std"], color=colors_e0[c_idx], alpha=0.15)
        e0_means.append(row_means)
        e0_stds.append(row_stds)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    # Hide extra subplots
    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes1[r][c].axis("off")

    fig1.suptitle("Analysis 1: E(0) Variants Learning Curves", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis1_e0_variants_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis1_e0_variants_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_e0_mean = pd.DataFrame(e0_means, index=envs, columns=e0_cols)
    df_e0_std = pd.DataFrame(e0_stds, index=envs, columns=e0_cols)
    save_table_bundle(
        df_e0_mean, df_e0_std,
        "Table 1: E(0) Q-Critic Variants Comparison (AUC Return)",
        os.path.join(tables_dir, "table1_e0_variants"),
        cell_annotations=e0_annots,
    )

    # -------------------------------------------------------------
    # 2. ANALYSIS 2: E(lambda=0.8) Variants (Naive vs v_aux vs Forward-Only)
    # -------------------------------------------------------------
    elam_cols = ["E(0.8) Naive", "E(0.8) v_aux", "E(0.8) Forward-Only"]
    elam_keys = ["E(0.8)_naive", "E(0.8)_v_aux", "E(0.8)_fwd"]
    elam_means, elam_stds, elam_annots = [], [], {}

    fig2, axes2 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    colors_elam = ["#4f46e5", "#0d9488", "#e11d48"]

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes2[r][c]
        row_means, row_stds = [], []
        for c_idx, k in enumerate(elam_keys):
            best = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key)
            row_means.append(best["auc_mean"])
            row_stds.append(best["auc_std"])
            elam_annots[(idx, c_idx)] = f"lr={best['lr']}"
            t = np.arange(1, len(best["curve"]) + 1) * batch_size
            ax.plot(t, best["curve"], label=elam_cols[c_idx], color=colors_elam[c_idx], lw=2.0)
            ax.fill_between(t, best["curve"] - best["std"], best["curve"] + best["std"], color=colors_elam[c_idx], alpha=0.15)
        elam_means.append(row_means)
        elam_stds.append(row_stds)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes2[r][c].axis("off")

    fig2.suptitle("Analysis 2: E(lambda=0.8) Variants Learning Curves", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis2_elambda_variants_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis2_elambda_variants_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_elam_mean = pd.DataFrame(elam_means, index=envs, columns=elam_cols)
    df_elam_std = pd.DataFrame(elam_stds, index=envs, columns=elam_cols)
    save_table_bundle(
        df_elam_mean, df_elam_std,
        "Table 2: E(lambda=0.8) Q-Critic Variants Comparison (AUC Return)",
        os.path.join(tables_dir, "table2_elambda_variants"),
        cell_annotations=elam_annots,
    )

    # -------------------------------------------------------------
    # 3. ANALYSIS 3: Naive Sampling Showdown (E(0) Naive vs E(0.8) Naive)
    # -------------------------------------------------------------
    naive_cols = ["E(0) Naive (Unbiased 1-Step)", "E(0.8) Naive (Multi-Step Jump)"]
    naive_keys = ["E(0)_naive", "E(0.8)_naive"]
    naive_means, naive_stds, naive_annots = [], [], {}

    fig3, axes3 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    colors_naive = ["#2563eb", "#7c3aed"]

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes3[r][c]
        row_means, row_stds = [], []
        for c_idx, k in enumerate(naive_keys):
            best = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key)
            row_means.append(best["auc_mean"])
            row_stds.append(best["auc_std"])
            naive_annots[(idx, c_idx)] = f"lr={best['lr']}"
            t = np.arange(1, len(best["curve"]) + 1) * batch_size
            ax.plot(t, best["curve"], label=naive_cols[c_idx], color=colors_naive[c_idx], lw=2.0)
            ax.fill_between(t, best["curve"] - best["std"], best["curve"] + best["std"], color=colors_naive[c_idx], alpha=0.15)
        naive_means.append(row_means)
        naive_stds.append(row_stds)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes3[r][c].axis("off")

    fig3.suptitle("Analysis 3: Naive Sampling Showdown (E(0) vs E(0.8))", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis3_naive_showdown_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis3_naive_showdown_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_naive_mean = pd.DataFrame(naive_means, index=envs, columns=naive_cols)
    df_naive_std = pd.DataFrame(naive_stds, index=envs, columns=naive_cols)
    save_table_bundle(
        df_naive_mean, df_naive_std,
        "Table 3: Naive Sampling Head-to-Head (E(0) Unbiased vs E(0.8) Jump)",
        os.path.join(tables_dir, "table3_naive_showdown_e0_vs_elambda"),
        cell_annotations=naive_annots,
    )

    # -------------------------------------------------------------
    # 4. ANALYSIS 4: Expected SARSA(lambda) Grid (Effect of Lambda on TD)
    # -------------------------------------------------------------
    sarsa_cols = [f"SARSA(λ={lam})" if lam < 1.0 else "MC (λ=1.0)" for lam in sarsa_lambdas]
    sarsa_means, sarsa_stds, sarsa_annots = [], [], {}

    fig4, axes4 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes4[r][c]
        row_means, row_stds = [], []
        for c_idx, lam in enumerate(sarsa_lambdas):
            best = select_best_lr_run(all_runs[env].get("SARSA", {}).get(lam, {}), critic_lrs, metric_key)
            row_means.append(best["auc_mean"])
            row_stds.append(best["auc_std"])
            sarsa_annots[(idx, c_idx)] = f"lr={best['lr']}"
        sarsa_means.append(row_means)
        sarsa_stds.append(row_stds)

        # Plot Lambda Sensitivity Curve for this env
        ax.plot(sarsa_lambdas, row_means, marker="o", color="#dc2626", lw=2.2, label="Expected SARSA(λ)")
        ax.fill_between(sarsa_lambdas, np.array(row_means) - np.array(row_stds), np.array(row_means) + np.array(row_stds), color="#dc2626", alpha=0.15)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Trace Parameter λ")
        ax.set_ylabel("AUC Return")
        ax.grid(True, alpha=0.3)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes4[r][c].axis("off")

    fig4.suptitle("Analysis 4: Expected SARSA(lambda) Lambda Sensitivity Curves", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis4_sarsa_lambda_sensitivity_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis4_sarsa_lambda_sensitivity_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_sarsa_mean = pd.DataFrame(sarsa_means, index=envs, columns=sarsa_cols)
    df_sarsa_std = pd.DataFrame(sarsa_stds, index=envs, columns=sarsa_cols)
    save_table_bundle(
        df_sarsa_mean, df_sarsa_std,
        "Table 4: Expected SARSA(lambda) Sensitivity Grid (AUC Return)",
        os.path.join(tables_dir, "table4_sarsa_lambda_grid"),
        cell_annotations=sarsa_annots,
    )

    # -------------------------------------------------------------
    # 5. ANALYSIS 5: Master Championship (Best E vs Best TD)
    # -------------------------------------------------------------
    champ_cols = ["Best TD / SARSA", "Best E-Minimization"]
    champ_means, champ_stds, champ_annots = [], [], {}

    fig5, axes5 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes5[r][c]

        # 1. Find Best TD run
        best_td = None
        for lam in sarsa_lambdas:
            cand = select_best_lr_run(all_runs[env].get("SARSA", {}).get(lam, {}), critic_lrs, metric_key)
            if best_td is None or cand["auc_mean"] > best_td["auc_mean"]:
                best_td = {**cand, "name": f"SARSA(λ={lam})"}

        # 2. Find Best E run
        best_e = None
        for e_key, clean_name in [
            ("E(0)_naive", "E(0) Naive"),
            ("E(0)_v_aux", "E(0) v_aux"),
            ("E(0)_fwd", "E(0) Fwd"),
            ("E(0.8)_naive", "E(0.8) Naive"),
            ("E(0.8)_v_aux", "E(0.8) v_aux"),
            ("E(0.8)_fwd", "E(0.8) Fwd"),
        ]:
            cand = select_best_lr_run(all_runs[env].get(e_key, {}), critic_lrs, metric_key)
            if best_e is None or cand["auc_mean"] > best_e["auc_mean"]:
                best_e = {**cand, "name": clean_name}

        champ_means.append([best_td["auc_mean"], best_e["auc_mean"]])
        champ_stds.append([best_td["auc_std"], best_e["auc_std"]])
        champ_annots[(idx, 0)] = f"{best_td['name']}, lr={best_td['lr']}"
        champ_annots[(idx, 1)] = f"{best_e['name']}, lr={best_e['lr']}"

        t = np.arange(1, len(best_td["curve"]) + 1) * batch_size
        ax.plot(t, best_td["curve"], label=f"Best TD ({best_td['name']})", color="#dc2626", lw=2.2)
        ax.fill_between(t, best_td["curve"] - best_td["std"], best_td["curve"] + best_td["std"], color="#dc2626", alpha=0.15)

        ax.plot(t, best_e["curve"], label=f"Best E ({best_e['name']})", color="#2563eb", lw=2.2)
        ax.fill_between(t, best_e["curve"] - best_e["std"], best_e["curve"] + best_e["std"], color="#2563eb", alpha=0.15)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes5[r][c].axis("off")

    fig5.suptitle("Analysis 5: Master Championship (Best TD vs Best E) Learning Curves", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis5_best_e_vs_best_td_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis5_best_e_vs_best_td_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_champ_mean = pd.DataFrame(champ_means, index=envs, columns=champ_cols)
    df_champ_std = pd.DataFrame(champ_stds, index=envs, columns=champ_cols)
    save_table_bundle(
        df_champ_mean, df_champ_std,
        "Table 5: Master Championship (Overall Best TD vs Overall Best E)",
        os.path.join(tables_dir, "table5_best_e_vs_best_td_championship"),
        cell_annotations=champ_annots,
    )

    print("\nAll 5 Cross-Algorithm Analyses, Plots, and Tables generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run Continuous Q-Critic (DDPG) Comprehensive Sweep on Gymnax")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, help="Gymnax Environment Names")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS, help="Critic learning rates")
    parser.add_argument("--actor_lr", type=float, default=gymnax_ddpg_config["ACTOR_LR"], help="Fixed actor learning rate")
    parser.add_argument("--sarsa_lambdas", nargs="+", type=float, default=DEFAULT_SARSA_LAMBDAS, help="SARSA lambdas to sweep")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps per env")
    parser.add_argument("--num_envs", type=int, default=gymnax_ddpg_config["NUM_ENVS"], help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=gymnax_ddpg_config["NUM_STEPS"], help="Rollout length")
    parser.add_argument("--minibatch_size", type=int, default=gymnax_ddpg_config["MINIBATCH_SIZE"], help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--exploration_noise", type=float, default=gymnax_ddpg_config["EXPLORATION_NOISE"], help="Rollout exploration noise std")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")
    parser.add_argument("--out_dir", type=str, default="results/sweep_continuous_q_critic_gymnax", help="Output directory")
    parser.add_argument("--load_existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

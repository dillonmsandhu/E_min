#!/usr/bin/env python3
"""
sweep_double_sample_critics.py

Comprehensive Double-Sampling Critic Comparison Sweep & Automated Publication Analysis Suite
for Gymnax (Discrete, Continuous & MinAtar).

Evaluates 4 critic algorithms in the double-sampling regime:
1. Fitted (Standard PPO baseline: 1-step double-sampled rollout buffer, GAE regression MSE)
2. MSBE (Baird double-sampled Mean Squared Bellman Error: 0.5 * (sg(delta_1)*delta_2 + sg(delta_2)*delta_1))
3. E-Min Inner (Double-sampled bilinear energy minimization: 0.5 * <e, bar_delta>_D without v_aux)
4. MSPBE (Nonlinear Mean Squared Projected Bellman Error with CG projection & exact Hessian-vector products)

Evaluated across 16 Gymnax environments:
- MinAtar (10M steps): Asterix-MinAtar, Breakout-MinAtar, Freeway-MinAtar, SpaceInvaders-MinAtar
- MountainCar (1M steps): MountainCar-v0, MountainCarContinuous-v0
- Classic Control & BSuite & Misc (300k steps): CartPole-v1, Pendulum-v1, Acrobot-v1,
  DeepSea-bsuite, DiscountingChain-bsuite, FourRooms-misc, PointRobot-misc, Reacher-misc,
  Swimmer-misc, Pong-misc

Automatically generates 4 publication-grade analyses (Figures in .pdf/.png and Tables in .md/.tex/.pdf/.png):
- Table 1 & Figure 1: Double Sampling Critic Comparison (Learning Curves & Final Return)
- Table 2 & Figure 2: Bellman Error Diagnostics (Unbiased Diagnostic MSBE E[delta_1 * delta_2] Reduction)
- Table 3 & Figure 3: Master Championship & Normalized Performance Profiles (Relative to Fitted Baseline)
- Table 4 & Figure 4: Critic Learning Rate Sensitivity (Hyperparameter Robustness)
"""

from __future__ import annotations
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

from gymnax_rl.core.gymnax_config import config as gymnax_ppo_config
from gymnax_rl.algos.ppo_double_sample import make_train as make_double_sample_train

ENV_TITLES = {
    # Classic Control
    "CartPole-v1": "CartPole-v1",
    "Pendulum-v1": "Pendulum-v1",
    "Acrobot-v1": "Acrobot-v1",
    "MountainCar-v0": "Mountain Car (Sparse)",
    "MountainCarContinuous-v0": "Mountain Car (Continuous)",
    # MinAtar (10M)
    "Asterix-MinAtar": "Asterix (MinAtar)",
    "Breakout-MinAtar": "Breakout (MinAtar)",
    "Freeway-MinAtar": "Freeway (MinAtar)",
    "SpaceInvaders-MinAtar": "Space Invaders (MinAtar)",
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

DEFAULT_ENVS = [
    # MinAtar (10M)
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
    # MountainCar (1M)
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    # Classic Control
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    # BSuite
    "DeepSea-bsuite",
    "DiscountingChain-bsuite",
    # Misc / Navigation / Continuous
    "FourRooms-misc",
    "PointRobot-misc",
    "Reacher-misc",
    "Swimmer-misc",
    "Pong-misc",
]

CRITIC_VARIANTS = [
    ("fitted", "Fitted Value (GAE)"),
    ("msbe", "MSBE (Baird Double)"),
    ("e_inner", "E-Min (Bilinear Inner)"),
    ("mspbe", "MSPBE (Tangent CG + HVP)"),
]

DEFAULT_CRITIC_LRS = [3e-4, 1e-3, 3e-3]
REAL_WIN_REL_TOL = 0.01  # 1% of environment return magnitude


def get_env_default_timesteps(env_name: str, fallback_timesteps: int | None = None) -> int:
    """Returns environment-specific timesteps: 10M for MinAtar, 1M for MountainCar, 300k for others."""
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
    raise KeyError(f"Metric {metric_key} not found in run data keys: {list(run_dict.keys())}")


def select_best_lr_run(lr_dict, critic_lrs, metric_key):
    """Selects the run with the highest mean AUC across available critic_lrs."""
    if not lr_dict:
        return {
            "lr": "N/A",
            "curve": np.zeros(1),
            "std": np.zeros(1),
            "final_mean": np.nan,
            "final_std": np.nan,
            "auc_mean": np.nan,
            "auc_std": np.nan,
            "all_runs": {},
        }

    best_lr = None
    best_auc = -np.inf
    best_curve = None
    best_std = None
    best_final_mean = np.nan
    best_final_std = np.nan
    best_auc_std = np.nan

    for lr in critic_lrs:
        if lr not in lr_dict:
            continue
        run_data = lr_dict[lr]
        raw_curve = get_metric_curve(run_data, metric_key)
        # raw_curve shape: (n_seeds, n_updates)
        mean_curve = np.mean(raw_curve, axis=0)
        std_curve = np.std(raw_curve, axis=0)
        sem_curve = std_curve / np.sqrt(max(1, raw_curve.shape[0]))

        auc_per_seed = np.mean(raw_curve, axis=1)
        mean_auc = float(np.mean(auc_per_seed))

        if mean_auc > best_auc or best_lr is None:
            best_auc = mean_auc
            best_auc_std = float(np.std(auc_per_seed))
            best_lr = lr
            best_curve = mean_curve
            best_std = sem_curve
            best_final_mean = float(mean_curve[-1])
            best_final_std = float(np.std(raw_curve[:, -1]))

    return {
        "lr": best_lr,
        "curve": best_curve,
        "std": best_std,
        "final_mean": best_final_mean,
        "final_std": best_final_std,
        "auc_mean": best_auc,
        "auc_std": best_auc_std,
        "all_runs": lr_dict,
    }


def run_experiment(config, seed_keys):
    """JIT-compiles and runs vmapped training across random seeds."""
    train_fn = make_double_sample_train(config)
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
        "diagnostic_msbe": np.array(metrics.get("diagnostic_msbe", np.zeros_like(metrics["returned_episode_returns"]))),
        "val_loss": np.array(metrics.get("val_loss", np.zeros_like(metrics["returned_episode_returns"]))),
        "actor_loss": np.array(metrics.get("actor_loss", np.zeros_like(metrics["returned_episode_returns"]))),
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


def _best_col_mask(vals, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL):
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return np.zeros(len(vals), dtype=bool)
    tol = _row_real_win_tol(vals, rel_tol=rel_tol)
    if higher_is_better:
        mx = np.nanmax(vals)
        return valid & (vals >= mx - tol)
    else:
        mn = np.nanmin(vals)
        return valid & (vals <= mn + tol)


def _max_int_mask(vals):
    vals = np.asarray(vals, dtype=float)
    mx = np.max(vals)
    return vals >= mx - 1e-9


def _compute_wtl_stats(mat_mean, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL):
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
        is_top = _best_col_mask(row, higher_is_better=higher_is_better, rel_tol=rel_tol)
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
                diff = row[c] - row[c2] if higher_is_better else row[c2] - row[c]
                if diff > tol:
                    pw_w[c] += 1
                elif diff < -tol:
                    pw_l[c] += 1
                else:
                    pw_t[c] += 1

    return {
        "top_w": top_w,
        "top_t": top_t,
        "top_l": top_l,
        "pw_w": pw_w,
        "pw_t": pw_t,
        "pw_l": pw_l,
    }


def generate_markdown_table(df_mean, df_std, title, cell_annotations=None, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL):
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_mean = df_mean.values
    mat_std = df_std.values

    wtl = _compute_wtl_stats(mat_mean, higher_is_better=higher_is_better, rel_tol=rel_tol)

    lines = []
    lines.append(f"### {title}")
    lines.append("")
    header = "| Environment | " + " | ".join(cols) + " |"
    sep = "|:---| " + " | ".join([":---:"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)

    for r_idx, env in enumerate(envs):
        best_mask = _best_col_mask(mat_mean[r_idx], higher_is_better=higher_is_better, rel_tol=rel_tol)
        row_cells = [f"**{ENV_TITLES.get(env, env)}**"]
        for c_idx, col in enumerate(cols):
            val = mat_mean[r_idx, c_idx]
            std = mat_std[r_idx, c_idx]
            cell_txt = _format_val_std(val, std)
            if best_mask[c_idx]:
                cell_txt = f"**{cell_txt}**"
            if cell_annotations and (r_idx, c_idx) in cell_annotations:
                cell_txt += f" <br><sub>({cell_annotations[(r_idx, c_idx)]})</sub>"
            row_cells.append(cell_txt)
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append(sep)
    # Top-1 Summary
    top_cells = ["**Top-1 (W/T/L)**"]
    top_w_mask = _max_int_mask(wtl["top_w"])
    for c_idx in range(len(cols)):
        s = f"{wtl['top_w'][c_idx]} / {wtl['top_t'][c_idx]} / {wtl['top_l'][c_idx]}"
        if top_w_mask[c_idx] and wtl["top_w"][c_idx] > 0:
            s = f"**{s}**"
        top_cells.append(s)
    lines.append("| " + " | ".join(top_cells) + " |")

    # Pairwise Summary
    pw_cells = ["**Pairwise (W/T/L)**"]
    pw_w_mask = _max_int_mask(wtl["pw_w"])
    for c_idx in range(len(cols)):
        s = f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}"
        if pw_w_mask[c_idx] and wtl["pw_w"][c_idx] > 0:
            s = f"**{s}**"
        pw_cells.append(s)
    lines.append("| " + " | ".join(pw_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def generate_latex_table(df_mean, df_std, title, label, cell_annotations=None, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL):
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_mean = df_mean.values
    mat_std = df_std.values

    wtl = _compute_wtl_stats(mat_mean, higher_is_better=higher_is_better, rel_tol=rel_tol)

    n_cols = len(cols)
    col_spec = "l" + "c" * n_cols
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{title}}}",
        f"\\label{{tab:{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Environment & " + " & ".join([f"\\textbf{{{c}}}" for c in cols]) + " \\\\",
        "\\midrule",
    ]

    for r_idx, env in enumerate(envs):
        best_mask = _best_col_mask(mat_mean[r_idx], higher_is_better=higher_is_better, rel_tol=rel_tol)
        row_str = ENV_TITLES.get(env, env).replace("_", "\\_")
        cells = []
        for c_idx in range(n_cols):
            val = mat_mean[r_idx, c_idx]
            std = mat_std[r_idx, c_idx]
            c_str = _format_val_std_tex(val, std, is_best=best_mask[c_idx])
            cells.append(c_str)
        lines.append(f"{row_str} & " + " & ".join(cells) + " \\\\")

    lines.append("\\midrule")
    top_str = "Top-1 (W/T/L) & " + " & ".join([f"{wtl['top_w'][c]} / {wtl['top_t'][c]} / {wtl['top_l'][c]}" for c in range(n_cols)]) + " \\\\"
    pw_str = "Pairwise (W/T/L) & " + " & ".join([f"{wtl['pw_w'][c]} / {wtl['pw_t'][c]} / {wtl['pw_l'][c]}" for c in range(n_cols)]) + " \\\\"
    lines.append(top_str)
    lines.append(pw_str)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])
    return "\n".join(lines)


def render_table_as_image(df_mean, df_std, title, out_path_png, out_path_pdf=None, cell_annotations=None, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL):
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_mean = df_mean.values
    mat_std = df_std.values
    n_rows, n_cols = mat_mean.shape

    wtl = _compute_wtl_stats(mat_mean, higher_is_better=higher_is_better, rel_tol=rel_tol)

    cell_data = []
    row_headers = []

    for r_idx, env in enumerate(envs):
        row_headers.append(ENV_TITLES.get(env, env))
        best_mask = _best_col_mask(mat_mean[r_idx], higher_is_better=higher_is_better, rel_tol=rel_tol)
        r_cells = []
        for c_idx, _ in enumerate(cols):
            val = mat_mean[r_idx, c_idx]
            std = mat_std[r_idx, c_idx]
            s = _format_val_std(val, std)
            if cell_annotations and (r_idx, c_idx) in cell_annotations:
                s += f"\n({cell_annotations[(r_idx, c_idx)]})"
            r_cells.append(s)
        cell_data.append(r_cells)

    row_headers.append("Top-1 (W/T/L)")
    top_w_mask = _max_int_mask(wtl["top_w"])
    r_top = []
    for c_idx in range(n_cols):
        r_top.append(f"{wtl['top_w'][c_idx]} / {wtl['top_t'][c_idx]} / {wtl['top_l'][c_idx]}")
    cell_data.append(r_top)

    row_headers.append("Pairwise (W/T/L)")
    pw_w_mask = _max_int_mask(wtl["pw_w"])
    r_pw = []
    for c_idx in range(n_cols):
        r_pw.append(f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}")
    cell_data.append(r_pw)

    fig_w = max(7.0, 2.2 * n_cols + 2.5)
    fig_h = max(3.5, 0.45 * (len(row_headers) + 1) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=row_headers,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.4)

    # Header styling
    for c_idx in range(n_cols):
        cell = table[0, c_idx]
        cell.set_facecolor("#2563eb")
        cell.set_text_props(color="white", weight="bold")

    # Data row styling
    for r_idx in range(n_rows):
        best_mask = _best_col_mask(mat_mean[r_idx], higher_is_better=higher_is_better, rel_tol=rel_tol)
        for c_idx in range(n_cols):
            cell = table[r_idx + 1, c_idx]
            if best_mask[c_idx]:
                cell.set_facecolor("#dbeafe")
                cell.set_text_props(weight="bold", color="#1e3a8a")
            else:
                cell.set_facecolor("#ffffff" if r_idx % 2 == 0 else "#f8fafc")

    # Summary row styling
    for c_idx in range(n_cols):
        cell_top = table[n_rows + 1, c_idx]
        cell_top.set_facecolor("#f1f5f9")
        if top_w_mask[c_idx] and wtl["top_w"][c_idx] > 0:
            cell_top.set_text_props(weight="bold", color="#0f766e")

        cell_pw = table[n_rows + 2, c_idx]
        cell_pw.set_facecolor("#e2e8f0")
        if pw_w_mask[c_idx] and wtl["pw_w"][c_idx] > 0:
            cell_pw.set_text_props(weight="bold", color="#0f766e")

    # Row label header styling
    for r_idx in range(len(row_headers)):
        table[r_idx + 1, -1].set_facecolor("#f8fafc")
        table[r_idx + 1, -1].set_text_props(weight="bold")

    ax.set_title(title, fontsize=12.0, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path_png, bbox_inches="tight", dpi=300)
    if out_path_pdf:
        plt.savefig(out_path_pdf, bbox_inches="tight")
    plt.close()


def save_table_bundle(df_mean, df_std, title, base_path, cell_annotations=None, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL):
    md_content = generate_markdown_table(df_mean, df_std, title, cell_annotations, higher_is_better=higher_is_better, rel_tol=rel_tol)
    with open(f"{base_path}.md", "w") as f:
        f.write(md_content)

    label = os.path.basename(base_path)
    tex_content = generate_latex_table(df_mean, df_std, title, label, cell_annotations, higher_is_better=higher_is_better, rel_tol=rel_tol)
    with open(f"{base_path}.tex", "w") as f:
        f.write(tex_content)

    render_table_as_image(
        df_mean, df_std, title,
        out_path_png=f"{base_path}.png",
        out_path_pdf=f"{base_path}.pdf",
        cell_annotations=cell_annotations,
        higher_is_better=higher_is_better,
        rel_tol=rel_tol,
    )
    print(f"Generated Table bundle: {base_path} (.md, .tex, .pdf, .png)")


# ==============================================================================
# Main Sweep Pipeline
# ==============================================================================

def run_sweep(args):
    envs = args.envs
    critic_lrs = args.critic_lrs
    actor_lr = args.actor_lr
    num_envs = args.num_envs
    num_steps = args.num_steps
    minibatch_size = args.minibatch_size
    n_seeds = args.n_seeds
    master_seed = args.master_seed
    out_dir = args.out_dir
    metric_key = args.metric_key

    batch_size = num_envs * num_steps

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    rng = jax.random.PRNGKey(master_seed)
    seed_keys = jax.random.split(rng, n_seeds)

    all_runs = {}
    timesteps = {env: get_env_default_timesteps(env, args.total_timesteps) for env in envs}

    if args.load_existing and os.path.exists(args.load_existing):
        print(f"Loading existing metrics from {args.load_existing}...")
        with open(args.load_existing, "rb") as f:
            saved = pickle.load(f)
            all_runs = saved["runs"]
            if "timesteps" in saved:
                timesteps.update(saved["timesteps"])

    for env_name in envs:
        env_timesteps = get_env_default_timesteps(env_name, args.total_timesteps)
        if env_name not in all_runs:
            all_runs[env_name] = {}

        env_dict = all_runs[env_name]
        for var_key, _ in CRITIC_VARIANTS:
            env_dict.setdefault(var_key, {})

        base_cfg = {
            **gymnax_ppo_config,
            "ENV_NAME": env_name,
            "ACTOR_LR": actor_lr,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": env_timesteps,
            "MINIBATCH_SIZE": minibatch_size,
            "ACTOR_MINIBATCH_SIZE": minibatch_size,
            "CRITIC_MINIBATCH_SIZE": minibatch_size,
            "GAE_LAMBDA": 0.95,
            "VALUE_LAMBDA": 0.95,
            "RETURN_LAMBDA": 1.0,
            "TD_LAMBDA": 0.0,
        }

        print(f"\n>>> Running Double-Sampling Sweep: {env_name} ({env_timesteps:,} steps) <<<")

        for var_key, var_label in CRITIC_VARIANTS:
            for lr in critic_lrs:
                if lr not in env_dict[var_key]:
                    print(f"[{var_label}] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_TYPE": var_key,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_key][lr] = res
                    final_ret = float(res["returned_episode_returns"].mean(axis=0)[-1])
                    final_msbe = float(res["diagnostic_msbe"].mean(axis=0)[-1])
                    print(f"   -> Final Return: {final_ret:.2f} | Diagnostic MSBE: {final_msbe:.5e}")

        # Save checkpoint per environment
        pkl_path = os.path.join(data_dir, "sweep_metrics_all.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"runs": all_runs, "timesteps": timesteps}, f)
        print(f"Checkpointed sweep data to {pkl_path}")

    # Generate Publication Analyses
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, metric_key, batch_size)


# ==============================================================================
# Automated Cross-Algorithm Diagnostics and Publication Tables
# ==============================================================================

def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, metric_key, batch_size):
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    n_envs = len(envs)
    n_cols_plot = min(4, n_envs)
    n_rows_plot = int(np.ceil(n_envs / n_cols_plot))

    var_keys = [k for k, _ in CRITIC_VARIANTS]
    var_labels = [label for _, label in CRITIC_VARIANTS]
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]  # Blue (Fitted), Red (MSBE), Green (E-Min), Purple (MSPBE)
    linestyles = ["-", "--", "-.", ":"]

    # -------------------------------------------------------------
    # 1. ANALYSIS 1: Double Sampling Critic Comparison (Learning Curves & Return)
    # -------------------------------------------------------------
    means_ret, stds_ret, annots_ret = [], [], {}
    auc_means, auc_stds = [], []

    fig1, axes1 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.2 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes1[r][c]
        row_means, row_stds = [], []
        row_auc_m, row_auc_s = [], []

        for c_idx, k in enumerate(var_keys):
            res = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key)
            row_means.append(res["final_mean"])
            row_stds.append(res["final_std"])
            row_auc_m.append(res["auc_mean"])
            row_auc_s.append(res["auc_std"])
            annots_ret[(idx, c_idx)] = f"lr={res['lr']}"

            t = np.arange(1, len(res["curve"]) + 1) * batch_size
            ax.plot(
                t, res["curve"],
                label=f"{var_labels[c_idx]} (lr={res['lr']})",
                color=colors[c_idx],
                lw=2.0,
                ls=linestyles[c_idx],
            )
            ax.fill_between(
                t, res["curve"] - res["std"], res["curve"] + res["std"],
                color=colors[c_idx], alpha=0.15
            )

        means_ret.append(row_means)
        stds_ret.append(row_stds)
        auc_means.append(row_auc_m)
        auc_stds.append(row_auc_s)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=11, fontweight="bold")
        ax.set_xlabel("Environment Steps", fontsize=9)
        ax.set_ylabel("Episode Return", fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7.5)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes1[r][c].axis("off")

    fig1.suptitle("Analysis 1: Double Sampling Critic Comparison — Learning Curves (PPO on Gymnax)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis1_double_sample_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis1_double_sample_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_ret_mean = pd.DataFrame(means_ret, index=envs, columns=var_labels)
    df_ret_std = pd.DataFrame(stds_ret, index=envs, columns=var_labels)
    save_table_bundle(
        df_ret_mean, df_ret_std,
        "Analysis 1: Double-Sampling Critic Final Returns (Mean ± Std over Seeds)",
        os.path.join(tables_dir, "table1_double_sample_final_returns"),
        cell_annotations=annots_ret,
        higher_is_better=True,
    )

    df_auc_mean = pd.DataFrame(auc_means, index=envs, columns=var_labels)
    df_auc_std = pd.DataFrame(auc_stds, index=envs, columns=var_labels)
    save_table_bundle(
        df_auc_mean, df_auc_std,
        "Analysis 1b: Double-Sampling Critic Mean AUC Returns (Mean ± Std over Seeds)",
        os.path.join(tables_dir, "table1b_double_sample_auc_returns"),
        cell_annotations=annots_ret,
        higher_is_better=True,
    )

    # -------------------------------------------------------------
    # 2. ANALYSIS 2: Bellman Error Diagnostics (Unbiased Diagnostic MSBE Reduction)
    # -------------------------------------------------------------
    means_msbe, stds_msbe = [], []
    fig2, axes2 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.2 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes2[r][c]
        row_msbe_m, row_msbe_s = [], []

        for c_idx, k in enumerate(var_keys):
            res = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key="diagnostic_msbe")
            row_msbe_m.append(res["final_mean"])
            row_msbe_s.append(res["final_std"])

            t = np.arange(1, len(res["curve"]) + 1) * batch_size
            msbe_curve = np.maximum(res["curve"], 1e-8)
            ax.plot(
                t, msbe_curve,
                label=f"{var_labels[c_idx]}",
                color=colors[c_idx],
                lw=2.0,
                ls=linestyles[c_idx],
            )

        means_msbe.append(row_msbe_m)
        stds_msbe.append(row_msbe_s)

        ax.set_yscale("log")
        ax.set_title(ENV_TITLES.get(env, env), fontsize=11, fontweight="bold")
        ax.set_xlabel("Environment Steps", fontsize=9)
        ax.set_ylabel("Diagnostic MSBE (log)", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7.5)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes2[r][c].axis("off")

    fig2.suptitle("Analysis 2: Bellman Error Diagnostics — Unbiased MSBE E[δ₁ · δ₂] over Training", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis2_diagnostic_msbe_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis2_diagnostic_msbe_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_msbe_mean = pd.DataFrame(means_msbe, index=envs, columns=var_labels)
    df_msbe_std = pd.DataFrame(stds_msbe, index=envs, columns=var_labels)
    save_table_bundle(
        df_msbe_mean, df_msbe_std,
        "Analysis 2: Diagnostic MSBE Reduction (Lower is Better, Mean ± Std over Seeds)",
        os.path.join(tables_dir, "table2_diagnostic_msbe_reduction"),
        higher_is_better=False,
    )

    # -------------------------------------------------------------
    # 3. ANALYSIS 3: Master Championship & Performance Profiles
    # -------------------------------------------------------------
    fitted_vals = df_ret_mean["Fitted Value (GAE)"].values
    norm_scores = {}
    for col in var_labels:
        raw_vals = df_ret_mean[col].values
        norm = np.where(
            np.abs(fitted_vals) > 1e-4,
            (raw_vals / np.maximum(fitted_vals, 1e-4)) * 100.0,
            100.0,
        )
        norm_scores[col] = norm

    df_norm = pd.DataFrame(norm_scores, index=envs)

    iqm_scores = {}
    median_scores = {}
    mean_norm_scores = {}
    for col in var_labels:
        vals = df_norm[col].values
        valid_vals = vals[~np.isnan(vals)]
        q25, q75 = np.percentile(valid_vals, [25, 75])
        iqm_mask = (valid_vals >= q25) & (valid_vals <= q75)
        iqm_scores[col] = float(np.mean(valid_vals[iqm_mask])) if np.any(iqm_mask) else float(np.mean(valid_vals))
        median_scores[col] = float(np.median(valid_vals))
        mean_norm_scores[col] = float(np.mean(valid_vals))

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Aggregate Score Bar Chart
    x_indices = np.arange(len(var_labels))
    bar_width = 0.35
    ax3a.bar(x_indices - bar_width / 2, [mean_norm_scores[c] for c in var_labels], width=bar_width, label="Mean Normalized Score", color=colors, alpha=0.85)
    ax3a.bar(x_indices + bar_width / 2, [iqm_scores[c] for c in var_labels], width=bar_width, label="Interquartile Mean (IQM)", color=colors, alpha=0.45, hatch="//")
    ax3a.axhline(100.0, color="gray", linestyle="--", lw=1.2, label="Fitted Baseline (100%)")
    ax3a.set_xticks(x_indices)
    ax3a.set_xticklabels(var_labels, rotation=15, ha="right", fontsize=9.5, fontweight="bold")
    ax3a.set_ylabel("Normalized Return (% of Fitted)", fontsize=10)
    ax3a.set_title(f"Championship Aggregate Performance ({n_envs} Gymnax Domains)", fontsize=11, fontweight="bold")
    ax3a.grid(True, alpha=0.3, axis="y")
    ax3a.legend(loc="upper left", fontsize=8.5)

    # Panel B: Performance Profiles (Empirical CDF of Normalized Scores)
    tau_vals = np.linspace(50.0, 150.0, 100)
    for c_idx, col in enumerate(var_labels):
        vals = df_norm[col].values
        cdf = [np.mean(vals >= tau) for tau in tau_vals]
        ax3b.plot(tau_vals, cdf, label=col, color=colors[c_idx], lw=2.2, ls=linestyles[c_idx])

    ax3b.axvline(100.0, color="gray", linestyle="--", lw=1.0)
    ax3b.set_xlabel("Normalized Score Threshold (τ %)", fontsize=10)
    ax3b.set_ylabel("Fraction of Envs with Score ≥ τ", fontsize=10)
    ax3b.set_title(f"Performance Profile across {n_envs} Gymnax Domains", fontsize=11, fontweight="bold")
    ax3b.grid(True, alpha=0.3)
    ax3b.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis3_master_championship_profile.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis3_master_championship_profile.pdf"), bbox_inches="tight")
    plt.close()

    champ_summary = []
    wtl = _compute_wtl_stats(df_ret_mean.values, higher_is_better=True, rel_tol=REAL_WIN_REL_TOL)
    for c_idx, col in enumerate(var_labels):
        champ_summary.append({
            "Algorithm": col,
            "Mean Norm Score (%)": f"{mean_norm_scores[col]:.1f}%",
            "IQM Score (%)": f"{iqm_scores[col]:.1f}%",
            "Median Score (%)": f"{median_scores[col]:.1f}%",
            "Top-1 Wins": wtl["top_w"][c_idx],
            "Pairwise (W/T/L)": f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}",
        })

    df_champ = pd.DataFrame(champ_summary)
    with open(os.path.join(tables_dir, "table3_master_championship.md"), "w") as f:
        f.write(f"### Analysis 3: Master Championship Summary ({n_envs} Gymnax Domains)\n\n")
        f.write(df_champ.to_markdown(index=False))
        f.write("\n")
    print(f"Generated Table 3 summary: {os.path.join(tables_dir, 'table3_master_championship.md')}")

    # -------------------------------------------------------------
    # 4. ANALYSIS 4: Critic Learning Rate Sensitivity
    # -------------------------------------------------------------
    fig4, axes4 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.2 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes4[r][c]

        for c_idx, k in enumerate(var_keys):
            lr_run = all_runs[env].get(k, {})
            lr_means = []
            for lr in critic_lrs:
                if lr in lr_run:
                    crv = get_metric_curve(lr_run[lr], metric_key)
                    lr_means.append(float(np.mean(crv[:, -1])))
                else:
                    lr_means.append(np.nan)

            ax.plot(
                critic_lrs, lr_means,
                marker="o", label=var_labels[c_idx],
                color=colors[c_idx], lw=1.8, ls=linestyles[c_idx],
            )

        ax.set_xscale("log")
        ax.set_title(ENV_TITLES.get(env, env), fontsize=11, fontweight="bold")
        ax.set_xlabel("Critic Learning Rate", fontsize=9)
        ax.set_ylabel("Final Return", fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7.5)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes4[r][c].axis("off")

    fig4.suptitle("Analysis 4: Critic Learning Rate Sensitivity Curves across Algorithms", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis4_critic_lr_sensitivity.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis4_critic_lr_sensitivity.pdf"), bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Double-Sampling Critic Comparison Suite on Gymnax")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, help="Environments to evaluate")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS, help="Critic learning rates")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Timesteps override per env")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout steps")
    parser.add_argument("--minibatch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key")
    parser.add_argument("--out_dir", type=str, default="results/sweep_double_sample_gymnax", help="Output directory")
    parser.add_argument("--load_existing", type=str, default=None, help="Existing pickle path to resume")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
sweep_ppo_critics.py

Comprehensive Critic Comparison Sweep & Automated Publication Analysis Suite for PPO on Brax:
Applies to continuous control environments with deterministic physics:
1. E(0) Family:
   - Naive (no v_aux, symmetric Dirichlet (e_i - e_j)^2)
   - v_aux (surgical auxiliary value network V_aux)
   - e_aux (empirical rollout return auxiliary target)
   - Forward-Only (no v_aux, semi-gradient (e_i - sg(e_j))^2)
2. E(lambda=0.8) Family:
   - Naive (no v_aux, geometric lookahead jumps)
   - v_aux (surgical auxiliary value network V_aux)
   - e_aux (empirical rollout return auxiliary target)
   - Forward-Only (no v_aux, semi-gradient)
3. TD(lambda) / Fitted Value Iteration Grid:
   - TD(0.0) / 1-step online TD(0)
   - TD(0.4)
   - TD(0.8)
   - GAE / TD(0.95)
   - MC (lambda=1.0 rollout return regression)

Automatically generates 5 publication-grade analyses (Plots in .pdf/.png and Tables in .md/.tex/.pdf/.png):
- Table 1 & Figure 1: E(0) Variants (Naive vs v_aux vs e_aux vs Forward-Only)
- Table 2 & Figure 2: E(lambda=0.8) Variants (Naive vs v_aux vs e_aux vs Forward-Only)
- Table 3 & Figure 3: Naive Head-to-Head (E(0) Naive vs E(0.8) Naive)
- Table 4 & Figure 4: TD(lambda) Grid (Effect of lambda on Fitted Critic)
- Table 5 & Figure 5: Master Championship (Overall Best E vs Overall Best TD)
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

from brax_rl.core.brax_config import config as brax_ppo_config
from brax_rl.algos.ppo import make_train as make_ppo_train

ENV_TITLES = {
    "hopper": "Hopper",
    "halfcheetah": "HalfCheetah",
    "walker2d": "Walker2d",
    "ant": "Ant",
    "inverted_pendulum": "Inverted Pendulum",
    "reacher": "Reacher",
}

DEFAULT_ENVS = ["hopper", "halfcheetah", "walker2d", "ant", "inverted_pendulum", "reacher"]
DEFAULT_CRITIC_LRS = [3e-4, 1e-3, 3e-3]
DEFAULT_TD_LAMBDAS = [0.0, 0.4, 0.8, 0.95, 1.0]
REAL_WIN_REL_TOL = 0.01  # 1% of environment return magnitude


def get_env_default_timesteps(env_name: str, fallback_timesteps: int | None = None) -> int:
    """Returns environment-specific timesteps for Brax."""
    if fallback_timesteps is not None:
        return fallback_timesteps
    if env_name in ["inverted_pendulum"]:
        return 2_000_000
    elif env_name in ["reacher"]:
        return 5_000_000
    elif env_name in ["hopper", "walker2d", "halfcheetah", "ant"]:
        return 10_000_000
    else:
        return 5_000_000


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
        "returned_episode_lengths": np.array(metrics.get("returned_episode_lengths", np.zeros_like(metrics["returned_episode_returns"]))),
        "critic_loss": np.array(
            metrics["value_loss"] if "value_loss" in metrics else metrics.get("val_loss", 0.0)
        ),
        "actor_loss": np.array(metrics.get("actor_loss", 0.0)),
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
                diff = row[c] - row[c2]
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


def generate_markdown_table(df_mean, df_std, title, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL):
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_mean = df_mean.values
    mat_std = df_std.values

    wtl = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)

    lines = []
    lines.append(f"### {title}")
    lines.append("")
    header = "| Environment | " + " | ".join(cols) + " |"
    sep = "|:---| " + " | ".join([":---:"] * len(cols)) + " |"
    lines.append(header)
    lines.append(sep)

    for r_idx, env in enumerate(envs):
        best_mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
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

    lines.append("|" + "---|" * (len(cols) + 1))

    # Top-1 W / T / L row
    top_cells = ["**Top-1 (W / T / L)**"]
    top_w_mask = _max_int_mask(wtl["top_w"])
    for c_idx in range(len(cols)):
        txt = f"{wtl['top_w'][c_idx]} / {wtl['top_t'][c_idx]} / {wtl['top_l'][c_idx]}"
        if top_w_mask[c_idx] and wtl["top_w"][c_idx] > 0:
            txt = f"**{txt}**"
        top_cells.append(txt)
    lines.append("| " + " | ".join(top_cells) + " |")

    # Pairwise W / T / L row
    pw_cells = ["**Pairwise (W / T / L)**"]
    pw_w_mask = _max_int_mask(wtl["pw_w"])
    for c_idx in range(len(cols)):
        txt = f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}"
        if pw_w_mask[c_idx] and wtl["pw_w"][c_idx] > 0:
            txt = f"**{txt}**"
        pw_cells.append(txt)
    lines.append("| " + " | ".join(pw_cells) + " |")

    lines.append("")
    return "\n".join(lines)


def generate_latex_table(df_mean, df_std, caption, label, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL):
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_mean = df_mean.values
    mat_std = df_std.values
    n_cols = len(cols)

    wtl = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)

    col_align = "l" + "c" * n_cols
    clean_cols = [c.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%") for c in cols]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{col_align}}}")
    lines.append("\\toprule")
    lines.append("Environment & " + " & ".join(clean_cols) + " \\\\")
    lines.append("\\midrule")

    for r_idx, env in enumerate(envs):
        best_mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
        clean_env = ENV_TITLES.get(env, env).replace("_", "\\_")
        row_cells = [clean_env]
        for c_idx, _ in enumerate(cols):
            val = mat_mean[r_idx, c_idx]
            std = mat_std[r_idx, c_idx]
            is_best = bool(best_mask[c_idx])
            cell_txt = _format_val_std_tex(val, std, is_best=is_best)
            row_cells.append(cell_txt)
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\midrule")

    # Top-1 row
    top_cells = ["Top-1 (W / T / L)"]
    top_w_mask = _max_int_mask(wtl["top_w"])
    for c_idx in range(n_cols):
        s = f"{wtl['top_w'][c_idx]} / {wtl['top_t'][c_idx]} / {wtl['top_l'][c_idx]}"
        if top_w_mask[c_idx] and wtl["top_w"][c_idx] > 0:
            s = f"\\textbf{{{s}}}"
        top_cells.append(s)
    lines.append(" & ".join(top_cells) + " \\\\")

    # Pairwise row
    pw_cells = ["Pairwise (W / T / L)"]
    pw_w_mask = _max_int_mask(wtl["pw_w"])
    for c_idx in range(n_cols):
        s = f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}"
        if pw_w_mask[c_idx] and wtl["pw_w"][c_idx] > 0:
            s = f"\\textbf{{{s}}}"
        pw_cells.append(s)
    lines.append(" & ".join(pw_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    lines.append("")
    return "\n".join(lines)


def render_table_as_image(df_mean, df_std, title, out_path_png, out_path_pdf=None, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL):
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_mean = df_mean.values
    mat_std = df_std.values
    n_rows, n_cols = mat_mean.shape

    wtl = _compute_wtl_stats(mat_mean, rel_tol=rel_tol)

    cell_data = []
    row_headers = []

    for r_idx, env in enumerate(envs):
        row_headers.append(ENV_TITLES.get(env, env))
        best_mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
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
        best_mask = _best_col_mask(mat_mean[r_idx], rel_tol=rel_tol)
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


def save_table_bundle(df_mean, df_std, title, base_path, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL):
    # 1. Markdown
    md_content = generate_markdown_table(df_mean, df_std, title, cell_annotations, rel_tol=rel_tol)
    with open(f"{base_path}.md", "w") as f:
        f.write(md_content)

    # 2. LaTeX
    label = os.path.basename(base_path)
    tex_content = generate_latex_table(df_mean, df_std, title, label, cell_annotations, rel_tol=rel_tol)
    with open(f"{base_path}.tex", "w") as f:
        f.write(tex_content)

    # 3. PNG & PDF
    render_table_as_image(
        df_mean, df_std, title,
        out_path_png=f"{base_path}.png",
        out_path_pdf=f"{base_path}.pdf",
        cell_annotations=cell_annotations,
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
    td_lambdas = args.td_lambdas
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
        env_dict.setdefault("E(0)_naive", {})
        env_dict.setdefault("E(0)_v_aux", {})
        env_dict.setdefault("E(0)_e_aux", {})
        env_dict.setdefault("E(0)_fwd", {})
        env_dict.setdefault("E(0.8)_naive", {})
        env_dict.setdefault("E(0.8)_v_aux", {})
        env_dict.setdefault("E(0.8)_e_aux", {})
        env_dict.setdefault("E(0.8)_fwd", {})
        env_dict.setdefault("TD", {lam: {} for lam in td_lambdas})

        for lam in td_lambdas:
            env_dict["TD"].setdefault(lam, {})

        critic_decay_ratio = brax_ppo_config.get(
            "CRITIC_LR_END", brax_ppo_config["NETWORK_CONFIG"]["LR_END"]
        ) / brax_ppo_config.get(
            "CRITIC_LR", brax_ppo_config["NETWORK_CONFIG"]["LR"]
        )
        base_cfg = {
            **brax_ppo_config,
            "ENV_NAME": env_name,
            "ACTOR_LR": actor_lr,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": env_timesteps,
            "MINIBATCH_SIZE": minibatch_size,
            "ACTOR_MINIBATCH_SIZE": minibatch_size,
            "CRITIC_MINIBATCH_SIZE": minibatch_size,
        }

        print(f"\n>>> Running Environment: {env_name} ({env_timesteps:,} steps) <<<")

        # 1. E(0) Variants
        for var_name, critic_type, e_var in [
            ("E(0)_naive", "e_0", "naive"),
            ("E(0)_v_aux", "e_0", "aux"),
            ("E(0)_e_aux", "e_0", "e_aux"),
            ("E(0)_fwd", "e_0", "fwd"),
        ]:
            for lr in critic_lrs:
                if lr not in env_dict[var_name]:
                    print(f"[{var_name}] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": critic_decay_ratio * lr,
                        "CRITIC_TYPE": critic_type,
                        "E_VARIANT": e_var,
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": 1.0,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_name][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 2. E(lambda=0.8) Variants
        for var_name, critic_type, e_var in [
            ("E(0.8)_naive", "e_lambda", "naive"),
            ("E(0.8)_v_aux", "e_lambda", "aux"),
            ("E(0.8)_e_aux", "e_lambda", "e_aux"),
            ("E(0.8)_fwd", "e_lambda", "fwd"),
        ]:
            for lr in critic_lrs:
                if lr not in env_dict[var_name]:
                    print(f"[{var_name}] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": critic_decay_ratio * lr,
                        "CRITIC_TYPE": critic_type,
                        "E_VARIANT": e_var,
                        "E_LAMBDA": 0.8,
                        "RETURN_LAMBDA": 1.0,
                        "AUX_LAMBDA": 1.0,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_name][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

        # 3. TD(lambda) Fitted Value Iteration Grid
        for lam in td_lambdas:
            for lr in critic_lrs:
                if lr not in env_dict["TD"][lam]:
                    c_type = "td_0" if lam == 0.0 else "fitted"
                    print(f"[TD(lambda={lam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": critic_decay_ratio * lr,
                        "CRITIC_TYPE": c_type,
                        "VALUE_LAMBDA": lam,
                        "TD_LAMBDA": 0.0,
                        "E_VARIANT": "naive",
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["TD"][lam][lr] = res
                    print(f"   -> Final Return: {res['returned_episode_returns'].mean(axis=0)[-1]:.2f}")

    # Save all raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate all analyses
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, td_lambdas, metric_key, batch_size)


# ==============================================================================
# Automated Cross-Algorithm Diagnostics and Tables
# ==============================================================================

def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, td_lambdas, metric_key, batch_size):
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    n_envs = len(envs)
    n_cols_plot = min(4, n_envs)
    n_rows_plot = int(np.ceil(n_envs / n_cols_plot))

    # -------------------------------------------------------------
    # 1. ANALYSIS 1: E(0) Variants (Naive vs v_aux vs e_aux vs Fwd)
    # -------------------------------------------------------------
    e0_cols = ["E(0) Naive", "E(0) v_aux", "E(0) e_aux", "E(0) Forward-Only"]
    e0_keys = ["E(0)_naive", "E(0)_v_aux", "E(0)_e_aux", "E(0)_fwd"]
    colors_e0 = ["#2563eb", "#059669", "#0891b2", "#d97706"]
    e0_means, e0_stds, e0_annots = [], [], {}

    fig1, axes1 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes1[r][c]
        row_means, row_stds = [], []
        for c_idx, k in enumerate(e0_keys):
            res = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key)
            row_means.append(res["auc_mean"])
            row_stds.append(res["auc_std"])
            e0_annots[(idx, c_idx)] = f"lr={res['lr']}"
            t = np.arange(1, len(res["curve"]) + 1) * batch_size
            ls = "--" if k == "E(0)_fwd" else "-"
            ax.plot(t, res["curve"], label=f"{e0_cols[c_idx]} (lr={res['lr']})", color=colors_e0[c_idx], lw=2.0, ls=ls)
            ax.fill_between(t, res["curve"] - res["std"], res["curve"] + res["std"], color=colors_e0[c_idx], alpha=0.15)
        e0_means.append(row_means)
        e0_stds.append(row_stds)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7.5)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes1[r][c].axis("off")

    fig1.suptitle("Analysis 1: E(0) Variants Learning Curves (PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis1_e0_variants_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis1_e0_variants_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_e0_mean = pd.DataFrame(e0_means, index=envs, columns=e0_cols)
    df_e0_std = pd.DataFrame(e0_stds, index=envs, columns=e0_cols)
    save_table_bundle(
        df_e0_mean, df_e0_std,
        "Table 1: E(0) Formulation Comparison on PPO (AUC Return)",
        os.path.join(tables_dir, "table1_e0_variants"),
        cell_annotations=e0_annots,
    )

    # -------------------------------------------------------------
    # 2. ANALYSIS 2: E(lambda=0.8) Variants (Naive vs v_aux vs e_aux vs Fwd)
    # -------------------------------------------------------------
    elam_cols = ["E(0.8) Naive", "E(0.8) v_aux (Debiased)", "E(0.8) e_aux", "E(0.8) Forward-Only"]
    elam_keys = ["E(0.8)_naive", "E(0.8)_v_aux", "E(0.8)_e_aux", "E(0.8)_fwd"]
    colors_elam = ["#7c3aed", "#059669", "#0891b2", "#ea580c"]
    elam_means, elam_stds, elam_annots = [], [], {}

    fig2, axes2 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes2[r][c]
        row_means, row_stds = [], []
        for c_idx, k in enumerate(elam_keys):
            res = select_best_lr_run(all_runs[env].get(k, {}), critic_lrs, metric_key)
            row_means.append(res["auc_mean"])
            row_stds.append(res["auc_std"])
            elam_annots[(idx, c_idx)] = f"lr={res['lr']}"
            t = np.arange(1, len(res["curve"]) + 1) * batch_size
            ls = "--" if k == "E(0.8)_fwd" else "-"
            ax.plot(t, res["curve"], label=f"{elam_cols[c_idx]} (lr={res['lr']})", color=colors_elam[c_idx], lw=2.0, ls=ls)
            ax.fill_between(t, res["curve"] - res["std"], res["curve"] + res["std"], color=colors_elam[c_idx], alpha=0.15)
        elam_means.append(row_means)
        elam_stds.append(row_stds)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7.5)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes2[r][c].axis("off")

    fig2.suptitle("Analysis 2: E(lambda=0.8) Variants Learning Curves (PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis2_elambda_variants_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis2_elambda_variants_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_elam_mean = pd.DataFrame(elam_means, index=envs, columns=elam_cols)
    df_elam_std = pd.DataFrame(elam_stds, index=envs, columns=elam_cols)
    save_table_bundle(
        df_elam_mean, df_elam_std,
        "Table 2: E(lambda=0.8) Formulation Comparison on PPO (AUC Return)",
        os.path.join(tables_dir, "table2_elambda_variants"),
        cell_annotations=elam_annots,
    )

    # -------------------------------------------------------------
    # 3. ANALYSIS 3: Naive Head-to-Head (E(0) Naive vs E(0.8) Naive)
    # -------------------------------------------------------------
    naive_cols = ["E(0) Naive (1-Step)", "E(0.8) Naive (Multi-Step Jump)"]
    naive_means, naive_stds, naive_annots = [], [], {}

    fig3, axes3 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes3[r][c]

        e0_naive = select_best_lr_run(all_runs[env].get("E(0)_naive", {}), critic_lrs, metric_key)
        elam_naive = select_best_lr_run(all_runs[env].get("E(0.8)_naive", {}), critic_lrs, metric_key)

        naive_means.append([e0_naive["auc_mean"], elam_naive["auc_mean"]])
        naive_stds.append([e0_naive["auc_std"], elam_naive["auc_std"]])
        naive_annots[(idx, 0)] = f"lr={e0_naive['lr']}"
        naive_annots[(idx, 1)] = f"lr={elam_naive['lr']}"

        t = np.arange(1, len(e0_naive["curve"]) + 1) * batch_size
        ax.plot(t, e0_naive["curve"], label=f"E(0) Naive (lr={e0_naive['lr']})", color="#2563eb", lw=2.2)
        ax.fill_between(t, e0_naive["curve"] - e0_naive["std"], e0_naive["curve"] + e0_naive["std"], color="#2563eb", alpha=0.15)

        ax.plot(t, elam_naive["curve"], label=f"E(0.8) Naive (lr={elam_naive['lr']})", color="#7c3aed", lw=2.2)
        ax.fill_between(t, elam_naive["curve"] - elam_naive["std"], elam_naive["curve"] + elam_naive["std"], color="#7c3aed", alpha=0.15)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes3[r][c].axis("off")

    fig3.suptitle("Analysis 3: Naive Head-to-Head (E(0) Naive vs E(0.8) Naive) Learning Curves (PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis3_naive_showdown_e0_vs_elambda.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis3_naive_showdown_e0_vs_elambda.pdf"), bbox_inches="tight")
    plt.close()

    df_naive_mean = pd.DataFrame(naive_means, index=envs, columns=naive_cols)
    df_naive_std = pd.DataFrame(naive_stds, index=envs, columns=naive_cols)
    save_table_bundle(
        df_naive_mean, df_naive_std,
        "Table 3: Naive Head-to-Head on PPO (E(0) 1-Step vs E(0.8) Multi-Step Jump)",
        os.path.join(tables_dir, "table3_naive_showdown_e0_vs_elambda"),
        cell_annotations=naive_annots,
    )

    # -------------------------------------------------------------
    # 4. ANALYSIS 4: TD(lambda) Fitted Value Iteration Grid
    # -------------------------------------------------------------
    td_cols = [
        "TD(0)" if lam == 0.0 else (f"TD(λ={lam})" if lam < 1.0 else "MC (λ=1.0)")
        for lam in td_lambdas
    ]
    td_means, td_stds, td_annots = [], [], {}

    fig4, axes4 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)
    cmap = matplotlib.colormaps["viridis"]
    td_colors = [cmap(i / max(1, len(td_lambdas) - 1)) for i in range(len(td_lambdas))]

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes4[r][c]

        env_td_means, env_td_stds = [], []
        for l_idx, lam in enumerate(td_lambdas):
            res = select_best_lr_run(all_runs[env].get("TD", {}).get(lam, {}), critic_lrs, metric_key)
            env_td_means.append(res["auc_mean"])
            env_td_stds.append(res["auc_std"])
            td_annots[(idx, l_idx)] = f"lr={res['lr']}"

            t = np.arange(1, len(res["curve"]) + 1) * batch_size
            lbl = f"λ={lam} (lr={res['lr']})" if lam > 0 else f"TD(0) (lr={res['lr']})"
            ax.plot(t, res["curve"], label=lbl, color=td_colors[l_idx], lw=1.8)

        td_means.append(env_td_means)
        td_stds.append(env_td_stds)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7.5)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes4[r][c].axis("off")

    fig4.suptitle("Analysis 4: TD(lambda) Fitted Value Iteration Grid Learning Curves (PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis4_td_lambda_grid_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis4_td_lambda_grid_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_td_mean = pd.DataFrame(td_means, index=envs, columns=td_cols)
    df_td_std = pd.DataFrame(td_stds, index=envs, columns=td_cols)
    save_table_bundle(
        df_td_mean, df_td_std,
        "Table 4: TD(lambda) Sensitivity Grid on PPO (AUC Return)",
        os.path.join(tables_dir, "table4_td_lambda_grid"),
        cell_annotations=td_annots,
    )

    # -------------------------------------------------------------
    # 5. ANALYSIS 5: Master Championship (Best E vs Best TD)
    # -------------------------------------------------------------
    champ_cols = ["Best TD / Fitted", "Best E-Minimization"]
    champ_means, champ_stds, champ_annots = [], [], {}

    fig5, axes5 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.0 * n_cols_plot, 3.2 * n_rows_plot), squeeze=False)

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes5[r][c]

        # 1. Find Best TD run
        best_td = None
        for lam in td_lambdas:
            cand = select_best_lr_run(all_runs[env].get("TD", {}).get(lam, {}), critic_lrs, metric_key)
            td_name = f"TD(λ={lam})" if lam > 0 else "TD(0)"
            if best_td is None or cand["auc_mean"] > best_td["auc_mean"]:
                best_td = {**cand, "name": td_name}

        # 2. Find Best E run
        best_e = None
        for e_key, clean_name in [
            ("E(0)_naive", "E(0) Naive"),
            ("E(0)_v_aux", "E(0) v_aux"),
            ("E(0)_e_aux", "E(0) e_aux"),
            ("E(0)_fwd", "E(0) Fwd"),
            ("E(0.8)_naive", "E(0.8) Naive"),
            ("E(0.8)_v_aux", "E(0.8) v_aux"),
            ("E(0.8)_e_aux", "E(0.8) e_aux"),
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

    fig5.suptitle("Analysis 5: Master Championship (Best TD vs Best E) Learning Curves (PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis5_best_e_vs_best_td_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis5_best_e_vs_best_td_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_champ_mean = pd.DataFrame(champ_means, index=envs, columns=champ_cols)
    df_champ_std = pd.DataFrame(champ_stds, index=envs, columns=champ_cols)
    save_table_bundle(
        df_champ_mean, df_champ_std,
        "Table 5: Master Championship on PPO (Overall Best TD vs Overall Best E)",
        os.path.join(tables_dir, "table5_best_e_vs_best_td_championship"),
        cell_annotations=champ_annots,
    )

    print("\nAll 5 Cross-Algorithm Analyses, Plots, and Tables generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run PPO Comprehensive Critic Sweep on Brax")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, help="Brax Environment Names")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS, help="Critic learning rates")
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=brax_ppo_config.get("ACTOR_LR", brax_ppo_config["NETWORK_CONFIG"]["LR"]),
        help="Fixed actor learning rate",
    )
    parser.add_argument("--td_lambdas", nargs="+", type=float, default=DEFAULT_TD_LAMBDAS, help="TD/GAE lambdas to sweep")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps per env")
    parser.add_argument("--num_envs", type=int, default=brax_ppo_config["NUM_ENVS"], help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=brax_ppo_config["NUM_STEPS"], help="Rollout length")
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=brax_ppo_config.get(
            "ACTOR_MINIBATCH_SIZE", brax_ppo_config["NETWORK_CONFIG"]["MINIBATCH_SIZE"]
        ),
        help="Minibatch size",
    )
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")
    parser.add_argument("--out_dir", type=str, default="results/sweep_ppo_critic_brax", help="Output directory")
    parser.add_argument("--load_existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

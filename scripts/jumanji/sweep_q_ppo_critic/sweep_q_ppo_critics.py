#!/usr/bin/env python3
"""
sweep_q_ppo_critics.py

Comprehensive Q-PPO Critic Comparison Sweep across Discrete Jumanji Environments:
1. E(0) Variants: Naive (Exact 1-Step Bellman Residual) vs v_aux vs Forward-Only
2. E(lambda=0.8) Variants: v_aux (Debiased Default!) vs Naive vs Forward-Only
3. Head-to-Head: E(0) Naive vs E(0.8) v_aux (Debiased)
4. TD(lambda) Fitted SARSA Grid: lambda in [0.0, 0.4, 0.8, 0.95, 1.0]
5. Master Championship: Best TD / Fitted vs Best E-Minimization

Generates:
- 5 Publication-grade cross-environment learning curve figures (.png, .pdf)
- 5 Publication-grade tables (.md, .tex, .png, .pdf) with bottom rows for
  'Top-1 (W/T/L)' (Number of won environments) and 'Pairwise (W/T/L)'
"""

from __future__ import annotations
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure E_min root is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from jumanji_rl.core.jumanji_config import config as base_default_config
from jumanji_rl.algos.q_ppo import make_train

DEFAULT_ENVS = [
    "Sokoban-v0",
    "Snake-v1",
    "Maze-v0",
    "SlidingTilePuzzle-v0",
    "RubiksCube-partly-scrambled-v0",
    "Game2048-v1",
]

ENV_TITLES = {
    "Sokoban-v0": "Sokoban (10x10)",
    "Snake-v1": "Snake (12x12)",
    "Maze-v0": "Maze Navigation (10x10)",
    "SlidingTilePuzzle-v0": "Sliding Tile Puzzle (5x5)",
    "RubiksCube-partly-scrambled-v0": "Rubik's Cube (3x3x3, 18-Action)",
    "Game2048-v1": "Game 2048 (4x4)",
}

DEFAULT_CRITIC_LRS = [0.0001, 0.0003, 0.001]
DEFAULT_TD_LAMBDAS = [0.0, 0.4, 0.8, 0.95, 1.0]

ENV_DEFAULT_TIMESTEPS = {
    "Sokoban-v0": 2_000_000,
    "Snake-v1": 2_000_000,
    "Maze-v0": 2_000_000,
    "SlidingTilePuzzle-v0": 2_000_000,
    "RubiksCube-partly-scrambled-v0": 2_000_000,
    "Game2048-v1": 2_000_000,
}

REAL_WIN_REL_TOL = 0.01  # 1% relative tolerance for ties


def get_env_default_timesteps(env_name: str, override: int | None = None) -> int:
    if override is not None:
        return override
    return ENV_DEFAULT_TIMESTEPS.get(env_name, 2_000_000)


def run_experiment(config: dict, seed_keys: jax.Array) -> dict:
    """Runs parallel seeds via jax.vmap(train) and averages metrics."""
    train_fn = make_train(config)
    vmap_train = jax.vmap(train_fn)
    out = vmap_train(seed_keys)
    metrics = out["metrics"]
    return {
        k: np.array(v)
        for k, v in metrics.items()
        if isinstance(v, (jax.Array, np.ndarray))
    }


def select_best_lr_run(lr_dict: dict, critic_lrs: list[float], metric_key: str = "returned_episode_returns") -> dict:
    """Finds best critic learning rate run based on Area Under Curve (AUC)."""
    best_auc = -np.inf
    best_lr = critic_lrs[0]
    best_curve = None
    best_std = None

    for lr in critic_lrs:
        if lr not in lr_dict:
            continue
        data = lr_dict[lr]
        if metric_key not in data:
            continue
        curves = data[metric_key]  # shape: (n_seeds, n_updates)
        mean_curve = np.nanmean(curves, axis=0)
        std_curve = np.nanstd(curves, axis=0)
        auc = np.nanmean(mean_curve)
        if np.isnan(auc):
            auc = -np.inf
        if auc > best_auc:
            best_auc = auc
            best_lr = lr
            best_curve = mean_curve
            best_std = std_curve

    if best_curve is None:
        best_curve = np.zeros(10)
        best_std = np.zeros(10)
        best_auc = 0.0

    return {
        "lr": best_lr,
        "auc_mean": float(best_auc),
        "auc_std": float(np.nanmean(best_std)),
        "curve": best_curve,
        "std": best_std,
    }


# ==============================================================================
# Statistical Win/Tie/Loss Calculation (Top-1 Wins and Pairwise Wins)
# ==============================================================================

def _compute_wtl_stats(mat_mean: np.ndarray, rel_tol: float = REAL_WIN_REL_TOL):
    n_rows, n_cols = mat_mean.shape
    top_w = [0] * n_cols
    top_t = [0] * n_cols
    top_l = [0] * n_cols

    pw_w = [0] * n_cols
    pw_t = [0] * n_cols
    pw_l = [0] * n_cols

    for r in range(n_rows):
        row = mat_mean[r]
        row_max = np.max(row)
        scale = max(abs(float(row_max)), 1.0)
        tol = rel_tol * scale

        # Top-1 W/T/L
        is_top = (row_max - row) <= tol
        n_tops = int(np.sum(is_top))
        for c in range(n_cols):
            if is_top[c]:
                if n_tops == 1:
                    top_w[c] += 1
                else:
                    top_t[c] += 1
            else:
                top_l[c] += 1

        # Pairwise W/T/L
        for c in range(n_cols):
            for c2 in range(n_cols):
                if c == c2:
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


def _best_col_mask(row: np.ndarray, rel_tol: float = REAL_WIN_REL_TOL) -> np.ndarray:
    row_max = np.max(row)
    scale = max(abs(float(row_max)), 1.0)
    return (row_max - row) <= (rel_tol * scale)


def _format_val_std(val: float, std: float) -> str:
    if abs(val) >= 100:
        return f"{val:.1f} ± {std:.1f}"
    elif abs(val) >= 1:
        return f"{val:.2f} ± {std:.2f}"
    else:
        return f"{val:.3f} ± {std:.3f}"


def generate_markdown_table(df_mean: pd.DataFrame, df_std: pd.DataFrame, title: str, cell_annotations=None, rel_tol=REAL_WIN_REL_TOL) -> str:
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
            if cell_annotations and (r_idx, c_idx) in cell_annotations:
                cell_txt += f" <br><sub>({cell_annotations[(r_idx, c_idx)]})</sub>"
            if best_mask[c_idx]:
                cell_txt = f"**{cell_txt}**"
            row_cells.append(cell_txt)
        lines.append("| " + " | ".join(row_cells) + " |")

    # Bottom summary row for Number of Won Environments
    top_cells = ["**Number of Won Environments (W / T / L)**"]
    for c_idx in range(len(cols)):
        txt = f"{wtl['top_w'][c_idx]} / {wtl['top_t'][c_idx]} / {wtl['top_l'][c_idx]}"
        if wtl["top_w"][c_idx] == max(wtl["top_w"]) and wtl["top_w"][c_idx] > 0:
            txt = f"**{txt}**"
        top_cells.append(txt)
    lines.append("| " + " | ".join(top_cells) + " |")

    # Pairwise W/T/L
    pw_cells = ["**Pairwise Tournament (W / T / L)**"]
    for c_idx in range(len(cols)):
        txt = f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}"
        if wtl["pw_w"][c_idx] == max(wtl["pw_w"]) and wtl["pw_w"][c_idx] > 0:
            txt = f"**{txt}**"
        pw_cells.append(txt)
    lines.append("| " + " | ".join(pw_cells) + " |")
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

    row_headers.append("Won Envs (W/T/L)")
    r_top = []
    for c_idx in range(n_cols):
        r_top.append(f"{wtl['top_w'][c_idx]} / {wtl['top_t'][c_idx]} / {wtl['top_l'][c_idx]}")
    cell_data.append(r_top)

    row_headers.append("Pairwise (W/T/L)")
    r_pw = []
    for c_idx in range(n_cols):
        r_pw.append(f"{wtl['pw_w'][c_idx]} / {wtl['pw_t'][c_idx]} / {wtl['pw_l'][c_idx]}")
    cell_data.append(r_pw)

    fig_w = max(7.5, 2.4 * n_cols + 2.5)
    fig_h = max(3.5, 0.5 * (len(row_headers) + 1) + 1.2)
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

    # Bottom summary row styling
    for c_idx in range(n_cols):
        cell_top = table[n_rows + 1, c_idx]
        cell_top.set_facecolor("#f1f5f9")
        if wtl["top_w"][c_idx] == max(wtl["top_w"]) and wtl["top_w"][c_idx] > 0:
            cell_top.set_text_props(weight="bold", color="#0f766e")

        cell_pw = table[n_rows + 2, c_idx]
        cell_pw.set_facecolor("#e2e8f0")
        if wtl["pw_w"][c_idx] == max(wtl["pw_w"]) and wtl["pw_w"][c_idx] > 0:
            cell_pw.set_text_props(weight="bold", color="#0f766e")

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

    # 2. PNG & PDF
    render_table_as_image(
        df_mean, df_std, title,
        out_path_png=f"{base_path}.png",
        out_path_pdf=f"{base_path}.pdf",
        cell_annotations=cell_annotations,
        rel_tol=rel_tol,
    )
    print(f"Generated Table bundle: {base_path} (.md, .pdf, .png)")


# ==============================================================================
# Cross-Environment Learning Curves & Diagnostic Generation
# ==============================================================================

def generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, td_lambdas, metric_key, batch_size):
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    n_envs = len(envs)
    n_cols_plot = min(3, n_envs)
    n_rows_plot = int(np.ceil(n_envs / n_cols_plot))

    # -------------------------------------------------------------
    # 1. ANALYSIS 1: E(0) Variants (Naive vs v_aux vs Fwd)
    # -------------------------------------------------------------
    e0_cols = ["E(0) Naive (Exact 1-Step)", "E(0) v_aux", "E(0) Forward-Only"]
    e0_keys = ["E(0)_naive", "E(0)_v_aux", "E(0)_fwd"]
    colors_e0 = ["#2563eb", "#059669", "#d97706"]
    e0_means, e0_stds, e0_annots = [], [], {}

    fig1, axes1 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.5 * n_cols_plot, 3.5 * n_rows_plot), squeeze=False)
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

        ax.set_title(ENV_TITLES.get(env, env), fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes1[r][c].axis("off")

    fig1.suptitle("Analysis 1: E(0) Variants Cross-Environment Learning Curves (Q-PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis1_e0_variants_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis1_e0_variants_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_e0_mean = pd.DataFrame(e0_means, index=envs, columns=e0_cols)
    df_e0_std = pd.DataFrame(e0_stds, index=envs, columns=e0_cols)
    save_table_bundle(
        df_e0_mean, df_e0_std,
        "Table 1: E(0) Formulation Comparison on Q-PPO (AUC Return)",
        os.path.join(tables_dir, "table1_e0_variants"),
        cell_annotations=e0_annots,
    )

    # -------------------------------------------------------------
    # 2. ANALYSIS 2: E(lambda=0.8) Variants (v_aux Debiased vs Naive vs Fwd)
    # -------------------------------------------------------------
    elam_cols = ["E(0.8) v_aux (Debiased Default)", "E(0.8) Naive", "E(0.8) Forward-Only"]
    elam_keys = ["E(0.8)_v_aux", "E(0.8)_naive", "E(0.8)_fwd"]
    colors_elam = ["#059669", "#7c3aed", "#ea580c"]
    elam_means, elam_stds, elam_annots = [], [], {}

    fig2, axes2 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.5 * n_cols_plot, 3.5 * n_rows_plot), squeeze=False)
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

        ax.set_title(ENV_TITLES.get(env, env), fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes2[r][c].axis("off")

    fig2.suptitle("Analysis 2: E(lambda=0.8) Debiasing Comparison (Q-PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis2_elambda_variants_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis2_elambda_variants_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_elam_mean = pd.DataFrame(elam_means, index=envs, columns=elam_cols)
    df_elam_std = pd.DataFrame(elam_stds, index=envs, columns=elam_cols)
    save_table_bundle(
        df_elam_mean, df_elam_std,
        "Table 2: E(lambda=0.8) Debiased vs Naive on Q-PPO (AUC Return)",
        os.path.join(tables_dir, "table2_elambda_variants"),
        cell_annotations=elam_annots,
    )

    # -------------------------------------------------------------
    # 3. ANALYSIS 3: Head-to-Head (E(0) Naive vs E(0.8) v_aux)
    # -------------------------------------------------------------
    naive_cols = ["E(0) Exact 1-Step", "E(0.8) v_aux (Debiased Jump)"]
    naive_means, naive_stds, naive_annots = [], [], {}

    fig3, axes3 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.5 * n_cols_plot, 3.5 * n_rows_plot), squeeze=False)
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes3[r][c]

        e0_run = select_best_lr_run(all_runs[env].get("E(0)_naive", {}), critic_lrs, metric_key)
        elam_run = select_best_lr_run(all_runs[env].get("E(0.8)_v_aux", {}), critic_lrs, metric_key)

        naive_means.append([e0_run["auc_mean"], elam_run["auc_mean"]])
        naive_stds.append([e0_run["auc_std"], elam_run["auc_std"]])
        naive_annots[(idx, 0)] = f"lr={e0_run['lr']}"
        naive_annots[(idx, 1)] = f"lr={elam_run['lr']}"

        t = np.arange(1, len(e0_run["curve"]) + 1) * batch_size
        ax.plot(t, e0_run["curve"], label=f"E(0) Exact (lr={e0_run['lr']})", color="#2563eb", lw=2.2)
        ax.fill_between(t, e0_run["curve"] - e0_run["std"], e0_run["curve"] + e0_run["std"], color="#2563eb", alpha=0.15)

        ax.plot(t, elam_run["curve"], label=f"E(0.8) v_aux (lr={elam_run['lr']})", color="#059669", lw=2.2)
        ax.fill_between(t, elam_run["curve"] - elam_run["std"], elam_run["curve"] + elam_run["std"], color="#059669", alpha=0.15)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes3[r][c].axis("off")

    fig3.suptitle("Analysis 3: Head-to-Head: E(0) Exact 1-Step vs E(0.8) v_aux (Q-PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis3_e0_vs_elambda_vaux.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis3_e0_vs_elambda_vaux.pdf"), bbox_inches="tight")
    plt.close()

    df_naive_mean = pd.DataFrame(naive_means, index=envs, columns=naive_cols)
    df_naive_std = pd.DataFrame(naive_stds, index=envs, columns=naive_cols)
    save_table_bundle(
        df_naive_mean, df_naive_std,
        "Table 3: Head-to-Head: E(0) Exact vs E(0.8) v_aux on Q-PPO (AUC Return)",
        os.path.join(tables_dir, "table3_e0_vs_elambda_vaux"),
        cell_annotations=naive_annots,
    )

    # -------------------------------------------------------------
    # 4. ANALYSIS 4: TD(lambda) Fitted SARSA Grid
    # -------------------------------------------------------------
    td_cols = [
        "SARSA(0)" if lam == 0.0 else (f"SARSA(λ={lam})" if lam < 1.0 else "MC (λ=1.0)")
        for lam in td_lambdas
    ]
    td_means, td_stds, td_annots = [], [], {}

    fig4, axes4 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.5 * n_cols_plot, 3.5 * n_rows_plot), squeeze=False)
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
            lbl = f"λ={lam} (lr={res['lr']})" if lam > 0 else f"SARSA(0) (lr={res['lr']})"
            ax.plot(t, res["curve"], label=lbl, color=td_colors[l_idx], lw=1.8)

        td_means.append(env_td_means)
        td_stds.append(env_td_stds)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes4[r][c].axis("off")

    fig4.suptitle("Analysis 4: TD(lambda) Fitted SARSA Grid Learning Curves (Q-PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis4_td_lambda_grid_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis4_td_lambda_grid_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_td_mean = pd.DataFrame(td_means, index=envs, columns=td_cols)
    df_td_std = pd.DataFrame(td_stds, index=envs, columns=td_cols)
    save_table_bundle(
        df_td_mean, df_td_std,
        "Table 4: TD(lambda) Fitted SARSA Grid on Q-PPO (AUC Return)",
        os.path.join(tables_dir, "table4_td_lambda_grid"),
        cell_annotations=td_annots,
    )

    # -------------------------------------------------------------
    # 5. ANALYSIS 5: Master Championship (Best TD vs Best E)
    # -------------------------------------------------------------
    champ_cols = ["Best TD / Fitted SARSA", "Best E-Minimization"]
    champ_means, champ_stds, champ_annots = [], [], {}

    fig5, axes5 = plt.subplots(n_rows_plot, n_cols_plot, figsize=(4.5 * n_cols_plot, 3.5 * n_rows_plot), squeeze=False)

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols_plot)
        ax = axes5[r][c]

        # 1. Best TD run
        best_td = None
        for lam in td_lambdas:
            cand = select_best_lr_run(all_runs[env].get("TD", {}).get(lam, {}), critic_lrs, metric_key)
            td_name = f"SARSA(λ={lam})" if lam > 0 else "SARSA(0)"
            if best_td is None or cand["auc_mean"] > best_td["auc_mean"]:
                best_td = {**cand, "name": td_name}

        # 2. Best E run
        best_e = None
        for e_key, clean_name in [
            ("E(0)_naive", "E(0) Exact"),
            ("E(0)_v_aux", "E(0) v_aux"),
            ("E(0)_fwd", "E(0) Fwd"),
            ("E(0.8)_v_aux", "E(0.8) v_aux"),
            ("E(0.8)_naive", "E(0.8) Naive"),
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

        ax.set_title(ENV_TITLES.get(env, env), fontsize=11.0, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.0)

    for idx in range(n_envs, n_rows_plot * n_cols_plot):
        r, c = divmod(idx, n_cols_plot)
        axes5[r][c].axis("off")

    fig5.suptitle("Analysis 5: Master Championship (Best TD vs Best E-Minimization) (Q-PPO)", fontsize=13.0, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "analysis5_best_e_vs_best_td_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "analysis5_best_e_vs_best_td_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    df_champ_mean = pd.DataFrame(champ_means, index=envs, columns=champ_cols)
    df_champ_std = pd.DataFrame(champ_stds, index=envs, columns=champ_cols)
    save_table_bundle(
        df_champ_mean, df_champ_std,
        "Table 5: Master Championship on Q-PPO (Overall Best TD vs Overall Best E-Minimization)",
        os.path.join(tables_dir, "table5_best_e_vs_best_td_championship"),
        cell_annotations=champ_annots,
    )

    print("\nAll 5 Cross-Algorithm Analyses, Plots, and Tables generated successfully!")


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
        env_dict.setdefault("E(0)_fwd", {})
        env_dict.setdefault("E(0.8)_v_aux", {})
        env_dict.setdefault("E(0.8)_naive", {})
        env_dict.setdefault("E(0.8)_fwd", {})
        env_dict.setdefault("TD", {lam: {} for lam in td_lambdas})

        for lam in td_lambdas:
            env_dict["TD"].setdefault(lam, {})

        base_cfg = {
            **base_default_config,
            "ENV_NAME": env_name,
            "ACTOR_LR": actor_lr,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": env_timesteps,
            "MINIBATCH_SIZE": minibatch_size,
            "ACTOR_MINIBATCH_SIZE": minibatch_size,
            "CRITIC_MINIBATCH_SIZE": minibatch_size,
            "NETWORK_TYPE": "cnn",
            "k": 64,
            "GAMMA": 0.99,
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
                        "CRITIC_LR_END": lr,
                        "CRITIC_TYPE": critic_type,
                        "E_VARIANT": e_var,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_name][lr] = res
                    final_ret = float(np.mean(res["returned_episode_returns"][:, -1]))
                    print(f"   -> Final Return: {final_ret:.2f}")

        # 2. E(lambda=0.8) Variants (Defaulting to v_aux debiasing!)
        for var_name, critic_type, e_var in [
            ("E(0.8)_v_aux", "e_lambda", "aux"),
            ("E(0.8)_naive", "e_lambda", "naive"),
            ("E(0.8)_fwd", "e_lambda", "fwd"),
        ]:
            for lr in critic_lrs:
                if lr not in env_dict[var_name]:
                    print(f"[{var_name}] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": lr,
                        "CRITIC_TYPE": critic_type,
                        "E_VARIANT": e_var,
                        "E_LAMBDA": 0.8,
                        "AUX_LAMBDA": 1.0,
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict[var_name][lr] = res
                    final_ret = float(np.mean(res["returned_episode_returns"][:, -1]))
                    print(f"   -> Final Return: {final_ret:.2f}")

        # 3. TD(lambda) Fitted SARSA Grid
        for lam in td_lambdas:
            for lr in critic_lrs:
                if lr not in env_dict["TD"][lam]:
                    c_type = "sarsa"
                    print(f"[TD(lambda={lam})] env={env_name}, critic_lr={lr}...")
                    cfg = {
                        **base_cfg,
                        "CRITIC_LR": lr,
                        "CRITIC_LR_END": lr,
                        "CRITIC_TYPE": c_type,
                        "VALUE_LAMBDA": lam,
                        "E_VARIANT": "naive",
                    }
                    res = run_experiment(cfg, seed_keys)
                    env_dict["TD"][lam][lr] = res
                    final_ret = float(np.mean(res["returned_episode_returns"][:, -1]))
                    print(f"   -> Final Return: {final_ret:.2f}")

    # Save all raw metrics
    save_path = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw metrics to {save_path}")

    # Generate all analyses
    generate_diagnostics_and_plots(all_runs, timesteps, out_dir, envs, critic_lrs, td_lambdas, metric_key, batch_size)


def main():
    parser = argparse.ArgumentParser(description="Run Q-PPO Comprehensive Critic Sweep on Jumanji")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, help="Jumanji Environment Names")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS, help="Critic learning rates")
    parser.add_argument("--actor_lr", type=float, default=0.0003, help="Fixed actor learning rate")
    parser.add_argument("--td_lambdas", nargs="+", type=float, default=DEFAULT_TD_LAMBDAS, help="TD lambdas to sweep")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override total timesteps per env")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout length (>=120 for full-episode Monte Carlo in Sokoban)")
    parser.add_argument("--minibatch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")
    parser.add_argument("--out_dir", type=str, default="results/sweep_q_ppo_critic_jumanji", help="Output directory")
    parser.add_argument("--load_existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

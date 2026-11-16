"""
scripts/sweep_standard_plotting.py

Shared canonical algorithm metadata, colors, summary tables, championship profiles,
default-vs-tuned learning curves, and pairwise comparison plots across all
Brax and Gymnax V-PPO / Q-PPO sweeps.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["axes.linewidth"] = 0.8

REAL_WIN_REL_TOL = 0.01

ALGO_REGISTRY: dict[str, dict[str, str]] = {
    "td_lambda": {
        "label": "TD(λ)",
        "color": "#2563eb",
        "ls": "-",
    },
    "sarsa_lambda": {
        "label": "SARSA(λ)",
        "color": "#2563eb",
        "ls": "-",
    },
    "e_0_v_aux": {
        "label": "E(0) (v_aux)",
        "color": "#059669",
        "ls": "-",
    },
    "e_0_bilinear": {
        "label": "E(0) (Bilinear, 2π)",
        "color": "#0d9488",
        "ls": "-.",
    },
    "e_lambda_v_aux": {
        "label": "E(λ) (v_aux)",
        "color": "#7c3aed",
        "ls": "-",
    },
    "bsmc_v_aux": {
        "label": "BSMC (v_aux)",
        "color": "#dc2626",
        "ls": "-.",
    },
    "bsmc_bilinear_v_aux": {
        "label": "BSMC (Bilinear, v_aux)",
        "color": "#dc2626",
        "ls": "-",
    },
    "bsmc_double_pi": {
        "label": "BSMC (2π)",
        "color": "#d97706",
        "ls": "-.",
    },
}

ENV_TITLES: dict[str, str] = {
    # Brax
    "hopper": "Hopper",
    "halfcheetah": "HalfCheetah",
    "walker2d": "Walker2d",
    "ant": "Ant",
    "humanoid": "Humanoid",
    "humanoidstandup": "HumanoidStandup",
    "inverted_pendulum": "Inverted Pendulum",
    "inverted_double_pendulum": "Inverted Double Pendulum",
    "pusher": "Pusher",
    "reacher": "Reacher",
    "swimmer": "Swimmer",
    # Gymnax
    "CartPole-v1": "CartPole-v1",
    "Pendulum-v1": "Pendulum-v1",
    "Acrobot-v1": "Acrobot-v1",
    "MountainCar-v0": "Mountain Car (Sparse)",
    "MountainCarContinuous-v0": "Mountain Car (Cont.)",
    "DeepSea-bsuite": "DeepSea (bsuite)",
    "DiscountingChain-bsuite": "Discounting Chain",
    "Catch-bsuite": "Catch (bsuite)",
    "FourRooms-misc": "FourRooms (Sparse)",
    "PointRobot-misc": "PointRobot (Cont.)",
    "Reacher-misc": "Reacher (Cont.)",
    "Swimmer-misc": "Swimmer (Cont.)",
    "Pong-misc": "Pong",
    "Asterix-MinAtar": "Asterix (MinAtar)",
    "Breakout-MinAtar": "Breakout (MinAtar)",
    "Freeway-MinAtar": "Freeway (MinAtar)",
    "SpaceInvaders-MinAtar": "Space Invaders (MinAtar)",
}

DEFAULT_PAIRWISE_V_PPO = [
    ("bsmc_v_aux", "td_lambda"),
    ("e_lambda_v_aux", "td_lambda"),
    ("e_lambda_v_aux", "bsmc_v_aux"),
    ("e_0_v_aux", "td_lambda"),
    ("e_0_v_aux", "bsmc_v_aux"),
    ("e_0_v_aux", "e_lambda_v_aux"),
]

DEFAULT_PAIRWISE_Q_PPO = [
    ("bsmc_bilinear_v_aux", "sarsa_lambda"),
    ("bsmc_double_pi", "sarsa_lambda"),
    ("e_lambda_v_aux", "sarsa_lambda"),
    ("e_0_v_aux", "sarsa_lambda"),
    ("e_0_bilinear", "sarsa_lambda"),
    ("e_0_v_aux", "e_0_bilinear"),
    ("bsmc_bilinear_v_aux", "bsmc_double_pi"),
    ("e_lambda_v_aux", "bsmc_bilinear_v_aux"),
    ("e_lambda_v_aux", "bsmc_double_pi"),
    ("e_0_v_aux", "bsmc_bilinear_v_aux"),
    ("e_0_bilinear", "bsmc_double_pi"),
]


def get_algo_meta(method_key: str) -> dict[str, str]:
    if method_key not in ALGO_REGISTRY:
        raise ValueError(f"Unrecognized canonical method_key '{method_key}'. Known: {list(ALGO_REGISTRY.keys())}")
    return ALGO_REGISTRY[method_key]


def compute_auc(returns_curve: np.ndarray) -> float:
    return float(np.mean(returns_curve))


def _format_val_std(val: float, std: float) -> str:
    if np.isnan(val):
        return "N/A"
    return f"{val:.1f} ± {std:.1f}"


def _format_val_std_tex(val: float, std: float, is_best: bool = False) -> str:
    if np.isnan(val):
        return "N/A"
    core = f"{val:.1f} \\pm {std:.1f}"
    return f"$\\mathbf{{{core}}}$" if is_best else f"${core}$"


def _row_real_win_tol(vals: np.ndarray, rel_tol: float = REAL_WIN_REL_TOL) -> float:
    valid = vals[~np.isnan(vals)]
    if valid.size == 0:
        return 0.0
    scale = max(float(np.max(np.abs(valid))), float(np.max(valid) - np.min(valid)), 1e-8)
    return rel_tol * scale


def _best_col_mask(vals: np.ndarray, higher_is_better: bool = True, rel_tol: float = REAL_WIN_REL_TOL) -> np.ndarray:
    mask = np.zeros(len(vals), dtype=bool)
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return mask
    best_val = np.nanmax(vals) if higher_is_better else np.nanmin(vals)
    tol = _row_real_win_tol(vals, rel_tol=rel_tol)
    for j in range(len(vals)):
        if valid[j] and abs(vals[j] - best_val) <= tol:
            mask[j] = True
    return mask


def _max_int_mask(vals: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(vals), dtype=bool)
    if len(vals) == 0:
        return mask
    best = np.max(vals)
    return vals == best


def _compute_wtl_stats(
    mat_mean: np.ndarray, higher_is_better: bool = True, rel_tol: float = REAL_WIN_REL_TOL
) -> dict[str, np.ndarray]:
    n_rows, n_cols = mat_mean.shape
    top_w = np.zeros(n_cols, dtype=int)
    top_t = np.zeros(n_cols, dtype=int)
    top_l = np.zeros(n_cols, dtype=int)
    pw_w = np.zeros(n_cols, dtype=int)
    pw_t = np.zeros(n_cols, dtype=int)
    pw_l = np.zeros(n_cols, dtype=int)

    for i in range(n_rows):
        row = mat_mean[i]
        tol = _row_real_win_tol(row, rel_tol=rel_tol)
        best_mask = _best_col_mask(row, higher_is_better=higher_is_better, rel_tol=rel_tol)
        n_best = int(np.sum(best_mask))
        for j in range(n_cols):
            if np.isnan(row[j]):
                continue
            if best_mask[j]:
                if n_best == 1:
                    top_w[j] += 1
                else:
                    top_t[j] += 1
            else:
                top_l[j] += 1

        for j in range(n_cols):
            if np.isnan(row[j]):
                continue
            for k in range(n_cols):
                if j == k or np.isnan(row[k]):
                    continue
                diff = (row[j] - row[k]) if higher_is_better else (row[k] - row[j])
                if abs(diff) <= tol:
                    pw_t[j] += 1
                elif diff > tol:
                    pw_w[j] += 1
                else:
                    pw_l[j] += 1

    return {
        "top_w": top_w,
        "top_t": top_t,
        "top_l": top_l,
        "pw_w": pw_w,
        "pw_t": pw_t,
        "pw_l": pw_l,
    }


def save_table_bundle(
    df_mean: pd.DataFrame,
    df_std: pd.DataFrame,
    title: str,
    base_path: str,
    cell_annotations: dict[tuple[int, int], str] | None = None,
    higher_is_better: bool = True,
    rel_tol: float = REAL_WIN_REL_TOL,
) -> None:
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    envs = list(df_mean.index)
    cols = list(df_mean.columns)
    mat_m = df_mean.values
    mat_s = df_std.values
    wtl = _compute_wtl_stats(mat_m, higher_is_better=higher_is_better, rel_tol=rel_tol)

    # 1. Markdown
    md_lines = [f"### {title}\n"]
    header = "| Environment | " + " | ".join(cols) + " |"
    sep = "|:---| " + " | ".join([":---:"] * len(cols)) + " |"
    md_lines.extend([header, sep])
    for i, env in enumerate(envs):
        best_mask = _best_col_mask(mat_m[i], higher_is_better=higher_is_better, rel_tol=rel_tol)
        cells = []
        for j in range(len(cols)):
            txt = _format_val_std(mat_m[i, j], mat_s[i, j])
            if best_mask[j] and txt != "N/A":
                txt = f"**{txt}**"
            if cell_annotations and (i, j) in cell_annotations:
                txt += f" <br><sub>({cell_annotations[(i, j)]})</sub>"
            cells.append(txt)
        md_lines.append(f"| **{ENV_TITLES.get(env, env)}** | " + " | ".join(cells) + " |")

    top_best = _max_int_mask(wtl["top_w"])
    pw_best = _max_int_mask(wtl["pw_w"])
    top_cells = [
        f"**{wtl['top_w'][j]} / {wtl['top_t'][j]} / {wtl['top_l'][j]}**" if top_best[j]
        else f"{wtl['top_w'][j]} / {wtl['top_t'][j]} / {wtl['top_l'][j]}"
        for j in range(len(cols))
    ]
    pw_cells = [
        f"**{wtl['pw_w'][j]} / {wtl['pw_t'][j]} / {wtl['pw_l'][j]}**" if pw_best[j]
        else f"{wtl['pw_w'][j]} / {wtl['pw_t'][j]} / {wtl['pw_l'][j]}"
        for j in range(len(cols))
    ]
    md_lines.append("|---" * (len(cols) + 1) + "|")
    md_lines.append("| **Top-1 (W / T / L)** | " + " | ".join(top_cells) + " |")
    md_lines.append("| **Pairwise (W / T / L)** | " + " | ".join(pw_cells) + " |")
    with open(base_path + ".md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # 2. LaTeX
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{title}}}",
        f"\\label{{{os.path.basename(base_path)}}}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Environment & " + " & ".join(c.replace("_", r"\_").replace("λ", r"$\lambda$").replace("π", r"$\pi$") for c in cols) + r" \\",
        r"\midrule",
    ]
    for i, env in enumerate(envs):
        best_mask = _best_col_mask(mat_m[i], higher_is_better=higher_is_better, rel_tol=rel_tol)
        cells = [_format_val_std_tex(mat_m[i, j], mat_s[i, j], is_best=bool(best_mask[j])) for j in range(len(cols))]
        env_tex = ENV_TITLES.get(env, env).replace("_", r"\_")
        tex_lines.append(f"{env_tex} & " + " & ".join(cells) + r" \\")
    tex_lines.append(r"\midrule")
    top_tex = [
        f"\\textbf{{{wtl['top_w'][j]} / {wtl['top_t'][j]} / {wtl['top_l'][j]}}}" if top_best[j]
        else f"{wtl['top_w'][j]} / {wtl['top_t'][j]} / {wtl['top_l'][j]}"
        for j in range(len(cols))
    ]
    pw_tex = [
        f"\\textbf{{{wtl['pw_w'][j]} / {wtl['pw_t'][j]} / {wtl['pw_l'][j]}}}" if pw_best[j]
        else f"{wtl['pw_w'][j]} / {wtl['pw_t'][j]} / {wtl['pw_l'][j]}"
        for j in range(len(cols))
    ]
    tex_lines.append("Top-1 (W / T / L) & " + " & ".join(top_tex) + r" \\")
    tex_lines.append("Pairwise (W / T / L) & " + " & ".join(pw_tex) + r" \\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    with open(base_path + ".tex", "w") as f:
        f.write("\n".join(tex_lines) + "\n")

    # 3. PNG + PDF table image
    n_rows = len(envs) + 3
    fig_w = max(8.5, 2.1 + 1.9 * len(cols))
    fig_h = max(3.5, 1.4 + 0.45 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_text = []
    cell_colors = []
    for i, env in enumerate(envs):
        best_mask = _best_col_mask(mat_m[i], higher_is_better=higher_is_better, rel_tol=rel_tol)
        row_t = [ENV_TITLES.get(env, env)]
        row_c = ["#f8fafc"]
        for j in range(len(cols)):
            txt = _format_val_std(mat_m[i, j], mat_s[i, j])
            if cell_annotations and (i, j) in cell_annotations:
                txt += f"\n({cell_annotations[(i, j)]})"
            row_t.append(txt)
            row_c.append("#dcfce7" if best_mask[j] else "#ffffff")
        cell_text.append(row_t)
        cell_colors.append(row_c)

    cell_text.append(["Top-1 (W/T/L)"] + [f"{wtl['top_w'][j]} / {wtl['top_t'][j]} / {wtl['top_l'][j]}" for j in range(len(cols))])
    cell_colors.append(["#f1f5f9"] + ["#dbeafe" if top_best[j] else "#f8fafc" for j in range(len(cols))])
    cell_text.append(["Pairwise (W/T/L)"] + [f"{wtl['pw_w'][j]} / {wtl['pw_t'][j]} / {wtl['pw_l'][j]}" for j in range(len(cols))])
    cell_colors.append(["#f1f5f9"] + ["#dbeafe" if pw_best[j] else "#f8fafc" for j in range(len(cols))])

    tbl = ax.table(
        cellText=cell_text,
        colLabels=["Environment"] + cols,
        cellColours=cell_colors,
        colColours=["#e2e8f0"] * (len(cols) + 1),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)
    tbl.scale(1.0, 1.6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(base_path + ".png", bbox_inches="tight", dpi=250)
    plt.savefig(base_path + ".pdf", bbox_inches="tight")
    plt.close()


def _format_tag_short(params: dict[str, Any]) -> str:
    parts = []
    if "CRITIC_LR" in params:
        parts.append(f"lr={params['CRITIC_LR']}")
    if "CRITIC_EPOCHS" in params:
        parts.append(f"ep={params['CRITIC_EPOCHS']}")
    for lk in ("VALUE_LAMBDA", "E_LAMBDA", "BSMC_LAMBDA"):
        if lk in params:
            parts.append(f"λ={params[lk]}")
            break
    return ", ".join(parts) if parts else "tuned"


def _compute_global_shared_winners(
    all_runs: dict,
    envs: list[str],
    method_keys: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Selects best shared hyperparameter tag per method across all environments."""
    best_tags: dict[str, str] = {}
    best_info: dict[str, dict[str, Any]] = {}

    for m_key in method_keys:
        common_tags = None
        for env in envs:
            m_runs = all_runs.get(env, {}).get(m_key, {})
            tags = set(m_runs.keys())
            common_tags = tags if common_tags is None else (common_tags & tags)
        if not common_tags:
            continue

        rows = []
        for tag in sorted(common_tags):
            row = {"tag": tag, "params": all_runs[envs[0]][m_key][tag].get("params", {})}
            for env in envs:
                r = all_runs[env][m_key][tag]
                row[f"{env}__auc"] = r["auc_mean"]
                row[f"{env}__final"] = r["final_mean"]
            rows.append(row)

        df = pd.DataFrame(rows)
        norm_aucs, norm_finals = [], []
        for env in envs:
            a_col, f_col = f"{env}__auc", f"{env}__final"
            a_min, a_max = df[a_col].min(), df[a_col].max()
            f_min, f_max = df[f_col].min(), df[f_col].max()
            na, nf = f"{env}__na", f"{env}__nf"
            df[na] = (df[a_col] - a_min) / max(a_max - a_min, 1e-8)
            df[nf] = (df[f_col] - f_min) / max(f_max - f_min, 1e-8)
            norm_aucs.append(na)
            norm_finals.append(nf)

        df["mean_norm_auc"] = df[norm_aucs].mean(axis=1)
        df["mean_norm_final"] = df[norm_finals].mean(axis=1)
        df["combined_score"] = 0.6 * df["mean_norm_auc"] + 0.4 * df["mean_norm_final"]
        df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        top = df.iloc[0]
        top_tag = str(top["tag"])
        best_tags[m_key] = top_tag
        best_info[m_key] = {
            "label": get_algo_meta(m_key)["label"],
            "best_tag": top_tag,
            "combined_score": float(top["combined_score"]),
            "mean_norm_auc": float(top["mean_norm_auc"]),
            "mean_norm_final": float(top["mean_norm_final"]),
            "params": top["params"],
        }
    return best_tags, best_info


def _plot_learning_curve_grid(
    all_runs: dict,
    envs: list[str],
    method_keys: list[str],
    per_env_best_tags: dict[tuple[str, str], str],
    batch_sizes: dict[str, int],
    title: str,
    out_base: str,
    include_default: bool = True,
) -> None:
    n_envs = len(envs)
    n_cols = min(3, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.3 * n_rows), squeeze=False)

    for r_idx, env in enumerate(envs):
        r, c = divmod(r_idx, n_cols)
        ax = axes[r][c]
        bs = batch_sizes[env]

        for m_key in method_keys:
            m_runs = all_runs[env].get(m_key, {})
            if not m_runs:
                continue
            meta = get_algo_meta(m_key)
            color = meta["color"]
            label = meta["label"]

            if include_default and "default" in m_runs:
                d_res = m_runs["default"]
                d_ret = np.asarray(d_res["returned_episode_returns"])
                d_mean = d_ret.mean(axis=0)
                d_sem = d_ret.std(axis=0) / np.sqrt(max(1, d_ret.shape[0]))
                t_d = np.arange(1, len(d_mean) + 1) * bs
                ax.plot(t_d, d_mean, label=f"{label} (Default)", color=color, ls="--", lw=1.5, alpha=0.85)
                ax.fill_between(t_d, d_mean - d_sem, d_mean + d_sem, color=color, alpha=0.07)

            t_tag = per_env_best_tags.get((env, m_key))
            if t_tag and t_tag in m_runs:
                t_res = m_runs[t_tag]
                t_ret = np.asarray(t_res["returned_episode_returns"])
                t_mean = t_ret.mean(axis=0)
                t_sem = t_ret.std(axis=0) / np.sqrt(max(1, t_ret.shape[0]))
                t_t = np.arange(1, len(t_mean) + 1) * bs
                lbl = f"{label} (Tuned)" if include_default else label
                ax.plot(t_t, t_mean, label=lbl, color=color, ls="-", lw=2.1)
                ax.fill_between(t_t, t_mean - t_sem, t_mean + t_sem, color=color, alpha=0.16)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps", fontsize=9)
        ax.set_ylabel("Episode Return", fontsize=9)
        ax.grid(True, alpha=0.3)
        if r_idx == 0:
            ax.legend(loc="best", fontsize=7.0, ncol=1 if len(method_keys) <= 4 else 2)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_base + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close()


def _plot_championship_profile(
    df_tuned_auc: pd.DataFrame,
    df_tuned_fin: pd.DataFrame,
    method_keys: list[str],
    reference_method: str,
    suite_title: str,
    plots_dir: str,
    tables_dir: str,
) -> None:
    envs = list(df_tuned_auc.index)
    n_envs = len(envs)
    labels = [get_algo_meta(m)["label"] for m in method_keys]
    colors = [get_algo_meta(m)["color"] for m in method_keys]
    linestyles = [get_algo_meta(m)["ls"] for m in method_keys]

    ref_label = get_algo_meta(reference_method)["label"] if reference_method in method_keys else labels[0]
    ref_vals = df_tuned_auc[ref_label].values

    norm_scores = {}
    for lbl in labels:
        raw_vals = df_tuned_auc[lbl].values
        norm = np.where(
            np.abs(ref_vals) > 1e-4,
            100.0 + ((raw_vals - ref_vals) / np.maximum(np.abs(ref_vals), 1e-4)) * 100.0,
            100.0,
        )
        norm_scores[lbl] = norm
    df_norm = pd.DataFrame(norm_scores, index=envs)

    mean_norm_scores, iqm_scores, median_scores = {}, {}, {}
    for lbl in labels:
        vals = df_norm[lbl].values
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            q25, q75 = np.percentile(valid, [25, 75])
            iqm_mask = (valid >= q25) & (valid <= q75)
            iqm_scores[lbl] = float(np.mean(valid[iqm_mask])) if np.any(iqm_mask) else float(np.mean(valid))
            median_scores[lbl] = float(np.median(valid))
            mean_norm_scores[lbl] = float(np.mean(valid))
        else:
            iqm_scores[lbl] = np.nan
            median_scores[lbl] = np.nan
            mean_norm_scores[lbl] = np.nan

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2))
    x_idx = np.arange(len(labels))
    bar_w = 0.35

    ax_a.bar(x_idx - bar_w / 2, [mean_norm_scores[l] for l in labels], width=bar_w, label="Mean Normalized Score", color=colors, alpha=0.88)
    ax_a.bar(x_idx + bar_w / 2, [iqm_scores[l] for l in labels], width=bar_w, label="Interquartile Mean (IQM)", color=colors, alpha=0.45, hatch="//")
    ax_a.axhline(100.0, color="gray", linestyle="--", lw=1.2, label=f"Baseline ({ref_label} = 100%)")
    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(labels, rotation=18, ha="right", fontsize=9.0, fontweight="bold")
    ax_a.set_ylabel(f"Normalized AUC Return (% of {ref_label})", fontsize=10)
    ax_a.set_title(f"{suite_title}: Aggregate Score ({n_envs} Envs)", fontsize=11, fontweight="bold")
    ax_a.grid(True, alpha=0.3, axis="y")
    ax_a.legend(loc="upper left", fontsize=8.0)

    all_vals = df_norm.values[~np.isnan(df_norm.values)]
    tau_min = float(min(50.0, np.percentile(all_vals, 5))) if len(all_vals) else 50.0
    tau_max = float(max(150.0, np.percentile(all_vals, 95))) if len(all_vals) else 150.0
    tau_vals = np.linspace(tau_min, tau_max, 120)
    for c_idx, lbl in enumerate(labels):
        vals = df_norm[lbl].values
        cdf = [np.mean(vals >= tau) for tau in tau_vals]
        ax_b.plot(tau_vals, cdf, label=lbl, color=colors[c_idx], lw=2.1, ls=linestyles[c_idx])

    ax_b.axvline(100.0, color="gray", linestyle="--", lw=1.0)
    ax_b.set_xlabel("Normalized Score Threshold (τ %)", fontsize=10)
    ax_b.set_ylabel("Fraction of Envs with Score ≥ τ", fontsize=10)
    ax_b.set_title(f"{suite_title}: Performance Profile ({n_envs} Envs)", fontsize=11, fontweight="bold")
    ax_b.grid(True, alpha=0.3)
    ax_b.legend(loc="lower left", fontsize=8.0)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "championship_profile.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "championship_profile.pdf"), bbox_inches="tight")
    plt.close()

    wtl_auc = _compute_wtl_stats(df_tuned_auc[labels].values, higher_is_better=True)
    wtl_fin = _compute_wtl_stats(df_tuned_fin[labels].values, higher_is_better=True)

    rows = []
    for c_idx, lbl in enumerate(labels):
        rows.append({
            "Algorithm": lbl,
            "Mean Norm (%)": f"{mean_norm_scores[lbl]:.1f}%",
            "IQM (%)": f"{iqm_scores[lbl]:.1f}%",
            "Median (%)": f"{median_scores[lbl]:.1f}%",
            "AUC Top-1": f"{wtl_auc['top_w'][c_idx]}W / {wtl_auc['top_t'][c_idx]}T / {wtl_auc['top_l'][c_idx]}L",
            "AUC Pairwise": f"{wtl_auc['pw_w'][c_idx]} / {wtl_auc['pw_t'][c_idx]} / {wtl_auc['pw_l'][c_idx]}",
            "Final Pairwise": f"{wtl_fin['pw_w'][c_idx]} / {wtl_fin['pw_t'][c_idx]} / {wtl_fin['pw_l'][c_idx]}",
        })
    df_champ = pd.DataFrame(rows)
    base_tbl = os.path.join(tables_dir, "table3_championship_summary")
    with open(base_tbl + ".md", "w") as f:
        f.write(f"### {suite_title}: Championship Summary ({n_envs} Environments, Tuned)\n\n")
        f.write(df_champ.to_markdown(index=False) + "\n")

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{suite_title}: Championship Summary ({n_envs} Environments, Tuned)}}",
        r"\label{table3_championship_summary}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Algorithm & Mean Norm (\%) & IQM (\%) & Median (\%) & AUC Top-1 & AUC Pairwise & Final Pairwise \\",
        r"\midrule",
    ]
    for r in rows:
        alg_tex = r["Algorithm"].replace("_", r"\_").replace("λ", r"$\lambda$").replace("π", r"$\pi$")
        mn_tex = r["Mean Norm (%)"].replace("%", r"\%")
        iqm_tex = r["IQM (%)"].replace("%", r"\%")
        med_tex = r["Median (%)"].replace("%", r"\%")
        tex_lines.append(
            f"{alg_tex} & {mn_tex} & {iqm_tex} & {med_tex} & "
            f"{r['AUC Top-1']} & {r['AUC Pairwise']} & {r['Final Pairwise']} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    with open(base_tbl + ".tex", "w") as f:
        f.write("\n".join(tex_lines) + "\n")


def generate_standardized_suite(
    all_runs: dict,
    method_keys: list[str],
    timesteps: dict[str, int],
    out_dir: str,
    suite_title: str,
    batch_sizes: dict[str, int],
    base_config: dict[str, Any],
    reference_method: str,
    pairwise_pairs: list[tuple[str, str]] | None = None,
    all_pairs: bool = False,
) -> None:
    """
    Produces standardized data/, tables/, and plots/ (default vs tuned learning curves,
    championship profile, and pairwise comparisons) from consolidated all_runs[env][method][tag].
    """
    plots_dir = os.path.join(out_dir, "plots")
    pairwise_dir = os.path.join(plots_dir, "pairwise")
    per_env_dir = os.path.join(plots_dir, "per_env")
    tables_dir = os.path.join(out_dir, "tables")
    data_dir = os.path.join(out_dir, "data")
    for d in (plots_dir, pairwise_dir, per_env_dir, tables_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    envs = sorted(all_runs.keys())
    global_tags, global_info = _compute_global_shared_winners(all_runs, envs, method_keys)

    per_env_best_tags: dict[tuple[str, str], str] = {}
    per_env_best_info: dict[str, dict[str, Any]] = {}
    summary_rows = []

    for env in envs:
        per_env_best_info[env] = {}
        for m_key in method_keys:
            m_runs = all_runs[env].get(m_key, {})
            if not m_runs:
                continue
            best_tag = max(m_runs.keys(), key=lambda t: m_runs[t]["auc_mean"])
            per_env_best_tags[(env, m_key)] = best_tag
            r_best = m_runs[best_tag]
            per_env_best_info[env][m_key] = {
                "best_tag": best_tag,
                "auc_mean": float(r_best["auc_mean"]),
                "auc_std": float(r_best["auc_std"]),
                "final_mean": float(r_best["final_mean"]),
                "final_std": float(r_best["final_std"]),
                "params": r_best.get("params", {}),
            }
            for tag, r in m_runs.items():
                summary_rows.append({
                    "env": env,
                    "method": m_key,
                    "method_label": get_algo_meta(m_key)["label"],
                    "tag": tag,
                    "is_default": tag == "default",
                    "is_best_per_env": tag == best_tag,
                    "auc_mean": float(r["auc_mean"]),
                    "auc_std": float(r["auc_std"]),
                    "final_mean": float(r["final_mean"]),
                    "final_std": float(r["final_std"]),
                    "elapsed_time": float(r.get("elapsed_time", 0.0)),
                })

    pd.DataFrame(summary_rows).to_csv(os.path.join(data_dir, "summary_returns.csv"), index=False)
    with open(os.path.join(data_dir, "best_configs.json"), "w") as f:
        json.dump({"global_shared_best": global_info, "per_env_best": per_env_best_info}, f, indent=2, default=str)

    # Build Default vs Tuned AUC and Final tables
    col_names, col_specs = [], []
    for m_key in method_keys:
        lbl = get_algo_meta(m_key)["label"]
        col_names.append(f"{lbl} (Default)")
        col_specs.append((m_key, "default"))
        col_names.append(f"{lbl} (Tuned)")
        col_specs.append((m_key, "tuned"))

    auc_m, auc_s, auc_ann = [], [], {}
    fin_m, fin_s, fin_ann = [], [], {}
    tuned_auc_only, tuned_fin_only = [], []
    tuned_labels = [get_algo_meta(m)["label"] for m in method_keys]

    for r_idx, env in enumerate(envs):
        r_am, r_as, r_fm, r_fs = [], [], [], []
        r_ta, r_tf = [], []
        for c_idx, (m_key, kind) in enumerate(col_specs):
            m_runs = all_runs[env].get(m_key, {})
            tag = "default" if kind == "default" else per_env_best_tags.get((env, m_key))
            if tag and tag in m_runs:
                res = m_runs[tag]
                r_am.append(res["auc_mean"])
                r_as.append(res["auc_std"])
                r_fm.append(res["final_mean"])
                r_fs.append(res["final_std"])
                ann = "default" if kind == "default" else _format_tag_short(res.get("params", {}))
                auc_ann[(r_idx, c_idx)] = ann
                fin_ann[(r_idx, c_idx)] = ann
                if kind == "tuned":
                    r_ta.append(res["auc_mean"])
                    r_tf.append(res["final_mean"])
            else:
                r_am.append(np.nan)
                r_as.append(np.nan)
                r_fm.append(np.nan)
                r_fs.append(np.nan)
                if kind == "tuned":
                    r_ta.append(np.nan)
                    r_tf.append(np.nan)
        auc_m.append(r_am)
        auc_s.append(r_as)
        fin_m.append(r_fm)
        fin_s.append(r_fs)
        tuned_auc_only.append(r_ta)
        tuned_fin_only.append(r_tf)

    save_table_bundle(
        pd.DataFrame(auc_m, index=envs, columns=col_names),
        pd.DataFrame(auc_s, index=envs, columns=col_names),
        f"{suite_title} — AUC Return (Default vs. Tuned)",
        os.path.join(tables_dir, "table1_auc_default_vs_tuned"),
        cell_annotations=auc_ann,
    )
    save_table_bundle(
        pd.DataFrame(fin_m, index=envs, columns=col_names),
        pd.DataFrame(fin_s, index=envs, columns=col_names),
        f"{suite_title} — Final Return (Default vs. Tuned)",
        os.path.join(tables_dir, "table2_final_default_vs_tuned"),
        cell_annotations=fin_ann,
    )

    # Championship plot + Table 3
    df_tuned_auc = pd.DataFrame(tuned_auc_only, index=envs, columns=tuned_labels)
    df_tuned_fin = pd.DataFrame(tuned_fin_only, index=envs, columns=tuned_labels)
    _plot_championship_profile(
        df_tuned_auc=df_tuned_auc,
        df_tuned_fin=df_tuned_fin,
        method_keys=method_keys,
        reference_method=reference_method,
        suite_title=suite_title,
        plots_dir=plots_dir,
        tables_dir=tables_dir,
    )

    # Config & winners summary markdown
    with open(os.path.join(tables_dir, "sweep_config_and_winners.md"), "w") as f:
        f.write(f"### {suite_title}: Base Config & Sweep Grid\n\n")
        for k, v in base_config.items():
            f.write(f"- **{k}:** {v}\n")
        if timesteps:
            ts_str = ", ".join(f"{ENV_TITLES.get(e, e)}: {timesteps[e]:,}" for e in envs if e in timesteps)
            f.write(f"- **Timesteps:** {ts_str}\n")
        f.write("\n### Winning Settings per (Environment, Algorithm) [Selected by Mean AUC]\n\n")
        win_rows = []
        for env in envs:
            row = {"Environment": ENV_TITLES.get(env, env)}
            for m_key in method_keys:
                lbl = get_algo_meta(m_key)["label"]
                tag = per_env_best_tags.get((env, m_key))
                if tag and tag in all_runs[env].get(m_key, {}):
                    r = all_runs[env][m_key][tag]
                    row[lbl] = f"{_format_tag_short(r.get('params', {}))} (Fin: {r['final_mean']:.1f}, AUC: {r['auc_mean']:.1f})"
                else:
                    row[lbl] = "N/A"
            win_rows.append(row)
        f.write(pd.DataFrame(win_rows).to_markdown(index=False) + "\n")

    # All-method learning curves (Default vs Tuned, and Tuned-only)
    _plot_learning_curve_grid(
        all_runs, envs, method_keys, per_env_best_tags, batch_sizes,
        f"{suite_title} — Default (Dashed) vs. Tuned (Solid)",
        os.path.join(plots_dir, "learning_curves_default_vs_tuned_all"),
        include_default=True,
    )
    _plot_learning_curve_grid(
        all_runs, envs, method_keys, per_env_best_tags, batch_sizes,
        f"{suite_title} — Best Tuned per Environment",
        os.path.join(plots_dir, "learning_curves_tuned_all"),
        include_default=False,
    )

    # Per-environment learning curve plots
    for env in envs:
        _plot_learning_curve_grid(
            all_runs, [env], method_keys, per_env_best_tags, batch_sizes,
            f"{suite_title}: {ENV_TITLES.get(env, env)} (Default vs. Tuned)",
            os.path.join(per_env_dir, f"learning_curves_{env}"),
            include_default=True,
        )

    # Pairwise learning curves
    if all_pairs:
        pairs_to_plot = list(itertools.combinations(method_keys, 2))
    elif pairwise_pairs is not None:
        pairs_to_plot = [(a, b) for a, b in pairwise_pairs if a in method_keys and b in method_keys]
    else:
        pairs_to_plot = list(itertools.combinations(method_keys, 2))

    for a_key, b_key in pairs_to_plot:
        a_lbl = get_algo_meta(a_key)["label"]
        b_lbl = get_algo_meta(b_key)["label"]
        _plot_learning_curve_grid(
            all_runs, envs, [a_key, b_key], per_env_best_tags, batch_sizes,
            f"{suite_title}: {a_lbl} vs. {b_lbl} (Default & Tuned)",
            os.path.join(pairwise_dir, f"{a_key}_vs_{b_key}_default_vs_tuned"),
            include_default=True,
        )

    # Q1 & Q2: Focused sampling comparison tables (Q-PPO)
    if "e_0_v_aux" in method_keys and "e_0_bilinear" in method_keys:
        _save_pairwise_comparison_table(
            all_runs, envs, ["e_0_v_aux", "e_0_bilinear"], per_env_best_tags,
            f"{suite_title}: Q1 — E(0) Sampling (v_aux vs. Bilinear 2π, Tuned AUC)",
            os.path.join(tables_dir, "q1_e0_sampling"),
        )
    if "bsmc_bilinear_v_aux" in method_keys and "bsmc_double_pi" in method_keys:
        _save_pairwise_comparison_table(
            all_runs, envs, ["bsmc_bilinear_v_aux", "bsmc_double_pi"], per_env_best_tags,
            f"{suite_title}: Q2 — BSMC Sampling (Bilinear v_aux vs. Double-Sample 2π, Tuned AUC)",
            os.path.join(tables_dir, "q2_bsmc_sampling"),
        )

    # Q3: E(lambda) vs BSMC (v_aux) at matched MC weight (same lambda)
    bsmc_v_key = "bsmc_v_aux" if "bsmc_v_aux" in method_keys else (
        "bsmc_bilinear_v_aux" if "bsmc_bilinear_v_aux" in method_keys else None
    )
    if "e_lambda_v_aux" in method_keys and bsmc_v_key is not None:
        _generate_q3_matched_lambda_comparison(
            all_runs=all_runs,
            envs=envs,
            e_key="e_lambda_v_aux",
            bsmc_key=bsmc_v_key,
            suite_title=suite_title,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
        )

    # Q5: Best E variant (including BSMC) vs TD(lambda) / SARSA(lambda)
    e_family = [m for m in method_keys if m != reference_method]
    if reference_method in method_keys and e_family:
        _generate_q5_best_e_vs_td(
            all_runs=all_runs,
            envs=envs,
            ref_key=reference_method,
            e_keys=e_family,
            per_env_best_tags=per_env_best_tags,
            batch_sizes=batch_sizes,
            suite_title=suite_title,
            plots_dir=plots_dir,
            tables_dir=tables_dir,
        )


def _save_pairwise_comparison_table(
    all_runs: dict,
    envs: list[str],
    pair_keys: list[str],
    per_env_best_tags: dict[tuple[str, str], str],
    title: str,
    base_path: str,
) -> None:
    cols = [get_algo_meta(k)["label"] for k in pair_keys]
    m_rows, s_rows, ann = [], [], {}
    for r_idx, env in enumerate(envs):
        rm, rs = [], []
        for c_idx, k in enumerate(pair_keys):
            tag = per_env_best_tags.get((env, k))
            if tag and tag in all_runs[env].get(k, {}):
                res = all_runs[env][k][tag]
                rm.append(res["auc_mean"])
                rs.append(res["auc_std"])
                ann[(r_idx, c_idx)] = _format_tag_short(res.get("params", {}))
            else:
                rm.append(np.nan)
                rs.append(np.nan)
        m_rows.append(rm)
        s_rows.append(rs)
    save_table_bundle(pd.DataFrame(m_rows, index=envs, columns=cols), pd.DataFrame(s_rows, index=envs, columns=cols), title, base_path, cell_annotations=ann)


def _generate_q3_matched_lambda_comparison(
    all_runs: dict,
    envs: list[str],
    e_key: str,
    bsmc_key: str,
    suite_title: str,
    plots_dir: str,
    tables_dir: str,
) -> None:
    """Q3: Compare E(lambda) vs BSMC (v_aux) at identical lambda (same MC anchor weight)."""
    lambdas_set = set()
    for env in envs:
        for k in (e_key, bsmc_key):
            for res in all_runs[env].get(k, {}).values():
                p = res.get("params", {})
                for lk in ("E_LAMBDA", "BSMC_LAMBDA"):
                    if lk in p:
                        lambdas_set.add(float(p[lk]))
    lambdas = sorted(lambdas_set)
    if not lambdas:
        return

    e_lbl = get_algo_meta(e_key)["label"]
    b_lbl = get_algo_meta(bsmc_key)["label"]
    cols = []
    for lmb in lambdas:
        cols.append(f"{e_lbl} (λ={lmb})")
        cols.append(f"{b_lbl} (λ={lmb})")

    m_rows, s_rows = [], []
    n_envs = len(envs)
    n_cols = min(3, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.2 * n_rows), squeeze=False)

    for r_idx, env in enumerate(envs):
        rm, rs = [], []
        e_curve_m, e_curve_s = [], []
        b_curve_m, b_curve_s = [], []
        for lmb in lambdas:
            for k, lk, cm, cs in [
                (e_key, "E_LAMBDA", e_curve_m, e_curve_s),
                (bsmc_key, "BSMC_LAMBDA", b_curve_m, b_curve_s),
            ]:
                cands = [
                    r for r in all_runs[env].get(k, {}).values()
                    if abs(float(r.get("params", {}).get(lk, -1.0)) - lmb) < 1e-6
                ]
                if cands:
                    best = max(cands, key=lambda r: r["auc_mean"])
                    rm.append(best["auc_mean"])
                    rs.append(best["auc_std"])
                    cm.append(best["auc_mean"])
                    cs.append(best["auc_std"])
                else:
                    rm.append(np.nan)
                    rs.append(np.nan)
                    cm.append(np.nan)
                    cs.append(np.nan)
        m_rows.append(rm)
        s_rows.append(rs)

        r, c = divmod(r_idx, n_cols)
        ax = axes[r][c]
        ax.errorbar(lambdas, e_curve_m, yerr=e_curve_s, marker="o", lw=2.0, color=get_algo_meta(e_key)["color"], label=e_lbl, capsize=3)
        ax.errorbar(lambdas, b_curve_m, yerr=b_curve_s, marker="s", lw=2.0, ls="--", color=get_algo_meta(bsmc_key)["color"], label=b_lbl, capsize=3)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("λ (Matched MC Weight)")
        ax.set_ylabel("AUC Return")
        ax.set_xticks(lambdas)
        ax.grid(True, alpha=0.3)
        if r_idx == 0:
            ax.legend(loc="best", fontsize=8.0)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{suite_title}: Q3 — {e_lbl} vs. {b_lbl} at Matched MC Weight (λ)", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "q3_elambda_vs_bsmc_matched_lambda.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "q3_elambda_vs_bsmc_matched_lambda.pdf"), bbox_inches="tight")
    plt.close()

    save_table_bundle(
        pd.DataFrame(m_rows, index=envs, columns=cols),
        pd.DataFrame(s_rows, index=envs, columns=cols),
        f"{suite_title}: Q3 — {e_lbl} vs. {b_lbl} at Matched MC Weight (AUC Return)",
        os.path.join(tables_dir, "q3_elambda_vs_bsmc_matched_lambda"),
    )


def _generate_q5_best_e_vs_td(
    all_runs: dict,
    envs: list[str],
    ref_key: str,
    e_keys: list[str],
    per_env_best_tags: dict[tuple[str, str], str],
    batch_sizes: dict[str, int],
    suite_title: str,
    plots_dir: str,
    tables_dir: str,
) -> None:
    """Q5: Best E-family variant (including BSMC) vs TD(lambda) / SARSA(lambda)."""
    ref_lbl = get_algo_meta(ref_key)["label"]
    cols = [f"Best {ref_lbl}", "Best E / BSMC"]
    m_rows, s_rows, ann = [], [], {}

    n_envs = len(envs)
    n_cols = min(3, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 3.2 * n_rows), squeeze=False)

    for r_idx, env in enumerate(envs):
        r, c = divmod(r_idx, n_cols)
        ax = axes[r][c]
        bs = batch_sizes[env]

        ref_tag = per_env_best_tags.get((env, ref_key))
        ref_res = all_runs[env].get(ref_key, {}).get(ref_tag) if ref_tag else None

        best_e_key, best_e_res = None, None
        for ek in e_keys:
            etag = per_env_best_tags.get((env, ek))
            if etag and etag in all_runs[env].get(ek, {}):
                cand = all_runs[env][ek][etag]
                if best_e_res is None or cand["auc_mean"] > best_e_res["auc_mean"]:
                    best_e_key = ek
                    best_e_res = cand

        rm = [
            ref_res["auc_mean"] if ref_res else np.nan,
            best_e_res["auc_mean"] if best_e_res else np.nan,
        ]
        rs = [
            ref_res["auc_std"] if ref_res else np.nan,
            best_e_res["auc_std"] if best_e_res else np.nan,
        ]
        m_rows.append(rm)
        s_rows.append(rs)
        if ref_res:
            ann[(r_idx, 0)] = _format_tag_short(ref_res.get("params", {}))
            ret_r = np.asarray(ref_res["returned_episode_returns"])
            mean_r = ret_r.mean(axis=0)
            sem_r = ret_r.std(axis=0) / np.sqrt(max(1, ret_r.shape[0]))
            t_r = np.arange(1, len(mean_r) + 1) * bs
            ax.plot(t_r, mean_r, label=ref_lbl, color=get_algo_meta(ref_key)["color"], lw=2.0)
            ax.fill_between(t_r, mean_r - sem_r, mean_r + sem_r, color=get_algo_meta(ref_key)["color"], alpha=0.14)
        if best_e_res and best_e_key:
            e_lbl = get_algo_meta(best_e_key)["label"]
            ann[(r_idx, 1)] = f"{e_lbl}; {_format_tag_short(best_e_res.get('params', {}))}"
            ret_e = np.asarray(best_e_res["returned_episode_returns"])
            mean_e = ret_e.mean(axis=0)
            sem_e = ret_e.std(axis=0) / np.sqrt(max(1, ret_e.shape[0]))
            t_e = np.arange(1, len(mean_e) + 1) * bs
            ax.plot(t_e, mean_e, label="Best E / BSMC", color="#7c3aed", lw=2.0)
            ax.fill_between(t_e, mean_e - sem_e, mean_e + sem_e, color="#7c3aed", alpha=0.14)

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Return")
        ax.grid(True, alpha=0.3)
        if r_idx == 0:
            ax.legend(loc="best", fontsize=8.0)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(f"{suite_title}: Q5 — Best E / BSMC vs. Best {ref_lbl}", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "q5_best_e_vs_td.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "q5_best_e_vs_td.pdf"), bbox_inches="tight")
    plt.close()

    save_table_bundle(
        pd.DataFrame(m_rows, index=envs, columns=cols),
        pd.DataFrame(s_rows, index=envs, columns=cols),
        f"{suite_title}: Q5 — Best E Variant (incl. BSMC) vs. Best {ref_lbl} (AUC Return)",
        os.path.join(tables_dir, "q5_best_e_vs_td"),
        cell_annotations=ann,
    )

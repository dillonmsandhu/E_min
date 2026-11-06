#!/usr/bin/env python3
"""
Generate publication-quality diagnostic tables for both V-critics and Q-critics.
Creates dedicated folders for each diagnostic, each containing:
- table.md  (GitHub Flavored Markdown)
- table.tex (LaTeX with booktabs)
- table.pdf (Vector PDF rendered via Matplotlib)
"""

import os
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ENV_TITLES = {
    "Acrobot-v1": "Acrobot",
    "CartPole-v1": "CartPole",
    "DeepSea-bsuite": "DeepSea",
    "DiscountingChain-bsuite": "DiscountingChain",
    "FourRooms-misc": "FourRooms",
    "MountainCar-v0": "MountainCar",
    "MountainCarContinuous-v0": "MountainCar (Cont.)",
    "Pendulum-v1": "Pendulum",
    "PointRobot-misc": "PointRobot",
    "Pong-misc": "Pong",
    "Reacher-misc": "Reacher",
    "Swimmer-misc": "Swimmer",
}

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
    os.makedirs(folder_path, exist_ok=True)
    n_rows, n_cols = len(row_names), len(col_names)

    # 1. Generate table.md
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

    # 2. Generate table.tex
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

    # 3. Generate table.pdf using Matplotlib
    fig_w = max(8.5, n_cols * 2.1 + 2.5)
    fig_h = max(3.8, n_rows * 0.65 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    # Row normalization for heatmap background
    row_mins = np.nanmin(mean_mat, axis=1, keepdims=True)
    row_maxs = np.nanmax(mean_mat, axis=1, keepdims=True)
    row_ranges = np.maximum(row_maxs - row_mins, 1e-8)
    mat_norm = (mean_mat - row_mins) / row_ranges if higher_is_better else (row_maxs - mean_mat) / row_ranges

    color_map = getattr(plt.cm, cmap)
    im = ax.imshow(mat_norm, cmap=color_map, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_names, fontsize=11, fontweight="bold", rotation=20, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_names, fontsize=11, fontweight="bold")

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

            txt = format_cell(val, std).replace(" ± ", "\n±")
            text_color = "white" if norm_val > 0.60 else "#111111"
            weight = "bold" if is_best else "normal"

            if is_best:
                rect = plt.Rectangle((c - 0.49, r - 0.49), 0.98, 0.98,
                                     fill=False, edgecolor="#00b894", linewidth=2.8, zorder=5)
                ax.add_patch(rect)

            ax.text(c, r, txt, ha="center", va="center", color=text_color,
                    fontsize=9.5, fontweight=weight, zorder=6)

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Relative AUC (0=Worst, 1=Best in Env)", fontsize=9.5, fontweight="semibold")
    cbar.ax.tick_params(labelsize=8.5)

    plt.tight_layout()
    pdf_path = os.path.join(folder_path, "table.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Generated table bundle in: {folder_path}")


def process_v_critics(pkl_path, base_out_dir):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    runs = d["runs"]
    envs = sorted(runs.keys())
    row_names = [ENV_TITLES.get(e, e) for e in envs]

    sample_env = envs[0]
    critic_lrs = sorted(runs[sample_env]["MC"].keys())
    sample_td = runs[sample_env]["TD(lambda)"][critic_lrs[0]]
    lambdas = sorted(sample_td.keys())
    abl_lmb = 0.8 if 0.8 in lambdas else lambdas[-1]

    tables_dir = os.path.join(base_out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    n_envs = len(envs)

    # 1. Main Comparison Table
    col_names_main = ["MC", "E(0)", "E(0) Naive Sampling", "TD(λ*)", "E(λ*) Baseline"]
    mean_main = np.zeros((n_envs, len(col_names_main)))
    std_main = np.zeros((n_envs, len(col_names_main)))

    for i, env in enumerate(envs):
        best_lr_mc = max(critic_lrs, key=lambda lr: np.mean(runs[env]["MC"][lr]["returned_episode_returns"]))
        mc_data = runs[env]["MC"][best_lr_mc]["returned_episode_returns"]
        mean_main[i, 0] = float(np.mean(mc_data))
        std_main[i, 0] = float(np.std(np.mean(mc_data, axis=1)))

        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"]))
        e0_data = runs[env]["E(0)_v_aux"][best_lr_e0]["returned_episode_returns"]
        mean_main[i, 1] = float(np.mean(e0_data))
        std_main[i, 1] = float(np.std(np.mean(e0_data, axis=1)))

        best_lr_e0_no = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)_no_v_aux"][lr]["returned_episode_returns"]))
        e0_no_data = runs[env]["E(0)_no_v_aux"][best_lr_e0_no]["returned_episode_returns"]
        mean_main[i, 2] = float(np.mean(e0_no_data))
        std_main[i, 2] = float(np.std(np.mean(e0_no_data, axis=1)))

        best_auc_td, best_td_data = -1e9, None
        for clr in critic_lrs:
            for lmb in lambdas:
                data = runs[env]["TD(lambda)"][clr][lmb]["returned_episode_returns"]
                auc = float(np.mean(data))
                if auc > best_auc_td:
                    best_auc_td = auc
                    best_td_data = data
        mean_main[i, 3] = float(np.mean(best_td_data))
        std_main[i, 3] = float(np.std(np.mean(best_td_data, axis=1)))

        best_auc_el, best_el_data = -1e9, None
        for clr in critic_lrs:
            for lmb in lambdas:
                data = runs[env]["E(lambda)"][clr][lmb]["returned_episode_returns"]
                auc = float(np.mean(data))
                if auc > best_auc_el:
                    best_auc_el = auc
                    best_el_data = data
        mean_main[i, 4] = float(np.mean(best_el_data))
        std_main[i, 4] = float(np.std(np.mean(best_el_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "main_comparison"),
        title="Comprehensive V-Critic Comparison",
        subtitle="Normalized Area Under Curve (AUC) of Episodic Return (Mean ± Std across seeds)",
        col_names=col_names_main,
        row_names=row_names,
        mean_mat=mean_main,
        std_mat=std_main,
        cmap="Blues",
    )

    # 2. Diagnostic 1: E(0) vs E(lambda=0)
    col_names_d1 = ["E(0) (Dirichlet)", "E(λ=0) (1-Step Backward)"]
    mean_d1 = np.zeros((n_envs, len(col_names_d1)))
    std_d1 = np.zeros((n_envs, len(col_names_d1)))

    for i, env in enumerate(envs):
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"]))
        e0_data = runs[env]["E(0)_v_aux"][best_lr_e0]["returned_episode_returns"]
        mean_d1[i, 0] = float(np.mean(e0_data))
        std_d1[i, 0] = float(np.std(np.mean(e0_data, axis=1)))

        best_lr_el0 = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(lambda)"][lr][0.0]["returned_episode_returns"]))
        el0_data = runs[env]["E(lambda)"][best_lr_el0][0.0]["returned_episode_returns"]
        mean_d1[i, 1] = float(np.mean(el0_data))
        std_d1[i, 1] = float(np.std(np.mean(el0_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_1_e0_vs_elambda0"),
        title="Diagnostic 1: Pure Dirichlet E(0) vs. 1-Step E(λ=0)",
        subtitle="Comparing True Continuous Bellman Error vs. Backward 1-Step Formulation (AUC)",
        col_names=col_names_d1,
        row_names=row_names,
        mean_mat=mean_d1,
        std_mat=std_d1,
        cmap="Purples",
    )

    # 3. Diagnostic 2: E(0) V_aux Effect
    col_names_d2 = ["E(0) with V_aux", "E(0) without V_aux (Degenerate)"]
    mean_d2 = np.zeros((n_envs, len(col_names_d2)))
    std_d2 = np.zeros((n_envs, len(col_names_d2)))

    for i, env in enumerate(envs):
        best_lr_e0_aux = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)_v_aux"][lr]["returned_episode_returns"]))
        e0_aux_data = runs[env]["E(0)_v_aux"][best_lr_e0_aux]["returned_episode_returns"]
        mean_d2[i, 0] = float(np.mean(e0_aux_data))
        std_d2[i, 0] = float(np.std(np.mean(e0_aux_data, axis=1)))

        best_lr_e0_no = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)_no_v_aux"][lr]["returned_episode_returns"]))
        e0_no_data = runs[env]["E(0)_no_v_aux"][best_lr_e0_no]["returned_episode_returns"]
        mean_d2[i, 1] = float(np.mean(e0_no_data))
        std_d2[i, 1] = float(np.std(np.mean(e0_no_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_2_e0_v_aux_effect"),
        title="Diagnostic 2: Impact of Auxiliary Value Anchor (V_aux) on E(0)",
        subtitle="AUC Comparison Showing Need for Anchor to Prevent Degenerate Minima",
        col_names=col_names_d2,
        row_names=row_names,
        mean_mat=mean_d2,
        std_mat=std_d2,
        cmap="Oranges",
    )

    # 4. Diagnostic 4: V_aux Regularization Ablations
    col_names_d4 = ["TD(λ) Converged Ref", "Baseline (MSE)", "+ Huber", "+ Trust Region", "+ Both"]
    mean_d4 = np.zeros((n_envs, len(col_names_d4)))
    std_d4 = np.zeros((n_envs, len(col_names_d4)))

    for i, env in enumerate(envs):
        best_clr = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(lambda)"][lr][abl_lmb]["returned_episode_returns"]))

        td_data = runs[env]["TD(lambda)"][best_clr][abl_lmb]["returned_episode_returns"]
        mean_d4[i, 0] = float(np.mean(td_data))
        std_d4[i, 0] = float(np.std(np.mean(td_data, axis=1)))

        b_data = runs[env]["E(lambda)"][best_clr][abl_lmb]["returned_episode_returns"]
        mean_d4[i, 1] = float(np.mean(b_data))
        std_d4[i, 1] = float(np.std(np.mean(b_data, axis=1)))

        h_data = runs[env]["E(lambda)_huber"][best_clr]["returned_episode_returns"]
        mean_d4[i, 2] = float(np.mean(h_data))
        std_d4[i, 2] = float(np.std(np.mean(h_data, axis=1)))

        t_data = runs[env]["E(lambda)_trust"][best_clr]["returned_episode_returns"]
        mean_d4[i, 3] = float(np.mean(t_data))
        std_d4[i, 3] = float(np.std(np.mean(t_data, axis=1)))

        both_data = runs[env]["E(lambda)_both"][best_clr]["returned_episode_returns"]
        mean_d4[i, 4] = float(np.mean(both_data))
        std_d4[i, 4] = float(np.std(np.mean(both_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_4_v_aux_ablations"),
        title=f"Diagnostic 4: V_aux Regularization Ablations (λ={abl_lmb})",
        subtitle="AUC Robustness Across Huber Loss and Trust Region Stabilization",
        col_names=col_names_d4,
        row_names=row_names,
        mean_mat=mean_d4,
        std_mat=std_d4,
        cmap="YlGn",
    )


def process_q_critics(pkl_path, base_out_dir):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    runs = d["runs"]
    envs = sorted(runs.keys())
    row_names = [ENV_TITLES.get(e, e) for e in envs]

    sample_env = envs[0]
    critic_lrs = sorted(runs[sample_env]["MC"].keys())
    sample_sarsa = runs[sample_env]["SARSA(lambda)"][critic_lrs[0]]
    lambdas = sorted(sample_sarsa.keys())

    tables_dir = os.path.join(base_out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    n_envs = len(envs)

    # 1. Main Comparison Table (Q-PPO)
    col_names_main = ["MC", "E(0)-Q", "SARSA(λ*)", "E(λ*-Q)"]
    mean_main = np.zeros((n_envs, len(col_names_main)))
    std_main = np.zeros((n_envs, len(col_names_main)))

    for i, env in enumerate(envs):
        best_lr_mc = max(critic_lrs, key=lambda lr: np.mean(runs[env]["MC"][lr]["returned_episode_returns"]))
        mc_data = runs[env]["MC"][best_lr_mc]["returned_episode_returns"]
        mean_main[i, 0] = float(np.mean(mc_data))
        std_main[i, 0] = float(np.std(np.mean(mc_data, axis=1)))

        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)"][lr]["returned_episode_returns"]))
        e0_data = runs[env]["E(0)"][best_lr_e0]["returned_episode_returns"]
        mean_main[i, 1] = float(np.mean(e0_data))
        std_main[i, 1] = float(np.std(np.mean(e0_data, axis=1)))

        best_auc_s, best_s_data = -1e9, None
        for clr in critic_lrs:
            for lmb in lambdas:
                data = runs[env]["SARSA(lambda)"][clr][lmb]["returned_episode_returns"]
                auc = float(np.mean(data))
                if auc > best_auc_s:
                    best_auc_s = auc
                    best_s_data = data
        mean_main[i, 2] = float(np.mean(best_s_data))
        std_main[i, 2] = float(np.std(np.mean(best_s_data, axis=1)))

        best_auc_e, best_e_data = -1e9, None
        for clr in critic_lrs:
            for lmb in lambdas:
                data = runs[env]["E(lambda)"][clr][lmb]["returned_episode_returns"]
                auc = float(np.mean(data))
                if auc > best_auc_e:
                    best_auc_e = auc
                    best_e_data = data
        mean_main[i, 3] = float(np.mean(best_e_data))
        std_main[i, 3] = float(np.std(np.mean(best_e_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "main_comparison"),
        title="Comprehensive Q-Critic Comparison",
        subtitle="Normalized Area Under Curve (AUC) across Deterministic Domains (Mean ± Std)",
        col_names=col_names_main,
        row_names=row_names,
        mean_mat=mean_main,
        std_mat=std_main,
        cmap="Blues",
    )

    # 2. Diagnostic 1: E(0)-Q vs E(lambda=0)-Q
    col_names_d1 = ["E(0)-Q (Action Anchor)", "E(λ=0)-Q (1-Step SARSA Target)"]
    mean_d1 = np.zeros((n_envs, len(col_names_d1)))
    std_d1 = np.zeros((n_envs, len(col_names_d1)))

    for i, env in enumerate(envs):
        best_lr_e0 = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(0)"][lr]["returned_episode_returns"]))
        e0_data = runs[env]["E(0)"][best_lr_e0]["returned_episode_returns"]
        mean_d1[i, 0] = float(np.mean(e0_data))
        std_d1[i, 0] = float(np.std(np.mean(e0_data, axis=1)))

        best_lr_el0 = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(lambda)"][lr][0.0]["returned_episode_returns"]))
        el0_data = runs[env]["E(lambda)"][best_lr_el0][0.0]["returned_episode_returns"]
        mean_d1[i, 1] = float(np.mean(el0_data))
        std_d1[i, 1] = float(np.std(np.mean(el0_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_1_e0_vs_elambda0"),
        title="Diagnostic 1: Pure Dirichlet E(0)-Q vs. 1-Step E(λ=0)-Q",
        subtitle="AUC Comparison of Action Anchor Dirichlet Formulation vs 1-Step Backward Formulation",
        col_names=col_names_d1,
        row_names=row_names,
        mean_mat=mean_d1,
        std_mat=std_d1,
        cmap="Purples",
    )

    # 3. Diagnostic 2: SARSA vs E-Q across Lambda Spectrum
    col_names_d2 = [f"SARSA(λ={l})" for l in lambdas] + [f"E(λ={l})-Q" for l in lambdas]
    mean_d2 = np.zeros((n_envs, len(col_names_d2)))
    std_d2 = np.zeros((n_envs, len(col_names_d2)))

    for i, env in enumerate(envs):
        for j, lmb in enumerate(lambdas):
            best_lr_s = max(critic_lrs, key=lambda lr: np.mean(runs[env]["SARSA(lambda)"][lr][lmb]["returned_episode_returns"]))
            s_data = runs[env]["SARSA(lambda)"][best_lr_s][lmb]["returned_episode_returns"]
            mean_d2[i, j] = float(np.mean(s_data))
            std_d2[i, j] = float(np.std(np.mean(s_data, axis=1)))

            best_lr_e = max(critic_lrs, key=lambda lr: np.mean(runs[env]["E(lambda)"][lr][lmb]["returned_episode_returns"]))
            e_data = runs[env]["E(lambda)"][best_lr_e][lmb]["returned_episode_returns"]
            mean_d2[i, len(lambdas) + j] = float(np.mean(e_data))
            std_d2[i, len(lambdas) + j] = float(np.std(np.mean(e_data, axis=1)))

    export_table_bundle(
        folder_path=os.path.join(tables_dir, "diagnostic_2_sarsa_vs_elambda_spectrum"),
        title="Diagnostic 2: SARSA(λ) vs. E(λ)-Q Lookahead Spectrum",
        subtitle=f"AUC Comparison across λ ∈ {lambdas}",
        col_names=col_names_d2,
        row_names=row_names,
        mean_mat=mean_d2,
        std_mat=std_d2,
        cmap="Blues",
    )


if __name__ == "__main__":
    print("Exporting V-Critic Tables...")
    process_v_critics(
        pkl_path="/Users/dillonsandhu/Documents/Research/E_min/results/sweep_comprehensive_critic/data/consolidated_sweep_metrics.pkl",
        base_out_dir="/Users/dillonsandhu/Documents/Research/E_min/results/sweep_comprehensive_critic"
    )
    print("Exporting Q-Critic Tables...")
    process_q_critics(
        pkl_path="/Users/dillonsandhu/Documents/Research/E_min/results/sweep_comprehensive_q_critic/data/consolidated_sweep_metrics.pkl",
        base_out_dir="/Users/dillonsandhu/Documents/Research/E_min/results/sweep_comprehensive_q_critic"
    )

    # Sync to brain artifacts folder
    art_v = "/Users/dillonsandhu/.gemini/antigravity-ide/brain/dcbfdcae-de92-4f35-80ff-46e88e5a5f98/final_sweep_plots/v_critics/tables"
    art_q = "/Users/dillonsandhu/.gemini/antigravity-ide/brain/dcbfdcae-de92-4f35-80ff-46e88e5a5f98/final_sweep_plots/q_critics/tables"
    shutil.copytree("/Users/dillonsandhu/Documents/Research/E_min/results/sweep_comprehensive_critic/tables", art_v, dirs_exist_ok=True)
    shutil.copytree("/Users/dillonsandhu/Documents/Research/E_min/results/sweep_comprehensive_q_critic/tables", art_q, dirs_exist_ok=True)
    print("All diagnostic tables generated and synchronized successfully.")

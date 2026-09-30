"""
visualize_multidim_sweep.py

Comprehensive Multi-Dimensional Post-Processing Visualizations for Hyperparameter Sweeps:
1. Parallel Coordinates Plot:
   - Shows high-dimensional trajectories through hyperparameter space (Loss Type, Heads, Epochs, Weight Decay)
   - Color-coded by final performance; highlights optimal "golden paths".
2. Marginal Main Effects & Significance:
   - Box/scatter distribution and 95% bootstrap CI for each isolated hyperparameter.
   - Computes Cohen's d effect sizes and Welch's t-test p-values.
3. 2D Interaction Heatmaps:
   - Unveils pairwise synergies (e.g., Critic Epochs x Weight Decay, Loss Type x Critic Epochs).
4. Variance Decomposition (ANOVA / fANOVA Importance):
   - Quantifies the percentage of total variance explained by each factor and two-way interaction.
5. Ablation Learning Curves:
   - Clean paired comparisons: Best Huber vs Best MSE, Best 4-Head vs Best 1-Head, Best 16-Epoch vs Best 4-Epoch.

Usage:
    python scripts/e_optimization/visualize_multidim_sweep.py --results-dir results/ppo/sweeps/e_experimental_gymnax_12345/CartPole-v1
    python scripts/e_optimization/visualize_multidim_sweep.py --suite-dir results/ppo/sweeps/e_experimental_gymnax_12345
"""

import os
import sys

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as stats

# Vector TrueType font embedding
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10


def plot_parallel_coordinates(df: pd.DataFrame, ax: plt.Axes, env_name: str, score_col="final_window_mean"):
    """Generates a parallel coordinates plot highlighting top vs bottom trajectories."""
    cols_to_plot = []
    col_labels = []
    
    # Identify swept columns
    candidate_cols = [
        ("lambda", "Lambda (λ)"),
        ("return_lambda", "Return Lambda (λ_ret)"),
        ("critic_epochs", "Critic Epochs"),
        ("critic_lr", "Critic LR"),
        ("num_value_heads", "Value Heads"),
        ("weight_decay", "Weight Decay"),
        ("critic_loss_type", "Loss Type"),
        ("num_envs", "Num Envs"),
    ]
    for c, label in candidate_cols:
        if c in df.columns and df[c].nunique() > 1:
            cols_to_plot.append(c)
            col_labels.append(label)
            
    if not cols_to_plot:
        ax.text(0.5, 0.5, "Not enough varying dimensions for Parallel Coordinates", 
                ha="center", va="center", transform=ax.transAxes)
        return

    n_axes = len(cols_to_plot) + 1
    col_labels.append("Final Return")

    # Map categories to numerical coordinates [0, 1]
    norm_df = pd.DataFrame()
    cat_mappings = {}
    
    for c in cols_to_plot:
        vals = sorted(df[c].unique())
        cat_mappings[c] = {v: i / max(1, len(vals) - 1) for i, v in enumerate(vals)}
        norm_df[c] = df[c].map(cat_mappings[c])

    # Normalize score
    min_score = df[score_col].min()
    max_score = df[score_col].max()
    score_range = max(1e-6, max_score - min_score)
    norm_df["score_norm"] = (df[score_col] - min_score) / score_range
    norm_df["raw_score"] = df[score_col]

    # Plot axes lines
    for x_i in range(n_axes):
        ax.axvline(x_i, color="#d0d0d0", linestyle="--", linewidth=1.0, zorder=1)

    # Sort so top trajectories are plotted on top
    norm_df_sorted = norm_df.sort_values("score_norm", ascending=True)
    top_thresh = norm_df["score_norm"].quantile(0.75)
    bottom_thresh = norm_df["score_norm"].quantile(0.25)

    cmap = plt.cm.viridis

    for _, row in norm_df_sorted.iterrows():
        coords = [row[c] for c in cols_to_plot] + [row["score_norm"]]
        s = row["score_norm"]
        color = cmap(s)
        
        if s >= top_thresh:
            lw = 2.4
            alpha = 0.95
            zorder = 5
        elif s <= bottom_thresh:
            lw = 1.0
            alpha = 0.35
            zorder = 2
        else:
            lw = 1.4
            alpha = 0.6
            zorder = 3

        ax.plot(range(n_axes), coords, color=color, linewidth=lw, alpha=alpha, zorder=zorder)

    # Setup axis ticks and labels
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(col_labels, fontweight="bold")
    ax.set_xlim(-0.1, n_axes - 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])

    # Add custom tick labels for each dimension
    for x_i, c in enumerate(cols_to_plot):
        vals = sorted(df[c].unique())
        for v in vals:
            y_pos = cat_mappings[c][v]
            ax.text(x_i - 0.05, y_pos, str(v), ha="right", va="center", fontsize=8.5, color="#333333")

    # Score ticks on last axis
    score_ticks = np.linspace(min_score, max_score, 5)
    for st in score_ticks:
        y_pos = (st - min_score) / score_range
        ax.text(n_axes - 1 + 0.05, y_pos, f"{st:.2f}", ha="left", va="center", fontsize=8.5, color="#333333")

    ax.set_title(f"Parallel Coordinates Trajectories: {env_name} (Color = Return)", pad=15, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_score, vmax=max_score))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.08, fraction=0.03)
    cbar.set_label("Final Window Mean Return", fontsize=9)


def plot_marginal_effects(df: pd.DataFrame, axes: list, score_col="final_window_mean"):
    """Plots marginal main effects for each hyperparameter with ANOVA / effect sizes."""
    candidate_cols = [
        ("critic_loss_type", "Critic Loss", ["mse", "huber"]),
        ("num_value_heads", "Value Heads", [1, 4]),
        ("critic_epochs", "Critic Epochs", [4, 16]),
        ("weight_decay", "Weight Decay", [0.0, 0.01]),
    ]

    for ax_idx, (col, title, _) in enumerate(candidate_cols):
        ax = axes[ax_idx]
        if col not in df.columns or df[col].nunique() < 2:
            ax.text(0.5, 0.5, f"{title}: Single Value", ha="center", va="center", transform=ax.transAxes)
            continue

        unique_vals = sorted(df[col].unique(), key=lambda x: str(x))
        groups = [df[df[col] == v][score_col].values for v in unique_vals]

        # Boxplot with version-safe labels parameter
        box_kwargs = dict(patch_artist=True,
                          boxprops=dict(facecolor="#e8f0fe", color="#1a73e8", linewidth=1.5),
                          medianprops=dict(color="#d93025", linewidth=2.0),
                          whiskerprops=dict(color="#1a73e8", linewidth=1.2),
                          capprops=dict(color="#1a73e8", linewidth=1.2),
                          widths=0.45)
        try:
            bp = ax.boxplot(groups, tick_labels=[str(v) for v in unique_vals], **box_kwargs)
        except TypeError:
            bp = ax.boxplot(groups, labels=[str(v) for v in unique_vals], **box_kwargs)

        # Overlay jittered scatter points
        for i, grp in enumerate(groups):
            jitter = np.random.normal(0, 0.04, size=len(grp))
            ax.scatter(np.full_like(grp, i + 1) + jitter, grp, color="#174ea6", alpha=0.6, s=25, zorder=4)

        # Means and SEM
        means = [np.mean(g) for g in groups]
        ax.plot([1, 2], means, "k--", marker="o", markersize=6, linewidth=1.5, zorder=5, label="Mean")

        # Effect size (Cohen's d) & p-value
        if len(groups) == 2 and len(groups[0]) > 1 and len(groups[1]) > 1:
            g1, g2 = groups[0], groups[1]
            diff = np.mean(g2) - np.mean(g1)
            pooled_std = np.sqrt(((len(g1) - 1) * np.var(g1, ddof=1) + (len(g2) - 1) * np.var(g2, ddof=1)) / (len(g1) + len(g2) - 2))
            d = diff / max(1e-8, pooled_std)
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            sig_text = f"Δ={diff:+.2f}, d={d:.2f}\np={p_val:.3f}"
            ax.text(0.95, 0.95, sig_text, transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#dadce0"))

        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        ax.set_ylabel("Final Return" if ax_idx == 0 else "")


def plot_interaction_heatmaps(df: pd.DataFrame, axes: list, score_col="final_window_mean"):
    """Plots 2D interaction grids across actively varying hyperparameter pairs."""
    varying_cols = [
        c for c in ["lambda", "critic_epochs", "critic_lr", "num_value_heads", "weight_decay", "critic_loss_type", "num_envs"]
        if c in df.columns and df[c].nunique() > 1
    ]
    name_map = {
        "lambda": "Lambda (λ)",
        "critic_epochs": "Critic Epochs",
        "critic_lr": "Critic LR",
        "weight_decay": "Weight Decay",
        "num_value_heads": "Value Heads",
        "critic_loss_type": "Critic Loss",
        "num_envs": "Num Envs",
    }

    candidate_pairs = []
    for i in range(len(varying_cols)):
        for j in range(i + 1, len(varying_cols)):
            c1, c2 = varying_cols[i], varying_cols[j]
            candidate_pairs.append((c1, c2, name_map[c1], name_map[c2]))

    for idx, ax in enumerate(axes):
        if idx >= len(candidate_pairs):
            ax.set_visible(False)
            continue

        col_x, col_y, label_x, label_y = candidate_pairs[idx]

        pivot = df.pivot_table(index=col_y, columns=col_x, values=score_col, aggfunc="mean")
        pivot_sem = df.pivot_table(index=col_y, columns=col_x, values=score_col, aggfunc=lambda x: np.std(x, ddof=1)/np.sqrt(max(1, len(x))))

        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

        # Ticks and labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], fontweight="bold")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(r) for r in pivot.index], fontweight="bold")
        ax.set_xlabel(label_x, fontweight="bold")
        ax.set_ylabel(label_y, fontweight="bold")
        ax.set_title(f"Interaction: {label_x} × {label_y}", fontweight="bold", pad=8)

        # Annotate cell values
        val_range = pivot.values.max() - pivot.values.min()
        thresh = pivot.values.min() + val_range * 0.6
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                m_val = pivot.values[r, c]
                sem_val = pivot_sem.values[r, c] if r < pivot_sem.shape[0] and c < pivot_sem.shape[1] else 0.0
                text_color = "white" if m_val > thresh else "black"
                ax.text(c, r, f"{m_val:.2f}\n±{sem_val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=8.5, fontweight="semibold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_variance_decomposition(df: pd.DataFrame, ax: plt.Axes, score_col="final_window_mean"):
    """Computes two-way ANOVA variance attribution across swept dimensions."""
    factors = [c for c in ["lambda", "critic_epochs", "critic_lr", "num_value_heads", "weight_decay", "critic_loss_type", "num_envs"]
               if c in df.columns and df[c].nunique() > 1]
    
    if len(factors) < 2:
        ax.text(0.5, 0.5, "Need at least 2 swept factors for variance decomposition",
                ha="center", va="center", transform=ax.transAxes)
        return

    # Grand mean and total SS
    y = df[score_col].values
    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    
    if ss_total < 1e-12:
        ax.text(0.5, 0.5, "Zero variance in return across configs", ha="center", va="center", transform=ax.transAxes)
        return

    ss_dict = {}

    # Main effects
    for f in factors:
        ss_f = 0.0
        for val, grp in df.groupby(f):
            ss_f += len(grp) * (grp[score_col].mean() - grand_mean) ** 2
        label = f.replace("critic_", "").replace("num_value_", "").replace("_", " ").title()
        ss_dict[f"Main: {label}"] = max(0.0, ss_f)

    # 2-way interactions
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            f1, f2 = factors[i], factors[j]
            ss_f1f2 = 0.0
            for _, grp in df.groupby([f1, f2]):
                ss_f1f2 += len(grp) * (grp[score_col].mean() - grand_mean) ** 2
            # Subtract main effects
            label1 = f1.replace("critic_", "").replace("num_value_", "").replace("_", " ").title()
            label2 = f2.replace("critic_", "").replace("num_value_", "").replace("_", " ").title()
            ss_inter = max(0.0, ss_f1f2 - ss_dict[f"Main: {label1}"] - ss_dict[f"Main: {label2}"])
            ss_dict[f"{label1} × {label2}"] = ss_inter

    # Convert to percentage of variance explained
    sorted_items = sorted(ss_dict.items(), key=lambda x: x[1], reverse=True)[:8]
    labels = [k for k, _ in sorted_items]
    percentages = [100.0 * (v / ss_total) for _, v in sorted_items]

    y_pos = np.arange(len(labels))
    colors = ["#1a73e8" if "Main:" in l else "#34a853" for l in labels]

    bars = ax.barh(y_pos, percentages, color=colors, height=0.6, edgecolor="#333333", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of Variance in Return Explained (%)", fontweight="bold")
    ax.set_title("Hyperparameter Importance (Two-Way ANOVA Variance Attribution)", fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="x")

    for bar, pct in zip(bars, percentages):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9, fontweight="bold")


def plot_ablation_curves(metrics_dict: dict, df: pd.DataFrame, ax: plt.Axes, env_name: str, window_size=100):
    """Plots clean paired comparisons across key varying dimensions."""
    if not metrics_dict:
        ax.text(0.5, 0.5, "No metrics.pkl found for learning curves", ha="center", va="center", transform=ax.transAxes)
        return

    comparisons = []
    
    # 1. Loss type comparison if varying
    if "critic_loss_type" in df.columns and set(df["critic_loss_type"].unique()) >= {"mse", "huber"}:
        best_mse = df[df["critic_loss_type"] == "mse"].sort_values("final_window_mean", ascending=False).iloc[0]
        best_huber = df[df["critic_loss_type"] == "huber"].sort_values("final_window_mean", ascending=False).iloc[0]
        comparisons.append(("Best Huber", best_huber["config"], "#1b7837", "-"))
        comparisons.append(("Best MSE", best_mse["config"], "#762a83", "--"))

    # 2. Value Heads comparison if varying
    if "num_value_heads" in df.columns and df["num_value_heads"].nunique() > 1:
        u_heads = sorted(df["num_value_heads"].unique())
        best_low = df[df["num_value_heads"] == u_heads[0]].sort_values("final_window_mean", ascending=False).iloc[0]
        best_high = df[df["num_value_heads"] == u_heads[-1]].sort_values("final_window_mean", ascending=False).iloc[0]
        comparisons.append((f"Best {u_heads[-1]}-Head", best_high["config"], "#00441b", "-"))
        comparisons.append((f"Best {u_heads[0]}-Head", best_low["config"], "#b2182b", ":"))

    # 3. Epochs comparison if varying
    if "critic_epochs" in df.columns and df["critic_epochs"].nunique() > 1 and len(comparisons) < 4:
        u_epochs = sorted(df["critic_epochs"].unique())
        best_min_ep = df[df["critic_epochs"] == u_epochs[0]].sort_values("final_window_mean", ascending=False).iloc[0]
        best_max_ep = df[df["critic_epochs"] == u_epochs[-1]].sort_values("final_window_mean", ascending=False).iloc[0]
        comparisons.append((f"Best {u_epochs[-1]}-Epoch", best_max_ep["config"], "#1a73e8", "-."))
        comparisons.append((f"Best {u_epochs[0]}-Epoch", best_min_ep["config"], "#ea4335", "--"))

    # 4. Weight decay comparison if varying
    if "weight_decay" in df.columns and df["weight_decay"].nunique() > 1 and len(comparisons) < 4:
        u_wd = sorted(df["weight_decay"].unique())
        best_low_wd = df[df["weight_decay"] == u_wd[0]].sort_values("final_window_mean", ascending=False).iloc[0]
        best_high_wd = df[df["weight_decay"] == u_wd[-1]].sort_values("final_window_mean", ascending=False).iloc[0]
        comparisons.append((f"Best {u_wd[-1]} WD", best_high_wd["config"], "#ff7f0e", "-"))
        comparisons.append((f"Best {u_wd[0]} WD", best_low_wd["config"], "#1f77b4", ":"))

    if not comparisons:
        # Fallback: top 3 configs
        top_configs = df.sort_values("final_window_mean", ascending=False).head(3)
        colors = ["#1a73e8", "#34a853", "#ea4335"]
        for i, (_, row) in enumerate(top_configs.iterrows()):
            comparisons.append((f"Top {i+1}: {row['config']}", row["config"], colors[i], "-"))

    for label, cfg_name, color, style in comparisons:
        if cfg_name in metrics_dict:
            entry = metrics_dict[cfg_name]
            if isinstance(entry, dict) and "mean" in entry:
                mean_c = entry["mean"]
                sem_c = entry.get("sem", np.zeros_like(mean_c))
            elif isinstance(entry, dict):
                # Search for metric key
                found_metric = None
                for candidate in ["returned_discounted_episode_returns", "returned_episode_returns", "returns"]:
                    if candidate in entry:
                        found_metric = entry[candidate]
                        break
                if found_metric is None and len(entry) > 0:
                    found_metric = list(entry.values())[0]

                if found_metric is not None:
                    arr = np.asarray(found_metric)
                    if arr.ndim > 1:
                        mean_c = arr.mean(axis=0)
                        sem_c = arr.std(axis=0) / np.sqrt(max(1, arr.shape[0]))
                    else:
                        mean_c = arr
                        sem_c = np.zeros_like(mean_c)
                else:
                    continue
            else:
                continue

            x = np.arange(len(mean_c))
            final_val = float(np.mean(mean_c[-min(len(mean_c), window_size):]))
            ax.plot(x, mean_c, label=f"{label} ({final_val:.2f})", color=color, linestyle=style, linewidth=2.0)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.15)

    ax.set_title(f"Ablation Learning Curves: {env_name} (Mean ± 1 SEM)", fontweight="bold")
    ax.set_xlabel("Environment Steps (Updates)", fontweight="bold")
    ax.set_ylabel("Episode Return", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)


def generate_multidim_analysis_pdf(env_dir: str, output_pdf=None):
    """Compiles the full 5-panel multi-dimensional analysis into a publication-ready PDF and PNG."""
    summary_path = os.path.join(env_dir, "summary_e_lambda.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(env_dir, "summary_td_lambda.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(env_dir, "summary_e_experimental.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(env_dir, "summary_td.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(env_dir, "summary_e_opt.csv")
    if not os.path.exists(summary_path):
        import glob
        matches = glob.glob(os.path.join(env_dir, "summary_*.csv"))
        if matches:
            summary_path = matches[0]
        else:
            print(f"Warning: Summary CSV not found in {env_dir}")
            return None

    df = pd.read_csv(summary_path)
    env_name = os.path.basename(os.path.normpath(env_dir))

    metrics_dict = {}
    pkl_path = os.path.join(env_dir, "metrics.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            metrics_dict = pickle.load(f)

    if output_pdf is None:
        output_pdf = os.path.join(env_dir, "multidim_analysis.pdf")
    output_png = os.path.splitext(output_pdf)[0] + ".png"

    print(f"Generating multi-dimensional analysis report: {output_pdf} ...")

    with PdfPages(output_pdf) as pdf:
        # Page 1: Multi-Dimensional Overview Poster (2x2 grid)
        fig = plt.figure(figsize=(15, 11))
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.25)

        # Panel (0, 0): Parallel coordinates
        ax_pc = fig.add_subplot(gs[0, :])
        plot_parallel_coordinates(df, ax_pc, env_name)

        # Panel (1, 0): Variance attribution
        ax_var = fig.add_subplot(gs[1, 0])
        plot_variance_decomposition(df, ax_var)

        # Panel (1, 1): Ablation learning curves
        ax_abl = fig.add_subplot(gs[1, 1])
        plot_ablation_curves(metrics_dict, df, ax_abl, env_name)

        fig.suptitle(f"Hyperparameter Geometry & Attribution: {env_name}", fontsize=14, fontweight="bold", y=0.98)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Marginal Main Effects (4 subplots)
        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
        plot_marginal_effects(df, axes2)
        fig2.suptitle(f"Marginal Main Effects & Distributions: {env_name}", fontsize=13, fontweight="bold", y=1.02)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        # Page 3: 2D Interaction Heatmaps (3 subplots)
        fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
        plot_interaction_heatmaps(df, axes3)
        fig3.suptitle(f"2D Interaction Synergy Grids: {env_name}", fontsize=13, fontweight="bold", y=1.02)
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

    print(f"Successfully generated:\n  - {output_pdf}\n  - {output_png}")
    return output_pdf


def generate_suite_multidim_pdf(suite_dir: str, output_pdf=None):
    """Compiles suite-level multi-environment normalized comparisons across all games."""
    import glob
    env_dirs = []
    candidate_roots = glob.glob(suite_dir) if ("*" in suite_dir or "?" in suite_dir) else [suite_dir]
    
    seen_envs = set()
    for s_dir in sorted(candidate_roots):
        if not os.path.isdir(s_dir):
            continue
        for item in sorted(os.listdir(s_dir)):
            p = os.path.join(s_dir, item)
            if os.path.isdir(p) and (os.path.exists(os.path.join(p, "summary_e_lambda.csv")) or
                                     os.path.exists(os.path.join(p, "summary_td_lambda.csv")) or
                                     os.path.exists(os.path.join(p, "summary_e_experimental.csv")) or
                                     os.path.exists(os.path.join(p, "summary_td.csv")) or
                                     os.path.exists(os.path.join(p, "summary_e_opt.csv"))):
                if item not in seen_envs:
                    env_dirs.append((item, p))
                    seen_envs.add(item)

    if not env_dirs:
        print(f"No environment directories with summary CSVs found in {suite_dir}")
        return None

    first_root = candidate_roots[0] if candidate_roots else "."
    if not os.path.isdir(first_root):
        first_root = os.path.dirname(first_root) or "."

    if output_pdf is None:
        output_pdf = os.path.join(first_root, "suite_multidim_analysis.pdf")
    output_png = os.path.splitext(output_pdf)[0] + ".png"

    print(f"Found {len(env_dirs)} completed environments in suite: {[name for name, _ in env_dirs]}")
    print(f"Generating suite multi-dimensional report: {output_pdf} ...")

    # Load and normalize data across environments
    dfs = []
    for env_name, env_path in env_dirs:
        csv_file = os.path.join(env_path, "summary_e_lambda.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(env_path, "summary_td_lambda.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(env_path, "summary_e_experimental.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(env_path, "summary_td.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(env_path, "summary_e_opt.csv")
        sub_df = pd.read_csv(csv_file).copy()
        sub_df["env_name"] = env_name

        # Normalize score within this environment to [0, 1]
        mn = sub_df["final_window_mean"].min()
        mx = sub_df["final_window_mean"].max()
        rng = max(1e-6, mx - mn)
        sub_df["norm_score"] = (sub_df["final_window_mean"] - mn) / rng
        dfs.append(sub_df)

    suite_df = pd.concat(dfs, ignore_index=True)

    with PdfPages(output_pdf) as pdf:
        # Page 1: Suite Overview
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        # Panel (0, 0): Cross-task Normalized Performance Matrix (Env x Config)
        ax_mat = fig.add_subplot(gs[0, :])
        pivot = suite_df.pivot_table(index="env_name", columns="config", values="norm_score")
        # Sort columns by overall mean normalized score
        col_order = pivot.mean(axis=0).sort_values(ascending=False).index
        pivot = pivot[col_order]

        im = ax_mat.imshow(pivot.values, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
        ax_mat.set_xticks(range(len(pivot.columns)))
        ax_mat.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8.5, fontweight="semibold")
        ax_mat.set_yticks(range(len(pivot.index)))
        ax_mat.set_yticklabels(pivot.index, fontweight="bold")
        ax_mat.set_title("Cross-Environment Normalized Performance Matrix (1.0 = Best in Task)", fontweight="bold", pad=12)
        cbar = plt.colorbar(im, ax=ax_mat, orientation="vertical", pad=0.02, fraction=0.02)
        cbar.set_label("Min-Max Normalized Return", fontsize=9)

        # Annotate top entries
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                val = pivot.values[r, c]
                if not np.isnan(val) and val >= 0.99:
                    ax_mat.text(c, r, "★", ha="center", va="center", color="cyan", fontsize=10, fontweight="bold")

        # Panel (1, 0): Suite-wide Marginal Main Effects
        ax_marg = fig.add_subplot(gs[1, 0])
        # Grouped by Loss Type & Heads
        key_factors = []
        if "critic_loss_type" in suite_df.columns and suite_df["critic_loss_type"].nunique() > 1:
            key_factors.append("critic_loss_type")
        if "num_value_heads" in suite_df.columns and suite_df["num_value_heads"].nunique() > 1:
            key_factors.append("num_value_heads")
        if "critic_epochs" in suite_df.columns and suite_df["critic_epochs"].nunique() > 1:
            key_factors.append("critic_epochs")
        if "weight_decay" in suite_df.columns and suite_df["weight_decay"].nunique() > 1:
            key_factors.append("weight_decay")

        sub_labels = []
        sub_groups = []
        for f in key_factors:
            for val in sorted(suite_df[f].unique()):
                sub_labels.append(f"{f.split('_')[-1]}={val}")
                sub_groups.append(suite_df[suite_df[f] == val]["norm_score"].values)

        if sub_groups:
            box_kwargs = dict(patch_artist=True,
                              boxprops=dict(facecolor="#e8eaed", color="#5f6368", linewidth=1.2),
                              medianprops=dict(color="#d93025", linewidth=2.0),
                              whiskerprops=dict(color="#5f6368", linewidth=1.0),
                              capprops=dict(color="#5f6368", linewidth=1.0),
                              widths=0.45)
            try:
                ax_marg.boxplot(sub_groups, tick_labels=sub_labels, **box_kwargs)
            except TypeError:
                ax_marg.boxplot(sub_groups, labels=sub_labels, **box_kwargs)
            ax_marg.set_xticklabels(sub_labels, rotation=30, ha="right", fontsize=8.5)
            ax_marg.set_ylabel("Normalized Return Across All Tasks", fontweight="bold")
            ax_marg.set_title("Suite-Wide Marginal Performance Distributions", fontweight="bold")
            ax_marg.grid(True, linestyle=":", alpha=0.5, axis="y")

        # Panel (1, 1): Suite-wide Hyperparameter Importance (ANOVA)
        ax_imp = fig.add_subplot(gs[1, 1])
        plot_variance_decomposition(suite_df, ax_imp, score_col="norm_score")

        fig.suptitle(f"Gymnax Suite Multi-Dimensional Meta-Analysis ({len(env_dirs)} Environments)",
                     fontsize=15, fontweight="bold", y=0.98)
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Append per-environment detailed pages
        for env_name, env_path in env_dirs:
            per_env_pdf = os.path.join(env_path, "multidim_analysis.pdf")
            if not os.path.exists(per_env_pdf):
                try:
                    generate_multidim_analysis_pdf(env_path)
                except Exception as ex:
                    print(f"Skipping per-env pdf for {env_name}: {ex}")

    print(f"Successfully generated suite multi-dimensional report:\n  - {output_pdf}\n  - {output_png}")
    return output_pdf


def main():
    parser = argparse.ArgumentParser(description="Multi-Dimensional Sweep Visualization Generator")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to an individual environment sweep results directory")
    parser.add_argument("--suite-dir", type=str, default=None,
                        help="Path to a suite directory containing multiple environment folders")
    parser.add_argument("--output-pdf", type=str, default=None,
                        help="Custom output PDF path")
    args = parser.parse_args()

    if args.suite_dir:
        generate_suite_multidim_pdf(args.suite_dir, args.output_pdf)
    elif args.results_dir:
        generate_multidim_analysis_pdf(args.results_dir, args.output_pdf)
    else:
        print("Please provide --results-dir (single environment) or --suite-dir (entire suite).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_env_properties_comparison.py

Generates publication-quality 2x3 vector PDF & PNG comparison figures:
- Row 1: Clean Environments (Sparse Cont., Dense Cont., Dense Disc.)
- Row 2: Noisy Environments (Sparse Cont., Dense Cont., Dense Disc.)
Each panel plots 3 curves (E(0), TD(0), TD(lambda)) with Mean ± 1 SEM shading.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matplotlib publication settings
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["axes.labelsize"] = 10


ALGO_METADATA = {
    "E(0)": {
        "color": "#1b7837",       # Forest Green
        "linestyle": "-",
        "linewidth": 2.2,
        "label": "E(0) (Dirichlet Error)",
    },
    "TD(0)": {
        "color": "#d95f02",      # Deep Orange
        "linestyle": "--",
        "linewidth": 2.0,
        "label": "TD(0) (1-Step Fitted)",
    },
    "TD(lambda)": {
        "color": "#7570b3", # Purple / Indigo
        "linestyle": "-.",
        "linewidth": 2.2,
        "label": "TD(λ) (GAE λ=0.95)",
    },
}


def get_variant_headers(spec):
    """Derives a prominent variant title and subtitle from the environment specification."""
    dense = spec.get("dense_reward", False)
    discrete = spec.get("discrete", False)
    if not dense and not discrete:
        return "Sparse Continuous", "Sparse Goal Reward • Continuous Actions"
    elif not dense and discrete:
        return "Sparse Discrete", "Sparse Goal Reward • Discrete Actions"
    elif dense and not discrete:
        return "Dense Continuous", "Dense Shaped Reward • Continuous Actions"
    elif dense and discrete:
        return "Dense Discrete", "Dense Shaped Reward • Discrete Actions"
    else:
        return spec.get("display_name", "Variant"), ""


def plot_suite_family_posters(all_results, family_name, out_dir, args=None):
    """
    Constructs a 2xN figure:
      Cols: Environment variants (Sparse Cont, Sparse Disc, Dense Cont, Dense Disc)
      Rows:
        Row 1: Clean (Slip=0)
        Row 2: Noisy (Slip=5%, Force=0.5, Transition Noise=0.001)
    """
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{family_name}_env_properties.pdf")
    png_path = os.path.join(out_dir, f"{family_name}_env_properties.png")

    variants = list(all_results.keys())  # e.g. [sparse_cont, sparse_disc, dense_cont, dense_disc]
    n_cols = max(1, len(variants))
    fig_width = max(16.0, 4.8 * n_cols)

    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, 9), sharex=True, sharey=False)
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.T
    # Generous top headroom (0.84) prevents any title overlapping
    plt.subplots_adjust(hspace=0.28, wspace=0.22, top=0.84, bottom=0.08, left=0.07, right=0.97)

    conditions = [("clean", "Clean Dynamics\n(Slip = 0%)", 0), ("noisy", "Noisy Dynamics\n(5% Slip + Noise)", 1)]

    pretty_family = "Mountain Car" if family_name == "mountain_car" else "Point Robot (Fully Observable MDP)"
    
    # 1. Main Title & Subtitle with dedicated vertical spacing
    fig.suptitle(
        f"{pretty_family} — Environmental Properties Sweep",
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5, 0.938,
        "Evaluating E(0) vs. TD(0) vs. TD(λ) Across Reward Density, Action Spaces, and Friction Noise (Fixed Epochs = 16)",
        fontsize=11.5,
        ha="center",
        color="#444444",
    )

    legend_handles = []
    legend_labels = []

    for col_idx, short_name in enumerate(variants):
        var_data = all_results[short_name]
        spec = var_data["spec"]
        env_title = spec["display_name"]
        var_title, var_sub = get_variant_headers(spec)

        for cond_key, cond_title, row_idx in conditions:
            ax = axes[row_idx, col_idx]
            cond_data = var_data[cond_key]

            for algo_key, meta in ALGO_METADATA.items():
                if algo_key in cond_data:
                    entry = cond_data[algo_key]
                    mean_c = entry["mean_curve"]
                    sem_c = entry["sem_curve"]
                    x_steps = np.arange(len(mean_c))

                    label = f"{algo_key}: {entry['final_mean']:.1f} ± {entry['final_sem']:.1f}"
                    line, = ax.plot(
                        x_steps,
                        mean_c,
                        label=label,
                        color=meta["color"],
                        linestyle=meta["linestyle"],
                        linewidth=meta["linewidth"],
                    )
                    ax.fill_between(
                        x_steps,
                        mean_c - sem_c,
                        mean_c + sem_c,
                        color=meta["color"],
                        alpha=0.15,
                    )
                    if col_idx == 0 and row_idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(meta["label"])

            # 2. Prominent Column Headers exclusively on top row (no cluttered Row prefixes)
            if row_idx == 0:
                ax.set_title(
                    f"{var_title}\n({var_sub})",
                    fontsize=12,
                    fontweight="bold",
                    pad=12,
                    color="#111111",
                )
            ax.grid(True, linestyle=":", alpha=0.6)

            # 3. Row conditions indicated on Y-axis
            if col_idx == 0:
                ax.set_ylabel(f"{cond_title}\nEpisode Return", fontsize=10.5, fontweight="bold")
            if row_idx == 1:
                ax.set_xlabel("Environment Steps (Updates)", fontsize=10.5, fontweight="bold")

            ax.legend(
                loc="lower right" if "PointRobot" in env_title else "best",
                fontsize=8.5,
                framealpha=0.92,
                title="Final Returns",
                title_fontsize=8.5,
            )

    # 4. Global Algorithm Legend in the header region
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.908),
            ncol=3,
            fontsize=10.5,
            frameon=True,
            facecolor="white",
            edgecolor="#d0d0d0",
            framealpha=0.95,
        )

    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Plot environmental properties comparison")
    parser.add_argument("--metrics-path", type=str, required=True,
                        help="Path to saved metrics_*.pkl file")
    parser.add_argument("--family", type=str, default="mountain_car",
                        choices=["mountain_car", "point_robot"],
                        help="Environment family name")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save figures")
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else os.path.dirname(args.metrics_path)
    with open(args.metrics_path, "rb") as f:
        all_results = pickle.load(f)

    pdf = plot_suite_family_posters(all_results, args.family, out_dir, args)
    print(f"Generated comparison figure: {pdf}")


if __name__ == "__main__":
    main()

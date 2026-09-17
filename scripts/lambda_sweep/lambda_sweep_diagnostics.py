"""
lambda_sweep_diagnostics.py
Consolidates the sweep results, generates cross-algorithm (lambda) plots,
and runs a diagnostic rollout for the best configurations to generate GIFs.
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import glob
import json
import shutil
import importlib
import argparse
import jax
import pandas as pd

import core.config as default_cfg
from notebooks.analyze_sweeps import (
    load_sweep_data,
    extract_best_configuration,
    plot_algorithm_comparison,
    summarize_algorithm_comparison,
)
from scripts.sweep_pipeline import ALGO_REGISTRY
from core.visualizations import generate_and_save_grid_gif


def generate_master_grid_summary(
    base_dir,
    all_tasks_data,
    win_size=750,
    pdf_filename="lambda_sweep_master_summary.pdf",
):
    """
    Generates a unified publication-quality 2-column grid PDF & PNG across all tasks:
      - Left column:  Exact E(lambda) progression
      - Right column: Exact TD(lambda) progression
      - Synchronized/shared Y-axis bounds per row for direct visual comparison.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not all_tasks_data:
        print("No task data found to generate master grid summary.")
        return None, None

    task_keys = list(all_tasks_data.keys())
    num_tasks = len(task_keys)

    print(f"\n{'='*80}")
    print(f"Generating Master 2-Column Summary ({num_tasks} Tasks) -> {pdf_filename}")
    print(f"{'='*80}")

    lambda_colors = {
        "0.0": "#1f77b4",   # Blue
        "0.5": "#ff7f0e",   # Orange
        "0.9": "#2ca02c",   # Green
        "0.95": "#d62728",  # Red
        "0.99": "#9467bd",  # Purple
        "1.0": "#8c564b",   # Brown
    }
    lambda_styles = {
        "0.0": ":",
        "0.5": "--",
        "0.9": "-.",
        "0.95": (0, (3, 1, 1, 1)),
        "0.99": "-",
        "1.0": "-",
    }

    fig, axes = plt.subplots(
        nrows=num_tasks,
        ncols=2,
        figsize=(15, 3.2 * num_tasks),
        sharex=False,
        squeeze=False,
    )

    # Column Super-Headers
    axes[0, 0].set_title("Exact $\mathbf{E(\lambda)}$ (Ours)", fontsize=14, fontweight="bold", pad=12)
    axes[0, 1].set_title("Exact $\mathbf{TD(\lambda)}$ (Baseline)", fontsize=14, fontweight="bold", pad=12)

    master_table_rows = []

    for row_idx, task_name in enumerate(task_keys):
        t_info = all_tasks_data[task_name]
        runs = t_info["runs"]
        pol = t_info["policy_type"].lower()

        if pol in ["ppo", "hybrid"]:
            metric_key = "V_start"
            rank_order = "higher"
            log_scale = False
            metric_label = "$V_{\\mathrm{start}}$"
        else:
            metric_key = "nn_weighted_VE"
            rank_order = "lower"
            log_scale = True
            metric_label = "Value Error (VE)"

        ax_e = axes[row_idx, 0]
        ax_td = axes[row_idx, 1]

        row_min_vals = []
        row_max_vals = []

        def _plot_family(ax, family_filter):
            fam_runs = {k: v for k, v in runs.items() if family_filter(k)}

            def _lmbda_val(name):
                parts = name.rsplit("_", 1)
                try:
                    return float(parts[1]) if len(parts) == 2 else 0.0
                except ValueError:
                    return 0.0

            sorted_keys = sorted(fam_runs.keys(), key=_lmbda_val)

            for pa in sorted_keys:
                s_data = fam_runs[pa]
                parts = pa.rsplit("_", 1)
                lmbda_str = parts[1] if len(parts) == 2 else "0.0"

                try:
                    actual_metric = metric_key
                    if actual_metric not in s_data["metrics"]:
                        avail = {k.lower(): k for k in s_data["metrics"].keys()}
                        if actual_metric.lower() in avail:
                            actual_metric = avail[actual_metric.lower()]
                        elif actual_metric == "V_start" and "returned_discounted_episode_returns" in s_data["metrics"]:
                            actual_metric = "returned_discounted_episode_returns"

                    trajs, best_label, _, _ = extract_best_configuration(
                        s_data,
                        metric_key=actual_metric,
                        rank_by="final_window",
                        rank_order=rank_order,
                        window_size=win_size,
                    )
                    n_seeds, time_steps = trajs.shape
                    x = np.arange(time_steps)
                    mean = trajs.mean(axis=0)
                    std = trajs.std(axis=0)

                    c = lambda_colors.get(lmbda_str, "#333333")
                    ls = lambda_styles.get(lmbda_str, "-")
                    lbl = f"$\\lambda={lmbda_str}$"

                    ax.plot(x, mean, label=lbl, color=c, linestyle=ls, linewidth=1.8)
                    if n_seeds > 1:
                        if log_scale:
                            lower = np.maximum(mean - std, mean * 0.05)
                        else:
                            lower = mean - std
                        upper = mean + std
                        ax.fill_between(x, lower, upper, color=c, alpha=0.15)
                        row_min_vals.append(lower)
                        row_max_vals.append(upper)
                    else:
                        row_min_vals.append(mean)
                        row_max_vals.append(mean)

                except Exception as e:
                    print(f"Error plotting {pa} in task {task_name}: {e}")
                    continue

        # Plot E(lambda) on Left, TD(lambda) on Right
        _plot_family(ax_e, lambda k: "exact_e" in k.lower() or "e_lambda" in k.lower())
        _plot_family(ax_td, lambda k: "exact_td" in k.lower() or "td_lambda" in k.lower())

        # Synchronize Y-Axis bounds across the two subplots in this row
        if row_min_vals and row_max_vals:
            all_low = np.concatenate([arr[arr > 0] if log_scale else arr for arr in row_min_vals] or [np.array([1e-3])])
            all_high = np.concatenate(row_max_vals)

            if len(all_low) > 0 and len(all_high) > 0:
                if log_scale:
                    ymin = max(np.percentile(all_low, 1) * 0.7, np.min(all_low) * 0.3)
                    ymax = np.percentile(all_high, 99) * 1.3
                else:
                    span = np.max(all_high) - np.min(all_low)
                    ymin = np.min(all_low) - 0.05 * span
                    ymax = np.max(all_high) + 0.05 * span

                if ymax > ymin and (ymin > 0 or not log_scale):
                    ax_e.set_ylim(bottom=ymin, top=ymax)
                    ax_td.set_ylim(bottom=ymin, top=ymax)

        if log_scale:
            ax_e.set_yscale("log")
            ax_td.set_yscale("log")

        ax_e.set_ylabel(f"{task_name}\n{metric_label}", fontsize=10, fontweight="bold")
        ax_e.grid(True, which="both", linestyle="--", alpha=0.35)
        ax_td.grid(True, which="both", linestyle="--", alpha=0.35)

        if row_idx == 0:
            ax_e.legend(loc="upper right", fontsize=9, framealpha=0.85, title="$\lambda$ (E)")
            ax_td.legend(loc="upper right", fontsize=9, framealpha=0.85, title="$\lambda$ (TD)")

        if row_idx == num_tasks - 1:
            ax_e.set_xlabel("Update Steps", fontsize=11)
            ax_td.set_xlabel("Update Steps", fontsize=11)

    fig.tight_layout()

    pdf_path = os.path.join(base_dir, pdf_filename)
    png_path = os.path.join(base_dir, pdf_filename.replace(".pdf", ".png"))

    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Master Vector PDF generated: {pdf_path}")
    print(f"Master PNG preview generated: {png_path}")
    return pdf_path, png_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, required=True, help="Sweep ID to analyze")
    return parser.parse_args()


def main():
    args = parse_args()
    sweep_id = args.sweep_id
    base_dir = os.path.join("results", "lambda_sweep", sweep_id)

    if not os.path.exists(base_dir):
        print(f"Sweep directory not found: {base_dir}")
        sys.exit(1)

    # Directory structure: results/lambda_sweep/<sweep_id>/<env_name>/<policy_type>/<pseudo_algo>/tuning/
    env_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")])

    all_tasks_data = {}

    for env_name in env_dirs:
        env_path = os.path.join(base_dir, env_name)
        policy_dirs = sorted([d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d)) and not d.startswith(".")])
        
        for policy_type in policy_dirs:
            policy_path = os.path.join(env_path, policy_type)
            task_key = f"{env_name} ({policy_type.capitalize()} Policy)"
            print("="*80)
            print(f"Analyzing Task: Env={env_name} | Policy={policy_type}")
            
            pseudo_algos = sorted([d for d in os.listdir(policy_path) if os.path.isdir(os.path.join(policy_path, d)) and d != "comparison" and not d.startswith(".")])
            if not pseudo_algos:
                continue
                
            completed_runs = {}
            for pa in pseudo_algos:
                run_dir = os.path.join(policy_path, pa, "tuning")
                if os.path.exists(run_dir):
                    try:
                        completed_runs[pa] = load_sweep_data(run_dir)
                    except Exception as e:
                        print(f"Failed to load data for {pa}: {e}")

            if not completed_runs:
                continue

            all_tasks_data[task_key] = {
                "env_name": env_name,
                "policy_type": policy_type,
                "policy_path": policy_path,
                "runs": completed_runs,
            }

            comparison_dir = os.path.join(policy_path, "comparison")
            os.makedirs(comparison_dir, exist_ok=True)

            metric_key = "V_start" if policy_type in ["ppo", "hybrid"] else "nn_weighted_VE"
            rank_by = "final_window"
            rank_order = "higher" if policy_type in ["ppo", "hybrid"] else "lower"
            window_size = 750
            log_scale = False if metric_key.lower() == "v_start" else True

            # 1. Generate Summary Dataframe
            try:
                summary_df = summarize_algorithm_comparison(
                    completed_runs,
                    metric_key=metric_key,
                    rank_by=rank_by,
                    rank_order=rank_order,
                    window_size=window_size,
                    save_path=os.path.join(comparison_dir, "comparison_summary.csv")
                )
                print(f"Summary generated for {env_name}/{policy_type}")
            except Exception as e:
                print(f"Failed to generate summary: {e}")
                summary_df = pd.DataFrame()

            # 2. Generate Comparison Plots
            try:
                import matplotlib.pyplot as plt
                base_colors = plt.cm.tab10.colors
                
                lambda_linestyles = {
                    "0.0": ":",
                    "0.5": (0, (5, 5)),
                    "0.9": "-.",
                    "0.95": "--",
                    "1.0": "-",
                }
                
                color_map = {}
                linestyle_map = {}
                base_algo_colors = {}
                color_idx = 0
                
                for pa in completed_runs.keys():
                    parts = pa.rsplit('_', 1)
                    if len(parts) == 2 and (parts[1] in lambda_linestyles or parts[1].replace('.','',1).isdigit()):
                        base_algo = parts[0]
                        lmbda = parts[1]
                    else:
                        base_algo = pa
                        lmbda = None
                        
                    if base_algo not in base_algo_colors:
                        base_algo_colors[base_algo] = base_colors[color_idx % len(base_colors)]
                        color_idx += 1
                        
                    color_map[pa] = base_algo_colors[base_algo]
                    linestyle_map[pa] = lambda_linestyles.get(lmbda, "-") if lmbda else "-"

                # A. Combined Comparison Plot (Both E and TD across all lambdas)
                plot_path = os.path.join(comparison_dir, "lambda_comparison_plot.png")
                plot_algorithm_comparison(
                    completed_runs,
                    metric_key=metric_key,
                    env_name=env_name,
                    log_scale=log_scale,
                    use_geom_mean=False,
                    rank_by=rank_by,
                    rank_order=rank_order,
                    window_size=window_size,
                    save_path=plot_path,
                    title=f"Lambda Comparison (E vs TD): {policy_type.capitalize()} Policy on {env_name}",
                    color_map=color_map,
                    linestyle_map=linestyle_map
                )
                print(f"Combined plot saved to {plot_path}")

                # B. E(lambda) Only Comparison Plot (Different colors per lambda)
                e_runs = {k: v for k, v in completed_runs.items() if "exact_e" in k.lower() or "e_lambda" in k.lower()}
                if e_runs:
                    e_color_map = {}
                    e_linestyle_map = {}
                    e_lambdas = sorted(list({pa.rsplit('_', 1)[1] for pa in e_runs.keys() if '_' in pa}))
                    for pa in e_runs.keys():
                        parts = pa.rsplit('_', 1)
                        lmbda = parts[1] if len(parts) == 2 else None
                        c_idx = e_lambdas.index(lmbda) if lmbda in e_lambdas else 0
                        e_color_map[pa] = base_colors[c_idx % len(base_colors)]
                        e_linestyle_map[pa] = lambda_linestyles.get(lmbda, "-") if lmbda else "-"

                    e_plot_path = os.path.join(comparison_dir, "exact_E_lambda_comparison_plot.png")
                    plot_algorithm_comparison(
                        e_runs,
                        metric_key=metric_key,
                        env_name=env_name,
                        log_scale=log_scale,
                        use_geom_mean=False,
                        rank_by=rank_by,
                        rank_order=rank_order,
                        window_size=window_size,
                        save_path=e_plot_path,
                        title=f"Exact E(λ) Comparison: {policy_type.capitalize()} Policy on {env_name}",
                        color_map=e_color_map,
                        linestyle_map=e_linestyle_map
                    )
                    # Also save as E_lambda_comparison_plot.png for convenient access
                    e_alt_path = os.path.join(comparison_dir, "E_lambda_comparison_plot.png")
                    if os.path.exists(e_plot_path):
                        shutil.copyfile(e_plot_path, e_alt_path)
                    print(f"E-only plot saved to {e_plot_path}")

                # C. TD(lambda) Only Comparison Plot (Different colors per lambda)
                td_runs = {k: v for k, v in completed_runs.items() if "exact_td" in k.lower() or "td_lambda" in k.lower()}
                if td_runs:
                    td_color_map = {}
                    td_linestyle_map = {}
                    td_lambdas = sorted(list({pa.rsplit('_', 1)[1] for pa in td_runs.keys() if '_' in pa}))
                    for pa in td_runs.keys():
                        parts = pa.rsplit('_', 1)
                        lmbda = parts[1] if len(parts) == 2 else None
                        c_idx = td_lambdas.index(lmbda) if lmbda in td_lambdas else 0
                        td_color_map[pa] = base_colors[c_idx % len(base_colors)]
                        td_linestyle_map[pa] = lambda_linestyles.get(lmbda, "-") if lmbda else "-"

                    td_plot_path = os.path.join(comparison_dir, "exact_td_lambda_comparison_plot.png")
                    plot_algorithm_comparison(
                        td_runs,
                        metric_key=metric_key,
                        env_name=env_name,
                        log_scale=log_scale,
                        use_geom_mean=False,
                        rank_by=rank_by,
                        rank_order=rank_order,
                        window_size=window_size,
                        save_path=td_plot_path,
                        title=f"Exact TD(λ) Comparison: {policy_type.capitalize()} Policy on {env_name}",
                        color_map=td_color_map,
                        linestyle_map=td_linestyle_map
                    )
                    # Also save as td_lambda_comparison_plot.png for convenient access
                    td_alt_path = os.path.join(comparison_dir, "td_lambda_comparison_plot.png")
                    if os.path.exists(td_plot_path):
                        shutil.copyfile(td_plot_path, td_alt_path)
                    print(f"TD-only plot saved to {td_plot_path}")

            except Exception as e:
                print(f"Failed to generate plots: {e}")
                import traceback
                traceback.print_exc()

            # 3. Generate Diagnostic GIFs for the best config of each pseudo-algorithm
            print(f"Generating diagnostic GIFs for best configs...")
            for pa, data in completed_runs.items():
                best_config = data.get("best_config")
                if not best_config:
                    print(f"No best_config found for {pa}, skipping GIF.")
                    continue

                print(f"  -> Rolling out {pa} for GIF...")
                base_algo = pa.rsplit('_', 1)[0]
                module_path = ALGO_REGISTRY.get(policy_type, {}).get(base_algo)
                if not module_path:
                    print(f"Could not find module for base algo {base_algo}")
                    continue
                
                try:
                    module = importlib.import_module(module_path)
                    make_train = getattr(module, "make_train")
                    
                    cfg_source = best_config.get("config", best_config) if isinstance(best_config, dict) else data.get("config", {})
                    cfg = cfg_source.copy()

                    for k, v in default_cfg.config.items():
                        if k not in cfg:
                            cfg[k] = v
                    if "TOTAL_TIMESTEPS" not in cfg:
                        cfg["TOTAL_TIMESTEPS"] = data.get("config", {}).get("TOTAL_TIMESTEPS", 1000)

                    cfg["LIGHT_METRICS"] = False
                    cfg["N_SEEDS"] = 1
                    
                    train_fn = make_train(cfg)
                    train_vjit = jax.jit(jax.vmap(train_fn))
                    rngs = jax.random.split(jax.random.PRNGKey(0), 1)
                    out = train_vjit(rngs)
                    metrics = out["metrics"]
                    
                    metrics = {k: jax.device_get(v) for k, v in metrics.items()}
                    
                    run_info = {
                        "name": pa,
                        "seed_idx": 0,
                        "gif_output": os.path.join(comparison_dir, f"{pa}_grid_vis.gif")
                    }
                    generate_and_save_grid_gif(run_info, save_gif=True, fps=5, max_frames=50, display_inline=False, metrics=metrics)
                    
                except Exception as e:
                    print(f"Failed to generate GIF for {pa}: {e}")
                    import traceback
                    traceback.print_exc()

    # 4. Generate Master 2-Column Summary PDF & PNG across all 12 tasks
    if all_tasks_data:
        try:
            generate_master_grid_summary(base_dir, all_tasks_data, win_size=750)
        except Exception as e:
            print(f"Failed to generate master grid summary: {e}")
            import traceback
            traceback.print_exc()

    print("="*80)
    print("Diagnostics complete!")


if __name__ == "__main__":
    main()

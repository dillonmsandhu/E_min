# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo>=0.17.6",
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
#     "pandas>=2.0.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="normal")


@app.cell
def _():
    import sys
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.path.abspath('.')
    notebooks_dir = current_dir if os.path.basename(current_dir) == 'notebooks' else os.path.join(current_dir, 'notebooks')
    repo_root = os.path.abspath(os.path.join(notebooks_dir, '..'))
    for p in [repo_root, notebooks_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Force JAX to CPU to prevent GPU VRAM exhaustion when unpickling sweep files
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

    import json
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo
    try:
        from notebooks.analyze_sweeps import (
            load_sweep_data,
            extract_best_configuration,
            find_latest_run_dir,
            discover_algorithm_sweeps,
        )
    except ImportError:
        from analyze_sweeps import (
            load_sweep_data,
            extract_best_configuration,
            find_latest_run_dir,
            discover_algorithm_sweeps,
        )
    from core.mail import email_pdf

    return (
        discover_algorithm_sweeps,
        email_pdf,
        extract_best_configuration,
        find_latest_run_dir,
        load_sweep_data,
        mo,
        np,
        os,
        pd,
        plt,
    )


@app.cell
def _():
    TASKS = [
        {"id": "mountain_car", "name": "MountainCar", "policy": "ppo", "env": "MountainCar-v0"},
        {"id": "four_rooms", "name": "FourRooms", "policy": "ppo", "env": "FourRooms-misc"},
        {"id": "eight_rooms", "name": "EightRooms", "policy": "ppo", "env": "EightRooms"},
        {"id": "whirlpool", "name": "Whirlpool", "policy": "ppo", "env": "Whirlpool"},
    ]

    EXACT_ALGOS = ["exact_E", "exact_td_lambda"]
    SAMPLED_ALGOS = ["sampled_E", "sampled_td_lambda"]

    ALGO_DISPLAY_NAMES = {
        "exact_E": "Exact E (GD)",
        "exact_E_gd": "Exact E (GD)",
        "exact_td_lambda": "Exact TD(λ)",
        "exact_td": "Exact TD(0)",
        "exact_mc": "Exact MC",
        "exact_Etd": "Exact E + TD",
        "exact_E_td": "Exact E + TD",
        "exact_td_symmetric": "Exact TD (Sym)",
        "sampled_E": "Sampled E",
        "sampled_td_lambda": "Sampled TD(λ)",
        "sampled_td": "Sampled TD(λ)",
        "sampled_mc": "Sampled MC",
        "monte_carlo": "Sampled MC",
        "td0": "Sampled TD(0)",
        "td": "Sampled TD(λ)",
        "E_td": "Sampled E + TD update",
    }

    ALGO_COLORS = {
        "exact_E": "#1f741fff",         # Green
        "exact_E_gd": "#1f741fff",      # Green
        "exact_td_lambda": "#cb8144ff", # Orange/Red
        "exact_td": "#1f77b4",          # Blue
        "exact_mc": "#492d14ff",        # Brown
        "exact_Etd": "#8c564b",         # Brown
        "exact_E_td": "#8c564b",        # Brown
        "exact_td_symmetric": "#9467bd", # Purple
        "sampled_E": "#1f741fff",       # Green
        "sampled_td_lambda": "#cb8144ff", # Orange/Red
        "sampled_td": "#cb8144ff",      # Orange/Red
        "sampled_mc": "#492d14ff",      # Brown
        "monte_carlo": "#492d14ff",     # Brown
        "td0": "#1f77b4",               # Blue
        "td": "#cb8144ff",              # Orange/Red
    }
    return ALGO_COLORS, ALGO_DISPLAY_NAMES, EXACT_ALGOS, SAMPLED_ALGOS, TASKS


@app.cell
def _(
    ALGO_DISPLAY_NAMES,
    TASKS,
    extract_best_configuration,
    get_selected_config_idx,
    os,
    pd,
):
    def generate_task_summary_table(
        all_task_data,
        algo_list,
        metric_key="V_start",
        window_size=500,
        selection_metric="V_start",
        save_path=None,
    ):
        """
        Creates a unified summary table across all 4 tasks for the specified algorithms,
        evaluating the single configuration selected by selection_metric.
        """
        task_keys = [t["id"] for t in TASKS]
        rows = []

        is_percent = ("correct" in metric_key.lower() or "acc" in metric_key.lower())

        for task_id in task_keys:
            if task_id not in all_task_data:
                continue

            task_obj = all_task_data[task_id]
            task_info = task_obj["task_info"]
            runs_dict = task_obj["runs"]

            for algo in algo_list:
                if algo not in runs_dict:
                    continue

                sweep_data = runs_dict[algo]
                try:
                    best_idx, best_label, _ = get_selected_config_idx(
                        sweep_data,
                        selection_metric=selection_metric,
                        window_size=window_size,
                        rank_by="final_window",
                    )

                    seed_trajectories, _, _, _ = extract_best_configuration(
                        sweep_data,
                        metric_key=metric_key,
                        config_idx=best_idx,
                    )
                    n_seeds, time_steps = seed_trajectories.shape
                    win = max(1, min(time_steps, window_size))

                    auc_val = float(seed_trajectories.mean())
                    window_vals = seed_trajectories[:, -win:]
                    window_mean = float(window_vals.mean())
                    window_std = float(window_vals.mean(axis=-1).std()) if n_seeds > 1 else 0.0

                    final_vals = seed_trajectories[:, -1]
                    final_mean = float(final_vals.mean())
                    final_std = float(final_vals.std()) if n_seeds > 1 else 0.0

                    min_val = float(seed_trajectories.min())
                    max_val = float(seed_trajectories.max())

                    display_name = ALGO_DISPLAY_NAMES.get(algo, algo)

                    # Format percentages for accuracy metrics, scientific notation for errors
                    if is_percent:
                        conv_str = f"{window_mean * 100:.2f}% ± {window_std * 100:.2f}%"
                        auc_str = f"{auc_val * 100:.2f}%"
                        final_str = f"{final_mean * 100:.2f}% ± {final_std * 100:.2f}%"
                        min_str = f"{min_val * 100:.2f}%"
                        max_str = f"{max_val * 100:.2f}%"
                    else:
                        conv_str = f"{window_mean:.4e} ± {window_std:.2e}"
                        auc_str = f"{auc_val:.4e}"
                        final_str = f"{final_mean:.4e} ± {final_std:.2e}"
                        min_str = f"{min_val:.4e}"
                        max_str = f"{max_val:.4e}"

                    rows.append({
                        "Task": task_info["name"],
                        "Algorithm": display_name,
                        f"Best Hyperparameters (by {selection_metric})": best_label,
                        f"Converged Window (Past {win} steps)": conv_str,
                        f"AUC ({metric_key})": auc_str,
                        f"Final ({metric_key})": final_str,
                        f"Min {metric_key}": min_str,
                        f"Max {metric_key}": max_str,
                        "Seeds": int(n_seeds),
                        "Total Steps": int(time_steps),
                        "Run Directory": sweep_data.get("run_dir", "N/A"),
                    })
                except Exception as e:
                    print(f"Error summarizing {algo} on {task_info['name']}: {e}")

        summary_df = pd.DataFrame(rows)

        if save_path and not summary_df.empty:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            if save_path.endswith(".csv"):
                summary_df.to_csv(save_path, index=False)
            elif save_path.endswith(".json"):
                summary_df.to_json(save_path, orient="records", indent=4)
            print(f"Summary table saved to {save_path}")

        return summary_df

    return (generate_task_summary_table,)


@app.cell
def _(
    TASKS,
    discover_algorithm_sweeps,
    find_latest_run_dir,
    load_sweep_data,
    os,
):
    def load_all_task_data(
        base_results_dir="results",
        custom_batches=None,
        custom_paths=None,
    ):
        """
        Discovers and loads sweep data across the 4 core tasks.

        Args:
            base_results_dir: Root results directory (default "results").
            custom_batches: List or string of batch directory paths.
            custom_paths: Dict mapping (policy, env) or task_id to batch dirs or algo paths.
        """
        # Resolve base_results_dir relative to current or parent directory
        if not os.path.exists(base_results_dir):
            parent_attempt = os.path.join("..", base_results_dir)
            if os.path.exists(parent_attempt):
                base_results_dir = parent_attempt

        all_task_data = {t["id"]: {"task_info": t, "runs": {}} for t in TASKS}

        # 1. If custom_batches provided (list or multi-line string)
        if custom_batches:
            if isinstance(custom_batches, str):
                custom_batches = [line.strip() for line in custom_batches.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
            for b_path in custom_batches:
                batch_dir = b_path
                if not os.path.exists(batch_dir):
                    parent_batch = os.path.join("..", b_path)
                    if os.path.exists(parent_batch):
                        batch_dir = parent_batch
                    else:
                        continue

                # Inspect algorithms inside batch directory
                for item in sorted(os.listdir(batch_dir)):
                    if item == "comparison" or item.startswith("."):
                        continue
                    item_path = os.path.join(batch_dir, item)
                    if not os.path.isdir(item_path):
                        continue

                    # Could be algo/tuning or direct algo dir
                    tuning_dir = os.path.join(item_path, "tuning")
                    search_dir = tuning_dir if os.path.exists(tuning_dir) else item_path
                    ts, env, run_path = find_latest_run_dir(search_dir)
                    if not run_path:
                        continue

                    try:
                        sweep = load_sweep_data(run_path)
                        env_name = sweep.get("env_name", env)
                        policy_type = "fixed" if "fixed" in batch_dir else ("ppo" if "ppo" in batch_dir else "random")
                        # Match to task
                        for task in TASKS:
                            if task["policy"] == policy_type and (task["env"] == env_name or env_name in task["env"] or task["env"] in batch_dir):
                                all_task_data[task["id"]]["runs"][item] = sweep
                    except Exception as e:
                        print(f"Error loading {item} from {batch_dir}: {e}")

        # 2. If custom_paths provided (explicit mapping)
        if custom_paths:
            for key, paths in custom_paths.items():
                target_tasks = []
                for t in TASKS:
                    if key == t["id"] or key == (t["policy"], t["env"]):
                        target_tasks.append(t["id"])
                if isinstance(paths, str):
                    paths = [paths]
                for p in paths:
                    resolved_p = p if os.path.exists(p) else os.path.join("..", p)
                    if not os.path.exists(resolved_p):
                        continue
                    for algo in os.listdir(resolved_p) if os.path.isdir(resolved_p) else [os.path.basename(resolved_p)]:
                        algo_path = os.path.join(resolved_p, algo, "tuning") if os.path.exists(os.path.join(resolved_p, algo, "tuning")) else resolved_p
                        ts, env, run_path = find_latest_run_dir(algo_path)
                        if run_path:
                            try:
                                sweep = load_sweep_data(run_path)
                                for tid in target_tasks:
                                    all_task_data[tid]["runs"][algo] = sweep
                            except Exception as e:
                                print(f"Error loading {algo} from {algo_path}: {e}")

        # 3. For any remaining tasks with missing algorithms, auto-discover latest
        for task in TASKS:
            task_id = task["id"]
            policy = task["policy"]
            env = task["env"]

            discovered = discover_algorithm_sweeps(policy=policy, env_name=env, base_results_dir=base_results_dir)
            for algo, path in discovered.items():
                if algo not in all_task_data[task_id]["runs"]:
                    try:
                        all_task_data[task_id]["runs"][algo] = load_sweep_data(path)
                    except Exception as e:
                        print(f"[{task['name']}] Could not load {algo} from {path}: {e}")

        return all_task_data

    return (load_all_task_data,)


@app.cell
def _(extract_best_configuration):
    def get_selected_config_idx(
        sweep_data,
        selection_metric="V_start",
        window_size=500,
        rank_by="final_window",
    ):
        """
        Selects the single best configuration index for this sweep run.
        Performs case-insensitive matching for selection_metric (default: 'V_start').
        """
        metrics = sweep_data.get("metrics", {})
        metric_to_use = None
        for k in metrics.keys():
            if k.lower() == selection_metric.lower():
                metric_to_use = k
                break

        if metric_to_use is None:
            for fallback in ["V_start", "mean_rew", "nn_greedy_performance", "nn_weighted_VE", "E"]:
                for k in metrics.keys():
                    if k.lower() == fallback.lower():
                        metric_to_use = k
                        break
                if metric_to_use:
                    break

        if metric_to_use is None and metrics:
            metric_to_use = list(metrics.keys())[0]

        if metric_to_use is None:
            return 0, "Default", {}

        k_lower = metric_to_use.lower()
        if any(x in k_lower for x in ["v_start", "rew", "perf", "acc", "correct"]):
            order = "higher"
        else:
            order = "lower"

        _, label, idx, hparams = extract_best_configuration(
            sweep_data,
            metric_key=metric_to_use,
            rank_by=rank_by,
            rank_order=order,
            window_size=window_size,
        )
        return idx, label, hparams

    return (get_selected_config_idx,)


@app.cell
def _(
    ALGO_COLORS,
    ALGO_DISPLAY_NAMES,
    TASKS,
    extract_best_configuration,
    get_selected_config_idx,
    np,
    os,
    plt,
):
    def plot_4task_grid(
        all_task_data,
        algo_list,
        title_prefix="Algorithms Comparison",
        metric_key="V_start",
        ylabel="Start State Value (V_start)",
        log_scale=False,
        use_geom_mean=False,
        selection_metric="V_start",
        selection_window=500,
        legend_loc="auto",
        save_path=None,
    ):
        """
        Plots a 2x2 grid comparing specified algorithms across the 4 core tasks.
        Uses the run selected via selection_metric (window_size=500).
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
        axes = axes.flatten()

        task_keys = [t["id"] for t in TASKS]

        # Determine automatic legend placement
        if legend_loc == "auto":
            loc = "lower right" if any(x in metric_key.lower() for x in ["correct", "acc", "v_start", "perf", "rew"]) else "upper right"
        else:
            loc = legend_loc

        for idx, task_id in enumerate(task_keys):
            ax = axes[idx]
            if task_id not in all_task_data:
                continue

            task_obj = all_task_data[task_id]
            task_info = task_obj["task_info"]
            runs_dict = task_obj["runs"]

            ax.set_title(task_info["name"], fontsize=13, fontweight="bold", pad=8)
            plotted_any = False

            for algo in algo_list:
                if algo not in runs_dict:
                    continue

                sweep_data = runs_dict[algo]
                try:
                    best_idx, best_label, _ = get_selected_config_idx(
                        sweep_data,
                        selection_metric=selection_metric,
                        window_size=selection_window,
                        rank_by="final_window",
                    )

                    seed_trajectories, _, _, _ = extract_best_configuration(
                        sweep_data,
                        metric_key=metric_key,
                        config_idx=best_idx,
                    )
                except Exception:
                    continue

                n_seeds, time_steps = seed_trajectories.shape
                x = list(range(time_steps))
                color = ALGO_COLORS.get(algo, "#333333")
                display_name = ALGO_DISPLAY_NAMES.get(algo, algo)
                label = f"{display_name} ({best_label})" if best_label else display_name

                # Geometric mean requires strictly positive values and log_scale
                can_use_geom = use_geom_mean and log_scale and not np.any(seed_trajectories <= 0)

                if can_use_geom:
                    safe_arr = np.maximum(seed_trajectories, 1e-18)
                    log_arr = np.log(safe_arr)
                    log_mean = np.mean(log_arr, axis=0)
                    log_std = np.std(log_arr, axis=0)
                    geom_mean = np.exp(log_mean)
                    lower = np.exp(log_mean - log_std)
                    upper = np.exp(log_mean + log_std)

                    ax.plot(x, geom_mean, label=label, color=color, linewidth=2.0)
                    if n_seeds > 1:
                        ax.fill_between(x, lower, upper, color=color, alpha=0.18)
                else:
                    mean_curve = seed_trajectories.mean(axis=0)
                    std_curve = seed_trajectories.std(axis=0)
                    ax.plot(x, mean_curve, label=label, color=color, linewidth=2.0)
                    if n_seeds > 1:
                        lower_bound = np.maximum(mean_curve - std_curve, 1e-18) if log_scale else mean_curve - std_curve
                        ax.fill_between(x, lower_bound, mean_curve + std_curve, color=color, alpha=0.18)

                plotted_any = True

            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel("Update Steps", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)

            if plotted_any:
                lines = ax.get_lines()
                if lines:
                    all_y = np.concatenate([l.get_ydata() for l in lines])
                    valid_y = all_y[np.isfinite(all_y)]
                    if log_scale:
                        valid_y = valid_y[valid_y > 0]

                    if len(valid_y) > 0:
                        ymin = np.min(valid_y)
                        ymax = np.max(valid_y)

                        if log_scale:
                            ax.set_ylim(ymin * 0.5, ymax * 2.0)
                        else:
                            pad = (ymax - ymin) * 0.08 if ymax > ymin else 0.1
                            ax.set_ylim(ymin - pad, ymax + pad)

            if plotted_any and loc is not None:
                ax.legend(loc=loc, fontsize=8.5, frameon=True)
            elif not plotted_any:
                ax.text(0.5, 0.5, "No Runs Found", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="gray")

        fig.suptitle(f"{title_prefix} — {metric_key} (Selected by {selection_metric})", fontsize=15, fontweight="bold", y=0.995)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight", dpi=120)
            print(f"Saved 4-task grid plot to {save_path}")

        return fig

    return (plot_4task_grid,)


@app.cell
def _(mo):
    mo.md("""
    # 🔬 PPO Control Sweep Benchmark Suite
    ### Cross-Algorithm & Cross-Task Learning Curves & Converged Metric Analysis
    Tasks evaluated:
    1. **MountainCar-v0**
    2. **FourRooms-misc**
    3. **EightRooms**
    4. **Whirlpool**
    """)
    return


@app.cell
def _():
    # Paste your sweep directories here to save them permanently in code/git:
    SWEEPS_TO_LOAD = [
        "results/ppo/sweeps/ppo_EightRooms_20260914_093440_exact",
        "results/ppo/sweeps/ppo_FourRooms-misc_20260914_100545_exact",
        # "results/ppo/sweeps/ppo_MountainCar-v0_20260914_125221_exact",
        "results/ppo/sweeps/ppo_Whirlpool_20260914_101645_exact",

        "results/ppo/sweeps/ppo_EightRooms_20260914_093940_sampled",
        "results/ppo/sweeps/ppo_FourRooms-misc_20260914_110726_sampled",
        "results/ppo/sweeps/ppo_MountainCar-v0_20260914_140547_sampled",
        "results/ppo/sweeps/ppo_Whirlpool_20260914_120623_sampled",
    ]
    return (SWEEPS_TO_LOAD,)


@app.cell
def _(mo):
    base_dir_input = mo.ui.text(value="results", label="Base Results Dir")
    window_size_slider = mo.ui.slider(start=10, stop=5000, step=10, value=500, label="Selection Tail Window Size")
    selection_metric_dropdown = mo.ui.dropdown(
        options=["V_start", "mean_rew", "E", "nn_weighted_VE", "nn_greedy_performance", "nn_greedy_correct"],
        value="V_start",
        label="Select Best Config By",
    )
    use_geom_mean_checkbox = mo.ui.checkbox(value=False, label="Geometric Mean Bands")

    controls = mo.hstack([base_dir_input, selection_metric_dropdown, window_size_slider, use_geom_mean_checkbox], justify="start")
    controls
    return (
        base_dir_input,
        selection_metric_dropdown,
        use_geom_mean_checkbox,
        window_size_slider,
    )


@app.cell
def _(SWEEPS_TO_LOAD, base_dir_input, load_all_task_data, mo):
    task_data = load_all_task_data(
        base_results_dir=base_dir_input.value,
        custom_batches=SWEEPS_TO_LOAD if SWEEPS_TO_LOAD else None,
    )
    total_loaded = sum(len(v["runs"]) for v in task_data.values())
    mo.md(f"✅ **Sweep Runs Loaded:** Loaded **{total_loaded}** algorithm sweep runs across 4 tasks.")
    return (task_data,)


@app.cell
def _(
    EXACT_ALGOS,
    mo,
    plot_4task_grid,
    selection_metric_dropdown,
    task_data,
    use_geom_mean_checkbox,
    window_size_slider,
):
    fig_exact = plot_4task_grid(
        task_data,
        algo_list=EXACT_ALGOS,
        title_prefix="Exact PPO Control Performance Across 4 Tasks",
        metric_key=selection_metric_dropdown.value,
        ylabel=f"Performance ({selection_metric_dropdown.value})",
        log_scale=False if any(x in selection_metric_dropdown.value.lower() for x in ["v_start", "perf", "rew", "correct", "acc"]) else True,
        use_geom_mean=use_geom_mean_checkbox.value,
        selection_metric=selection_metric_dropdown.value,
        selection_window=window_size_slider.value,
        save_path="results/comparison_exact_ppo.png",
    )

    mo.vstack([
        mo.md(f"## 📊 1. Exact Algorithms: 4-Task Performance (`{selection_metric_dropdown.value}`)"),
        mo.image(src="results/comparison_exact_ppo.png"),
    ])
    return


@app.cell
def _(email_pdf):
    # email_pdf("results/comparison_exact_ppo.png")
    return


@app.cell
def _(
    EXACT_ALGOS,
    generate_task_summary_table,
    mo,
    selection_metric_dropdown,
    task_data,
    window_size_slider,
):
    exact_table = generate_task_summary_table(
        task_data,
        algo_list=EXACT_ALGOS,
        metric_key=selection_metric_dropdown.value,
        window_size=window_size_slider.value,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/exact_algorithms_converged_summary.csv",
    )
    mo.vstack([
        mo.md(f"## 📋 Exact Algorithms: Converged Summary Table (Tail Window: Past {window_size_slider.value} steps)"),
        mo.ui.table(exact_table),
    ])
    return


@app.cell
def _(
    SAMPLED_ALGOS,
    mo,
    plot_4task_grid,
    selection_metric_dropdown,
    task_data,
    use_geom_mean_checkbox,
    window_size_slider,
):
    fig_sampled = plot_4task_grid(
        task_data,
        algo_list=SAMPLED_ALGOS,
        title_prefix="Sampled PPO Control Performance Across 4 Tasks",
        metric_key=selection_metric_dropdown.value,
        ylabel=f"Performance ({selection_metric_dropdown.value})",
        log_scale=False if any(x in selection_metric_dropdown.value.lower() for x in ["v_start", "perf", "rew", "correct", "acc"]) else True,
        use_geom_mean=use_geom_mean_checkbox.value,
        selection_metric=selection_metric_dropdown.value,
        selection_window=window_size_slider.value,
        save_path="results/comparison_sampled_ppo.png",
    )

    mo.vstack([
        mo.md(f"## 📊 2. Sampled Algorithms: 4-Task Performance (`{selection_metric_dropdown.value}`)"),
        mo.image(src="results/comparison_sampled_ppo.png"),
    ])
    return


@app.cell
def _(email_pdf):
    # email_pdf("results/comparison_sampled_ppo.png")
    return


@app.cell
def _(
    SAMPLED_ALGOS,
    generate_task_summary_table,
    mo,
    selection_metric_dropdown,
    task_data,
    window_size_slider,
):
    sampled_table = generate_task_summary_table(
        task_data,
        algo_list=SAMPLED_ALGOS,
        metric_key=selection_metric_dropdown.value,
        window_size=window_size_slider.value,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/sampled_algorithms_converged_summary.csv",
    )

    mo.vstack([
        mo.md(f"## 📋 Sampled Algorithms: Converged Summary Table (Tail Window: Past {window_size_slider.value} steps)"),
        mo.ui.table(sampled_table),
    ])
    return


if __name__ == "__main__":
    app.run()

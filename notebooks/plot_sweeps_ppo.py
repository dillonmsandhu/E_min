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
        {"id": "mountain_car", "name": " MountainCar", "policy": "ppo", "env": "MountainCar-v0"},
        {"id": "four_rooms", "name": "FourRooms", "policy": "ppo", "env": "FourRooms-misc"},
        {"id": "eight_rooms", "name": "EightRooms", "policy": "ppo", "env": "EightRooms"},
        {"id": "whirlpool", "name": "Whirlpool", "policy": "ppo", "env": "Whirlpool"},
    ]

    EXACT_ALGOS = ["exact_td", "exact E"]
    SAMPLED_ALGOS = ["sampled_td", "sampled_E"]

    ALGO_DISPLAY_NAMES = {
        "exact_td": "Exact TD(0)",
        "exact_mc": "Exact MC",
        "exact_E_gd": "Exact E (GD)",
        "exact_E": "Exact E (GD)",
        "exact_Etd": "Exact E + TD",
        "exact_E_td": "Exact E + TD",
        "exact_td_lambda": "Exact TD(λ)",
        "exact_td_symmetric": "Exact TD (Sym)",
        "sampled_td": "Sampled TD(λ)",
        "sampled_E": "Sampled E",
        "monte_carlo": "Sampled MC",
        "sampled_mc": "Sampled MC",
        "td0": "Sampled TD(0)",
        "E_td": "Sampled E + TD update",
    }

    ALGO_COLORS = {
        "exact_td": "#1f77b4",        # Blue
        "exact_mc": "#492d14ff",        # Orange
        "exact_E_gd": "#1f741fff",      # Green
        "exact_E": "#1f741fff",         # Green
        "exact_Etd": "#8c564b",       # Brown
        "exact_E_td": "#8c564b",      # Brown
        "exact_td_lambda": "#cb8144ff", # Red
        "exact_td_symmetric": "#9467bd", # Purple
        "sampled_td": "#cb8144ff",              # Blue
        "sampled_E": "#1f741fff",        # Green
        "monte_carlo": "#492d14ff",     # Orange
        "sampled_mc": "#492d14ff",              # Orange
        "td0": "#1f77b4",             # Purple
    }
    return ALGO_COLORS, ALGO_DISPLAY_NAMES, EXACT_ALGOS, SAMPLED_ALGOS, TASKS


@app.cell
def _(
    ALGO_DISPLAY_NAMES,
    extract_best_configuration,
    get_selected_config_idx,
    os,
    pd,
):
    def generate_task_summary_table(
        all_task_data,
        algo_list,
        metric_key="nn_weighted_VE",
        window_size=20,
        selection_metric="nn_greedy_correct",
        save_path=None,
    ):
        """
        Creates a unified summary table across all 4 tasks for the specified algorithms,
        evaluating the single configuration selected by selection_metric (window_size=1000).
        """
        task_keys = ["fixed_mountaincar", "fixed_fourrooms", "random_mountaincar", "random_fourrooms"]
        rows = []

        is_percent = ("correct" in metric_key or "acc" in metric_key)

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
                    # Single selection based on greedy accuracy
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
                        "Best Hyperparameters (by Greedy Acc)": best_label,
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
            custom_batches: List or string of batch directory paths, e.g.:
                ["results/fixed/sweeps/fixed_MountainCar-v0_20260828_094649", ...]
            custom_paths: Dict mapping (policy, env) or task_id to batch dirs or algo paths.
        """
        all_task_data = {t["id"]: {"task_info": t, "runs": {}} for t in TASKS}

        # 1. If custom_batches provided (list or multi-line string)
        if custom_batches:
            if isinstance(custom_batches, str):
                custom_batches = [line.strip() for line in custom_batches.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
            for batch_dir in custom_batches:
                if not os.path.exists(batch_dir):
                    print(f"Warning: Custom batch directory not found: {batch_dir}")
                    continue
                # Inspect algorithms inside batch directory
                for item in os.listdir(batch_dir):
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
                    for algo in os.listdir(p) if os.path.isdir(p) else [os.path.basename(p)]:
                        algo_path = os.path.join(p, algo, "tuning") if os.path.exists(os.path.join(p, algo, "tuning")) else p
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
        selection_metric="v_start",
        window_size=500,
        rank_by="final_window",
    ):
        """
        Selects the single best configuration index once for this sweep run based on greedy accuracy.
        Uses rank_by='final_window', window_size=1000, rank_order='higher'.
        Falls back to 'nn_weighted_VE' (rank_order='lower') if selection_metric is not available.
        """
        metrics = sweep_data.get("metrics", {})
        if selection_metric in metrics:
            metric_to_use = selection_metric
            order = "higher"
        elif "nn_weighted_VE" in metrics:
            metric_to_use = "nn_weighted_VE"
            order = "lower"
        else:
            metric_to_use = list(metrics.keys())[0]
            order = "higher"

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
        metric_key="nn_weighted_VE",
        ylabel="Value Error (nn_weighted_VE)",
        log_scale=True,
        use_geom_mean=True,
        selection_metric="nn_greedy_correct",
        selection_window=1000,
        legend_loc="auto",
        save_path=None,
    ):
        """
        Plots a 2x2 grid comparing specified algorithms across the 4 core tasks.
        Uses the single run selected via selection_metric (window_size=1000, higher is better).
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=False)
        axes = axes.flatten()

        task_keys = ["fixed_mountaincar", "fixed_fourrooms", "random_mountaincar", "random_fourrooms"]

        # Determine automatic legend placement
        if legend_loc == "auto":
            loc = "lower right" if ("correct" in metric_key or "acc" in metric_key) else "upper right"
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
                    # 1. Select the single winning configuration once based on greedy accuracy
                    best_idx, best_label, _ = get_selected_config_idx(
                        sweep_data,
                        selection_metric=selection_metric,
                        window_size=selection_window,
                        rank_by="final_window",
                    )

                    # 2. Extract trajectory for the requested metric_key for that exact best_idx
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

                if use_geom_mean:
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
                # Collect y-values strictly from the mean lines (excluding CI shading)
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
                            # 50% lower padding, 2x upper padding in log space
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
            fig.savefig(save_path, bbox_inches="tight", dpi=90)
            print(f"Saved 4-task grid plot to {save_path}")

        return fig

    return (plot_4task_grid,)


@app.cell
def _(mo):
    mo.md("""
    # 🔬 Reinforcement Learning Sweep Benchmark Suite
    ### Cross-Algorithm & Cross-Task Learning Curves & Converged Metric Analysis
    Tasks evaluated:
    1. **Fixed Policy** — MountainCar-v0
    2. **Fixed Policy** — FourRooms-misc
    3. **Random Policy** — MountainCar-v0
    4. **Random Policy** — FourRooms-misc
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
        "results/ppo/sweeps/ppo/ppo_FourRooms-misc_20260914_110726_sampled",
        "results/ppo/sweeps/ppo_MountainCar-v0_20260914_140547_sampled",
        "results/ppo/sweeps/ppo_Whirlpool_20260914_120623_sampled",
    ]
    return (SWEEPS_TO_LOAD,)


@app.cell
def _(mo):
    base_dir_input = mo.ui.text(value="../results", label="Base Results Dir")
    window_size_slider = mo.ui.slider(start=10, stop=20000, step=10, value=1800, label="Selection Tail Window Size")
    selection_metric_dropdown = mo.ui.dropdown(
        options=["nn_greedy_correct", "nn_weighted_VE", "E", "nn_greedy_performance", "V_start"],
        value="nn_greedy_performance",
        label="Select Best Config By",
    )
    use_geom_mean_checkbox = mo.ui.checkbox(value=True, label="Geometric Mean Bands")

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
        title_prefix="Exact Algorithms Performance Across 4 Tasks",
        metric_key="V_start",
        ylabel="Performance V_start",
        log_scale=True,
        use_geom_mean=use_geom_mean_checkbox.value,
        selection_metric=selection_metric_dropdown.value,
        selection_window=window_size_slider.value,
        save_path="results/comparison_exact_ppo.png",
    )

    # fig_greedy_exact = plot_4task_grid(
    #     task_data,
    #     algo_list=EXACT_ALGOS,
    #     title_prefix="Exact Algorithms Performance Across 4 Tasks",
    #     metric_key="nn_greedy_correct",
    #     ylabel="Greedy Policy Accuracy",
    #     log_scale=False,
    #     use_geom_mean=use_geom_mean_checkbox.value,
    #     selection_metric=selection_metric_dropdown.value,
    #     selection_window=window_size_slider.value,
    #     legend_loc='lower right',
    #     save_path="results/comparison_exact_4tasks_greedy_acc.png",
    # )

    # fig_greedy_val = plot_4task_grid(
    #     task_data,
    #     algo_list=EXACT_ALGOS,
    #     title_prefix="Exact Algorithms Performance Across 4 Tasks",
    #     metric_key="nn_greedy_performance",
    #     ylabel="Greedy Policy Performance",
    #     log_scale=False,
    #     use_geom_mean=use_geom_mean_checkbox.value,
    #     selection_metric=selection_metric_dropdown.value,
    #     selection_window=window_size_slider.value,
    #     legend_loc='lower right',
    #     save_path="results/comparison_exact_4tasks_greedy_val.png",
    # )

    # fig_E_exact = plot_4task_grid(
    #     task_data,
    #     algo_list=EXACT_ALGOS,
    #     title_prefix="Exact Algorithms Performance Across 4 Tasks",
    #     metric_key="E",
    #     ylabel="E (A-weighted value error)",
    #     log_scale=True,
    #     use_geom_mean=use_geom_mean_checkbox.value,
    #     selection_metric=selection_metric_dropdown.value,
    #     selection_window=window_size_slider.value,
    #     save_path="results/comparison_exact_4tasks_E.png",
    # )

    mo.vstack([
        mo.md("## 📊 2. Exact Algorithms: 4-Task Value Error (`nn_weighted_VE`) & Greedy Accuracy"),
        mo.image(src="results/comparison_exact_ppo.png"),
        # mo.image(src="results/comparison_exact_4tasks_greedy_acc.png"),
        # mo.image(src="results/comparison_exact_4tasks_greedy_val.png"),  
    ])
    return


@app.cell
def _(email_pdf):
    # email_pdf("results/comparison_exact_4tasks_VE.png")
    # email_pdf("results/comparison_exact_4tasks_greedy_acc.png")
    email_pdf("results/comparison_exact_4tasks_greedy_val.png")
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
        metric_key="nn_greedy_performance",
        window_size=window_size_slider.value,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/exact_algorithms_converged_summary.csv",
    )
    mo.ui.table(exact_table)
    return


@app.cell
def _(
    SAMPLED_ALGOS,
    mo,
    plot_4task_grid,
    plt,
    selection_metric_dropdown,
    task_data,
    use_geom_mean_checkbox,
):
    fig_sampled = plot_4task_grid(
        task_data,
        algo_list=SAMPLED_ALGOS,
        title_prefix="Sampled Algorithms Performance Across 4 Tasks",
        metric_key="nn_weighted_VE",
        ylabel="Value Error (nn_weighted_VE)",
        log_scale=True,
        selection_window = 40,
        use_geom_mean=use_geom_mean_checkbox.value,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/comparison_sampled_4tasks_VE.png",
    )

    fig_greedy_sampled = plot_4task_grid(
        task_data,
        algo_list=SAMPLED_ALGOS,
        title_prefix="Sampled Algorithms Performance Across 4 Tasks",
        metric_key="nn_greedy_correct",
        ylabel="Greedy Policy Accuracy",
        log_scale=False,
        selection_window = 40,
        legend_loc = 'lower right',
        use_geom_mean=use_geom_mean_checkbox.value,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/comparison_sampled_4tasks_greedy_acc.png",
    )

    fig_greedy_val_sampled = plot_4task_grid(
        task_data,
        algo_list=SAMPLED_ALGOS,
        title_prefix="Sampled Algorithms Performance Across 4 Tasks",
        metric_key="nn_greedy_performance",
        ylabel="Greedy Policy Value",
        log_scale=False,
        selection_window = 40,
        legend_loc = 'lower right',
        use_geom_mean=use_geom_mean_checkbox.value,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/comparison_sampled_4tasks_greedy_val.png",
    )

    plt.close('all')

    mo.vstack([
        mo.md("## 📊 1. Sampled Algorithms: 4-Task Value Error (`nn_weighted_VE`) & Greedy Accuracy"),
        mo.image(src="results/comparison_sampled_4tasks_VE.png"), 
        mo.image(src="results/comparison_sampled_4tasks_greedy_acc.png"), 
        mo.image(src="results/comparison_sampled_4tasks_greedy_val.png"), 
    ])
    return


@app.cell
def _(email_pdf):

    # email_pdf("results/comparison_sampled_4tasks_VE.png")
    # email_pdf("results/comparison_sampled_4tasks_greedy_acc.png")
    email_pdf("results/comparison_sampled_4tasks_greedy_val.png")
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
        metric_key="nn_weighted_VE",
        window_size=20,
        selection_metric=selection_metric_dropdown.value,
        save_path="results/sampled_algorithms_converged_summary.csv",
    )

    mo.vstack([
        mo.md(f"## 📋 4. Sampled Algorithms: Converged Summary Table (Tail Window: Past {window_size_slider.value} steps)"),
        mo.ui.table(sampled_table)
    ])
    return


if __name__ == "__main__":
    app.run()

# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo>=0.17.6",
#     "matplotlib>=3.7.0",
#     "numpy>=1.24.0",
#     "pandas>=2.0.0",
#     "jax>=0.4.20",
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

    # Force JAX to CPU to prevent GPU memory allocation during analysis
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

    import json
    import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import marimo as mo

    from notebooks.analyze_sweeps import (
        load_sweep_data,
        extract_best_configuration,
        plot_algorithm_comparison,
        summarize_algorithm_comparison,
        find_latest_run_dir,
    )

    return (
        extract_best_configuration,
        find_latest_run_dir,
        glob,
        json,
        load_sweep_data,
        mo,
        np,
        os,
        pd,
        plot_algorithm_comparison,
        plt,
        repo_root,
        summarize_algorithm_comparison,
        sys,
    )


@app.cell
def _(mo, os, repo_root):
    # Discover available lambda sweep directories
    _lambda_sweep_base = os.path.join(repo_root, "results", "lambda_sweep")
    discovered_sweeps = []
    if os.path.exists(_lambda_sweep_base):
        # Look for sweep timestamp folders
        for _d in sorted(os.listdir(_lambda_sweep_base), reverse=True):
            _full_p = os.path.join(_lambda_sweep_base, _d)
            if os.path.isdir(_full_p) and not _d.startswith("."):
                discovered_sweeps.append(f"results/lambda_sweep/{_d}")

    default_sweep = discovered_sweeps[0] if discovered_sweeps else "results/lambda_sweep/20260916_222811"

    mo.md(
        f"""
        # 🔬 Lambda Sweep Multi-Task Analyzer
        Analyze cross-algorithm performance across $\\lambda$ values for both **Policy Evaluation** ($E(\\lambda)$ vs. $\\text{{TD}}(\\lambda)$) and **Policy Optimization** (PPO).
        """
    )
    return default_sweep, discovered_sweeps


@app.cell
def _(default_sweep, discovered_sweeps, mo):
    sweep_input = mo.ui.dropdown(
        options=discovered_sweeps if discovered_sweeps else [default_sweep],
        value=default_sweep if default_sweep in discovered_sweeps else (discovered_sweeps[0] if discovered_sweeps else default_sweep),
        label="Select Lambda Sweep Batch:",
        allow_select_none=False,
    ) if discovered_sweeps else mo.ui.text(
        value=default_sweep,
        label="Sweep Directory Path:",
        full_width=True,
    )

    custom_sweep_text = mo.ui.text(
        value="",
        placeholder="Or type custom path (e.g. results/lambda_sweep/20260916_222811)",
        label="Custom Path Override:",
    )

    window_size_ui = mo.ui.number(
        start=1,
        stop=5000,
        step=10,
        value=750,
        label="Evaluation Window (Steps):",
    )

    ranking_criterion_ui = mo.ui.dropdown(
        options={
            "Final Window Mean": "final_window",
            "Area Under Curve (AUC / Full Trajectory Mean)": "auc",
            "Final Step Value": "final_step",
            "Minimum Across Trajectory": "min",
            "Maximum Across Trajectory": "max",
        },
        value="Final Window Mean",
        label="Ranking Criterion:",
    )

    metric_override_ui = mo.ui.dropdown(
        options={
            "Auto (nn_weighted_VE for Eval, V_start for PPO)": "auto",
            "Value Error (nn_weighted_VE)": "nn_weighted_VE",
            "Start State Value (V_start)": "V_start",
            "Greedy Policy Accuracy (nn_greedy_correct)": "nn_greedy_correct",
            "Advantage Cosine Similarity (nn_advantage_cossim)": "nn_advantage_cossim",
            "Dirichlet Energy Error (E)": "E",
            "Greedy Policy Return (nn_greedy_performance)": "nn_greedy_performance",
            "Total Loss (total_loss)": "total_loss",
        },
        value="Auto (nn_weighted_VE for Eval, V_start for PPO)",
        label="Metric Mode:",
    )

    show_plots_ui = mo.ui.checkbox(value=True, label="Show Comparison Learning Curves")
    show_individual_plots_ui = mo.ui.checkbox(value=True, label="Show Separate E(λ) and TD(λ) Plots")

    _controls = mo.vstack([
        mo.hstack([sweep_input, custom_sweep_text], justify="start", align="center"),
        mo.hstack([window_size_ui, ranking_criterion_ui, metric_override_ui], justify="start", align="center"),
        mo.hstack([show_plots_ui, show_individual_plots_ui], justify="start", align="center"),
    ])

    mo.accordion({"⚙️ Sweep Configuration & Ranking Settings": _controls})
    return (
        custom_sweep_text,
        metric_override_ui,
        ranking_criterion_ui,
        show_individual_plots_ui,
        show_plots_ui,
        sweep_input,
        window_size_ui,
    )


@app.cell
def _(
    custom_sweep_text,
    load_sweep_data,
    os,
    repo_root,
    sweep_input,
):
    # Resolve active sweep directory
    _raw_path = custom_sweep_text.value.strip() if custom_sweep_text.value.strip() else sweep_input.value
    if not os.path.isabs(_raw_path):
        active_sweep_dir = os.path.join(repo_root, _raw_path)
    else:
        active_sweep_dir = _raw_path

    # Structure: results/lambda_sweep/<sweep_id>/<env_name>/<policy_type>/<pseudo_algo>/tuning/
    tasks_data = {}
    _load_errors = []

    if os.path.exists(active_sweep_dir):
        _env_names = sorted([_d for _d in os.listdir(active_sweep_dir) if os.path.isdir(os.path.join(active_sweep_dir, _d)) and not _d.startswith(".")])
        for _env_name in _env_names:
            _env_path = os.path.join(active_sweep_dir, _env_name)
            _policy_names = sorted([_d for _d in os.listdir(_env_path) if os.path.isdir(os.path.join(_env_path, _d)) and not _d.startswith(".")])
            for _pol_name in _policy_names:
                _pol_path = os.path.join(_env_path, _pol_name)
                _task_key = f"{_env_name} ({_pol_name.capitalize()} Policy)"
                _task_id = f"{_env_name}_{_pol_name}"
                
                _algo_runs = {}
                _pseudo_algos = sorted([_d for _d in os.listdir(_pol_path) if os.path.isdir(os.path.join(_pol_path, _d)) and _d != "comparison" and not _d.startswith(".")])
                for _pa_name in _pseudo_algos:
                    _tuning_dir = os.path.join(_pol_path, _pa_name, "tuning")
                    if os.path.exists(_tuning_dir):
                        try:
                            _sweep_data = load_sweep_data(_tuning_dir)
                            _algo_runs[_pa_name] = _sweep_data
                        except Exception as _e:
                            _load_errors.append(f"Error loading {_task_key} / {_pa_name}: {_e}")
                
                if _algo_runs:
                    tasks_data[_task_key] = {
                        "task_id": _task_id,
                        "env_name": _env_name,
                        "policy_type": _pol_name,
                        "pol_path": _pol_path,
                        "runs": _algo_runs,
                    }

    return active_sweep_dir, tasks_data


@app.cell
def _(
    active_sweep_dir,
    extract_best_configuration,
    metric_override_ui,
    mo,
    pd,
    ranking_criterion_ui,
    tasks_data,
    window_size_ui,
):
    if not tasks_data:
        task_view = mo.md(
            f"""
            ⚠️ **No completed lambda sweep runs found at:** `{active_sweep_dir}`
            
            Please verify the directory path above.
            """
        )
        master_df = pd.DataFrame()
        task_summaries = {}
    else:
        # Collect all unique pseudo-algorithms across all tasks
        _all_algo_set = set()
        for _t_info in tasks_data.values():
            _all_algo_set.update(_t_info["runs"].keys())

        # Sort pseudo-algorithms logically (e.g. exact_E_lambda before exact_td_lambda, then sorted by lambda value)
        def _algo_sort_key(name):
            parts = name.rsplit('_', 1)
            base = parts[0]
            try:
                lmbda = float(parts[1]) if len(parts) == 2 else 0.0
            except ValueError:
                lmbda = 0.0
            algo_priority = 0 if "E" in base else 1
            return (algo_priority, base, lmbda)

        _all_pseudo_algos = sorted(list(_all_algo_set), key=_algo_sort_key)

        def _pretty_algo_name(name):
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                base = parts[0].replace('exact_', '').replace('_lambda', '')
                lmbda = parts[1]
                return f"{base.upper()}(λ={lmbda})"
            return name

        _win_size = int(window_size_ui.value)

        # Robustly resolve ranking criterion (handles dictionary value or label)
        _raw_rank = str(ranking_criterion_ui.value).strip()
        _rank_map = {
            "Final Window Mean": "final_window",
            "Area Under Curve (AUC / Full Trajectory Mean)": "auc",
            "Final Step Value": "final_step",
            "Minimum Across Trajectory": "min",
            "Maximum Across Trajectory": "max",
            "final_window": "final_window",
            "auc": "auc",
            "final_step": "final_step",
            "min": "min",
            "max": "max",
        }
        _rank_crit = _rank_map.get(_raw_rank, "final_window")

        # Robustly resolve metric mode (handles dictionary value, label, or custom string)
        _raw_metric = str(metric_override_ui.value).strip()
        _metric_map = {
            "Auto (nn_weighted_VE for Eval, V_start for PPO)": "auto",
            "Value Error (nn_weighted_VE)": "nn_weighted_VE",
            "Start State Value (V_start)": "V_start",
            "Greedy Policy Accuracy (nn_greedy_correct)": "nn_greedy_correct",
            "Advantage Cosine Similarity (nn_advantage_cossim)": "nn_advantage_cossim",
            "Dirichlet Energy Error (E)": "E",
            "Greedy Policy Return (nn_greedy_performance)": "nn_greedy_performance",
            "Total Loss (total_loss)": "total_loss",
            "Total Loss": "total_loss",
            "auto": "auto",
            "nn_weighted_ve": "nn_weighted_VE",
            "nn_weighted_VE": "nn_weighted_VE",
            "v_start": "V_start",
            "V_start": "V_start",
        }
        _metric_choice = _metric_map.get(_raw_metric, _raw_metric)
        if _metric_choice not in ["auto", "nn_weighted_VE", "V_start", "nn_greedy_correct", "nn_advantage_cossim", "E", "nn_greedy_performance", "total_loss"]:
            if "auto" in _metric_choice.lower():
                _metric_choice = "auto"
            elif "weighted_ve" in _metric_choice.lower():
                _metric_choice = "nn_weighted_VE"
            elif "v_start" in _metric_choice.lower():
                _metric_choice = "V_start"

        # Build master comparison table
        _table_rows = []
        task_summaries = {}

        _policy_priority = {"random": 0, "fixed": 1, "ppo": 2, "hybrid": 3}
        def _task_sort_key(item):
            _name, _info = item
            _env = _info["env_name"]
            _pol = _info["policy_type"].lower()
            return (_env, _policy_priority.get(_pol, 99))

        _sorted_tasks = sorted(tasks_data.items(), key=_task_sort_key)

        for _task_name, _t_info in _sorted_tasks:
            _pol = _t_info["policy_type"].lower()
            _runs = _t_info["runs"]

            # Determine task metric
            if _metric_choice == "auto":
                if _pol in ["ppo", "hybrid"]:
                    _metric_key = "V_start"
                    _rank_order = "higher"
                    _metric_label = f"V_start, Final Window ({_win_size} steps)"
                else:
                    _metric_key = "nn_weighted_VE"
                    _rank_order = "lower"
                    _metric_label = f"nn_weighted_VE, Final Window ({_win_size} steps)"
            else:
                _metric_key = _metric_choice
                _is_higher = _metric_key in ["V_start", "nn_greedy_correct", "nn_greedy_performance", "nn_advantage_cossim", "reward", "return", "returned_discounted_episode_returns"]
                _rank_order = "higher" if _is_higher else "lower"
                _metric_label = f"{_metric_key}, Final Window ({_win_size} steps)"

            _row_data = {
                "Task": _task_name,
                "Evaluation Metric": _metric_label,
            }

            _task_algo_scores = {}
            _task_algo_formatted = {}

            for _pa in _all_pseudo_algos:
                if _pa not in _runs:
                    _row_data[_pretty_algo_name(_pa)] = "—"
                    continue

                _sweep_data = _runs[_pa]
                try:
                    # Fallback check for PPO return metrics if V_start is not present
                    _actual_metric = _metric_key
                    if _actual_metric not in _sweep_data["metrics"]:
                        _avail = list(_sweep_data["metrics"].keys())
                        _lower_avail = {k.lower(): k for k in _avail}
                        if _actual_metric.lower() in _lower_avail:
                            _actual_metric = _lower_avail[_actual_metric.lower()]
                        elif _actual_metric == "V_start" and "returned_discounted_episode_returns" in _avail:
                            _actual_metric = "returned_discounted_episode_returns"

                    _seed_trajectories, _best_label, _best_idx, _best_hparams = extract_best_configuration(
                        _sweep_data,
                        metric_key=_actual_metric,
                        rank_by=_rank_crit,
                        rank_order=_rank_order,
                        window_size=_win_size,
                    )
                    _n_seeds, _time_steps = _seed_trajectories.shape
                    _actual_win = max(1, min(_time_steps, _win_size))

                    # Compute metric score for cell display (mean over seeds in final window)
                    _seed_window_means = _seed_trajectories[:, -_actual_win:].mean(axis=-1)
                    _mean_val = float(_seed_window_means.mean())
                    _std_val = float(_seed_window_means.std()) if _n_seeds > 1 else 0.0

                    _task_algo_scores[_pa] = _mean_val

                    # Formatting: clean scientific or fixed float
                    if abs(_mean_val) < 1e-2 and abs(_mean_val) > 0:
                        _fmt_str = f"{_mean_val:.2e} ± {_std_val:.1e}" if _n_seeds > 1 else f"{_mean_val:.2e}"
                    else:
                        _fmt_str = f"{_mean_val:.4f} ± {_std_val:.4f}" if _n_seeds > 1 else f"{_mean_val:.4f}"

                    _task_algo_formatted[_pa] = {
                        "mean": _mean_val,
                        "std": _std_val,
                        "fmt": _fmt_str,
                        "best_label": _best_label,
                    }
                    _row_data[_pretty_algo_name(_pa)] = _fmt_str
                except Exception as _e:
                    _row_data[_pretty_algo_name(_pa)] = f"Err ({type(_e).__name__})"
                    _task_algo_scores[_pa] = float("inf") if _rank_order == "lower" else float("-inf")

            # Identify winning algorithm for this task
            if _task_algo_scores:
                _valid_scores = {_k: _v for _k, _v in _task_algo_scores.items() if _v not in [float("inf"), float("-inf")]}
                if _valid_scores:
                    if _rank_order == "lower":
                        _winning_algo = min(_valid_scores, key=_valid_scores.get)
                    else:
                        _winning_algo = max(_valid_scores, key=_valid_scores.get)
                    
                    _winning_col = _pretty_algo_name(_winning_algo)
                    if _winning_col in _row_data:
                        _row_data[_winning_col] = f"**{_row_data[_winning_col]} 🏆**"
                    _row_data["Winning Algorithm"] = f"**{_winning_col}**"
                else:
                    _winning_algo = None
                    _row_data["Winning Algorithm"] = "—"
            else:
                _winning_algo = None
                _row_data["Winning Algorithm"] = "—"

            _table_rows.append(_row_data)
            task_summaries[_task_name] = {
                "info": _t_info,
                "metric_key": _metric_key,
                "rank_order": _rank_order,
                "metric_label": _metric_label,
                "winning_algo": _winning_algo,
                "algo_results": _task_algo_formatted,
            }

        master_df = pd.DataFrame(_table_rows)

        # Move Winning Algorithm to the front after Task and Metric
        _cols = list(master_df.columns)
        if "Winning Algorithm" in _cols:
            _cols.remove("Winning Algorithm")
            _cols.insert(2, "Winning Algorithm")
            master_df = master_df[_cols]

        # Convert DataFrame to high-impact Markdown table
        _md_table_lines = []
        _headers = list(master_df.columns)
        _md_table_lines.append("| " + " | ".join(_headers) + " |")
        _md_table_lines.append("| " + " | ".join(["---"] * len(_headers)) + " |")
        for _, _r in master_df.iterrows():
            _row_cells = [str(_r[_h]) for _h in _headers]
            _md_table_lines.append("| " + " | ".join(_row_cells) + " |")
        
        _markdown_table_str = "\n".join(_md_table_lines)

        task_view = mo.vstack([
            mo.md("### 📊 Master Sweep Comparison Table"),
            mo.md(_markdown_table_str),
        ])

    task_view
    return (
        master_df,
        task_summaries,
        task_view,
    )


@app.cell
def _(mo, task_summaries):
    if not task_summaries:
        task_selector_ui = None
        selector_view = mo.md("")
    else:
        _options = ["All Tasks"] + list(task_summaries.keys())
        task_selector_ui = mo.ui.dropdown(
            options=_options,
            value=_options[0],
            label="🎯 Select Task to View Learning Curves:",
        )
        selector_view = mo.hstack([task_selector_ui], justify="start", align="center")

    selector_view
    return selector_view, task_selector_ui


@app.cell
def _(
    active_sweep_dir,
    mo,
    os,
    plot_algorithm_comparison,
    plt,
    show_individual_plots_ui,
    show_plots_ui,
    task_selector_ui,
    task_summaries,
    window_size_ui,
):
    if not task_summaries or not show_plots_ui.value or task_selector_ui is None:
        plots_view = mo.md("")
    else:
        _selected_task = task_selector_ui.value
        _tasks_to_render = (
            list(task_summaries.items())
            if _selected_task == "All Tasks"
            else [(_selected_task, task_summaries[_selected_task])]
            if _selected_task in task_summaries
            else []
        )

        _plots_dir = os.path.join(active_sweep_dir, "plots")
        os.makedirs(_plots_dir, exist_ok=True)

        # Intuitive Cold-to-Warm chromatic spectrum:
        # Cold Blue (TD(0), pure bootstrapping) -> Teal -> Orange -> Crimson Red -> Deep Purple -> Black (Monte Carlo, pure returns)
        _lambda_colors = {
            "0.0": "#1f77b4",   # Deep Royal Blue (Pure Bootstrap, lambda=0)
            "0.5": "#17becf",   # Cyan / Teal (Intermediate)
            "0.9": "#ff7f0e",   # Amber Orange
            "0.95": "#d62728",  # Crimson Red
            "0.99": "#6a1b9a",  # Deep Royal Purple (Near-MC, lambda=0.99)
            "1.0": "#111111",   # Dark Charcoal / Black (Pure Monte Carlo)
        }
        _lambda_linestyles = {
            "0.0": ":",             # Dotted
            "0.5": "-.",            # Dash-dot
            "0.9": "--",            # Dashed
            "0.95": (0, (5, 2)),    # Long dash
            "0.99": "-",            # Solid
            "1.0": "-",             # Solid bold
        }

        _win_size = int(window_size_ui.value)
        _task_plot_elements = []

        for _task_name, _summary in _tasks_to_render:
            _t_info = _summary["info"]
            _task_id = _t_info["task_id"]
            _env = _t_info["env_name"]
            _runs = _t_info["runs"]
            _metric_key = _summary["metric_key"]
            _rank_order = _summary["rank_order"]
            _winning_algo = _summary["winning_algo"]

            _log_scale = False if _metric_key.lower() == "v_start" or _rank_order == "higher" else True

            # Color and Linestyle maps for combined plot
            _color_map = {}
            _linestyle_map = {}

            for _pa in _runs.keys():
                _parts = _pa.rsplit('_', 1)
                _lmbda = _parts[1] if len(_parts) == 2 else None

                _color_map[_pa] = _lambda_colors.get(_lmbda, "#333333")
                _linestyle_map[_pa] = _lambda_linestyles.get(_lmbda, "-") if _lmbda else "-"

            # 1. Combined Plot (E vs TD) saved directly to disk
            _combined_png_path = os.path.join(_plots_dir, f"{_task_id}_combined.png")
            _fig_combined = plot_algorithm_comparison(
                _runs,
                metric_key=_metric_key,
                env_name=_env,
                log_scale=_log_scale,
                use_geom_mean=False,
                rank_by="final_window",
                rank_order=_rank_order,
                window_size=_win_size,
                title=f"Combined Lambda Comparison: {_task_name}",
                color_map=_color_map,
                linestyle_map=_linestyle_map,
                save_path=_combined_png_path,
            )
            if _fig_combined is not None:
                plt.close(_fig_combined)

            # 2. Sub-plots for E only and TD only if requested
            _e_png_path = None
            _td_png_path = None
            if show_individual_plots_ui.value:
                # E-only
                _e_runs = {_k: _v for _k, _v in _runs.items() if "exact_e" in _k.lower() or "e_lambda" in _k.lower()}
                if _e_runs:
                    _e_color_map = {}
                    _e_linestyle_map = {}
                    for _pa_e in _e_runs.keys():
                        _parts_e = _pa_e.rsplit('_', 1)
                        _lmbda_e = _parts_e[1] if len(_parts_e) == 2 else None
                        _e_color_map[_pa_e] = _lambda_colors.get(_lmbda_e, "#333333")
                        _e_linestyle_map[_pa_e] = _lambda_linestyles.get(_lmbda_e, "-") if _lmbda_e else "-"

                    _e_png_path = os.path.join(_plots_dir, f"{_task_id}_e_lambda.png")
                    _fig_e = plot_algorithm_comparison(
                        _e_runs,
                        metric_key=_metric_key,
                        env_name=_env,
                        log_scale=_log_scale,
                        use_geom_mean=False,
                        rank_by="final_window",
                        rank_order=_rank_order,
                        window_size=_win_size,
                        title=f"Exact E(λ) Progression: {_task_name}",
                        color_map=_e_color_map,
                        linestyle_map=_e_linestyle_map,
                        save_path=_e_png_path,
                    )
                    if _fig_e is not None:
                        plt.close(_fig_e)

                # TD-only
                _td_runs = {_k: _v for _k, _v in _runs.items() if "exact_td" in _k.lower() or "td_lambda" in _k.lower()}
                if _td_runs:
                    _td_color_map = {}
                    _td_linestyle_map = {}
                    for _pa_td in _td_runs.keys():
                        _parts_td = _pa_td.rsplit('_', 1)
                        _lmbda_td = _parts_td[1] if len(_parts_td) == 2 else None
                        _td_color_map[_pa_td] = _lambda_colors.get(_lmbda_td, "#333333")
                        _td_linestyle_map[_pa_td] = _lambda_linestyles.get(_lmbda_td, "-") if _lmbda_td else "-"

                    _td_png_path = os.path.join(_plots_dir, f"{_task_id}_td_lambda.png")
                    _fig_td = plot_algorithm_comparison(
                        _td_runs,
                        metric_key=_metric_key,
                        env_name=_env,
                        log_scale=_log_scale,
                        use_geom_mean=False,
                        rank_by="final_window",
                        rank_order=_rank_order,
                        window_size=_win_size,
                        title=f"Exact TD(λ) Progression: {_task_name}",
                        color_map=_td_color_map,
                        linestyle_map=_td_linestyle_map,
                        save_path=_td_png_path,
                    )
                    if _fig_td is not None:
                        plt.close(_fig_td)

            # Ensure all figures are closed
            plt.close("all")

            # Build Per-Task UI Card using mo.image with local file path
            _winning_text = f"🏆 **Winning Algorithm:** `{_winning_algo}`" if _winning_algo else ""
            _task_card_content = [
                mo.md(f"## 🎯 Task: {_task_name}"),
                mo.md(f"**Metric Evaluated:** `{_summary['metric_label']}` | {_winning_text}"),
            ]

            if os.path.exists(_combined_png_path):
                _task_card_content.append(mo.image(src=_combined_png_path, rounded=True))

            if _e_png_path and _td_png_path and os.path.exists(_e_png_path) and os.path.exists(_td_png_path):
                _task_card_content.append(
                    mo.hstack([
                        mo.image(src=_e_png_path, rounded=True),
                        mo.image(src=_td_png_path, rounded=True),
                    ], justify="space-around")
                )
            elif _e_png_path and os.path.exists(_e_png_path):
                _task_card_content.append(mo.image(src=_e_png_path, rounded=True))
            elif _td_png_path and os.path.exists(_td_png_path):
                _task_card_content.append(mo.image(src=_td_png_path, rounded=True))

            _task_plot_elements.append(mo.vstack(_task_card_content))

        plots_view = mo.vstack([
            mo.md("---"),
            mo.md("## 📈 Cross-Algorithm Learning Curves per Task"),
            *_task_plot_elements
        ])

    plots_view
    return (plots_view,)


@app.cell
def _(
    active_sweep_dir,
    extract_best_configuration,
    mo,
    np,
    os,
    plt,
    task_summaries,
    tasks_data,
    window_size_ui,
):
    if not tasks_data or not task_summaries:
        export_view = mo.md("")
    else:
        _policy_priority = {"random": 0, "fixed": 1, "ppo": 2, "hybrid": 3}
        def _task_sort_key(name):
            _info = tasks_data[name]
            _env = _info["env_name"]
            _pol = _info["policy_type"].lower()
            return (_env, _policy_priority.get(_pol, 99))

        _task_keys = sorted(list(tasks_data.keys()), key=_task_sort_key)
        _num_tasks = len(_task_keys)

        # Intuitive Cold-to-Warm chromatic spectrum:
        # Cold Blue (TD(0), pure bootstrapping) -> Teal -> Orange -> Crimson Red -> Deep Purple -> Black (Monte Carlo, pure returns)
        _lambda_colors = {
            "0.0": "#1f77b4",   # Deep Royal Blue (Pure Bootstrap, lambda=0)
            "0.5": "#17becf",   # Cyan / Teal (Intermediate)
            "0.9": "#ff7f0e",   # Amber Orange
            "0.95": "#d62728",  # Crimson Red
            "0.99": "#6a1b9a",  # Deep Royal Purple (Near-MC, lambda=0.99)
            "1.0": "#111111",   # Dark Charcoal / Black (Pure Monte Carlo)
        }
        _lambda_styles = {
            "0.0": ":",             # Dotted
            "0.5": "-.",            # Dash-dot
            "0.9": "--",            # Dashed
            "0.95": (0, (5, 2)),    # Long dash
            "0.99": "-",            # Solid
            "1.0": "-",             # Solid bold
        }

        _win_size = int(window_size_ui.value)

        _fig, _axes = plt.subplots(
            nrows=_num_tasks,
            ncols=2,
            figsize=(15, 3.2 * _num_tasks),
            sharex=False,
            squeeze=False,
        )

        _axes[0, 0].set_title("Exact $\mathbf{E(\lambda)}$", fontsize=14, fontweight="bold", pad=12)
        _axes[0, 1].set_title("Exact $\mathbf{TD(\lambda)}$", fontsize=14, fontweight="bold", pad=12)

        for _row_idx, _task_name in enumerate(_task_keys):
            _t_info = tasks_data[_task_name]
            _runs = _t_info["runs"]
            _pol = _t_info["policy_type"].lower()

            if _pol in ["ppo", "hybrid"]:
                _metric_key = "V_start"
                _rank_order = "higher"
                _log_scale = False
                _metric_label = "$V_{\\mathrm{start}}$"
            else:
                _metric_key = "nn_weighted_VE"
                _rank_order = "lower"
                _log_scale = True
                _metric_label = "Value Error (VE)"

            _ax_e = _axes[_row_idx, 0]
            _ax_td = _axes[_row_idx, 1]

            _row_min_vals = []
            _row_max_vals = []

            def _plot_family(ax, family_filter):
                _fam_runs = {_k: _v for _k, _v in _runs.items() if family_filter(_k)}

                def _lmbda_val(name):
                    parts = name.rsplit("_", 1)
                    try:
                        return float(parts[1]) if len(parts) == 2 else 0.0
                    except ValueError:
                        return 0.0

                _sorted_keys = sorted(_fam_runs.keys(), key=_lmbda_val)

                for _pa in _sorted_keys:
                    _s_data = _fam_runs[_pa]
                    _parts = _pa.rsplit("_", 1)
                    _lmbda_str = _parts[1] if len(_parts) == 2 else "0.0"

                    try:
                        _actual_metric = _metric_key
                        if _actual_metric not in _s_data["metrics"]:
                            _avail = {_k.lower(): _k for _k in _s_data["metrics"].keys()}
                            if _actual_metric.lower() in _avail:
                                _actual_metric = _avail[_actual_metric.lower()]
                            elif _actual_metric == "V_start" and "returned_discounted_episode_returns" in _s_data["metrics"]:
                                _actual_metric = "returned_discounted_episode_returns"

                        _trajs, _best_label, _, _ = extract_best_configuration(
                            _s_data,
                            metric_key=_actual_metric,
                            rank_by="final_window",
                            rank_order=_rank_order,
                            window_size=_win_size,
                        )
                        _n_seeds, _time_steps = _trajs.shape
                        _x = np.arange(_time_steps)
                        _mean = _trajs.mean(axis=0)
                        _std = _trajs.std(axis=0)

                        _c = _lambda_colors.get(_lmbda_str, "#333333")
                        _ls = _lambda_styles.get(_lmbda_str, "-")
                        _lbl = f"$\\lambda={_lmbda_str}$"

                        ax.plot(_x, _mean, label=_lbl, color=_c, linestyle=_ls, linewidth=1.8)
                        if _n_seeds > 1:
                            if _log_scale:
                                _lower = np.maximum(_mean - _std, _mean * 0.05)
                            else:
                                _lower = _mean - _std
                            _upper = _mean + _std
                            ax.fill_between(_x, _lower, _upper, color=_c, alpha=0.15)
                            _row_min_vals.append(_lower)
                            _row_max_vals.append(_upper)
                        else:
                            _row_min_vals.append(_mean)
                            _row_max_vals.append(_mean)

                    except Exception as _e:
                        continue

            _plot_family(_ax_e, lambda k: "exact_e" in k.lower() or "e_lambda" in k.lower())
            _plot_family(_ax_td, lambda k: "exact_td" in k.lower() or "td_lambda" in k.lower())

            if _row_min_vals and _row_max_vals:
                _all_low = np.concatenate([_arr[_arr > 0] if _log_scale else _arr for _arr in _row_min_vals] or [np.array([1e-3])])
                _all_high = np.concatenate(_row_max_vals)

                if len(_all_low) > 0 and len(_all_high) > 0:
                    if _log_scale:
                        _ymin = max(np.percentile(_all_low, 1) * 0.7, np.min(_all_low) * 0.3)
                        _ymax = np.percentile(_all_high, 99) * 1.3
                    else:
                        _span = np.max(_all_high) - np.min(_all_low)
                        _ymin = np.min(_all_low) - 0.05 * _span
                        _ymax = np.max(_all_high) + 0.05 * _span

                    if _ymax > _ymin and (_ymin > 0 or not _log_scale):
                        _ax_e.set_ylim(bottom=_ymin, top=_ymax)
                        _ax_td.set_ylim(bottom=_ymin, top=_ymax)

            if _log_scale:
                _ax_e.set_yscale("log")
                _ax_td.set_yscale("log")

            _ax_e.set_ylabel(f"{_task_name}\n{_metric_label}", fontsize=10, fontweight="bold")
            _ax_e.grid(True, which="both", linestyle="--", alpha=0.35)
            _ax_td.grid(True, which="both", linestyle="--", alpha=0.35)

            if _row_idx == 0:
                _ax_e.legend(loc="upper right", fontsize=9, framealpha=0.85, title="$\lambda$ (E)")
                _ax_td.legend(loc="upper right", fontsize=9, framealpha=0.85, title="$\lambda$ (TD)")

            if _row_idx == _num_tasks - 1:
                _ax_e.set_xlabel("Update Steps", fontsize=11)
                _ax_td.set_xlabel("Update Steps", fontsize=11)

        _fig.tight_layout()

        _pdf_path = os.path.join(active_sweep_dir, "lambda_sweep_master_summary.pdf")
        _fig.savefig(_pdf_path, format="pdf", bbox_inches="tight")
        plt.close(_fig)

        export_view = mo.vstack([
            mo.md("---"),
            mo.md("### 📄 Master 12-Task Advisor Vector PDF"),
            mo.md(f"✅ **Master PDF Saved:** `{_pdf_path}`"),
            mo.download(
                data=open(_pdf_path, "rb").read() if os.path.exists(_pdf_path) else b"",
                filename="lambda_sweep_master_summary.pdf",
                label="⬇️ Download Advisor Summary (Vector PDF)",
            ),
        ])

    export_view
    return (export_view,)


if __name__ == "__main__":
    app.run()

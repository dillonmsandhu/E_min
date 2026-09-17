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
app = marimo.App(width="wide")


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
def _(glob, mo, os, repo_root):
    # Discover available lambda sweep directories
    lambda_sweep_base = os.path.join(repo_root, "results", "lambda_sweep")
    discovered_sweeps = []
    if os.path.exists(lambda_sweep_base):
        # Look for sweep timestamp folders
        for d in sorted(os.listdir(lambda_sweep_base), reverse=True):
            full_p = os.path.join(lambda_sweep_base, d)
            if os.path.isdir(full_p) and not d.startswith("."):
                discovered_sweeps.append(f"results/lambda_sweep/{d}")

    default_sweep = discovered_sweeps[0] if discovered_sweeps else "results/lambda_sweep/20260916_222811"

    mo.md(
        f"""
        # 🔬 Lambda Sweep Multi-Task Analyzer
        Analyze cross-algorithm performance across $\\lambda$ values for both **Policy Evaluation** ($E(\\lambda)$ vs. $\\text{{TD}}(\\lambda)$) and **Policy Optimization** (PPO).
        """
    )
    return default_sweep, discovered_sweeps, lambda_sweep_base


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
            "final_window": "Final Window Mean",
            "auc": "Area Under Curve (AUC / Full Trajectory Mean)",
            "final_step": "Final Step Value",
            "min": "Minimum Across Trajectory",
            "max": "Maximum Across Trajectory",
        },
        value="final_window",
        label="Ranking Criterion:",
    )

    metric_override_ui = mo.ui.dropdown(
        options={
            "auto": "Auto (nn_weighted_VE for Eval, V_start for PPO)",
            "nn_weighted_VE": "Value Error (nn_weighted_VE)",
            "V_start": "Start State Value (V_start)",
            "nn_greedy_correct": "Greedy Policy Accuracy (nn_greedy_correct)",
            "nn_advantage_cossim": "Advantage Cosine Similarity (nn_advantage_cossim)",
            "E": "Dirichlet Energy Error (E)",
            "nn_greedy_performance": "Greedy Policy Return (nn_greedy_performance)",
            "total_loss": "Total Loss",
        },
        value="auto",
        label="Metric Mode:",
    )

    show_plots_ui = mo.ui.checkbox(value=True, label="Show Comparison Learning Curves")
    show_individual_plots_ui = mo.ui.checkbox(value=True, label="Show Separate E(λ) and TD(λ) Plots")

    controls = mo.vstack([
        mo.hstack([sweep_input, custom_sweep_text], justify="start", align="center"),
        mo.hstack([window_size_ui, ranking_criterion_ui, metric_override_ui], justify="start", align="center"),
        mo.hstack([show_plots_ui, show_individual_plots_ui], justify="start", align="center"),
    ])

    mo.accordion({"⚙️ Sweep Configuration & Ranking Settings": controls})
    return (
        controls,
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
    raw_path = custom_sweep_text.value.strip() if custom_sweep_text.value.strip() else sweep_input.value
    if not os.path.isabs(raw_path):
        active_sweep_dir = os.path.join(repo_root, raw_path)
    else:
        active_sweep_dir = raw_path

    # Structure: results/lambda_sweep/<sweep_id>/<env_name>/<policy_type>/<pseudo_algo>/tuning/
    tasks_data = {}
    load_errors = []

    if os.path.exists(active_sweep_dir):
        env_names = sorted([d for d in os.listdir(active_sweep_dir) if os.path.isdir(os.path.join(active_sweep_dir, d)) and not d.startswith(".")])
        for env in env_names:
            env_path = os.path.join(active_sweep_dir, env)
            policy_names = sorted([d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d)) and not d.startswith(".")])
            for pol in policy_names:
                pol_path = os.path.join(env_path, pol)
                task_key = f"{env} ({pol.capitalize()} Policy)"
                task_id = f"{env}_{pol}"
                
                algo_runs = {}
                pseudo_algos = sorted([d for d in os.listdir(pol_path) if os.path.isdir(os.path.join(pol_path, d)) and d != "comparison" and not d.startswith(".")])
                for pa in pseudo_algos:
                    tuning_dir = os.path.join(pol_path, pa, "tuning")
                    if os.path.exists(tuning_dir):
                        try:
                            sweep_data = load_sweep_data(tuning_dir)
                            algo_runs[pa] = sweep_data
                        except Exception as e:
                            load_errors.append(f"Error loading {task_key} / {pa}: {e}")
                
                if algo_runs:
                    tasks_data[task_key] = {
                        "task_id": task_id,
                        "env_name": env,
                        "policy_type": pol,
                        "pol_path": pol_path,
                        "runs": algo_runs,
                    }

    return active_sweep_dir, load_errors, raw_path, tasks_data


@app.cell
def _(
    active_sweep_dir,
    extract_best_configuration,
    load_errors,
    metric_override_ui,
    mo,
    np,
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
        all_pseudo_algos = []
    else:
        # Collect all unique pseudo-algorithms across all tasks
        all_algo_set = set()
        for t_info in tasks_data.values():
            all_algo_set.update(t_info["runs"].keys())

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

        all_pseudo_algos = sorted(list(all_algo_set), key=_algo_sort_key)

        def _pretty_algo_name(name):
            parts = name.rsplit('_', 1)
            if len(parts) == 2:
                base = parts[0].replace('exact_', '').replace('_lambda', '')
                lmbda = parts[1]
                return f"{base.upper()}(λ={lmbda})"
            return name

        win_size = int(window_size_ui.value)
        rank_crit = ranking_criterion_ui.value
        metric_choice = metric_override_ui.value

        # Build master comparison table
        table_rows = []
        task_summaries = {}

        for task_name, t_info in tasks_data.items():
            env = t_info["env_name"]
            pol = t_info["policy_type"]
            runs = t_info["runs"]

            # Determine task metric
            if metric_choice == "auto":
                if pol in ["ppo", "hybrid"]:
                    metric_key = "V_start"
                    rank_order = "higher"
                    metric_label = f"V_start, Final Window ({win_size} steps)"
                else:
                    metric_key = "nn_weighted_VE"
                    rank_order = "lower"
                    metric_label = f"nn_weighted_VE, Final Window ({win_size} steps)"
            else:
                metric_key = metric_choice
                is_higher = metric_key in ["V_start", "nn_greedy_correct", "nn_greedy_performance", "nn_advantage_cossim", "reward", "return"]
                rank_order = "higher" if is_higher else "lower"
                metric_label = f"{metric_key}, Final Window ({win_size} steps)"

            row_data = {
                "Task": task_name,
                "Evaluation Metric": metric_label,
            }

            task_algo_scores = {}
            task_algo_formatted = {}

            for pa in all_pseudo_algos:
                if pa not in runs:
                    row_data[_pretty_algo_name(pa)] = "—"
                    continue

                sweep_data = runs[pa]
                try:
                    seed_trajectories, best_label, best_idx, best_hparams = extract_best_configuration(
                        sweep_data,
                        metric_key=metric_key,
                        rank_by=rank_crit,
                        rank_order=rank_order,
                        window_size=win_size,
                    )
                    n_seeds, time_steps = seed_trajectories.shape
                    actual_win = max(1, min(time_steps, win_size))

                    # Compute metric score for cell display (mean over seeds in final window)
                    seed_window_means = seed_trajectories[:, -actual_win:].mean(axis=-1)
                    mean_val = float(seed_window_means.mean())
                    std_val = float(seed_window_means.std()) if n_seeds > 1 else 0.0

                    task_algo_scores[pa] = mean_val

                    # Formatting: clean scientific or fixed float
                    if abs(mean_val) < 1e-2 and abs(mean_val) > 0:
                        fmt_str = f"{mean_val:.2e} ± {std_val:.1e}" if n_seeds > 1 else f"{mean_val:.2e}"
                    else:
                        fmt_str = f"{mean_val:.4f} ± {std_val:.4f}" if n_seeds > 1 else f"{mean_val:.4f}"

                    task_algo_formatted[pa] = {
                        "mean": mean_val,
                        "std": std_val,
                        "fmt": fmt_str,
                        "best_label": best_label,
                    }
                    row_data[_pretty_algo_name(pa)] = fmt_str
                except Exception as e:
                    row_data[_pretty_algo_name(pa)] = "Error"
                    task_algo_scores[pa] = float("inf") if rank_order == "lower" else float("-inf")

            # Identify winning algorithm for this task
            if task_algo_scores:
                valid_scores = {k: v for k, v in task_algo_scores.items() if v not in [float("inf"), float("-inf")]}
                if valid_scores:
                    if rank_order == "lower":
                        winning_algo = min(valid_scores, key=valid_scores.get)
                    else:
                        winning_algo = max(valid_scores, key=valid_scores.get)
                    
                    winning_col = _pretty_algo_name(winning_algo)
                    if winning_col in row_data:
                        row_data[winning_col] = f"**{row_data[winning_col]} 🏆**"
                    row_data["Winning Algorithm"] = f"**{winning_col}**"
                else:
                    winning_algo = None
                    row_data["Winning Algorithm"] = "—"
            else:
                winning_algo = None
                row_data["Winning Algorithm"] = "—"

            table_rows.append(row_data)
            task_summaries[task_name] = {
                "info": t_info,
                "metric_key": metric_key,
                "rank_order": rank_order,
                "metric_label": metric_label,
                "winning_algo": winning_algo,
                "algo_results": task_algo_formatted,
            }

        master_df = pd.DataFrame(table_rows)

        # Move Winning Algorithm to the front after Task and Metric
        cols = list(master_df.columns)
        if "Winning Algorithm" in cols:
            cols.remove("Winning Algorithm")
            cols.insert(2, "Winning Algorithm")
            master_df = master_df[cols]

        # Convert DataFrame to high-impact Markdown table
        md_table_lines = []
        headers = list(master_df.columns)
        md_table_lines.append("| " + " | ".join(headers) + " |")
        md_table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, r in master_df.iterrows():
            row_cells = [str(r[h]) for h in headers]
            md_table_lines.append("| " + " | ".join(row_cells) + " |")
        
        markdown_table_str = "\n".join(md_table_lines)

        task_view = mo.vstack([
            mo.md("### 📊 Master Sweep Comparison Table"),
            mo.md(markdown_table_str),
        ])

    task_view
    return (
        all_pseudo_algos,
        master_df,
        task_summaries,
        task_view,
    )


@app.cell
def _(
    all_pseudo_algos,
    extract_best_configuration,
    mo,
    np,
    plot_algorithm_comparison,
    plt,
    show_individual_plots_ui,
    show_plots_ui,
    task_summaries,
    tasks_data,
    window_size_ui,
):
    if not task_summaries or not show_plots_ui.value:
        plots_view = mo.md("")
    else:
        task_plot_elements = []

        base_colors = plt.cm.tab10.colors
        lambda_linestyles = {
            "0.0": ":",
            "0.5": (0, (5, 5)),
            "0.9": "-.",
            "0.95": "--",
            "1.0": "-",
        }

        win_size = int(window_size_ui.value)

        for task_name, summary in task_summaries.items():
            t_info = summary["info"]
            env = t_info["env_name"]
            pol = t_info["policy_type"]
            runs = t_info["runs"]
            metric_key = summary["metric_key"]
            rank_order = summary["rank_order"]
            winning_algo = summary["winning_algo"]

            log_scale = False if metric_key.lower() == "v_start" or rank_order == "higher" else True

            # Color and Linestyle maps for combined plot
            color_map = {}
            linestyle_map = {}
            base_algo_colors = {}
            c_idx = 0

            for pa in runs.keys():
                parts = pa.rsplit('_', 1)
                if len(parts) == 2 and (parts[1] in lambda_linestyles or parts[1].replace('.', '', 1).isdigit()):
                    base_algo = parts[0]
                    lmbda = parts[1]
                else:
                    base_algo = pa
                    lmbda = None

                if base_algo not in base_algo_colors:
                    base_algo_colors[base_algo] = base_colors[c_idx % len(base_colors)]
                    c_idx += 1

                color_map[pa] = base_algo_colors[base_algo]
                linestyle_map[pa] = lambda_linestyles.get(lmbda, "-") if lmbda else "-"

            # 1. Combined Plot (E vs TD)
            fig_combined = plot_algorithm_comparison(
                runs,
                metric_key=metric_key,
                env_name=env,
                log_scale=log_scale,
                use_geom_mean=False,
                rank_by="final_window",
                rank_order=rank_order,
                window_size=win_size,
                title=f"Combined Lambda Comparison: {task_name}",
                color_map=color_map,
                linestyle_map=linestyle_map,
            )

            # Sub-plots for E only and TD only if requested
            fig_e = None
            fig_td = None
            if show_individual_plots_ui.value:
                # E-only
                e_runs = {k: v for k, v in runs.items() if "exact_e" in k.lower() or "e_lambda" in k.lower()}
                if e_runs:
                    e_color_map = {}
                    e_linestyle_map = {}
                    e_lambdas = sorted(list({pa.rsplit('_', 1)[1] for pa in e_runs.keys() if '_' in pa}))
                    for pa in e_runs.keys():
                        parts = pa.rsplit('_', 1)
                        lmbda = parts[1] if len(parts) == 2 else None
                        color_pos = e_lambdas.index(lmbda) if lmbda in e_lambdas else 0
                        e_color_map[pa] = base_colors[color_pos % len(base_colors)]
                        e_linestyle_map[pa] = lambda_linestyles.get(lmbda, "-") if lmbda else "-"

                    fig_e = plot_algorithm_comparison(
                        e_runs,
                        metric_key=metric_key,
                        env_name=env,
                        log_scale=log_scale,
                        use_geom_mean=False,
                        rank_by="final_window",
                        rank_order=rank_order,
                        window_size=win_size,
                        title=f"Exact E(λ) Progression: {task_name}",
                        color_map=e_color_map,
                        linestyle_map=e_linestyle_map,
                    )

                # TD-only
                td_runs = {k: v for k, v in runs.items() if "exact_td" in k.lower() or "td_lambda" in k.lower()}
                if td_runs:
                    td_color_map = {}
                    td_linestyle_map = {}
                    td_lambdas = sorted(list({pa.rsplit('_', 1)[1] for pa in td_runs.keys() if '_' in pa}))
                    for pa in td_runs.keys():
                        parts = pa.rsplit('_', 1)
                        lmbda = parts[1] if len(parts) == 2 else None
                        color_pos = td_lambdas.index(lmbda) if lmbda in td_lambdas else 0
                        td_color_map[pa] = base_colors[color_pos % len(base_colors)]
                        td_linestyle_map[pa] = lambda_linestyles.get(lmbda, "-") if lmbda else "-"

                    fig_td = plot_algorithm_comparison(
                        td_runs,
                        metric_key=metric_key,
                        env_name=env,
                        log_scale=log_scale,
                        use_geom_mean=False,
                        rank_by="final_window",
                        rank_order=rank_order,
                        window_size=win_size,
                        title=f"Exact TD(λ) Progression: {task_name}",
                        color_map=td_color_map,
                        linestyle_map=td_linestyle_map,
                    )

            # Build Per-Task UI Card
            winning_text = f"🏆 **Winning Algorithm:** `{winning_algo}`" if winning_algo else ""
            task_card_content = [
                mo.md(f"## 🎯 Task: {task_name}"),
                mo.md(f"**Metric Evaluated:** `{summary['metric_label']}` | {winning_text}"),
            ]

            if fig_combined:
                task_card_content.append(mo.as_html(fig_combined))

            if fig_e and fig_td:
                task_card_content.append(
                    mo.hstack([mo.as_html(fig_e), mo.as_html(fig_td)], justify="space-around")
                )
            elif fig_e:
                task_card_content.append(mo.as_html(fig_e))
            elif fig_td:
                task_card_content.append(mo.as_html(fig_td))

            task_plot_elements.append(mo.vstack(task_card_content))

        plots_view = mo.vstack([
            mo.md("---"),
            mo.md("## 📈 Cross-Algorithm Learning Curves per Task"),
            *task_plot_elements
        ])

    plots_view
    return (
        base_algo_colors,
        c_idx,
        color_map,
        color_pos,
        e_color_map,
        e_lambdas,
        e_linestyle_map,
        e_runs,
        env,
        fig_combined,
        fig_e,
        fig_td,
        lambda_linestyles,
        lmbda,
        log_scale,
        metric_key,
        pa,
        parts,
        plots_view,
        pol,
        rank_order,
        runs,
        summary,
        t_info,
        task_card_content,
        task_name,
        task_plot_elements,
        td_color_map,
        td_lambdas,
        td_linestyle_map,
        td_runs,
        win_size,
        winning_algo,
        winning_text,
    )


if __name__ == "__main__":
    app.run()

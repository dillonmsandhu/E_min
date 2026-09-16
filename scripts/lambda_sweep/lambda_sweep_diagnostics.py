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
import importlib
import argparse
import jax
import pandas as pd

from notebooks.analyze_sweeps import load_sweep_data, plot_algorithm_comparison, summarize_algorithm_comparison
from scripts.sweep_pipeline import ALGO_REGISTRY
from core.visualizations import generate_and_save_grid_gif

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
    env_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for env_name in env_dirs:
        env_path = os.path.join(base_dir, env_name)
        policy_dirs = [d for d in os.listdir(env_path) if os.path.isdir(os.path.join(env_path, d))]
        
        for policy_type in policy_dirs:
            policy_path = os.path.join(env_path, policy_type)
            print("="*80)
            print(f"Analyzing Task: Env={env_name} | Policy={policy_type}")
            
            pseudo_algos = [d for d in os.listdir(policy_path) if os.path.isdir(os.path.join(policy_path, d)) and d != "comparison"]
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

            comparison_dir = os.path.join(policy_path, "comparison")
            os.makedirs(comparison_dir, exist_ok=True)

            metric_key = "v_start" if policy_type in ["ppo", "hybrid"] else "nn_weighted_VE"
            rank_by = "auc"
            rank_order = "higher" if policy_type in ["ppo", "hybrid"] else "lower"
            window_size = 40

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

            # 2. Generate Comparison Plot
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

                plot_path = os.path.join(comparison_dir, "lambda_comparison_plot.png")
                plot_algorithm_comparison(
                    completed_runs,
                    metric_key=metric_key,
                    env_name=env_name,
                    log_scale=True,
                    use_geom_mean=False,
                    rank_by=rank_by,
                    rank_order=rank_order,
                    window_size=window_size,
                    save_path=plot_path,
                    title=f"Lambda Comparison: {policy_type.capitalize()} Policy on {env_name}",
                    color_map=color_map,
                    linestyle_map=linestyle_map
                )
                print(f"Plot saved to {plot_path}")
            except Exception as e:
                print(f"Failed to generate plot: {e}")

            # 3. Generate Diagnostic GIFs for the best config of each pseudo-algorithm
            print(f"Generating diagnostic GIFs for best configs...")
            for pa, data in completed_runs.items():
                best_config = data.get("best_config")
                if not best_config:
                    print(f"No best_config found for {pa}, skipping GIF.")
                    continue

                print(f"  -> Rolling out {pa} for GIF...")
                # Strip the _<lambda> suffix to get base algo
                base_algo = pa.rsplit('_', 1)[0]
                module_path = ALGO_REGISTRY.get(policy_type, {}).get(base_algo)
                if not module_path:
                    print(f"Could not find module for base algo {base_algo}")
                    continue
                
                try:
                    module = importlib.import_module(module_path)
                    make_train = getattr(module, "make_train")
                    
                    cfg = best_config.copy()
                    cfg["LIGHT_METRICS"] = False
                    cfg["N_SEEDS"] = 1
                    
                    # Some algos might run on CPU, but we assume JAX takes care of dispatch
                    train_fn = make_train(cfg)
                    train_vjit = jax.jit(jax.vmap(train_fn))
                    rngs = jax.random.split(jax.random.PRNGKey(0), 1)
                    out = train_vjit(rngs)
                    metrics = out["metrics"]
                    
                    # Convert device arrays to numpy for plotting
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

    print("="*80)
    print("Diagnostics complete!")

if __name__ == "__main__":
    main()

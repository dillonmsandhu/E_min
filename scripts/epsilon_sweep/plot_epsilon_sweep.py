import os
import glob
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_epsilon_sweep(results_dir, env_name, metric, rank_order, save_path, dir_filter, allowed_algos, plot_column):
    # 1. Find all sweep directories
    sweep_dirs = glob.glob(os.path.join(results_dir, "*"))
    
    data = []
    used_dirs = set()
    
    for s_dir in sweep_dirs:
        if dir_filter and dir_filter not in os.path.basename(s_dir):
            continue
        config_path = os.path.join(s_dir, "pipeline_config.json")
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            meta = json.load(f)
            
        # Only look at greedy policy sweeps for the correct environment
        base_config = meta.get("base_config", {})
        if not base_config.get("USE_GREEDY_POLICY", False):
            continue
        if meta.get("env_name") != env_name:
            continue
            
        epsilon = base_config.get("POLICY_EPSILON", 0.0)
        algos = meta.get("algos", [])
        
        for algo in algos:
            if allowed_algos is not None and algo not in allowed_algos:
                continue
            csv_pattern = os.path.join(s_dir, algo, "tuning", "*", env_name, "tuning_summary.csv")
            csv_files = glob.glob(csv_pattern)
            if not csv_files:
                continue
                
            tuning_csv = csv_files[0]
            df = pd.read_csv(tuning_csv)
            
            plot_col = plot_column if plot_column else f"auc_{metric}"
            
            if plot_col not in df.columns:
                # fallback for some older formats
                if metric in df.columns:
                    plot_col = metric
                else:
                    print(f"skipping sweep {s_dir} due to lack of key {plot_col}.")
                    continue

                
            
            # Check for lambda column (GAE_LAMBDA for TD, VALUE_LAMBDA for exact TD)
            lambda_col = None
            for col in ["GAE_LAMBDA", "VALUE_LAMBDA", "lambda"]:
                if col in df.columns:
                    lambda_col = col
                    break
            
            # 2. Group by Lambda (if applicable) and find the best LR for the requested metric
            if lambda_col is not None:
                for lam_val, group in df.groupby(lambda_col):
                    if rank_order == "lower":
                        best_val = group[plot_col].min()
                    else:
                        best_val = group[plot_col].max()
                    
                    data.append({
                        "epsilon": epsilon,
                        "algorithm": f"{algo} (λ={lam_val})",
                        "best_metric": best_val
                    })
            else:
                # No lambda (e.g. Sampled E)
                if rank_order == "lower":
                    best_val = df[plot_col].min()
                else:
                    best_val = df[plot_col].max()
                    
                data.append({
                    "epsilon": epsilon,
                    "algorithm": algo,
                    "best_metric": best_val
                })

    if not data:
        print(f"No epsilon sweep data found in {results_dir} for env {env_name}.")
        return
        
    print(f"Plotting data aggregated from {len(used_dirs)} sweep directories:")
    for d in sorted(used_dirs):
        print(f"  - {d}")

        
    df_plot = pd.DataFrame(data)
    
    # If multiple directories had the same (algorithm, epsilon), take the best overall
    if rank_order == "lower":
        df_plot = df_plot.groupby(["algorithm", "epsilon"], as_index=False)["best_metric"].min()
    else:
        df_plot = df_plot.groupby(["algorithm", "epsilon"], as_index=False)["best_metric"].max()
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Group by algorithm and sort by epsilon
    # Sort algorithm names so that E-minimization always comes first for consistent coloring
    algo_names = sorted(df_plot["algorithm"].unique(), key=lambda x: (not ('E' in x or 'e_gd' in x), x))
    for algo_name in algo_names:
        group = df_plot[df_plot["algorithm"] == algo_name].sort_values("epsilon")
        plt.plot(group["epsilon"], group["best_metric"], marker='o', linewidth=2, label=algo_name)
        
    plt.xlabel("Policy Epsilon (Noise)", fontsize=12)
    plt.ylabel(f"Best {metric} (over all LRs)", fontsize=12)
    plt.title(f"Epsilon Sweep on {env_name}: {metric}", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"Saved epsilon sweep plot to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/fixed/sweeps")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc")
    parser.add_argument("--metric", type=str, default="nn_weighted_VE")
    parser.add_argument("--plot-column", type=str, default=None, help="Exact column to plot (e.g. auc_nn_advantage_cossim_uniform). Defaults to auc_{metric}")
    parser.add_argument("--rank-order", type=str, default="lower", choices=["lower", "higher"])
    parser.add_argument("--dir-filter", type=str, default="", help="Only include sweep directories containing this string.")
    parser.add_argument("--algos", nargs="+", default=None, help="List of specific algorithms to plot.")
    parser.add_argument("--save-path", type=str, default="epsilon_sweep_plot.png")
    
    args = parser.parse_args()
    
    plot_epsilon_sweep(
        results_dir=args.results_dir,
        env_name=args.env_name,
        metric=args.metric,
        rank_order=args.rank_order,
        save_path=args.save_path,
        dir_filter=args.dir_filter,
        allowed_algos=args.algos,
        plot_column=args.plot_column
    )

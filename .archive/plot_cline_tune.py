import os
import json
import cloudpickle
import numpy as np
import matplotlib.pyplot as plt
from core.utils import load_run_data

def get_latest_run_dir(tuning_base_path):
    if not os.path.exists(tuning_base_path):
        return None
    timestamps = sorted(os.listdir(tuning_base_path))
    if not timestamps:
        return None
    latest_timestamp = timestamps[-1]
    timestamp_path = os.path.join(tuning_base_path, latest_timestamp)
    envs = os.listdir(timestamp_path)
    if not envs:
        return None
    env_name = envs[0]
    return latest_timestamp, env_name

def plot_seeds_nn_weighted_ve():
    algorithms = {
        "Exact TD": "results/random/td_exact/tuning",
        "Exact MC": "results/random/mc_exact/tuning",
        "Exact MC PPO": "results/ppo/ppo_mc_exact/tuning",
        "Exact TD PPO": "results/ppo/ppo_td_exact/tuning",
        "Exact TD Fixed": "results/fixed/td_exact/tuning",
        "Exact MC Fixed": "results/fixed/mc_exact/tuning", 
        "Exact E Fixed": "results/fixed/exact_E_gd/tuning",
    }

    for algo_name, tuning_dir in algorithms.items():
        res = get_latest_run_dir(tuning_dir)
        if res is None:
            print(f"No run found for {algo_name} in {tuning_dir}")
            continue
        
        timestamp, env_name = res
        print(f"Loading {algo_name} from timestamp {timestamp}, env {env_name}...")
        
        try:
            # FIX: Pass only 'timestamp' instead of os.path.join(timestamp, env_name)
            config, metrics = load_run_data(timestamp, env_name, results_base_path=tuning_dir)
        except Exception as e:
            print(f"Error loading {algo_name}: {e}")
            continue

        if "nn_weighted_VE" not in metrics:
            print(f"'nn_weighted_VE' not found in metrics for {algo_name}. Available keys: {list(metrics.keys())}")
            continue

        ve_data = np.array(metrics["nn_weighted_VE"]) # Shape: (n_combos, n_seeds, time_steps) or (n_seeds, time_steps)
        
        if ve_data.ndim == 3:
            # Select the best configuration based on final performance across seeds
            final_means = ve_data[:, :, -1].mean(axis=1)
            best_combo_idx = int(np.argmin(final_means))
            print(f"Selected best hyperparameter combination index {best_combo_idx} out of {ve_data.shape[0]}")
            seed_trajectories = ve_data[best_combo_idx] # Shape: (n_seeds, time_steps)
        elif ve_data.ndim == 2:
            seed_trajectories = ve_data # Shape: (n_seeds, time_steps)
        else:
            print(f"Unexpected shape for nn_weighted_VE: {ve_data.shape}")
            continue

        steps_per_pi = config.get("NUM_ENVS", 1) * config.get("NUM_STEPS", 1)
        n_seeds, time_steps = seed_trajectories.shape
        x = [i * steps_per_pi for i in range(time_steps)]

        plt.figure(figsize=(10, 6))
        for seed_idx in range(n_seeds):
            y = seed_trajectories[seed_idx]
            plt.plot(x, y, label=f"Seed {seed_idx}", linewidth=1.5, alpha=0.8)

        plt.yscale('log')
        plt.xlabel("Env. Step")
        plt.ylabel("Value Error (nn_weighted_VE)")
        plt.title(f"{algo_name} ({env_name}) - Seed Trajectories")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(loc='upper right')

        save_dir = os.path.join(tuning_dir, timestamp, env_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "nn_weighted_VE_seeds.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_seeds_nn_weighted_ve()

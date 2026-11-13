#!/usr/bin/env python3
"""
plot_consolidated_results.py

Consolidates outputs across all Gymnax environments from single-env / cluster array runs:
1. Discovers completed environment runs in <out_dir>/*/data/sweep_metrics_all.pkl
2. Combines them into a single consolidated dataset
3. Automatically generates the full 4 publication-grade cross-environment analyses:
   - Table 1 & Figure 1: Double Sampling Critic Comparison (Learning Curves & Final Return)
   - Table 2 & Figure 2: Bellman Error Diagnostics (Unbiased Diagnostic MSBE E[delta_1 * delta_2] Reduction)
   - Table 3 & Figure 3: Master Championship & Normalized Performance Profiles (Relative to Fitted Baseline)
   - Table 4 & Figure 4: Critic Learning Rate Sensitivity (Hyperparameter Robustness)
"""

from __future__ import annotations
import os
import sys
import glob
import pickle
import argparse
import numpy as np

# Ensure E_min root is strictly prioritized on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))  # E_min root
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))   # E_min/scripts

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

try:
    from scripts.gymnax.sweep_double_sample.sweep_double_sample_critics import (
        generate_diagnostics_and_plots,
        DEFAULT_CRITIC_LRS,
        CRITIC_VARIANTS,
    )
except ImportError:
    from sweep_double_sample_critics import (
        generate_diagnostics_and_plots,
        DEFAULT_CRITIC_LRS,
        CRITIC_VARIANTS,
    )


def main():
    parser = argparse.ArgumentParser(description="Consolidate Double-Sampling Critic Sweep Results across Gymnax Environments")
    parser.add_argument("--out_dir", type=str, default="results/sweep_double_sample_gymnax", help="Base output directory")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout length")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")

    args = parser.parse_args()
    out_dir = args.out_dir
    batch_size = args.num_envs * args.num_steps

    print("=" * 64)
    print("Consolidating Double Sampling Critic Sweep Results (Gymnax)")
    print(f"Base Directory: {out_dir}")
    print(f"Metric Key:     {args.metric_key}")
    print("=" * 64)

    env_pickles = glob.glob(os.path.join(out_dir, "*", "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        env_pickles = glob.glob(os.path.join(out_dir, "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        print(f"[Warning] No sweep_metrics_all.pkl found in {out_dir}/*/")
        return

    consolidated_runs = {}
    consolidated_timesteps = {}
    found_envs = []

    for pkl_file in sorted(env_pickles):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
            runs = data.get("runs", {})
            timesteps = data.get("timesteps", {})
            for env_name, env_data in runs.items():
                consolidated_runs[env_name] = env_data
                if env_name in timesteps:
                    consolidated_timesteps[env_name] = timesteps[env_name]
                found_envs.append(env_name)
                print(f"Loaded {env_name} from {pkl_file}")
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

    found_envs = sorted(list(set(found_envs)))
    if not found_envs:
        print("[Error] No valid environment runs found to consolidate.")
        return

    print(f"\nSuccessfully discovered {len(found_envs)} completed Gymnax environments: {found_envs}")

    # Save consolidated dataset
    consolidated_data_dir = os.path.join(out_dir, "data")
    os.makedirs(consolidated_data_dir, exist_ok=True)
    consolidated_pkl = os.path.join(consolidated_data_dir, "sweep_metrics_all.pkl")
    with open(consolidated_pkl, "wb") as f:
        pickle.dump({"runs": consolidated_runs, "timesteps": consolidated_timesteps}, f)
    print(f"Saved consolidated metrics to {consolidated_pkl}")

    # Generate full suite plots and tables
    critic_lrs = DEFAULT_CRITIC_LRS
    generate_diagnostics_and_plots(
        consolidated_runs,
        consolidated_timesteps,
        out_dir,
        found_envs,
        critic_lrs,
        args.metric_key,
        batch_size,
    )
    print("\nConsolidated cross-environment report and publication tables completed successfully!")


if __name__ == "__main__":
    main()

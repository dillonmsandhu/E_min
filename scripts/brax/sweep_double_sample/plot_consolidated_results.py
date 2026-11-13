#!/usr/bin/env python3
"""
plot_consolidated_results.py

Consolidates outputs across all 11 continuous Brax environments from single-env / cluster array runs:
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
    from scripts.brax.sweep_double_sample.sweep_double_sample_critics import (
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
    parser = argparse.ArgumentParser(description="Consolidate Double-Sampling Critic Sweep Results across Brax Environments")
    parser.add_argument("--out_dir", type=str, default="results/sweep_double_sample_brax", help="Base output directory")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")

    args = parser.parse_args()
    out_dir = args.out_dir
    batch_size = args.num_envs * args.num_steps

    print("=" * 64)
    print("Consolidating Double Sampling Critic Sweep Results (Brax)")
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
    timesteps = {}
    all_critic_lrs = set()

    for pkl_path in sorted(env_pickles):
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            runs = d.get("runs", {})
            for env_name, env_data in runs.items():
                print(f"  Loaded environment: {env_name} from {pkl_path}")
                consolidated_runs[env_name] = env_data
                if "timesteps" in d and env_name in d["timesteps"]:
                    timesteps[env_name] = d["timesteps"][env_name]

                # Detect critic_lrs
                for var_key, _ in CRITIC_VARIANTS:
                    if var_key in env_data:
                        all_critic_lrs.update(env_data[var_key].keys())
        except Exception as e:
            print(f"  [Error] Failed loading {pkl_path}: {e}")

    if not consolidated_runs:
        print("No valid runs loaded. Exiting.")
        return

    critic_lrs = sorted(list(all_critic_lrs)) if all_critic_lrs else DEFAULT_CRITIC_LRS
    envs = list(consolidated_runs.keys())

    print("\nConsolidated Summary:")
    print(f"  Environments ({len(envs)}): {envs}")
    print(f"  Critic LRs:   {critic_lrs}")

    # Save consolidated dataset
    cons_data_dir = os.path.join(out_dir, "data")
    os.makedirs(cons_data_dir, exist_ok=True)
    cons_pkl = os.path.join(cons_data_dir, "sweep_metrics_all.pkl")
    with open(cons_pkl, "wb") as f:
        pickle.dump({"runs": consolidated_runs, "timesteps": timesteps}, f)
    print(f"\nSaved consolidated metrics to {cons_pkl}")

    # Generate Publication Analyses
    generate_diagnostics_and_plots(
        consolidated_runs, timesteps, out_dir, envs, critic_lrs, args.metric_key, batch_size
    )


if __name__ == "__main__":
    main()

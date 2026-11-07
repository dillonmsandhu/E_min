#!/usr/bin/env python3
"""
external_E_min/scripts/gymnax/sweep_comprehensive_q_critic/plot_consolidated_results.py

Consolidates outputs across all deterministic discrete environments from SLURM array jobs:
1. Discovers all completed environments in <out_dir>/*/data/sweep_metrics_all.pkl
2. Combines them into a consolidated dataset
3. Generates the full cross-environment diagnostic and benchmark suite:
   - Diagnostic 1: E(0)-Q vs E(lambda=0)-Q cross-environment plot
   - Diagnostic 2: SARSA(lambda) vs E(lambda)-Q spectrum comparison across lambdas
   - Main Benchmark: Head-to-head learning curves
   - Normalized AUC Heatmap across all environments
   - Summary CSV table
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
    from scripts.gymnax.sweep_comprehensive_q_critic.sweep_comprehensive_q_critics import generate_diagnostics_and_plots
except ImportError:
    from sweep_comprehensive_q_critics import generate_diagnostics_and_plots


def main():
    parser = argparse.ArgumentParser(description="Consolidate and plot Comprehensive Q-PPO Critic Sweep results")
    parser.add_argument("--out_dir", type=str, default="results/sweep_comprehensive_q_critic", help="Base output directory")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel envs")
    parser.add_argument("--num_steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to analyze (default: returned_episode_returns)")

    args = parser.parse_args()
    out_dir = args.out_dir

    print("================================================================")
    print("Consolidating Q-PPO Critic Sweep Results Across Environments")
    print(f"Scanning base directory: {out_dir}")
    print(f"Metric Key:              {args.metric_key}")
    print("================================================================")

    env_pickles = glob.glob(os.path.join(out_dir, "*", "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        env_pickles = glob.glob(os.path.join(out_dir, "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        print(f"[Warning] No sweep_metrics_all.pkl found under {out_dir}/*/")
        return

    consolidated_runs = {}
    timesteps = None
    all_critic_lrs = set()
    all_lambdas = set()

    for pkl_path in sorted(env_pickles):
        try:
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            runs = d.get("runs", {})
            for env_name, env_data in runs.items():
                print(f"  Loaded environment: {env_name} from {pkl_path}")
                consolidated_runs[env_name] = env_data
                if timesteps is None and "timesteps" in d:
                    timesteps = d["timesteps"]

                # Extract lrs and lambdas
                if "E(lambda)" in env_data:
                    for clr, lmb_dict in env_data["E(lambda)"].items():
                        all_critic_lrs.add(clr)
                        for lmb in lmb_dict.keys():
                            all_lambdas.add(lmb)
        except Exception as e:
            print(f"  [Error] Failed loading {pkl_path}: {e}")

    envs = list(consolidated_runs.keys())
    critic_lrs = sorted(list(all_critic_lrs)) if all_critic_lrs else [1e-4, 3e-4, 1e-3]
    lambdas = sorted(list(all_lambdas)) if all_lambdas else [0.0, 0.4, 0.8, 0.95]

    print(f"\nDiscovered {len(envs)} environments: {envs}")
    print(f"Critic LRs: {critic_lrs}")
    print(f"Lambdas:    {lambdas}")

    # Save consolidated dataset
    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    cons_path = os.path.join(out_dir, "data", "consolidated_sweep_metrics.pkl")
    with open(cons_path, "wb") as f:
        pickle.dump({"runs": consolidated_runs, "timesteps": timesteps, "envs": envs}, f)
    print(f"Saved consolidated data to: {cons_path}")

    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "benchmarks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    class ArgsWrapper:
        num_envs = args.num_envs
        num_steps = args.num_steps
        metric_key = args.metric_key

    generate_diagnostics_and_plots(
        all_runs=consolidated_runs,
        timesteps=timesteps,
        out_dir=out_dir,
        envs=envs,
        critic_lrs=critic_lrs,
        lambdas=lambdas,
        args=ArgsWrapper(),
    )

    print("\nConsolidated plotting completed successfully!")


if __name__ == "__main__":
    main()

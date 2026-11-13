#!/usr/bin/env python3
"""
plot_consolidated_results.py

Consolidates outputs across all discrete Jumanji environments from single-env / cluster array runs:
1. Discovers completed environment runs in <out_dir>/*/data/sweep_metrics_all.pkl
2. Combines them into a single consolidated dataset
3. Automatically generates the full 5 publication-grade cross-environment analyses:
   - Table 1 & Figure 1: E(0) Variants (Naive vs v_aux vs Forward-Only)
   - Table 2 & Figure 2: E(lambda=0.8) Variants (v_aux Debiased vs Naive vs Forward-Only)
   - Table 3 & Figure 3: Head-to-Head (E(0) Exact vs E(0.8) v_aux Debiased)
   - Table 4 & Figure 4: TD(lambda) Fitted SARSA Grid
   - Table 5 & Figure 5: Master Championship (Best TD vs Best E) with bottom row: Number of Won Envs
"""

from __future__ import annotations
import os
import sys
import glob
import pickle
import argparse
import numpy as np

# Ensure E_min root is strictly on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.jumanji.sweep_q_ppo_critic.sweep_q_ppo_critics import (
    generate_diagnostics_and_plots,
    DEFAULT_CRITIC_LRS,
    DEFAULT_TD_LAMBDAS,
)


def main():
    parser = argparse.ArgumentParser(description="Consolidate Q-PPO Critic Sweep Results across Jumanji Environments")
    parser.add_argument("--out_dir", type=str, default="results/sweep_q_ppo_critic_jumanji", help="Base output directory")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=64, help="Rollout length")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")

    args = parser.parse_args()
    out_dir = args.out_dir
    batch_size = args.num_envs * args.num_steps

    print("=" * 64)
    print("Consolidating Q-PPO Critic Sweep Results (Jumanji)")
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
    all_td_lambdas = set()

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

                # Detect critic_lrs and td_lambdas
                for var_key in [
                    "E(0)_naive", "E(0)_v_aux", "E(0)_fwd",
                    "E(0.8)_v_aux", "E(0.8)_naive", "E(0.8)_fwd",
                ]:
                    if var_key in env_data:
                        all_critic_lrs.update(env_data[var_key].keys())
                if "TD" in env_data:
                    for lam, lr_dict in env_data["TD"].items():
                        all_td_lambdas.add(lam)
                        all_critic_lrs.update(lr_dict.keys())
        except Exception as e:
            print(f"  [Error] Failed loading {pkl_path}: {e}")

    if not consolidated_runs:
        print("No valid runs loaded. Exiting.")
        return

    critic_lrs = sorted(list(all_critic_lrs)) if all_critic_lrs else DEFAULT_CRITIC_LRS
    td_lambdas = sorted(list(all_td_lambdas)) if all_td_lambdas else DEFAULT_TD_LAMBDAS
    envs = list(consolidated_runs.keys())

    print("\nConsolidated Summary:")
    print(f"  Environments ({len(envs)}): {envs}")
    print(f"  Critic LRs:   {critic_lrs}")
    print(f"  TD Lambdas:   {td_lambdas}")

    # Save consolidated dataset
    cons_data_dir = os.path.join(out_dir, "data")
    os.makedirs(cons_data_dir, exist_ok=True)
    cons_pkl = os.path.join(cons_data_dir, "sweep_metrics_all.pkl")
    with open(cons_pkl, "wb") as f:
        pickle.dump({"runs": consolidated_runs, "timesteps": timesteps}, f)
    print(f"\nSaved consolidated metrics to {cons_pkl}")

    # Generate 5 Publication Analyses
    generate_diagnostics_and_plots(
        consolidated_runs, timesteps, out_dir, envs, critic_lrs, td_lambdas, args.metric_key, batch_size
    )


if __name__ == "__main__":
    main()

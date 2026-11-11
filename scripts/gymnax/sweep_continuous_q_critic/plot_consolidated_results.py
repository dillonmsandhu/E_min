#!/usr/bin/env python3
"""
plot_consolidated_results.py

Consolidates outputs across all continuous Gymnax environments from single-env / cluster array runs:
1. Discovers completed environment runs in <out_dir>/*/data/sweep_metrics_all.pkl
2. Combines them into a single consolidated dataset
3. Automatically generates the full 5 publication-grade cross-environment analyses:
   - Table 1 & Figure 1: E(0) Variants (Naive vs v_aux vs Forward-Only)
   - Table 2 & Figure 2: E(lambda=0.8) Variants (Naive vs v_aux vs Forward-Only)
   - Table 3 & Figure 3: Naive Head-to-Head (E(0) Naive vs E(0.8) Naive)
   - Table 4 & Figure 4: Expected SARSA(lambda) Grid (Effect of lambda on TD)
   - Table 5 & Figure 5: Master Championship (Best E vs Best TD)
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
    from scripts.gymnax.sweep_continuous_q_critic.sweep_continuous_q_critics import (
        generate_diagnostics_and_plots,
        DEFAULT_CRITIC_LRS,
        DEFAULT_SARSA_LAMBDAS,
    )
except ImportError:
    from sweep_continuous_q_critics import (
        generate_diagnostics_and_plots,
        DEFAULT_CRITIC_LRS,
        DEFAULT_SARSA_LAMBDAS,
    )


def main():
    parser = argparse.ArgumentParser(description="Consolidate Continuous Q-Critic Sweep Results across Gymnax Environments")
    parser.add_argument("--out_dir", type=str, default="results/sweep_continuous_q_critic_gymnax", help="Base output directory")
    parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=64, help="Rollout length")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")

    args = parser.parse_args()
    out_dir = args.out_dir
    batch_size = args.num_envs * args.num_steps

    print("=" * 64)
    print("Consolidating Continuous Q-Critic Sweep Results (Gymnax)")
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
    all_sarsa_lambdas = set()

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

                # Detect critic_lrs and sarsa_lambdas
                for var_key in ["E(0)_naive", "E(0)_v_aux", "E(0)_fwd", "E(0.8)_naive", "E(0.8)_v_aux", "E(0.8)_fwd"]:
                    if var_key in env_data:
                        all_critic_lrs.update(env_data[var_key].keys())
                if "SARSA" in env_data:
                    for lam, lr_dict in env_data["SARSA"].items():
                        all_sarsa_lambdas.add(lam)
                        all_critic_lrs.update(lr_dict.keys())
        except Exception as e:
            print(f"  [Error] Failed loading {pkl_path}: {e}")

    envs = list(consolidated_runs.keys())
    critic_lrs = sorted(list(all_critic_lrs)) if all_critic_lrs else DEFAULT_CRITIC_LRS
    sarsa_lambdas = sorted(list(all_sarsa_lambdas)) if all_sarsa_lambdas else DEFAULT_SARSA_LAMBDAS

    print(f"\nDiscovered {len(envs)} environments: {envs}")
    print(f"Critic LRs:     {critic_lrs}")
    print(f"SARSA Lambdas:  {sarsa_lambdas}")

    os.makedirs(os.path.join(out_dir, "data"), exist_ok=True)
    cons_path = os.path.join(out_dir, "data", "consolidated_sweep_metrics.pkl")
    with open(cons_path, "wb") as f:
        pickle.dump({"runs": consolidated_runs, "timesteps": timesteps, "envs": envs}, f)
    print(f"Saved consolidated dataset to {cons_path}")

    # Generate all 5 cross-algorithm tables and figures
    generate_diagnostics_and_plots(
        consolidated_runs, timesteps, out_dir, envs, critic_lrs, sarsa_lambdas, args.metric_key, batch_size
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_single_env.py

Executes the Continuous Q-Critic (DDPG) Comparison Sweep on a single Brax continuous control environment:
1. E(0) Variants (Naive, v_aux, Forward-Only)
2. E(lambda=0.8) Variants (Naive, v_aux, Forward-Only)
3. Expected SARSA(lambda) Grid (0.0, 0.4, 0.8, 0.95, 1.0)

Outputs saved to <out_dir>/<env_name>/:
- data/sweep_metrics_all.pkl
- plots/analysis[1-5]_*.png, .pdf
- tables/table[1-5]_*.md, .tex, .pdf, .png
"""

from __future__ import annotations
import os
import sys
import argparse

# Ensure E_min root is strictly prioritized on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))  # E_min root
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))   # E_min/scripts

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

try:
    from scripts.brax.sweep_continuous_q_critic.sweep_continuous_q_critics import (
        run_sweep,
        get_env_default_timesteps,
        DEFAULT_CRITIC_LRS,
        DEFAULT_SARSA_LAMBDAS,
    )
except ImportError:
    from sweep_continuous_q_critics import (
        run_sweep,
        get_env_default_timesteps,
        DEFAULT_CRITIC_LRS,
        DEFAULT_SARSA_LAMBDAS,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Continuous Q-Critic Sweep on a Single Brax Environment")
    parser.add_argument("--env_name", type=str, required=True, help="Deterministic Continuous Brax Environment Name")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS, help="Critic learning rates")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--sarsa_lambdas", nargs="+", type=float, default=DEFAULT_SARSA_LAMBDAS, help="SARSA lambdas to sweep")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total training timesteps (default: auto per env)")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout length")
    parser.add_argument("--minibatch_size", type=int, default=1024, help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--exploration_noise", type=float, default=0.1, help="Rollout exploration noise std")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")
    parser.add_argument("--out_dir", type=str, default="results/sweep_continuous_q_critic_brax", help="Base output directory")
    parser.add_argument("--load_existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    cli_args = parser.parse_args()

    env_out_dir = os.path.join(cli_args.out_dir, cli_args.env_name)
    os.makedirs(env_out_dir, exist_ok=True)
    env_timesteps = get_env_default_timesteps(cli_args.env_name, cli_args.total_timesteps)

    class SweepArgs:
        envs = [cli_args.env_name]
        critic_lrs = cli_args.critic_lrs
        actor_lr = cli_args.actor_lr
        sarsa_lambdas = cli_args.sarsa_lambdas
        total_timesteps = env_timesteps
        num_envs = cli_args.num_envs
        num_steps = cli_args.num_steps
        minibatch_size = cli_args.minibatch_size
        n_seeds = cli_args.n_seeds
        master_seed = cli_args.master_seed
        exploration_noise = cli_args.exploration_noise
        metric_key = cli_args.metric_key
        out_dir = env_out_dir
        load_existing = cli_args.load_existing

    print("=" * 64)
    print(f"Starting Continuous Q-Critic Worker: {cli_args.env_name}")
    print(f"Output Directory: {env_out_dir}")
    print(f"Timesteps:        {env_timesteps:,}")
    print("=" * 64)

    run_sweep(SweepArgs())
    print(f"\nWorker for Environment: {cli_args.env_name} completed successfully.")


if __name__ == "__main__":
    main()

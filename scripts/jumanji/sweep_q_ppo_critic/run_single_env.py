#!/usr/bin/env python3
"""
run_single_env.py

Executes the Q-PPO Critic Comparison Sweep on a single discrete Jumanji environment:
1. E(0) Variants (Naive, v_aux, Forward-Only)
2. E(lambda=0.8) Variants (v_aux Debiased, Naive, Forward-Only)
3. TD(lambda) Fitted SARSA Grid (0.0, 0.4, 0.8, 0.95, 1.0)

Outputs saved to <out_dir>/<env_name>/:
- data/sweep_metrics_all.pkl
- plots/analysis[1-5]_*.png, .pdf
- tables/table[1-5]_*.md, .pdf, .png
"""

from __future__ import annotations
import os
import sys
import argparse

# Ensure E_min root is strictly on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.jumanji.sweep_q_ppo_critic.sweep_q_ppo_critics import (
    run_sweep,
    get_env_default_timesteps,
    DEFAULT_CRITIC_LRS,
    DEFAULT_TD_LAMBDAS,
)


def main():
    parser = argparse.ArgumentParser(description="Run Q-PPO Critic Sweep on a Single Jumanji Environment")
    parser.add_argument("--env_name", type=str, required=True, help="Jumanji Environment Name (e.g. Sokoban-v0)")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS, help="Critic learning rates")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--td_lambdas", nargs="+", type=float, default=DEFAULT_TD_LAMBDAS, help="TD lambdas to sweep")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps (default: auto per env)")
    parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=128, help="Rollout length (>=120 for full-episode Monte Carlo in Sokoban)")
    parser.add_argument("--minibatch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--metric_key", type=str, default="returned_episode_returns", help="Metric key to evaluate")
    parser.add_argument("--out_dir", type=str, default="results/sweep_q_ppo_critic_jumanji", help="Base output directory")
    parser.add_argument("--load_existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    cli_args = parser.parse_args()

    env_out_dir = os.path.join(cli_args.out_dir, cli_args.env_name)
    os.makedirs(env_out_dir, exist_ok=True)
    env_timesteps = get_env_default_timesteps(cli_args.env_name, cli_args.total_timesteps)

    class SweepArgs:
        envs = [cli_args.env_name]
        critic_lrs = cli_args.critic_lrs
        actor_lr = cli_args.actor_lr
        td_lambdas = cli_args.td_lambdas
        total_timesteps = env_timesteps
        num_envs = cli_args.num_envs
        num_steps = cli_args.num_steps
        minibatch_size = cli_args.minibatch_size
        n_seeds = cli_args.n_seeds
        master_seed = cli_args.master_seed
        metric_key = cli_args.metric_key
        out_dir = env_out_dir
        load_existing = cli_args.load_existing

    print("=" * 64)
    print(f"Starting Q-PPO Critic Worker: {cli_args.env_name}")
    print(f"Output Directory: {env_out_dir}")
    print(f"Timesteps:        {env_timesteps:,}")
    print("=" * 64)

    run_sweep(SweepArgs())


if __name__ == "__main__":
    main()

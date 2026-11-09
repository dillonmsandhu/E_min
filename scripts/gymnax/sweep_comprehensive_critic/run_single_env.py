#!/usr/bin/env python3
"""
external_E_min/scripts/gymnax/sweep_comprehensive_critic/run_single_env.py

Executes the Comprehensive Critic Comparison Sweep on a single environment:
1. E(lambda) with geometric lookahead jumps (sweeping lambda)
2. TD(lambda) fitted value iteration (sweeping lambda)
3. E(0) with V_aux
4. E(0) without V_aux
5. MC rollout return regression
6. V_aux Regularization Ablations (Huber, Trust Region, Both) for a single lambda

Saves per-environment results to <out_dir>/<env_name>/:
- data/sweep_metrics_all.pkl
- diagnostics/diagnostic_1_e0_vs_elambda0.png
- diagnostics/diagnostic_2_e0_v_aux_effect.png
- diagnostics/diagnostic_3_v_aux_vs_v_main.png
- diagnostics/diagnostic_4_v_aux_ablations.png (with TD(lambda) converged performance dotted line)
- benchmarks/main_comparison_learning_curves.png
- benchmarks/main_comparison_auc_heatmap.png
- benchmarks/main_comparison_summary.csv
"""

from __future__ import annotations
import os
import sys
import argparse

# Ensure E_min root is strictly prioritized on sys.path (avoiding collisions with other envs packages)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))  # E_min root
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))   # E_min/scripts

sys.path = [_repo_root, _scripts_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

try:
    from scripts.gymnax.sweep_comprehensive_critic.sweep_comprehensive_critics import run_sweep, get_env_default_timesteps
except ImportError:
    from sweep_comprehensive_critics import run_sweep, get_env_default_timesteps


def main():
    parser = argparse.ArgumentParser(description="Run Comprehensive Critic Sweep on a Single Environment")
    parser.add_argument("--env_name", type=str, required=True, help="Gymnax Environment Name")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.4, 0.8, 0.95], help="Lambdas to sweep")
    parser.add_argument("--aux_lambdas", nargs="+", type=float, default=[1.0, 0.9, 0.8], help="Auxiliary critic lambdas to sweep")
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3], help="Critic learning rates")
    parser.add_argument("--actor_lr", type=float, default=3e-4, help="Fixed actor learning rate")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total training timesteps (default: auto per env: 1M for MountainCar, 300k for small envs)")
    parser.add_argument("--num_envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--num_steps", type=int, default=256, help="Rollout length")
    parser.add_argument("--minibatch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=3, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--ablation_lambda", type=float, default=0.8, help="Lambda for V_aux ablations")
    parser.add_argument("--huber_delta", type=float, default=10.0, help="Huber loss delta")
    parser.add_argument("--trust_coef", type=float, default=1.0, help="Trust-region L2 coefficient")
    parser.add_argument("--out_dir", type=str, default="results/sweep_comprehensive_critic", help="Base output directory")
    parser.add_argument("--metric_key", type=str, default="returned_discounted_episode_returns", help="Metric key to plot")
    parser.add_argument("--load_existing", type=str, default=None, help="Path to existing sweep_metrics_all.pkl")

    cli_args = parser.parse_args()

    # Setup per-environment output directory
    env_out_dir = os.path.join(cli_args.out_dir, cli_args.env_name)
    os.makedirs(env_out_dir, exist_ok=True)

    env_timesteps = get_env_default_timesteps(cli_args.env_name, cli_args.total_timesteps)

    # Adapt arguments to match run_sweep schema
    class SweepArgs:
        envs = [cli_args.env_name]
        lambdas = cli_args.lambdas
        aux_lambdas = cli_args.aux_lambdas
        critic_lrs = cli_args.critic_lrs
        actor_lr = cli_args.actor_lr
        total_timesteps = env_timesteps
        num_envs = cli_args.num_envs
        num_steps = cli_args.num_steps
        minibatch_size = cli_args.minibatch_size
        n_seeds = cli_args.n_seeds
        master_seed = cli_args.master_seed
        ablation_lambda = cli_args.ablation_lambda
        huber_delta = cli_args.huber_delta
        trust_coef = cli_args.trust_coef
        metric_key = cli_args.metric_key
        out_dir = env_out_dir
        load_existing = cli_args.load_existing

    print(f"================================================================")
    print(f"Starting Worker for Environment: {cli_args.env_name}")
    print(f"Output Directory: {env_out_dir}")
    print(f"Lambdas:          {cli_args.lambdas}")
    print(f"Critic LRs:       {cli_args.critic_lrs}")
    print(f"Ablation Lambda:  {cli_args.ablation_lambda}")
    print(f"Timesteps:        {cli_args.total_timesteps}")
    print(f"================================================================")

    run_sweep(SweepArgs())
    print(f"\nWorker for Environment: {cli_args.env_name} completed successfully.")


if __name__ == "__main__":
    main()

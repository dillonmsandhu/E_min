#!/usr/bin/env python3
"""
run_single_env.py

Runs TD(lambda), Q-PPO E(0) (Bilinear + Hybrid), and E(lambda) (v_aux) tuning/comparison
on a single Brax environment and saves outputs to <out_dir>/<env_name>/.
"""

from __future__ import annotations
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

from scripts.brax.sweep_q_ppo_vs_baselines.sweep_q_ppo_vs_baselines import (
    build_arg_parser,
    get_env_default_timesteps,
    run_sweep,
)


def main():
    parser = build_arg_parser()
    parser.add_argument("--env_name", type=str, required=True, help="Single Brax environment name")
    args = parser.parse_args()

    args.envs = [args.env_name]
    args.out_dir = os.path.join(args.out_dir, args.env_name)
    os.makedirs(args.out_dir, exist_ok=True)
    env_timesteps = get_env_default_timesteps(args.env_name, args.total_timesteps)

    print("=" * 68)
    print(f"Running Brax Q-PPO vs Baselines Worker: {args.env_name}")
    print(f"  Mode:       {args.mode}")
    print(f"  Timesteps:  {env_timesteps:,}")
    print(f"  Output Dir: {args.out_dir}")
    print("=" * 68)

    run_sweep(args)
    print(f"\nWorker for Environment '{args.env_name}' completed successfully.")


if __name__ == "__main__":
    main()

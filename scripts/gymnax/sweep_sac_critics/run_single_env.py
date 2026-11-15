#!/usr/bin/env python3
"""
run_single_env.py

Runs the Gymnax SAC Critic Comparison Sweep (TD(lambda), E(0), Online TD(0), E(lambda) with v_aux)
on a single Gymnax environment and saves outputs to <out_dir>/<env_name>/.
"""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (
        os.path.exists(os.path.join(p, "fixed_policy"))
        or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy")))
    )
]

from scripts.gymnax.sweep_sac_critics.sweep_sac_critics import (
    build_arg_parser,
    get_env_default_timesteps,
    run_sweep,
)


def main() -> None:
    parser = build_arg_parser()
    parser.add_argument("--env_name", type=str, required=True, help="Single Gymnax environment name")
    args = parser.parse_args()

    args.envs = [args.env_name]
    args.out_dir = os.path.join(args.out_dir, args.env_name)
    os.makedirs(args.out_dir, exist_ok=True)
    env_timesteps = get_env_default_timesteps(args.env_name, args.total_timesteps)

    print("=" * 68)
    print(f"Running Gymnax SAC Critics Worker: {args.env_name}")
    print(f"  Mode:       {args.mode}")
    print(f"  Timesteps:  {env_timesteps:,}")
    print(f"  Seeds:      {args.n_seeds} (Master: {args.master_seed})")
    print(f"  Output Dir: {args.out_dir}")
    print("=" * 68)

    run_sweep(args)
    print(f"\nWorker for Environment '{args.env_name}' completed successfully.")


if __name__ == "__main__":
    main()

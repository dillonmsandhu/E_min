#!/usr/bin/env python3
"""Single-environment worker for Octax V-PPO sweep."""

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

from scripts.octax.sweep_v_ppo.sweep_v_ppo import build_arg_parser, run_sweep


def main():
    parser = build_arg_parser()
    parser.add_argument("--env_name", type=str, required=True, help="Single Octax environment name (e.g. Octax-pong)")
    args = parser.parse_args()
    args.envs = [args.env_name]
    args.out_dir = os.path.join(args.out_dir, args.env_name)
    os.makedirs(args.out_dir, exist_ok=True)
    run_sweep(args)


if __name__ == "__main__":
    main()

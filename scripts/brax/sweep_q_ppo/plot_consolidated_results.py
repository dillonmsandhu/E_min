#!/usr/bin/env python3
"""Consolidates per-environment results under results/brax/sweep_q_ppo."""

from __future__ import annotations

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

from scripts.brax.sweep_q_ppo.sweep_q_ppo import METHOD_KEYS, REFERENCE_METHOD, SUITE_TITLE
from scripts.sweep_standard_runner import consolidate_standardized_sweep


def main():
    parser = argparse.ArgumentParser(description="Consolidate Brax Q-PPO Sweep Results")
    parser.add_argument("--out_dir", type=str, default="results/brax/sweep_q_ppo")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--all_pairs", action="store_true")
    parser.add_argument("--exclude_methods", nargs="*", default=[])
    args = parser.parse_args()
    active_methods = [m for m in METHOD_KEYS if m not in set(args.exclude_methods)]
    consolidate_standardized_sweep(
        out_dir=args.out_dir,
        method_keys=active_methods,
        suite_title=SUITE_TITLE,
        reference_method=REFERENCE_METHOD,
        is_v_ppo=False,
        save_dir=args.save_dir,
        all_pairs=args.all_pairs,
    )


if __name__ == "__main__":
    main()

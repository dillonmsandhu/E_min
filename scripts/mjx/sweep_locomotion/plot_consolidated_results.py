#!/usr/bin/env python3
"""Consolidates per-environment MJX Locomotion V-PPO sweep results and generates tables/plots."""

from __future__ import annotations

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p
    for p in sys.path
    if not (
        os.path.exists(os.path.join(p, "fixed_policy"))
        or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy")))
    )
]

from scripts.mjx.sweep_locomotion.sweep_locomotion import METHOD_KEYS, REFERENCE_METHOD, SUITE_TITLE
from scripts.sweep_standard_runner import consolidate_standardized_sweep


def main():
    parser = argparse.ArgumentParser(description="Consolidate MJX Locomotion V-PPO per-env sweep outputs")
    parser.add_argument("--out_dir", type=str, default="results/mjx/sweep_locomotion")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=METHOD_KEYS)
    parser.add_argument("--all_pairs", action="store_true")
    args = parser.parse_args()

    consolidate_standardized_sweep(
        out_dir=args.out_dir,
        method_keys=list(args.methods),
        suite_title=SUITE_TITLE,
        reference_method=REFERENCE_METHOD,
        is_v_ppo=True,
        save_dir=args.save_dir,
        all_pairs=args.all_pairs,
    )


if __name__ == "__main__":
    main()

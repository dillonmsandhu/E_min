#!/usr/bin/env python3
"""Consolidates per-environment (and per-method) MJX Locomotion V-PPO sweep results."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
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
from scripts.sweep_standard_plotting import DEFAULT_PAIRWISE_V_PPO, generate_standardized_suite


def main():
    parser = argparse.ArgumentParser(description="Consolidate MJX Locomotion V-PPO per-env sweep outputs")
    parser.add_argument("--out_dir", type=str, default="results/mjx/sweep_locomotion")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--methods", nargs="+", default=METHOD_KEYS)
    parser.add_argument("--all_pairs", action="store_true")
    args = parser.parse_args()

    env_pickles = sorted(glob.glob(os.path.join(args.out_dir, "*", "data", "sweep_metrics_all.pkl")))
    if not env_pickles:
        print(f"[Warning] No sweep_metrics_all.pkl found under {args.out_dir}")
        return

    consolidated_runs: dict = {}
    timesteps: dict = {}
    batch_sizes: dict = {}
    for pkl_path in env_pickles:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        for env, env_data in d.get("runs", {}).items():
            consolidated_runs.setdefault(env, {})
            for m_key, m_data in env_data.items():
                if m_data:
                    consolidated_runs[env].setdefault(m_key, {}).update(m_data)
        timesteps.update(d.get("timesteps", {}))
        batch_sizes.update(d.get("batch_sizes", {}))

    req_methods = list(args.methods)
    complete_envs = [
        e for e in sorted(consolidated_runs.keys())
        if all(m in consolidated_runs[e] and consolidated_runs[e][m] for m in req_methods)
    ]
    if not complete_envs:
        print(f"[Warning] No environment has all requested methods {req_methods} yet.")
        for e, ed in sorted(consolidated_runs.items()):
            print(f"  {e}: { {m: len(v) for m, v in ed.items()} }")
        return

    filtered_runs = {e: consolidated_runs[e] for e in complete_envs}
    target_dir = args.save_dir or args.out_dir
    os.makedirs(os.path.join(target_dir, "data"), exist_ok=True)
    with open(os.path.join(target_dir, "data", "sweep_metrics_all.pkl"), "wb") as f:
        pickle.dump(
            {
                "runs": filtered_runs,
                "timesteps": {e: timesteps[e] for e in complete_envs},
                "batch_sizes": {e: batch_sizes[e] for e in complete_envs},
                "method_keys": req_methods,
            },
            f,
        )

    first_env = complete_envs[0]
    first_r = next(iter(next(iter(filtered_runs[first_env].values())).values()))
    sample_cfg = first_r.get("config", {})
    base_summary = {
        "ACTOR_LR": sample_cfg.get("ACTOR_LR"),
        "GAE_LAMBDA": sample_cfg.get("GAE_LAMBDA"),
        "NUM_ENVS": sample_cfg.get("NUM_ENVS"),
        "NUM_STEPS": sample_cfg.get("NUM_STEPS"),
    }
    generate_standardized_suite(
        all_runs=filtered_runs,
        method_keys=req_methods,
        timesteps={e: timesteps[e] for e in complete_envs},
        out_dir=target_dir,
        suite_title=SUITE_TITLE,
        batch_sizes={e: batch_sizes[e] for e in complete_envs},
        base_config=base_summary,
        reference_method=REFERENCE_METHOD,
        pairwise_pairs=DEFAULT_PAIRWISE_V_PPO,
        all_pairs=args.all_pairs,
    )


if __name__ == "__main__":
    main()

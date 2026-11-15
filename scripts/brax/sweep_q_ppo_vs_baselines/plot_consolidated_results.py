#!/usr/bin/env python3
"""
plot_consolidated_results.py

Consolidates per-environment runs in <out_dir>/*/data/sweep_metrics_all.pkl,
computes cross-environment normalized rankings and per-environment winners for:
  - TD(lambda) (PPO Fitted)
  - Q-PPO E(0) (Bilinear + Hybrid)
  - E(lambda) (v_aux PPO)
Writes consolidated learning curves, Markdown/LaTeX/PNG/PDF tables, and
best_configs_consolidated.json.
"""

from __future__ import annotations
import os
import sys
import glob
import json
import copy
import pickle
import argparse
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

from scripts.brax.sweep_q_ppo_vs_baselines.sweep_q_ppo_vs_baselines import (
    METHODS,
    generate_comparison_outputs,
    score_method_candidates,
)


def main():
    parser = argparse.ArgumentParser(description="Consolidate Brax Q-PPO vs Baselines Sweep Results")
    parser.add_argument("--out_dir", type=str, default="results/sweep_q_ppo_vs_baselines_brax")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=128)
    args = parser.parse_args()

    out_dir = args.out_dir
    batch_size = args.num_envs * args.num_steps

    env_pickles = glob.glob(os.path.join(out_dir, "*", "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        env_pickles = glob.glob(os.path.join(out_dir, "data", "sweep_metrics_all.pkl"))
    if not env_pickles:
        print(f"[Warning] No sweep_metrics_all.pkl found under {out_dir}")
        return

    consolidated_runs = {}
    timesteps = {}
    for pkl_path in sorted(env_pickles):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        for env_name, env_data in d.get("runs", {}).items():
            print(f"Loaded {env_name} from {pkl_path}")
            consolidated_runs[env_name] = env_data
            if "timesteps" in d and env_name in d["timesteps"]:
                timesteps[env_name] = d["timesteps"][env_name]

    envs = sorted(consolidated_runs.keys())
    generate_comparison_outputs(consolidated_runs, timesteps, out_dir, batch_size)

    # Also compute strict cross-env shared-hyperparameter best + per-env best summary
    data_dir = os.path.join(out_dir, "data")
    consolidated_summary = {"global_shared_best": {}, "per_env_best": {}}
    for method_key, method_label in METHODS:
        env_method_runs = {
            env: consolidated_runs[env][method_key]
            for env in envs
            if method_key in consolidated_runs[env] and consolidated_runs[env][method_key]
        }
        df_global = score_method_candidates(env_method_runs, method_key, shared_only=True)
        if not df_global.empty:
            df_global.to_csv(os.path.join(data_dir, f"global_shared_ranking_{method_key}.csv"), index=False)
            best_row = df_global.iloc[0]
            best_tag = str(best_row["tag"])
            consolidated_summary["global_shared_best"][method_key] = {
                "label": method_label,
                "tag": best_tag,
                "combined_score": float(best_row["combined_score"]),
                "params": copy.deepcopy(env_method_runs[envs[0]][best_tag]["params"]),
            }

        consolidated_summary["per_env_best"][method_key] = {}
        for env in envs:
            if method_key not in consolidated_runs[env]:
                continue
            df_env = score_method_candidates({env: consolidated_runs[env][method_key]}, method_key, shared_only=False)
            if not df_env.empty:
                top = df_env.iloc[0]
                t_name = str(top["tag"])
                r = consolidated_runs[env][method_key][t_name]
                consolidated_summary["per_env_best"][method_key][env] = {
                    "tag": t_name,
                    "auc_mean": float(r["auc_mean"]),
                    "final_mean": float(r["final_mean"]),
                    "params": copy.deepcopy(r["params"]),
                }

    with open(os.path.join(data_dir, "best_configs_consolidated.json"), "w") as f:
        json.dump(consolidated_summary, f, indent=2, default=str)

    print(f"Consolidated plots, tables, and tuned configs saved under {out_dir}/")


if __name__ == "__main__":
    main()

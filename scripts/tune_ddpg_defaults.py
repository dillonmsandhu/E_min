#!/usr/bin/env python3
"""
scripts/tune_ddpg_defaults.py

Tunes standard On-Policy DDPG using Expected SARSA / TD(lambda=0.8) starting from
the baseline defaults in core/ddpg_config.py, for either Gymnax or Brax.

Outputs ranked results, normalized cross-environment scores, and a ready-to-apply
best_ddpg_config.json so core/ddpg_config.py can be updated with the tuned defaults.

Usage:
  # Fast coordinate tuning on Gymnax (around core/ddpg_config.py starting params)
  python3 scripts/tune_ddpg_defaults.py --backend gymnax --mode coordinate

  # Full Cartesian grid search on Brax (GPU recommended)
  python3 scripts/tune_ddpg_defaults.py --backend brax --mode grid --n_seeds 3
"""

from __future__ import annotations
import os
import sys
import json
import copy
import time
import itertools
import argparse
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path = [_repo_root, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

import jax
from gymnax_rl.core.ddpg_config import config as gymnax_ddpg_config
from brax_rl.core.ddpg_config import config as brax_ddpg_config


DEFAULT_ENVS = {
    "gymnax": ["Pendulum-v1", "MountainCarDenseContinuous-v0", "PointRobot-misc"],
    "brax": ["inverted_pendulum", "hopper", "halfcheetah"],
}

DEFAULT_TIMESTEPS = {
    "Pendulum-v1": 300_000,
    "MountainCarContinuous-v0": 500_000,
    "MountainCarDenseContinuous-v0": 500_000,
    "PointRobot-misc": 300_000,
    "inverted_pendulum": 2_000_000,
    "reacher": 5_000_000,
    "hopper": 5_000_000,
    "walker2d": 5_000_000,
    "halfcheetah": 5_000_000,
    "ant": 5_000_000,
}


def build_base_ddpg_td_config(backend: str, env_name: str, total_timesteps: int | None) -> dict:
    """Builds the standard TD(lambda=0.8) DDPG starting config from core/ddpg_config.py."""
    if backend == "gymnax":
        cfg = copy.deepcopy(gymnax_ddpg_config)
    elif backend == "brax":
        cfg = copy.deepcopy(brax_ddpg_config)
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Expected 'gymnax' or 'brax'.")

    cfg["ENV_NAME"] = env_name
    cfg["CRITIC_TYPE"] = "sarsa"
    cfg["VALUE_LAMBDA"] = 0.8
    cfg["TOTAL_TIMESTEPS"] = (
        total_timesteps if total_timesteps is not None else DEFAULT_TIMESTEPS.get(env_name, cfg["TOTAL_TIMESTEPS"])
    )
    return cfg


def build_search_space(backend: str, args) -> tuple[dict, list[dict]]:
    """
    Returns (starting_params, list_of_candidate_override_dicts).
    Every candidate fixes TD(lambda=0.8) and perturbs around core/ddpg_config.py defaults.
    """
    base = gymnax_ddpg_config if backend == "gymnax" else brax_ddpg_config
    starting_params = {
        "ACTOR_LR": base["ACTOR_LR"],
        "CRITIC_LR": base["CRITIC_LR"],
        "ACTOR_EPOCHS": base["ACTOR_EPOCHS"],
        "CRITIC_EPOCHS": base["CRITIC_EPOCHS"],
        "ACTION_REG_COEF": base["ACTION_REG_COEF"],
        "EXPLORATION_NOISE": base["EXPLORATION_NOISE"],
        "LR_DECAY_RATIO": 0.1,
    }
    if backend == "gymnax":
        starting_params["k"] = base["k"]
        starting_params["NORMALIZE_OBS"] = base["NORMALIZE_OBS"]

    axes = {
        "ACTOR_LR": args.actor_lrs,
        "CRITIC_LR": args.critic_lrs,
        "ACTOR_EPOCHS": args.actor_epochs,
        "CRITIC_EPOCHS": args.critic_epochs,
        "ACTION_REG_COEF": args.action_reg_coefs,
        "EXPLORATION_NOISE": args.exploration_noises,
        "LR_DECAY_RATIO": args.lr_decay_ratios,
    }
    if backend == "gymnax":
        axes["k"] = args.k_dims
        axes["NORMALIZE_OBS"] = [bool(v) for v in args.normalize_obs]

    candidates = []
    seen = set()

    def _add(param_dict, tag):
        key = tuple(sorted(param_dict.items()))
        if key not in seen:
            seen.add(key)
            candidates.append({"tag": tag, "params": dict(param_dict)})

    # Always evaluate starting baseline first
    _add(starting_params, "baseline_start")

    if args.mode == "coordinate":
        for axis_name, vals in axes.items():
            for v in vals:
                cand = dict(starting_params)
                cand[axis_name] = v
                _add(cand, f"coord_{axis_name}={v}")
    elif args.mode == "grid":
        keys = list(axes.keys())
        for combo in itertools.product(*(axes[k] for k in keys)):
            cand = dict(zip(keys, combo))
            _add(cand, "grid")
    else:
        raise ValueError(f"Unrecognized tuning mode '{args.mode}'. Expected 'coordinate' or 'grid'.")

    return starting_params, candidates


def eval_candidate_on_env(backend: str, make_train_fn, base_env_cfg: dict, cand_params: dict, seed_keys) -> dict:
    """Runs one candidate config across seed_keys on a single environment."""
    cfg = copy.deepcopy(base_env_cfg)
    for k, v in cand_params.items():
        if k != "LR_DECAY_RATIO":
            cfg[k] = v

    ratio = cand_params["LR_DECAY_RATIO"]
    cfg["ACTOR_LR_END"] = ratio * cfg["ACTOR_LR"]
    cfg["CRITIC_LR_END"] = ratio * cfg["CRITIC_LR"]
    if backend == "gymnax":
        cfg["LR"] = cfg["ACTOR_LR"]

    train_fn = make_train_fn(cfg)
    vmapped_fn = jax.jit(jax.vmap(train_fn))
    t0 = time.time()
    out = vmapped_fn(seed_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - t0

    returns = np.array(out["metrics"]["returned_episode_returns"])
    seed_aucs = returns.mean(axis=-1)
    seed_finals = returns[:, -1]
    return {
        "auc_mean": float(seed_aucs.mean()),
        "auc_std": float(seed_aucs.std()),
        "final_mean": float(seed_finals.mean()),
        "final_std": float(seed_finals.std()),
        "elapsed_s": float(elapsed),
    }


def evaluate_candidates(
    backend: str,
    make_train_fn,
    envs: list[str],
    candidates: list[dict],
    seed_keys,
    total_timesteps: int | None,
) -> list[dict]:
    records = []
    for c_idx, cand in enumerate(candidates):
        params = cand["params"]
        tag = cand["tag"]
        print(f"\n[{c_idx + 1}/{len(candidates)}] Evaluating ({tag}): {params}")
        env_metrics = {}
        for env_name in envs:
            base_env_cfg = build_base_ddpg_td_config(backend, env_name, total_timesteps)
            m = eval_candidate_on_env(backend, make_train_fn, base_env_cfg, params, seed_keys)
            env_metrics[env_name] = m
            print(
                f"  -> {env_name:30s} | AUC={m['auc_mean']:8.2f} ± {m['auc_std']:5.2f} "
                f"| Final={m['final_mean']:8.2f} ({m['elapsed_s']:.1f}s)"
            )
        records.append({"tag": tag, "params": params, "env_metrics": env_metrics})
    return records


def rank_records(records: list[dict], envs: list[str]) -> pd.DataFrame:
    """Computes per-env min-max normalized AUC & Final scores and aggregates mean rank."""
    rows = []
    for rec in records:
        row = {"tag": rec["tag"], **rec["params"]}
        for env in envs:
            m = rec["env_metrics"][env]
            row[f"{env}__auc"] = m["auc_mean"]
            row[f"{env}__auc_std"] = m["auc_std"]
            row[f"{env}__final"] = m["final_mean"]
            row[f"{env}__final_std"] = m["final_std"]
        rows.append(row)

    df = pd.DataFrame(rows)
    norm_auc_cols = []
    norm_final_cols = []
    for env in envs:
        auc_col = f"{env}__auc"
        fin_col = f"{env}__final"
        auc_min, auc_max = df[auc_col].min(), df[auc_col].max()
        fin_min, fin_max = df[fin_col].min(), df[fin_col].max()

        n_auc = f"{env}__norm_auc"
        n_fin = f"{env}__norm_final"
        df[n_auc] = (df[auc_col] - auc_min) / max(auc_max - auc_min, 1e-8)
        df[n_fin] = (df[fin_col] - fin_min) / max(fin_max - fin_min, 1e-8)
        norm_auc_cols.append(n_auc)
        norm_final_cols.append(n_fin)

    df["mean_norm_auc"] = df[norm_auc_cols].mean(axis=1)
    df["mean_norm_final"] = df[norm_final_cols].mean(axis=1)
    df["combined_score"] = 0.6 * df["mean_norm_auc"] + 0.4 * df["mean_norm_final"]
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Tune standard TD(lambda=0.8) DDPG defaults for core/ddpg_config.py")
    parser.add_argument("--backend", type=str, choices=["gymnax", "brax"], default="gymnax")
    parser.add_argument("--mode", type=str, choices=["coordinate", "grid"], default="coordinate")
    parser.add_argument("--envs", nargs="+", default=None, help="Environments to tune on")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Override timesteps per env")
    parser.add_argument("--n_seeds", type=int, default=3, help="Random seeds per configuration")
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/tune_ddpg_defaults")

    # Search axes anchored around core/ddpg_config.py starting defaults
    parser.add_argument("--actor_lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--actor_epochs", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--critic_epochs", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--action_reg_coefs", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25])
    parser.add_argument("--exploration_noises", nargs="+", type=float, default=None)
    parser.add_argument("--lr_decay_ratios", nargs="+", type=float, default=[1.0, 0.1, 0.03])
    parser.add_argument("--k_dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--normalize_obs", nargs="+", type=int, default=[0, 1], help="0 or 1 for Gymnax NORMALIZE_OBS")

    args = parser.parse_args()
    if args.exploration_noises is None:
        args.exploration_noises = [0.1, 0.2, 0.3] if args.backend == "gymnax" else [0.05, 0.1, 0.2]

    envs = args.envs if args.envs is not None else DEFAULT_ENVS[args.backend]
    if args.backend == "gymnax":
        from gymnax_rl.algos.ddpg import make_train as make_train_fn
        base_template = copy.deepcopy(gymnax_ddpg_config)
    else:
        from brax_rl.algos.ddpg import make_train as make_train_fn
        base_template = copy.deepcopy(brax_ddpg_config)

    out_dir = os.path.join(args.out_dir, args.backend)
    os.makedirs(out_dir, exist_ok=True)

    rng = jax.random.PRNGKey(args.master_seed)
    seed_keys = jax.random.split(rng, args.n_seeds)

    starting_params, candidates = build_search_space(args.backend, args)
    print("=" * 72)
    print(f"Tuning Standard DDPG (TD lambda=0.8) on backend={args.backend} | mode={args.mode}")
    print(f"Envs:            {envs}")
    print(f"Seeds:           {args.n_seeds}")
    print(f"Starting Params: {starting_params}")
    print(f"Total Configs:   {len(candidates)}")
    print("=" * 72)

    records = evaluate_candidates(args.backend, make_train_fn, envs, candidates, seed_keys, args.total_timesteps)

    # In coordinate mode, synthesize and validate the combined coordinate-wise winner
    if args.mode == "coordinate" and len(envs) > 0:
        df_coord = rank_records(records, envs)
        combined_params = dict(starting_params)
        for axis_name in starting_params.keys():
            axis_rows = df_coord[df_coord["tag"].str.startswith(f"coord_{axis_name}=") | (df_coord["tag"] == "baseline_start")]
            best_row = axis_rows.iloc[0]
            val = best_row[axis_name]
            if isinstance(val, np.generic):
                val = val.item()
            combined_params[axis_name] = val

        if tuple(sorted(combined_params.items())) != tuple(sorted(starting_params.items())):
            print("\nValidating combined coordinate-optimal candidate against starting params...")
            records.extend(
                evaluate_candidates(
                    args.backend,
                    make_train_fn,
                    envs,
                    [{"tag": "combined_coordinate_best", "params": combined_params}],
                    seed_keys,
                    args.total_timesteps,
                )
            )

    df_ranked = rank_records(records, envs)
    csv_path = os.path.join(out_dir, "tuning_results.csv")
    df_ranked.to_csv(csv_path, index=False)

    best_row = df_ranked.iloc[0]
    tuned_cfg = copy.deepcopy(base_template)
    for k in starting_params.keys():
        if k != "LR_DECAY_RATIO":
            val = best_row[k]
            tuned_cfg[k] = val.item() if isinstance(val, np.generic) else val

    decay_ratio = float(best_row["LR_DECAY_RATIO"])
    tuned_cfg["ACTOR_LR_END"] = decay_ratio * float(tuned_cfg["ACTOR_LR"])
    tuned_cfg["CRITIC_LR_END"] = decay_ratio * float(tuned_cfg["CRITIC_LR"])
    if args.backend == "gymnax":
        tuned_cfg["LR"] = tuned_cfg["ACTOR_LR"]

    best_json_path = os.path.join(out_dir, "best_ddpg_config.json")
    with open(best_json_path, "w") as f:
        json.dump(tuned_cfg, f, indent=2)

    # Write Markdown summary
    display_cols = ["tag", "combined_score", "mean_norm_auc", "mean_norm_final"] + list(starting_params.keys())
    top10 = df_ranked[display_cols].head(10)
    md_path = os.path.join(out_dir, "tuning_summary.md")
    target_cfg_path = "gymnax_rl/core/ddpg_config.py" if args.backend == "gymnax" else "brax_rl/core/ddpg_config.py"
    with open(md_path, "w") as f:
        f.write(f"# DDPG TD(lambda=0.8) Tuning Summary ({args.backend}, mode={args.mode})\n\n")
        f.write(f"- Environments: `{envs}`\n")
        f.write(f"- Seeds per config: `{args.n_seeds}`\n")
        f.write(f"- Best tag: `{best_row['tag']}` (combined_score={best_row['combined_score']:.4f})\n\n")
        f.write("## Top 10 Configurations\n\n")
        f.write(top10.to_markdown(index=False))
        f.write(f"\n\n## Recommended `{target_cfg_path}` Overrides\n\n```json\n")
        f.write(json.dumps({k: tuned_cfg[k] for k in tuned_cfg if tuned_cfg[k] != base_template.get(k)}, indent=2))
        f.write("\n```\n")

    print("\n" + "=" * 72)
    print(f"Tuning complete! Results written to {out_dir}/")
    print(f"  - Ranked CSV:     {csv_path}")
    print(f"  - Summary MD:     {md_path}")
    print(f"  - Best Config:    {best_json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

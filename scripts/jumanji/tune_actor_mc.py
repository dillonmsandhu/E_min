#!/usr/bin/env python3
"""
scripts/jumanji/tune_actor_mc.py

Tunes PPO Actor hyperparameters across discrete Jumanji environments using an
unbiased Monte Carlo critic (lambda = 1.0, GAE lambda = 1.0) with extended rollout
lengths (NUM_STEPS = 256) so episodes terminate within the rollout window,
achieving true non-bootstrapped Monte Carlo policy evaluation.

Finds a single, robust, standardized actor configuration across:
- Sokoban-v0 (time_limit = 120 <= 256 -> 100% exact full-episode MC)
- Snake-v1
- Game2048-v1

Search Grid:
- ACTOR_LR: [1e-4, 3e-4, 1e-3]
- ENT_COEF: [0.001, 0.005, 0.01, 0.02]
- CLIP_EPS: [0.1, 0.2]

Outputs:
- Ranked cross-environment Min-Max score tables
- Saves scripts/jumanji/jumanji_best_actor_config.json
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

# Ensure E_min root is on sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
from jumanji_rl.core.jumanji_config import config as base_default_config
from jumanji_rl.algos.q_ppo import make_train

DEFAULT_ENVS = [
    "Sokoban-v0",
    "Snake-v1",
    "Maze-v0",
    "SlidingTilePuzzle-v0",
    "RubiksCube-partly-scrambled-v0",
    "Game2048-v1",
]

DEFAULT_SEARCH_SPACE = {
    "ACTOR_LR": [0.0001, 0.0003, 0.001],
    "ENT_COEF": [0.001, 0.005, 0.01, 0.02],
    "CLIP_EPS": [0.1, 0.2],
}

DEFAULT_BASE_PARAMS = {
    "ACTOR_LR": 0.0003,
    "ENT_COEF": 0.01,
    "CLIP_EPS": 0.1,
}


def build_experiment_config(env_name: str, hparams: dict, num_envs: int, num_steps: int, total_timesteps: int, minibatch_size: int) -> dict:
    return {
        **base_default_config,
        "ENV_NAME": env_name,
        "TOTAL_TIMESTEPS": total_timesteps,
        "NUM_ENVS": num_envs,
        "NUM_STEPS": num_steps,           # Extended horizon (e.g. 256) to encompass complete episodes
        "MINIBATCH_SIZE": minibatch_size,
        "ACTOR_MINIBATCH_SIZE": minibatch_size,
        "CRITIC_MINIBATCH_SIZE": minibatch_size,
        "NUM_EPOCHS": 4,
        "NETWORK_TYPE": "cnn",
        "k": 64,
        "GAMMA": 0.99,
        # Pure Monte Carlo Critic & Advantage (Zero Bootstrapping)
        "CRITIC_TYPE": "sarsa",
        "E_VARIANT": "naive",
        "VALUE_LAMBDA": 1.0,              # Pure empirical return target G_t
        "GAE_LAMBDA": 1.0,                # REINFORCE advantage: A_t = G_t - V(s_t)
        "RETURN_LAMBDA": 1.0,
        "CRITIC_LR": 0.001,               # Fast regression on Monte Carlo targets
        "CRITIC_LR_END": 0.001,
        # Actor parameters under evaluation
        "ACTOR_LR": hparams["ACTOR_LR"],
        "ACTOR_LR_END": hparams["ACTOR_LR"],
        "ENT_COEF": hparams["ENT_COEF"],
        "CLIP_EPS": hparams["CLIP_EPS"],
    }


def evaluate_condition(env_name: str, hparams: dict, seed_keys: jax.Array, num_envs: int, num_steps: int, total_timesteps: int, minibatch_size: int) -> float:
    cfg = build_experiment_config(env_name, hparams, num_envs, num_steps, total_timesteps, minibatch_size)
    train_fn = make_train(cfg)
    vmap_train = jax.vmap(train_fn)
    out = vmap_train(seed_keys)
    ret_arr = np.array(out["metrics"]["returned_episode_returns"])  # (n_seeds, n_updates)
    # Area under the learning curve (AUC) across seeds
    auc_score = float(np.nanmean(ret_arr))
    return auc_score


def run_coordinate_tuning(args):
    """Fast coordinate-descent search along each hyperparameter axis."""
    envs = args.envs
    rng = jax.random.PRNGKey(args.master_seed)
    seed_keys = jax.random.split(rng, args.n_seeds)

    current_best = dict(DEFAULT_BASE_PARAMS)
    history = []

    print("=" * 64)
    print("Running Coordinate Tuning with Monte Carlo Critic (lambda = 1.0)")
    print(f"Environments:     {envs}")
    print(f"Rollout Horizon:  {args.num_steps} steps (NUM_ENVS={args.num_envs})")
    print(f"Total Timesteps:  {args.total_timesteps:,} per condition")
    print(f"Seeds:            {args.n_seeds}")
    print("=" * 64)

    # 1. Sweep ACTOR_LR
    for param_name, param_values in DEFAULT_SEARCH_SPACE.items():
        print(f"\n>>> Tuning {param_name} across values: {param_values} <<<")
        scores = {}
        for val in param_values:
            candidate = dict(current_best)
            candidate[param_name] = val
            print(f"Testing {param_name}={val}...")
            env_scores = {}
            for env in envs:
                t0 = time.time()
                score = evaluate_condition(env, candidate, seed_keys, args.num_envs, args.num_steps, args.total_timesteps, args.minibatch_size)
                dt = time.time() - t0
                env_scores[env] = score
                print(f"  {env}: AUC Return = {score:.2f} ({dt:.1f}s)")
            scores[val] = env_scores

        # Rank values across envs via Min-Max score normalization
        env_mins = {env: min(scores[v][env] for v in param_values) for env in envs}
        env_maxs = {env: max(scores[v][env] for v in param_values) for env in envs}

        best_val = None
        best_norm_score = -np.inf
        for val in param_values:
            norm_scores = []
            for env in envs:
                span = env_maxs[env] - env_mins[env]
                norm = (scores[val][env] - env_mins[env]) / (span if span > 1e-6 else 1.0)
                norm_scores.append(norm)
            avg_norm = float(np.mean(norm_scores))
            history.append({
                "param_tuned": param_name,
                "value": val,
                "avg_norm_score": avg_norm,
                **{f"{env}_raw": scores[val][env] for env in envs},
                **copy.deepcopy(current_best),
                param_name: val,
            })
            if avg_norm > best_norm_score:
                best_norm_score = avg_norm
                best_val = val

        current_best[param_name] = best_val
        print(f"==> Best {param_name}: {best_val} (Avg Norm Score = {best_norm_score:.3f})")

    return current_best, pd.DataFrame(history)


def run_grid_tuning(args):
    """Full Cartesian grid search across all combinations."""
    envs = args.envs
    rng = jax.random.PRNGKey(args.master_seed)
    seed_keys = jax.random.split(rng, args.n_seeds)

    keys = list(DEFAULT_SEARCH_SPACE.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*DEFAULT_SEARCH_SPACE.values())]

    print("=" * 64)
    print(f"Running Full Grid Tuning: {len(combos)} combinations x {len(envs)} envs")
    print(f"Rollout Horizon:  {args.num_steps} steps (NUM_ENVS={args.num_envs})")
    print("=" * 64)

    raw_results = []
    for idx, combo in enumerate(combos):
        print(f"\n[{idx+1}/{len(combos)}] Testing config: {combo}")
        env_scores = {}
        for env in envs:
            t0 = time.time()
            score = evaluate_condition(env, combo, seed_keys, args.num_envs, args.num_steps, args.total_timesteps, args.minibatch_size)
            dt = time.time() - t0
            env_scores[env] = score
            print(f"  {env}: AUC Return = {score:.2f} ({dt:.1f}s)")
        raw_results.append({"config": combo, "env_scores": env_scores})

    # Min-Max Normalization
    env_mins = {env: min(r["env_scores"][env] for r in raw_results) for env in envs}
    env_maxs = {env: max(r["env_scores"][env] for r in raw_results) for env in envs}

    ranked = []
    for r in raw_results:
        norm_scores = []
        for env in envs:
            span = env_maxs[env] - env_mins[env]
            norm = (r["env_scores"][env] - env_mins[env]) / (span if span > 1e-6 else 1.0)
            norm_scores.append(norm)
        ranked.append({
            **r["config"],
            "avg_norm_score": float(np.mean(norm_scores)),
            **{f"{env}_return": r["env_scores"][env] for env in envs},
        })

    df = pd.DataFrame(ranked).sort_values(by="avg_norm_score", ascending=False)
    best_config = dict(df.iloc[0][keys].to_dict())
    return best_config, df


def main():
    parser = argparse.ArgumentParser(description="Tune PPO Actor Defaults on Jumanji via Pure Monte Carlo Critic")
    parser.add_argument("--mode", type=str, choices=["coordinate", "grid"], default="coordinate", help="Search strategy")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, help="Jumanji Environments")
    parser.add_argument("--num_steps", type=int, default=256, help="Extended rollout length for true MC (>= Sokoban time_limit 120)")
    parser.add_argument("--num_envs", type=int, default=128, help="Parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=500_000, help="Timesteps per evaluation condition")
    parser.add_argument("--minibatch_size", type=int, default=512, help="Minibatch size")
    parser.add_argument("--n_seeds", type=int, default=2, help="Seeds per condition")
    parser.add_argument("--master_seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--out_json", type=str, default="scripts/jumanji/jumanji_best_actor_config.json", help="Output JSON path")

    args = parser.parse_args()

    if args.mode == "coordinate":
        best_cfg, df = run_coordinate_tuning(args)
    else:
        best_cfg, df = run_grid_tuning(args)

    print("\n" + "=" * 64)
    print("FINAL TUNING RESULTS (Monte Carlo Grounded)")
    print("=" * 64)
    print(df.to_string(index=False))

    out_path = os.path.join(_repo_root, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_cfg, f, indent=4)

    print(f"\n[Success] Best cross-environment actor configuration saved to: {out_path}")
    print(json.dumps(best_cfg, indent=4))


if __name__ == "__main__":
    main()

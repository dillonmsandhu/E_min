#!/usr/bin/env python3
"""
scripts/brax/ablate_shared_torso.py

Ablation comparing:
  1. shared_torso    (SHARED_TORSO=True,  AUX_STOP_GRAD_TORSO=False)
  2. shared_torso_sg (SHARED_TORSO=True,  AUX_STOP_GRAD_TORSO=True)
  3. separate_torso  (SHARED_TORSO=False, AUX_STOP_GRAD_TORSO=False)
for E(lambda=0.8) on Brax environments.
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time

import jax
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from brax_rl.algos.ppo import make_train as make_ppo_train
from brax_rl.core.brax_config import config as brax_ppo_config
from scripts.brax.sweep_v_ppo.sweep_v_ppo import get_env_default_timesteps

DEFAULT_BRAX_5_ENVS = [
    "hopper",
    "walker2d",
    "halfcheetah",
    "ant",
    "inverted_pendulum",
]

VARIANTS = [
    ("shared_torso", True, False, False),
    ("shared_torso_sg", True, True, False),
    ("separate_torso", False, False, False),
    ("separate_torso_split_opt", False, False, True),
]


def main():
    p = argparse.ArgumentParser(description="Ablate SHARED_TORSO and stop_gradient on Brax E(lambda)")
    p.add_argument("--envs", nargs="+", default=DEFAULT_BRAX_5_ENVS)
    p.add_argument("--e_lambda", type=float, default=0.8)
    p.add_argument("--gae_lambda", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--minibatch_size", type=int, default=1024)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/brax/ablate_shared_torso")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_keys = jax.random.split(jax.random.PRNGKey(args.master_seed), args.n_seeds)
    results = {}

    for env in args.envs:
        results[env] = {}
        ts = get_env_default_timesteps(env, args.total_timesteps)
        print(f"\n=== {env} ({ts:,} steps | E(lambda={args.e_lambda}), lr={args.lr}, ep={args.epochs}) ===")
        for label, shared_torso, stop_grad, split_opt in VARIANTS:
            cfg = copy.deepcopy(brax_ppo_config)
            cfg["ENV_NAME"] = env
            cfg["TOTAL_TIMESTEPS"] = int(ts)
            cfg["CRITIC_TYPE"] = "e_lambda"
            cfg["E_VARIANT"] = "aux"
            cfg["E_LAMBDA"] = float(args.e_lambda)
            cfg["GAE_LAMBDA"] = float(args.gae_lambda)
            cfg["ACTOR_GAE_LAMBDA"] = float(args.gae_lambda)
            cfg["ACTOR_LR"] = float(args.lr)
            cfg["CRITIC_LR"] = float(args.lr)
            cfg["ACTOR_EPOCHS"] = int(args.epochs)
            cfg["CRITIC_EPOCHS"] = int(args.epochs)
            cfg["NUM_ENVS"] = int(args.num_envs)
            cfg["NUM_STEPS"] = int(args.num_steps)
            cfg["MINIBATCH_SIZE"] = int(args.minibatch_size)
            cfg["ACTOR_MINIBATCH_SIZE"] = int(args.minibatch_size)
            cfg["CRITIC_MINIBATCH_SIZE"] = int(args.minibatch_size)
            cfg["SHARED_TORSO"] = shared_torso
            cfg["AUX_STOP_GRAD_TORSO"] = stop_grad
            cfg["SPLIT_AUX_OPTIMIZER"] = split_opt

            t0 = time.time()
            if args.n_seeds == 1:
                train_fn = jax.jit(make_ppo_train(cfg))
                out = train_fn(seed_keys[0])
            else:
                train_fn = jax.jit(jax.vmap(make_ppo_train(cfg)))
                out = train_fn(seed_keys)
            jax.block_until_ready(out["metrics"]["returned_episode_returns"])
            dt = time.time() - t0

            rets = np.asarray(out["metrics"]["returned_episode_returns"])
            if rets.ndim == 1:
                rets = rets[None, :]
            auc = float(rets.mean(axis=-1).mean())
            auc_std = float(rets.mean(axis=-1).std())
            fin = float(rets[:, -1].mean())
            results[env][label] = {"returns": rets, "auc_mean": auc, "auc_std": auc_std, "final_mean": fin, "time": dt}
            print(f"  {label:20s} | AUC={auc:8.2f} ± {auc_std:6.2f} | Fin={fin:8.2f} ({dt:.1f}s)")

    with open(os.path.join(args.out_dir, "shared_torso_ablation.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/mjx/ablate_shared_torso.py

2-seed ablation comparing the 4 v_aux torso/optimizer variants on the 5 MJX DMC tasks:
  1. shared_torso             (SHARED_TORSO=True,  AUX_STOP_GRAD_TORSO=False, SPLIT_AUX_OPTIMIZER=False)
  2. shared_torso_sg          (SHARED_TORSO=True,  AUX_STOP_GRAD_TORSO=True,  SPLIT_AUX_OPTIMIZER=False)
  3. separate_torso           (SHARED_TORSO=False, AUX_STOP_GRAD_TORSO=False, SPLIT_AUX_OPTIMIZER=False)
  4. separate_torso_split_opt (SHARED_TORSO=False, AUX_STOP_GRAD_TORSO=False, SPLIT_AUX_OPTIMIZER=True)
"""

from __future__ import annotations

import argparse
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

from mjx_rl.algos.ppo import make_train as make_ppo_train
from mjx_rl.core.mjx_config import DEFAULT_DMC_5_ENVS, get_dmc_brax_parity_config

VARIANTS = [
    ("shared_torso", True, False, False),
    ("shared_torso_sg", True, True, False),
    ("separate_torso", False, False, False),
    ("separate_torso_split_opt", False, False, True),
]


def main():
    p = argparse.ArgumentParser(description="Ablate SHARED_TORSO, stop_gradient, and SPLIT_AUX_OPTIMIZER on MJX E(lambda)")
    p.add_argument("--envs", nargs="+", default=DEFAULT_DMC_5_ENVS)
    p.add_argument("--e_lambda", type=float, default=0.8)
    p.add_argument("--gae_lambda", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--minibatch_size", type=int, default=2048)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/mjx/ablate_shared_torso")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_keys = jax.random.split(jax.random.PRNGKey(args.master_seed), args.n_seeds)
    results = {}

    for env in args.envs:
        results[env] = {}
        cfg0 = get_dmc_brax_parity_config(env, impl="jax")
        ts = int(args.total_timesteps) if args.total_timesteps is not None else int(cfg0["TOTAL_TIMESTEPS"])
        print(f"\n=== {env} ({ts:,} steps | E(lambda={args.e_lambda}), lr={args.lr}, ep={args.epochs}, seeds={args.n_seeds}) ===")
        for label, shared_torso, stop_grad, split_opt in VARIANTS:
            cfg = get_dmc_brax_parity_config(env, impl="jax")
            cfg["TOTAL_TIMESTEPS"] = ts
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
            print(f"  {label:25s} | AUC={auc:8.2f} ± {auc_std:6.2f} | Fin={fin:8.2f} ({dt:.1f}s)")

    with open(os.path.join(args.out_dir, "shared_torso_ablation.pkl"), "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

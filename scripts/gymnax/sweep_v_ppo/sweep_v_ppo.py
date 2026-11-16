#!/usr/bin/env python3
"""
scripts/gymnax/sweep_v_ppo/sweep_v_ppo.py

Standardized Gymnax V-PPO sweep comparing:
  - td_lambda      : TD(λ) (fitted state-value critic)
  - e_0_v_aux      : E(0) (v_aux)
  - e_lambda_v_aux : E(λ) (v_aux)
  - bsmc_v_aux     : BSMC (v_aux)
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

from gymnax_rl.algos.ppo import make_train as make_ppo_train
from gymnax_rl.core.gymnax_config import config as gymnax_ppo_config
from scripts.sweep_standard_runner import run_standardized_sweep

METHOD_KEYS = ["td_lambda", "e_0_v_aux", "e_lambda_v_aux", "bsmc_v_aux"]
SUITE_TITLE = "Gymnax V-PPO"
REFERENCE_METHOD = "td_lambda"

DEFAULT_ENVS = [
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    "DeepSea-bsuite",
    "DiscountingChain-bsuite",
    "FourRooms-misc",
    "PointRobot-misc",
    "Reacher-misc",
    "Swimmer-misc",
    "Pong-misc",
]


def get_env_default_timesteps(env_name: str, fallback: int | None = None) -> int:
    if fallback is not None:
        return int(fallback)
    if "MinAtar" in env_name:
        return 10_000_000
    if "MountainCar" in env_name:
        return 1_000_000
    return 300_000


def build_base_config(
    env_name: str,
    actor_lr: float,
    gae_lambda: float,
    num_envs: int,
    num_steps: int,
    minibatch_size: int,
    aux_vf_coef: float,
) -> dict:
    cfg = copy.deepcopy(gymnax_ppo_config)
    cfg["ENV_NAME"] = env_name
    cfg["ACTOR_LR"] = float(actor_lr)
    cfg["CRITIC_LR"] = float(actor_lr)
    cfg["GAE_LAMBDA"] = float(gae_lambda)
    cfg["NUM_ENVS"] = int(num_envs)
    cfg["NUM_STEPS"] = int(num_steps)
    cfg["MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["ACTOR_MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["CRITIC_MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["ACTOR_EPOCHS"] = int(cfg["NUM_EPOCHS"])
    cfg["CRITIC_EPOCHS"] = int(cfg["NUM_EPOCHS"])
    cfg["AUX_VF_COEF"] = float(aux_vf_coef)
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standardized Gymnax V-PPO Critic Sweep")
    p.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    p.add_argument("--critic_lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--critic_epochs", nargs="+", type=int, default=[4, 8])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.4, 0.8, 0.95])
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--gae_lambda", type=float, default=0.8)
    p.add_argument("--default_critic_lr", type=float, default=3e-4)
    p.add_argument("--default_critic_epochs", type=int, default=4)
    p.add_argument("--default_lambda", type=float, default=0.8)
    p.add_argument("--aux_vf_coef", type=float, default=1.0)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--minibatch_size", type=int, default=512)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--metric_key", type=str, default="returned_episode_returns")
    p.add_argument("--out_dir", type=str, default="results/gymnax/sweep_v_ppo")
    p.add_argument("--load_existing", type=str, default=None)
    p.add_argument("--all_pairs", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p


def run_sweep(args) -> dict:
    return run_standardized_sweep(
        envs=args.envs,
        method_keys=METHOD_KEYS,
        make_train_fn=make_ppo_train,
        build_base_config_fn=lambda env: build_base_config(
            env_name=env,
            actor_lr=args.actor_lr,
            gae_lambda=args.gae_lambda,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            minibatch_size=args.minibatch_size,
            aux_vf_coef=args.aux_vf_coef,
        ),
        get_timesteps_fn=lambda env: get_env_default_timesteps(env, args.total_timesteps),
        is_brax=False,
        is_v_ppo=True,
        suite_title=SUITE_TITLE,
        reference_method=REFERENCE_METHOD,
        out_dir=args.out_dir,
        n_seeds=args.n_seeds,
        master_seed=args.master_seed,
        critic_lrs=args.critic_lrs,
        critic_epochs_list=args.critic_epochs,
        lambdas=args.lambdas,
        default_critic_lr=args.default_critic_lr,
        default_critic_epochs=args.default_critic_epochs,
        default_lambda=args.default_lambda,
        metric_key=args.metric_key,
        load_existing=args.load_existing,
        generate_plots=not args.no_plots,
        all_pairs=args.all_pairs,
    )


def main():
    args = build_arg_parser().parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

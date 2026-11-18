#!/usr/bin/env python3
"""
scripts/mjx/sweep_v_ppo/sweep_v_ppo.py

Standardized MuJoCo Playground (MJX) V-PPO sweep comparing:
  - td_lambda      : TD(λ) (fitted state-value critic)
  - e_lambda_v_aux : E(λ) (v_aux)

Starts from the Brax PPO parity configuration (`get_dmc_brax_parity_config`)
or any tuned base hyperparameters passed via CLI flags.
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
    p
    for p in sys.path
    if not (
        os.path.exists(os.path.join(p, "fixed_policy"))
        or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy")))
    )
]

from mjx_rl.algos.ppo import make_train as make_ppo_train
from mjx_rl.core.mjx_config import DEFAULT_DMC_5_ENVS, get_dmc_brax_parity_config
from scripts.sweep_standard_runner import run_standardized_sweep

METHOD_KEYS = ["td_lambda", "e_lambda_v_aux"]
SUITE_TITLE = "MJX Playground V-PPO"
REFERENCE_METHOD = "td_lambda"


def get_env_default_timesteps(env_name: str, fallback: int | None = None) -> int:
    if fallback is not None:
        return int(fallback)
    cfg = get_dmc_brax_parity_config(env_name)
    return int(cfg["TOTAL_TIMESTEPS"])


def build_base_config(
    env_name: str,
    impl: str,
    actor_lr: float,
    gae_lambda: float,
    num_envs: int,
    num_steps: int,
    minibatch_size: int,
    actor_epochs: int,
    clip_eps: float,
    ent_coef: float,
    reward_scaling: float,
    aux_vf_coef: float,
) -> dict:
    cfg = get_dmc_brax_parity_config(env_name, impl=impl)
    cfg["ACTOR_LR"] = float(actor_lr)
    cfg["CRITIC_LR"] = float(actor_lr)
    cfg["GAE_LAMBDA"] = float(gae_lambda)
    cfg["ACTOR_GAE_LAMBDA"] = float(gae_lambda)
    cfg["NUM_ENVS"] = int(num_envs)
    cfg["NUM_STEPS"] = int(num_steps)
    cfg["MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["ACTOR_MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["CRITIC_MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["ACTOR_EPOCHS"] = int(actor_epochs)
    cfg["CRITIC_EPOCHS"] = int(actor_epochs)
    cfg["CLIP_EPS"] = float(clip_eps)
    cfg["ENT_COEF"] = float(ent_coef)
    cfg["REWARD_SCALING"] = float(reward_scaling)
    cfg["AUX_VF_COEF"] = float(aux_vf_coef)
    net_cfg = copy.deepcopy(cfg["NETWORK_CONFIG"])
    net_cfg["LR"] = float(actor_lr)
    net_cfg["NUM_EPOCHS"] = int(actor_epochs)
    net_cfg["MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["NETWORK_CONFIG"] = net_cfg
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standardized MJX Playground V-PPO Critic Sweep (E(λ) vs TD(λ))")
    p.add_argument("--envs", nargs="+", default=DEFAULT_DMC_5_ENVS)
    p.add_argument("--methods", nargs="+", default=METHOD_KEYS)
    p.add_argument("--impl", type=str, default="jax", choices=["jax", "warp"])
    p.add_argument("--critic_lrs", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--critic_epochs", nargs="+", type=int, default=[8, 16])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.8, 0.95, 0.99])
    p.add_argument("--actor_lr", type=float, default=1e-3)
    p.add_argument("--actor_epochs", type=int, default=16)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.3)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--reward_scaling", type=float, default=10.0)
    p.add_argument("--default_critic_lr", type=float, default=1e-3)
    p.add_argument("--default_critic_epochs", type=int, default=16)
    p.add_argument("--default_lambda", type=float, default=0.95)
    p.add_argument("--aux_vf_coef", type=float, default=1.0)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--minibatch_size", type=int, default=1920)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--metric_key", type=str, default="returned_episode_returns")
    p.add_argument("--out_dir", type=str, default="results/mjx/sweep_v_ppo")
    p.add_argument("--load_existing", type=str, default=None)
    p.add_argument("--all_pairs", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p


def run_sweep(args) -> dict:
    return run_standardized_sweep(
        envs=args.envs,
        method_keys=list(args.methods),
        make_train_fn=make_ppo_train,
        build_base_config_fn=lambda env: build_base_config(
            env_name=env,
            impl=args.impl,
            actor_lr=args.actor_lr,
            gae_lambda=args.gae_lambda,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            minibatch_size=args.minibatch_size,
            actor_epochs=args.actor_epochs,
            clip_eps=args.clip_eps,
            ent_coef=args.ent_coef,
            reward_scaling=args.reward_scaling,
            aux_vf_coef=args.aux_vf_coef,
        ),
        get_timesteps_fn=lambda env: get_env_default_timesteps(env, args.total_timesteps),
        is_brax=True,
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

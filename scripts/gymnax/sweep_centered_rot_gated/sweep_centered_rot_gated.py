#!/usr/bin/env python3
"""
scripts/gymnax/sweep_centered_rot_gated/sweep_centered_rot_gated.py

Gymnax V-PPO sweep using Centered Rotation Gating (J_c = J - E[J], tau = 0.25)
across 3 targeted conditions:
  1. SpaceInvaders-MinAtar         : Grid CNN (localized, low aliasing)
  2. SpaceInvaders-MinAtar-AvgPool : Grid CNN + 3x3 AvgPool (aliased spatial features)
  3. MountainCarContinuous-v0      : Continuous (x, v) + MLP (smooth aliased spiral)

Methods compared:
  - td_0               : Online 1-step TD(0)
  - e_0_v_aux          : Single-sample Dirichlet E(0) (v_aux)
  - e_0_rot_gated      : Centered Rotation-Gated E(0) (tau=0.25, slow EMA)
  - td_lambda          : Fitted TD(lambda)
  - e_lambda_v_aux     : Single-sample Geometric E(lambda) (v_aux)
  - e_lambda_rot_gated : Centered Rotation-Gated E(lambda) (tau=0.25, lambda-trace probe)
"""

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

from gymnax_rl.algos.ppo import make_train as make_v_ppo_train
from scripts.gymnax.sweep_v_ppo.sweep_v_ppo import build_base_config, get_env_default_timesteps
from scripts.sweep_standard_runner import run_standardized_sweep

METHOD_KEYS = [
    "td_0",
    "e_0_v_aux",
    "e_0_rot_gated",
    "td_lambda",
    "e_lambda_v_aux",
    "e_lambda_rot_gated",
]
SUITE_TITLE = "Centered Rotation-Gated E (τ=0.25): SpaceInvaders (CNN vs AvgPool) & Continuous MountainCar"
REFERENCE_METHOD = "td_lambda"

DEFAULT_ENVS = [
    "SpaceInvaders-MinAtar",
    "SpaceInvaders-MinAtar-AvgPool",
    "MountainCarContinuous-v0",
]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Centered Rotation-Gated E(0) & E(lambda) V-PPO Sweep")
    p.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    p.add_argument("--critic_lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    p.add_argument("--critic_epochs", nargs="+", type=int, default=[4, 8])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.5, 0.8])
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--gae_lambda", type=float, default=0.8)
    p.add_argument("--default_critic_lr", type=float, default=3e-4)
    p.add_argument("--default_critic_epochs", type=int, default=4)
    p.add_argument("--default_lambda", type=float, default=0.8)
    p.add_argument("--aux_vf_coef", type=float, default=1.0)
    p.add_argument("--rot_gate_tau", type=float, default=0.25)
    p.add_argument("--rot_gate_ema", type=float, default=0.95)
    p.add_argument("--rot_probe_envs", type=int, default=16)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--minibatch_size", type=int, default=512)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--metric_key", type=str, default="returned_episode_returns")
    p.add_argument("--out_dir", type=str, default="results/gymnax/sweep_centered_rot_gated")
    p.add_argument("--load_existing", type=str, default=None)
    p.add_argument("--all_pairs", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p


def _resolve_env_name(env_key: str) -> tuple[str, bool]:
    if env_key == "SpaceInvaders-MinAtar-AvgPool":
        return "SpaceInvaders-MinAtar", True
    return env_key, False


def _build_env_base_config(env_key: str, args) -> dict:
    real_env, use_avg_pool = _resolve_env_name(env_key)
    cfg = build_base_config(
        env_name=real_env,
        actor_lr=args.actor_lr,
        gae_lambda=args.gae_lambda,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        minibatch_size=args.minibatch_size,
        aux_vf_coef=args.aux_vf_coef,
    )
    cfg["NETWORK_TYPE"] = "cnn" if "MinAtar" in real_env else "mlp"
    cfg["CNN_AVG_POOL"] = bool(use_avg_pool)
    cfg["ROT_GATE_TAU"] = float(args.rot_gate_tau)
    cfg["ROT_GATE_EMA"] = float(args.rot_gate_ema)
    cfg["ROT_PROBE_ENVS"] = int(args.rot_probe_envs)
    return cfg


def _get_timesteps(env_key: str, fallback: int | None) -> int:
    real_env, _ = _resolve_env_name(env_key)
    return get_env_default_timesteps(real_env, fallback)


def run_sweep(args) -> dict:
    return run_standardized_sweep(
        envs=args.envs,
        method_keys=METHOD_KEYS,
        make_train_fn=make_v_ppo_train,
        build_base_config_fn=lambda env: _build_env_base_config(env, args),
        get_timesteps_fn=lambda env: _get_timesteps(env, args.total_timesteps),
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

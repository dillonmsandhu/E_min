#!/usr/bin/env python3
"""
scripts/mjx/sweep_locomotion/sweep_locomotion.py

Standardized MuJoCo Playground (MJX) Locomotion V-PPO sweep comparing:
  - td_lambda      : TD(λ) (fitted state-value critic)
  - td_0           : TD(0) (online 1-step TD)
  - e_0_v_aux      : E(0) (v_aux)
  - e_lambda_v_aux : E(λ) (v_aux)

Uses per-environment Brax-parity locomotion settings from
`get_locomotion_brax_parity_config` (privileged_state critic, (512, 256, 128) MLPs,
γ=0.97, reward_scaling=1.0) with NUM_ENVS=512, NUM_STEPS=128.
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

from mjx_rl.algos.ppo import (
    extract_hparams,
    make_train as make_ppo_train,
    static_compile_signature,
)
from mjx_rl.core.mjx_config import DEFAULT_LOCOMOTION_5_ENVS, get_locomotion_brax_parity_config
from scripts.sweep_standard_plotting import DEFAULT_PAIRWISE_V_PPO, generate_standardized_suite
from scripts.sweep_standard_runner import _apply_params, build_candidates_for_method

METHOD_KEYS = ["td_lambda", "td_0", "e_0_v_aux", "e_lambda_v_aux"]
SUITE_TITLE = "MJX Playground Locomotion V-PPO"
REFERENCE_METHOD = "td_lambda"


def get_env_default_timesteps(env_name: str, fallback: int | None = None) -> int:
    if fallback is not None:
        return int(fallback)
    return int(get_locomotion_brax_parity_config(env_name)["TOTAL_TIMESTEPS"])


def build_base_config(
    env_name: str,
    impl: str,
    actor_lr: float,
    gae_lambda: float,
    num_envs: int,
    num_steps: int,
    minibatch_size: int,
    actor_epochs: int,
    aux_vf_coef: float,
) -> dict:
    cfg = get_locomotion_brax_parity_config(env_name, impl=impl)
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
    cfg["AUX_VF_COEF"] = float(aux_vf_coef)
    net_cfg = copy.deepcopy(cfg["NETWORK_CONFIG"])
    net_cfg["LR"] = float(actor_lr)
    net_cfg["NUM_EPOCHS"] = int(actor_epochs)
    net_cfg["MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["NETWORK_CONFIG"] = net_cfg
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standardized MJX Playground Locomotion V-PPO Critic Sweep")
    p.add_argument("--envs", nargs="+", default=DEFAULT_LOCOMOTION_5_ENVS)
    p.add_argument("--methods", nargs="+", default=METHOD_KEYS)
    p.add_argument("--impl", type=str, default="jax", choices=["jax", "warp"])
    p.add_argument("--critic_lrs", nargs="+", type=float, default=[3e-4, 1e-3])
    p.add_argument("--critic_epochs", nargs="+", type=int, default=[4, 8])
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.6, 0.8, 0.95, 1.0])
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--actor_epochs", type=int, default=4)
    p.add_argument("--gae_lambda", type=float, default=0.8)
    p.add_argument("--default_critic_lr", type=float, default=3e-4)
    p.add_argument("--default_critic_epochs", type=int, default=4)
    p.add_argument("--default_lambda", type=float, default=0.95)
    p.add_argument("--aux_vf_coef", type=float, default=1.0)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--minibatch_size", type=int, default=2048)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--metric_key", type=str, default="returned_episode_returns")
    p.add_argument("--out_dir", type=str, default="results/mjx/sweep_locomotion")
    p.add_argument("--load_existing", type=str, default=None)
    p.add_argument("--all_pairs", action="store_true")
    p.add_argument("--no_plots", action="store_true")
    return p


def run_sweep(args) -> dict:
    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, "sweep_metrics_all.pkl")
    method_keys = list(args.methods)

    if args.load_existing and os.path.exists(args.load_existing):
        with open(args.load_existing, "rb") as f:
            payload = pickle.load(f)
        all_runs = payload["runs"]
        timesteps = payload["timesteps"]
        batch_sizes = payload["batch_sizes"]
    else:
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)
            all_runs = payload.get("runs", {})
            timesteps = payload.get("timesteps", {})
            batch_sizes = payload.get("batch_sizes", {})
        else:
            all_runs = {}
            timesteps = {}
            batch_sizes = {}
        seed_keys = jax.random.split(jax.random.PRNGKey(args.master_seed), args.n_seeds)

        for env in args.envs:
            env_ts = int(get_env_default_timesteps(env, args.total_timesteps))
            base_cfg = build_base_config(
                env_name=env,
                impl=args.impl,
                actor_lr=args.actor_lr,
                gae_lambda=args.gae_lambda,
                num_envs=args.num_envs,
                num_steps=args.num_steps,
                minibatch_size=args.minibatch_size,
                actor_epochs=args.actor_epochs,
                aux_vf_coef=args.aux_vf_coef,
            )
            base_cfg["TOTAL_TIMESTEPS"] = env_ts
            timesteps[env] = env_ts
            batch_sizes[env] = int(base_cfg["NUM_ENVS"]) * int(base_cfg["NUM_STEPS"])
            if env not in all_runs:
                all_runs[env] = {}

            compiled_cache = {}
            print(f"\n=== Locomotion: {env} ({env_ts:,} steps, bs={batch_sizes[env]}, val_obs={base_cfg['VALUE_OBS_KEY']}) ===")
            for m_key in method_keys:
                if m_key not in all_runs[env]:
                    all_runs[env][m_key] = {}
                cands = build_candidates_for_method(
                    method_key=m_key,
                    critic_lrs=args.critic_lrs,
                    critic_epochs_list=args.critic_epochs,
                    lambdas=args.lambdas,
                    default_critic_lr=args.default_critic_lr,
                    default_critic_epochs=args.default_critic_epochs,
                    default_lambda=args.default_lambda,
                )
                print(f"  [{m_key}] Running {len(cands)} configs x {args.n_seeds} seeds...")
                for cand in cands:
                    tag = cand["tag"]
                    if tag in all_runs[env][m_key]:
                        res = all_runs[env][m_key][tag]
                        print(
                            f"    {tag:28s} | [CACHED] AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} "
                            f"| Fin={res['final_mean']:8.2f}"
                        )
                        continue
                    cfg = _apply_params(base_cfg, m_key, cand["params"], is_brax=True, is_v_ppo=True)
                    sig = static_compile_signature(cfg)
                    if sig not in compiled_cache:
                        train_fn = make_ppo_train(cfg, dynamic_hparams=True)
                        compiled_cache[sig] = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))

                    t0 = time.time()
                    out = compiled_cache[sig](seed_keys, extract_hparams(cfg))
                    jax.tree_util.tree_map(
                        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                        out["metrics"][args.metric_key],
                    )
                    elapsed = time.time() - t0

                    returns = np.asarray(out["metrics"][args.metric_key])
                    if returns.ndim == 1:
                        returns = returns[None, :]
                    seed_aucs = returns.mean(axis=-1)
                    seed_finals = returns[:, -1]

                    res = {
                        "returned_episode_returns": returns,
                        "auc_mean": float(seed_aucs.mean()),
                        "auc_std": float(seed_aucs.std()),
                        "final_mean": float(seed_finals.mean()),
                        "final_std": float(seed_finals.std()),
                        "elapsed_time": float(elapsed),
                        "params": cand["params"],
                        "config": cfg,
                    }
                    for extra_key in (
                        "diag_rho_c",
                        "diag_r_rot_c",
                        "diag_env_ret_var",
                        "diag_ret_var_ratio",
                        "diag_v_r2",
                        "diag_v_rel_mse",
                        "diag_aux_r2",
                        "diag_aux_vs_v",
                        "diag_alpha_eff",
                        "diag_beta_eff",
                    ):
                        if extra_key in out["metrics"]:
                            arr = np.asarray(out["metrics"][extra_key])
                            if arr.ndim == 1:
                                arr = arr[None, :]
                            res[extra_key] = arr
                    all_runs[env][m_key][tag] = res
                    print(
                        f"    {tag:28s} | AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} "
                        f"| Fin={res['final_mean']:8.2f} ({res['elapsed_time']:.1f}s)"
                    )
                    with open(pkl_path, "wb") as f:
                        pickle.dump(
                            {
                                "runs": all_runs,
                                "timesteps": timesteps,
                                "batch_sizes": batch_sizes,
                                "method_keys": method_keys,
                            },
                            f,
                        )

    if not args.no_plots and all_runs:
        sample_cfg = build_base_config(
            env_name=args.envs[0],
            impl=args.impl,
            actor_lr=args.actor_lr,
            gae_lambda=args.gae_lambda,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            minibatch_size=args.minibatch_size,
            actor_epochs=args.actor_epochs,
            aux_vf_coef=args.aux_vf_coef,
        )
        base_summary = {
            "ACTOR_LR": sample_cfg["ACTOR_LR"],
            "GAE_LAMBDA": sample_cfg["GAE_LAMBDA"],
            "DEFAULT_CRITIC_LR": args.default_critic_lr,
            "DEFAULT_CRITIC_EPOCHS": args.default_critic_epochs,
            "DEFAULT_LAMBDA": args.default_lambda,
            "CRITIC_LRS": args.critic_lrs,
            "CRITIC_EPOCHS": args.critic_epochs,
            "LAMBDAS": args.lambdas,
            "N_SEEDS": args.n_seeds,
        }
        generate_standardized_suite(
            all_runs=all_runs,
            method_keys=method_keys,
            timesteps=timesteps,
            out_dir=args.out_dir,
            suite_title=SUITE_TITLE,
            batch_sizes=batch_sizes,
            base_config=base_summary,
            reference_method=REFERENCE_METHOD,
            pairwise_pairs=DEFAULT_PAIRWISE_V_PPO,
            all_pairs=args.all_pairs,
        )

    return all_runs


def main():
    args = build_arg_parser().parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()

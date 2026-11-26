"""
scripts/sweep_standard_runner.py

Shared execution, candidate-grid construction, and pickle consolidation
utilities for the 4 standardized sweeps:
  - scripts/brax/sweep_v_ppo
  - scripts/gymnax/sweep_v_ppo
  - scripts/brax/sweep_q_ppo
  - scripts/gymnax/sweep_q_ppo
"""

from __future__ import annotations

import copy
import glob
import os
import pickle
import time
from typing import Any, Callable

import jax
import numpy as np

from scripts.sweep_standard_plotting import (
    DEFAULT_PAIRWISE_Q_PPO,
    DEFAULT_PAIRWISE_V_PPO,
    compute_auc,
    generate_standardized_suite,
)


def _method_critic_type(method_key: str) -> str:
    mapping = {
        "td_lambda": "fitted",
        "td_0": "td_0",
        "msbe": "msbe",
        "sarsa_lambda": "sarsa_lambda",
        "e_0_v_aux": "e_0_v_aux",
        "e_0_rot_gated": "e_0",
        "e_0_bilinear": "e_0_bilinear",
        "e_0_dir_marg": "e_0_dir_marg",
        "e_0_dir_double": "e_dirichlet_double",
        "e_lambda_v_aux": "e_lambda_v_aux",
        "e_lambda_rot_gated": "e_lambda",
        "bsmc_v_aux": "bsmc_v_aux",
        "bsmc_fwd": "bsmc",
        "bsmc_double_pi": "bsmc_double_pi",
        "bsmc_bilinear_v_aux": "bsmc_bilinear_v_aux",
        "bsmc_dir_marg": "bsmc_dir_marg",
        "bsmc_dir_double": "bsmc_dir_double",
    }
    if method_key not in mapping:
        raise ValueError(f"Unknown canonical method_key '{method_key}'.")
    return mapping[method_key]


def _method_lambda_key(method_key: str) -> str | None:
    if method_key in ("td_lambda", "sarsa_lambda"):
        return "VALUE_LAMBDA"
    if method_key in ("e_lambda_v_aux", "e_lambda_rot_gated"):
        return "E_LAMBDA"
    if method_key in ("bsmc_v_aux", "bsmc_fwd", "bsmc_double_pi", "bsmc_bilinear_v_aux", "bsmc_dir_marg", "bsmc_dir_double"):
        return "BSMC_LAMBDA"
    if method_key in ("e_0_v_aux", "e_0_rot_gated", "e_0_bilinear", "e_0_dir_marg", "e_0_dir_double", "td_0", "msbe"):
        return None
    raise ValueError(f"Unknown canonical method_key '{method_key}'.")


def build_candidates_for_method(
    method_key: str,
    critic_lrs: list[float],
    critic_epochs_list: list[int],
    lambdas: list[float],
    default_critic_lr: float,
    default_critic_epochs: int,
    default_lambda: float,
) -> list[dict[str, Any]]:
    """
    Constructs a list of {'tag', 'params'} dictionaries starting with 'default'
    followed by the tuned hyperparameter grid (CRITIC_LR x CRITIC_EPOCHS [x LAMBDA]).
    """
    lambda_key = _method_lambda_key(method_key)
    candidates: list[dict[str, Any]] = []
    seen = set()

    def _add(params: dict[str, Any], tag: str) -> None:
        key = tuple(sorted(params.items()))
        if key not in seen:
            seen.add(key)
            candidates.append({"tag": tag, "params": dict(params)})

    default_params: dict[str, Any] = {
        "CRITIC_LR": float(default_critic_lr),
        "CRITIC_EPOCHS": int(default_critic_epochs),
    }
    if lambda_key is not None:
        default_params[lambda_key] = float(default_lambda)
    _add(default_params, "default")

    lambda_vals = [float(x) for x in lambdas] if lambda_key is not None else [None]
    for lr in critic_lrs:
        for ep in critic_epochs_list:
            for lmb in lambda_vals:
                p: dict[str, Any] = {
                    "CRITIC_LR": float(lr),
                    "CRITIC_EPOCHS": int(ep),
                }
                tag = f"clr={lr}_cep={ep}"
                if lambda_key is not None and lmb is not None:
                    p[lambda_key] = float(lmb)
                    tag += f"_lam={lmb}"
                _add(p, tag)

    return candidates


def _apply_params(
    base_cfg: dict[str, Any],
    method_key: str,
    params: dict[str, Any],
    is_brax: bool,
    is_v_ppo: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if is_v_ppo and method_key in ("e_0_v_aux", "e_0_rot_gated"):
        cfg["CRITIC_TYPE"] = "e_0"
    elif is_v_ppo and method_key in ("e_lambda_v_aux", "e_lambda_rot_gated"):
        cfg["CRITIC_TYPE"] = "e_lambda"
    elif is_v_ppo and method_key == "bsmc_v_aux":
        cfg["CRITIC_TYPE"] = "bsmc"
    else:
        cfg["CRITIC_TYPE"] = _method_critic_type(method_key)

    if method_key in ("e_0_rot_gated", "e_lambda_rot_gated"):
        cfg["ROT_GATE_ENABLED"] = True
    else:
        cfg["ROT_GATE_ENABLED"] = False

    if method_key in ("e_0_bilinear", "e_0_dir_marg", "e_0_dir_double", "td_0", "msbe", "bsmc_double_pi", "bsmc_dir_marg", "bsmc_dir_double"):
        cfg["E_VARIANT"] = "naive"
    elif method_key == "bsmc_fwd":
        cfg["E_VARIANT"] = "fwd"
    else:
        cfg["E_VARIANT"] = "aux"

    for k, v in params.items():
        cfg[k] = v

    if is_brax:
        net_cfg = copy.deepcopy(cfg["NETWORK_CONFIG"])
        ratio = float(net_cfg["LR_END"]) / max(float(net_cfg["LR"]), 1e-12)
        net_cfg["LR"] = float(cfg["ACTOR_LR"])
        net_cfg["LR_END"] = ratio * float(cfg["ACTOR_LR"])
        cfg["NETWORK_CONFIG"] = net_cfg
        cfg["ACTOR_LR_END"] = ratio * float(cfg["ACTOR_LR"])
        cfg["CRITIC_LR_END"] = ratio * float(cfg["CRITIC_LR"])
    return cfg


def run_candidate(make_train_fn: Callable, cfg: dict[str, Any], seed_keys: jax.Array, metric_key: str) -> dict[str, Any]:
    train_fn = make_train_fn(cfg)
    vmapped = jax.jit(jax.vmap(train_fn))
    t0 = time.time()
    out = vmapped(seed_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - t0

    metrics = out["metrics"]
    returns = np.asarray(metrics[metric_key])
    if returns.ndim == 1:
        returns = returns[None, :]
    seed_aucs = returns.mean(axis=-1)
    seed_finals = returns[:, -1]

    result = {
        "returned_episode_returns": returns,
        "auc_mean": float(seed_aucs.mean()),
        "auc_std": float(seed_aucs.std()),
        "final_mean": float(seed_finals.mean()),
        "final_std": float(seed_finals.std()),
        "elapsed_time": float(elapsed),
    }
    for extra_key in (
        "rot_omega",
        "rot_r_rot_ema",
        "rot_alpha_eff",
        "rot_beta_eff",
        "rot_rho_c",
        "diag_rho_c",
        "diag_r_rot_c",
        "diag_ret_var_ratio",
        "diag_v_r2",
        "diag_aux_r2",
        "diag_aux_vs_v",
        "diag_alpha_eff",
        "diag_beta_eff",
    ):
        if extra_key in metrics:
            arr = np.asarray(metrics[extra_key])
            if arr.ndim == 1:
                arr = arr[None, :]
            result[extra_key] = arr
    return result


def run_standardized_sweep(
    envs: list[str],
    method_keys: list[str],
    make_train_fn: Callable,
    build_base_config_fn: Callable[[str], dict[str, Any]],
    get_timesteps_fn: Callable[[str], int],
    is_brax: bool,
    is_v_ppo: bool,
    suite_title: str,
    reference_method: str,
    out_dir: str,
    n_seeds: int,
    master_seed: int,
    critic_lrs: list[float],
    critic_epochs_list: list[int],
    lambdas: list[float],
    default_critic_lr: float,
    default_critic_epochs: int,
    default_lambda: float,
    metric_key: str = "returned_episode_returns",
    load_existing: str | None = None,
    generate_plots: bool = True,
    all_pairs: bool = False,
) -> dict[str, Any]:
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, "sweep_metrics_all.pkl")

    if load_existing and os.path.exists(load_existing):
        with open(load_existing, "rb") as f:
            payload = pickle.load(f)
        all_runs = payload["runs"]
        timesteps = payload["timesteps"]
        batch_sizes = payload["batch_sizes"]
    else:
        all_runs: dict[str, dict[str, dict[str, Any]]] = {}
        timesteps: dict[str, int] = {}
        batch_sizes: dict[str, int] = {}

        master_key = jax.random.PRNGKey(master_seed)
        seed_keys = jax.random.split(master_key, n_seeds)

        for env in envs:
            env_ts = int(get_timesteps_fn(env))
            base_cfg = build_base_config_fn(env)
            base_cfg["TOTAL_TIMESTEPS"] = env_ts
            timesteps[env] = env_ts
            batch_sizes[env] = int(base_cfg["NUM_ENVS"]) * int(base_cfg["NUM_STEPS"])
            all_runs[env] = {}

            print(f"\n=== Environment: {env} ({env_ts:,} steps, bs={batch_sizes[env]}) ===")
            for m_key in method_keys:
                all_runs[env][m_key] = {}
                cands = build_candidates_for_method(
                    method_key=m_key,
                    critic_lrs=critic_lrs,
                    critic_epochs_list=critic_epochs_list,
                    lambdas=lambdas,
                    default_critic_lr=default_critic_lr,
                    default_critic_epochs=default_critic_epochs,
                    default_lambda=default_lambda,
                )
                print(f"  [{m_key}] Running {len(cands)} configs x {n_seeds} seeds...")
                for cand in cands:
                    tag = cand["tag"]
                    cfg = _apply_params(base_cfg, m_key, cand["params"], is_brax=is_brax, is_v_ppo=is_v_ppo)
                    train_factory = make_train_fn[m_key] if isinstance(make_train_fn, dict) else make_train_fn
                    res = run_candidate(train_factory, cfg, seed_keys, metric_key=metric_key)
                    res["params"] = cand["params"]
                    res["config"] = cfg
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

    if generate_plots and all_runs:
        sample_cfg = build_base_config_fn(envs[0]) if envs else {}
        base_summary = {
            "ACTOR_LR": sample_cfg.get("ACTOR_LR"),
            "GAE_LAMBDA": sample_cfg.get("GAE_LAMBDA"),
            "DEFAULT_CRITIC_LR": default_critic_lr,
            "DEFAULT_CRITIC_EPOCHS": default_critic_epochs,
            "DEFAULT_LAMBDA": default_lambda,
            "CRITIC_LRS": critic_lrs,
            "CRITIC_EPOCHS": critic_epochs_list,
            "LAMBDAS": lambdas,
            "N_SEEDS": n_seeds,
        }
        if not is_v_ppo:
            base_summary["K_ACTIONS"] = sample_cfg.get("K_ACTIONS")
            base_summary["GAE_CONTINUOUS_MODE"] = sample_cfg.get("GAE_CONTINUOUS_MODE")

        pairs = DEFAULT_PAIRWISE_V_PPO if is_v_ppo else DEFAULT_PAIRWISE_Q_PPO
        generate_standardized_suite(
            all_runs=all_runs,
            method_keys=method_keys,
            timesteps=timesteps,
            out_dir=out_dir,
            suite_title=suite_title,
            batch_sizes=batch_sizes,
            base_config=base_summary,
            reference_method=reference_method,
            pairwise_pairs=pairs,
            all_pairs=all_pairs,
        )

    return all_runs


def consolidate_standardized_sweep(
    out_dir: str,
    method_keys: list[str],
    suite_title: str,
    reference_method: str,
    is_v_ppo: bool,
    save_dir: str | None = None,
    all_pairs: bool = False,
) -> None:
    env_pickles = sorted(glob.glob(os.path.join(out_dir, "*", "data", "sweep_metrics_all.pkl")))
    if not env_pickles:
        single_pkl = os.path.join(out_dir, "data", "sweep_metrics_all.pkl")
        if os.path.exists(single_pkl):
            env_pickles = [single_pkl]
    if not env_pickles:
        print(f"[Warning] No sweep_metrics_all.pkl found under {out_dir}")
        return

    consolidated_runs: dict[str, Any] = {}
    timesteps: dict[str, int] = {}
    batch_sizes: dict[str, int] = {}

    for pkl_path in env_pickles:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        for env, env_data in d.get("runs", {}).items():
            consolidated_runs[env] = env_data
            if env in d.get("timesteps", {}):
                timesteps[env] = d["timesteps"][env]
            if env in d.get("batch_sizes", {}):
                batch_sizes[env] = d["batch_sizes"][env]

    if not consolidated_runs:
        print("No valid runs loaded. Exiting.")
        return

    target_dir = save_dir if save_dir else out_dir
    cons_data_dir = os.path.join(target_dir, "data")
    os.makedirs(cons_data_dir, exist_ok=True)
    with open(os.path.join(cons_data_dir, "sweep_metrics_all.pkl"), "wb") as f:
        pickle.dump(
            {
                "runs": consolidated_runs,
                "timesteps": timesteps,
                "batch_sizes": batch_sizes,
                "method_keys": method_keys,
            },
            f,
        )

    first_env = sorted(consolidated_runs.keys())[0]
    first_m = next(iter(consolidated_runs[first_env].values()))
    first_r = next(iter(first_m.values()))
    sample_cfg = first_r.get("config", {})
    base_summary = {
        "ACTOR_LR": sample_cfg.get("ACTOR_LR"),
        "GAE_LAMBDA": sample_cfg.get("GAE_LAMBDA"),
        "NUM_ENVS": sample_cfg.get("NUM_ENVS"),
        "NUM_STEPS": sample_cfg.get("NUM_STEPS"),
    }
    if not is_v_ppo:
        base_summary["K_ACTIONS"] = sample_cfg.get("K_ACTIONS")
        base_summary["GAE_CONTINUOUS_MODE"] = sample_cfg.get("GAE_CONTINUOUS_MODE")

    pairs = DEFAULT_PAIRWISE_V_PPO if is_v_ppo else DEFAULT_PAIRWISE_Q_PPO
    generate_standardized_suite(
        all_runs=consolidated_runs,
        method_keys=method_keys,
        timesteps=timesteps,
        out_dir=target_dir,
        suite_title=suite_title,
        batch_sizes=batch_sizes,
        base_config=base_summary,
        reference_method=reference_method,
        pairwise_pairs=pairs,
        all_pairs=all_pairs,
    )

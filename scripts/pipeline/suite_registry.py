"""
scripts/pipeline/suite_registry.py

Single source of truth for all 5 Deep RL suites in the ICML 2027 two-stage pipeline:
  - brax           : 10 Brax classic continuous-control envs (γ=0.99)
  - mjx_locomotion : 5 MuJoCo Playground locomotion envs (γ=0.97, no Brax overlap)
  - octax          : 8 Octax visual/discrete CNN envs (5 tune -> 8 eval)
  - minatar        : 4 MinAtar visual/discrete CNN envs (separate from Gymnax)
  - gymnax         : 12 Gymnax non-MinAtar diagnostic envs (Acrobot, MountainCar, etc.)

Enforces strict parity invariants:
  - VF_COEF = 1.0 (both top-level and NETWORK_CONFIG["VF_COEF"])
  - VF_CLIP = None (regular fitted TD(λ), no heuristic PPO value clipping)
  - SPLIT_AUX_OPTIMIZER = True (dedicated optimizer + grad clipping for aux_loss)
  - MAX_GRAD_NORM propagated to both top-level and NETWORK_CONFIG["MAX_GRAD_NORM"]
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import jax
import numpy as np

from scripts.sweep_standard_runner import _apply_params, _method_lambda_key

# The 4 separately tuned base algorithms (Stage 1)
TUNED_BASE_ALGOS: list[str] = [
    "td_0",
    "e_0_v_aux",
    "td_lambda",
    "e_lambda_v_aux",
]

# Inherited ablations: not separately tuned in Stage 1; inherit base algo's JSON in Stage 2
ABLATION_INHERITANCE: dict[str, str] = {
    "bsmc_v_aux": "e_lambda_v_aux",
    "e_0_rot_gated": "e_0_v_aux",
    "e_lambda_rot_gated": "e_lambda_v_aux",
}

DIAGNOSTIC_KEYS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    title: str
    tune_envs: tuple[str, ...]
    eval_envs: tuple[str, ...]
    is_brax_style: bool
    uses_mjx_dynamic_cache: bool
    critic_lrs: tuple[float, ...]
    critic_epochs: tuple[int, ...]
    max_grad_norms: tuple[float, ...]
    lambdas: tuple[float, ...]
    actor_lr: float
    gae_lambda: float
    stage1_seeds: int
    stage2_seeds: int
    stage2_seed_chunk: int
    stage1_master_seed: int = 42
    stage2_master_seed: int = 1000
    vf_coef: float = 1.0
    aux_vf_coef: float = 1.0
    reference_method: str = "td_lambda"


SUITES: dict[str, SuiteSpec] = {
    "brax": SuiteSpec(
        name="brax",
        title="Brax V-PPO (Held-Out Global Config)",
        tune_envs=(
            "hopper",
            "halfcheetah",
            "walker2d",
            "ant",
            "humanoid",
            "inverted_pendulum",
            "inverted_double_pendulum",
            "pusher",
            "reacher",
            "swimmer",
        ),
        eval_envs=(
            "hopper",
            "halfcheetah",
            "walker2d",
            "ant",
            "humanoid",
            "inverted_pendulum",
            "inverted_double_pendulum",
            "pusher",
            "reacher",
            "swimmer",
        ),
        is_brax_style=True,
        uses_mjx_dynamic_cache=False,
        critic_lrs=(3e-4, 1e-3, 3e-3),
        critic_epochs=(4, 8),
        max_grad_norms=(1.0,),
        lambdas=(0.4, 0.8, 0.95, 1.0),
        actor_lr=3e-4,
        gae_lambda=0.8,
        stage1_seeds=3,
        stage2_seeds=10,
        stage2_seed_chunk=5,
    ),
    "mjx_locomotion": SuiteSpec(
        name="mjx_locomotion",
        title="MJX Playground Locomotion V-PPO (Held-Out Global Config)",
        tune_envs=(
            "Go1JoystickFlatTerrain",
            "Go1Getup",
            "Go1Handstand",
            "G1JoystickFlatTerrain",
            "BerkeleyHumanoidJoystickFlatTerrain",
        ),
        eval_envs=(
            "Go1JoystickFlatTerrain",
            "Go1Getup",
            "Go1Handstand",
            "G1JoystickFlatTerrain",
            "BerkeleyHumanoidJoystickFlatTerrain",
        ),
        is_brax_style=True,
        uses_mjx_dynamic_cache=True,
        critic_lrs=(3e-4, 1e-3),
        critic_epochs=(4, 8),
        max_grad_norms=(1.0,),
        lambdas=(0.6, 0.8, 0.95, 1.0),
        actor_lr=3e-4,
        gae_lambda=0.8,
        stage1_seeds=3,
        stage2_seeds=6,
        stage2_seed_chunk=3,
    ),
    "octax": SuiteSpec(
        name="octax",
        title="Octax V-PPO (Held-Out Global Config)",
        tune_envs=(
            "Octax-pong",
            "Octax-brix",
            "Octax-tetris",
            "Octax-missile",
            "Octax-tank",
        ),
        eval_envs=(
            "Octax-pong",
            "Octax-brix",
            "Octax-tetris",
            "Octax-blinky",
            "Octax-missile",
            "Octax-ufo",
            "Octax-tank",
            "Octax-worm",
        ),
        is_brax_style=False,
        uses_mjx_dynamic_cache=False,
        critic_lrs=(1e-4, 3e-4, 1e-3),
        critic_epochs=(4, 8),
        max_grad_norms=(1.0,),
        lambdas=(0.4, 0.8, 0.95, 1.0),
        actor_lr=3e-4,
        gae_lambda=0.8,
        stage1_seeds=3,
        stage2_seeds=10,
        stage2_seed_chunk=5,
    ),
    "minatar": SuiteSpec(
        name="minatar",
        title="MinAtar V-PPO (Held-Out Global Config)",
        tune_envs=(
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ),
        eval_envs=(
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "SpaceInvaders-MinAtar",
        ),
        is_brax_style=False,
        uses_mjx_dynamic_cache=False,
        critic_lrs=(1e-4, 3e-4, 1e-3),
        critic_epochs=(4, 8),
        max_grad_norms=(1.0,),
        lambdas=(0.4, 0.8, 0.95, 1.0),
        actor_lr=3e-4,
        gae_lambda=0.8,
        stage1_seeds=3,
        stage2_seeds=10,
        stage2_seed_chunk=5,
    ),
    "gymnax": SuiteSpec(
        name="gymnax",
        title="Gymnax V-PPO Diagnostic (Held-Out Global Config)",
        tune_envs=(
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
        ),
        eval_envs=(
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
        ),
        is_brax_style=False,
        uses_mjx_dynamic_cache=False,
        critic_lrs=(1e-4, 3e-4, 1e-3),
        critic_epochs=(4, 8),
        max_grad_norms=(1.0,),
        lambdas=(0.4, 0.8, 0.95, 1.0),
        actor_lr=3e-4,
        gae_lambda=0.8,
        stage1_seeds=3,
        stage2_seeds=10,
        stage2_seed_chunk=10,
    ),
}


def get_suite_spec(suite_name: str) -> SuiteSpec:
    if suite_name not in SUITES:
        raise ValueError(f"Unknown suite '{suite_name}'. Valid suites: {sorted(SUITES.keys())}")
    return SUITES[suite_name]


def enforce_parity(cfg: dict[str, Any], vf_coef: float = 1.0, max_grad_norm: float | None = None) -> dict[str, Any]:
    """
    Enforces strict apples-to-apples critic parity across all suites:
      1. VF_COEF identical at top-level and inside NETWORK_CONFIG
      2. VF_CLIP = None (no heuristic PPO value clipping)
      3. SPLIT_AUX_OPTIMIZER = True (dedicated optimizer for aux_loss)
      4. MAX_GRAD_NORM synchronized across top-level and NETWORK_CONFIG
    """
    out = copy.deepcopy(cfg)
    out["VF_COEF"] = float(vf_coef)
    out["VF_CLIP"] = None
    out["SPLIT_AUX_OPTIMIZER"] = True
    if max_grad_norm is not None:
        out["MAX_GRAD_NORM"] = float(max_grad_norm)

    if "NETWORK_CONFIG" in out and isinstance(out["NETWORK_CONFIG"], dict):
        net_cfg = copy.deepcopy(out["NETWORK_CONFIG"])
        net_cfg["VF_COEF"] = float(vf_coef)
        if max_grad_norm is not None:
            net_cfg["MAX_GRAD_NORM"] = float(max_grad_norm)
        out["NETWORK_CONFIG"] = net_cfg
    return out


def load_suite_backend(suite_name: str) -> tuple[Callable, Callable[[str, SuiteSpec, dict[str, Any]], dict[str, Any]], Callable[[str, int | None], int]]:
    """
    Lazily imports the suite's PPO factory, base config builder, and default timestep getter.
    Lazy imports ensure MJX is not imported in Brax/Gymnax environments and vice versa.
    """
    if suite_name == "brax":
        from brax_rl.algos.ppo import make_train as make_ppo_train
        from scripts.brax.sweep_v_ppo.sweep_v_ppo import (
            build_base_config as brax_base_cfg,
            get_env_default_timesteps,
        )

        def build_cfg(env_name: str, spec: SuiteSpec, overrides: dict[str, Any]) -> dict[str, Any]:
            cfg = brax_base_cfg(
                env_name=env_name,
                actor_lr=overrides.get("actor_lr", spec.actor_lr),
                gae_lambda=overrides.get("gae_lambda", spec.gae_lambda),
                num_envs=overrides.get("num_envs", 512),
                num_steps=overrides.get("num_steps", 128),
                minibatch_size=overrides.get("minibatch_size", 1024),
                aux_vf_coef=overrides.get("aux_vf_coef", spec.aux_vf_coef),
            )
            return enforce_parity(cfg, vf_coef=spec.vf_coef)

        return make_ppo_train, build_cfg, get_env_default_timesteps

    if suite_name == "mjx_locomotion":
        from mjx_rl.algos.ppo import make_train as make_ppo_train
        from scripts.mjx.sweep_locomotion.sweep_locomotion import (
            build_base_config as mjx_base_cfg,
            get_env_default_timesteps,
        )

        def build_cfg(env_name: str, spec: SuiteSpec, overrides: dict[str, Any]) -> dict[str, Any]:
            cfg = mjx_base_cfg(
                env_name=env_name,
                impl=overrides.get("impl", "jax"),
                actor_lr=overrides.get("actor_lr", spec.actor_lr),
                gae_lambda=overrides.get("gae_lambda", spec.gae_lambda),
                num_envs=overrides.get("num_envs", 512),
                num_steps=overrides.get("num_steps", 128),
                minibatch_size=overrides.get("minibatch_size", 2048),
                actor_epochs=overrides.get("actor_epochs", 4),
                aux_vf_coef=overrides.get("aux_vf_coef", spec.aux_vf_coef),
            )
            return enforce_parity(cfg, vf_coef=spec.vf_coef)

        return make_ppo_train, build_cfg, get_env_default_timesteps

    if suite_name == "octax":
        from octax_rl.algos.ppo import make_train as make_ppo_train
        from scripts.octax.sweep_v_ppo.sweep_v_ppo import (
            build_base_config as octax_base_cfg,
            get_env_default_timesteps,
        )

        def build_cfg(env_name: str, spec: SuiteSpec, overrides: dict[str, Any]) -> dict[str, Any]:
            cfg = octax_base_cfg(
                env_name=env_name,
                actor_lr=overrides.get("actor_lr", spec.actor_lr),
                gae_lambda=overrides.get("gae_lambda", spec.gae_lambda),
                num_envs=overrides.get("num_envs", 64),
                num_steps=overrides.get("num_steps", 128),
                minibatch_size=overrides.get("minibatch_size", 1024),
                aux_vf_coef=overrides.get("aux_vf_coef", spec.aux_vf_coef),
                sticky_action_prob=overrides.get("sticky_action_prob", 0.0),
            )
            return enforce_parity(cfg, vf_coef=spec.vf_coef)

        return make_ppo_train, build_cfg, get_env_default_timesteps

    if suite_name in ("minatar", "gymnax"):
        from gymnax_rl.algos.ppo import make_train as make_ppo_train
        from scripts.gymnax.sweep_v_ppo.sweep_v_ppo import (
            build_base_config as gymnax_base_cfg,
            get_env_default_timesteps,
        )
        default_num_envs = 64 if suite_name == "minatar" else 32
        default_num_steps = 128 if suite_name == "minatar" else 256

        def build_cfg(env_name: str, spec: SuiteSpec, overrides: dict[str, Any]) -> dict[str, Any]:
            cfg = gymnax_base_cfg(
                env_name=env_name,
                actor_lr=overrides.get("actor_lr", spec.actor_lr),
                gae_lambda=overrides.get("gae_lambda", spec.gae_lambda),
                num_envs=overrides.get("num_envs", default_num_envs),
                num_steps=overrides.get("num_steps", default_num_steps),
                minibatch_size=overrides.get("minibatch_size", 512),
                aux_vf_coef=overrides.get("aux_vf_coef", spec.aux_vf_coef),
            )
            return enforce_parity(cfg, vf_coef=spec.vf_coef)

        return make_ppo_train, build_cfg, get_env_default_timesteps

    raise ValueError(f"Unsupported suite '{suite_name}'.")


def build_stage1_candidates(
    method_key: str,
    critic_lrs: list[float] | tuple[float, ...],
    critic_epochs_list: list[int] | tuple[int, ...],
    max_grad_norms: list[float] | tuple[float, ...],
    lambdas: list[float] | tuple[float, ...],
) -> list[dict[str, Any]]:
    """
    Builds the Stage 1 tuning grid:
      CRITIC_LR x CRITIC_EPOCHS x MAX_GRAD_NORM [x LAMBDA]
    Every tuned algorithm gets the exact same grid on its applicable axes.
    """
    lambda_key = _method_lambda_key(method_key)
    if lambda_key is not None:
        # At λ=1.0, E(1.0) == TD(1.0) == MC; only sweep λ=1.0 inside td_lambda (exported as `mc`)
        if method_key == "e_lambda_v_aux":
            lambda_vals = [float(x) for x in lambdas if float(x) < 1.0 - 1e-6]
        else:
            lambda_vals = [float(x) for x in lambdas]
    else:
        lambda_vals = [None]

    candidates: list[dict[str, Any]] = []
    seen = set()
    for lr in critic_lrs:
        for ep in critic_epochs_list:
            for gn in max_grad_norms:
                for lmb in lambda_vals:
                    params: dict[str, Any] = {
                        "CRITIC_LR": float(lr),
                        "CRITIC_EPOCHS": int(ep),
                        "MAX_GRAD_NORM": float(gn),
                    }
                    tag = f"clr={lr:g}_cep={ep}_gn={gn:g}"
                    if lambda_key is not None and lmb is not None:
                        params[lambda_key] = float(lmb)
                        tag += f"_lam={lmb:g}"
                    key = tuple(sorted(params.items()))
                    if key not in seen:
                        seen.add(key)
                        candidates.append({"tag": tag, "params": params})
    return candidates


def apply_pipeline_params(
    base_cfg: dict[str, Any],
    method_key: str,
    params: dict[str, Any],
    spec: SuiteSpec,
) -> dict[str, Any]:
    cfg = _apply_params(
        base_cfg=base_cfg,
        method_key=method_key,
        params=params,
        is_brax=spec.is_brax_style,
        is_v_ppo=True,
    )
    return enforce_parity(cfg, vf_coef=spec.vf_coef, max_grad_norm=params.get("MAX_GRAD_NORM"))


def run_candidate_chunked(
    make_train_fn: Callable,
    cfg: dict[str, Any],
    seed_keys: jax.Array,
    metric_key: str,
    seed_chunk_size: int,
    uses_mjx_dynamic_cache: bool = False,
    compiled_cache: dict[tuple, Any] | None = None,
) -> dict[str, Any]:
    """
    Runs a candidate across `seed_keys`, chunking the vmap over `seed_chunk_size`
    so Stage 2 can safely run 10+ held-out seeds without GPU OOM.
    """
    n_seeds = int(seed_keys.shape[0])
    chunk = max(1, min(int(seed_chunk_size), n_seeds))

    if uses_mjx_dynamic_cache:
        from mjx_rl.algos.ppo import extract_hparams, static_compile_signature

        sig = static_compile_signature(cfg)
        if compiled_cache is None:
            compiled_cache = {}
        if sig not in compiled_cache:
            train_fn = make_train_fn(cfg, dynamic_hparams=True)
            compiled_cache[sig] = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))
        vmapped = compiled_cache[sig]
        hparams = extract_hparams(cfg)
    else:
        train_fn = make_train_fn(cfg)
        vmapped = jax.jit(jax.vmap(train_fn))
        hparams = None

    ret_chunks: list[np.ndarray] = []
    extra_chunks: dict[str, list[np.ndarray]] = {}
    t0 = time.time()

    for start in range(0, n_seeds, chunk):
        sub_keys = seed_keys[start : start + chunk]
        out = vmapped(sub_keys, hparams) if uses_mjx_dynamic_cache else vmapped(sub_keys)
        metrics = out["metrics"]
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            metrics[metric_key],
        )
        ret = np.asarray(metrics[metric_key])
        if ret.ndim == 1:
            ret = ret[None, :]
        ret_chunks.append(ret)

        for extra_key in DIAGNOSTIC_KEYS:
            if extra_key in metrics:
                arr = np.asarray(metrics[extra_key])
                if arr.ndim == 1:
                    arr = arr[None, :]
                extra_chunks.setdefault(extra_key, []).append(arr)

    elapsed = time.time() - t0
    returns = np.concatenate(ret_chunks, axis=0)
    seed_aucs = returns.mean(axis=-1)
    seed_finals = returns[:, -1]

    result: dict[str, Any] = {
        "returned_episode_returns": returns,
        "auc_mean": float(seed_aucs.mean()),
        "auc_std": float(seed_aucs.std()),
        "final_mean": float(seed_finals.mean()),
        "final_std": float(seed_finals.std()),
        "elapsed_time": float(elapsed),
        "n_seeds": int(n_seeds),
    }
    for extra_key, chunks in extra_chunks.items():
        result[extra_key] = np.concatenate(chunks, axis=0)

    # Prevent accumulated XLA executable bloat across 36-48 sequential candidates on 24GB GPUs
    if not uses_mjx_dynamic_cache:
        del vmapped
        jax.clear_caches()
    return result


def tuned_config_path(configs_dir: str, suite_name: str, algo_key: str, lambda_val: float | None = None) -> str:
    if lambda_val is None:
        fname = f"{algo_key}.json"
    else:
        fname = f"{algo_key}_lam{float(lambda_val):g}.json"
    return os.path.join(configs_dir, suite_name, fname)


def load_tuned_config(configs_dir: str, suite_name: str, algo_key: str, lambda_val: float | None = None) -> dict[str, Any]:
    """
    Loads a Stage 1 tuned JSON config. For inherited ablations, resolves to the
    parent base algorithm's config and rewrites the lambda key if needed.
    Fails fast if the required JSON does not exist.
    """
    source_algo = ABLATION_INHERITANCE.get(algo_key, algo_key)
    path = tuned_config_path(configs_dir, suite_name, source_algo, lambda_val=lambda_val)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing tuned config for suite='{suite_name}', algo='{algo_key}' "
            f"(resolved from '{source_algo}', lambda={lambda_val}): {path}. "
            f"Run Stage 1 + select_configs.py first."
        )
    with open(path, "r") as f:
        payload = json.load(f)

    params = dict(payload["params"])
    src_lkey = _method_lambda_key(source_algo)
    dst_lkey = _method_lambda_key(algo_key)
    if src_lkey != dst_lkey:
        if src_lkey is not None and src_lkey in params:
            lval = params.pop(src_lkey)
            if dst_lkey is not None:
                params[dst_lkey] = lval
        elif dst_lkey is not None and lambda_val is not None:
            params[dst_lkey] = float(lambda_val)

    return {
        "algo": algo_key,
        "inherited_from": source_algo,
        "source_path": path,
        "params": params,
        "metadata": payload.get("metadata", {}),
    }

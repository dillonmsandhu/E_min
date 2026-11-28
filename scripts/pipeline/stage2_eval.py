#!/usr/bin/env python3
"""
scripts/pipeline/stage2_eval.py

Stage 2: Held-out comparative evaluation using ONLY the global suite-wide configs
exported by Stage 1 (`configs/tuned/<suite>/<algo>.json`). ZERO per-env tuning.

Evaluates:
  1. Global Championship: 1 fixed suite-wide config per algorithm
     (e.g., td_0, e_0_v_aux, td_lambda, e_lambda_v_aux + optional inherited ablations)
  2. Per-λ Head-to-Head: 1 fixed suite-wide config per (algo, λ) for td_lambda & e_lambda
     (deduplicated automatically if identical to the global winner!)

Runs on disjoint held-out seeds (default master_seed=1000, n_seeds=10 for Brax/Octax/MinAtar/Gymnax,
5 for MJX Locomotion) with chunked vmap (`--seed_chunk_size`) to prevent GPU OOM.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, ".."))
sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (
        os.path.exists(os.path.join(p, "fixed_policy"))
        or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy")))
    )
]

import jax

from scripts.pipeline.suite_registry import (
    SUITES,
    TUNED_BASE_ALGOS,
    apply_pipeline_params,
    get_suite_spec,
    load_suite_backend,
    load_tuned_config,
    run_candidate_chunked,
)
from scripts.sweep_standard_runner import _method_lambda_key


def _params_signature(algo: str, params: dict[str, Any]) -> tuple:
    return (algo, tuple(sorted((k, float(v) if isinstance(v, (int, float)) else v) for k, v in params.items())))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2: Held-Out Comparative Evaluation (Zero Per-Env Tuning)")
    p.add_argument("--suite", type=str, required=True, choices=sorted(SUITES.keys()))
    p.add_argument("--envs", nargs="+", default=None, help="Subset of envs (defaults to spec.eval_envs)")
    p.add_argument(
        "--algos",
        nargs="+",
        default=None,
        help="Algorithms to evaluate (base + inherited ablations, e.g. td_0 e_0_v_aux td_lambda e_lambda_v_aux bsmc_v_aux e_0_rot_gated e_lambda_rot_gated)",
    )
    p.add_argument("--lambdas", nargs="+", type=float, default=None, help="Per-λ configs to also evaluate")
    p.add_argument("--no_per_lambda", action="store_true", help="Skip per-λ evaluation runs")
    p.add_argument("--actor_lr", type=float, default=None)
    p.add_argument("--gae_lambda", type=float, default=None)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=None)
    p.add_argument("--seed_chunk_size", type=int, default=None)
    p.add_argument("--master_seed", type=int, default=None)
    p.add_argument("--metric_key", type=str, default="returned_episode_returns")
    p.add_argument("--configs_dir", type=str, default="configs/tuned")
    p.add_argument("--out_root", type=str, default="results/pipeline")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_suite_spec(args.suite)
    make_train_fn, build_cfg_fn, get_ts_fn = load_suite_backend(args.suite)

    envs = list(args.envs) if args.envs else list(spec.eval_envs)
    algos = list(args.algos) if args.algos else list(TUNED_BASE_ALGOS)
    lambdas = list(args.lambdas) if args.lambdas else list(spec.lambdas)

    n_seeds = int(args.n_seeds) if args.n_seeds is not None else spec.stage2_seeds
    seed_chunk = int(args.seed_chunk_size) if args.seed_chunk_size is not None else spec.stage2_seed_chunk
    master_seed = int(args.master_seed) if args.master_seed is not None else spec.stage2_master_seed
    seed_keys = jax.random.split(jax.random.PRNGKey(master_seed), n_seeds)

    # Pre-load all required JSON configs up front (fail fast before any GPU work!)
    global_cfgs: dict[str, dict[str, Any]] = {}
    per_lambda_cfgs: dict[str, dict[float, dict[str, Any]]] = {}
    for algo in algos:
        global_cfgs[algo] = load_tuned_config(args.configs_dir, args.suite, algo, lambda_val=None)
        if (not args.no_per_lambda) and _method_lambda_key(algo) is not None:
            per_lambda_cfgs[algo] = {}
            for lmb in lambdas:
                if float(lmb) >= 1.0 - 1e-6:
                    continue
                per_lambda_cfgs[algo][float(lmb)] = load_tuned_config(
                    args.configs_dir, args.suite, algo, lambda_val=lmb
                )

    overrides: dict[str, Any] = {}
    if args.actor_lr is not None:
        overrides["actor_lr"] = float(args.actor_lr)
    if args.gae_lambda is not None:
        overrides["gae_lambda"] = float(args.gae_lambda)

    for env in envs:
        env_ts = int(get_ts_fn(env, args.total_timesteps))
        base_cfg = build_cfg_fn(env, spec, overrides)
        base_cfg["TOTAL_TIMESTEPS"] = env_ts
        bs = int(base_cfg["NUM_ENVS"]) * int(base_cfg["NUM_STEPS"])
        compiled_cache: dict = {}

        global_dir = os.path.join(args.out_root, args.suite, "stage2_eval", "global", env, "data")
        per_lam_dir = os.path.join(args.out_root, args.suite, "stage2_eval", "per_lambda", env, "data")
        os.makedirs(global_dir, exist_ok=True)
        os.makedirs(per_lam_dir, exist_ok=True)
        global_pkl = os.path.join(global_dir, "sweep_metrics_all.pkl")
        per_lam_pkl = os.path.join(per_lam_dir, "sweep_metrics_all.pkl")

        if (not args.overwrite) and os.path.exists(global_pkl):
            with open(global_pkl, "rb") as f:
                global_runs = pickle.load(f).get("runs", {})
        else:
            global_runs = {}

        if (not args.overwrite) and os.path.exists(per_lam_pkl):
            with open(per_lam_pkl, "rb") as f:
                per_lam_runs = pickle.load(f).get("runs", {})
        else:
            per_lam_runs = {}

        global_runs.setdefault(env, {})
        per_lam_runs.setdefault(env, {})

        # Deduplication cache within this env: (algo, params_tuple) -> result dict
        eval_cache: dict[tuple, dict[str, Any]] = {}
        for m_k, t_dict in global_runs[env].items():
            for res in t_dict.values():
                if "params" in res:
                    eval_cache[_params_signature(m_k, res["params"])] = res
        for m_k, t_dict in per_lam_runs[env].items():
            for res in t_dict.values():
                if "params" in res:
                    eval_cache[_params_signature(m_k, res["params"])] = res

        print(
            f"\n[Stage 2 | {args.suite}] env={env} "
            f"({len(algos)} algos x {n_seeds} held-out seeds [chunk={seed_chunk}], ts={env_ts:,}, bs={bs})"
        )

        def _get_or_run(algo: str, loaded: dict[str, Any], label: str) -> dict[str, Any]:
            params = loaded["params"]
            sig = _params_signature(algo, params)
            if (not args.overwrite) and sig in eval_cache:
                res = eval_cache[sig]
                print(
                    f"  [{algo:18s} | {label:10s}] [CACHED/DEDUP] "
                    f"AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} | Fin={res['final_mean']:8.2f}"
                )
                return res

            cfg = apply_pipeline_params(base_cfg, algo, params, spec)
            res = run_candidate_chunked(
                make_train_fn=make_train_fn,
                cfg=cfg,
                seed_keys=seed_keys,
                metric_key=args.metric_key,
                seed_chunk_size=seed_chunk,
                uses_mjx_dynamic_cache=spec.uses_mjx_dynamic_cache,
                compiled_cache=compiled_cache,
            )
            res["params"] = params
            res["config"] = cfg
            res["inherited_from"] = loaded["inherited_from"]
            res["source_config_path"] = loaded["source_path"]
            eval_cache[sig] = res
            print(
                f"  [{algo:18s} | {label:10s}] "
                f"AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} "
                f"| Fin={res['final_mean']:8.2f} ({res['elapsed_time']:.1f}s)"
            )
            return res

        # 1. Global suite-wide config runs (strictly 1 tag = "global" per algo -> zero per-env tuning)
        for algo in algos:
            global_runs[env].setdefault(algo, {})
            res = _get_or_run(algo, global_cfgs[algo], label="global")
            global_runs[env][algo]["global"] = res
            with open(global_pkl, "wb") as f:
                pickle.dump(
                    {
                        "suite": args.suite,
                        "stage": "stage2_eval_global",
                        "runs": global_runs,
                        "timesteps": {env: env_ts},
                        "batch_sizes": {env: bs},
                        "method_keys": algos,
                        "master_seed": master_seed,
                        "n_seeds": n_seeds,
                    },
                    f,
                )

        # 2. Per-λ suite-wide config runs (strictly 1 tag per λ -> zero per-env tuning)
        if not args.no_per_lambda:
            for algo, lam_map in per_lambda_cfgs.items():
                per_lam_runs[env].setdefault(algo, {})
                for lmb, loaded in lam_map.items():
                    tag = f"lam={lmb:g}"
                    res = _get_or_run(algo, loaded, label=tag)
                    per_lam_runs[env][algo][tag] = res
                with open(per_lam_pkl, "wb") as f:
                    pickle.dump(
                        {
                            "suite": args.suite,
                            "stage": "stage2_eval_per_lambda",
                            "runs": per_lam_runs,
                            "timesteps": {env: env_ts},
                            "batch_sizes": {env: bs},
                            "method_keys": list(per_lambda_cfgs.keys()),
                            "master_seed": master_seed,
                            "n_seeds": n_seeds,
                        },
                        f,
                    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/pipeline/stage1_tune.py

Stage 1: Hyperparameter tuning sweep for a single (suite, env, algo) or subset.
Sweeps CRITIC_LR x CRITIC_EPOCHS x MAX_GRAD_NORM [x LAMBDA] on tuning seeds
(default master_seed=42, n_seeds=3) with strict critic parity:
  - VF_COEF = 1.0
  - VF_CLIP = None
  - SPLIT_AUX_OPTIMIZER = True

Saves incremental pickle checkpoints to:
  <out_root>/<suite>/stage1_tune/<env>/<algo>/sweep_metrics_all.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

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
    build_stage1_candidates,
    get_suite_spec,
    load_suite_backend,
    run_candidate_chunked,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1: Suite-wide Hyperparameter Tuning Runner")
    p.add_argument("--suite", type=str, required=True, choices=sorted(SUITES.keys()))
    p.add_argument("--envs", nargs="+", default=None, help="Subset of envs (defaults to spec.tune_envs)")
    p.add_argument("--algos", nargs="+", default=None, help="Algos to tune (defaults to TUNED_BASE_ALGOS)")
    p.add_argument("--critic_lrs", nargs="+", type=float, default=None)
    p.add_argument("--critic_epochs", nargs="+", type=int, default=None)
    p.add_argument("--max_grad_norms", nargs="+", type=float, default=None)
    p.add_argument("--lambdas", nargs="+", type=float, default=None)
    p.add_argument("--actor_lr", type=float, default=None)
    p.add_argument("--gae_lambda", type=float, default=None)
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=None)
    p.add_argument("--seed_chunk_size", type=int, default=None)
    p.add_argument("--master_seed", type=int, default=None)
    p.add_argument("--metric_key", type=str, default="returned_episode_returns")
    p.add_argument("--out_root", type=str, default="results/pipeline")
    p.add_argument("--overwrite", action="store_true", help="Ignore existing cached candidate runs")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_suite_spec(args.suite)
    make_train_fn, build_cfg_fn, get_ts_fn = load_suite_backend(args.suite)

    envs = list(args.envs) if args.envs else list(spec.tune_envs)
    algos = list(args.algos) if args.algos else list(TUNED_BASE_ALGOS)
    critic_lrs = list(args.critic_lrs) if args.critic_lrs else list(spec.critic_lrs)
    critic_epochs = list(args.critic_epochs) if args.critic_epochs else list(spec.critic_epochs)
    max_grad_norms = list(args.max_grad_norms) if args.max_grad_norms else list(spec.max_grad_norms)
    lambdas = list(args.lambdas) if args.lambdas else list(spec.lambdas)

    n_seeds = int(args.n_seeds) if args.n_seeds is not None else spec.stage1_seeds
    seed_chunk = int(args.seed_chunk_size) if args.seed_chunk_size is not None else n_seeds
    master_seed = int(args.master_seed) if args.master_seed is not None else spec.stage1_master_seed
    seed_keys = jax.random.split(jax.random.PRNGKey(master_seed), n_seeds)

    overrides: dict = {}
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

        for algo in algos:
            out_dir = os.path.join(args.out_root, args.suite, "stage1_tune", env, algo)
            os.makedirs(out_dir, exist_ok=True)
            pkl_path = os.path.join(out_dir, "sweep_metrics_all.pkl")

            if (not args.overwrite) and os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    payload = pickle.load(f)
                all_runs = payload.get("runs", {})
            else:
                all_runs = {}

            all_runs.setdefault(env, {})
            all_runs[env].setdefault(algo, {})

            cands = build_stage1_candidates(
                method_key=algo,
                critic_lrs=critic_lrs,
                critic_epochs_list=critic_epochs,
                max_grad_norms=max_grad_norms,
                lambdas=lambdas,
            )
            print(
                f"\n[Stage 1 | {args.suite}] env={env} algo={algo} "
                f"({len(cands)} configs x {n_seeds} seeds, ts={env_ts:,}, bs={bs})"
            )

            for cand in cands:
                tag = cand["tag"]
                if (not args.overwrite) and tag in all_runs[env][algo]:
                    res = all_runs[env][algo][tag]
                    print(
                        f"  {tag:36s} | [CACHED] AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} "
                        f"| Fin={res['final_mean']:8.2f}"
                    )
                    continue

                cfg = apply_pipeline_params(base_cfg, algo, cand["params"], spec)
                res = run_candidate_chunked(
                    make_train_fn=make_train_fn,
                    cfg=cfg,
                    seed_keys=seed_keys,
                    metric_key=args.metric_key,
                    seed_chunk_size=seed_chunk,
                    uses_mjx_dynamic_cache=spec.uses_mjx_dynamic_cache,
                    compiled_cache=compiled_cache,
                )
                res["params"] = cand["params"]
                res["config"] = cfg
                all_runs[env][algo][tag] = res
                print(
                    f"  {tag:36s} | AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} "
                    f"| Fin={res['final_mean']:8.2f} ({res['elapsed_time']:.1f}s)"
                )

                with open(pkl_path, "wb") as f:
                    pickle.dump(
                        {
                            "suite": args.suite,
                            "stage": "stage1_tune",
                            "runs": all_runs,
                            "timesteps": {env: env_ts},
                            "batch_sizes": {env: bs},
                            "method_keys": [algo],
                            "master_seed": master_seed,
                            "n_seeds": n_seeds,
                        },
                        f,
                    )


if __name__ == "__main__":
    main()

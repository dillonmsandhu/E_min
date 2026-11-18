#!/usr/bin/env python3
"""
scripts/mjx/tune_base_td/tune_base_td.py

Phase 1 base-config verification and hyperparameter tuning for TD(λ) on MuJoCo Playground
(5 diverse DM Control Suite tasks: CheetahRun, WalkerWalk, HopperHop, CartpoleSwingup, FingerSpin).

Always runs the exact Brax PPO parity baseline first ('brax_parity_default'), followed by
one-at-a-time (OAT) or full-grid variations over:
  - Rollout geometry & minibatch size: (num_envs, num_steps, num_minibatches, epochs)
  - Actor/Critic base learning rate & schedule
  - Clipping epsilon & entropy coefficient
  - GAE / TD lambda
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import pickle
import sys
from typing import Any

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from scripts.sweep_standard_runner import run_candidate

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def build_base_td_candidates(mode: str = "oat") -> list[dict[str, Any]]:
    """
    Builds candidate configurations around the Brax PPO DMC parity baseline:
      num_envs=2048, num_steps=30, num_minibatches=32 (mb=1920), epochs=16,
      lr=1e-3 (constant), clip_eps=0.3, ent_coef=0.01, gae_lambda=0.95.
    """
    candidates: list[dict[str, Any]] = []

    def _add(tag: str, overrides: dict[str, Any]):
        candidates.append({"tag": tag, "overrides": overrides})

    # 1. Exact Brax PPO DMC parity configuration
    _add("00_brax_parity_default", {})

    if mode == "parity_only":
        return candidates

    # 2. Rollout horizon & batch geometry variations
    _add(
        "01_steps60_ep8",
        {
            "NUM_ENVS": 2048,
            "NUM_STEPS": 60,
            "ACTOR_EPOCHS": 8,
            "CRITIC_EPOCHS": 8,
            "ACTOR_MINIBATCH_SIZE": 3840,
            "CRITIC_MINIBATCH_SIZE": 3840,
        },
    )
    _add(
        "02_envs1024_steps60_ep8",
        {
            "NUM_ENVS": 1024,
            "NUM_STEPS": 60,
            "ACTOR_EPOCHS": 8,
            "CRITIC_EPOCHS": 8,
            "ACTOR_MINIBATCH_SIZE": 1920,
            "CRITIC_MINIBATCH_SIZE": 1920,
        },
    )
    _add(
        "03_ep8",
        {
            "ACTOR_EPOCHS": 8,
            "CRITIC_EPOCHS": 8,
        },
    )

    # 3. Learning rate & schedule variations
    _add("04_lr3e-4_const", {"ACTOR_LR": 3e-4, "CRITIC_LR": 3e-4})
    _add(
        "05_lr1e-3_linear",
        {"ACTOR_LR": 1e-3, "CRITIC_LR": 1e-3, "LR_SCHEDULE": "linear"},
    )

    # 4. Clipping & entropy variations
    _add("06_clip0.2", {"CLIP_EPS": 0.2})
    _add("07_ent1e-3", {"ENT_COEF": 1e-3})

    # 5. GAE / Value lambda variations
    _add(
        "08_lam0.8",
        {"GAE_LAMBDA": 0.8, "ACTOR_GAE_LAMBDA": 0.8, "VALUE_LAMBDA": 0.8},
    )
    return candidates


def apply_candidate_overrides(env_name: str, impl: str, overrides: dict[str, Any], total_timesteps: int | None) -> dict[str, Any]:
    cfg = get_dmc_brax_parity_config(env_name, impl=impl)
    cfg["CRITIC_TYPE"] = "fitted"
    if total_timesteps is not None:
        cfg["TOTAL_TIMESTEPS"] = int(total_timesteps)

    # Ensure default minibatch size matches 2048 * 30 // 32 = 1920
    default_mb = (int(cfg["NUM_ENVS"]) * int(cfg["NUM_STEPS"])) // int(cfg["NETWORK_CONFIG"]["NUM_MINIBATCHES"])
    cfg["MINIBATCH_SIZE"] = default_mb
    cfg["ACTOR_MINIBATCH_SIZE"] = default_mb
    cfg["CRITIC_MINIBATCH_SIZE"] = default_mb

    net_cfg = copy.deepcopy(cfg["NETWORK_CONFIG"])
    for k, v in overrides.items():
        if k == "LR_SCHEDULE":
            net_cfg["LR_SCHEDULE"] = v
        else:
            cfg[k] = v

    net_cfg["LR"] = float(cfg["ACTOR_LR"])
    net_cfg["NUM_EPOCHS"] = int(cfg["ACTOR_EPOCHS"])
    net_cfg["MINIBATCH_SIZE"] = int(cfg["ACTOR_MINIBATCH_SIZE"])
    cfg["NETWORK_CONFIG"] = net_cfg
    return cfg


def plot_and_summarize(all_runs: dict[str, dict[str, Any]], timesteps: dict[str, int], out_dir: str) -> None:
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    rows = []
    envs = sorted(all_runs.keys())
    for env in envs:
        env_runs = all_runs[env]
        fig, ax = plt.subplots(figsize=(8, 5))
        ts = timesteps.get(env, 60_000_000)
        for tag, res in env_runs.items():
            rets = np.asarray(res["returned_episode_returns"])
            n_seeds, n_updates = rets.shape
            xs = np.linspace(0, ts / 1e6, n_updates)
            mean = rets.mean(axis=0)
            sem = rets.std(axis=0) / max(1.0, np.sqrt(n_seeds))
            lw = 2.4 if "parity_default" in tag else 1.5
            ax.plot(xs, mean, label=f"{tag} (AUC {res['auc_mean']:.0f})", linewidth=lw)
            ax.fill_between(xs, mean - sem, mean + sem, alpha=0.15)
            rows.append(
                {
                    "env": env,
                    "tag": tag,
                    "auc_mean": res["auc_mean"],
                    "auc_std": res["auc_std"],
                    "auc_sem": res["auc_std"] / max(1.0, np.sqrt(n_seeds)),
                    "final_mean": res["final_mean"],
                    "final_std": res["final_std"],
                    "final_sem": res["final_std"] / max(1.0, np.sqrt(n_seeds)),
                    "elapsed_s": res["elapsed_time"],
                }
            )
        ax.set_title(f"{env} — Base TD(λ) Config Tuning")
        ax.set_xlabel("Environment Steps (Millions)")
        ax.set_ylabel("Episodic Return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"{env}_base_td_tuning.pdf"))
        plt.close(fig)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(tables_dir, "summary_returned_episode_returns.csv")
        df.to_csv(csv_path, index=False)

        pivot_auc = df.pivot(index="tag", columns="env", values="auc_mean")
        pivot_auc["Mean_AUC"] = pivot_auc.mean(axis=1)
        pivot_auc = pivot_auc.sort_values("Mean_AUC", ascending=False)
        md_path = os.path.join(tables_dir, "base_td_ranking.md")
        with open(md_path, "w") as f:
            f.write("### MJX Playground Base TD(λ) Config Ranking (AUC Return)\n\n")
            f.write(pivot_auc.to_markdown(floatfmt=".1f"))
            f.write("\n")


def consolidate_base_td(out_dir: str) -> None:
    pkls = sorted(glob.glob(os.path.join(out_dir, "*", "data", "base_td_metrics.pkl")))
    if not pkls:
        single = os.path.join(out_dir, "data", "base_td_metrics.pkl")
        if os.path.exists(single):
            pkls = [single]
    if not pkls:
        print(f"No base_td_metrics.pkl found under {out_dir}")
        return

    all_runs: dict[str, dict[str, Any]] = {}
    timesteps: dict[str, int] = {}
    for p in pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        for env, runs in d.get("runs", {}).items():
            all_runs[env] = runs
        timesteps.update(d.get("timesteps", {}))

    cons_dir = os.path.join(out_dir, "data")
    os.makedirs(cons_dir, exist_ok=True)
    with open(os.path.join(cons_dir, "base_td_metrics.pkl"), "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps}, f)

    plot_and_summarize(all_runs, timesteps, out_dir)
    print(f"Consolidated {len(all_runs)} environments into {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tune / verify base TD(λ) config on MJX Playground DMC tasks")
    p.add_argument("--envs", nargs="+", default=DEFAULT_DMC_5_ENVS)
    p.add_argument("--env_name", type=str, default=None, help="Single env override (writes to out_dir/<env_name>)")
    p.add_argument("--impl", type=str, default="jax", choices=["jax", "warp"])
    p.add_argument("--mode", type=str, default="oat", choices=["parity_only", "oat"])
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/mjx/tune_base_td")
    p.add_argument("--consolidate_only", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()
    if args.consolidate_only:
        consolidate_base_td(args.out_dir)
        return

    envs = [args.env_name] if args.env_name else list(args.envs)
    target_out = os.path.join(args.out_dir, args.env_name) if args.env_name else args.out_dir
    data_dir = os.path.join(target_out, "data")
    os.makedirs(data_dir, exist_ok=True)

    candidates = build_base_td_candidates(args.mode)
    seed_keys = jax.random.split(jax.random.PRNGKey(args.master_seed), args.n_seeds)

    all_runs: dict[str, dict[str, Any]] = {}
    timesteps: dict[str, int] = {}

    for env in envs:
        all_runs[env] = {}
        print(f"\n=== Base TD(λ) Tuning: {env} ({len(candidates)} candidates x {args.n_seeds} seeds) ===")
        for cand in candidates:
            tag = cand["tag"]
            cfg = apply_candidate_overrides(env, args.impl, cand["overrides"], args.total_timesteps)
            timesteps[env] = int(cfg["TOTAL_TIMESTEPS"])
            res = run_candidate(make_ppo_train, cfg, seed_keys, metric_key="returned_episode_returns")
            res["overrides"] = cand["overrides"]
            res["config"] = cfg
            all_runs[env][tag] = res
            print(
                f"  {tag:28s} | AUC={res['auc_mean']:8.2f} ± {res['auc_std']:6.2f} "
                f"| Fin={res['final_mean']:8.2f} ({res['elapsed_time']:.1f}s)"
            )
            with open(os.path.join(data_dir, "base_td_metrics.pkl"), "wb") as f:
                pickle.dump({"runs": all_runs, "timesteps": timesteps}, f)

    plot_and_summarize(all_runs, timesteps, target_out)


if __name__ == "__main__":
    main()

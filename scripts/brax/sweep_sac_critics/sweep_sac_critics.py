#!/usr/bin/env python3
"""
sweep_sac_critics.py

Compares four critic algorithms in Brax SAC (brax_rl/algos/sac.py):
  1. td_lambda : TD(lambda)
  2. e_0       : E(0)
  3. td_0      : Online TD(0)
  4. e_lambda  : E(lambda) with v_aux

Supports:
  - `default` mode: head-to-head comparison using brax_rl/core/sac_config.py defaults
  - `tune_lr` mode: sweeps CRITIC_LR per method
  - `tune_full` mode: sweeps CRITIC_LR x LAMBDA (for td_lambda and e_lambda)

Outputs under <out_dir>/:
  - data/sweep_metrics_all.pkl
  - data/summary_returns.csv
  - data/best_configs.json
  - plots/learning_curves_<env>.pdf
  - tables/summary_table.md, .tex
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["axes.edgecolor"] = "#333333"
matplotlib.rcParams["axes.linewidth"] = 0.8

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (
        os.path.exists(os.path.join(p, "fixed_policy"))
        or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy")))
    )
]

import jax

from brax_rl.algos.sac import make_train
from brax_rl.core.sac_config import config as brax_sac_config

METHODS = [
    ("td_lambda", "TD(λ)"),
    ("e_0", "E(0)"),
    ("td_0", "Online TD(0)"),
    ("e_lambda", "E(λ) (v_aux)"),
]

METHOD_COLORS = {
    "td_lambda": "#2563eb",
    "e_0": "#059669",
    "td_0": "#d97706",
    "e_lambda": "#7c3aed",
}

ENV_TITLES = {
    "hopper": "Hopper",
    "halfcheetah": "HalfCheetah",
    "walker2d": "Walker2d",
    "ant": "Ant",
    "humanoid": "Humanoid",
    "humanoidstandup": "HumanoidStandup",
    "inverted_pendulum": "Inverted Pendulum",
    "inverted_double_pendulum": "Inverted Double Pendulum",
    "pusher": "Pusher",
    "reacher": "Reacher",
    "swimmer": "Swimmer",
}

DEFAULT_ENVS = [
    "hopper",
    "halfcheetah",
    "walker2d",
    "ant",
    "humanoid",
    "humanoidstandup",
    "inverted_pendulum",
    "inverted_double_pendulum",
    "pusher",
    "reacher",
    "swimmer",
]

DEFAULT_CRITIC_LRS = [1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_LAMBDAS = [0.4, 0.8, 0.95]


def get_env_default_timesteps(env_name: str, fallback_timesteps: int | None = None) -> int:
    if fallback_timesteps is not None:
        return int(fallback_timesteps)
    if env_name in ("inverted_pendulum", "inverted_double_pendulum"):
        return 2_000_000
    if env_name in ("reacher", "pusher"):
        return 5_000_000
    if env_name in ("humanoid", "humanoidstandup"):
        return 20_000_000
    if env_name in ("hopper", "walker2d", "halfcheetah", "ant", "swimmer"):
        return 10_000_000
    return 5_000_000


def build_method_config(
    method_key: str,
    env_name: str,
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
    minibatch_size: int,
    critic_lr: float | None = None,
    lam: float | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(brax_sac_config)
    cfg["ENV_NAME"] = env_name
    cfg["TOTAL_TIMESTEPS"] = int(total_timesteps)
    cfg["NUM_ENVS"] = int(num_envs)
    cfg["NUM_STEPS"] = int(num_steps)
    cfg["ACTOR_MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["CRITIC_MINIBATCH_SIZE"] = int(minibatch_size)

    if method_key == "td_lambda":
        cfg["CRITIC_TYPE"] = "td_lambda"
        if lam is not None:
            cfg["TD_LAMBDA"] = float(lam)
    elif method_key == "e_0":
        cfg["CRITIC_TYPE"] = "e_0"
    elif method_key == "td_0":
        cfg["CRITIC_TYPE"] = "td_0"
    elif method_key == "e_lambda":
        cfg["CRITIC_TYPE"] = "e_lambda"
        if lam is not None:
            cfg["E_LAMBDA"] = float(lam)
    else:
        raise ValueError(f"Unknown method_key '{method_key}'")

    if critic_lr is not None:
        cfg["CRITIC_LR"] = float(critic_lr)

    return cfg


def build_candidates(
    mode: str,
    critic_lrs: list[float],
    lambdas: list[float],
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {m: [] for m, _ in METHODS}

    if mode == "default":
        for m, _ in METHODS:
            candidates[m].append({"tag": "default", "critic_lr": None, "lambda": None})
        return candidates

    if mode == "tune_lr":
        for lr in critic_lrs:
            for m, _ in METHODS:
                candidates[m].append({
                    "tag": f"clr={lr:g}",
                    "critic_lr": float(lr),
                    "lambda": None,
                })
        return candidates

    if mode == "tune_full":
        for lr in critic_lrs:
            for m in ("e_0", "td_0"):
                candidates[m].append({
                    "tag": f"clr={lr:g}",
                    "critic_lr": float(lr),
                    "lambda": None,
                })
            for lam in lambdas:
                for m in ("td_lambda", "e_lambda"):
                    candidates[m].append({
                        "tag": f"clr={lr:g}_lam={lam:g}",
                        "critic_lr": float(lr),
                        "lambda": float(lam),
                    })
        return candidates

    raise ValueError(f"Unknown mode '{mode}'")


def run_condition(
    cfg: dict[str, Any],
    n_seeds: int,
    master_seed: int,
) -> dict[str, Any]:
    train_fn = jax.jit(jax.vmap(make_train(cfg)))
    rng = jax.random.PRNGKey(master_seed)
    train_keys = jax.random.split(rng, n_seeds)

    t0 = time.time()
    out = train_fn(train_keys)
    jax.block_until_ready(out["metrics"])
    elapsed = time.time() - t0

    # returned_episode_returns shape: (n_seeds, num_updates)
    curves = np.asarray(out["metrics"]["returned_episode_returns"], dtype=np.float64)
    window = max(1, curves.shape[1] // 10)
    final_seed_means = np.mean(curves[:, -window:], axis=1)
    auc_seed_means = np.mean(curves, axis=1)

    return {
        "curves": curves,
        "final_mean": float(np.mean(final_seed_means)),
        "final_sem": float(np.std(final_seed_means, ddof=0) / np.sqrt(n_seeds)),
        "auc_mean": float(np.mean(auc_seed_means)),
        "auc_sem": float(np.std(auc_seed_means, ddof=0) / np.sqrt(n_seeds)),
        "elapsed_s": float(elapsed),
        "params": {
            "CRITIC_TYPE": cfg["CRITIC_TYPE"],
            "CRITIC_LR": cfg["CRITIC_LR"],
            "TD_LAMBDA": cfg["TD_LAMBDA"],
            "E_LAMBDA": cfg["E_LAMBDA"],
        },
    }


def score_candidates(env_runs: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    ranked = sorted(
        env_runs.items(),
        key=lambda kv: (kv[1]["final_mean"], kv[1]["auc_mean"]),
        reverse=True,
    )
    return [(tag, res["final_mean"]) for tag, res in ranked]


def generate_outputs(
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]],
    timesteps: dict[str, int],
    out_dir: str,
) -> None:
    data_dir = os.path.join(out_dir, "data")
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    summary_rows = []
    best_configs: dict[str, dict[str, Any]] = {}

    envs = sorted(runs.keys())
    for env_name in envs:
        best_configs[env_name] = {}
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        total_steps = timesteps[env_name]
        title = ENV_TITLES[env_name] if env_name in ENV_TITLES else env_name

        for method_key, method_label in METHODS:
            cand_dict = runs[env_name].get(method_key, {})
            if not cand_dict:
                continue
            ranked = score_candidates(cand_dict)
            best_tag = ranked[0][0]
            best_res = cand_dict[best_tag]
            best_configs[env_name][method_key] = {
                "tag": best_tag,
                "final_mean": best_res["final_mean"],
                "final_sem": best_res["final_sem"],
                "auc_mean": best_res["auc_mean"],
                "params": best_res["params"],
            }

            for tag, res in cand_dict.items():
                summary_rows.append({
                    "env": env_name,
                    "method": method_key,
                    "method_label": method_label,
                    "tag": tag,
                    "is_best": tag == best_tag,
                    "final_mean": res["final_mean"],
                    "final_sem": res["final_sem"],
                    "auc_mean": res["auc_mean"],
                    "auc_sem": res["auc_sem"],
                    "elapsed_s": res["elapsed_s"],
                })

            curves = best_res["curves"]
            n_seeds, n_updates = curves.shape
            x = np.linspace(0, total_steps, n_updates)
            mean_c = np.mean(curves, axis=0)
            sem_c = np.std(curves, axis=0, ddof=0) / np.sqrt(n_seeds)
            color = METHOD_COLORS[method_key]
            lbl = f"{method_label} ({best_tag})" if best_tag != "default" else method_label
            ax.plot(x, mean_c, label=lbl, color=color, linewidth=1.8)
            ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color=color, alpha=0.18)

        ax.set_title(f"Brax SAC Critics: {title}")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Return")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"learning_curves_{env_name}.pdf"))
        plt.close(fig)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(data_dir, "summary_returns.csv"), index=False)

    with open(os.path.join(data_dir, "best_configs.json"), "w") as f:
        json.dump(best_configs, f, indent=2)

    best_df = df[df["is_best"]].copy() if not df.empty else df
    if not best_df.empty:
        best_df["Return (Mean ± SEM)"] = best_df.apply(
            lambda r: f"{r['final_mean']:.2f} ± {r['final_sem']:.2f}", axis=1
        )
        cols = ["env", "method_label", "tag", "Return (Mean ± SEM)", "auc_mean"]
        md_str = best_df[cols].to_markdown(index=False)
        with open(os.path.join(tables_dir, "summary_table.md"), "w") as f:
            f.write(md_str + "\n")
        tex_str = best_df[cols].to_latex(index=False, escape=True)
        with open(os.path.join(tables_dir, "summary_table.tex"), "w") as f:
            f.write(tex_str + "\n")


def run_sweep(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    candidates = build_candidates(args.mode, args.critic_lrs, args.lambdas)

    runs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    timesteps: dict[str, int] = {}

    for env_name in args.envs:
        env_steps = get_env_default_timesteps(env_name, args.total_timesteps)
        timesteps[env_name] = env_steps
        runs[env_name] = {}
        print(f"\n=== Brax SAC Sweep: {env_name} ({env_steps:,} steps, {args.n_seeds} seeds) ===")

        for method_key, method_label in METHODS:
            runs[env_name][method_key] = {}
            for cand in candidates[method_key]:
                tag = cand["tag"]
                cfg = build_method_config(
                    method_key=method_key,
                    env_name=env_name,
                    total_timesteps=env_steps,
                    num_envs=args.num_envs,
                    num_steps=args.num_steps,
                    minibatch_size=args.minibatch_size,
                    critic_lr=cand["critic_lr"],
                    lam=cand["lambda"],
                )
                print(f"  -> {method_label} [{tag}] ...", end=" ", flush=True)
                res = run_condition(
                    cfg=cfg,
                    n_seeds=args.n_seeds,
                    master_seed=args.master_seed,
                )
                runs[env_name][method_key][tag] = res
                print(
                    f"final={res['final_mean']:.2f}±{res['final_sem']:.2f} "
                    f"auc={res['auc_mean']:.2f} ({res['elapsed_s']:.1f}s)"
                )

    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, "sweep_metrics_all.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"runs": runs, "timesteps": timesteps}, f)
    print(f"\nSaved metrics to {pkl_path}")

    generate_outputs(runs, timesteps, args.out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brax SAC Critic Comparison Sweep")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--mode", type=str, default="tune_lr", choices=["default", "tune_lr", "tune_full"])
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=brax_sac_config["NUM_ENVS"])
    parser.add_argument("--num_steps", type=int, default=brax_sac_config["NUM_STEPS"])
    parser.add_argument("--minibatch_size", type=int, default=brax_sac_config["CRITIC_MINIBATCH_SIZE"])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/sweep_sac_critics_brax")
    return parser


if __name__ == "__main__":
    run_sweep(build_arg_parser().parse_args())

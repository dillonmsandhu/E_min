#!/usr/bin/env python3
"""
sweep_sac_critics.py

Compares four critic algorithms in Gymnax SAC (gymnax_rl/algos/sac.py):
  1. td_lambda : TD(lambda)
  2. e_0       : E(0)
  3. td_0      : Online TD(0)
  4. e_lambda  : E(lambda) with v_aux

Supports:
  - `default` mode: head-to-head comparison using default SAC hyperparameters
  - `tune_lr` mode: sweeps CRITIC_LR per method (plus lambda=0.8 default for lambda methods)
  - `tune_full` mode: sweeps CRITIC_LR x LAMBDA (for td_lambda and e_lambda)

Outputs under <out_dir>/:
  - data/sweep_metrics_all.pkl
  - data/summary_returns.csv
  - data/best_configs.json
  - plots/learning_curves_<env>.pdf (and consolidated learning_curves_grid.pdf)
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

# Publication vector PDF formatting
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
import jax.numpy as jnp

from gymnax_rl.algos.sac import DEFAULT_CONFIG as SAC_DEFAULT_CONFIG
from gymnax_rl.algos.sac import make_evaluate, make_train

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

DEFAULT_ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
    "Catch-bsuite",
    "PointRobot-misc",
    "Reacher-misc",
]

DEFAULT_CRITIC_LRS = [1e-4, 3e-4, 1e-3, 3e-3]
DEFAULT_LAMBDAS = [0.4, 0.8, 0.95]


def get_env_default_timesteps(env_name: str, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    if "MountainCar" in env_name:
        return 500_000
    if env_name in ("Pendulum-v1", "PointRobot-misc", "Reacher-misc", "Swimmer-misc"):
        return 300_000
    return 150_000


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
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg["ENV_NAME"] = env_name
    cfg["TOTAL_TIMESTEPS"] = int(total_timesteps)
    cfg["NUM_ENVS"] = int(num_envs)
    cfg["NUM_STEPS"] = int(num_steps)
    cfg["ACTOR_MINIBATCH_SIZE"] = int(minibatch_size)
    cfg["CRITIC_MINIBATCH_SIZE"] = int(minibatch_size)
    # Explicit deterministic flag for non-builtin Gymnax envs
    if env_name not in {
        "CartPole-v1",
        "Pendulum-v1",
        "MountainCarContinuous-v0",
        "MountainCar-v0",
        "Acrobot-v1",
    }:
        cfg["DETERMINISTIC_DYNAMICS"] = True

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


def _compute_mean_return_curve(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """Computes (n_seeds, n_updates) episode return curves from Gymnax SAC metrics."""
    completed = np.asarray(metrics["completed_episodes"], dtype=np.float64)
    ret_sum = np.asarray(metrics["episode_return_sum"], dtype=np.float64)
    # Forward-fill any update where 0 episodes finished so curves remain smooth
    n_seeds, n_updates = completed.shape
    curves = np.zeros((n_seeds, n_updates), dtype=np.float64)
    for s in range(n_seeds):
        last_val = np.nan
        for t in range(n_updates):
            if completed[s, t] > 0:
                last_val = ret_sum[s, t] / completed[s, t]
                curves[s, t] = last_val
            else:
                curves[s, t] = last_val
        # Backfill leading NaNs if first few updates had 0 completions
        valid = np.where(~np.isnan(curves[s]))[0]
        if len(valid) > 0:
            curves[s, :valid[0]] = curves[s, valid[0]]
        else:
            curves[s, :] = 0.0
    return curves


def run_condition(
    cfg: dict[str, Any],
    n_seeds: int,
    master_seed: int,
    eval_episodes: int = 32,
) -> dict[str, Any]:
    train_fn = jax.jit(jax.vmap(make_train(cfg)))
    eval_fn = jax.jit(jax.vmap(make_evaluate(cfg, num_episodes=eval_episodes, greedy=False)))

    rng = jax.random.PRNGKey(master_seed)
    train_keys = jax.random.split(rng, n_seeds)
    eval_keys = jax.random.split(jax.random.PRNGKey(master_seed + 10_000), n_seeds)

    t0 = time.time()
    out = train_fn(train_keys)
    jax.block_until_ready(out["metrics"])
    elapsed = time.time() - t0

    runner_state = out["runner_state"]
    actor_params = runner_state[0].params
    eval_returns = np.asarray(eval_fn(actor_params, eval_keys), dtype=np.float64)

    metrics_np = {k: np.asarray(v) for k, v in out["metrics"].items()}
    curves = _compute_mean_return_curve(metrics_np)
    window = max(1, curves.shape[1] // 10)
    final_seed_means = np.mean(curves[:, -window:], axis=1)
    auc_seed_means = np.mean(curves, axis=1)

    return {
        "curves": curves,  # (n_seeds, n_updates)
        "final_mean": float(np.mean(final_seed_means)),
        "final_sem": float(np.std(final_seed_means, ddof=0) / np.sqrt(n_seeds)),
        "auc_mean": float(np.mean(auc_seed_means)),
        "auc_sem": float(np.std(auc_seed_means, ddof=0) / np.sqrt(n_seeds)),
        "eval_mean": float(np.mean(eval_returns)),
        "eval_sem": float(np.std(eval_returns, ddof=0) / np.sqrt(n_seeds)),
        "eval_returns": eval_returns,
        "elapsed_s": float(elapsed),
        "params": {
            "CRITIC_TYPE": cfg["CRITIC_TYPE"],
            "CRITIC_LR": cfg["CRITIC_LR"],
            "TD_LAMBDA": cfg["TD_LAMBDA"],
            "E_LAMBDA": cfg["E_LAMBDA"],
        },
    }


def score_candidates(env_runs: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    """Ranks candidate tags by final_mean (tie-break by auc_mean)."""
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
                "eval_mean": best_res["eval_mean"],
                "eval_sem": best_res["eval_sem"],
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
                    "eval_mean": res["eval_mean"],
                    "eval_sem": res["eval_sem"],
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

        ax.set_title(f"Gymnax SAC Critics: {env_name}")
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

    # Markdown & LaTeX table for best per-method results
    best_df = df[df["is_best"]].copy() if not df.empty else df
    if not best_df.empty:
        best_df["Return (Mean ± SEM)"] = best_df.apply(
            lambda r: f"{r['final_mean']:.2f} ± {r['final_sem']:.2f}", axis=1
        )
        best_df["Eval (Mean ± SEM)"] = best_df.apply(
            lambda r: f"{r['eval_mean']:.2f} ± {r['eval_sem']:.2f}", axis=1
        )
        cols = ["env", "method_label", "tag", "Return (Mean ± SEM)", "Eval (Mean ± SEM)", "auc_mean"]
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
        print(f"\n=== Gymnax SAC Sweep: {env_name} ({env_steps:,} steps, {args.n_seeds} seeds) ===")

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
                    eval_episodes=args.eval_episodes,
                )
                runs[env_name][method_key][tag] = res
                print(
                    f"final={res['final_mean']:.2f}±{res['final_sem']:.2f} "
                    f"eval={res['eval_mean']:.2f}±{res['eval_sem']:.2f} "
                    f"({res['elapsed_s']:.1f}s)"
                )

    data_dir = os.path.join(args.out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    pkl_path = os.path.join(data_dir, "sweep_metrics_all.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"runs": runs, "timesteps": timesteps}, f)
    print(f"\nSaved metrics to {pkl_path}")

    generate_outputs(runs, timesteps, args.out_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gymnax SAC Critic Comparison Sweep")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    parser.add_argument("--mode", type=str, default="tune_lr", choices=["default", "tune_lr", "tune_full"])
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=DEFAULT_CRITIC_LRS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=SAC_DEFAULT_CONFIG["NUM_ENVS"])
    parser.add_argument("--num_steps", type=int, default=SAC_DEFAULT_CONFIG["NUM_STEPS"])
    parser.add_argument("--minibatch_size", type=int, default=SAC_DEFAULT_CONFIG["CRITIC_MINIBATCH_SIZE"])
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--eval_episodes", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default="results/sweep_sac_critics_gymnax")
    return parser


if __name__ == "__main__":
    run_sweep(build_arg_parser().parse_args())

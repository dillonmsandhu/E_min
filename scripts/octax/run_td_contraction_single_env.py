#!/usr/bin/env python3
"""
scripts/octax/run_td_contraction_single_env.py

Runs TD(lambda) across multiple lambda values (default: 0.0, 0.5, 0.8, 0.95)
on a single Octax environment (default: Octax-blinky — maze navigation with
directed corridors and high CNN aliasing) and tracks all four contraction-loss
and return-variance diagnostics over training:
  1. diag_rho_c         : Centered gradient aliasing (rho_c)
  2. diag_r_rot_c       : Centered split-half rotation ratio (r_rot,c)
  3. diag_ret_var_ratio : Return-to-TD error ratio E[(G-V)^2] / E[delta^2]
  4. diag_aux_unexp_var : Critic / v_aux unexplained variance (1 - R^2)

Saves learning-curve figure (.png + .pdf), summary table (.csv + .md), and raw .pkl.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from octax_rl.algos.ppo import make_train
from octax_rl.core.octax_config import config as OCTAX_BASE

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]


PALETTE = ["#dc2626", "#ea580c", "#2563eb", "#059669", "#7c3aed", "#0891b2"]


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    p = argparse.ArgumentParser(description="Track TD(lambda) contraction diagnostics on a single Octax env")
    p.add_argument("--env_name", type=str, default="Octax-blinky")
    p.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 0.8, 0.95])
    p.add_argument("--include_e_lambda", type=float, default=None, help="Optionally also run E(lambda) at this lambda")
    p.add_argument("--total_timesteps", type=int, default=2_000_000)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=128)
    p.add_argument("--minibatch_size", type=int, default=1024)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gae_lambda", type=float, default=0.8)
    p.add_argument("--sticky_action_prob", type=float, default=0.0)
    p.add_argument("--diag_probe_envs", type=int, default=16)
    p.add_argument("--diag_ema", type=float, default=0.95)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--master_seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default=None)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join("results", "octax", f"td_contraction_{args.env_name}")
    os.makedirs(out_dir, exist_ok=True)

    run_specs = []
    for idx, lmb in enumerate(args.lambdas):
        c_type = "td_0" if abs(lmb) < 1e-9 else "fitted"
        lbl = "TD(0)" if abs(lmb) < 1e-9 else f"TD({lmb:g})"
        run_specs.append((lbl, c_type, lmb, PALETTE[idx % len(PALETTE)], "-"))
    if args.include_e_lambda is not None:
        lmb = float(args.include_e_lambda)
        c_type = "e_0" if abs(lmb) < 1e-9 else "e_lambda"
        lbl = "E(0) (v_aux)" if abs(lmb) < 1e-9 else f"E({lmb:g}) (v_aux)"
        run_specs.append((lbl, c_type, lmb, "#7c3aed", "--"))

    seed_keys = jax.random.split(jax.random.PRNGKey(args.master_seed), args.n_seeds)
    batch_size = args.num_envs * args.num_steps
    results = {}
    summary_rows = []

    print(f"=== Octax Single-Env Contraction Tracking: {args.env_name} ({args.total_timesteps:,} steps, {args.n_seeds} seeds) ===")
    for lbl, c_type, lmb, color, ls in run_specs:
        cfg = dict(OCTAX_BASE)
        cfg.update({
            "ENV_NAME": args.env_name,
            "CRITIC_TYPE": c_type,
            "E_VARIANT": "aux",
            "VALUE_LAMBDA": lmb,
            "E_LAMBDA": lmb,
            "GAE_LAMBDA": args.gae_lambda,
            "STICKY_ACTION_PROB": args.sticky_action_prob,
            "NUM_ENVS": args.num_envs,
            "NUM_STEPS": args.num_steps,
            "TOTAL_TIMESTEPS": args.total_timesteps,
            "NUM_EPOCHS": args.num_epochs,
            "ACTOR_EPOCHS": args.num_epochs,
            "CRITIC_EPOCHS": args.num_epochs,
            "MINIBATCH_SIZE": args.minibatch_size,
            "ACTOR_MINIBATCH_SIZE": args.minibatch_size,
            "CRITIC_MINIBATCH_SIZE": args.minibatch_size,
            "ACTOR_LR": args.lr,
            "CRITIC_LR": args.lr,
            "DIAG_PROBE_ENVS": args.diag_probe_envs,
            "DIAG_EMA": args.diag_ema,
            "LOG_CONTRACTION_DIAGNOSTICS": True,
        })
        t0 = time.time()
        out = jax.jit(jax.vmap(make_train(cfg)))(seed_keys)
        jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
        dt = time.time() - t0
        m = out["metrics"]

        res = {
            "color": color,
            "ls": ls,
            "ret": np.asarray(m["returned_episode_returns"]),
            "rho_c": np.asarray(m["diag_rho_c"]),
            "r_rot_c": np.asarray(m["diag_r_rot_c"]),
            "vr": np.asarray(m["diag_ret_var_ratio"]),
            "uv": np.asarray(m["diag_aux_unexp_var"]),
            "alpha": np.asarray(m["diag_alpha_eff"]),
            "beta": np.asarray(m["diag_beta_eff"]),
            "elapsed_sec": dt,
        }
        results[lbl] = res
        row = {
            "Method": lbl,
            "Return_AUC": float(res["ret"].mean()),
            "Return_Final": float(res["ret"][:, -1].mean()),
            "AUC_rho_c": float(res["rho_c"].mean()),
            "AUC_r_rot_c": float(res["r_rot_c"].mean()),
            "Peak_r_rot_c": float(res["r_rot_c"].mean(axis=0).max()),
            "AUC_VR_ret": float(res["vr"].mean()),
            "AUC_1_minus_R2": float(res["uv"].mean()),
            "Time_s": round(dt, 1),
        }
        summary_rows.append(row)
        print(
            f"  {lbl:14s} ({dt:5.1f}s) | Ret AUC={row['Return_AUC']:7.2f} Fin={row['Return_Final']:7.2f} "
            f"| rho_c={row['AUC_rho_c']:.3f} | r_rot={row['AUC_r_rot_c']:.3f} (peak {row['Peak_r_rot_c']:.3f}) "
            f"| VR={row['AUC_VR_ret']:6.1f} | 1-R2={row['AUC_1_minus_R2']:.3f}"
        )

    with open(os.path.join(out_dir, "diagnostics.pkl"), "wb") as f:
        pickle.dump({"env_name": args.env_name, "batch_size": batch_size, "results": results}, f)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"### {args.env_name}: TD(λ) Contraction-Loss & Return-Variance Diagnostics\n\n")
        f.write(df.to_markdown(index=False) + "\n")

    # 2x3 diagnostic figure
    n_upd = next(iter(results.values()))["ret"].shape[-1]
    xs = (np.arange(1, n_upd + 1) * batch_size) / 1e6
    win = max(1, n_upd // 40)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    panels = [
        (axes[0, 0], "ret",     "Episode Return",                       "Return",           None),
        (axes[0, 1], "rho_c",   "Centered Gradient Aliasing (ρ_c)",     "ρ_c",              0.85),
        (axes[0, 2], "r_rot_c", "Centered Rotation Ratio (r_rot,c)",    "r_rot,c",          0.25),
        (axes[1, 0], "vr",      "Return-to-TD Error Ratio (VR_ret)",    "E[(G-V)²]/E[δ²]",  1.0),
        (axes[1, 1], "uv",      "Critic / Aux Unexplained Var (1 - R²)","1 - R²",           1.0),
    ]

    for ax, key, title, ylabel, ref in panels:
        for lbl, _, _, _, _ in run_specs:
            r = results[lbl]
            mu = _smooth_1d(np.nanmean(r[key], axis=0), win)
            sd = _smooth_1d(np.nanstd(r[key], axis=0), win)
            ax.plot(xs, mu, lw=2.0, ls=r["ls"], color=r["color"], label=lbl)
            if args.n_seeds > 1:
                ax.fill_between(xs, mu - sd, mu + sd, color=r["color"], alpha=0.14)
        if ref is not None:
            ax.axhline(ref, color="#64748b", ls=":", lw=1.1)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Env Steps (M)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if key == "ret":
            ax.legend(loc="best", fontsize=8.5)
        if key == "vr":
            ax.set_yscale("log")

    # Summary bar panel: AUC of rho_c and r_rot_c
    ax = axes[1, 2]
    lbls = [s[0] for s in run_specs]
    x = np.arange(len(lbls))
    w = 0.36
    ax.bar(x - w / 2, [results[l]["rho_c"].mean() for l in lbls], width=w, color="#2563eb", alpha=0.85, label="AUC ρ_c")
    ax.bar(x + w / 2, [results[l]["r_rot_c"].mean() for l in lbls], width=w, color="#dc2626", alpha=0.85, label="AUC r_rot,c")
    ax.axhline(0.25, color="#dc2626", ls=":", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=20, ha="right", fontsize=8.5)
    ax.set_title("Contraction-Loss AUC vs. λ", fontsize=10.5, fontweight="bold")
    ax.legend(loc="best", fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"{args.env_name}: TD(λ) Contraction-Loss & Return-Variance Diagnostics ({args.n_seeds} seeds)",
        fontsize=13.0,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    base = os.path.join(out_dir, f"td_contraction_{args.env_name}")
    plt.savefig(base + ".png", dpi=250, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

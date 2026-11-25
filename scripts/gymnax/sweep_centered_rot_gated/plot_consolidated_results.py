#!/usr/bin/env python3
"""Consolidates per-environment results under results/gymnax/sweep_centered_rot_gated
and plots centered r_rot_c(t) and backward gate omega_t by lambda across all conditions.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

from scripts.gymnax.sweep_centered_rot_gated.sweep_centered_rot_gated import METHOD_KEYS, REFERENCE_METHOD, SUITE_TITLE
from scripts.sweep_standard_plotting import ENV_TITLES
from scripts.sweep_standard_runner import consolidate_standardized_sweep


def plot_rotation_gate_by_lambda(pkl_path: str, fig_dir: str) -> None:
    """Plots omega_t (top) and centered r_rot_c(t) (bottom) comparing lambda=0 vs each E(lambda) value."""
    if not os.path.exists(pkl_path):
        return
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    runs = payload["runs"]
    timesteps = payload["timesteps"]
    preferred_order = [
        "SpaceInvaders-MinAtar",
        "SpaceInvaders-MinAtar-AvgPool",
        "MountainCarContinuous-v0",
    ]
    envs = [e for e in preferred_order if e in runs] + [e for e in sorted(runs.keys()) if e not in preferred_order]
    if not envs:
        return

    n_envs = len(envs)
    fig, axes = plt.subplots(2, n_envs, figsize=(4.8 * n_envs, 6.4), squeeze=False, constrained_layout=True)

    palette = ["#059669", "#2563eb", "#7c3aed", "#ea580c", "#db2777"]
    tau_val = 0.25

    for c_i, env in enumerate(envs):
        env_runs = runs[env]
        env_ts = float(timesteps[env])
        ax_w = axes[0, c_i]
        ax_r = axes[1, c_i]

        series = []
        if "e_0_rot_gated" in env_runs:
            best_0 = max(env_runs["e_0_rot_gated"].values(), key=lambda e: e["auc_mean"])
            if "rot_omega" in best_0:
                series.append(("E(0) ($\\lambda=0.0$)", best_0))

        if "e_lambda_rot_gated" in env_runs:
            by_lam = {}
            for entry in env_runs["e_lambda_rot_gated"].values():
                if "rot_omega" not in entry:
                    continue
                lam = float(entry["params"].get("E_LAMBDA", entry["config"]["E_LAMBDA"]))
                if lam not in by_lam or entry["auc_mean"] > by_lam[lam]["auc_mean"]:
                    by_lam[lam] = entry
            for lam in sorted(by_lam.keys()):
                series.append((f"E($\\lambda={lam:g}$)", by_lam[lam]))

        for s_i, (label, entry) in enumerate(series):
            color = palette[s_i % len(palette)]
            om = np.asarray(entry["rot_omega"])
            rr = np.asarray(entry["rot_r_rot_ema"])
            tau_val = float(entry["config"].get("ROT_GATE_TAU", tau_val))
            n_seeds, n_upd = om.shape
            xs = np.linspace(0.0, env_ts / 1e6, n_upd)

            om_mean, om_sem = om.mean(axis=0), om.std(axis=0) / np.sqrt(n_seeds)
            rr_mean, rr_sem = rr.mean(axis=0), rr.std(axis=0) / np.sqrt(n_seeds)

            ax_w.plot(xs, om_mean, color=color, lw=2.0, label=label)
            ax_w.fill_between(xs, np.clip(om_mean - om_sem, 0.0, 1.0), np.clip(om_mean + om_sem, 0.0, 1.0), color=color, alpha=0.16)

            ax_r.plot(xs, rr_mean, color=color, lw=2.0, label=label)
            ax_r.fill_between(xs, np.maximum(rr_mean - rr_sem, 0.0), rr_mean + rr_sem, color=color, alpha=0.16)

        ax_w.set_title(ENV_TITLES.get(env, env), fontweight="bold")
        ax_w.set_ylim(-0.04, 1.04)
        ax_w.grid(True, alpha=0.22)
        if c_i == 0:
            ax_w.set_ylabel("Backward Gate $\\omega_t$", fontweight="bold")

        ax_r.axhline(tau_val, color="#444444", lw=1.2, ls="--", label=f"Threshold $\\tau={tau_val:g}$")
        ax_r.set_xlabel("Env Steps (M)")
        ax_r.grid(True, alpha=0.22)
        if c_i == 0:
            ax_r.set_ylabel("Centered Rotation Ratio $\\bar{r}_{\\mathrm{rot},c}(t)$", fontweight="bold")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=9.2, frameon=True, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Centered Online Rotation Gate $\\omega_t$ and Spectral Ratio $\\bar{r}_{\\mathrm{rot},c}(t)$ by $\\lambda$", fontsize=11.8, fontweight="bold", y=1.11)

    os.makedirs(fig_dir, exist_ok=True)
    out_pdf = os.path.join(fig_dir, "rot_gate_diagnostics_by_lambda.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved centered rotation gate diagnostics -> {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate Centered Rotation-Gated E V-PPO Sweep Results")
    parser.add_argument("--out_dir", type=str, default="results/gymnax/sweep_centered_rot_gated")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--all_pairs", action="store_true")
    parser.add_argument("--exclude_methods", nargs="*", default=[])
    args = parser.parse_args()
    active_methods = [m for m in METHOD_KEYS if m not in set(args.exclude_methods)]
    consolidate_standardized_sweep(
        out_dir=args.out_dir,
        method_keys=active_methods,
        suite_title=SUITE_TITLE,
        reference_method=REFERENCE_METHOD,
        is_v_ppo=True,
        save_dir=args.save_dir,
        all_pairs=args.all_pairs,
    )
    target_dir = args.save_dir if args.save_dir is not None else args.out_dir
    pkl_path = os.path.join(target_dir, "data", "sweep_metrics_all.pkl")
    fig_dir = os.path.join(target_dir, "figures")
    plot_rotation_gate_by_lambda(pkl_path, fig_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Consolidates per-environment results under results/brax/sweep_v_ppo."""

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

from scripts.brax.sweep_v_ppo.sweep_v_ppo import DEFAULT_ENVS, METHOD_KEYS, REFERENCE_METHOD, SUITE_TITLE
from scripts.sweep_standard_plotting import ENV_TITLES
from scripts.sweep_standard_runner import consolidate_standardized_sweep


def plot_rotation_gate_omega(pkl_path: str, plots_dir: str) -> None:
    """Plots backward gate omega_t by lambda across all environments."""
    if not os.path.exists(pkl_path):
        return
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    runs = payload["runs"]
    timesteps = payload["timesteps"]
    envs = [e for e in DEFAULT_ENVS if e in runs] + [e for e in sorted(runs.keys()) if e not in DEFAULT_ENVS]
    if not envs:
        return

    has_omega = any(
        "rot_omega" in entry
        for env in envs
        for m_key in ("e_0_rot_gated", "e_lambda_rot_gated")
        for entry in runs[env].get(m_key, {}).values()
    )
    if not has_omega:
        return

    n_envs = len(envs)
    n_cols = min(4, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.8 * n_rows), squeeze=False, constrained_layout=True)

    palette = ["#059669", "#2563eb", "#7c3aed", "#ea580c", "#db2777"]
    legend_handles = {}

    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        env_runs = runs[env]
        env_ts = float(timesteps[env])

        series = []
        if "e_0_rot_gated" in env_runs:
            cands = [e for e in env_runs["e_0_rot_gated"].values() if "rot_omega" in e]
            if cands:
                series.append(("E(0) ($\\lambda=0$)", max(cands, key=lambda e: e["auc_mean"])))

        if "e_lambda_rot_gated" in env_runs:
            by_lam = {}
            for entry in env_runs["e_lambda_rot_gated"].values():
                if "rot_omega" not in entry:
                    continue
                lam = float(entry["params"]["E_LAMBDA"]) if "E_LAMBDA" in entry["params"] else float(entry["config"]["E_LAMBDA"])
                if lam not in by_lam or entry["auc_mean"] > by_lam[lam]["auc_mean"]:
                    by_lam[lam] = entry
            for lam in sorted(by_lam.keys()):
                series.append((f"E($\\lambda={lam:g}$)", by_lam[lam]))

        for s_i, (label, entry) in enumerate(series):
            color = palette[s_i % len(palette)]
            om = np.asarray(entry["rot_omega"])
            n_seeds, n_upd = om.shape
            xs = np.linspace(0.0, env_ts / 1e6, n_upd)
            om_mean = om.mean(axis=0)
            om_sem = om.std(axis=0) / np.sqrt(n_seeds)
            line, = ax.plot(xs, om_mean, color=color, lw=1.8, label=label)
            ax.fill_between(xs, np.clip(om_mean - om_sem, 0.0, 1.0), np.clip(om_mean + om_sem, 0.0, 1.0), color=color, alpha=0.16)
            if label not in legend_handles:
                legend_handles[label] = line

        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.0, fontweight="bold")
        ax.set_ylim(-0.04, 1.04)
        ax.grid(True, alpha=0.25)
        if r == n_rows - 1:
            ax.set_xlabel("Env Steps (M)")
        if c == 0:
            ax.set_ylabel("Backward Gate $\\omega_t$")

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            ncol=len(legend_handles),
            fontsize=9.0,
            frameon=True,
            bbox_to_anchor=(0.5, 1.04),
        )
    fig.suptitle(f"{SUITE_TITLE}: Online Rotation Gate $\\omega_t$ by $\\lambda$", fontsize=12.0, fontweight="bold", y=1.08)

    diag_dir = os.path.join(plots_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    out_pdf = os.path.join(diag_dir, "rot_gate_omega_by_lambda.pdf")
    out_png = os.path.join(diag_dir, "rot_gate_omega_by_lambda.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"Saved rotation gate omega plot -> {out_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate Brax V-PPO Sweep Results")
    parser.add_argument("--out_dir", type=str, default="results/brax/sweep_v_ppo")
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
    plot_rotation_gate_omega(
        pkl_path=os.path.join(target_dir, "data", "sweep_metrics_all.pkl"),
        plots_dir=os.path.join(target_dir, "plots"),
    )


if __name__ == "__main__":
    main()

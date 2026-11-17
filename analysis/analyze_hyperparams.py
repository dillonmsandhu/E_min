#!/usr/bin/env python3
"""
analysis/analyze_hyperparams.py

Cross-suite hyperparameter & sensitivity analysis across results/brax/ and results/gymnax/:
  1. Tuning sensitivity (Default vs. Tuned % gain, and mean-over-grid / best-tuned ratio).
  2. Selected CRITIC_LR, CRITIC_EPOCHS, and LAMBDA distributions per algorithm.
  3. Discrete (exact marginalization) vs. Continuous (K=4 probe) LR preferences for E(0) (2pi).
"""

from __future__ import annotations

import argparse
from collections import Counter
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.sweep_standard_plotting import _extract_run_lambda, get_algo_meta


SUITES = [
    ("Brax V-PPO", "brax/sweep_v_ppo"),
    ("Brax Q-PPO", "brax/sweep_q_ppo"),
    ("Gymnax V-PPO", "gymnax/sweep_v_ppo"),
    ("Gymnax Q-PPO", "gymnax/sweep_q_ppo"),
]

GYMNAX_CONTINUOUS_ENVS = {
    "MountainCarContinuous-v0",
    "Pendulum-v1",
    "PointRobot-misc",
    "Reacher-misc",
    "Swimmer-misc",
}


def load_suite_runs(results_root: str, rel_path: str) -> dict:
    pkl_path = os.path.join(results_root, rel_path, "data", "sweep_metrics_all.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing consolidated results: {pkl_path}")
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    return payload["runs"] if "runs" in payload else payload


def analyze_suites(results_root: str, out_dir: str, exclude_methods: set[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    e0_split_rows = []

    for suite_name, rel_path in SUITES:
        runs = load_suite_runs(results_root, rel_path)
        envs = sorted(runs.keys())
        methods = sorted(
            {m for e in envs for m in runs[e].keys() if m not in exclude_methods}
        )
        for m_key in methods:
            lrs, eps, lams = Counter(), Counter(), Counter()
            tune_gains, grid_ratios = [], []
            for env in envs:
                m_runs = runs[env][m_key]
                best_tag = max(m_runs.keys(), key=lambda t: m_runs[t]["auc_mean"])
                best = m_runs[best_tag]
                p = best.get("params", {})
                lrs[p["CRITIC_LR"]] += 1
                eps[p["CRITIC_EPOCHS"]] += 1
                lv = _extract_run_lambda(p)
                if lv is not None:
                    lams[lv] += 1

                if "default" in m_runs:
                    d_auc = m_runs["default"]["auc_mean"]
                    b_auc = best["auc_mean"]
                    if abs(d_auc) > 1e-4:
                        tune_gains.append((b_auc - d_auc) / abs(d_auc) * 100.0)
                all_aucs = [r["auc_mean"] for r in m_runs.values()]
                if abs(best["auc_mean"]) > 1e-4 and all_aucs:
                    mean_auc = float(np.mean(all_aucs))
                    # Fraction of best performance achieved on average across the sweep grid
                    grid_ratios.append(100.0 - abs(best["auc_mean"] - mean_auc) / abs(best["auc_mean"]) * 100.0)

            lbl = get_algo_meta(m_key)["label"]
            summary_rows.append({
                "Suite": suite_name,
                "Method": m_key,
                "Label": lbl,
                "N_Envs": len(envs),
                "Med_Tune_Gain_%": float(np.median(tune_gains)) if tune_gains else 0.0,
                "Mean_Tune_Gain_%": float(np.mean(tune_gains)) if tune_gains else 0.0,
                "Grid_Robustness_%": float(np.median(grid_ratios)) if grid_ratios else np.nan,
                "LR_Counts": dict(sorted(lrs.items())),
                "EP_Counts": dict(sorted(eps.items())),
                "LAM_Counts": dict(sorted(lams.items())),
                "Pct_Top_LR": 100.0 * lrs[max(lrs.keys())] / len(envs) if lrs else 0.0,
                "Pct_Lam_095": 100.0 * lams.get(0.95, 0) / len(envs) if lams else np.nan,
            })

        # Discrete vs Continuous LR breakdown for E(0) Bilinear (2pi) vs E(0) v_aux
        if "e_0_bilinear" in methods and "e_0_v_aux" in methods:
            for m_key in ("e_0_v_aux", "e_0_bilinear"):
                if "Gymnax" in suite_name:
                    groups = [
                        ("Gymnax Discrete (Marg.)", [e for e in envs if e not in GYMNAX_CONTINUOUS_ENVS]),
                        ("Gymnax Continuous (K=4)", [e for e in envs if e in GYMNAX_CONTINUOUS_ENVS]),
                    ]
                else:
                    groups = [("Brax Continuous (K=4)", envs)]
                for g_name, g_envs in groups:
                    lrs_g, eps_g = Counter(), Counter()
                    for env in g_envs:
                        best_tag = max(runs[env][m_key].keys(), key=lambda t: runs[env][m_key][t]["auc_mean"])
                        p = runs[env][m_key][best_tag]["params"]
                        lrs_g[p["CRITIC_LR"]] += 1
                        eps_g[p["CRITIC_EPOCHS"]] += 1
                    e0_split_rows.append({
                        "Group": g_name,
                        "Method": get_algo_meta(m_key)["label"],
                        "N_Envs": len(g_envs),
                        "LR_Counts": dict(sorted(lrs_g.items())),
                        "EP_Counts": dict(sorted(eps_g.items())),
                    })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(out_dir, "hyperparam_summary.csv"), index=False)

    df_e0 = pd.DataFrame(e0_split_rows)
    df_e0.to_csv(os.path.join(out_dir, "e0_lr_discrete_vs_continuous.csv"), index=False)

    md_path = os.path.join(out_dir, "hyperparam_summary.md")
    with open(md_path, "w") as f:
        f.write("# Cross-Suite Hyperparameter Analysis (Brax & Gymnax)\n\n")
        f.write("## 1. Tuning Sensitivity & Selected Hyperparameters\n\n")
        cols = ["Suite", "Label", "Med_Tune_Gain_%", "Grid_Robustness_%", "LR_Counts", "EP_Counts", "LAM_Counts"]
        df_disp = df[cols].copy()
        df_disp["Med_Tune_Gain_%"] = df_disp["Med_Tune_Gain_%"].map(lambda x: f"{x:+.1f}%")
        df_disp["Grid_Robustness_%"] = df_disp["Grid_Robustness_%"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        f.write(df_disp.to_markdown(index=False) + "\n\n")

        if not df_e0.empty:
            f.write("## 2. E(0) Learning Rate Selection: Discrete (Exact Marg.) vs. Continuous (K=4)\n\n")
            f.write(df_e0.to_markdown(index=False) + "\n")

    # Visual 3-panel summary figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # Panel A: Tuning sensitivity (Median % gain Default -> Tuned)
    key_methods = ["td_lambda", "sarsa_lambda", "e_lambda_v_aux", "bsmc_v_aux", "bsmc_double_pi"]
    sub = df[df["Method"].isin(key_methods)].copy()
    sub["Tag"] = sub["Suite"].str.replace(" V-PPO", " V").str.replace(" Q-PPO", " Q") + "\n" + sub["Label"]
    colors = [get_algo_meta(m)["color"] for m in sub["Method"]]
    y_pos = np.arange(len(sub))
    ax1.barh(y_pos, sub["Med_Tune_Gain_%"], color=colors, alpha=0.88, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sub["Tag"], fontsize=7.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Median AUC Gain: Default → Tuned (%)")
    ax1.set_title("A. Hyperparameter Sensitivity (Lower = Robust)", fontsize=10.0, fontweight="bold")
    ax1.grid(True, alpha=0.25, axis="x")

    # Panel B: Fraction of envs selecting lambda=0.95 (TD/SARSA vs E(lambda) vs BSMC)
    lam_sub = df[df["Pct_Lam_095"].notna()].copy()
    lam_sub["Tag"] = lam_sub["Suite"].str.replace(" V-PPO", " V").str.replace(" Q-PPO", " Q") + "\n" + lam_sub["Label"]
    colors_l = [get_algo_meta(m)["color"] for m in lam_sub["Method"]]
    y_l = np.arange(len(lam_sub))
    ax2.barh(y_l, lam_sub["Pct_Lam_095"], color=colors_l, alpha=0.88, height=0.6)
    ax2.set_yticks(y_l)
    ax2.set_yticklabels(lam_sub["Tag"], fontsize=7.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    ax2.set_xlabel("% of Envs Selecting λ = 0.95")
    ax2.set_title("B. Reliance on Near-MC (λ = 0.95)", fontsize=10.0, fontweight="bold")
    ax2.grid(True, alpha=0.25, axis="x")

    # Panel C: E(0) Bilinear (2pi) vs E(0) v_aux LR choices on Discrete vs Continuous
    if not df_e0.empty:
        labels_c = [f"{r['Group']}\n{r['Method']}" for _, r in df_e0.iterrows()]
        lr_vals = sorted({lr for d in df_e0["LR_Counts"] for lr in d.keys()})
        lr_colors = {1e-4: "#94a3b8", 3e-4: "#38bdf8", 1e-3: "#2563eb", 3e-3: "#7c3aed"}
        y_c = np.arange(len(df_e0))
        left = np.zeros(len(df_e0))
        for lr in lr_vals:
            pcts = np.array([
                100.0 * r["LR_Counts"].get(lr, 0) / max(1, r["N_Envs"])
                for _, r in df_e0.iterrows()
            ])
            ax3.barh(y_c, pcts, left=left, height=0.58, color=lr_colors.get(lr, "#64748b"), label=f"lr={lr:g}")
            left += pcts
        ax3.set_yticks(y_c)
        ax3.set_yticklabels(labels_c, fontsize=7.8)
        ax3.invert_yaxis()
        ax3.set_xlim(0, 100)
        ax3.set_xlabel("% of Envs Selecting CRITIC_LR")
        ax3.set_title("C. E(0) LR: Discrete Marg. vs. Cont. K=4", fontsize=10.0, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=7.5)
        ax3.grid(True, alpha=0.25, axis="x")

    fig.suptitle("Cross-Suite Hyperparameter Insights (Brax & Gymnax)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hyperparam_insights.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(out_dir, "hyperparam_insights.pdf"), bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-suite hyperparameter analysis for Brax & Gymnax")
    parser.add_argument("--results_root", type=str, default=os.path.join(_repo_root, "results"))
    parser.add_argument("--out_dir", type=str, default=os.path.join(_script_dir, "outputs"))
    parser.add_argument("--exclude_methods", nargs="*", default=["bsmc_bilinear_v_aux"])
    args = parser.parse_args()
    analyze_suites(args.results_root, args.out_dir, set(args.exclude_methods))


if __name__ == "__main__":
    main()

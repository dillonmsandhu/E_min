#!/usr/bin/env python3
"""
scripts/pipeline/compile_cross_suite.py

Compiles Stage 2 `cross_suite_summary.csv` files across ALL completed suites
(brax, mjx_locomotion, octax, minatar, gymnax) and generates publication-ready
vector PDF scatter plots & aggregate tables:
  1. Rotation Ratio (r_rot,c) vs. Relative Performance (E vs. TD) across ALL envs
  2. Gradient Aliasing (ρ_c) vs. Relative Performance (E vs. TD) across ALL envs
  3. Per-λ Cross-Suite Trend (r_rot,c and E-vs-TD gain as a function of λ)
  4. Master CSV: results/pipeline/cross_suite/all_suites_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

SUITE_STYLES: dict[str, dict[str, str]] = {
    "brax": {"label": "Brax (γ=0.99)", "color": "#2563EB", "marker": "o"},
    "mjx_locomotion": {"label": "MJX Locomotion (γ=0.97)", "color": "#DC2626", "marker": "s"},
    "octax": {"label": "Octax (Visual)", "color": "#7C3AED", "marker": "^"},
    "minatar": {"label": "MinAtar (Visual)", "color": "#D97706", "marker": "D"},
    "gymnax": {"label": "Gymnax (Diagnostic)", "color": "#059669", "marker": "P"},
}


def _to_float(val: Any) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def load_all_csvs(out_root: str, suites: list[str] | None = None) -> list[dict[str, Any]]:
    pattern = os.path.join(out_root, "*", "stage2_eval", "data", "cross_suite_summary.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No cross_suite_summary.csv files found matching: {pattern}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if suites and r["suite"] not in suites:
                    continue
                rows.append(r)
    return rows


def _plot_diagnostic_scatter(
    rows: list[dict[str, Any]],
    x_col: str,
    x_label: str,
    y_col: str,
    y_label: str,
    e_method: str,
    td_method: str,
    eval_mode: str,
    title: str,
    out_pdf: str,
    filter_tag: str | None = None,
) -> None:
    """
    Plots x = TD diagnostic (e.g., diag_r_rot_c_mean of TD) vs.
          y = E relative performance vs. TD (e.g., rel_auc_vs_td_lambda)
    across all environments from all available suites.
    """
    # Index TD diagnostics by (suite, env, tag)
    td_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        if r["eval_mode"] != eval_mode or r["method"] != td_method:
            continue
        if filter_tag is not None and r["tag"] != filter_tag:
            continue
        td_lookup[(r["suite"], r["env"], r["tag"])] = r

    pts: list[tuple[str, str, float, float]] = []
    for r in rows:
        if r["eval_mode"] != eval_mode or r["method"] != e_method:
            continue
        if filter_tag is not None and r["tag"] != filter_tag:
            continue
        td_row = td_lookup.get((r["suite"], r["env"], r["tag"]))
        if td_row is None:
            continue
        x = _to_float(td_row.get(x_col))
        y = _to_float(r.get(y_col))
        if np.isfinite(x) and np.isfinite(y):
            pts.append((r["suite"], r["env"], x, y))

    if len(pts) < 2:
        print(f"[Skip] Not enough finite points ({len(pts)}) for {out_pdf}")
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axhline(1.0, color="#6B7280", linestyle="--", linewidth=1.0, alpha=0.8, label="Parity (E = TD)")

    xs_all = np.array([p[2] for p in pts], dtype=np.float64)
    ys_all = np.array([p[3] for p in pts], dtype=np.float64)

    for suite in sorted(set(p[0] for p in pts)):
        style = SUITE_STYLES.get(suite, {"label": suite, "color": "#111827", "marker": "o"})
        sub = [p for p in pts if p[0] == suite]
        sx = [p[2] for p in sub]
        sy = [p[3] for p in sub]
        ax.scatter(
            sx,
            sy,
            c=style["color"],
            marker=style["marker"],
            s=64,
            alpha=0.88,
            edgecolors="white",
            linewidths=0.6,
            label=f"{style['label']} (n={len(sub)})",
            zorder=3,
        )
        for _, env, xv, yv in sub:
            short_env = env.replace("JoystickFlatTerrain", "").replace("-MinAtar", "").replace("-misc", "")
            ax.annotate(
                short_env,
                (xv, yv),
                fontsize=6.5,
                alpha=0.78,
                xytext=(3, 3),
                textcoords="offset points",
            )

    rho, p_rho = spearmanr(xs_all, ys_all)
    r_val, p_r = pearsonr(xs_all, ys_all)
    ax.set_title(
        f"{title}\nSpearman ρ = {rho:+.2f} (p={p_rho:.3g}) | Pearson r = {r_val:+.2f} (p={p_r:.3g}) | N = {len(pts)}",
        fontsize=10.5,
        pad=10,
    )
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved {out_pdf} (N={len(pts)}, Spearman ρ={rho:+.3f})")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compile cross-suite Stage 2 results & publication scatters")
    p.add_argument("--out_root", type=str, default="results/pipeline")
    p.add_argument("--suites", nargs="+", default=None, help="Optional subset of suites to compile")
    p.add_argument("--save_dir", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    save_dir = args.save_dir or os.path.join(args.out_root, "cross_suite")
    os.makedirs(save_dir, exist_ok=True)

    rows = load_all_csvs(args.out_root, suites=args.suites)
    if not rows:
        print("No rows loaded; exiting.")
        return

    master_csv = os.path.join(save_dir, "all_suites_summary.csv")
    fieldnames = list(rows[0].keys())
    with open(master_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Master CSV] Wrote {len(rows)} rows -> {master_csv}")

    # 1. Global E(λ) vs TD(λ): r_rot,c and ρ_c scatters
    _plot_diagnostic_scatter(
        rows=rows,
        x_col="diag_r_rot_c_mean",
        x_label="TD(λ) Centered Split-Half Rotation Ratio ($r_{\\mathrm{rot},c}$)",
        y_col="rel_auc_vs_td_lambda",
        y_label="Relative Performance: $E(\\lambda)$ vs. $\\mathrm{TD}(\\lambda)$ (1.0 = Parity)",
        e_method="e_lambda_v_aux",
        td_method="td_lambda",
        eval_mode="global",
        title="All-Suite Cross-Benchmark: $E(\\lambda)$ vs. $\\mathrm{TD}(\\lambda)$ by Rotation Ratio",
        out_pdf=os.path.join(save_dir, "scatter_r_rot_vs_e_lambda_over_td_lambda.pdf"),
    )
    _plot_diagnostic_scatter(
        rows=rows,
        x_col="diag_rho_c_mean",
        x_label="TD(λ) Centered Gradient Aliasing ($\\rho_c$)",
        y_col="rel_auc_vs_td_lambda",
        y_label="Relative Performance: $E(\\lambda)$ vs. $\\mathrm{TD}(\\lambda)$ (1.0 = Parity)",
        e_method="e_lambda_v_aux",
        td_method="td_lambda",
        eval_mode="global",
        title="All-Suite Cross-Benchmark: $E(\\lambda)$ vs. $\\mathrm{TD}(\\lambda)$ by Gradient Aliasing",
        out_pdf=os.path.join(save_dir, "scatter_rho_c_vs_e_lambda_over_td_lambda.pdf"),
    )

    # 2. 1-Step E(0) vs TD(0): r_rot,c and ρ_c scatters
    _plot_diagnostic_scatter(
        rows=rows,
        x_col="diag_r_rot_c_mean",
        x_label="TD(0) Centered Split-Half Rotation Ratio ($r_{\\mathrm{rot},c}$)",
        y_col="rel_auc_vs_td0",
        y_label="Relative Performance: $E(0)$ vs. $\\mathrm{TD}(0)$ (1.0 = Parity)",
        e_method="e_0_v_aux",
        td_method="td_0",
        eval_mode="global",
        title="All-Suite Cross-Benchmark: 1-Step $E(0)$ vs. $\\mathrm{TD}(0)$ by Rotation Ratio",
        out_pdf=os.path.join(save_dir, "scatter_r_rot_vs_e0_over_td0.pdf"),
    )

    # 3. Matched-λ (λ = 0.8) scatter across all suites
    _plot_diagnostic_scatter(
        rows=rows,
        x_col="diag_r_rot_c_mean",
        x_label="TD(0.8) Centered Split-Half Rotation Ratio ($r_{\\mathrm{rot},c}$)",
        y_col="rel_auc_vs_matched_td",
        y_label="Relative Performance: $E(0.8)$ vs. $\\mathrm{TD}(0.8)$ (1.0 = Parity)",
        e_method="e_lambda_v_aux",
        td_method="td_lambda",
        eval_mode="per_lambda",
        filter_tag="lam=0.8",
        title="All-Suite Matched-$\\lambda$ ($\\lambda=0.8$): $E(0.8)$ vs. $\\mathrm{TD}(0.8)$ by Rotation Ratio",
        out_pdf=os.path.join(save_dir, "scatter_r_rot_vs_e08_over_td08.pdf"),
    )


if __name__ == "__main__":
    main()

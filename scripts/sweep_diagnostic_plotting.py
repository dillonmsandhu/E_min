"""
scripts/sweep_diagnostic_plotting.py

Generates standardized contraction-loss and return-variance diagnostic plots
and tables at the end of any Gymnax, Brax, or MJX sweep:
  1. Per-environment learning curves for each of the 4 diagnostic measures.
  2. Per-environment AUC bar chart + tables for all 4 measures.
  3. Benchmark-average AUC bar chart across all environments.
  4. 2x2 scatter grid: AUC of each diagnostic measure (x-axis) vs. relative
     performance of E(lambda) / TD(lambda) at fixed matched lambda (y-axis).
"""

from __future__ import annotations

import os
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


PRIMARY_DIAG_SPECS: list[dict[str, Any]] = [
    {
        "key": "diag_rho_c",
        "title": "1. Aliasing — Centered Gradient Aliasing (ρ_c of v)",
        "short": "Aliasing ρ_c (v)",
        "ylabel": "Centered Aliasing ρ_c (main critic v)",
        "ref_line": 0.85,
        "ref_label": "High aliasing (0.85)",
        "scatter_source": "baseline",
        "expected_sign": "+",
    },
    {
        "key": "diag_r_rot_c",
        "title": "2. Rotation — Centered Split-Half Ratio (r_rot,c of v)",
        "short": "Rotation r_rot,c (v)",
        "ylabel": "Rotation Ratio r_rot,c (main critic v)",
        "ref_line": 0.25,
        "ref_label": "Gate threshold τ=0.25",
        "scatter_source": "baseline",
        "expected_sign": "+",
    },
    {
        "key": "diag_ret_var_ratio",
        "title": "3. Variance — Return-to-TD Ratio (VR_ret of v)",
        "short": "Return Variance VR_ret (v)",
        "ylabel": "E[(G−v)²] / E[δ²] (main critic v)",
        "ref_line": 1.0,
        "ref_label": "1-step parity (1.0)",
        "scatter_source": "baseline",
        "expected_sign": "-",
    },
    {
        "key": "diag_v_r2",
        "title": "4. Main Critic Accuracy — R²(v) (Higher is Better)",
        "short": "Main Critic R²(v)",
        "ylabel": "R²(v) vs. Rollout Return (1.0 = perfect)",
        "ref_line": 0.0,
        "ref_label": "Mean baseline (0.0)",
        "scatter_source": "baseline",
        "expected_sign": "-",
    },
]

AUX_DIAG_SPECS: list[dict[str, Any]] = [
    {
        "key": "diag_aux_r2",
        "title": "v_aux Accuracy — R²(v_aux) (Higher is Better)",
        "short": "v_aux R²",
        "ylabel": "R²(v_aux) vs. G^aux (1.0 = perfect)",
        "ref_line": 0.0,
        "ref_label": "Mean baseline (0.0)",
        "scatter_source": "e_method",
        "expected_sign": "+",
    },
    {
        "key": "diag_aux_vs_v",
        "title": "v_aux vs. v Skill (> 0: v_aux better than v)",
        "short": "v_aux vs. v (>0: better)",
        "ylabel": "1 − MSE(v_aux)/MSE(v)  [>0: v_aux better]",
        "ref_line": 0.0,
        "ref_label": "> 0: v_aux better than v",
        "scatter_source": "e_method",
        "expected_sign": "+",
    },
]

DIAG_SPECS: list[dict[str, Any]] = PRIMARY_DIAG_SPECS + AUX_DIAG_SPECS


def _smooth_1d(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _extract_run_lambda(params: dict[str, Any]) -> float | None:
    for k in ("VALUE_LAMBDA", "E_LAMBDA", "BSMC_LAMBDA", "GAE_LAMBDA"):
        if k in params and params[k] is not None:
            return float(params[k])
    return None


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _active_diag_specs(all_runs: dict, envs: list[str], method_keys: list[str]) -> list[dict[str, Any]]:
    present = set()
    for env in envs:
        for m_key in method_keys:
            for res in all_runs.get(env, {}).get(m_key, {}).values():
                for spec in DIAG_SPECS:
                    if spec["key"] in res:
                        present.add(spec["key"])
                if "diag_aux_unexp_var" in res:
                    present.add("diag_v_r2")
                    present.add("diag_aux_r2")
    return [spec for spec in DIAG_SPECS if spec["key"] in present]


def _has_any_diagnostics(all_runs: dict, envs: list[str], method_keys: list[str]) -> bool:
    return len(_active_diag_specs(all_runs, envs, method_keys)) > 0


def _metric_auc_stats(res: dict[str, Any], key: str, method_key: str | None = None) -> tuple[float, float]:
    arr_raw = res.get(key)
    # Backwards-compat for older pickles that logged overloaded `diag_aux_unexp_var`
    if arr_raw is None and "diag_aux_unexp_var" in res and method_key is not None:
        is_non_aux = method_key in ("td_lambda", "td_0", "sarsa_lambda", "msbe", "bsmc_fwd")
        if key == "diag_v_r2" and is_non_aux:
            arr_raw = 1.0 - np.asarray(res["diag_aux_unexp_var"], dtype=np.float64)
        elif key == "diag_aux_r2" and not is_non_aux:
            arr_raw = 1.0 - np.asarray(res["diag_aux_unexp_var"], dtype=np.float64)
        elif key == "diag_v_r2" and not is_non_aux and "diag_aux_vs_v" in res:
            # Recover 1 - R^2(v) = (1 - R^2(v_aux)) / (1 - aux_vs_v) -> R^2(v) = 1 - (1 - R^2(v_aux)) / (1 - aux_vs_v)
            u_aux = np.asarray(res["diag_aux_unexp_var"], dtype=np.float64)
            s_aux = np.asarray(res["diag_aux_vs_v"], dtype=np.float64)
            denom = np.where(np.abs(1.0 - s_aux) > 1e-6, 1.0 - s_aux, np.nan)
            arr_raw = 1.0 - (u_aux / denom)
    if arr_raw is None:
        return float("nan"), float("nan")
    arr = np.asarray(arr_raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if not np.any(np.isfinite(arr)):
        return float("nan"), float("nan")
    # Mean over updates per seed (ignoring non-finite entries if any)
    seed_aucs = np.array([
        float(np.nanmean(row)) if np.any(np.isfinite(row)) else float("nan")
        for row in arr
    ], dtype=np.float64)
    valid = seed_aucs[np.isfinite(seed_aucs)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(valid)), float(np.std(valid))


def _plot_diagnostic_learning_curves(
    all_runs: dict,
    envs: list[str],
    method_keys: list[str],
    per_env_best_tags: dict[tuple[str, str], str],
    batch_sizes: dict[str, int],
    spec: dict[str, Any],
    suite_title: str,
    diag_plots_dir: str,
    get_algo_meta,
    env_titles: dict[str, str],
) -> None:
    key = spec["key"]
    n_envs = len(envs)
    n_cols = min(3, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.4 * n_rows), squeeze=False)

    any_plotted = False
    for idx, env in enumerate(envs):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        bs = float(batch_sizes.get(env, 1))
        for m_key in method_keys:
            tag = per_env_best_tags.get((env, m_key))
            if not tag or tag not in all_runs.get(env, {}).get(m_key, {}):
                continue
            res = all_runs[env][m_key][tag]
            if key not in res:
                continue
            arr = np.asarray(res[key], dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[None, :]
            if not np.any(np.isfinite(arr)):
                continue
            n_upd = arr.shape[-1]
            xs = (np.arange(1, n_upd + 1) * bs) / 1e6
            win = max(1, n_upd // 40)
            mean_c = _smooth_1d(np.nanmean(arr, axis=0), win)
            std_c = _smooth_1d(np.nanstd(arr, axis=0), win)
            meta = get_algo_meta(m_key)
            ax.plot(xs, mean_c, lw=1.9, ls=meta["ls"], color=meta["color"], label=meta["label"])
            if arr.shape[0] > 1:
                ax.fill_between(xs, mean_c - std_c, mean_c + std_c, color=meta["color"], alpha=0.14)
            any_plotted = True

        if spec.get("ref_line") is not None:
            ax.axhline(spec["ref_line"], color="#64748b", ls=":", lw=1.1, alpha=0.8)
        ax.set_title(env_titles.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Env Steps (M)")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=7.5)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    if not any_plotted:
        plt.close(fig)
        return

    fig.suptitle(f"{suite_title}: {spec['title']}", fontsize=12.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    base = os.path.join(diag_plots_dir, f"learning_curves_{key}")
    plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_diagnostic_auc_per_env_and_benchmark(
    all_runs: dict,
    envs: list[str],
    method_keys: list[str],
    per_env_best_tags: dict[tuple[str, str], str],
    suite_title: str,
    diag_plots_dir: str,
    tables_dir: str,
    get_algo_meta,
    env_titles: dict[str, str],
) -> None:
    n_envs = len(envs)
    n_methods = len(method_keys)
    env_labels = [env_titles.get(e, e) for e in envs]
    active_specs = _active_diag_specs(all_runs, envs, method_keys)

    # Collect per-env AUC means/stds: dict[key] -> (n_envs, n_methods)
    auc_means: dict[str, np.ndarray] = {}
    auc_stds: dict[str, np.ndarray] = {}

    for spec in active_specs:
        key = spec["key"]
        mat_m = np.full((n_envs, n_methods), np.nan, dtype=np.float64)
        mat_s = np.full((n_envs, n_methods), np.nan, dtype=np.float64)
        for i, env in enumerate(envs):
            for j, m_key in enumerate(method_keys):
                tag = per_env_best_tags.get((env, m_key))
                if not tag or tag not in all_runs.get(env, {}).get(m_key, {}):
                    continue
                m_val, s_val = _metric_auc_stats(all_runs[env][m_key][tag], key, method_key=m_key)
                mat_m[i, j] = m_val
                mat_s[i, j] = s_val
        auc_means[key] = mat_m
        auc_stds[key] = mat_s

        # Save per-metric CSV + Markdown table with Benchmark Average row
        rows = []
        for i, env in enumerate(envs):
            row = {"Environment": env_labels[i]}
            for j, m_key in enumerate(method_keys):
                lbl = get_algo_meta(m_key)["label"]
                mv, sv = mat_m[i, j], mat_s[i, j]
                row[lbl] = "N/A" if np.isnan(mv) else f"{mv:.4f} ± {sv:.4f}"
            rows.append(row)
        avg_row = {"Environment": "Benchmark Average"}
        for j, m_key in enumerate(method_keys):
            lbl = get_algo_meta(m_key)["label"]
            col = mat_m[:, j]
            valid = col[~np.isnan(col)]
            if valid.size == 0:
                avg_row[lbl] = "N/A"
            else:
                se = float(np.std(valid) / np.sqrt(valid.size)) if valid.size > 1 else 0.0
                avg_row[lbl] = f"{float(np.mean(valid)):.4f} ± {se:.4f}"
        rows.append(avg_row)

        df = pd.DataFrame(rows)
        t_base = os.path.join(tables_dir, f"diagnostic_auc_{key}")
        df.to_csv(t_base + ".csv", index=False)
        with open(t_base + ".md", "w") as f:
            f.write(f"### {suite_title}: {spec['title']} (AUC)\n\n")
            f.write(df.to_markdown(index=False) + "\n")

    # 1. Per-environment AUC grouped bar chart (2x2 or 2x3 grid)
    n_specs = len(active_specs)
    n_cols_grid = 2 if n_specs <= 4 else 3
    n_rows_grid = int(np.ceil(n_specs / n_cols_grid))
    fig, axes = plt.subplots(
        n_rows_grid,
        n_cols_grid,
        figsize=(max(5.5 * n_cols_grid, (0.7 * n_envs + 3.0) * n_cols_grid), 4.1 * n_rows_grid),
        squeeze=False,
    )
    x = np.arange(n_envs)
    width = 0.78 / max(n_methods, 1)

    for idx, spec in enumerate(active_specs):
        ax = axes[idx // n_cols_grid][idx % n_cols_grid]
        key = spec["key"]
        mat_m = auc_means[key]
        mat_s = auc_stds[key]
        for j, m_key in enumerate(method_keys):
            meta = get_algo_meta(m_key)
            offset = (j - (n_methods - 1) / 2.0) * width
            vals = np.nan_to_num(mat_m[:, j], nan=0.0)
            errs = np.nan_to_num(mat_s[:, j], nan=0.0)
            ax.bar(
                x + offset,
                vals,
                width=width * 0.92,
                yerr=errs,
                capsize=2.0,
                color=meta["color"],
                alpha=0.88,
                label=meta["label"],
                edgecolor="#222222",
                linewidth=0.4,
            )
        if spec.get("ref_line") is not None:
            ax.axhline(spec["ref_line"], color="#64748b", ls=":", lw=1.1, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(env_labels, rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel(spec["ylabel"], fontsize=9.0)
        ax.set_title(spec["title"], fontsize=10.5, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=7.5, ncol=2)
    for idx in range(n_specs, n_rows_grid * n_cols_grid):
        axes[idx // n_cols_grid][idx % n_cols_grid].axis("off")

    fig.suptitle(f"{suite_title}: Per-Environment Diagnostic AUCs", fontsize=13.0, fontweight="bold", y=1.01)
    plt.tight_layout()
    base_env = os.path.join(diag_plots_dir, "diagnostic_auc_per_env")
    plt.savefig(base_env + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(base_env + ".pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Benchmark-average AUC bar chart (1 x n_specs grid)
    fig, axes = plt.subplots(1, n_specs, figsize=(3.9 * n_specs, 3.9), squeeze=False)
    axes_flat = axes[0]
    xm = np.arange(n_methods)
    m_labels = [get_algo_meta(k)["label"] for k in method_keys]
    m_colors = [get_algo_meta(k)["color"] for k in method_keys]

    summary_rows = []
    for idx, spec in enumerate(active_specs):
        ax = axes_flat[idx]
        key = spec["key"]
        mat_m = auc_means[key]
        means, ses = [], []
        row = {"Diagnostic": spec["title"]}
        for j, m_key in enumerate(method_keys):
            valid = mat_m[:, j][~np.isnan(mat_m[:, j])]
            if valid.size == 0:
                mv, se = np.nan, 0.0
                row[m_labels[j]] = "N/A"
            else:
                mv = float(np.mean(valid))
                se = float(np.std(valid) / np.sqrt(valid.size)) if valid.size > 1 else 0.0
                row[m_labels[j]] = f"{mv:.4f} ± {se:.4f}"
            means.append(0.0 if np.isnan(mv) else mv)
            ses.append(se)
        summary_rows.append(row)

        bars = ax.bar(xm, means, yerr=ses, capsize=3.0, color=m_colors, alpha=0.88, edgecolor="#222222", linewidth=0.5)
        for bar, mv in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{mv:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        if spec.get("ref_line") is not None:
            ax.axhline(spec["ref_line"], color="#64748b", ls=":", lw=1.1, alpha=0.8)
        ax.set_xticks(xm)
        ax.set_xticklabels(m_labels, rotation=32, ha="right", fontsize=8.0)
        ax.set_ylabel(spec["ylabel"], fontsize=8.5)
        ax.set_title(spec["short"], fontsize=10.5, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{suite_title}: Benchmark-Average Diagnostic AUCs (Mean ± SE across Envs)", fontsize=12.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    base_avg = os.path.join(diag_plots_dir, "diagnostic_benchmark_avg_auc")
    plt.savefig(base_avg + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(base_avg + ".pdf", bbox_inches="tight")
    plt.close(fig)

    df_sum = pd.DataFrame(summary_rows)
    s_base = os.path.join(tables_dir, "diagnostic_benchmark_summary")
    df_sum.to_csv(s_base + ".csv", index=False)
    with open(s_base + ".md", "w") as f:
        f.write(f"### {suite_title}: Benchmark-Average Contraction & Variance Diagnostics\n\n")
        f.write(df_sum.to_markdown(index=False) + "\n")


def _select_best_at_lambda(
    all_runs: dict,
    env: str,
    method_key: str,
    target_lambda: float | None,
) -> dict[str, Any] | None:
    runs = all_runs.get(env, {}).get(method_key, {})
    if not runs:
        return None
    if target_lambda is None:
        return max(runs.values(), key=lambda r: r["auc_mean"])
    matched = [
        r for r in runs.values()
        if (lv := _extract_run_lambda(r.get("params", {}))) is not None and abs(lv - target_lambda) < 1e-6
    ]
    if not matched:
        return None
    return max(matched, key=lambda r: r["auc_mean"])


def _plot_2x2_scatter_for_pair(
    all_runs: dict,
    envs: list[str],
    e_key: str,
    td_key: str,
    target_lambda: float | None,
    lam_label: str,
    suite_title: str,
    diag_plots_dir: str,
    get_algo_meta,
    env_titles: dict[str, str],
) -> None:
    e_lbl = get_algo_meta(e_key)["label"]
    td_lbl = get_algo_meta(td_key)["label"]

    # Primary 2x2 scatter grid uses ONLY main-critic PRIMARY_DIAG_SPECS (all measured on td_key's main critic v)
    present_keys = {s["key"] for s in _active_diag_specs(all_runs, envs, [e_key, td_key])}
    active_specs = [s for s in PRIMARY_DIAG_SPECS if s["key"] in present_keys]
    if not active_specs:
        return

    # Gather matched per-env data
    records = []
    for env in envs:
        r_e = _select_best_at_lambda(all_runs, env, e_key, target_lambda)
        r_td = _select_best_at_lambda(all_runs, env, td_key, target_lambda)
        if r_e is None or r_td is None:
            continue
        auc_e = float(r_e["auc_mean"])
        auc_td = float(r_td["auc_mean"])
        denom = max(abs(auc_td), 1e-8)
        rel_gain_pct = 100.0 * (auc_e - auc_td) / denom

        rec: dict[str, Any] = {
            "env": env,
            "label": env_titles.get(env, env),
            "rel_gain_pct": rel_gain_pct,
        }
        for spec in active_specs:
            key = spec["key"]
            if spec["scatter_source"] == "e_method":
                val, _ = _metric_auc_stats(r_e, key, method_key=e_key)
                if np.isnan(val):
                    val, _ = _metric_auc_stats(r_td, key, method_key=td_key)
            else:
                val, _ = _metric_auc_stats(r_td, key, method_key=td_key)
                if np.isnan(val):
                    val, _ = _metric_auc_stats(r_e, key, method_key=e_key)
            rec[key] = val
        records.append(rec)

    if len(records) < 2:
        return

    n_specs = len(active_specs)
    n_cols = 2 if n_specs <= 4 else 3
    n_rows = int(np.ceil(n_specs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 4.6 * n_rows), squeeze=False)
    ys_all = np.array([r["rel_gain_pct"] for r in records], dtype=np.float64)

    for idx, spec in enumerate(active_specs):
        ax = axes[idx // n_cols][idx % n_cols]
        key = spec["key"]
        xs = np.array([r[key] for r in records], dtype=np.float64)
        valid = np.isfinite(xs) & np.isfinite(ys_all)
        xv, yv = xs[valid], ys_all[valid]
        lbls = [records[i]["label"] for i in range(len(records)) if valid[i]]

        ax.axhline(0.0, color="#334155", ls="--", lw=1.1, alpha=0.8)
        if spec.get("ref_line") is not None:
            ax.axvline(spec["ref_line"], color="#94a3b8", ls=":", lw=1.0, alpha=0.8)

        if xv.size > 0:
            use_symlog = bool(np.max(np.abs(yv)) > 350.0)
            if use_symlog:
                ax.set_yscale("symlog", linthresh=25.0, linscale=1.2)

            # Set natural x-axis bounds for bounded metrics without clipping any data point
            xmin_d, xmax_d = float(np.min(xv)), float(np.max(xv))
            if key == "diag_rho_c":
                ax.set_xlim(min(0.0, xmin_d - 0.05), max(1.02, xmax_d + 0.03))
            elif key == "diag_r_rot_c":
                ax.set_xlim(0.0, max(1.0, xmax_d * 1.10))
            elif key == "diag_ret_var_ratio":
                ax.set_xlim(0.0, max(1.0, xmax_d * 1.08))
            elif key in ("diag_v_r2", "diag_aux_r2", "diag_aux_vs_v"):
                ax.set_xlim(min(-1.0, xmin_d - 0.15), 1.05)

            colors = [
                "#16a34a" if y > 1.0 else ("#dc2626" if y < -1.0 else "#64748b")
                for y in yv
            ]
            ax.scatter(xv, yv, s=72, c=colors, edgecolors="#111827", linewidths=0.7, zorder=4)

            # Simple greedy vertical offset staggering to reduce label overlap
            x_span = max(float(np.max(xv) - np.min(xv)), 1e-8)
            y_trans = np.sign(yv) * np.log1p(np.abs(yv) / 25.0) if use_symlog else yv
            y_span = max(float(np.max(y_trans) - np.min(y_trans)), 1e-8)
            order = np.argsort(xv)
            placed: list[tuple[float, float]] = []
            for oi in order:
                xi, yi, txt = float(xv[oi]), float(yv[oi]), lbls[oi]
                xn = xi / x_span
                yn = float(y_trans[oi]) / y_span
                dy = 5
                dx = 5
                for px, py in placed:
                    if abs(xn - px) < 0.16 and abs(yn - py) < 0.08:
                        dy = -11 if dy > 0 else dy - 10
                placed.append((xn, yn))
                ax.annotate(
                    txt,
                    (xi, yi),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=7.8,
                    color="#1e293b",
                )

            if xv.size >= 3 and np.std(xv) > 1e-12:
                slope, intercept = np.polyfit(xv, yv, 1)
                x_line = np.linspace(float(np.min(xv)), float(np.max(xv)), 80)
                ax.plot(x_line, slope * x_line + intercept, color="#2563eb", lw=1.4, ls="-", alpha=0.7, zorder=3)

            r_p = _pearson_corr(xv, yv)
            r_s = _spearman_corr(xv, yv)
            stats_txt = f"Pearson r = {r_p:+.2f}\nSpearman ρ = {r_s:+.2f}\n(n = {xv.size})"
            ax.text(
                0.03,
                0.96,
                stats_txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8.2,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.92),
            )

        src_note = f"measured on {e_lbl}" if spec["scatter_source"] == "e_method" else f"measured on {td_lbl}"
        ax.set_xlabel(f"{spec['short']} ({src_note})", fontsize=9.5)
        ax.set_ylabel("E / TD Performance Gain (%)", fontsize=9.5)
        ax.set_title(spec["title"], fontsize=10.5, fontweight="bold")
        ax.grid(True, alpha=0.3)

    for idx in range(n_specs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        f"{suite_title}: Diagnostic AUCs vs. Relative Performance ({e_lbl} vs. {td_lbl}, {lam_label})",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5,
        -0.015,
        f"Y-axis: relative AUC gain ({e_lbl} − {td_lbl}) / |{td_lbl}| × 100%  (> 0%: {e_lbl} wins; < 0%: {td_lbl} wins).",
        ha="center",
        fontsize=8.5,
        color="#475569",
        style="italic",
    )
    plt.tight_layout()
    fname = f"scatter_2x2_vs_rel_perf_{lam_label.replace('=', '').replace('.', 'p').replace(' ', '_')}"
    base = os.path.join(diag_plots_dir, fname)
    plt.savefig(base + ".png", bbox_inches="tight", dpi=300)
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def generate_contraction_diagnostics_suite(
    all_runs: dict,
    envs: list[str],
    method_keys: list[str],
    reference_method: str,
    per_env_best_tags: dict[tuple[str, str], str],
    batch_sizes: dict[str, int],
    suite_title: str,
    plots_dir: str,
    tables_dir: str,
    get_algo_meta,
    env_titles: dict[str, str],
    scatter_target_lambda: float = 0.8,
) -> None:
    """Entry point called from `generate_standardized_suite`."""
    if not _has_any_diagnostics(all_runs, envs, method_keys):
        return

    diag_plots_dir = os.path.join(plots_dir, "diagnostics")
    os.makedirs(diag_plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Learning curves for each of the 4 diagnostic measures
    for spec in DIAG_SPECS:
        _plot_diagnostic_learning_curves(
            all_runs=all_runs,
            envs=envs,
            method_keys=method_keys,
            per_env_best_tags=per_env_best_tags,
            batch_sizes=batch_sizes,
            spec=spec,
            suite_title=suite_title,
            diag_plots_dir=diag_plots_dir,
            get_algo_meta=get_algo_meta,
            env_titles=env_titles,
        )

    # 2. Per-env AUC bar chart + Benchmark-average AUC bar chart + tables
    _plot_diagnostic_auc_per_env_and_benchmark(
        all_runs=all_runs,
        envs=envs,
        method_keys=method_keys,
        per_env_best_tags=per_env_best_tags,
        suite_title=suite_title,
        diag_plots_dir=diag_plots_dir,
        tables_dir=tables_dir,
        get_algo_meta=get_algo_meta,
        env_titles=env_titles,
    )

    # 3. 2x2 Scatter grid: E(lambda) / TD(lambda) at fixed lambda (default 0.8 or closest)
    if reference_method in method_keys and "e_lambda_v_aux" in method_keys:
        # Discover available matched lambdas across envs
        lam_set = set()
        for env in envs:
            for m_key in (reference_method, "e_lambda_v_aux"):
                for res in all_runs.get(env, {}).get(m_key, {}).values():
                    lv = _extract_run_lambda(res.get("params", {}))
                    if lv is not None:
                        lam_set.add(round(lv, 6))
        if lam_set:
            chosen_lam = min(lam_set, key=lambda x: abs(x - scatter_target_lambda))
            _plot_2x2_scatter_for_pair(
                all_runs=all_runs,
                envs=envs,
                e_key="e_lambda_v_aux",
                td_key=reference_method,
                target_lambda=chosen_lam,
                lam_label=f"lam={chosen_lam:g}",
                suite_title=suite_title,
                diag_plots_dir=diag_plots_dir,
                get_algo_meta=get_algo_meta,
                env_titles=env_titles,
            )

    # Also generate 2x2 scatter for E(0) vs TD(0) when both are present
    if "td_0" in method_keys and "e_0_v_aux" in method_keys:
        _plot_2x2_scatter_for_pair(
            all_runs=all_runs,
            envs=envs,
            e_key="e_0_v_aux",
            td_key="td_0",
            target_lambda=None,
            lam_label="lam=0",
            suite_title=suite_title,
            diag_plots_dir=diag_plots_dir,
            get_algo_meta=get_algo_meta,
            env_titles=env_titles,
        )

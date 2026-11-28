#!/usr/bin/env python3
"""
scripts/pipeline/select_configs.py

Stage 1 -> Stage 2 bridge: reads all Stage 1 tuning pickles for a suite,
selects ONE global winning hyperparameter configuration per algorithm
(maximizing suite-wide IQM of normalized AUC return), and also selects
per-lambda global winners for `td_lambda` and `e_lambda`.

Writes JSON configs to:
  <configs_dir>/<suite>/<algo>.json
  <configs_dir>/<suite>/<algo>_lam<lam>.json
and a human-readable summary to:
  <configs_dir>/<suite>/selection_summary.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np

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

from scripts.pipeline.suite_registry import (
    SUITES,
    TUNED_BASE_ALGOS,
    get_suite_spec,
    tuned_config_path,
)
from scripts.sweep_standard_runner import _method_lambda_key


def compute_iqm(values: list[float] | np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.size
    if n == 0:
        return float("nan")
    if n < 4:
        return float(arr.mean())
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(arr[lo:hi].mean())


def load_stage1_runs(out_root: str, suite: str) -> dict[str, dict[str, dict[str, Any]]]:
    pattern = os.path.join(out_root, suite, "stage1_tune", "*", "*", "sweep_metrics_all.pkl")
    pkls = sorted(glob.glob(pattern))
    if not pkls:
        raise FileNotFoundError(f"No Stage 1 pickles found matching: {pattern}")

    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        for env, env_dict in payload.get("runs", {}).items():
            merged.setdefault(env, {})
            for algo, tag_dict in env_dict.items():
                merged[env].setdefault(algo, {})
                merged[env][algo].update(tag_dict)
    return merged


def compute_env_scales(
    merged_runs: dict[str, dict[str, dict[str, Any]]],
    env_scales_file: str | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Returns per-environment (low, high) reference bounds for normalizing AUC returns
    to [0, 1] before computing suite-wide IQM.
    If `env_scales_file` exists, uses fixed published bounds; otherwise uses the
    global observed [min, max] across ALL Stage 1 algorithms on that environment
    so every algorithm is scored on the exact same per-environment scale.
    """
    if env_scales_file and os.path.exists(env_scales_file):
        with open(env_scales_file, "r") as f:
            raw = json.load(f)
        return {env: (float(v["low"]), float(v["high"])) for env, v in raw.items()}

    scales: dict[str, tuple[float, float]] = {}
    for env, env_dict in merged_runs.items():
        all_aucs = [
            float(res["auc_mean"])
            for tag_dict in env_dict.values()
            for res in tag_dict.values()
            if np.isfinite(res["auc_mean"])
        ]
        if not all_aucs:
            raise ValueError(f"No finite AUC values found for env '{env}'.")
        lo = float(min(all_aucs))
        hi = float(max(all_aucs))
        if hi - lo < 1e-8:
            hi = lo + 1.0
        scales[env] = (lo, hi)
    return scales


def rank_all_tags(
    merged_runs: dict[str, dict[str, dict[str, Any]]],
    envs: list[str],
    algo: str,
    scales: dict[str, tuple[float, float]],
    filter_lambda: float | None = None,
    exclude_mc_lambda: bool = False,
    filter_max_grad_norm: float | None = None,
) -> list[dict[str, Any]]:
    lkey = _method_lambda_key(algo)
    # Find tags present across ALL tuning envs
    tag_sets = []
    for env in envs:
        env_tags = merged_runs.get(env, {}).get(algo, {})
        valid_tags = set()
        for tag, res in env_tags.items():
            if filter_max_grad_norm is not None:
                gn = res.get("params", {}).get("MAX_GRAD_NORM")
                if gn is not None and abs(float(gn) - float(filter_max_grad_norm)) > 1e-6:
                    continue
            if filter_lambda is not None:
                if lkey is None:
                    continue
                lval = res.get("params", {}).get(lkey)
                if lval is None or abs(float(lval) - float(filter_lambda)) > 1e-6:
                    continue
            elif exclude_mc_lambda and lkey is not None:
                lval = res.get("params", {}).get(lkey)
                if lval is not None and float(lval) >= 1.0 - 1e-6:
                    continue
            valid_tags.add(tag)
        tag_sets.append(valid_tags)

    if not tag_sets:
        return []
    common_tags = sorted(set.intersection(*tag_sets))
    if not common_tags:
        return []

    ranked: list[dict[str, Any]] = []
    for tag in common_tags:
        norm_scores = []
        per_env_auc = {}
        for env in envs:
            res = merged_runs[env][algo][tag]
            auc = float(res["auc_mean"])
            lo, hi = scales[env]
            norm_scores.append((auc - lo) / (hi - lo))
            per_env_auc[env] = auc

        iqm = compute_iqm(norm_scores)
        mean_norm = float(np.mean(norm_scores))
        sample_res = merged_runs[envs[0]][algo][tag]
        ranked.append(
            {
                "tag": tag,
                "params": dict(sample_res["params"]),
                "iqm_norm_auc": float(iqm),
                "mean_norm_auc": float(mean_norm),
                "per_env_auc": per_env_auc,
            }
        )

    ranked.sort(key=lambda e: (e["iqm_norm_auc"], e["mean_norm_auc"]), reverse=True)
    return ranked


def select_best_tag(
    merged_runs: dict[str, dict[str, dict[str, Any]]],
    envs: list[str],
    algo: str,
    scales: dict[str, tuple[float, float]],
    filter_lambda: float | None = None,
    filter_max_grad_norm: float | None = None,
) -> dict[str, Any] | None:
    ranked = rank_all_tags(
        merged_runs,
        envs,
        algo,
        scales,
        filter_lambda=filter_lambda,
        filter_max_grad_norm=filter_max_grad_norm,
    )
    return ranked[0] if ranked else None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select global suite-wide configs from Stage 1 sweeps")
    p.add_argument("--suite", type=str, required=True, choices=sorted(SUITES.keys()))
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--envs", nargs="+", default=None, help="Override tuning envs (defaults to spec.tune_envs)")
    p.add_argument("--lambdas", nargs="+", type=float, default=None)
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Restrict selection to this MAX_GRAD_NORM (default: 1.0; pass <=0 to allow all)",
    )
    p.add_argument("--out_root", type=str, default="results/pipeline")
    p.add_argument("--configs_dir", type=str, default="configs/tuned")
    p.add_argument("--env_scales_file", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_suite_spec(args.suite)
    algos = list(args.algos) if args.algos else list(TUNED_BASE_ALGOS)
    lambdas = list(args.lambdas) if args.lambdas else list(spec.lambdas)

    merged_runs = load_stage1_runs(args.out_root, args.suite)
    expected_envs = list(args.envs) if args.envs else list(spec.tune_envs)
    present_envs = [e for e in expected_envs if e in merged_runs]
    if not present_envs:
        raise RuntimeError(f"None of the expected tuning envs {expected_envs} found in Stage 1 runs.")
    if len(present_envs) < len(expected_envs):
        missing = sorted(set(expected_envs) - set(present_envs))
        print(f"[Warning] Missing Stage 1 envs for {args.suite}: {missing}. Selecting over {present_envs}.")

    scales_file = args.env_scales_file or os.path.join("configs", "env_scales", f"{args.suite}.json")
    scales = compute_env_scales(merged_runs, env_scales_file=scales_file)

    suite_cfg_dir = os.path.join(args.configs_dir, args.suite)
    os.makedirs(suite_cfg_dir, exist_ok=True)
    mirror_cfg_dir = os.path.join(args.out_root, args.suite, "selected_configs")
    os.makedirs(mirror_cfg_dir, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()

    summary_lines = [
        f"# Stage 1 Global Config Selection — `{args.suite}`",
        "",
        f"- **Generated:** `{now_iso}`",
        f"- **Tuning Envs ({len(present_envs)}):** `{', '.join(present_envs)}`",
        "- **Selection Metric:** Suite-wide IQM of normalized AUC return (tie-break: Mean normalized AUC)",
        "",
        "## Global Winners (1 Config per Algorithm)",
        "",
        "| Algorithm | Winning Tag | Params | Suite IQM (Norm) | Suite Mean (Norm) |",
        "|:---|:---|:---|:---:|:---:|",
    ]

    per_lambda_rows: list[str] = []
    full_leaderboard_sections: list[str] = []

    filter_gn = float(args.max_grad_norm) if args.max_grad_norm is not None and args.max_grad_norm > 0 else None

    for algo in algos:
        # Global winners for TD(λ) and E(λ) restrict strictly to bootstrapped λ < 1.0;
        # λ = 1.0 is exported separately as the dedicated `mc` baseline below.
        exclude_mc = _method_lambda_key(algo) is not None
        ranked_all = rank_all_tags(
            merged_runs,
            present_envs,
            algo,
            scales,
            filter_lambda=None,
            exclude_mc_lambda=exclude_mc,
            filter_max_grad_norm=filter_gn,
        )
        if not ranked_all:
            print(f"[Warning] No complete Stage 1 grid found for algo='{algo}' across {present_envs}; skipping.")
            continue
        winner = ranked_all[0]

        payload = {
            "suite": args.suite,
            "algo": algo,
            "lambda_filter": None,
            "tag": winner["tag"],
            "params": winner["params"],
            "metadata": {
                "generated_utc": now_iso,
                "tuning_envs": present_envs,
                "iqm_norm_auc": winner["iqm_norm_auc"],
                "mean_norm_auc": winner["mean_norm_auc"],
                "per_env_auc": winner["per_env_auc"],
            },
        }
        out_path = tuned_config_path(args.configs_dir, args.suite, algo, lambda_val=None)
        for target_path in (out_path, os.path.join(mirror_cfg_dir, os.path.basename(out_path))):
            with open(target_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[Selected] {out_path} -> {winner['tag']} (IQM={winner['iqm_norm_auc']:.4f})")
        summary_lines.append(
            f"| `{algo}` | `{winner['tag']}` | `{winner['params']}` | "
            f"{winner['iqm_norm_auc']:.4f} | {winner['mean_norm_auc']:.4f} |"
        )

        # Also export per-lambda global winners for lambda-parameterized algorithms (bootstrapped λ < 1.0)
        lam_winner_tags: dict[str, float] = {}
        if _method_lambda_key(algo) is not None:
            for lmb in lambdas:
                if float(lmb) >= 1.0 - 1e-6:
                    continue
                lw = select_best_tag(
                    merged_runs, present_envs, algo, scales, filter_lambda=lmb, filter_max_grad_norm=filter_gn
                )
                if lw is None:
                    continue
                lam_winner_tags[lw["tag"]] = float(lmb)
                l_payload = {
                    "suite": args.suite,
                    "algo": algo,
                    "lambda_filter": float(lmb),
                    "tag": lw["tag"],
                    "params": lw["params"],
                    "metadata": {
                        "generated_utc": now_iso,
                        "tuning_envs": present_envs,
                        "iqm_norm_auc": lw["iqm_norm_auc"],
                        "mean_norm_auc": lw["mean_norm_auc"],
                        "per_env_auc": lw["per_env_auc"],
                    },
                }
                l_path = tuned_config_path(args.configs_dir, args.suite, algo, lambda_val=lmb)
                for target_path in (l_path, os.path.join(mirror_cfg_dir, os.path.basename(l_path))):
                    with open(target_path, "w") as f:
                        json.dump(l_payload, f, indent=2, sort_keys=True)
                print(f"[Selected] {l_path} -> {lw['tag']} (IQM={lw['iqm_norm_auc']:.4f})")
                per_lambda_rows.append(
                    f"| `{algo}` | `{lmb:g}` | `{lw['tag']}` | `{lw['params']}` | "
                    f"{lw['iqm_norm_auc']:.4f} | {lw['mean_norm_auc']:.4f} |"
                )

        # Build full ranked leaderboard of ALL options tuned for this algo
        sec_lines = [
            f"### `{algo}` — All Tuned Options ({len(ranked_all)} configs, ranked by Suite IQM)",
            "",
            "| Rank | Status | Tag | Params | Suite IQM (Norm) | Suite Mean (Norm) |",
            "|:---:|:---|:---|:---|:---:|:---:|",
        ]
        for idx, entry in enumerate(ranked_all, start=1):
            badges = []
            if entry["tag"] == winner["tag"]:
                badges.append("**★ GLOBAL**")
            if entry["tag"] in lam_winner_tags:
                badges.append(f"◆ λ={lam_winner_tags[entry['tag']]:g} best")
            status_str = " / ".join(badges) if badges else ""
            sec_lines.append(
                f"| {idx} | {status_str} | `{entry['tag']}` | `{entry['params']}` | "
                f"{entry['iqm_norm_auc']:.4f} | {entry['mean_norm_auc']:.4f} |"
            )
        full_leaderboard_sections.append("\n".join(sec_lines))

    # Export dedicated `mc` (λ=1.0) baseline extracted from `td_lambda` Stage 1 runs (zero extra GPU work)
    if "td_lambda" in algos:
        mc_ranked = rank_all_tags(
            merged_runs, present_envs, "td_lambda", scales, filter_lambda=1.0, filter_max_grad_norm=filter_gn
        )
        if mc_ranked:
            mc_winner = mc_ranked[0]
            mc_params = dict(mc_winner["params"])
            mc_params["VALUE_LAMBDA"] = 1.0
            mc_payload = {
                "suite": args.suite,
                "algo": "mc",
                "lambda_filter": 1.0,
                "tag": mc_winner["tag"],
                "params": mc_params,
                "metadata": {
                    "generated_utc": now_iso,
                    "tuning_envs": present_envs,
                    "iqm_norm_auc": mc_winner["iqm_norm_auc"],
                    "mean_norm_auc": mc_winner["mean_norm_auc"],
                    "per_env_auc": mc_winner["per_env_auc"],
                    "extracted_from": "td_lambda (λ=1.0)",
                },
            }
            mc_path = tuned_config_path(args.configs_dir, args.suite, "mc", lambda_val=None)
            for target_path in (mc_path, os.path.join(mirror_cfg_dir, os.path.basename(mc_path))):
                with open(target_path, "w") as f:
                    json.dump(mc_payload, f, indent=2, sort_keys=True)
            print(f"[Selected] {mc_path} -> {mc_winner['tag']} (IQM={mc_winner['iqm_norm_auc']:.4f})")
            summary_lines.append(
                f"| `mc` | `{mc_winner['tag']}` | `{mc_params}` | "
                f"{mc_winner['iqm_norm_auc']:.4f} | {mc_winner['mean_norm_auc']:.4f} |"
            )
            mc_sec = [
                f"### `mc` (λ=1.0, extracted from `td_lambda`) — All Tuned Options ({len(mc_ranked)} configs)",
                "",
                "| Rank | Status | Tag | Params | Suite IQM (Norm) | Suite Mean (Norm) |",
                "|:---:|:---|:---|:---|:---:|:---:|",
            ]
            for idx, entry in enumerate(mc_ranked, start=1):
                badge = "**★ GLOBAL**" if idx == 1 else ""
                mc_sec.append(
                    f"| {idx} | {badge} | `{entry['tag']}` | `{entry['params']}` | "
                    f"{entry['iqm_norm_auc']:.4f} | {entry['mean_norm_auc']:.4f} |"
                )
            full_leaderboard_sections.append("\n".join(mc_sec))

    if per_lambda_rows:
        summary_lines.extend(
            [
                "",
                "## Per-λ Global Winners (Independently Tuned per λ)",
                "",
                "| Algorithm | λ | Winning Tag | Params | Suite IQM (Norm) | Suite Mean (Norm) |",
                "|:---|:---:|:---|:---|:---:|:---:|",
                *per_lambda_rows,
            ]
        )

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = os.path.join(suite_cfg_dir, "selection_summary.md")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"[Summary] Wrote {summary_path}")

    # Write full tuning report (winners + ALL tuned options ranked) inside results/pipeline/<suite>/
    report_lines = [
        summary_text.rstrip(),
        "",
        "---",
        "",
        "## Full Stage 1 Tuning Grid Results (All Options Evaluated)",
        "",
        "\n\n".join(full_leaderboard_sections),
        "",
    ]
    report_path = os.path.join(args.out_root, args.suite, "tuning_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"[Tuning Report] Wrote {report_path}")


if __name__ == "__main__":
    main()

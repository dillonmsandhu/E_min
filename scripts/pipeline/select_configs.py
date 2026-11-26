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


def select_best_tag(
    merged_runs: dict[str, dict[str, dict[str, Any]]],
    envs: list[str],
    algo: str,
    scales: dict[str, tuple[float, float]],
    filter_lambda: float | None = None,
) -> dict[str, Any] | None:
    lkey = _method_lambda_key(algo)
    # Find tags present across ALL tuning envs
    tag_sets = []
    for env in envs:
        env_tags = merged_runs.get(env, {}).get(algo, {})
        valid_tags = set()
        for tag, res in env_tags.items():
            if filter_lambda is not None:
                if lkey is None:
                    continue
                lval = res.get("params", {}).get(lkey)
                if lval is None or abs(float(lval) - float(filter_lambda)) > 1e-6:
                    continue
            valid_tags.add(tag)
        tag_sets.append(valid_tags)

    if not tag_sets:
        return None
    common_tags = sorted(set.intersection(*tag_sets))
    if not common_tags:
        return None

    best_entry: dict[str, Any] | None = None
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
        # Tie-break IQM by mean_norm
        key_score = (iqm, mean_norm)
        if best_entry is None or key_score > best_entry["_sort_key"]:
            sample_res = merged_runs[envs[0]][algo][tag]
            best_entry = {
                "_sort_key": key_score,
                "tag": tag,
                "params": dict(sample_res["params"]),
                "iqm_norm_auc": float(iqm),
                "mean_norm_auc": float(mean_norm),
                "per_env_auc": per_env_auc,
            }

    if best_entry is not None:
        best_entry.pop("_sort_key", None)
    return best_entry


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select global suite-wide configs from Stage 1 sweeps")
    p.add_argument("--suite", type=str, required=True, choices=sorted(SUITES.keys()))
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--envs", nargs="+", default=None, help="Override tuning envs (defaults to spec.tune_envs)")
    p.add_argument("--lambdas", nargs="+", type=float, default=None)
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

    for algo in algos:
        winner = select_best_tag(merged_runs, present_envs, algo, scales, filter_lambda=None)
        if winner is None:
            print(f"[Warning] No complete Stage 1 grid found for algo='{algo}' across {present_envs}; skipping.")
            continue

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
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"[Selected] {out_path} -> {winner['tag']} (IQM={winner['iqm_norm_auc']:.4f})")
        summary_lines.append(
            f"| `{algo}` | `{winner['tag']}` | `{winner['params']}` | "
            f"{winner['iqm_norm_auc']:.4f} | {winner['mean_norm_auc']:.4f} |"
        )

        # Also export per-lambda global winners for lambda-parameterized algorithms
        if _method_lambda_key(algo) is not None:
            for lmb in lambdas:
                lw = select_best_tag(merged_runs, present_envs, algo, scales, filter_lambda=lmb)
                if lw is None:
                    continue
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
                with open(l_path, "w") as f:
                    json.dump(l_payload, f, indent=2, sort_keys=True)
                print(f"[Selected] {l_path} -> {lw['tag']} (IQM={lw['iqm_norm_auc']:.4f})")
                per_lambda_rows.append(
                    f"| `{algo}` | `{lmb:g}` | `{lw['tag']}` | `{lw['params']}` | "
                    f"{lw['iqm_norm_auc']:.4f} | {lw['mean_norm_auc']:.4f} |"
                )

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

    summary_path = os.path.join(suite_cfg_dir, "selection_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[Summary] Wrote {summary_path}")


if __name__ == "__main__":
    main()

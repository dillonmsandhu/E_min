#!/usr/bin/env python3
"""
scripts/pipeline/stage2_consolidate.py

Stage 2 post-processing:
  1. Consolidates per-environment Stage 2 held-out pickles for both:
       - global     (1 suite-wide config per algorithm -> championship profiles & diagnostics)
       - per_lambda (1 suite-wide config per (algo, λ) -> per-λ head-to-head curves & diagnostics)
  2. Runs `consolidate_standardized_sweep` so all standard plots/tables match existing sweeps
     (with ZERO per-environment tuning!).
  3. Exports a clean, flat cross-suite CSV:
       <out_root>/<suite>/stage2_eval/data/cross_suite_summary.csv
     so `compile_cross_suite.py` can plot all environments across all suites in a single scatter!
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pickle
import sys
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
    DIAGNOSTIC_KEYS,
    SUITES,
    TUNED_BASE_ALGOS,
    get_suite_spec,
)
from scripts.sweep_standard_runner import _method_lambda_key, consolidate_standardized_sweep


def _diag_auc_mean(res: dict[str, Any], key: str) -> float:
    if key not in res:
        return float("nan")
    arr = np.asarray(res[key], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size > 0 else float("nan")


def _extract_lambda_val(method_key: str, res: dict[str, Any]) -> float:
    lkey = _method_lambda_key(method_key)
    if lkey is None:
        return 0.0
    params = res.get("params", {})
    if lkey in params and params[lkey] is not None:
        return float(params[lkey])
    return float("nan")


def _load_consolidated_pickle(pkl_path: str) -> dict[str, Any]:
    if not os.path.exists(pkl_path):
        return {}
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def export_cross_suite_csv(suite: str, stage2_root: str) -> str:
    global_pkl = os.path.join(stage2_root, "global", "data", "sweep_metrics_all.pkl")
    per_lam_pkl = os.path.join(stage2_root, "per_lambda", "data", "sweep_metrics_all.pkl")

    global_payload = _load_consolidated_pickle(global_pkl)
    per_lam_payload = _load_consolidated_pickle(per_lam_pkl)

    global_runs = global_payload.get("runs", {})
    per_lam_runs = per_lam_payload.get("runs", {})

    rows: list[dict[str, Any]] = []

    def _collect(runs_dict: dict[str, Any], mode: str) -> None:
        for env, env_data in sorted(runs_dict.items()):
            # Reference baselines for relative ratios
            ref_td_lam_auc = float("nan")
            ref_td_lam_fin = float("nan")
            if "td_lambda" in global_runs.get(env, {}) and "global" in global_runs[env]["td_lambda"]:
                ref_td_lam_auc = float(global_runs[env]["td_lambda"]["global"]["auc_mean"])
                ref_td_lam_fin = float(global_runs[env]["td_lambda"]["global"]["final_mean"])

            ref_td0_auc = float("nan")
            if "td_0" in global_runs.get(env, {}) and "global" in global_runs[env]["td_0"]:
                ref_td0_auc = float(global_runs[env]["td_0"]["global"]["auc_mean"])

            for method, tag_dict in sorted(env_data.items()):
                for tag, res in sorted(tag_dict.items()):
                    returns = np.asarray(res["returned_episode_returns"], dtype=np.float64)
                    n_seeds = int(returns.shape[0]) if returns.ndim >= 1 else 1
                    auc_mean = float(res["auc_mean"])
                    auc_std = float(res["auc_std"])
                    auc_sem = auc_std / max(np.sqrt(n_seeds), 1.0)
                    fin_mean = float(res["final_mean"])
                    fin_std = float(res["final_std"])
                    fin_sem = fin_std / max(np.sqrt(n_seeds), 1.0)
                    lmb = _extract_lambda_val(method, res)

                    # Matched-lambda TD reference if in per_lambda mode
                    matched_td_auc = float("nan")
                    if mode == "per_lambda" and "td_lambda" in env_data and tag in env_data["td_lambda"]:
                        matched_td_auc = float(env_data["td_lambda"][tag]["auc_mean"])

                    def _signed_ratio(val: float, ref: float) -> float:
                        if not (np.isfinite(val) and np.isfinite(ref)) or abs(ref) < 1e-8:
                            return float("nan")
                        # Handle negative-return envs (e.g., Reacher, Pusher, Acrobot, Pendulum)
                        # using standard sign-aware improvement ratio: 1 + (val - ref) / |ref|
                        return 1.0 + (val - ref) / abs(ref)

                    row: dict[str, Any] = {
                        "suite": suite,
                        "env": env,
                        "eval_mode": mode,
                        "method": method,
                        "tag": tag,
                        "lambda_val": lmb,
                        "n_seeds": n_seeds,
                        "auc_mean": auc_mean,
                        "auc_sem": auc_sem,
                        "final_mean": fin_mean,
                        "final_sem": fin_sem,
                        "rel_auc_vs_td_lambda": _signed_ratio(auc_mean, ref_td_lam_auc),
                        "rel_final_vs_td_lambda": _signed_ratio(fin_mean, ref_td_lam_fin),
                        "rel_auc_vs_td0": _signed_ratio(auc_mean, ref_td0_auc),
                        "rel_auc_vs_matched_td": _signed_ratio(auc_mean, matched_td_auc),
                        "critic_lr": res.get("params", {}).get("CRITIC_LR", float("nan")),
                        "critic_epochs": res.get("params", {}).get("CRITIC_EPOCHS", float("nan")),
                        "max_grad_norm": res.get("params", {}).get("MAX_GRAD_NORM", float("nan")),
                    }
                    for dk in DIAGNOSTIC_KEYS:
                        row[f"{dk}_mean"] = _diag_auc_mean(res, dk)
                    rows.append(row)

    _collect(global_runs, "global")
    _collect(per_lam_runs, "per_lambda")

    out_data_dir = os.path.join(stage2_root, "data")
    os.makedirs(out_data_dir, exist_ok=True)
    csv_path = os.path.join(out_data_dir, "cross_suite_summary.csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Cross-Suite CSV] Wrote {len(rows)} rows -> {csv_path}")
    else:
        print(f"[Warning] No rows found to write to {csv_path}")
    return csv_path


def _discover_methods(sub_root: str, fallback: list[str]) -> list[str]:
    pkls = sorted(glob.glob(os.path.join(sub_root, "*", "data", "sweep_metrics_all.pkl")))
    found: list[str] = []
    for p in pkls:
        with open(p, "rb") as f:
            d = pickle.load(f)
        for m in d.get("method_keys", []):
            if m not in found:
                found.append(m)
    return found if found else list(fallback)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2: Consolidate Held-Out Results & Export Cross-Suite CSV")
    p.add_argument("--suite", type=str, required=True, choices=sorted(SUITES.keys()))
    p.add_argument("--algos", nargs="+", default=None)
    p.add_argument("--out_root", type=str, default="results/pipeline")
    p.add_argument("--all_pairs", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_suite_spec(args.suite)
    stage2_root = os.path.join(args.out_root, args.suite, "stage2_eval")

    global_root = os.path.join(stage2_root, "global")
    if os.path.isdir(global_root):
        global_methods = list(args.algos) if args.algos else _discover_methods(global_root, TUNED_BASE_ALGOS)
        print(f"\n[Consolidate Global | {args.suite}] methods={global_methods}")
        consolidate_standardized_sweep(
            out_dir=global_root,
            method_keys=global_methods,
            suite_title=spec.title,
            reference_method=spec.reference_method,
            is_v_ppo=True,
            all_pairs=args.all_pairs,
        )

    per_lam_root = os.path.join(stage2_root, "per_lambda")
    if os.path.isdir(per_lam_root):
        pl_pkls = glob.glob(os.path.join(per_lam_root, "*", "data", "sweep_metrics_all.pkl"))
        if pl_pkls:
            pl_methods = _discover_methods(per_lam_root, ["td_lambda", "e_lambda_v_aux"])
            print(f"\n[Consolidate Per-λ | {args.suite}] methods={pl_methods}")
            consolidate_standardized_sweep(
                out_dir=per_lam_root,
                method_keys=pl_methods,
                suite_title=f"{spec.title} (Per-λ)",
                reference_method=spec.reference_method,
                is_v_ppo=True,
                all_pairs=args.all_pairs,
            )

    export_cross_suite_csv(args.suite, stage2_root)


if __name__ == "__main__":
    main()

"""Shared summary utilities for hyperparameter-tuning sweep scripts."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


def _iter_hp_runs(hp_dict, prefix=()):
    """Yield (hp_tuple, run_data) from flat or nested hyperparameter dicts."""
    if not isinstance(hp_dict, dict):
        return
    if "returned_episode_returns" in hp_dict or "metrics" in hp_dict:
        yield prefix, hp_dict
        return
    for k, v in hp_dict.items():
        if isinstance(k, tuple):
            yield prefix + k, v
        elif isinstance(v, dict):
            yield from _iter_hp_runs(v, prefix + (k,))


def _extract_returns(run_data):
    """Return 2D (seeds, steps) episode returns array, or None."""
    if not isinstance(run_data, dict):
        return None
    if "returned_episode_returns" in run_data:
        arr = np.asarray(run_data["returned_episode_returns"])
        return arr if arr.ndim == 2 else arr[None, :]
    if "metrics" in run_data and isinstance(run_data["metrics"], dict):
        ret = run_data["metrics"].get("returned_episode_returns")
        if ret is not None:
            arr = np.asarray(ret)
            return arr if arr.ndim == 2 else arr[None, :]
    return None


def _fmt_hp(hp_tuple):
    if not hp_tuple:
        return "default"
    return ", ".join(str(x) for x in hp_tuple)


def write_sweep_config_and_winners(
    tables_dir: str,
    all_runs: dict,
    envs: list[str],
    base_config: dict,
    timesteps: dict | None = None,
) -> str:
    """
    Write `<tables_dir>/sweep_config_and_winners.md` containing:
      1. Base config & swept hyperparameter grid
      2. Winning hyperparameter setting per (Environment, Algorithm) by mean AUC.
    """
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, "sweep_config_and_winners.md")

    algos = []
    for env in envs:
        for algo in all_runs.get(env, {}).keys():
            if algo not in algos:
                algos.append(algo)

    with open(out_path, "w") as f:
        f.write("### Base Config & Sweep Grid\n\n")
        for k, v in base_config.items():
            f.write(f"- **{k}:** {v}\n")
        if timesteps:
            ts_str = ", ".join(f"{e}: {timesteps[e]:,}" for e in envs if e in timesteps)
            if ts_str:
                f.write(f"- **Timesteps:** {ts_str}\n")

        f.write("\n### Winning Settings per (Environment, Algorithm) [Selected by Mean AUC]\n\n")
        rows = []
        for env in envs:
            row = {"Environment": env}
            env_runs = all_runs.get(env, {})
            for algo in algos:
                best_auc = -np.inf
                best_fin = np.nan
                best_hp = None
                for hp_tuple, run_data in _iter_hp_runs(env_runs.get(algo, {})):
                    ret = _extract_returns(run_data)
                    if ret is None or ret.size == 0:
                        continue
                    auc = float(np.mean(ret))
                    fin = float(np.mean(ret[:, -1]))
                    if auc > best_auc or best_hp is None:
                        best_auc = auc
                        best_fin = fin
                        best_hp = hp_tuple
                if best_hp is None:
                    row[algo] = "N/A"
                else:
                    row[algo] = f"{_fmt_hp(best_hp)} (Fin: {best_fin:.1f}, AUC: {best_auc:.1f})"
            rows.append(row)

        f.write(pd.DataFrame(rows).to_markdown(index=False))
        f.write("\n")

    print(f"Generated Config & Winners summary: {out_path}")
    return out_path

#!/usr/bin/env python3
"""
sweep_q_ppo_vs_baselines.py

Head-to-head comparison and hyperparameter tuning on Brax continuous control across:
1. td_lambda:        Standard PPO TD(lambda) (brax_rl/algos/ppo.py, CRITIC_TYPE='fitted')
2. q_ppo_e0_hybrid:  Q-PPO E(0) with Bilinear Loss + Hybrid GAE (brax_rl/algos/q_ppo.py)
3. e_lambda_v_aux:   Standard PPO E(lambda) with v_aux (brax_rl/algos/ppo.py, CRITIC_TYPE='e_lambda', E_VARIANT='aux')

Every method starts from the same PPO baseline configuration (brax_config.py / q_ppo_config.py),
then runs hyperparameter tuning (coordinate search, critic_lr sweep, or Cartesian grid) and
outputs best tuned configurations, learning curves, and Markdown/LaTeX/PNG/PDF tables.
"""

from __future__ import annotations
import os
import sys
import json
import copy
import time
import pickle
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_script_dir, "../../.."))
_scripts_dir = os.path.abspath(os.path.join(_script_dir, "../.."))

sys.path = [_repo_root, _scripts_dir, _script_dir] + [
    p for p in sys.path
    if not (os.path.exists(os.path.join(p, "fixed_policy")) or (p == "" and os.path.exists(os.path.join(os.getcwd(), "fixed_policy"))))
]

import jax
from brax_rl.core.brax_config import config as brax_ppo_config
from brax_rl.core.q_ppo_config import config as brax_q_ppo_config
from brax_rl.algos.ppo import make_train as make_ppo_train
from brax_rl.algos.q_ppo import make_train as make_q_ppo_train
from scripts.brax.sweep_ppo_critic.sweep_ppo_critics import (
    ENV_TITLES,
    DEFAULT_ENVS,
    get_env_default_timesteps,
    save_table_bundle,
)

METHODS = [
    ("td_lambda", "TD(λ) (PPO Fitted)"),
    ("q_ppo_e0_hybrid", "Q-PPO E(0) (Bilinear+Hybrid)"),
    ("e_lambda_v_aux", "E(λ) (v_aux PPO)"),
]

METHOD_COLORS = {
    "td_lambda": "#2563eb",
    "q_ppo_e0_hybrid": "#059669",
    "e_lambda_v_aux": "#7c3aed",
}


def build_base_method_config(
    method_key: str,
    env_name: str,
    total_timesteps: int,
    num_envs: int,
    num_steps: int,
    minibatch_size: int,
) -> dict:
    """Starts from PPO defaults (or q_ppo_config initialized from PPO defaults)."""
    if method_key == "q_ppo_e0_hybrid":
        cfg = copy.deepcopy(brax_q_ppo_config)
        cfg["CRITIC_TYPE"] = "e_0"
        cfg["GAE_CONTINUOUS_MODE"] = "hybrid"
        cfg["K_ACTIONS"] = 2
        cfg["RETURN_LAMBDA"] = 1.0
    elif method_key == "td_lambda":
        cfg = copy.deepcopy(brax_ppo_config)
        cfg["CRITIC_TYPE"] = "fitted"
        cfg["VALUE_LAMBDA"] = cfg["GAE_LAMBDA"]
    elif method_key == "e_lambda_v_aux":
        cfg = copy.deepcopy(brax_ppo_config)
        cfg["CRITIC_TYPE"] = "e_lambda"
        cfg["E_VARIANT"] = "aux"
        cfg["E_LAMBDA"] = cfg["GAE_LAMBDA"]
        cfg["RETURN_LAMBDA"] = 1.0
        cfg["AUX_LAMBDA"] = 1.0
        cfg["AUX_VF_COEF"] = 1.0
    else:
        raise ValueError(f"Unknown method_key '{method_key}'")

    net_cfg = copy.deepcopy(cfg["NETWORK_CONFIG"])
    cfg["NETWORK_CONFIG"] = net_cfg
    cfg["ENV_NAME"] = env_name
    cfg["TOTAL_TIMESTEPS"] = total_timesteps
    cfg["NUM_ENVS"] = num_envs
    cfg["NUM_STEPS"] = num_steps
    cfg["MINIBATCH_SIZE"] = minibatch_size
    cfg["ACTOR_MINIBATCH_SIZE"] = minibatch_size
    cfg["CRITIC_MINIBATCH_SIZE"] = minibatch_size
    cfg["ACTOR_LR"] = net_cfg["LR"]
    cfg["CRITIC_LR"] = net_cfg["LR"]
    cfg["ACTOR_LR_END"] = net_cfg["LR_END"]
    cfg["CRITIC_LR_END"] = net_cfg["LR_END"]
    cfg["ACTOR_EPOCHS"] = net_cfg["NUM_EPOCHS"]
    cfg["CRITIC_EPOCHS"] = net_cfg["NUM_EPOCHS"]
    return cfg


def build_candidates_for_method(method_key: str, args) -> tuple[dict, list[dict]]:
    """Constructs candidate hyperparameter dictionaries anchored at PPO defaults."""
    base_net = brax_ppo_config["NETWORK_CONFIG"]
    default_decay_ratio = float(base_net["LR_END"]) / float(base_net["LR"])

    starting_params = {
        "ACTOR_LR": float(base_net["LR"]),
        "CRITIC_LR": float(base_net["LR"]),
        "ACTOR_EPOCHS": int(base_net["NUM_EPOCHS"]),
        "CRITIC_EPOCHS": int(base_net["NUM_EPOCHS"]),
        "GAE_LAMBDA": float(brax_ppo_config["GAE_LAMBDA"]),
        "ENT_COEF": float(brax_ppo_config["ENT_COEF"]),
        "LR_DECAY_RATIO": default_decay_ratio,
    }
    axes = {
        "CRITIC_LR": [float(x) for x in args.critic_lrs],
        "CRITIC_EPOCHS": [int(x) for x in args.critic_epochs],
        "ACTOR_LR": [float(x) for x in args.actor_lrs],
        "ACTOR_EPOCHS": [int(x) for x in args.actor_epochs],
    }

    if method_key == "td_lambda":
        starting_params["VALUE_LAMBDA"] = float(brax_ppo_config["VALUE_LAMBDA"])
    elif method_key == "q_ppo_e0_hybrid":
        starting_params["K_ACTIONS"] = int(brax_q_ppo_config["K_ACTIONS"])
        starting_params["VF_COEF"] = float(brax_q_ppo_config["VF_COEF"])
        axes["K_ACTIONS"] = [int(x) for x in args.k_actions]
    elif method_key == "e_lambda_v_aux":
        starting_params["E_LAMBDA"] = float(brax_ppo_config["GAE_LAMBDA"])
        starting_params["AUX_VF_COEF"] = 1.0
        axes["AUX_VF_COEF"] = [float(x) for x in args.aux_vf_coefs]
    else:
        raise ValueError(f"Unknown method_key '{method_key}'")

    candidates = []
    seen = set()

    def _add(param_dict, tag):
        key = tuple(sorted(param_dict.items()))
        if key not in seen:
            seen.add(key)
            candidates.append({"tag": tag, "params": dict(param_dict)})

    _add(starting_params, "baseline_start")

    if args.mode == "critic_lr":
        for lr in axes["CRITIC_LR"]:
            cand = dict(starting_params)
            cand["CRITIC_LR"] = lr
            _add(cand, f"coord_CRITIC_LR={lr}")
    elif args.mode == "tune_coordinate":
        for axis_name, vals in axes.items():
            for v in vals:
                cand = dict(starting_params)
                cand[axis_name] = v
                _add(cand, f"coord_{axis_name}={v}")
    elif args.mode == "tune_grid":
        grid_axes = {
            "CRITIC_LR": axes["CRITIC_LR"],
            "ACTOR_LR": axes["ACTOR_LR"],
            "CRITIC_EPOCHS": axes["CRITIC_EPOCHS"],
        }
        keys = list(grid_axes.keys())
        for combo in itertools.product(*(grid_axes[k] for k in keys)):
            cand = dict(starting_params)
            cand.update(dict(zip(keys, combo)))
            _add(cand, "grid_" + "_".join(f"{k}={v}" for k, v in zip(keys, combo)))
    else:
        raise ValueError(f"Unrecognized mode '{args.mode}'. Expected 'critic_lr', 'tune_coordinate', or 'tune_grid'.")

    return starting_params, candidates


def apply_candidate_params(base_cfg: dict, cand_params: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in cand_params.items():
        if k != "LR_DECAY_RATIO":
            cfg[k] = v
    ratio = float(cand_params["LR_DECAY_RATIO"])
    cfg["ACTOR_LR_END"] = ratio * float(cfg["ACTOR_LR"])
    cfg["CRITIC_LR_END"] = ratio * float(cfg["CRITIC_LR"])
    cfg["NETWORK_CONFIG"]["LR"] = float(cfg["ACTOR_LR"])
    cfg["NETWORK_CONFIG"]["LR_END"] = float(cfg["ACTOR_LR_END"])
    return cfg


def run_candidate(method_key: str, cfg: dict, seed_keys) -> dict:
    make_fn = make_q_ppo_train if method_key == "q_ppo_e0_hybrid" else make_ppo_train
    train_fn = make_fn(cfg)
    vmapped_fn = jax.jit(jax.vmap(train_fn))
    t0 = time.time()
    out = vmapped_fn(seed_keys)
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - t0

    metrics = out["metrics"]
    returns = np.array(metrics["returned_episode_returns"])
    seed_aucs = returns.mean(axis=-1)
    seed_finals = returns[:, -1]
    val_loss_key = "val_loss" if "val_loss" in metrics else "value_loss"
    return {
        "returned_episode_returns": returns,
        "returned_episode_lengths": np.array(metrics["returned_episode_lengths"]),
        "val_loss": np.array(metrics[val_loss_key]),
        "actor_loss": np.array(metrics["actor_loss"]),
        "auc_mean": float(seed_aucs.mean()),
        "auc_std": float(seed_aucs.std()),
        "final_mean": float(seed_finals.mean()),
        "final_std": float(seed_finals.std()),
        "elapsed_time": float(elapsed),
    }


def score_method_candidates(
    env_method_runs: dict[str, dict[str, dict]],
    method_key: str,
    shared_only: bool = False,
) -> pd.DataFrame:
    """
    Scores all tags evaluated for `method_key` across all environments in `env_method_runs`:
      env_method_runs[env][tag] = run_result_dict
    If shared_only=True, restricts to identical hyperparameter tags ('baseline_start',
    'coord_*', 'grid_*') so cross-env consolidation compares identical parameter tuples.
    Returns DataFrame sorted descending by combined_score (0.6*norm_auc + 0.4*norm_final).
    """
    envs = sorted(env_method_runs.keys())
    common_tags = None
    for env in envs:
        tags = {
            t for t in env_method_runs[env].keys()
            if (not shared_only) or t == "baseline_start" or t.startswith("coord_") or t.startswith("grid_")
        }
        common_tags = tags if common_tags is None else (common_tags & tags)
    if not common_tags:
        return pd.DataFrame()

    rows = []
    for tag in sorted(common_tags):
        first_run = env_method_runs[envs[0]][tag]
        row = {"method": method_key, "tag": tag, **first_run["params"]}
        for env in envs:
            r = env_method_runs[env][tag]
            row[f"{env}__auc"] = r["auc_mean"]
            row[f"{env}__auc_std"] = r["auc_std"]
            row[f"{env}__final"] = r["final_mean"]
            row[f"{env}__final_std"] = r["final_std"]
        rows.append(row)

    df = pd.DataFrame(rows)
    norm_aucs, norm_finals = [], []
    for env in envs:
        a_col, f_col = f"{env}__auc", f"{env}__final"
        a_min, a_max = df[a_col].min(), df[a_col].max()
        f_min, f_max = df[f_col].min(), df[f_col].max()
        na = f"{env}__norm_auc"
        nf = f"{env}__norm_final"
        df[na] = (df[a_col] - a_min) / max(a_max - a_min, 1e-8)
        df[nf] = (df[f_col] - f_min) / max(f_max - f_min, 1e-8)
        norm_aucs.append(na)
        norm_finals.append(nf)

    df["mean_norm_auc"] = df[norm_aucs].mean(axis=1)
    df["mean_norm_final"] = df[norm_finals].mean(axis=1)
    df["combined_score"] = 0.6 * df["mean_norm_auc"] + 0.4 * df["mean_norm_final"]
    return df.sort_values("combined_score", ascending=False).reset_index(drop=True)


def generate_comparison_outputs(
    all_runs: dict,
    timesteps: dict,
    out_dir: str,
    batch_size: int,
):
    """Generates tuned config JSONs, tuning CSVs, learning curve plots, and tables."""
    plots_dir = os.path.join(out_dir, "plots")
    tables_dir = os.path.join(out_dir, "tables")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    envs = sorted(all_runs.keys())
    best_configs = {}

    for method_key, method_label in METHODS:
        env_method_runs = {
            env: all_runs[env][method_key]
            for env in envs
            if method_key in all_runs[env] and all_runs[env][method_key]
        }
        df_ranked = score_method_candidates(env_method_runs, method_key)
        if df_ranked.empty:
            continue
        csv_path = os.path.join(data_dir, f"tuning_results_{method_key}.csv")
        df_ranked.to_csv(csv_path, index=False)

        best_row = df_ranked.iloc[0]
        best_tag = str(best_row["tag"])
        sample_cfg = copy.deepcopy(env_method_runs[envs[0]][best_tag]["config"])
        best_configs[method_key] = {
            "label": method_label,
            "best_tag": best_tag,
            "combined_score": float(best_row["combined_score"]),
            "mean_norm_auc": float(best_row["mean_norm_auc"]),
            "mean_norm_final": float(best_row["mean_norm_final"]),
            "params": env_method_runs[envs[0]][best_tag]["params"],
            "config_sample": sample_cfg,
        }

    with open(os.path.join(data_dir, "best_configs.json"), "w") as f:
        json.dump(best_configs, f, indent=2, default=str)

    # Human-readable coordinate search report per (env, method)
    report_lines = []
    for env in envs:
        report_lines.append("=" * 72)
        report_lines.append(f"Environment: {env}")
        report_lines.append("=" * 72)
        for method_key, method_label in METHODS:
            method_runs = all_runs[env].get(method_key, {})
            if not method_runs:
                continue
            df_env = score_method_candidates({env: method_runs}, method_key)
            if df_env.empty:
                continue
            base_params = method_runs["baseline_start"]["params"] if "baseline_start" in method_runs else {}
            best_tag = str(df_env.iloc[0]["tag"])
            best_params = method_runs[best_tag]["params"]
            diffs = {k: (base_params.get(k), v) for k, v in best_params.items() if base_params.get(k) != v}

            report_lines.append(f"\n[{method_label}] ({method_key})")
            report_lines.append(f"  Baseline params: {base_params}")
            report_lines.append(f"  Picked: {best_tag} | AUC={method_runs[best_tag]['auc_mean']:.2f}, Final={method_runs[best_tag]['final_mean']:.2f}")
            if "combined_coordinate_best" in method_runs and best_tag != "combined_coordinate_best":
                comb = method_runs["combined_coordinate_best"]
                report_lines.append(
                    f"  NOTE: combined_coordinate_best (AUC={comb['auc_mean']:.2f}, Final={comb['final_mean']:.2f}) "
                    f"was beaten by '{best_tag}' -> possible parameter interaction!"
                )
            report_lines.append(f"  Changed vs baseline: {diffs if diffs else 'None (baseline_start won)'}")
            report_lines.append("  All tried:")
            for _, row in df_env.iterrows():
                t = str(row["tag"])
                r = method_runs[t]
                t_diffs = {k: v for k, v in r["params"].items() if base_params.get(k) != v}
                marker = "*" if t == best_tag else " "
                report_lines.append(
                    f"    {marker} {t:32s} | score={row['combined_score']:.4f} | "
                    f"AUC={r['auc_mean']:8.2f} | Final={r['final_mean']:8.2f} | diffs={t_diffs}"
                )
        report_lines.append("")

    with open(os.path.join(data_dir, "coordinate_search_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    # Learning Curve Comparison (PPO Baseline Default vs Best Tuned per Method)
    n_envs = len(envs)
    n_cols = min(3, n_envs)
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows), squeeze=False)

    auc_rows_mean, auc_rows_std, auc_annots = [], [], {}
    fin_rows_mean, fin_rows_std, fin_annots = [], [], {}
    col_names = []
    col_specs = []
    for method_key, method_label in METHODS:
        col_names.append(f"{method_label}\n(Default)")
        col_specs.append((method_key, "baseline"))
        col_names.append(f"{method_label}\n(Tuned)")
        col_specs.append((method_key, "tuned"))

    for r_idx, env in enumerate(envs):
        r, c = divmod(r_idx, n_cols)
        ax = axes[r][c]
        row_auc_m, row_auc_s = [], []
        row_fin_m, row_fin_s = [], []

        for c_idx, (method_key, kind) in enumerate(col_specs):
            method_runs = all_runs[env].get(method_key, {})
            if kind == "baseline":
                tag = "baseline_start"
                ls = "--"
                lw = 1.6
                alpha_fill = 0.08
                tag_short = "default"
            else:
                # Per-env best or global best if available; pick per-env best by AUC for table fairness,
                # while plotting global best when multiple envs are consolidated.
                tag = best_configs[method_key]["best_tag"] if method_key in best_configs else "baseline_start"
                if tag not in method_runs and method_runs:
                    tag = max(method_runs.keys(), key=lambda t: method_runs[t]["auc_mean"])
                ls = "-"
                lw = 2.2
                alpha_fill = 0.16
                tag_short = tag.replace("coord_", "")

            run_res = method_runs[tag]
            returns = run_res["returned_episode_returns"]
            mean_curve = returns.mean(axis=0)
            std_curve = returns.std(axis=0)
            t = np.arange(1, len(mean_curve) + 1) * batch_size
            color = METHOD_COLORS[method_key]
            label_txt = f"{dict(METHODS)[method_key]} ({kind})"
            ax.plot(t, mean_curve, label=label_txt, color=color, ls=ls, lw=lw)
            ax.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=alpha_fill)

            row_auc_m.append(run_res["auc_mean"])
            row_auc_s.append(run_res["auc_std"])
            row_fin_m.append(run_res["final_mean"])
            row_fin_s.append(run_res["final_std"])
            auc_annots[(r_idx, c_idx)] = tag_short
            fin_annots[(r_idx, c_idx)] = tag_short

        auc_rows_mean.append(row_auc_m)
        auc_rows_std.append(row_auc_s)
        fin_rows_mean.append(row_fin_m)
        fin_rows_std.append(row_fin_s)
        ax.set_title(ENV_TITLES.get(env, env), fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Episode Return")
        ax.grid(True, alpha=0.3)
        if r_idx == 0:
            ax.legend(loc="best", fontsize=7.0)

    for idx in range(n_envs, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle("Brax: TD(λ) vs Q-PPO E(0) (Bilinear+Hybrid) vs E(λ) (v_aux) — Default & Tuned", fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "baseline_vs_tuned_learning_curves.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(plots_dir, "baseline_vs_tuned_learning_curves.pdf"), bbox_inches="tight")
    plt.close()

    flat_cols = [c.replace("\n", " ") for c in col_names]
    df_auc_m = pd.DataFrame(auc_rows_mean, index=envs, columns=flat_cols)
    df_auc_s = pd.DataFrame(auc_rows_std, index=envs, columns=flat_cols)
    save_table_bundle(
        df_auc_m,
        df_auc_s,
        "Table 1: TD(lambda) vs Q-PPO E(0) (Bilinear+Hybrid) vs E(lambda) (v_aux) — AUC Return",
        os.path.join(tables_dir, "table1_auc_baseline_vs_tuned"),
        cell_annotations=auc_annots,
    )

    df_fin_m = pd.DataFrame(fin_rows_mean, index=envs, columns=flat_cols)
    df_fin_s = pd.DataFrame(fin_rows_std, index=envs, columns=flat_cols)
    save_table_bundle(
        df_fin_m,
        df_fin_s,
        "Table 2: TD(lambda) vs Q-PPO E(0) (Bilinear+Hybrid) vs E(lambda) (v_aux) — Final Return",
        os.path.join(tables_dir, "table2_final_baseline_vs_tuned"),
        cell_annotations=fin_annots,
    )


def run_sweep(args):
    envs = args.envs
    batch_size = args.num_envs * args.num_steps
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "data"), exist_ok=True)

    rng = jax.random.PRNGKey(args.master_seed)
    seed_keys = jax.random.split(rng, args.n_seeds)

    all_runs = {}
    timesteps = {env: get_env_default_timesteps(env, args.total_timesteps) for env in envs}

    for env_name in envs:
        env_timesteps = timesteps[env_name]
        all_runs[env_name] = {}
        print(f"\n>>> Environment: {env_name} ({env_timesteps:,} steps, mode={args.mode}) <<<")

        for method_key, method_label in METHODS:
            all_runs[env_name][method_key] = {}
            base_cfg = build_base_method_config(
                method_key=method_key,
                env_name=env_name,
                total_timesteps=env_timesteps,
                num_envs=args.num_envs,
                num_steps=args.num_steps,
                minibatch_size=args.minibatch_size,
            )
            starting_params, candidates = build_candidates_for_method(method_key, args)
            print(f"\n  [{method_label}] Evaluating {len(candidates)} candidate(s)...")

            for idx, cand in enumerate(candidates):
                tag = cand["tag"]
                cand_params = cand["params"]
                cfg = apply_candidate_params(base_cfg, cand_params)
                res = run_candidate(method_key, cfg, seed_keys)
                all_runs[env_name][method_key][tag] = {
                    **res,
                    "params": cand_params,
                    "config": cfg,
                }
                print(
                    f"    ({idx + 1}/{len(candidates)}) {tag:28s} | "
                    f"AUC={res['auc_mean']:8.2f} ± {res['auc_std']:5.2f} | "
                    f"Final={res['final_mean']:8.2f} ({res['elapsed_time']:.1f}s)"
                )

            # In coordinate mode, synthesize & evaluate combined coordinate winner per method
            if args.mode == "tune_coordinate":
                df_coord = score_method_candidates({env_name: all_runs[env_name][method_key]}, method_key)
                combined_params = dict(starting_params)
                for axis_name in starting_params.keys():
                    axis_rows = df_coord[
                        df_coord["tag"].str.startswith(f"coord_{axis_name}=") | (df_coord["tag"] == "baseline_start")
                    ]
                    val = axis_rows.iloc[0][axis_name]
                    if isinstance(val, np.generic):
                        val = val.item()
                    combined_params[axis_name] = val

                if tuple(sorted(combined_params.items())) != tuple(sorted(starting_params.items())):
                    cfg = apply_candidate_params(base_cfg, combined_params)
                    res = run_candidate(method_key, cfg, seed_keys)
                    all_runs[env_name][method_key]["combined_coordinate_best"] = {
                        **res,
                        "params": combined_params,
                        "config": cfg,
                    }
                    print(
                        f"    [combined_coordinate_best]    | "
                        f"AUC={res['auc_mean']:8.2f} ± {res['auc_std']:5.2f} | "
                        f"Final={res['final_mean']:8.2f} ({res['elapsed_time']:.1f}s)"
                    )

    save_path = os.path.join(args.out_dir, "data", "sweep_metrics_all.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"runs": all_runs, "timesteps": timesteps, "args": vars(args)}, f)
    print(f"\nSaved raw sweep metrics to {save_path}")

    generate_comparison_outputs(all_runs, timesteps, args.out_dir, batch_size)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare & tune TD(lambda), Q-PPO E(0) (Bilinear+Hybrid), and E(lambda) (v_aux) on Brax"
    )
    parser.add_argument("--envs", nargs="+", default=["inverted_pendulum", "hopper", "halfcheetah", "walker2d"])
    parser.add_argument(
        "--mode",
        type=str,
        default="tune_coordinate",
        choices=["critic_lr", "tune_coordinate", "tune_grid"],
        help="Tuning strategy: coordinate search around PPO defaults, critic_lr sweep, or small grid",
    )
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps (default: per-env)")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--minibatch_size", type=int, default=1024)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--master_seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results/sweep_q_ppo_vs_baselines_brax")

    # Hyperparameter tuning axes anchored at PPO defaults (brax_config.py)
    parser.add_argument("--critic_lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    parser.add_argument("--actor_lrs", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--critic_epochs", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--actor_epochs", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--gae_lambdas", nargs="+", type=float, default=[0.8, 0.9, 0.95])
    parser.add_argument("--value_lambdas", nargs="+", type=float, default=[0.5, 0.8, 0.95])
    parser.add_argument("--e_lambdas", nargs="+", type=float, default=[0.5, 0.8, 0.95])
    parser.add_argument("--k_actions", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--vf_coefs", nargs="+", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--aux_vf_coefs", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--ent_coefs", nargs="+", type=float, default=[0.0, 0.001, 0.005])
    parser.add_argument(
        "--lr_decay_ratios",
        nargs="+",
        type=float,
        default=[1.0, 0.1, float(brax_ppo_config["NETWORK_CONFIG"]["LR_END"]) / float(brax_ppo_config["NETWORK_CONFIG"]["LR"])],
    )
    return parser


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    run_sweep(cli_args)

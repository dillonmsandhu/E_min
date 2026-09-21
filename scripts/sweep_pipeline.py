"""
sweep_pipeline.py
A unified, modular hyperparameter sweep and cross-algorithm comparison pipeline.

Usage Examples:
    # 1. Run full sweep over all core algorithms for fixed policy on FourRooms-misc:
    python scripts/sweep_pipeline.py --policy fixed --env-name FourRooms-misc --n-seeds 3 --total-timesteps 1000

    # 2. Run sweep for specific algorithms:
    python scripts/sweep_pipeline.py --policy fixed --algos exact_td exact_mc --n-seeds 5

    # 3. Use custom learning rate grid:
    python scripts/sweep_pipeline.py --policy random --lr-grid 0.05 0.01 0.005 0.001 0.0005 0.0001
"""

import os
import sys

# Ensure repository root is in sys.path when running from scripts/ or anywhere
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
import importlib
import datetime
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import core.config as default_cfg
from core.sweep import tune
from notebooks.analyze_sweeps import (
    plot_algorithm_comparison,
    summarize_algorithm_comparison,
    plot_lambda_spectrum_vs_E,
)


# Map algorithm shorthand names to module paths
ALGO_REGISTRY = {
    "fixed": {
        "exact_td": "fixed_policy.exact_td",
        "exact_mc": "fixed_policy.exact_mc",
        "exact_E_gd": "fixed_policy.exact_E_gd",
        "exact_E": "fixed_policy.exact_E_gd",
        "exact_E_lambda": "fixed_policy.exact_E_lambda",
        "exact_E_td": "fixed_policy.exact_E_td",
        "exact_Etd": "fixed_policy.exact_E_td",
        "exact_td_lambda": "fixed_policy.exact_td_lambda",
        "exact_td_symmetric": "fixed_policy.exact_td_symmetric",
        "td": "fixed_policy.td",
        "td0": "fixed_policy.td0",
        "mc": "fixed_policy.mc",
        "monte_carlo": "fixed_policy.mc",
        "sampled_E": "fixed_policy.sampled_E",
        "unbiased_sampled_E": "fixed_policy.unbiased_sampled_E",
    },
    "random": {
        "exact_td": "random_policy.exact_td",
        "exact_mc": "random_policy.exact_mc",
        "exact_E_gd": "random_policy.exact_E_gd",
        "exact_E": "random_policy.exact_E_gd",
        "exact_E_lambda": "random_policy.exact_E_lambda",
        "exact_td_lambda": "random_policy.exact_td_lambda",
        "exact_E_td": "random_policy.exact_E_td",
        "exact_Etd": "random_policy.exact_E_td",
        "exact_E_sampling_form": "random_policy.exact_E_sampling_form",
        "exact_td_lambda": "random_policy.exact_td_lambda",
        "exact_td_symmetric": "random_policy.exact_td_symmetric",
        "td": "random_policy.td",
        "td0": "random_policy.td0",
        "mc": "random_policy.mc",
        "monte_carlo": "random_policy.mc",
        "sampled_E": "random_policy.sampled_E",
        "unbiased_sampled_E": "random_policy.unbiased_sampled_E",
    },
    "ppo": {
        "sampled_E": "algos.E",
        "sampled_E_gd": "algos.E",
        "E": "algos.E",
        "E_min": "algos.E",
        "sampled_td_lambda": "algos.ppo",
        "td_lambda": "algos.ppo",
        "td": "algos.ppo",
        "ppo": "algos.ppo",
        "sampled_mc": "algos.mc",
        "mc": "algos.mc",
        "monte_carlo": "algos.mc",
    },
    "hybrid": {
        "sampled_E": "algos.E",
        "sampled_td_lambda": "algos.ppo",
        "sampled_mc": "algos.mc",
    },
}

DEFAULT_ALGOS = ["sampled_E", "sampled_td_lambda", "sampled_mc"]
DEFAULT_SAMPLED_ALGOS = ["sampled_E", "sampled_td_lambda", "sampled_mc"]
DEFAULT_HYBRID_ALGOS = ["sampled_E", "sampled_td_lambda", "sampled_mc"]


def get_default_param_grid(
    algo_name,
    lr_list=None,
    lambda_list=None,
    actor_lr_list=None,
    gae_lambda_list=None,
    value_lambda_list=None,
):
    """Returns sensible default parameter grids for standard and multi-param algorithms."""
    standard_lrs = lr_list if lr_list is not None else [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    grid = {"LR": standard_lrs}

    mc_algos = ["mc", "monte_carlo", "exact_mc", "sampled_mc", "hybrid_mc", "hybrid_exact_mc"]

    # 1. GAE lambda grid (strictly for policy advantages)
    if gae_lambda_list is not None and algo_name not in mc_algos:
        grid["GAE_LAMBDA"] = gae_lambda_list

    # 2. Value lambda grid (strictly for critic returns)
    if value_lambda_list is not None and algo_name not in mc_algos:
        grid["VALUE_LAMBDA"] = value_lambda_list
    elif lambda_list is not None and algo_name not in mc_algos:
        # Default lambda_list maps directly to VALUE_LAMBDA for critic return target
        grid["VALUE_LAMBDA"] = lambda_list

    # 3. Actor LR grid (for policy net)
    if actor_lr_list is not None:
        grid["ACTOR_LR"] = actor_lr_list

    return grid


def resolve_model_load_dir(model_load_dir, env_name, base_results_dir="results"):
    """Finds a valid existing PPO model load directory containing runner_state for fixed policy evaluation."""
    import cloudpickle
    candidates = []

    # If a dict or JSON string mapping env -> model was passed
    if isinstance(model_load_dir, dict):
        model_load_dir = model_load_dir.get(env_name)
    elif isinstance(model_load_dir, str) and model_load_dir.startswith("{"):
        try:
            mapping = json.loads(model_load_dir)
            model_load_dir = mapping.get(env_name)
        except Exception:
            pass

    if model_load_dir and not str(model_load_dir).startswith("PLACEHOLDER"):
        candidates.append(str(model_load_dir))
        candidates.append(f"ground_truth/{model_load_dir}")
    
    gt_dir = os.path.join(base_results_dir, "ppo", "ground_truth")
    if os.path.exists(gt_dir):
        timestamps = sorted([d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d)) and not d.startswith(".")])
        for ts in reversed(timestamps):
            candidates.append(f"ground_truth/{ts}")

    for cand in candidates:
        env_path = os.path.join(base_results_dir, "ppo", cand, env_name)
        pkl_path = os.path.join(env_path, "out.pkl")
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, "rb") as f:
                    data = cloudpickle.load(f)
                if isinstance(data, dict) and "runner_state" in data:
                    print(f"[{env_name}] Auto-resolved MODEL_LOAD_DIR to valid checkpoint: '{cand}'")
                    return cand
            except Exception:
                continue

    return model_load_dir or "ground_truth/short_run"


def run_sweep_pipeline(
    policy_type="fixed",
    env_name="FourRooms-misc",
    algos=None,
    n_seeds=3,
    total_timesteps=1000,
    num_envs=None,
    num_steps=None,
    minibatch_size=None,
    model_load_dir="short_run",
    metric_key="returned_episode_returns",
    rank_by="auc",
    rank_order="higher",
    window_size=40,
    lr_grid=None,
    actor_lr_grid=None,
    lambda_grid=None,
    gae_lambda_grid=None,
    value_lambda_grid=None,
    custom_grids=None,
    config_overrides=None,
    base_save_dir="results",
    log_scale=False,
    use_geom_mean=False,
    sweep_suffix="",
    sweep_root_dir_arg=None,
    lr_end=None,
    actor_lr_end=None,
    light_metrics=None,
):
    """
    Runs a parallel hyperparameter sweep for multiple algorithms on a fixed task,
    saves outputs in an interpretable directory hierarchy, and generates comparison plots.
    
    Returns:
        pipeline_results: dict containing results for each algorithm and comparison artifacts.
    """
    if policy_type == "fixed":
        model_load_dir = resolve_model_load_dir(model_load_dir, env_name, base_save_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if sweep_root_dir_arg:
        sweep_root_dir = sweep_root_dir_arg
    else:
        sweep_batch_name = f"{policy_type}_{env_name}_{timestamp}"
        if sweep_suffix:
            sweep_batch_name += f"_{sweep_suffix}"
        sweep_root_dir = os.path.join(base_save_dir, policy_type, "sweeps", sweep_batch_name)
    os.makedirs(sweep_root_dir, exist_ok=True)
    if algos is not None:
        algos = [a.rstrip(",").strip() for a in algos if a.rstrip(",").strip()]

    if algos is None or algos == ["exact"]:
        algos = [a for a in DEFAULT_ALGOS if a in ALGO_REGISTRY.get(policy_type, {})]
        if not algos and policy_type == "hybrid":
            algos = DEFAULT_HYBRID_ALGOS
    elif algos == ["sampled"]:
        algos = [a for a in DEFAULT_SAMPLED_ALGOS if a in ALGO_REGISTRY.get(policy_type, {})]
    elif algos == ["hybrid"]:
        algos = [a for a in DEFAULT_HYBRID_ALGOS if a in ALGO_REGISTRY.get(policy_type, {})]
    elif algos == ["all"]:
        algos = list(ALGO_REGISTRY.get(policy_type, {}).keys())

    # Base configuration template
    base_config = default_cfg.config.copy()
    base_config["ENV_NAME"] = env_name
    base_config["N_SEEDS"] = n_seeds
    base_config["MODEL_LOAD_DIR"] = model_load_dir

    if light_metrics is not None:
        base_config["LIGHT_METRICS"] = light_metrics
    if total_timesteps is not None:
        base_config["TOTAL_TIMESTEPS"] = total_timesteps
    if num_envs is not None:
        base_config["NUM_ENVS"] = num_envs
    if num_steps is not None:
        base_config["NUM_STEPS"] = num_steps
    if minibatch_size is not None:
        base_config["MINIBATCH_SIZE"] = minibatch_size
    if lr_end is not None:
        base_config["LR_END"] = lr_end
    if actor_lr_end is not None:
        base_config["ACTOR_LR_END"] = actor_lr_end
    if config_overrides:
        base_config.update(config_overrides)

    total_timesteps = base_config.get("TOTAL_TIMESTEPS", 1000)
    num_envs = base_config.get("NUM_ENVS", 1)
    num_steps = base_config.get("NUM_STEPS", 1)
    minibatch_size = base_config.get("MINIBATCH_SIZE", 1)
    base_config["NUM_MINIBATCHES"] = max(1, (num_envs * num_steps) // minibatch_size)
    num_epochs = base_config.get("NUM_EPOCHS", 1)

    print("\n" + "=" * 70)
    print(f"STARTING SWEEP PIPELINE: {policy_type.upper()} POLICY")
    print(f"Environment: {env_name} | Timesteps: {total_timesteps} | Seeds: {n_seeds}")
    print(f"NUM_ENVS: {num_envs} | NUM_STEPS: {num_steps} | MINIBATCH_SIZE: {minibatch_size} | NUM_EPOCHS: {num_epochs}")
    print(f"Critic LR_END: {base_config.get('LR_END')} | Actor LR_END: {base_config.get('ACTOR_LR_END')}")
    print(f"Algorithms to sweep: {algos}")
    print(f"Ranking: {rank_by} ({'lower' if rank_order in ['lower', 'min', 'asc', 'ascending'] else 'higher'} is better)")
    print(f"Output Root Directory: {sweep_root_dir}")
    print("=" * 70 + "\n")

    # Save pipeline run specification
    pipeline_meta = {
        "policy_type": policy_type,
        "env_name": env_name,
        "algos": algos,
        "n_seeds": n_seeds,
        "total_timesteps": total_timesteps,
        "model_load_dir": model_load_dir,
        "metric_key": metric_key,
        "rank_by": rank_by,
        "rank_order": rank_order,
        "window_size": window_size,
        "timestamp": timestamp,
        "base_config": base_config,
    }
    with open(os.path.join(sweep_root_dir, "pipeline_config.json"), "w") as f:
        json.dump(pipeline_meta, f, indent=4)

    algorithm_results = {}
    completed_runs_for_comparison = {}

    for algo_idx, algo_name in enumerate(algos):
        print(f"\n[{algo_idx+1}/{len(algos)}] Running sweep for algorithm: {algo_name}...")
        
        module_path = ALGO_REGISTRY.get(policy_type, {}).get(algo_name)
        if not module_path:
            print(f"Warning: Algorithm '{algo_name}' not found in registry for policy '{policy_type}'. Skipping.")
            continue

        try:
            module = importlib.import_module(module_path)
            make_train = getattr(module, "make_train")
        except Exception as e:
            print(f"Error importing {module_path}: {e}")
            continue

        # Determine hyperparameter grid
        if custom_grids and algo_name in custom_grids:
            param_grid = custom_grids[algo_name]
        else:
            param_grid = get_default_param_grid(
                algo_name,
                lr_list=lr_grid,
                lambda_list=lambda_grid,
                actor_lr_list=actor_lr_grid,
                gae_lambda_list=gae_lambda_grid,
                value_lambda_list=value_lambda_grid,
            )

        # Output folder for this specific algorithm inside the sweep root
        algo_save_dir = os.path.join(sweep_root_dir, algo_name, "tuning")

        try:
            res = tune(
                make_train=make_train,
                base_config=base_config.copy(),
                param_grid=param_grid,
                metric_key=metric_key,
                rank_by=rank_by,
                rank_order=rank_order,
                window_size=window_size,
                save_dir=algo_save_dir,
                log_scale=log_scale,
                save_metrics=True,
            )
            algorithm_results[algo_name] = res
            completed_runs_for_comparison[algo_name] = res
        except Exception as e:
            print(f"!!! Error during sweep of {algo_name} !!!: {e}")
            import traceback
            traceback.print_exc()

    # Generate Cross-Algorithm Comparison
    print("\n" + "=" * 70)
    print("GENERATING CROSS-ALGORITHM COMPARISON")
    print("=" * 70)

    comparison_dir = os.path.join(sweep_root_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Discover any other algorithm runs that were previously completed in sweep_root_dir
    from notebooks.analyze_sweeps import load_sweep_data, find_latest_run_dir
    for item in sorted(os.listdir(sweep_root_dir)):
        if item in completed_runs_for_comparison or item.startswith(".") or item == "comparison":
            continue
        item_path = os.path.join(sweep_root_dir, item)
        if not os.path.isdir(item_path):
            continue
        tuning_dir = os.path.join(item_path, "tuning")
        search_dir = tuning_dir if os.path.exists(tuning_dir) else item_path
        ts, env_sub, run_path = find_latest_run_dir(search_dir)
        if run_path:
            try:
                completed_runs_for_comparison[item] = load_sweep_data(run_path)
            except Exception as load_err:
                print(f"Note: Could not auto-load existing run for {item}: {load_err}")

    summary_df = summarize_algorithm_comparison(
        completed_runs_for_comparison,
        metric_key=metric_key,
        rank_by=rank_by,
        rank_order=rank_order,
        window_size=window_size,
        save_path=os.path.join(comparison_dir, "comparison_summary.csv"),
    )
    # Save JSON format
    summary_df.to_json(os.path.join(comparison_dir, "comparison_summary.json"), orient="records", indent=4)

    print("\nSUMMARY OF BEST PERFORMING CONFIGURATIONS:")
    print(summary_df.to_string(index=False))

    plot_path = os.path.join(comparison_dir, "comparison_best_configs.png")
    plot_algorithm_comparison(
        completed_runs_for_comparison,
        metric_key=metric_key,
        env_name=env_name,
        log_scale=log_scale,
        use_geom_mean=use_geom_mean,
        rank_by=rank_by,
        rank_order=rank_order,
        window_size=window_size,
        title=f"PPO Policy Improvement ({env_name}) - Algorithm Comparison" if policy_type in ["ppo", "hybrid"] else f"{policy_type.capitalize()} Policy Evaluation ({env_name}) - Algorithm Comparison",
    )

    # Generate Lambda Spectrum vs E comparison if both algorithms are present
    td_key = next((k for k in ["sampled_td_lambda", "td_lambda", "td"] if k in completed_runs_for_comparison), None)
    e_key = next((k for k in ["sampled_E", "E", "E_min"] if k in completed_runs_for_comparison), None)
    if td_key and e_key:
        try:
            spectrum_plot_path = os.path.join(comparison_dir, "comparison_lambda_spectrum_vs_E.png")
            plot_lambda_spectrum_vs_E(
                td_sweep_data=completed_runs_for_comparison[td_key],
                e_sweep_data=completed_runs_for_comparison[e_key],
                metric_key=metric_key,
                title=f"E-Minimization vs. TD(λ) Spectrum ({env_name})",
                save_path=spectrum_plot_path,
                log_scale=log_scale,
            )
            print(f"Lambda spectrum comparison plot saved to: {spectrum_plot_path}")
        except Exception as e:
            print(f"Note: Could not generate lambda spectrum plot: {e}")

    print("\n" + "=" * 70)
    print(f"SWEEP PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Results & Plots saved in: {sweep_root_dir}")
    print(f"Comparison Plot: {plot_path}")
    print("=" * 70 + "\n")

    return {
        "sweep_root_dir": sweep_root_dir,
        "comparison_dir": comparison_dir,
        "summary_df": summary_df,
        "algorithm_results": algorithm_results,
    }


ALL_GYMNAX_ENVS_NO_CATCH = [
    # Classic Control
    "CartPole-v1",
    "Pendulum-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "MountainCarContinuous-v0",
    # MinAtar
    "Asterix-MinAtar",
    "Breakout-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
    # BSuite (excluding Catch-bsuite)
    "DeepSea-bsuite",
    "MemoryChain-bsuite",
    "UmbrellaChain-bsuite",
    "DiscountingChain-bsuite",
    # Misc / Navigation / Continuous
    "FourRooms-misc",
    "MetaMaze-misc",
    "PointRobot-misc",
    "Reacher-misc",
    "Swimmer-misc",
    "Pong-misc",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Modular Hyperparameter Sweep Pipeline")
    parser.add_argument("--policy", type=str, default="fixed", choices=["fixed", "random", "ppo", "hybrid"],
                        help="Policy type to evaluate (fixed, random, ppo, hybrid)")
    parser.add_argument("--env-name", type=str, default="FourRooms-misc",
                        help="Environment name (e.g. FourRooms-misc, MountainCar-v0, or 'all_gymnax')")
    parser.add_argument("--env-names", nargs="+", default=None,
                        help="List of environments to sweep, or 'all_gymnax' for all 23 Gymnax envs (excluding Catch)")
    parser.add_argument("--algos", nargs="+", default=None,
                        help="List of algorithms to sweep (e.g. exact_td exact_mc exact_E_gd exact_td_lambda, or group keywords: exact, sampled, hybrid, all)")
    parser.add_argument("--n-seeds", type=int, default=3,
                        help="Number of random seeds to evaluate per configuration")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total training/evaluation updates (timesteps)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments (NUM_ENVS)")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of rollout steps per env (NUM_STEPS)")
    parser.add_argument("--minibatch-size", type=int, default=None,
                        help="Minibatch size (MINIBATCH_SIZE)")
    parser.add_argument("--model-dir", type=str, default="short_run",
                        help="Model load directory for fixed/ppo evaluation policy")
    parser.add_argument("--metric", type=str, default=None,
                        help="Primary metric to optimize and rank by (default: returned_episode_returns for PPO)")
    parser.add_argument("--rank-by", type=str, default="auc", choices=["auc", "final_window", "final_step", "min", "max"],
                        help="Criterion to rank and select best config (default: auc)")
    parser.add_argument("--rank-order", type=str, default=None, choices=["lower", "higher"],
                        help="Optimization goal: lower or higher (defaults to higher for return/reward)")
    parser.add_argument("--higher-is-better", action="store_true",
                        help="Shortcut for --rank-order higher (e.g. for nn_greedy_accuracy or reward)")
    parser.add_argument("--window-size", type=int, default=20,
                        help="Number of final steps to average when using final_window ranking (default: 20)")
    parser.add_argument("--lr-grid", nargs="+", type=float, default=None,
                        help="Custom learning rate grid (e.g. --lr-grid 0.01 0.001 0.0001)")
    parser.add_argument("--actor-lr-grid", nargs="+", type=float, default=None,
                        help="Custom actor learning rate grid for policy algorithms (e.g. --actor-lr-grid 0.001 0.0001)")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="Ending learning rate for value/critic net (default: low value from config, e.g. 0.00003)")
    parser.add_argument("--actor-lr-end", type=float, default=None,
                        help="Ending learning rate for actor net (default: low value from config, e.g. 0.00003)")
    parser.add_argument("--lambda-grid", nargs="+", type=float, default=None,
                        help="Custom lambda grid for TD algorithms (e.g. --lambda-grid 0.0 0.3 0.6 0.9 0.95 1.0)")
    parser.add_argument("--gae-lambda-grid", nargs="+", type=float, default=None,
                        help="Custom GAE lambda grid for policy advantages (e.g. --gae-lambda-grid 0.9 0.99 1.0)")
    parser.add_argument("--value-lambda-grid", nargs="+", type=float, default=None,
                        help="Custom value lambda grid for critic returns (e.g. --value-lambda-grid 0.9 0.99 1.0)")
    parser.add_argument("--custom-grids-json", type=str, default=None,
                        help="Path to JSON file specifying custom grids per algorithm")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON string or path to JSON file with additional config overrides")
    parser.add_argument("--use-geom-mean", action="store_true",
                        help="Use geometric mean for error bands in comparison plot")
    parser.add_argument("--light-metrics", dest="light_metrics", action="store_true", default=None,
                        help="Only log lightweight scalar value metrics (skip grids and heavy matrices)")
    parser.add_argument("--no-light-metrics", dest="light_metrics", action="store_false",
                        help="Log full value metrics including spatial grids and heavy matrices")
    parser.add_argument("--log-scale", dest="log_scale", action="store_true", default=None,
                        help="Force log scale on y-axis for curve plots")
    parser.add_argument("--no-log-scale", dest="log_scale", action="store_false",
                        help="Disable log scale on y-axis (recommended for performance metrics like V_start, reward)")
    parser.add_argument("--sweep-root-dir", type=str, default=None, help="Explicit root directory to save this sweep run.")
    parser.add_argument("--sweep-suffix", type=str, default="", help="Optional suffix for the sweep directory name.")
    parser.add_argument("--use-greedy-policy", action="store_true",
                        help="Evaluate epsilon-greedy optimal policy instead of a loaded PPO network")
    parser.add_argument("--policy-epsilon", type=float, default=0.0,
                        help="Epsilon noise for the greedy policy (default: 0.0)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    custom_grids = None
    if args.custom_grids_json and os.path.exists(args.custom_grids_json):
        with open(args.custom_grids_json, "r") as f:
            custom_grids = json.load(f)

    config_overrides = None
    if args.config:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                config_overrides = json.load(f)
        else:
            config_overrides = json.loads(args.config)

    if args.use_greedy_policy:
        if config_overrides is None:
            config_overrides = {}
        config_overrides["USE_GREEDY_POLICY"] = True
        config_overrides["POLICY_EPSILON"] = args.policy_epsilon

    # Default metric based on policy type
    if args.metric is not None:
        metric = args.metric
    elif args.policy in ["ppo", "hybrid"]:
        metric = "returned_episode_returns"
    else:
        metric = "nn_weighted_VE"

    if args.rank_order is not None:
        rank_order = args.rank_order
    elif args.higher_is_better or metric in ["returned_episode_returns", "returned_discounted_episode_returns", "V_start", "v_pred_start", "reward", "return", "nn_greedy_correct", "nn_greedy_accuracy"]:
        rank_order = "higher"
    else:
        rank_order = "lower"

    # Automatically choose sensible default for log_scale based on metric:
    if args.log_scale is None:
        if metric in ["V_start", "reward", "return", "returned_episode_returns", "nn_greedy_correct", "nn_greedy_accuracy"] or rank_order == "higher":
            log_scale = False
        else:
            log_scale = True
    else:
        log_scale = args.log_scale

    # Resolve environment list:
    if args.env_names:
        if "all_gymnax" in args.env_names:
            env_list = ALL_GYMNAX_ENVS_NO_CATCH
        else:
            env_list = args.env_names
    elif args.env_name == "all_gymnax":
        env_list = ALL_GYMNAX_ENVS_NO_CATCH
    else:
        env_list = [args.env_name]

    for env_idx, target_env in enumerate(env_list):
        print(f"\n=======================================================")
        print(f"[{env_idx + 1}/{len(env_list)}] RUNNING SWEEP PIPELINE FOR ENV: {target_env}")
        print(f"=======================================================")
        run_sweep_pipeline(
            policy_type=args.policy,
            env_name=target_env,
            algos=args.algos,
            n_seeds=args.n_seeds,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            minibatch_size=args.minibatch_size,
            model_load_dir=args.model_dir,
            metric_key=metric,
            rank_by=args.rank_by,
            rank_order=rank_order,
            window_size=args.window_size,
            lr_grid=args.lr_grid,
            actor_lr_grid=args.actor_lr_grid,
            lr_end=args.lr_end,
            actor_lr_end=args.actor_lr_end,
            light_metrics=args.light_metrics,
            lambda_grid=args.lambda_grid,
            gae_lambda_grid=args.gae_lambda_grid,
            value_lambda_grid=args.value_lambda_grid,
            custom_grids=custom_grids,
            config_overrides=config_overrides,
            log_scale=log_scale,
            use_geom_mean=args.use_geom_mean,
            sweep_suffix=args.sweep_suffix,
            sweep_root_dir_arg=args.sweep_root_dir,
        )


if __name__ == "__main__":
    main()

"""
lambda_sweep_pipeline.py
A specialized sweep pipeline for comparing exact_td_lambda and exact_E_lambda
across different values of VALUE_LAMBDA, treating each lambda as a separate algorithm.
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import json
import importlib
import argparse
import datetime

import core.config as default_cfg
from core.sweep import tune
from scripts.sweep_pipeline import ALGO_REGISTRY, get_default_param_grid, resolve_model_load_dir

def parse_args():
    parser = argparse.ArgumentParser(description="Lambda Sweep Pipeline Worker")
    parser.add_argument("--env-name", type=str, required=True, help="Environment name")
    parser.add_argument("--policy", type=str, required=True, choices=["fixed", "random", "ppo", "hybrid"], help="Policy type")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm base name (e.g., exact_td_lambda)")
    parser.add_argument("--sweep-id", type=str, required=True, help="Sweep batch ID")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 0.9, 0.95, 1.0], help="List of VALUE_LAMBDA to sweep")
    parser.add_argument("--lr-grid", type=float, nargs="+", default=[0.005, 0.001, 0.0005], help="Learning rate grid")
    parser.add_argument("--actor-lr-grid", type=float, nargs="+", default=[0.005, 0.001, 0.0005], help="Actor learning rate grid for PPO")
    return parser.parse_args()

def run_lambda_sweep_worker():
    args = parse_args()
    policy_type = args.policy
    env_name = args.env_name
    algo_name = args.algo
    sweep_id = args.sweep_id
    lambdas = args.lambdas

    base_save_dir = "results/lambda_sweep"
    sweep_root_dir = os.path.join(base_save_dir, sweep_id, env_name, policy_type)
    os.makedirs(sweep_root_dir, exist_ok=True)

    if policy_type == "fixed":
        model_load_dir = resolve_model_load_dir("short_run", env_name, "results")
    else:
        model_load_dir = "short_run"

    base_config = default_cfg.config.copy()
    base_config["ENV_NAME"] = env_name
    base_config["N_SEEDS"] = 3
    base_config["MODEL_LOAD_DIR"] = model_load_dir

    print("=" * 70)
    print(f"Starting Lambda Sweep Worker")
    print(f"Sweep ID: {sweep_id} | Env: {env_name} | Policy: {policy_type} | Algo: {algo_name}")
    print(f"Lambdas: {lambdas}")
    print("=" * 70)

    module_path = ALGO_REGISTRY.get(policy_type, {}).get(algo_name)
    if not module_path:
        print(f"Error: Algorithm '{algo_name}' not found in registry for policy '{policy_type}'.")
        sys.exit(1)

    try:
        module = importlib.import_module(module_path)
        make_train = getattr(module, "make_train")
    except Exception as e:
        print(f"Error importing {module_path}: {e}")
        sys.exit(1)

    metric_key = "v_start" if policy_type in ["ppo", "hybrid"] else "nn_weighted_VE"
    rank_by = "auc"
    rank_order = "higher" if policy_type in ["ppo", "hybrid"] else "lower"
    window_size = 40

    for lmbda in lambdas:
        # Create a pseudo-algorithm name so it is treated as a separate algorithm in plots
        pseudo_algo = f"{algo_name}_{lmbda}"
        print(f"\n---> Running sweep for pseudo-algorithm: {pseudo_algo}")

        algo_save_dir = os.path.join(sweep_root_dir, pseudo_algo, "tuning")
        
        param_grid = get_default_param_grid(
            algo_name,
            lr_list=args.lr_grid,
            actor_lr_list=args.actor_lr_grid if policy_type in ["ppo", "hybrid"] else None,
            value_lambda_list=[lmbda]
        )

        try:
            tune(
                make_train=make_train,
                base_config=base_config.copy(),
                param_grid=param_grid,
                metric_key=metric_key,
                rank_by=rank_by,
                rank_order=rank_order,
                window_size=window_size,
                save_dir=algo_save_dir,
                log_scale=True,
                save_metrics=True,
            )
            print(f"Successfully completed tuning for {pseudo_algo}")
        except Exception as e:
            print(f"!!! Error during sweep of {pseudo_algo} !!!: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_lambda_sweep_worker()

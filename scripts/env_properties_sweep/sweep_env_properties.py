#!/usr/bin/env python3
"""
sweep_env_properties.py

Environmental Properties Sweep:
Evaluates E(0) vs. TD(0) vs. TD(lambda) under varying reward density, action spaces,
and environmental noise (5% traction slip with 50% force and transition noise).

Environments:
  1. MountainCar:
     - MountainCarContinuous-v0 (Sparse, Continuous)
     - MountainCarDenseContinuous-v0 (Dense PBRS, Continuous)
     - MountainCarDenseDiscrete-v0 (Dense PBRS, Discrete)
  2. PointRobot (Fully Observable MDP):
     - PointRobot-misc [Sparse, Continuous]
     - PointRobot-misc [Dense, Continuous]
     - PointRobotDiscrete-misc [Dense, Discrete]

Algorithms:
  - E(0): Symmetrized Dirichlet error minimization (algos/E.py)
  - TD(0): Fitted 1-step TD (algos/ppo.py with VALUE_LAMBDA=0.0)
  - TD(lambda): Standard PPO GAE baseline (algos/ppo.py with VALUE_LAMBDA=0.95)

Hyperparameters:
  - NUM_EPOCHS: 16 (held fixed)
  - LR: 0.0003, ACTOR_LR: 0.0003
  - Minibatch size: 1024
  - Environments: 64, Rollout steps: 64

Conditions:
  - Clean: SLIP_PROB=0.0, TRANSITION_NOISE=0.0
  - Noisy: SLIP_PROB=0.05, SLIP_FORCE_SCALE=0.5, TRANSITION_NOISE=0.001
"""

import os
import sys
import time
import argparse
import datetime
import pickle
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from core.config import config as default_config
from algos.E import make_train as make_train_e0
from algos.ppo import make_train as make_train_ppo
from scripts.env_properties_sweep.plot_env_properties_comparison import plot_suite_family_posters


# Environment Definitions
ENV_DEFINITIONS = {
    "mountain_car": [
        {
            "env_id": "MountainCarDenseContinuous-v0",
            "display_name": "MountainCar (Sparse Cont.)",
            "dense_reward": False,
            "discrete": False,
            "short_name": "mc_sparse_cont",
        },
        {
            "env_id": "MountainCarDenseDiscrete-v0",
            "display_name": "MountainCar (Sparse Disc.)",
            "dense_reward": False,
            "discrete": True,
            "short_name": "mc_sparse_disc",
        },
        {
            "env_id": "MountainCarDenseContinuous-v0",
            "display_name": "MountainCar (Dense Cont.)",
            "dense_reward": True,
            "discrete": False,
            "short_name": "mc_dense_cont",
        },
        {
            "env_id": "MountainCarDenseDiscrete-v0",
            "display_name": "MountainCar (Dense Disc.)",
            "dense_reward": True,
            "discrete": True,
            "short_name": "mc_dense_disc",
        },
    ],
    "point_robot": [
        {
            "env_id": "PointRobot-misc",
            "display_name": "PointRobot (Sparse Cont.)",
            "dense_reward": False,
            "discrete": False,
            "short_name": "pr_sparse_cont",
        },
        {
            "env_id": "PointRobotDiscrete-misc",
            "display_name": "PointRobot (Sparse Disc.)",
            "dense_reward": False,
            "discrete": True,
            "short_name": "pr_sparse_disc",
        },
        {
            "env_id": "PointRobot-misc",
            "display_name": "PointRobot (Dense Cont.)",
            "dense_reward": True,
            "discrete": False,
            "short_name": "pr_dense_cont",
        },
        {
            "env_id": "PointRobotDiscrete-misc",
            "display_name": "PointRobot (Dense Disc.)",
            "dense_reward": True,
            "discrete": True,
            "short_name": "pr_dense_disc",
        },
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Environmental Properties Sweep: E(0) vs TD(0) vs TD(lambda)")
    parser.add_argument("--env-family", type=str, default="both", choices=["mountain_car", "point_robot", "both"],
                        help="Environment family to run (default: both)")
    parser.add_argument("--total-timesteps", type=int, default=2_048_000,
                        help="Total environment timesteps per run (default: 2,048,000)")
    parser.add_argument("--n-seeds", type=int, default=8,
                        help="Number of independent random seeds (default: 8)")
    parser.add_argument("--num-epochs", type=int, default=16,
                        help="Critic/Actor training epochs per rollout update (default: 16)")
    parser.add_argument("--num-envs", type=int, default=64,
                        help="Number of parallel vectorized environments (default: 64)")
    parser.add_argument("--num-steps", type=int, default=64,
                        help="Number of rollout steps per env (default: 64)")
    parser.add_argument("--minibatch-size", type=int, default=1024,
                        help="Minibatch size for SGD updates (default: 1024)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="Learning rate for actor and critic (default: 0.0003)")
    
    # Noise parameters
    parser.add_argument("--slip-prob", type=float, default=0.05,
                        help="Probability of traction slip on each step in noisy mode (default: 0.05)")
    parser.add_argument("--slip-force-scale", type=float, default=0.5,
                        help="Fraction of force delivered during slip (default: 0.5)")
    parser.add_argument("--transition-noise", type=float, default=0.001,
                        help="Additive Gaussian transition noise std (default: 0.001)")
    
    # Algorithm parameters
    parser.add_argument("--td-lambda", type=float, default=0.95,
                        help="Lambda value for TD(lambda) (default: 0.95)")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Window size for final mean evaluation (default: 100)")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Unified sweep identifier across SLURM jobs (default: SLURM_ARRAY_JOB_ID or timestamp)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit directory to save results")
    return parser.parse_args()


def run_configuration(algo_type, env_spec, is_noisy, args):
    """
    Executes a single algorithm on a specific environment under clean or noisy condition.
    Returns:
        mean_curve, sem_curve, final_mean, final_sem, auc, elapsed, full_metrics
    """
    cfg = default_config.copy()
    cfg["ENV_NAME"] = env_spec["env_id"]
    cfg["DENSE_REWARD"] = env_spec["dense_reward"]
    cfg["FULLY_OBSERVABLE"] = True
    cfg["TOTAL_TIMESTEPS"] = args.total_timesteps
    cfg["NUM_ENVS"] = args.num_envs
    cfg["NUM_STEPS"] = args.num_steps
    cfg["MINIBATCH_SIZE"] = min(args.minibatch_size, args.num_envs * args.num_steps)
    cfg["NUM_EPOCHS"] = args.num_epochs
    cfg["CRITIC_EPOCHS"] = args.num_epochs
    cfg["ACTOR_EPOCHS"] = args.num_epochs
    cfg["LR"] = args.lr
    cfg["ACTOR_LR"] = args.lr

    if is_noisy:
        cfg["SLIP_PROB"] = args.slip_prob
        cfg["SLIP_FORCE_SCALE"] = args.slip_force_scale
        cfg["TRANSITION_NOISE"] = args.transition_noise
    else:
        cfg["SLIP_PROB"] = 0.0
        cfg["SLIP_FORCE_SCALE"] = 0.0
        cfg["TRANSITION_NOISE"] = 0.0

    # Select Algorithm
    if algo_type == "E(0)":
        cfg["RETURN_LAMBDA"] = 1.0
        make_train_fn = make_train_e0
    elif algo_type == "TD(0)":
        cfg["VALUE_LAMBDA"] = 0.0
        make_train_fn = make_train_ppo
    elif algo_type == "TD(lambda)":
        cfg["VALUE_LAMBDA"] = args.td_lambda
        make_train_fn = make_train_ppo
    else:
        raise ValueError(f"Unknown algorithm: {algo_type}")

    start_time = time.time()
    train_fn = make_train_fn(cfg)
    train_vjit = jax.jit(jax.vmap(train_fn))

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, args.n_seeds)

    out = train_vjit(rngs)
    # Block until execution finishes
    _ = jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out)
    elapsed = time.time() - start_time

    # Extract return metric
    if "returned_episode_returns" in out["metrics"]:
        m_arr = out["metrics"]["returned_episode_returns"]
    elif "returned_discounted_episode_returns" in out["metrics"]:
        m_arr = out["metrics"]["returned_discounted_episode_returns"]
    else:
        first_key = list(out["metrics"].keys())[0]
        m_arr = out["metrics"][first_key]

    tensor = np.asarray(m_arr)  # (n_seeds, updates)
    mean_curve = tensor.mean(axis=0)
    sem_curve = tensor.std(axis=0) / np.sqrt(max(1, args.n_seeds))

    win = min(len(mean_curve), args.window_size)
    final_mean = float(np.mean(mean_curve[-win:]))
    final_sem = float(np.mean(sem_curve[-win:]))
    auc = float(np.mean(mean_curve))

    full_metrics = {k: np.asarray(v) for k, v in out["metrics"].items()}
    return mean_curve, sem_curve, final_mean, final_sem, auc, elapsed, full_metrics


def run_suite(args):
    sweep_id = args.sweep_id
    if sweep_id is None:
        array_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_id:
            sweep_id = f"env_props_{array_id}"
        else:
            job_id = os.environ.get("SLURM_JOB_ID")
            sweep_id = f"env_props_{job_id}" if job_id else f"env_props_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    base_out_dir = args.output_dir if args.output_dir else os.path.join("results/sweeps", sweep_id)
    os.makedirs(base_out_dir, exist_ok=True)

    families = ["mountain_car", "point_robot"] if args.env_family == "both" else [args.env_family]
    algos = ["E(0)", "TD(0)", f"TD(lambda)"]

    print("=" * 80)
    print("ENVIRONMENT PROPERTIES SWEEP: E(0) vs TD(0) vs TD(lambda)")
    print(f"  Suite ID:          {sweep_id}")
    print(f"  Output Directory:  {base_out_dir}")
    print(f"  Families:          {families}")
    print(f"  Total Timesteps:   {args.total_timesteps:,}")
    print(f"  Num Epochs:        {args.num_epochs} (held fixed)")
    print(f"  Seeds per curve:   {args.n_seeds}")
    print(f"  Slippage Settings: prob={args.slip_prob}, force_scale={args.slip_force_scale}, trans_noise={args.transition_noise}")
    print("=" * 80)

    for family in families:
        family_out_dir = os.path.join(base_out_dir, family)
        os.makedirs(family_out_dir, exist_ok=True)
        env_specs = ENV_DEFINITIONS[family]

        all_results = {}
        summary_rows = []

        total_runs = len(env_specs) * 2 * len(algos)
        run_count = 0

        for env_spec in env_specs:
            env_name = env_spec["display_name"]
            short_name = env_spec["short_name"]
            all_results[short_name] = {"clean": {}, "noisy": {}, "spec": env_spec}

            for is_noisy, condition in [(False, "clean"), (True, "noisy")]:
                cond_label = "Noisy" if is_noisy else "Clean"
                print(f"\n--- [{family.upper()}] {env_name} | Condition: {cond_label} ---")

                for algo in algos:
                    run_count += 1
                    algo_label = f"TD(λ={args.td_lambda})" if algo == "TD(lambda)" else algo
                    print(f"[{run_count:02d}/{total_runs}] Running {algo_label} on {env_name} ({cond_label})...")

                    mean_c, sem_c, f_mean, f_sem, auc, elapsed, full_m = run_configuration(
                        algo, env_spec, is_noisy, args
                    )

                    print(f"       Finished in {elapsed:.1f}s | Final Return: {f_mean:.2f} ± {f_sem:.2f} | AUC: {auc:.2f}")

                    all_results[short_name][condition][algo] = {
                        "mean_curve": mean_c,
                        "sem_curve": sem_c,
                        "final_mean": f_mean,
                        "final_sem": f_sem,
                        "auc": auc,
                        "elapsed": elapsed,
                        "metrics": full_m,
                    }

                    summary_rows.append({
                        "family": family,
                        "env_id": env_spec["env_id"],
                        "env_variant": env_name,
                        "short_name": short_name,
                        "condition": condition,
                        "algo": algo,
                        "num_epochs": args.num_epochs,
                        "final_mean": f_mean,
                        "final_sem": f_sem,
                        "auc": auc,
                        "elapsed_seconds": elapsed,
                    })

        # Save summary CSV
        os.makedirs(family_out_dir, exist_ok=True)
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(family_out_dir, f"summary_{family}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary CSV: {csv_path}")

        # Save metrics pickle
        pkl_path = os.path.join(family_out_dir, f"metrics_{family}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Saved metrics data: {pkl_path}")

        # Generate 2x3 publication comparison poster for this family
        try:
            pdf_path = plot_suite_family_posters(all_results, family, family_out_dir, args)
            print(f"Generated Vector Publication Figure: {pdf_path}")
        except Exception as e:
            print(f"Error generating plot for {family}: {e}")

    print("\n" + "=" * 80)
    print("ENVIRONMENT PROPERTIES SWEEP COMPLETE!")
    print(f"Results located at: {base_out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    cli_args = parse_args()
    run_suite(cli_args)

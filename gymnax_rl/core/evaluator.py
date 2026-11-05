import os
import sys
import copy
import argparse
import datetime
import traceback
import jax
import jax.numpy as jnp

from gymnax_rl.core.gymnax_config import config as default_config
from gymnax_rl.core.gymnax_utils import parse_config_override, save_results, save_plot
from gymnax_rl.core.evaluate import evaluate
from gymnax_rl.core.sweep import tune


def train_and_evaluate_seeds(make_train, config, env_name, rng, run_dir, num_seeds):
    """Train and evaluate the model across multiple seeds using jax.vmap."""
    config["ENV_NAME"] = env_name
    config["N_SEEDS"] = num_seeds
    steps_per_pi = config["NUM_ENVS"] * config["NUM_STEPS"]

    train_vjit = jax.jit(jax.vmap(make_train(config)))
    rngs = jax.random.split(rng, num_seeds)
    outs = jax.block_until_ready(train_vjit(rngs))
    metrics = outs["metrics"]

    env_dir = os.path.join(run_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)
    save_results(metrics, config, env_name, env_dir)

    for metric_name in ["returned_episode_returns", "value_loss", "actor_loss"]:
        if metric_name in metrics:
            series = metrics[metric_name].mean(axis=0)
            save_plot(env_dir, env_name, steps_per_pi, series, metric_name)

    return outs


def run_experiment_main(make_train, SAVE_DIR="ppo"):
    """
    Unified CLI entry point for running Gymnax experiments and sweeps.
    """
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--run-suffix", type=str, default=run_timestamp)
    parser.add_argument("--n-seeds", type=int, default=0)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-metrics", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--env-ids", "--env-id", dest="env_ids", nargs="+", default=[])
    parser.add_argument("--sweep", nargs="?", const="default", default=None)

    args = parser.parse_args()

    base_cfg = default_config.copy()
    env_list = [base_cfg.get("ENV_NAME")]
    if args.env_ids:
        env_list = args.env_ids

    for i, env_name in enumerate(env_list):
        if env_name is None:
            continue

        run_config = base_cfg.copy()
        run_config["ENV_NAME"] = env_name

        if args.config:
            run_config.update(parse_config_override(args.config))

        if args.n_seeds > 0:
            run_config["N_SEEDS"] = args.n_seeds

        print(f"\n{'='*50}")
        print(f"RUNNING ENV {i+1}/{len(env_list)}: {env_name}")
        print(f"Config: {args.config}")
        print(f"{'='*50}")

        rng = jax.random.PRNGKey(run_config.get("SEED", 42))
        run_dir = os.path.join(f"results/{SAVE_DIR}/{args.run_suffix}")

        try:
            if args.sweep:
                if args.sweep == "default":
                    param_grid = {"LR": [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]}
                else:
                    import json
                    if os.path.exists(args.sweep):
                        with open(args.sweep, "r") as f:
                            param_grid = json.load(f)
                    else:
                        param_grid = json.loads(args.sweep)

                save_base = f"results/{SAVE_DIR}/tuning" if args.run_suffix == run_timestamp else f"{run_dir}/tuning"
                tune(
                    make_train,
                    run_config,
                    param_grid,
                    save_dir=save_base,
                    save_checkpoint=args.save_checkpoint,
                    save_metrics=args.save_metrics or True,
                )
            else:
                evaluate(run_config, make_train, run_dir, args, rng)
        except Exception as e:
            print(f"!!! CRITICAL ERROR running {env_name} !!!")
            traceback.print_exc()
            print("Continuing to next environment...")

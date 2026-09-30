import os
import sys
import copy
import warnings
import argparse
from datetime import datetime
import jax
import jax.numpy as jnp

from brax_rl.core.brax_config import config as default_config
from brax_rl.core.brax_utils import (
    save_results,
    save_plot,
    load_config_dict,
    load_config,
)


def tune(make_train, in_config, env_name):
    """Hyperparameter sweep with wandb."""
    import wandb
    alg_name = in_config.get("ALG", "PPO")
    project_name = f"{alg_name}_hyperparam_tuning"

    def wrapped_make_train():
        wandb.init(project=project_name)
        config = copy.deepcopy(in_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("Running experiment with params:", config)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))
        metrics = outs["metrics"]
        mean_test_ret = float(jnp.mean(metrics["returned_episode_returns"]))
        max_test_ret = float(jnp.max(metrics["returned_episode_returns"].mean(-1)))
        wandb.log({
            "mean_test_return": mean_test_ret,
            "max_test_return": max_test_ret,
        })

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "grid",
        "metric": {"name": "max_test_return", "goal": "maximize"},
        "parameters": {
            "LR": {"values": [2.5e-3, 5e-4, 1e-4]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


def train_and_evaluate_seeds(make_train, config, env_name, rng, run_dir, num_seeds):
    """Train and evaluate the model across multiple seeds."""
    config["ENV_NAME"] = env_name
    steps_per_pi = config["NUM_ENVS"] * config["NUM_STEPS"]
    rngs = jax.random.split(rng, num_seeds)
    outs = jax.jit(jax.vmap(make_train(config)))(rngs)
    metrics = outs["metrics"]

    env_dir = os.path.join(run_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)
    save_results(metrics, config, env_name, env_dir)
    save_plot(env_dir, env_name, steps_per_pi, metrics["returned_episode_returns"].mean(0))

    mean_ret = float(jnp.mean(metrics["returned_episode_returns"]))
    max_ret = float(jnp.max(metrics["returned_episode_returns"]))
    print(f"Mean Test return on {env_name} across {num_seeds} seeds: {mean_ret:.2f}")
    print(f"Max Test return on {env_name}: {max_ret:.2f}")
    return metrics


def run_experiment_main(make_train, default_critic="fitted", save_dir=None, args=None):
    """
    Main entry point for running Brax RL training and evaluation from CLI or script.
    """
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(description="Run Brax RL algorithm with selectable critic")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--critic",
        type=str,
        default=default_critic,
        choices=["fitted", "ppo", "td_0", "td", "e_0", "e0", "e", "e_lambda", "elambda", "e_geometric"],
        help="Critic loss formulation: 'fitted', 'td_0', 'e_0', or 'e_lambda'",
    )
    parser.add_argument("--e-lambda", dest="e_lambda", type=float, default=None, help="Lambda value for E(lambda) critic")
    parser.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=None, help="Lambda value for GAE advantages")
    parser.add_argument("--value-lambda", dest="value_lambda", type=float, default=None, help="Lambda value for fitted critic value targets")
    parser.add_argument("--return-lambda", dest="return_lambda", type=float, default=None, help="Lambda value for E-minimization return targets")
    parser.add_argument("--env", "--envs", dest="envs", type=str, nargs="*", default=[], help="Environment names to run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=None, help="Override NUM_ENVS")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override TOTAL_TIMESTEPS")
    parser.add_argument("--num-steps", type=int, default=None, help="Override NUM_STEPS")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--actor-lr", dest="actor_lr", type=float, default=None, help="Override actor learning rate")
    parser.add_argument("--critic-lr", dest="critic_lr", type=float, default=None, help="Override critic learning rate")
    parser.add_argument("--actor-epochs", dest="actor_epochs", type=int, default=None, help="Override actor epochs")
    parser.add_argument("--critic-epochs", dest="critic_epochs", type=int, default=None, help="Override critic epochs")
    parser.add_argument("--actor-minibatch-size", dest="actor_minibatch_size", type=int, default=None, help="Override actor minibatch size")
    parser.add_argument("--critic-minibatch-size", dest="critic_minibatch_size", type=int, default=None, help="Override critic minibatch size")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--run-suffix", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    cli_args = parser.parse_args(args)

    config = copy.deepcopy(default_config)
    if cli_args.config is not None:
        if os.path.isfile(cli_args.config):
            loaded = load_config_dict(cli_args.config) if cli_args.config.endswith(".py") else load_config(cli_args.config)
            config.update(loaded)
        elif os.path.isfile(os.path.join("configs/ppo", cli_args.config)):
            cfg_path = os.path.join("configs/ppo", cli_args.config)
            loaded = load_config_dict(cfg_path) if cfg_path.endswith(".py") else load_config(cfg_path)
            config.update(loaded)
        else:
            print(f"Warning: Config file {cli_args.config} not found. Using default brax config.")

    # Apply command-line overrides
    config["CRITIC_TYPE"] = cli_args.critic
    if cli_args.num_envs is not None:
        config["NUM_ENVS"] = cli_args.num_envs
    if cli_args.total_timesteps is not None:
        config["TOTAL_TIMESTEPS"] = cli_args.total_timesteps
    if cli_args.num_steps is not None:
        config["NUM_STEPS"] = cli_args.num_steps
    if cli_args.lr is not None:
        config["NETWORK_CONFIG"]["LR"] = cli_args.lr
        if cli_args.actor_lr is None:
            config["ACTOR_LR"] = cli_args.lr
        if cli_args.critic_lr is None:
            config["CRITIC_LR"] = cli_args.lr
    if cli_args.actor_lr is not None:
        config["ACTOR_LR"] = cli_args.actor_lr
    if cli_args.critic_lr is not None:
        config["CRITIC_LR"] = cli_args.critic_lr
    if cli_args.actor_epochs is not None:
        config["ACTOR_EPOCHS"] = cli_args.actor_epochs
    if cli_args.critic_epochs is not None:
        config["CRITIC_EPOCHS"] = cli_args.critic_epochs
    if cli_args.actor_minibatch_size is not None:
        config["ACTOR_MINIBATCH_SIZE"] = cli_args.actor_minibatch_size
    if cli_args.critic_minibatch_size is not None:
        config["CRITIC_MINIBATCH_SIZE"] = cli_args.critic_minibatch_size
    if cli_args.e_lambda is not None:
        config["E_LAMBDA"] = cli_args.e_lambda
    if cli_args.gae_lambda is not None:
        config["GAE_LAMBDA"] = cli_args.gae_lambda
    if cli_args.value_lambda is not None:
        config["VALUE_LAMBDA"] = cli_args.value_lambda
    if cli_args.return_lambda is not None:
        config["RETURN_LAMBDA"] = cli_args.return_lambda

    config["ALG"] = f"PPO_{config['CRITIC_TYPE'].upper()}"
    config["NUM_SEEDS"] = cli_args.num_seeds
    config["SEED"] = cli_args.seed

    base_dir = save_dir if save_dir is not None else f"brax_{config['CRITIC_TYPE']}"
    run_dir = os.path.join("results", f"{base_dir}/{cli_args.run_suffix}")
    print(f"Algorithm: {config['ALG']} | Critic: {config['CRITIC_TYPE']}")
    print(f"Saving all results to {run_dir}")

    envs_to_run = cli_args.envs if cli_args.envs else [config.get("ENV_NAME", "hopper")]

    if cli_args.tune:
        for env in envs_to_run:
            config["ENV_NAME"] = env
            tune(make_train, config, env)

    for env_name in envs_to_run:
        print(f"Running environment: {env_name} with {config['CRITIC_TYPE']} critic")
        config["ENV_NAME"] = env_name
        rng = jax.random.PRNGKey(cli_args.seed)
        if cli_args.num_seeds > 1:
            metrics = train_and_evaluate_seeds(make_train, config, env_name, rng, run_dir, cli_args.num_seeds)
        else:
            steps_per_pi = config["NUM_ENVS"] * config["NUM_STEPS"]
            out = jax.jit(make_train(config))(rng)
            metrics = out["metrics"]
            mean_ret = float(jnp.mean(metrics["returned_episode_returns"]))
            max_ret = float(jnp.max(metrics["returned_episode_returns"]))
            num_policies = len(metrics["returned_episode_returns"])
            print(f"Mean return is {mean_ret:.2f}")
            print(f"Max return is {max_ret:.2f}")
            print(f"Num Policies: {num_policies}")
            if "delta_mag" in metrics:
                print(f"Mean TD/Error magnitude: {float(jnp.mean(metrics['delta_mag'])):.4f}")

            os.makedirs(run_dir, exist_ok=True)
            env_dir = os.path.join(run_dir, env_name)
            save_results(out, config, env_name, env_dir)
            save_plot(env_dir, env_name, steps_per_pi, metrics["returned_episode_returns"])

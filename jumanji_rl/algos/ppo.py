# jumanji_rl/algos/ppo.py
# Pure JAX standard V-PPO implementation for discrete Jumanji environments

from __future__ import annotations
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional

from jumanji_rl.core.jumanji_config import config as base_default_config
from jumanji_rl.core.jumanji_wrappers import make_env
from jumanji_rl.core.jumanji_network import (
    initialize_actor_critic,
    initialize_split_train_state,
)
from jumanji_rl.core.jumanji_critics import (
    calculate_gae,
    calculate_targets,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
)
from jumanji_rl.core.jumanji_utils import (
    Transition,
    shuffle_and_batch,
    compute_runtime_metrics,
)


def make_train(base_config: dict):
    """
    Factory creating standard discrete V-PPO training function for Jumanji environments.
    """
    cfg = base_default_config.copy()
    cfg.update(base_config)
    config = cfg

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_UPDATES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)

    env, env_params = make_env(config)
    obs_shape = env.observation_space(env_params).shape

    def train(rng: jax.Array, hparams: Optional[dict] = None):
        if hparams is not None:
            cfg_run = config.copy()
            cfg_run.update(hparams)
        else:
            cfg_run = config

        actor_epochs = cfg_run.get("ACTOR_EPOCHS") or cfg_run["NUM_EPOCHS"]
        critic_epochs = cfg_run.get("CRITIC_EPOCHS") or cfg_run["NUM_EPOCHS"]

        actor_mb_size = min(batch_size, cfg_run.get("ACTOR_MINIBATCH_SIZE") or cfg_run["MINIBATCH_SIZE"])
        critic_mb_size = min(batch_size, cfg_run.get("CRITIC_MINIBATCH_SIZE") or cfg_run["MINIBATCH_SIZE"])

        actor_num_mb = max(1, batch_size // actor_mb_size)
        critic_num_mb = max(1, batch_size // critic_mb_size)

        gae_lambda = cfg_run["GAE_LAMBDA"]
        value_lambda = cfg_run.get("VALUE_LAMBDA", 1.0)

        rng, net_rng = jax.random.split(rng)
        network, network_params = initialize_actor_critic(
            net_rng, obs_shape, env, env_params, cfg_run
        )
        train_state = initialize_split_train_state(cfg_run, network, network_params)

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, cfg_run["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        start_obs = obsv
        last_true_terminal = jnp.zeros((cfg_run["NUM_ENVS"],), dtype=bool)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # 1. Rollout Collection
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                step_keys = jax.random.split(_rng, cfg_run["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, env_state, action, env_params)

                next_obs = info["real_next_obs"]
                is_timeout = info["is_timeout"]
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=next_obs,
                    info=info,
                    is_start=is_start,
                    is_timeout=is_timeout,
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, cfg_run["NUM_STEPS"]
            )

            # 2. Advantage & Value Target Calculation
            _, last_val = network.apply(train_state.params, last_obs)
            regular_next_val = jnp.concatenate([traj_batch.value[1:], last_val[None]], axis=0)
            timeout_next_val = network.apply(train_state.params, traj_batch.next_obs, method=network.value)
            next_value = jnp.where(traj_batch.is_timeout, timeout_next_val, regular_next_val)

            advantages, targets = calculate_gae(
                traj_batch,
                next_value,
                gamma=cfg_run["GAMMA"],
                gae_lambda=gae_lambda,
            )

            # 3. Critic Update Phase
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    traj_mb, targets_mb = mb
                    def _c_loss(critic_params):
                        return ppo_critic_loss_fn(
                            critic_params,
                            train_state.actor.params,
                            network,
                            traj_mb,
                            targets_mb,
                            cfg_run,
                        )
                    (loss, aux), grads = jax.value_and_grad(_c_loss, has_aux=True)(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads=grads)
                    return train_state, aux

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(critic_batch, batch_size, critic_num_mb, _rng)
                train_state, c_losses = jax.lax.scan(_critic_minibatch, train_state, minibatches)
                return (train_state, critic_batch, rng), c_losses

            critic_batch = (traj_batch, targets)
            rng, _rng = jax.random.split(rng)
            c_init_state = (train_state, critic_batch, _rng)
            (train_state, _, rng), c_loss_epochs = jax.lax.scan(
                _critic_epoch, c_init_state, None, critic_epochs
            )

            # 4. Actor Update Phase
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    traj_mb, adv_mb = mb
                    def _a_loss(actor_params):
                        return ppo_actor_loss_fn(
                            actor_params,
                            train_state.critic.params,
                            network,
                            traj_mb,
                            adv_mb,
                            cfg_run,
                        )
                    (loss, aux), grads = jax.value_and_grad(_a_loss, has_aux=True)(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads=grads)
                    return train_state, aux

                train_state, actor_batch, rng = actor_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(actor_batch, batch_size, actor_num_mb, _rng)
                train_state, a_losses = jax.lax.scan(_actor_minibatch, train_state, minibatches)
                return (train_state, actor_batch, rng), a_losses

            actor_batch = (traj_batch, advantages)
            rng, _rng = jax.random.split(rng)
            a_init_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), a_loss_epochs = jax.lax.scan(
                _actor_epoch, a_init_state, None, actor_epochs
            )

            # 5. Runtime Metrics
            combined_loss_info = {**c_loss_epochs, **a_loss_epochs}
            metric = compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=combined_loss_info,
                start_obs=start_obs,
                config=cfg_run,
            )
            metric["advantage"] = advantages.mean()

            runner_state = (train_state, env_state, last_obs, last_true_terminal, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

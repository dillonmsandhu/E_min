from gymnax_rl.core.gymnax_config import config as base_default_config
import os
import sys

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp

from gymnax_rl.core.gymnax_utils import (
    Double_Sample_Transition,
    shuffle_and_batch,
    merge_hparams,
)
from gymnax_rl.core.gymnax_wrappers import make_env
from gymnax_rl.core.gymnax_network import (
    initialize_actor_critic,
    initialize_split_flax_train_state,
)
from gymnax_rl.core.gymnax_critics import (
    get_critic_loss_fn,
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    ppo_actor_loss_fn,
)
from gymnax_rl.core.runtime_metrics import compute_runtime_metrics

SAVE_DIR = "ppo_double_sample"


def make_train(base_config):
    """
    Factory creating the JAX-vmappable and JIT-compilable training function
    for Gymnax with native 1-step double sampling.
    """
    cfg = base_default_config.copy()
    cfg.update(base_config)
    config_copy = cfg
    batch_size = config_copy["NUM_STEPS"] * config_copy["NUM_ENVS"]
    config_copy["NUM_UPDATES"] = max(1, config_copy["TOTAL_TIMESTEPS"] // batch_size)

    env, env_params = make_env(config_copy)
    obs_shape = env.observation_space(env_params).shape

    def train(rng, hparams=None):
        config = merge_hparams(config_copy, hparams)
        c_type = config["CRITIC_TYPE"].lower()
        critic_loss_fn = get_critic_loss_fn(c_type, split=True)
        actor_loss_fn = ppo_actor_loss_fn
        batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
        config["NUM_UPDATES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)

        # Actor and Critic epochs & minibatch sizes
        actor_epochs = config["ACTOR_EPOCHS"] if config["ACTOR_EPOCHS"] is not None else config["NUM_EPOCHS"]
        critic_epochs = config["CRITIC_EPOCHS"] if config["CRITIC_EPOCHS"] is not None else config["NUM_EPOCHS"]

        actor_mb_size = (
            config["ACTOR_MINIBATCH_SIZE"] if config["ACTOR_MINIBATCH_SIZE"] is not None else config["MINIBATCH_SIZE"]
        )
        actor_mb_size = min(batch_size, actor_mb_size)

        critic_mb_size = (
            config["CRITIC_MINIBATCH_SIZE"] if config["CRITIC_MINIBATCH_SIZE"] is not None else config["MINIBATCH_SIZE"]
        )
        critic_mb_size = min(batch_size, critic_mb_size)

        config["ACTOR_EPOCHS"] = actor_epochs
        config["CRITIC_EPOCHS"] = critic_epochs
        config["ACTOR_MINIBATCH_SIZE"] = actor_mb_size
        config["CRITIC_MINIBATCH_SIZE"] = critic_mb_size
        config["ACTOR_NUM_MINIBATCHES"] = max(1, batch_size // actor_mb_size)
        config["CRITIC_NUM_MINIBATCHES"] = max(1, batch_size // critic_mb_size)

        # Lambdas
        gae_lambda = config["GAE_LAMBDA"]
        target_lambda = get_critic_target_lambda(config)

        # Network & Split TrainState (Standard ActorCritic network, no auxiliary network needed)
        network, network_params = initialize_actor_critic(
            rng=rng,
            obs_shape=obs_shape,
            env=env,
            env_params=env_params,
            config=config,
        )
        train_state = initialize_split_flax_train_state(config, network, network_params)

        # Initialize Environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        start_obs = obsv
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES WITH 1-STEP DOUBLE SAMPLING
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                # SELECT ACTIONS & VALUE
                rng, rng_a1, rng_a2, rng_s1, rng_s2 = jax.random.split(rng, 5)
                pi, value = network.apply(train_state.params, last_obs)

                # Sample two independent policy actions from same observation
                action_1 = pi.sample(seed=rng_a1)
                action_2 = pi.sample(seed=rng_a2)
                log_prob_1 = pi.log_prob(action_1)

                # STEP ENV - Branch 1 (Primary rollout)
                rng_step1 = jax.random.split(rng_s1, config["NUM_ENVS"])
                obsv_1, next_env_state, reward_1, done_1, info_1 = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step1, env_state, action_1, env_params)

                next_obs_1 = info_1["real_next_obs"]
                is_timeout_1 = info_1["is_timeout"]
                true_terminal_1 = done_1 & ~is_timeout_1
                is_start = last_true_terminal

                # STEP ENV - Branch 2 (Auxiliary probe from exact same env_state)
                rng_step2 = jax.random.split(rng_s2, config["NUM_ENVS"])
                obsv_2, _, reward_2, done_2, info_2 = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step2, env_state, action_2, env_params)

                probe_next_obs = info_2["real_next_obs"]
                probe_is_timeout = info_2["is_timeout"]

                transition = Double_Sample_Transition(
                    done=done_1,
                    action=action_1,
                    value=value,
                    reward=reward_1,
                    log_prob=log_prob_1,
                    obs=last_obs,
                    next_obs=next_obs_1,
                    info=info_1,
                    is_start=is_start,
                    is_timeout=is_timeout_1,
                    probe_action=action_2,
                    probe_reward=reward_2,
                    probe_done=done_2,
                    probe_next_obs=probe_next_obs,
                    probe_is_timeout=probe_is_timeout,
                )
                # Rollout loop continues along Branch 1
                return (train_state, next_env_state, obsv_1, true_terminal_1, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. CALCULATE ADVANTAGES & TARGETS
            final_next_val = network.apply(
                train_state.params, traj_batch.next_obs[-1], method=network.value
            )
            if final_next_val.ndim > traj_batch.value.ndim - 1:
                final_next_val = final_next_val.squeeze(-1)
            regular_next_val = jnp.concatenate([traj_batch.value[1:], final_next_val[None]], axis=0)
            timeout_next_val = network.apply(train_state.params, traj_batch.next_obs, method=network.value)
            if timeout_next_val.ndim > regular_next_val.ndim:
                timeout_next_val = timeout_next_val.squeeze(-1)
            next_value = jnp.where(traj_batch.is_timeout, timeout_next_val, regular_next_val)

            # Probe branch next value continuation
            probe_next_val = network.apply(train_state.params, traj_batch.probe_next_obs, method=network.value)
            if probe_next_val.ndim > traj_batch.probe_reward.ndim:
                probe_next_val = probe_next_val.squeeze(-1)

            advantages, gae_targets = calculate_gae(
                traj_batch,
                next_value,
                gamma=config["GAMMA"],
                gae_lambda=gae_lambda,
            )
            if target_lambda == gae_lambda:
                targets = gae_targets
            else:
                targets = calculate_targets(
                    traj_batch,
                    next_value,
                    gamma=config["GAMMA"],
                    target_lambda=target_lambda,
                )

            traj_batch = traj_batch._replace(
                next_value=next_value,
                probe_next_value=probe_next_val,
                targets=targets,
            )

            # 3. CRITIC UPDATE PHASE
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    traj_mb = mb

                    def _c_loss(critic_params):
                        return critic_loss_fn(
                            critic_params,
                            train_state.actor.params,
                            network,
                            traj_mb,
                            traj_mb.targets,
                            config,
                        )

                    (loss, aux), grads = jax.value_and_grad(_c_loss, has_aux=True)(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads=grads)
                    return train_state, aux

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(_rng, critic_batch, config["CRITIC_NUM_MINIBATCHES"])
                train_state, c_losses = jax.lax.scan(_critic_minibatch, train_state, minibatches)
                return (train_state, critic_batch, rng), c_losses

            critic_batch = traj_batch
            rng, _rng = jax.random.split(rng)
            c_init_state = (train_state, critic_batch, _rng)
            (train_state, _, rng), c_loss_epochs = jax.lax.scan(
                _critic_epoch, c_init_state, None, config["CRITIC_EPOCHS"]
            )

            # 4. ACTOR UPDATE PHASE
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    traj_mb, adv_mb = mb

                    def _a_loss(actor_params):
                        return actor_loss_fn(
                            actor_params,
                            train_state.critic.params,
                            network,
                            traj_mb,
                            adv_mb,
                            config,
                        )

                    (loss, aux), grads = jax.value_and_grad(_a_loss, has_aux=True)(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads=grads)
                    return train_state, aux

                train_state, actor_batch, rng = actor_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(_rng, actor_batch, config["ACTOR_NUM_MINIBATCHES"])
                train_state, a_losses = jax.lax.scan(_actor_minibatch, train_state, minibatches)
                return (train_state, actor_batch, rng), a_losses

            actor_batch = (traj_batch, advantages)
            rng, _rng = jax.random.split(rng)
            a_init_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), a_loss_epochs = jax.lax.scan(
                _actor_epoch, a_init_state, None, config["ACTOR_EPOCHS"]
            )

            # 5. RUNTIME & DIAGNOSTIC METRICS
            combined_loss_info = {**c_loss_epochs, **a_loss_epochs}

            # Unbiased MSBE diagnostic computed on rollout batch: E[delta_1 * delta_2]
            true_term_1 = traj_batch.done & ~traj_batch.is_timeout
            true_term_2 = traj_batch.probe_done & ~traj_batch.probe_is_timeout
            delta_1 = (
                traj_batch.reward
                + config["GAMMA"] * (1.0 - true_term_1) * next_value
                - traj_batch.value
            )
            delta_2 = (
                traj_batch.probe_reward
                + config["GAMMA"] * (1.0 - true_term_2) * probe_next_val
                - traj_batch.value
            )
            diagnostic_msbe_rollout = jnp.mean(delta_1 * delta_2)

            metric = compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=combined_loss_info,
                start_obs=start_obs,
                config=config,
            )
            metric["pi_idx"] = idx
            metric["pi_num"] = idx
            metric["advantage"] = advantages.mean()
            metric["diagnostic_msbe"] = diagnostic_msbe_rollout

            runner_state = (train_state, env_state, last_obs, last_true_terminal, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from gymnax_rl.core.evaluator import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

# q_ppo.py
# PPO with Action-Value Function Q(s, a) and Closed-Form Marginalized State Value:
# V(s) = sum_{a in A} pi(a | s) * Q(s, a)
# Critic can be trained via:
# 1. SARSA(lambda) regression (fitted)
# 2. E(0) State-Action Dirichlet Minimization (No V_aux needed!)
# 3. E(lambda) Geometric Jump State-Action Dirichlet Minimization
from __future__ import annotations
import os
import sys
from typing import NamedTuple, Any, Optional

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp

from gymnax_rl.core.gymnax_config import config as base_default_config
from gymnax_rl.core.gymnax_utils import (
    shuffle_and_batch,
    merge_hparams,
    post_process_advantage,
)
from gymnax_rl.core.gymnax_wrappers import make_env
from gymnax_rl.core.gymnax_network import initialize_split_flax_train_state
from gymnax_rl.core.gymnax_q_network import initialize_q_actor_critic
from gymnax_rl.core.gymnax_q_critics import get_q_critic_loss_fn, extract_q_taken
from gymnax_rl.core.gymnax_critics import calculate_gae, calculate_targets, get_critic_target_lambda, ppo_actor_loss_fn
from gymnax_rl.core.runtime_metrics import compute_runtime_metrics

SAVE_DIR = "q_ppo"


class QTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray          # Marginalized V(s) = sum_a pi(a|s) Q(s,a)
    q_taken: jnp.ndarray        # Q(s, a_taken)
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    next_action: Optional[jnp.ndarray] = None
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    dest_obs: Optional[jnp.ndarray] = None
    dest_action: Optional[jnp.ndarray] = None
    dest_targets: Optional[jnp.ndarray] = None
    hit_true_terminal: Optional[jnp.ndarray] = None
    valid_jump: Optional[jnp.ndarray] = None


def prepare_q_jump_transitions(traj_batch, targets, last_obs, last_action, gamma, lmbda, rng):
    """Samples geometric lookahead jump state-action pairs for E(lambda) Q-critic."""
    T, B = targets.shape[:2]
    gl = gamma * lmbda

    u = jax.random.uniform(rng, shape=(T, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    jumps = jnp.maximum(jumps, 0)

    t_arr = jnp.arange(T)[:, None]
    target_idx = t_arr + jumps + 1
    t_clamped = jnp.minimum(target_idx, T)

    t_prev = jnp.minimum(t_arr + jumps, T - 1)
    done_cumsum = jnp.cumsum(traj_batch.done.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    is_absorbing = true_terminal

    next_target_T = jnp.where(true_terminal[-1], 0.0, traj_batch.next_value[-1])
    obs_ext = jnp.concatenate([traj_batch.obs, last_obs[None]], axis=0)
    act_ext = jnp.concatenate([traj_batch.action, last_action[None]], axis=0)
    targets_ext = jnp.concatenate([targets, next_target_T[None]], axis=0)

    idx_expanded = t_clamped.reshape(t_clamped.shape + (1,) * (traj_batch.obs.ndim - 2))
    idx_broadcast = jnp.broadcast_to(idx_expanded, (T, B) + traj_batch.obs.shape[2:])
    dest_obs_gathered = jnp.take_along_axis(obs_ext, idx_broadcast, axis=0)
    dest_act_gathered = jnp.take_along_axis(act_ext, t_clamped, axis=0)
    dest_targets_gathered = jnp.take_along_axis(targets_ext, t_clamped, axis=0)

    dest_obs = jnp.where(traj_batch.is_timeout[..., None], traj_batch.next_obs, dest_obs_gathered)
    dest_act = jnp.where(traj_batch.is_timeout, traj_batch.next_action, dest_act_gathered)
    dest_targets = jnp.where(traj_batch.is_timeout, traj_batch.next_value, dest_targets_gathered)

    valid_jump = jnp.where(
        true_terminal,
        False,
        jnp.where(traj_batch.is_timeout, jumps == 0, ~has_reset_between),
    )

    return traj_batch._replace(
        dest_obs=dest_obs,
        dest_action=dest_act,
        dest_targets=dest_targets,
        hit_true_terminal=is_absorbing,
        valid_jump=valid_jump,
    )


def make_train(base_config):
    """
    Factory creating the Q-PPO training function with marginalized GAE
    and modular Q-critic loss (SARSA, E(0)-Q, E(lambda)-Q).
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
        c_type = config.get("CRITIC_TYPE", "sarsa").lower()
        critic_loss_fn = get_q_critic_loss_fn(c_type)
        actor_loss_fn = ppo_actor_loss_fn

        actor_epochs = config.get("ACTOR_EPOCHS")
        if actor_epochs is None:
            actor_epochs = config.get("NUM_EPOCHS", 4)
        critic_epochs = config.get("CRITIC_EPOCHS")
        if critic_epochs is None:
            critic_epochs = config.get("NUM_EPOCHS", 4)

        actor_mb_size = config.get("ACTOR_MINIBATCH_SIZE")
        if actor_mb_size is None:
            actor_mb_size = config.get("MINIBATCH_SIZE", 512)
        actor_mb_size = min(batch_size, actor_mb_size)

        critic_mb_size = config.get("CRITIC_MINIBATCH_SIZE")
        if critic_mb_size is None:
            critic_mb_size = config.get("MINIBATCH_SIZE", 512)
        critic_mb_size = min(batch_size, critic_mb_size)

        config["ACTOR_EPOCHS"] = actor_epochs
        config["CRITIC_EPOCHS"] = critic_epochs
        config["ACTOR_MINIBATCH_SIZE"] = actor_mb_size
        config["CRITIC_MINIBATCH_SIZE"] = critic_mb_size
        config["ACTOR_NUM_MINIBATCHES"] = max(1, batch_size // actor_mb_size)
        config["CRITIC_NUM_MINIBATCHES"] = max(1, batch_size // critic_mb_size)

        gae_lambda = config.get("GAE_LAMBDA", 0.8)
        target_lambda = get_critic_target_lambda(config)

        rng, net_rng = jax.random.split(rng)
        network, network_params = initialize_q_actor_critic(
            net_rng, obs_shape, env, env_params, config
        )
        train_state = initialize_split_flax_train_state(config, network, network_params)

        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env_params)
        start_obs = obsv
        last_true_terminal = jnp.zeros((config["NUM_ENVS"],), dtype=bool)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # 1. ROLLOUT COLLECTION
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value, q_all = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                q_taken = extract_q_taken(q_all, action)

                rng, _rng = jax.random.split(rng)
                step_keys = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, env_state, action, env_params)

                next_obs = info["real_next_obs"]
                is_timeout = info["is_timeout"]
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                transition = QTransition(
                    done=done,
                    action=action,
                    value=value,
                    q_taken=q_taken,
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
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # Next-action alignment along rollout
            rng, _rng = jax.random.split(rng)
            pi_last, last_val, q_last = network.apply(train_state.params, last_obs)
            last_action = pi_last.sample(seed=_rng)
            next_action = jnp.concatenate([traj_batch.action[1:], last_action[None]], axis=0)

            # 2. ADVANTAGE & VALUE TARGET CALCULATION (Analytical Marginalized GAE)
            regular_next_val = jnp.concatenate([traj_batch.value[1:], last_val[None]], axis=0)
            timeout_next_val = network.apply(train_state.params, traj_batch.next_obs, method=network.value)
            if timeout_next_val.ndim > regular_next_val.ndim:
                timeout_next_val = timeout_next_val.squeeze(-1)
            next_value = jnp.where(traj_batch.is_timeout, timeout_next_val, regular_next_val)

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

            true_terminal = traj_batch.done & ~traj_batch.is_timeout
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(last_val)
            next_targets = jnp.where(traj_batch.is_timeout, next_value, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)

            traj_batch = traj_batch._replace(
                next_action=next_action,
                next_value=next_value,
                targets=targets,
                next_targets=next_targets,
            )

            # Prepare geometric jumps if E(lambda)
            if "lambda" in c_type or "geometric" in c_type:
                rng, _rng = jax.random.split(rng)
                traj_batch = prepare_q_jump_transitions(
                    traj_batch,
                    targets,
                    last_obs,
                    last_action,
                    gamma=config["GAMMA"],
                    lmbda=config.get("E_LAMBDA", config.get("VALUE_LAMBDA", 0.8)),
                    rng=_rng,
                )

            # 3. CRITIC UPDATE PHASE (Updates Q)
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

            # 4. ACTOR UPDATE PHASE (Updates Policy via PPO clipped surrogate on GAE)
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

            # 5. RUNTIME METRICS
            combined_loss_info = {**c_loss_epochs, **a_loss_epochs}
            metric = compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=combined_loss_info,
                start_obs=start_obs,
                config=config,
            )
            metric["advantage"] = advantages.mean()
            metric["q_pred"] = traj_batch.q_taken.mean()

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

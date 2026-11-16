"""Fresh-batch SAC for Brax with TD(lambda), TD(0), E(0), and E(lambda) + v_aux.

Follows brax_rl/algos/ppo.py environment preprocessing, split TrainState
optimization, and metrics structure, using a tanh-squashed Gaussian policy.
"""
from __future__ import annotations

import copy
import os
import sys
from typing import Any, NamedTuple

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import jax
import jax.numpy as jnp
import numpy as np
import optax

from brax_rl.core.brax_network import initialize_split_train_state
from brax_rl.core.brax_sac_network import BraxSACDualCritic, BraxSACNetwork
from brax_rl.core.brax_utils import shuffle_and_batch
from brax_rl.core.brax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    NormalizeVecObservation,
    NormalizeVecReward,
    VecEnv,
)
from brax_rl.core.sac_config import config as base_default_config


class BraxSACTransition(NamedTuple):
    done: jax.Array
    action: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    next_obs: jax.Array
    info: Any
    is_timeout: jax.Array


def _soft_lambda_targets(reward, done, is_timeout, next_value, log_prob, gamma, lambda_, alpha):
    next_logp = jnp.concatenate([log_prob[1:], jnp.zeros_like(log_prob[:1])])
    trace = jnp.where(done, 0.0, lambda_)
    trace = trace.at[-1].set(0.0)
    terminal = done & ~is_timeout

    def backward(carry, step):
        r, end, v, next_lp, weight = step
        continuation = (1.0 - weight) * v + weight * (carry - alpha * next_lp)
        target = r + gamma * jnp.where(end, 0.0, continuation)
        return target, target

    _, targets = jax.lax.scan(
        backward, jnp.zeros_like(reward[0]),
        (reward, terminal, next_value, next_logp, trace), reverse=True)
    return jax.lax.stop_gradient(targets)


def _prepare_e_lambda_jumps(batch, gamma, e_lambda, rng, steps_since_start):
    T, B = batch.reward.shape
    gl = gamma * e_lambda
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    u = jax.random.uniform(rng, shape=(T, B))
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T)[:, None]
    j_idx = jnp.minimum(t_arr + m, T)

    is_timeout = batch.is_timeout
    true_terminals = batch.done & ~is_timeout
    term_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)], axis=0)
    timeout_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)], axis=0)
    hit_true_terminal = (jnp.take_along_axis(term_cumsum, j_idx, axis=0) - term_cumsum[:T]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum, j_idx, axis=0) - timeout_cumsum[:T]) > 0
    hit_horizon = (t_arr + m) > T

    def scan_steps(k, flags):
        term, timeout = flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    next_steps_since_start, all_steps = jax.lax.scan(
        scan_steps, steps_since_start, (true_terminals, is_timeout))
    tilde_gamma = (gamma * (1.0 - e_lambda)) / jnp.maximum(1.0 - gl, 1e-8)
    start_trace = jnp.where(gl < 1e-6, jnp.where(all_steps == 0, 1.0, 0.0), safe_gl ** all_steps)
    start_weight = 0.25 * tilde_gamma * start_trace

    obs_ext = jnp.concatenate([batch.obs, batch.next_obs[-1][None]], axis=0)
    j_obs = jnp.broadcast_to(j_idx.reshape(j_idx.shape + (1,) * (obs_ext.ndim - 2)), (T, B) + obs_ext.shape[2:])
    dest_obs = jnp.take_along_axis(obs_ext, j_obs, axis=0)
    return (dest_obs, j_idx, hit_true_terminal, hit_timeout, hit_horizon, start_weight), next_steps_since_start


def _soft_e_lambda_loss(
    q_start, q_aux_start, v_jump, v_aux_jump,
    targets, dest_targets,
    hit_true_terminal, hit_timeout, hit_horizon, start_weight,
    gamma, e_lambda,
):
    sg = jax.lax.stop_gradient
    e_start = sg(targets) - q_start
    e_jump = jnp.where(hit_true_terminal, 0.0, sg(dest_targets) - v_jump)
    e_aux_start = sg(q_aux_start) - q_start
    e_aux_jump = jnp.where(hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

    inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2
    diff_sq = jnp.where(hit_true_terminal | hit_timeout | hit_horizon, e_start ** 2, inner_diff_sq)

    gl = gamma * e_lambda
    denom = jnp.maximum(1.0 - gl, 1e-8)
    tilde_gamma = (gamma * (1.0 - e_lambda)) / denom
    magnitude_weight = 0.5 * (1.0 - gamma) / denom
    dirichlet_weight = 0.5 * tilde_gamma

    main_loss = (
        magnitude_weight * jnp.mean(e_start ** 2)
        + dirichlet_weight * jnp.mean(diff_sq)
        + jnp.mean(start_weight * (e_start ** 2))
    )
    aux_loss = 0.5 * jnp.mean((q_aux_start - sg(targets)) ** 2)
    return main_loss + aux_loss


def make_train(config):
    cfg = copy.deepcopy(base_default_config)
    cfg.update(config)
    config = cfg

    critic_aliases = {
        "td_lambda": "td_lambda", "td": "td_lambda", "fitted": "td_lambda",
        "td_0": "td_0", "td0": "td_0", "online_td0": "td_0",
        "e": "e_0", "e_0": "e_0", "e0": "e_0", "e_min": "e_0",
        "e_lambda": "e_lambda", "elambda": "e_lambda", "e_geometric": "e_lambda",
    }
    c_type = critic_aliases.get(str(config["CRITIC_TYPE"]).lower())
    if c_type is None:
        raise ValueError(f"Unsupported CRITIC_TYPE '{config['CRITIC_TYPE']}'")
    if not 0 <= config["TD_LAMBDA"] <= 1 or not 0 <= config["RETURN_LAMBDA"] <= 1 or not 0 <= config["E_LAMBDA"] <= 1:
        raise ValueError("Require TD_LAMBDA, RETURN_LAMBDA, E_LAMBDA in [0, 1]")
    if not 0 <= config["GAMMA"] < 1:
        raise ValueError("Require GAMMA in [0, 1)")
    if config["ALPHA"] < 0 or config["KL_COEFF"] < 0:
        raise ValueError("ALPHA and KL_COEFF must be nonnegative")
    if config["KL_TARGET"] is not None and (config["KL_TARGET"] <= 0 or config["KL_COEFF"] <= 0):
        raise ValueError("Adaptive KL requires positive KL_TARGET and initial KL_COEFF")
    if config["AUTOTUNE_ALPHA"] and (config["ALPHA"] <= 0 or config["ALPHA_LR"] <= 0):
        raise ValueError("AUTOTUNE_ALPHA requires positive initial ALPHA and ALPHA_LR")

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["MAX_NUM_POLICIES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)
    config["NUM_UPDATES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)

    net_cfg = config["NETWORK_CONFIG"]
    actor_minibatch_size = min(batch_size, config["ACTOR_MINIBATCH_SIZE"])
    critic_minibatch_size = min(batch_size, config["CRITIC_MINIBATCH_SIZE"])
    if batch_size % actor_minibatch_size or batch_size % critic_minibatch_size:
        raise ValueError("Minibatch sizes must divide batch_size = NUM_ENVS * NUM_STEPS")
    config["ACTOR_MINIBATCH_SIZE"] = actor_minibatch_size
    config["CRITIC_MINIBATCH_SIZE"] = critic_minibatch_size
    config["ACTOR_NUM_MINIBATCHES"] = batch_size // actor_minibatch_size
    config["CRITIC_NUM_MINIBATCHES"] = batch_size // critic_minibatch_size

    # Identical environment pre-processing to brax_rl/algos/ppo.py
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if net_cfg["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    action_dim = env.action_space(env_params).shape[0]
    target_entropy = config["TARGET_ENTROPY"]
    if target_entropy is None:
        target_entropy = action_dim * (float(np.log(2.0)) - 0.39)
    target_lambda = (
        config["RETURN_LAMBDA"] if c_type in ("e_0", "e_lambda")
        else 0.0 if c_type == "td_0" else config["TD_LAMBDA"]
    )

    def train(rng):
        if c_type == "e_lambda":
            network = BraxSACDualCritic(
                action_dim=action_dim,
                activation=net_cfg["ACTIVATION"],
                layer_norm=config["LAYER_NORM"],
                shared_torso=config["SHARED_TORSO"],
            )
        else:
            network = BraxSACNetwork(
                action_dim=action_dim,
                activation=net_cfg["ACTIVATION"],
                layer_norm=config["LAYER_NORM"],
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        init_a = jnp.zeros((action_dim,))
        network_params = network.init(_rng, init_x)
        # Ensure Q-heads are initialized cleanly with action inputs.
        _ = network.apply(network_params, init_x, init_a, method=network.q_value)

        train_state = initialize_split_train_state(config, network, network_params)
        temperature_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": jnp.log(max(config["ALPHA"], 1e-8))},
            tx=optax.adam(config["ALPHA_LR"]),
        )
        kl_coeff = jnp.asarray(config["KL_COEFF"], dtype=jnp.float32)
        steps_since_start = jnp.zeros((config["NUM_ENVS"],), dtype=jnp.int32)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(jax.random.split(_rng, config["NUM_ENVS"]), env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, temperature_state, kl_coeff, steps_since_start, idx = runner_state
            alpha = (
                jnp.exp(temperature_state.params["log_alpha"])
                if config["AUTOTUNE_ALPHA"] else jnp.asarray(config["ALPHA"], dtype=jnp.float32)
            )

            # 1. COLLECT ON-POLICY ROLLOUT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state
                rng, ak, ek = jax.random.split(rng, 3)
                action, log_prob = network.apply(
                    train_state.params, last_obs, ak, method=network.sample_action)
                obsv, env_state, reward, done, info = env.step(
                    jax.random.split(ek, config["NUM_ENVS"]), env_state, action, env_params)
                transition = BraxSACTransition(
                    done=done,
                    action=action,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=info["real_next_obs"],
                    info=info,
                    is_timeout=info["is_timeout"],
                )
                return (train_state, env_state, obsv, rng), transition

            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["NUM_STEPS"])

            # 2. SOFT BOOTSTRAP & TARGETS
            rng, return_key = jax.random.split(rng)
            return_action, return_logp = network.apply(
                train_state.params, traj_batch.next_obs, return_key, method=network.sample_action)

            def _compute_targets(critic_params):
                full_params = {"params": {**train_state.actor.params, **critic_params}}
                next_q = network.apply(
                    full_params, traj_batch.next_obs, return_action, method=network.q_value)
                next_val = next_q - alpha * return_logp
                new_targets = _soft_lambda_targets(
                    traj_batch.reward, traj_batch.done, traj_batch.is_timeout,
                    next_val, traj_batch.log_prob, config["GAMMA"], target_lambda, alpha)
                new_last_val = jnp.where(
                    traj_batch.done[-1] & ~traj_batch.is_timeout[-1], 0.0, next_val[-1])
                return new_targets, new_last_val

            targets, last_val = _compute_targets(train_state.critic.params)
            if c_type == "e_lambda":
                rng, jump_key = jax.random.split(rng)
                jump_data, steps_since_start = _prepare_e_lambda_jumps(
                    traj_batch, config["GAMMA"], config["E_LAMBDA"], jump_key, steps_since_start)
                dest_obs, j_idx, hit_true_terminal, hit_timeout, hit_horizon, start_weight = jump_data

            # 3. CRITIC UPDATE PHASE
            def _critic_epoch(carry, unused):
                train_state, rng = carry
                rng, key = jax.random.split(rng)

                if c_type == "e_lambda":
                    state_targets = jnp.concatenate(
                        [targets - alpha * traj_batch.log_prob, last_val[None]], axis=0)
                    dest_targets = jnp.take_along_axis(state_targets, j_idx, axis=0)
                    mb_data = (
                        traj_batch, targets, dest_obs, dest_targets,
                        hit_true_terminal, hit_timeout, hit_horizon, start_weight,
                    )
                else:
                    mb_data = (traj_batch, targets)

                minibatches = shuffle_and_batch(key, mb_data, config["CRITIC_NUM_MINIBATCHES"])

                def _critic_minibatch(carry, mb):
                    train_state, rng = carry
                    if c_type == "e_lambda":
                        (traj_mb, target_mb, mb_dest_obs, mb_dest_target,
                         mb_hit_term, mb_hit_to, mb_hit_hor, mb_sw) = mb
                        rng, dest_key = jax.random.split(rng)
                        dest_action, dest_logp = network.apply(
                            train_state.params, mb_dest_obs, dest_key, method=network.sample_action)
                        dest_action, dest_logp = jax.tree_util.tree_map(
                            jax.lax.stop_gradient, (dest_action, dest_logp))
                    else:
                        traj_mb, target_mb = mb
                        if c_type in ("e_0", "td_0"):
                            rng, probe_key = jax.random.split(rng)
                            probe_action, probe_logp = network.apply(
                                train_state.params, traj_mb.next_obs, probe_key, method=network.sample_action)
                            probe_action, probe_logp = jax.tree_util.tree_map(
                                jax.lax.stop_gradient, (probe_action, probe_logp))

                    def _loss_fn(critic_params):
                        full_params = {"params": {**train_state.actor.params, **critic_params}}
                        if c_type == "e_lambda":
                            q_start, q_aux_start = network.apply(
                                full_params, traj_mb.obs, traj_mb.action, method=network.both_q)
                            q_jump, q_aux_jump = network.apply(
                                full_params, mb_dest_obs, dest_action, method=network.both_q)
                            v_jump = q_jump - alpha * dest_logp
                            v_aux_jump = q_aux_jump - alpha * dest_logp
                            return _soft_e_lambda_loss(
                                q_start, q_aux_start, v_jump, v_aux_jump,
                                target_mb, mb_dest_target,
                                mb_hit_term, mb_hit_to, mb_hit_hor, mb_sw,
                                config["GAMMA"], config["E_LAMBDA"],
                            )
                        q_taken = network.apply(
                            full_params, traj_mb.obs, traj_mb.action, method=network.q_value)
                        if c_type == "e_0":
                            probe_q = network.apply(
                                full_params, traj_mb.next_obs, probe_action, method=network.q_value)
                            terminal = traj_mb.done & ~traj_mb.is_timeout
                            continuation = probe_q - alpha * jax.lax.stop_gradient(probe_logp)
                            delta = traj_mb.reward + config["GAMMA"] * jnp.where(terminal, 0.0, continuation) - q_taken
                            error = jax.lax.stop_gradient(target_mb) - q_taken
                            return 0.5 * jnp.mean(error * delta)
                        if c_type == "td_0":
                            next_q = network.apply(
                                full_params, traj_mb.next_obs, probe_action, method=network.q_value)
                            terminal = traj_mb.done & ~traj_mb.is_timeout
                            td_target = jax.lax.stop_gradient(
                                traj_mb.reward + config["GAMMA"] * jnp.where(
                                    terminal, 0.0, next_q - alpha * probe_logp))
                            return 0.5 * jnp.mean((q_taken - td_target) ** 2)
                        return 0.5 * jnp.mean((q_taken - target_mb) ** 2)

                    loss, grads = jax.value_and_grad(_loss_fn)(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads)
                    return (train_state, rng), loss

                (train_state, rng), losses = jax.lax.scan(_critic_minibatch, (train_state, rng), minibatches)
                return (train_state, rng), losses.mean()

            (train_state, rng), critic_losses = jax.lax.scan(
                _critic_epoch, (train_state, rng), None, config["CRITIC_EPOCHS"])

            # 4. ACTOR + TEMPERATURE + ADAPTIVE KL UPDATE PHASE
            old_mean, old_log_std = jax.tree_util.tree_map(
                jax.lax.stop_gradient,
                network.apply(train_state.params, traj_batch.obs, method=network.actor_stats),
            )

            def _actor_epoch(carry, unused):
                train_state, rng, temperature_state = carry
                rng, key = jax.random.split(rng)
                minibatches = shuffle_and_batch(
                    key, (traj_batch.obs, old_mean, old_log_std), config["ACTOR_NUM_MINIBATCHES"])

                def _actor_minibatch(carry, mb):
                    train_state, rng, temperature_state = carry
                    obs_mb, om_mb, ols_mb = mb
                    actor_alpha = (
                        jnp.exp(temperature_state.params["log_alpha"])
                        if config["AUTOTUNE_ALPHA"] else alpha
                    )
                    rng, ak = jax.random.split(rng)

                    def _loss_fn(actor_params):
                        full_params = {"params": {**actor_params, **train_state.critic.params}}
                        action, logp = network.apply(
                            full_params, obs_mb, ak, method=network.sample_action)
                        q = network.apply(full_params, obs_mb, action, method=network.q_value)
                        loss = jnp.mean(actor_alpha * logp - q)
                        if config["KL_COEFF"] > 0 or config["KL_TARGET"] is not None:
                            m, ls = network.apply(full_params, obs_mb, method=network.actor_stats)
                            kl = ols_mb - ls + (jnp.exp(2 * ls) + (m - om_mb)**2) / (2 * jnp.exp(2 * ols_mb)) - 0.5
                            loss = loss + kl_coeff * jnp.sum(kl, axis=-1).mean()
                        return loss, -jnp.mean(logp)

                    (loss, ent), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads)

                    alpha_loss = jnp.asarray(0.0)
                    if config["AUTOTUNE_ALPHA"]:
                        rng, tk = jax.random.split(rng)
                        _, lp_new = network.apply(
                            train_state.params, obs_mb, tk, method=network.sample_action)
                        ent_new = -jnp.mean(lp_new)

                        def _alpha_loss(p):
                            return p["log_alpha"] * jax.lax.stop_gradient(ent_new - target_entropy)

                        alpha_loss, a_grads = jax.value_and_grad(_alpha_loss)(temperature_state.params)
                        temperature_state = temperature_state.apply_gradients(grads=a_grads)

                    return (train_state, rng, temperature_state), (loss, ent, alpha_loss)

                (train_state, rng, temperature_state), (losses, ents, alpha_losses) = jax.lax.scan(
                    _actor_minibatch, (train_state, rng, temperature_state), minibatches)
                return (train_state, rng, temperature_state), (losses.mean(), ents.mean(), alpha_losses.mean())

            (train_state, rng, temperature_state), (actor_losses, entropies, alpha_losses) = jax.lax.scan(
                _actor_epoch, (train_state, rng, temperature_state), None, config["ACTOR_EPOCHS"])

            new_mean, new_log_std = network.apply(
                train_state.params, traj_batch.obs, method=network.actor_stats)
            step_kl = jnp.sum(
                old_log_std - new_log_std + (jnp.exp(2 * new_log_std) + (new_mean - old_mean)**2) / (
                    2 * jnp.exp(2 * old_log_std)) - 0.5,
                axis=-1,
            ).mean()
            if config["KL_TARGET"] is not None:
                kl_target = config["KL_TARGET"] * action_dim
                kl_coeff = jnp.where(
                    step_kl > 1.5 * kl_target,
                    jnp.minimum(kl_coeff * 2.0, 100.0),
                    jnp.where(step_kl < kl_target / 1.5, jnp.maximum(kl_coeff * 0.5, 1e-4), kl_coeff),
                )

            # 5. RUNTIME METRICS (matching brax_rl/algos/ppo.py)
            metric = {
                "pi_idx": idx,
                "pi_num": idx,
                "critic_loss": critic_losses.mean(),
                "actor_loss": actor_losses.mean(),
                "entropy": entropies.mean(),
                "alpha": (
                    jnp.exp(temperature_state.params["log_alpha"])
                    if config["AUTOTUNE_ALPHA"] else jnp.asarray(config["ALPHA"])
                ),
                "alpha_loss": alpha_losses.mean(),
                "actor_step_kl": step_kl,
                "kl_coeff": kl_coeff,
            }
            metric.update({
                k: v.mean() for k, v in traj_batch.info.items()
                if k not in ["real_next_obs", "obs_jump", "targets_jump", "valid_jump", "is_absorbing"]
                and hasattr(v, "mean") and v.ndim <= 2
            })

            runner_state = (
                train_state, env_state, last_obs, rng, temperature_state,
                kl_coeff, steps_since_start, idx + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, temperature_state, kl_coeff, steps_since_start, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from brax_rl.core.evaluator import run_experiment_main
    run_experiment_main(make_train)

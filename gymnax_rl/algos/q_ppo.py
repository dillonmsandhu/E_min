# q_ppo.py
# PPO with Action-Value Function Q(s, a):
# - Discrete actions: Closed-form Marginalized State Value V(s) = sum_{a in A} pi(a | s) * Q(s, a)
# - Continuous actions: K-action sampling (K=2 default) with zero-bias probe E(0) and LOO/Hybrid GAE
# Critic can be trained via:
# 1. SARSA(lambda) regression (fitted)
# 2. Bilinear E(0) with exact successor marginalization (discrete) or independent probe action (continuous)
# 3. Legacy sampled E(lambda) Geometric Jump State-Action Dirichlet Minimization
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
from gymnax.environments import spaces

from gymnax_rl.core.gymnax_config import config as base_default_config
from gymnax_rl.core.gymnax_utils import (
    shuffle_and_batch,
    merge_hparams,
    post_process_advantage,
)
from gymnax_rl.core.gymnax_wrappers import make_env
from gymnax_rl.core.gymnax_network import initialize_split_flax_train_state, uses_aux_network
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
    actions_k: Optional[jnp.ndarray] = None
    q_all_k: Optional[jnp.ndarray] = None
    next_probe_actions: Optional[jnp.ndarray] = None
    next_action: Optional[jnp.ndarray] = None
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    next_reward: Optional[jnp.ndarray] = None
    next_next_obs: Optional[jnp.ndarray] = None
    next_true_terminal: Optional[jnp.ndarray] = None
    is_horizon: Optional[jnp.ndarray] = None
    aux_targets: Optional[jnp.ndarray] = None
    dest_obs: Optional[jnp.ndarray] = None
    dest_action: Optional[jnp.ndarray] = None
    dest_targets: Optional[jnp.ndarray] = None
    dest_reward: Optional[jnp.ndarray] = None
    dest_next_obs: Optional[jnp.ndarray] = None
    dest_true_terminal: Optional[jnp.ndarray] = None
    hit_true_terminal: Optional[jnp.ndarray] = None
    valid_jump: Optional[jnp.ndarray] = None


def prepare_q_jump_transitions(traj_batch, targets, last_obs, last_action, gamma, lmbda, rng):
    """Samples geometric lookahead jump state-action pairs within the rollout for E(lambda) Q-critic."""
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
    in_bounds = target_idx < T
    t_clamped = jnp.minimum(target_idx, T - 1)

    t_prev = jnp.minimum(t_arr + jumps, T - 1)
    done_bool = traj_batch.done > 0.5
    timeout_bool = traj_batch.is_timeout > 0.5
    true_terminal = done_bool & ~timeout_bool

    done_cumsum = jnp.cumsum(done_bool.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    def _gather(arr):
        idx = t_clamped.reshape(t_clamped.shape + (1,) * (arr.ndim - 2))
        return jnp.take_along_axis(arr, jnp.broadcast_to(idx, (T, B) + arr.shape[2:]), axis=0)

    valid_jump = (~true_terminal) & (~timeout_bool) & in_bounds & (~has_reset_between)

    return traj_batch._replace(
        dest_obs=_gather(traj_batch.obs),
        dest_action=_gather(traj_batch.action),
        dest_targets=jnp.take_along_axis(targets, t_clamped, axis=0),
        dest_reward=jnp.take_along_axis(traj_batch.reward, t_clamped, axis=0),
        dest_next_obs=_gather(traj_batch.next_obs),
        dest_true_terminal=jnp.take_along_axis(true_terminal, t_clamped, axis=0),
        hit_true_terminal=true_terminal,
        valid_jump=valid_jump,
    )


def make_train(base_config):
    """
    Factory creating the Q-PPO training function (Discrete marginalized GAE /
    Continuous K-action sampling GAE) with modular Q-critic loss (SARSA,
    bilinear E(0), sampled E(lambda)).
    """
    cfg = base_default_config.copy()
    cfg.update(base_config)
    config_copy = cfg

    batch_size = config_copy["NUM_STEPS"] * config_copy["NUM_ENVS"]
    config_copy["NUM_UPDATES"] = max(1, config_copy["TOTAL_TIMESTEPS"] // batch_size)

    env, env_params = make_env(config_copy)
    obs_shape = env.observation_space(env_params).shape
    act_space = env.action_space(env_params)
    is_continuous = isinstance(act_space, spaces.Box)

    def train(rng, hparams=None):
        config = merge_hparams(config_copy, hparams)
        c_type = config["CRITIC_TYPE"].lower()
        if c_type in ("e_0", "e_0_q", "e_0_bilinear", "e_0_dir_marg", "e0", "e", "bsmc_double_pi", "bsmc_dir_marg"):
            # Non-aux bilinear/marginalized Q critics have no auxiliary branch.
            config = {**config, "E_VARIANT": "naive"}
        uses_aux = uses_aux_network(config)
        critic_loss_fn = get_q_critic_loss_fn(c_type)
        actor_loss_fn = ppo_actor_loss_fn

        if is_continuous:
            k_actions = config["K_ACTIONS"]
            if k_actions < 2:
                raise ValueError(f"Continuous Q-PPO requires K_ACTIONS >= 2, got {k_actions}.")
            gae_mode = config["GAE_CONTINUOUS_MODE"].lower()
            if gae_mode not in ("loo", "hybrid"):
                raise ValueError(
                    f"Unrecognized GAE_CONTINUOUS_MODE '{config['GAE_CONTINUOUS_MODE']}'. "
                    "Expected 'loo' or 'hybrid'."
                )
        else:
            k_actions = None
            gae_mode = None

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

        def _sample_and_eval_k(params, obs_batch, key):
            pi = network.apply(params, obs_batch, method=network.policy)
            act_keys = jax.random.split(key, k_actions)
            actions_k = jnp.moveaxis(jax.vmap(lambda k: pi.sample(seed=k))(act_keys), 0, -2)
            q_all_k = jax.vmap(
                lambda a: network.apply(params, obs_batch, a, method=network.q_value),
                in_axes=-2,
                out_axes=-1,
            )(actions_k)
            return pi, actions_k, q_all_k

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
                if is_continuous:
                    pi, actions_k, q_all_k = _sample_and_eval_k(train_state.params, last_obs, _rng)
                    action = actions_k[..., 0, :]
                    log_prob = pi.log_prob(action)
                    q_taken = q_all_k[..., 0]
                    value = jnp.mean(q_all_k[..., 1:], axis=-1)
                    action_env = action.reshape((config["NUM_ENVS"],) + act_space.shape)
                else:
                    pi, value, q_all = network.apply(train_state.params, last_obs)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    q_taken = extract_q_taken(q_all, action)
                    actions_k = None
                    q_all_k = None
                    action_env = action

                rng, _rng = jax.random.split(rng)
                step_keys = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(step_keys, env_state, action_env, env_params)

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
                    actions_k=actions_k,
                    q_all_k=q_all_k,
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. ADVANTAGE & VALUE TARGET CALCULATION
            if is_continuous:
                rng, last_rng, timeout_rng = jax.random.split(rng, 3)
                _, last_actions_k, last_q_k = _sample_and_eval_k(train_state.params, last_obs, last_rng)
                _, timeout_actions_k, timeout_q_k = _sample_and_eval_k(
                    train_state.params, traj_batch.next_obs, timeout_rng
                )

                regular_next_actions_k = jnp.concatenate([traj_batch.actions_k[1:], last_actions_k[None]], axis=0)
                next_actions_k = jnp.where(
                    traj_batch.is_timeout[..., None, None], timeout_actions_k, regular_next_actions_k
                )
                regular_next_q_k = jnp.concatenate([traj_batch.q_all_k[1:], last_q_k[None]], axis=0)
                next_q_k = jnp.where(
                    traj_batch.is_timeout[..., None], timeout_q_k, regular_next_q_k
                )

                last_action = last_actions_k[..., 0, :]
                next_action = next_actions_k[..., 0, :]
                next_probe_actions = next_actions_k[..., 1:, :]

                # Return targets use slot 0 only so targets remain strictly independent
                # of probe actions (slots 1..K-1) at interior, timeout, and final steps.
                v_ret = traj_batch.q_all_k[..., 0]
                next_v_ret = next_q_k[..., 0]
                targets = calculate_targets(
                    traj_batch._replace(value=v_ret),
                    next_v_ret,
                    gamma=config["GAMMA"],
                    target_lambda=target_lambda,
                )
                last_val = last_q_k[..., 0]
                timeout_target_val = next_v_ret

                v_loo = jnp.mean(traj_batch.q_all_k[..., 1:], axis=-1)
                next_v_loo = jnp.mean(next_q_k[..., 1:], axis=-1)
                if gae_mode == "loo":
                    advantages, _ = calculate_gae(
                        traj_batch._replace(value=v_loo),
                        next_v_loo,
                        gamma=config["GAMMA"],
                        gae_lambda=gae_lambda,
                    )
                    next_value = next_v_loo
                else:
                    # Hybrid GAE: averaged bootstrap target + leave-one-out baseline
                    v_bar = jnp.mean(traj_batch.q_all_k, axis=-1)
                    next_v_bar = jnp.mean(next_q_k, axis=-1)
                    v_loo_from_bar = (k_actions * v_bar - traj_batch.q_taken) / (k_actions - 1)
                    _, gae_returns_bar = calculate_gae(
                        traj_batch._replace(value=v_bar),
                        next_v_bar,
                        gamma=config["GAMMA"],
                        gae_lambda=gae_lambda,
                    )
                    advantages = gae_returns_bar - v_loo_from_bar
                    next_value = next_v_bar
            else:
                # Discrete: Analytical Marginalized GAE
                rng, _rng = jax.random.split(rng)
                pi_last, last_val, q_last = network.apply(train_state.params, last_obs)
                last_action = pi_last.sample(seed=_rng)
                next_action = jnp.concatenate([traj_batch.action[1:], last_action[None]], axis=0)
                next_probe_actions = None

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
                timeout_target_val = next_value

            true_terminal = traj_batch.done & ~traj_batch.is_timeout
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(last_val)
            next_targets = jnp.where(traj_batch.is_timeout, timeout_target_val, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)

            T_steps = config["NUM_STEPS"]
            is_horizon = (jnp.arange(T_steps) == (T_steps - 1))[:, None]
            is_horizon = jnp.broadcast_to(is_horizon, true_terminal.shape)
            next_reward = jnp.roll(traj_batch.reward, shift=-1, axis=0)
            next_next_obs = jnp.roll(traj_batch.next_obs, shift=-1, axis=0)
            next_true_terminal = jnp.roll(true_terminal, shift=-1, axis=0)

            if uses_aux:
                v_aux_vals = network.apply(train_state.params, traj_batch.obs, method=network.v_aux)
                final_next_aux_val = network.apply(
                    train_state.params, traj_batch.next_obs[-1], method=network.v_aux
                )
                regular_aux_next = jnp.concatenate([v_aux_vals[1:], final_next_aux_val[None]], axis=0)
                timeout_aux_next = network.apply(
                    train_state.params, traj_batch.next_obs, method=network.v_aux
                )
                aux_next_value = jnp.where(traj_batch.is_timeout, timeout_aux_next, regular_aux_next)
                aux_lambda = config["AUX_LAMBDA"]
                _, aux_targets = calculate_gae(
                    traj_batch._replace(value=v_aux_vals),
                    aux_next_value,
                    gamma=config["GAMMA"],
                    gae_lambda=aux_lambda,
                )
            else:
                aux_targets = targets

            traj_batch = traj_batch._replace(
                next_probe_actions=next_probe_actions,
                next_action=next_action,
                next_value=next_value,
                targets=targets,
                next_targets=next_targets,
                next_reward=next_reward,
                next_next_obs=next_next_obs,
                next_true_terminal=next_true_terminal,
                is_horizon=is_horizon,
                aux_targets=aux_targets,
            )

            # Prepare geometric jumps only for E(lambda)
            if c_type in ("e_lambda", "e_lambda_q", "e_lambda_v_aux", "elambda", "e_geometric"):
                rng, _rng = jax.random.split(rng)
                traj_batch = prepare_q_jump_transitions(
                    traj_batch,
                    targets,
                    last_obs,
                    last_action,
                    gamma=config["GAMMA"],
                    lmbda=config["E_LAMBDA"],
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

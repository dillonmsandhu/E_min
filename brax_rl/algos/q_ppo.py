# q_ppo.py
# Continuous Q-PPO for Brax environments:
# - K-action sampling (K=2 default) with slot 0 applied to the environment
# - Leave-One-Out ('loo') and Hybrid ('hybrid') Generalized Advantage Estimation
# - Bilinear E(0) Q-critic minimization via independent successor probe actions
# - SARSA(lambda) / fitted Q regression baseline
from __future__ import annotations
import copy
import os
import sys
from typing import Any, NamedTuple, Optional

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from gymnax.wrappers.purerl import LogWrapper

from brax_rl.core.q_ppo_config import config as base_default_config
from brax_rl.core.brax_utils import shuffle_and_batch
from brax_rl.core.brax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
)
from brax_rl.core.brax_network import initialize_split_train_state
from brax_rl.core.brax_critics import (
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    ppo_actor_loss_fn,
)
from brax_rl.core.brax_q_network import BraxQPPOActorCritic
from brax_rl.core.brax_q_critics import get_brax_q_ppo_critic_loss_fn

SAVE_DIR = "q_ppo"


class BraxQTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    q_taken: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    actions_k: jnp.ndarray
    q_all_k: jnp.ndarray
    next_probe_actions: Optional[jnp.ndarray] = None
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None


def make_train(config):
    """
    Factory creating the Brax continuous Q-PPO training function using K-action
    sampling (slot 0 applied, slots 1..K-1 unapplied probes) and decoupled
    actor/critic optimization.
    """
    cfg = copy.deepcopy(base_default_config)
    if "NETWORK_CONFIG" in config:
        merged_net_cfg = {**cfg["NETWORK_CONFIG"], **config["NETWORK_CONFIG"]}
    else:
        merged_net_cfg = cfg["NETWORK_CONFIG"]
    cfg.update(config)
    cfg["NETWORK_CONFIG"] = merged_net_cfg
    config = cfg

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["MAX_NUM_POLICIES"] = config["TOTAL_TIMESTEPS"] // batch_size
    config["NUM_UPDATES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)

    net_cfg = config["NETWORK_CONFIG"]

    # Decoupled Actor settings
    actor_epochs = config.get("ACTOR_EPOCHS", net_cfg.get("NUM_EPOCHS", 4))
    actor_minibatch_size = config.get("ACTOR_MINIBATCH_SIZE", net_cfg.get("MINIBATCH_SIZE", 1024))
    actor_minibatch_size = min(batch_size, actor_minibatch_size)
    config["ACTOR_EPOCHS"] = actor_epochs
    config["ACTOR_MINIBATCH_SIZE"] = actor_minibatch_size
    config["ACTOR_NUM_MINIBATCHES"] = max(1, batch_size // actor_minibatch_size)

    # Decoupled Critic settings
    critic_epochs = config.get("CRITIC_EPOCHS", net_cfg.get("NUM_EPOCHS", 4))
    critic_minibatch_size = config.get("CRITIC_MINIBATCH_SIZE", net_cfg.get("MINIBATCH_SIZE", 1024))
    critic_minibatch_size = min(batch_size, critic_minibatch_size)
    config["CRITIC_EPOCHS"] = critic_epochs
    config["CRITIC_MINIBATCH_SIZE"] = critic_minibatch_size
    config["CRITIC_NUM_MINIBATCHES"] = max(1, batch_size // critic_minibatch_size)

    c_type = config["CRITIC_TYPE"].lower()
    critic_loss_fn = get_brax_q_ppo_critic_loss_fn(c_type)
    actor_loss_fn = ppo_actor_loss_fn

    k_actions = config["K_ACTIONS"]
    if not isinstance(k_actions, int) or isinstance(k_actions, bool) or k_actions < 2:
        raise ValueError(f"Brax continuous Q-PPO requires integer K_ACTIONS >= 2, got {k_actions!r}.")

    gae_mode = config["GAE_CONTINUOUS_MODE"].lower()
    if gae_mode not in ("loo", "hybrid"):
        raise ValueError(
            f"Unrecognized GAE_CONTINUOUS_MODE '{config['GAE_CONTINUOUS_MODE']}'. "
            "Expected 'loo' or 'hybrid'."
        )

    gae_lambda = config.get("ACTOR_GAE_LAMBDA", config["GAE_LAMBDA"])
    target_lambda = get_critic_target_lambda(config)

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if net_cfg.get("NORMALIZE_ENV", True):
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def train(rng):
        network = BraxQPPOActorCritic(
            action_dim=env.action_space(env_params).shape[0],
            activation=net_cfg.get("ACTIVATION", "tanh"),
            layer_norm=config.get("LAYER_NORM", False),
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        train_state = initialize_split_train_state(config, network, network_params)

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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES WITH K-ACTION SAMPLING
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, actions_k, q_all_k = _sample_and_eval_k(train_state.params, last_obs, _rng)
                action = actions_k[..., 0, :]
                log_prob = pi.log_prob(action)
                q_taken = q_all_k[..., 0]
                value = jnp.mean(q_all_k[..., 1:], axis=-1)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

                next_obs = info["real_next_obs"]
                is_timeout = info["is_timeout"]
                true_terminal = (done > 0.5) & ~(is_timeout > 0.5)
                is_start = last_true_terminal

                transition = BraxQTransition(
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

            # 2. CALCULATE ADVANTAGES, RETURN TARGETS & PROBE ACTIONS
            rng, last_rng, timeout_rng = jax.random.split(rng, 3)
            _, last_actions_k, last_q_k = _sample_and_eval_k(train_state.params, last_obs, last_rng)
            _, timeout_actions_k, timeout_q_k = _sample_and_eval_k(
                train_state.params, traj_batch.next_obs, timeout_rng
            )

            regular_next_actions_k = jnp.concatenate([traj_batch.actions_k[1:], last_actions_k[None]], axis=0)
            next_actions_k = jnp.where(
                (traj_batch.is_timeout > 0.5)[..., None, None], timeout_actions_k, regular_next_actions_k
            )
            regular_next_q_k = jnp.concatenate([traj_batch.q_all_k[1:], last_q_k[None]], axis=0)
            next_q_k = jnp.where(
                (traj_batch.is_timeout > 0.5)[..., None], timeout_q_k, regular_next_q_k
            )
            next_probe_actions = next_actions_k[..., 1:, :]

            # Critic return targets strictly use applied slot-0 actions so G_t
            # stays independent of probe actions a'_{2..K} at all steps.
            v_ret = traj_batch.q_all_k[..., 0]
            next_v_ret = next_q_k[..., 0]
            targets = calculate_targets(
                traj_batch._replace(value=v_ret),
                next_v_ret,
                gamma=config["GAMMA"],
                target_lambda=target_lambda,
            )

            true_terminal = (traj_batch.done > 0.5) & ~(traj_batch.is_timeout > 0.5)
            last_ret_val = jnp.where(true_terminal[-1], 0.0, next_v_ret[-1])
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(last_ret_val)
            next_targets = jnp.where(traj_batch.is_timeout > 0.5, next_v_ret, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)

            # Actor GAE: 'loo' uses slots 1..K-1 everywhere; 'hybrid' computes
            # lambda-returns with the K-action average V_bar and subtracts V_LOO.
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

            traj_batch = traj_batch._replace(
                next_probe_actions=next_probe_actions,
                next_value=next_value,
                targets=targets,
                next_targets=next_targets,
            )

            # 3. CRITIC UPDATE PHASE (Separate epochs, minibatches, optimizer)
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    def _loss_fn(critic_params):
                        return critic_loss_fn(
                            critic_params,
                            train_state.actor.params,
                            network,
                            mb,
                            mb.targets,
                            config,
                        )

                    (_, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads)
                    return train_state, aux

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(_rng, critic_batch, config["CRITIC_NUM_MINIBATCHES"])
                train_state, critic_metrics = jax.lax.scan(
                    _critic_minibatch, train_state, minibatches
                )
                return (train_state, critic_batch, rng), critic_metrics

            rng, _rng = jax.random.split(rng)
            critic_state = (train_state, traj_batch, _rng)
            (train_state, _, rng), critic_loss_info = jax.lax.scan(
                _critic_epoch, critic_state, None, config["CRITIC_EPOCHS"]
            )

            # 4. ACTOR UPDATE PHASE (Separate epochs, minibatches, optimizer)
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    traj_mb, adv_mb = mb

                    def _loss_fn(actor_params):
                        return actor_loss_fn(
                            actor_params,
                            train_state.critic.params,
                            network,
                            traj_mb,
                            adv_mb,
                            config,
                        )

                    (_, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads)
                    return train_state, aux

                train_state, actor_batch, rng = actor_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(_rng, actor_batch, config["ACTOR_NUM_MINIBATCHES"])
                train_state, actor_metrics = jax.lax.scan(
                    _actor_minibatch, train_state, minibatches
                )
                return (train_state, actor_batch, rng), actor_metrics

            actor_batch = (traj_batch, advantages)
            rng, _rng = jax.random.split(rng)
            actor_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), actor_loss_info = jax.lax.scan(
                _actor_epoch, actor_state, None, config["ACTOR_EPOCHS"]
            )

            # 5. RUNTIME METRICS
            combined_loss_info = {**critic_loss_info, **actor_loss_info}
            metric = {
                "pi_idx": idx,
                "pi_num": idx,
                "advantage": advantages.mean(),
            }
            metric.update({k: v.mean() for k, v in combined_loss_info.items()})
            metric.update({
                k: v.mean() for k, v in traj_batch.info.items()
                if k not in ["real_next_obs", "obs_jump", "targets_jump", "valid_jump", "is_absorbing"]
                and hasattr(v, "mean") and v.ndim <= 2
            })

            runner_state = (train_state, env_state, last_obs, last_true_terminal, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, _rng, 1)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from brax_rl.core.evaluator import run_experiment_main
    run_experiment_main(make_train, default_critic="e_0", save_dir=SAVE_DIR)

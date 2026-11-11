# ddpg.py
# On-Policy Continuous DDPG for Brax environments with State-Action Critic Q(s, a)
# and Expected State Value V(s) = Q(s, mu(s)).
# Critic loss is swappable via config["CRITIC_TYPE"]:
#   - "sarsa" / "fitted" / "td_lambda": On-policy Expected SARSA(lambda) regression
#   - "td_0" / "sarsa_0": Online 1-step Expected SARSA(0)
#   - "e_0" / "e_0_basic": Sampled E(0) State-Action Dirichlet minimization
#   - "e_lambda" / "e_geometric": Geometric Jump E(lambda) State-Action Dirichlet minimization
from __future__ import annotations
import os
import sys
from typing import NamedTuple, Any, Optional

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import copy
import jax
import jax.numpy as jnp
from gymnax.wrappers.purerl import LogWrapper

from core.ddpg_config import brax_ddpg_config as base_default_config
from brax_rl.core.brax_utils import shuffle_and_batch
from brax_rl.core.brax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
)
from brax_rl.core.brax_q_network import (
    BraxDDPGNetwork,
    BraxDDPGDualCritic,
    initialize_split_train_state,
)
from brax_rl.core.brax_q_critics import (
    get_brax_ddpg_critic_loss_fn,
    ddpg_actor_loss_fn,
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    prepare_critic_transitions,
)


class BraxDDPGTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    mu: jnp.ndarray
    value: jnp.ndarray
    q_taken: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    v_aux: Optional[jnp.ndarray] = None
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    aux_targets: Optional[jnp.ndarray] = None


def make_train(config):
    cfg = copy.deepcopy(base_default_config)
    cfg.update(config)
    config = cfg
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["MAX_NUM_POLICIES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)
    config["NUM_UPDATES"] = max(1, config["TOTAL_TIMESTEPS"] // batch_size)

    net_cfg = config["NETWORK_CONFIG"]

    # Decoupled Actor settings
    actor_epochs = config["ACTOR_EPOCHS"] if config["ACTOR_EPOCHS"] is not None else net_cfg["NUM_EPOCHS"]
    actor_minibatch_size = (
        config["ACTOR_MINIBATCH_SIZE"] if config["ACTOR_MINIBATCH_SIZE"] is not None else net_cfg["MINIBATCH_SIZE"]
    )
    actor_minibatch_size = min(batch_size, actor_minibatch_size)
    config["ACTOR_EPOCHS"] = actor_epochs
    config["ACTOR_MINIBATCH_SIZE"] = actor_minibatch_size
    config["ACTOR_NUM_MINIBATCHES"] = max(1, batch_size // actor_minibatch_size)

    # Decoupled Critic settings
    critic_epochs = config["CRITIC_EPOCHS"] if config["CRITIC_EPOCHS"] is not None else net_cfg["NUM_EPOCHS"]
    critic_minibatch_size = (
        config["CRITIC_MINIBATCH_SIZE"] if config["CRITIC_MINIBATCH_SIZE"] is not None else net_cfg["MINIBATCH_SIZE"]
    )
    critic_minibatch_size = min(batch_size, critic_minibatch_size)
    config["CRITIC_EPOCHS"] = critic_epochs
    config["CRITIC_MINIBATCH_SIZE"] = critic_minibatch_size
    config["CRITIC_NUM_MINIBATCHES"] = max(1, batch_size // critic_minibatch_size)

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if net_cfg["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def train(rng):
        c_type = config["CRITIC_TYPE"].lower()
        e_variant = config["E_VARIANT"].lower()
        critic_loss_fn = get_brax_ddpg_critic_loss_fn(c_type)
        actor_loss_fn = ddpg_actor_loss_fn

        target_lambda = get_critic_target_lambda(config)
        uses_aux = (
            c_type in ["e_0", "e_0_q", "e0", "e", "sampled_e", "surgical_v_aux", "surgical", "e_lambda", "e_lambda_q", "elambda", "e_geometric"]
            and e_variant in ["aux", "e_aux"]
        )

        # INIT NETWORK
        action_dim = env.action_space(env_params).shape[0]
        if uses_aux:
            network = BraxDDPGDualCritic(
                action_dim=action_dim,
                activation=net_cfg["ACTIVATION"],
                layer_norm=config["LAYER_NORM"],
                exploration_noise=config["EXPLORATION_NOISE"],
                shared_torso=config["SHARED_TORSO"],
            )
        else:
            network = BraxDDPGNetwork(
                action_dim=action_dim,
                activation=net_cfg["ACTIVATION"],
                layer_norm=config["LAYER_NORM"],
                exploration_noise=config["EXPLORATION_NOISE"],
            )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        # Independent TrainStates for Actor and Critic
        train_state = initialize_split_train_state(config, network, network_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        last_true_terminal = jnp.zeros(config["NUM_ENVS"], dtype=bool)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx = runner_state

            # 1. COLLECT ON-POLICY ROLLOUT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                mu = network.apply(train_state.params, last_obs, method=network.actor)
                pi = network.apply(train_state.params, last_obs, method=network.policy)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                if uses_aux:
                    value, v_aux = network.apply(train_state.params, last_obs, method=network.both)
                else:
                    value = network.apply(train_state.params, last_obs, method=network.value)
                    v_aux = jnp.zeros_like(value)

                q_taken = network.apply(train_state.params, last_obs, action, method=network.q_value)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

                next_obs = info["real_next_obs"]
                is_timeout = info["is_timeout"]
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                transition = BraxDDPGTransition(
                    done=done,
                    action=action,
                    mu=mu,
                    value=value,
                    q_taken=q_taken,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    next_obs=next_obs,
                    info=info,
                    is_start=is_start,
                    is_timeout=is_timeout,
                    v_aux=v_aux,
                )
                return (train_state, env_state, obsv, true_terminal, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. EXPECTED SARSA(lambda) MULTI-STEP TARGET CALCULATION
            last_val_raw = network.apply(train_state.params, last_obs, method=network.value)
            regular_next_val = jnp.concatenate([traj_batch.value[1:], last_val_raw[None]], axis=0)
            timeout_next_val = network.apply(train_state.params, traj_batch.next_obs, method=network.value)
            next_value = jnp.where(traj_batch.is_timeout, timeout_next_val, regular_next_val)
            true_terminal = traj_batch.done & ~traj_batch.is_timeout
            last_val = jnp.where(true_terminal[-1], 0.0, next_value[-1])

            targets = calculate_targets(
                traj_batch,
                next_value,
                gamma=config["GAMMA"],
                target_lambda=target_lambda,
            )

            # AUXILIARY TARGETS
            if uses_aux and e_variant == "aux":
                final_next_aux_val = network.apply(
                    train_state.params, traj_batch.next_obs[-1], method=network.v_aux
                )
                regular_aux_next = jnp.concatenate([traj_batch.v_aux[1:], final_next_aux_val[None]], axis=0)
                timeout_aux_next = network.apply(train_state.params, traj_batch.next_obs, method=network.v_aux)
                aux_next_value = jnp.where(traj_batch.is_timeout, timeout_aux_next, regular_aux_next)
                traj_batch_aux = traj_batch._replace(value=traj_batch.v_aux)
                _, aux_targets = calculate_gae(
                    traj_batch_aux,
                    aux_next_value,
                    gamma=config["GAMMA"],
                    gae_lambda=config["AUX_LAMBDA"],
                )
            elif uses_aux and e_variant == "e_aux":
                aux_targets = calculate_targets(
                    traj_batch,
                    next_value,
                    gamma=config["GAMMA"],
                    target_lambda=config["AUX_LAMBDA"],
                )
            else:
                aux_targets = targets

            # Align next targets G_{t+1}
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(last_val)
            next_targets = jnp.where(traj_batch.is_timeout, next_value, next_targets)
            next_targets = jnp.where(true_terminal, 0.0, next_targets)

            traj_batch = traj_batch._replace(
                next_value=next_value,
                targets=targets,
                next_targets=next_targets,
                aux_targets=aux_targets,
            )

            # 3. PREPARE CRITIC TRANSITIONS (e.g. geometric lookahead jumps for E(lambda))
            rng, _rng = jax.random.split(rng)
            traj_batch = prepare_critic_transitions(
                critic_type=c_type,
                traj_batch=traj_batch,
                targets=targets,
                last_val=last_val,
                gamma=config["GAMMA"],
                e_lambda=config["E_LAMBDA"],
                rng=_rng,
            )

            # 4. CRITIC UPDATE PHASE (Separate epochs, minibatches, optimizer)
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    traj_mb = mb

                    def _loss_fn(critic_params):
                        return critic_loss_fn(
                            critic_params,
                            train_state.actor.params,
                            network,
                            traj_mb,
                            traj_mb.targets,
                            config,
                        )

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads)
                    return train_state, aux

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(_rng, critic_batch, config["CRITIC_NUM_MINIBATCHES"])
                train_state, critic_metrics = jax.lax.scan(
                    _critic_minibatch, train_state, minibatches
                )
                return (train_state, critic_batch, rng), critic_metrics

            critic_batch = traj_batch
            rng, _rng = jax.random.split(rng)
            critic_state = (train_state, critic_batch, _rng)
            (train_state, _, rng), critic_loss_info = jax.lax.scan(
                _critic_epoch, critic_state, None, config["CRITIC_EPOCHS"]
            )

            # 5. ACTOR UPDATE PHASE (Updates mu_phi via -E[Q(s, mu(s))])
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    traj_mb = mb

                    def _loss_fn(actor_params):
                        return actor_loss_fn(
                            actor_params,
                            train_state.critic.params,
                            network,
                            traj_mb,
                            config,
                        )

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads)
                    return train_state, aux

                train_state, actor_batch, rng = actor_state
                rng, _rng = jax.random.split(rng)
                minibatches = shuffle_and_batch(_rng, actor_batch, config["ACTOR_NUM_MINIBATCHES"])
                train_state, actor_metrics = jax.lax.scan(
                    _actor_minibatch, train_state, minibatches
                )
                return (train_state, actor_batch, rng), actor_metrics

            actor_batch = traj_batch
            rng, _rng = jax.random.split(rng)
            actor_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), actor_loss_info = jax.lax.scan(
                _actor_epoch, actor_state, None, config["ACTOR_EPOCHS"]
            )

            # 6. RUNTIME METRICS
            combined_loss_info = {**critic_loss_info, **actor_loss_info}
            metric = {
                "pi_idx": idx,
                "pi_num": idx,
                "q_pred": traj_batch.q_taken.mean(),
                "v_pred": traj_batch.value.mean(),
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
    run_experiment_main(make_train, default_critic="sarsa", save_dir="brax_ddpg")

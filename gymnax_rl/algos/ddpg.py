# ddpg.py
# On-Policy Continuous DDPG with State-Action Critic Q(s, a) and Expected State Value V(s) = Q(s, mu(s)).
# Critic loss is swappable via config["CRITIC_TYPE"]:
#   - "sarsa" / "fitted" / "td_lambda": On-policy Expected SARSA(lambda) regression
#   - "td_0" / "sarsa_0": Online 1-step Expected SARSA(0)
#   - "e_0" / "e_0_basic": Sampled E(0) State-Action Dirichlet minimization
#   - "e_lambda" / "e_geometric": Geometric Jump E(lambda) State-Action Dirichlet minimization
# Actor is updated via Deterministic Policy Gradient:
#   L_actor(phi) = -E_s[ Q_theta(s, mu_phi(s)) ]
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
)
from gymnax_rl.core.gymnax_wrappers import make_env
from gymnax_rl.core.gymnax_network import (
    initialize_split_flax_train_state,
    uses_aux_network,
)
from gymnax_rl.core.gymnax_q_network import initialize_ddpg_network
from gymnax_rl.core.gymnax_q_critics import (
    get_ddpg_critic_loss_fn,
    ddpg_actor_loss_fn,
)
from gymnax_rl.core.gymnax_critics import (
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    prepare_critic_transitions,
)
from gymnax_rl.core.runtime_metrics import compute_runtime_metrics

SAVE_DIR = "ddpg"


class DDPGTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    mu: jnp.ndarray
    value: jnp.ndarray          # Target policy value V(s) = Q(s, mu(s))
    q_taken: jnp.ndarray        # Executed action value Q(s, a_t)
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any
    is_start: jnp.ndarray
    is_timeout: jnp.ndarray
    next_value: Optional[jnp.ndarray] = None
    targets: Optional[jnp.ndarray] = None
    next_targets: Optional[jnp.ndarray] = None
    aux_targets: Optional[jnp.ndarray] = None
    dest_obs: Optional[jnp.ndarray] = None
    dest_targets: Optional[jnp.ndarray] = None
    hit_true_terminal: Optional[jnp.ndarray] = None
    hit_timeout: Optional[jnp.ndarray] = None
    hit_horizon: Optional[jnp.ndarray] = None
    start_weight: Optional[jnp.ndarray] = None
    critic_mask: Optional[jnp.ndarray] = None
    v_aux: Optional[jnp.ndarray] = None


def make_train(base_config):
    """
    Factory creating the JAX-vmappable and JIT-compilable training function
    for On-Policy Continuous DDPG with modular Q-critic choices selected via config['CRITIC_TYPE'].
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
        critic_loss_fn = get_ddpg_critic_loss_fn(c_type)
        actor_loss_fn = ddpg_actor_loss_fn
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

        # Lambdas & Auxiliary Critic Dispatch
        target_lambda = get_critic_target_lambda(config)
        uses_aux = uses_aux_network(config)

        # Network & Split TrainState
        rng, net_rng = jax.random.split(rng)
        network, network_params = initialize_ddpg_network(
            rng=net_rng,
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
        steps_since_start = jnp.full((config["NUM_ENVS"],), 10_000, dtype=jnp.int32)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_true_terminal, rng, idx, steps_since_start = runner_state

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

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                next_obs = info["real_next_obs"]
                is_timeout = info["is_timeout"]
                true_terminal = done & ~is_timeout
                is_start = last_true_terminal

                transition = DDPGTransition(
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
            final_next_val = network.apply(
                train_state.params, traj_batch.next_obs[-1], method=network.value
            )
            regular_next_val = jnp.concatenate([traj_batch.value[1:], final_next_val[None]], axis=0)
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
            e_variant = config["E_VARIANT"].lower()
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
            elif uses_aux:
                raise ValueError(f"uses_aux=True with unsupported E_VARIANT '{config['E_VARIANT']}'.")
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

            # 3. PREPARE CRITIC TRANSITIONS (Geometric jumps for E(lambda))
            rng, _rng = jax.random.split(rng)
            traj_batch, steps_since_start = prepare_critic_transitions(
                critic_type=c_type,
                traj_batch=traj_batch,
                targets=targets,
                last_val=last_val,
                gamma=config["GAMMA"],
                e_lambda=config["E_LAMBDA"],
                rng=_rng,
                steps_since_start=steps_since_start,
            )

            # 4. CRITIC UPDATE PHASE (Updates Q_theta)
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

            # 5. ACTOR UPDATE PHASE (Updates mu_phi via -E[Q(s, mu(s))])
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    traj_mb = mb

                    def _a_loss(actor_params):
                        return actor_loss_fn(
                            actor_params,
                            train_state.critic.params,
                            network,
                            traj_mb,
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

            actor_batch = traj_batch
            rng, _rng = jax.random.split(rng)
            a_init_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), a_loss_epochs = jax.lax.scan(
                _actor_epoch, a_init_state, None, config["ACTOR_EPOCHS"]
            )

            # 6. RUNTIME METRICS
            combined_loss_info = {**c_loss_epochs, **a_loss_epochs}
            metric = compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=combined_loss_info,
                start_obs=start_obs,
                config=config,
            )
            metric["q_pred"] = traj_batch.q_taken.mean()

            runner_state = (train_state, env_state, last_obs, last_true_terminal, rng, idx + 1, steps_since_start)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, last_true_terminal, _rng, 1, steps_since_start)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from gymnax_rl.core.evaluator import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

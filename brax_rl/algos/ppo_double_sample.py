from brax_rl.core.brax_config import config as base_default_config
import os
import sys

# Ensure repository root is on sys.path
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
from gymnax.wrappers.purerl import LogWrapper

from brax_rl.core.brax_utils import (
    Double_Sample_Transition,
    scale_rms,
    shuffle_and_batch,
)
from brax_rl.core.brax_wrappers import (
    BraxGymnaxWrapper,
    ClipAction,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
)
from brax_rl.core.brax_network import (
    ActorCritic,
    initialize_split_train_state,
)
from brax_rl.core.brax_critics import (
    get_critic_loss_fn,
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    ppo_actor_loss_fn,
)


def make_train(config):
    cfg = base_default_config.copy()
    cfg.update(config)
    config = cfg
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["MAX_NUM_POLICIES"] = config["TOTAL_TIMESTEPS"] // batch_size
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    net_cfg = config.get("NETWORK_CONFIG", {})

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

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if net_cfg.get("NORMALIZE_ENV", True):
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])

    def train(rng):
        c_type = config.get("CRITIC_TYPE", "fitted").lower()
        critic_loss_fn = get_critic_loss_fn(c_type, split=True)
        actor_loss_fn = ppo_actor_loss_fn

        gae_lambda = config.get("GAE_LAMBDA", 0.95)
        target_lambda = get_critic_target_lambda(config)

        # Standard ActorCritic network (single value head, no auxiliary head needed)
        network = ActorCritic(
            action_dim=env.action_space(env_params).shape[0],
            activation=net_cfg.get("ACTIVATION", "tanh"),
            n_value_heads=config.get("NUM_VALUE_HEADS", 1),
            layer_norm=config.get("LAYER_NORM", False),
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

            # 1. COLLECT TRAJECTORIES WITH 1-STEP DOUBLE SAMPLING
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, last_true_terminal, rng = env_scan_state

                # SELECT ACTION & VALUE
                rng, rng_a1, rng_a2, rng_s1, rng_s2 = jax.random.split(rng, 5)
                pi, value = network.apply(train_state.params, last_obs)

                # Sample two independent policy actions from same observation
                action_1 = pi.sample(seed=rng_a1)
                action_2 = pi.sample(seed=rng_a2)
                log_prob_1 = pi.log_prob(action_1)

                # STEP ENV - Branch 1 (Primary rollout)
                rng_step1 = jax.random.split(rng_s1, config["NUM_ENVS"])
                obsv_1, next_env_state, reward_1, done_1, info_1 = env.step(
                    rng_step1, env_state, action_1, env_params
                )
                next_obs_1 = info_1["real_next_obs"]
                is_timeout_1 = info_1["is_timeout"]
                true_terminal_1 = done_1 & ~is_timeout_1
                is_start = last_true_terminal

                # STEP ENV - Branch 2 (Auxiliary probe from exact same env_state)
                rng_step2 = jax.random.split(rng_s2, config["NUM_ENVS"])
                obsv_2, _, reward_2, done_2, info_2 = env.step(
                    rng_step2, env_state, action_2, env_params
                )
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
                # Next step continues with next_env_state from Branch 1
                return (train_state, next_env_state, obsv_1, true_terminal_1, rng), transition

            env_step_state = (train_state, env_state, last_obs, last_true_terminal, rng)
            (_, env_state, last_obs, last_true_terminal, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. CALCULATE ADVANTAGES & TARGETS
            last_val_raw = network.apply(train_state.params, last_obs, method=network.value)
            regular_next_val = jnp.concatenate([traj_batch.value[1:], last_val_raw[None]], axis=0)
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

            # 3. CRITIC UPDATE PHASE (Separate epochs, minibatches, optimizer)
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

            actor_batch = (traj_batch, advantages)
            rng, _rng = jax.random.split(rng)
            actor_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), actor_loss_info = jax.lax.scan(
                _actor_epoch, actor_state, None, config["ACTOR_EPOCHS"]
            )

            # 5. RUNTIME & DIAGNOSTIC METRICS
            combined_loss_info = {**critic_loss_info, **actor_loss_info}

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

            metric = {
                "pi_idx": idx,
                "pi_num": idx,
                "advantage": advantages.mean(),
                "diagnostic_msbe": diagnostic_msbe_rollout,
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
    run_experiment_main(make_train)

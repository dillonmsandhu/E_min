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
    Transition,
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
from brax_rl.core.brax_network import ActorCritic, initialize_split_train_state
from brax_rl.core.brax_critics import (
    get_critic_loss_fn,
    calculate_gae,
    prepare_critic_transitions,
    ppo_actor_loss_fn,
)


def make_train(config):
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["MAX_NUM_POLICIES"] = config["TOTAL_TIMESTEPS"] // batch_size
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size

    net_cfg = config.get("NETWORK_CONFIG", {})

    # Decoupled Actor settings
    actor_epochs = config.get("ACTOR_EPOCHS", net_cfg.get("NUM_EPOCHS", 4))
    actor_minibatch_size = config.get("ACTOR_MINIBATCH_SIZE", net_cfg.get("MINIBATCH_SIZE", 1024))
    if actor_minibatch_size > batch_size:
        actor_minibatch_size = batch_size
    config["ACTOR_EPOCHS"] = actor_epochs
    config["ACTOR_MINIBATCH_SIZE"] = actor_minibatch_size
    config["ACTOR_NUM_MINIBATCHES"] = max(1, batch_size // actor_minibatch_size)

    # Decoupled Critic settings
    critic_epochs = config.get("CRITIC_EPOCHS", net_cfg.get("NUM_EPOCHS", 4))
    critic_minibatch_size = config.get("CRITIC_MINIBATCH_SIZE", net_cfg.get("MINIBATCH_SIZE", 1024))
    if critic_minibatch_size > batch_size:
        critic_minibatch_size = batch_size
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

    critic_loss_fn = get_critic_loss_fn(config.get("CRITIC_TYPE", "fitted"), split=True)
    actor_loss_fn = ppo_actor_loss_fn

    def train(rng):
        n_value_heads = config.get("NUM_VALUE_HEADS", 1)
        network = ActorCritic(
            action_dim=env.action_space(env_params).shape[0],
            activation=net_cfg.get("ACTIVATION", "tanh"),
            n_value_heads=n_value_heads,
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)

        # Independent TrainStates for Actor and Critic with separate LR and schedules
        train_state = initialize_split_train_state(config, network, network_params)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, idx = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
                true_next_obs = info["real_next_obs"] if isinstance(info, dict) and "real_next_obs" in info else obsv
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, true_next_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng, idx)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            train_state, env_state, last_obs, rng, idx = runner_state

            # CALCULATE ADVANTAGE & TARGETS
            _, last_val = network.apply(train_state.params, last_obs)
            advantages, targets = calculate_gae(
                traj_batch,
                last_val,
                gamma=config["GAMMA"],
                gae_lambda=config["GAE_LAMBDA"],
            )

            # PREPARE CRITIC TRANSITIONS (e.g. geometric lookahead jumps for E0 / E(lambda))
            critic_type = config.get("CRITIC_TYPE", "fitted").lower()
            rng, _rng = jax.random.split(rng)
            traj_batch = prepare_critic_transitions(
                critic_type=critic_type,
                traj_batch=traj_batch,
                targets=targets,
                last_val=last_val,
                gamma=config["GAMMA"],
                e_lambda=config.get("E_LAMBDA", 0.8),
                rng=_rng,
            )

            # 1. CRITIC UPDATE PHASE (Separate epochs, minibatches, optimizer)
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    traj_mb, targets_mb = mb

                    def _loss_fn(critic_params):
                        return critic_loss_fn(
                            critic_params, train_state.actor.params, network, traj_mb, targets_mb, config
                        )

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads)
                    return train_state, aux

                train_state, traj_batch, targets, rng = critic_state
                rng, _rng = jax.random.split(rng)
                critic_batch = (traj_batch, targets)
                minibatches = shuffle_and_batch(_rng, critic_batch, config["CRITIC_NUM_MINIBATCHES"])
                train_state, critic_metrics = jax.lax.scan(
                    _critic_minibatch, train_state, minibatches
                )
                return (train_state, traj_batch, targets, rng), critic_metrics

            critic_state = (train_state, traj_batch, targets, rng)
            (train_state, _, _, rng), critic_loss_info = jax.lax.scan(
                _critic_epoch, critic_state, None, config["CRITIC_EPOCHS"]
            )

            # 2. ACTOR UPDATE PHASE (Separate epochs, minibatches, optimizer)
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    traj_mb, adv_mb = mb

                    def _loss_fn(actor_params):
                        return actor_loss_fn(
                            actor_params, train_state.critic.params, network, traj_mb, adv_mb, config
                        )

                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads)
                    return train_state, aux

                train_state, traj_batch, advantages, rng = actor_state
                rng, _rng = jax.random.split(rng)
                actor_batch = (traj_batch, advantages)
                minibatches = shuffle_and_batch(_rng, actor_batch, config["ACTOR_NUM_MINIBATCHES"])
                train_state, actor_metrics = jax.lax.scan(
                    _actor_minibatch, train_state, minibatches
                )
                return (train_state, traj_batch, advantages, rng), actor_metrics

            actor_state = (train_state, traj_batch, advantages, rng)
            (train_state, _, _, rng), actor_loss_info = jax.lax.scan(
                _actor_epoch, actor_state, None, config["ACTOR_EPOCHS"]
            )

            # METRICS COLLECTION
            metric = {
                "pi_idx": idx,
                "pi_num": idx,
                "advantage": advantages.mean(),
            }
            metric.update({k: v.mean() for k, v in critic_loss_info.items()})
            metric.update({k: v.mean() for k, v in actor_loss_info.items()})
            metric.update({
                k: v.mean() for k, v in traj_batch.info.items()
                if k not in ["real_next_obs", "obs_jump", "targets_jump", "valid_jump", "is_absorbing"]
                and hasattr(v, "mean") and v.ndim <= 2
            })
            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from brax_rl.core.evaluator import run_experiment_main
    run_experiment_main(make_train)
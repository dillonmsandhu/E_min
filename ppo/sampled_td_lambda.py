# Sampled version of PPO: TD(lambda) for critic, standard GAE for actor
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import core.bellman_error as bellman_error

SAVE_DIR = "ppo/td_lambda"

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(base_config):
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = batch_size // base_config["MINIBATCH_SIZE"]
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get("k", 32)

        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config["LAYER_NORM"]
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        runner_state = (train_state, env_state, obsv, rng, 1)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"]
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                clean_info = {k: v for k, v in info.items() if k not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. SEPARATE ADVANTAGE AND VALUE TARGET CALCULATIONS
            # GAE_LAMBDA is strictly for policy advantages
            gae_lambda = config.get("GAE_LAMBDA", 0.95)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            # VALUE_LAMBDA is strictly for critic targets
            value_lambda = config.get("VALUE_LAMBDA", 0.95)
            _, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], value_lambda)

            # 3. UPDATE EPOCHS
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    obs_mb, action_mb, log_prob_mb, value_mb, advantages_mb, targets_mb = batch_info

                    def loss_fn(params, network):
                        # A) Actor Loss (PPO clipped surrogate)
                        pi = network.apply(params, obs_mb, method=network.policy)
                        log_prob = pi.log_prob(action_mb)
                        entropy = pi.entropy().mean()
                        ratio = jnp.exp(log_prob - log_prob_mb)

                        adv_norm = helpers.post_process_advantage(advantages_mb, config)


                        surr1 = ratio * adv_norm
                        surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
                        actor_loss = -jnp.minimum(surr1, surr2).mean()

                        # B) Critic Loss (TD(lambda) with optional PPO clipping)
                        value_pred = network.apply(params, obs_mb, method=network.value)
                        if config.get("VF_CLIP", 0.0) > 0:
                            e = config["VF_CLIP"]
                            value_pred_clipped = value_mb + (value_pred - value_mb).clip(-e, e)
                            value_losses = jnp.square(value_pred - targets_mb)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets_mb)
                            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        else:
                            value_loss = 0.5 * jnp.mean((value_pred - targets_mb) ** 2)

                        total_loss = (
                            actor_loss
                            + config.get("VF_COEF", 0.5) * value_loss
                            - entropy * config.get("ENT_COEF", 0.01)
                        )
                        return total_loss, {
                            "total_loss": total_loss,
                            "value_loss": value_loss,
                            "actor_loss": actor_loss,
                            "entropy": entropy,
                        }

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(train_state.params, network)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (
                    traj_batch.obs,
                    traj_batch.action,
                    traj_batch.log_prob,
                    traj_batch.value,
                    advantages,
                    targets,
                )
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])

                train_state, epoch_losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), epoch_losses

            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # 4. METRICS
            metric = {k: v.mean() for k, v in traj_batch.info.items()}
            metric.update({k: v.mean() for k, v in loss_info.items()})
            metric.update({"mean_rew": traj_batch.reward.mean()})

            if evaluator is not None:
                value_metrics = bellman_error.value_metrics(
                    evaluator, network, train_state.params, random_policy=False, light=config.get("LIGHT_METRICS", True)
                )
                metric.update(value_metrics)

                if config.get("LOG_FEATURE_METRICS", False):
                    from core.feature_metrics import feature_metrics
                    metric.update(feature_metrics(
                        evaluator, network, train_state.params, random_policy=False
                    ))

                if hasattr(evaluator, "obs_stack"):
                    s_states = evaluator.obs_stack
                    v_pred_all = network.apply(train_state.params, s_states, method=network.value)
                    if hasattr(evaluator, "start_idx"):
                        metric["v_pred_start"] = v_pred_all[evaluator.start_idx]

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

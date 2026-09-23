# PPO with Differentiable E(lambda) Critic: backward moment scan (Method 2)
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.runtime_metrics as runtime_metrics
import core.utils as utils

SAVE_DIR = "E_lambda_diff"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_train(base_config):
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // base_config.get("MINIBATCH_SIZE", batch_size))
    base_config["NUM_UPDATES"] = max(1, base_config["TOTAL_TIMESTEPS"] // batch_size)
    env, env_params = helpers.make_env(base_config)
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
        start_obs = obsv

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
                true_next_obs = info["real_next_obs"].reshape(last_obs.shape)
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                clean_info = {k: v for k, v in info.items() if k not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. POLICY ADVANTAGES & BASELINE RETURNS
            gae_lambda = config.get("GAE_LAMBDA", 0.8)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            return_lambda = config.get("RETURN_LAMBDA", 0.99)
            _, returns = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)

            # 3. UPDATE NETWORK OVER EPOCHS AND TRAJECTORY MINIBATCHES
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch_mb, advantages_mb, returns_mb = batch_info
                    grad_fn = jax.value_and_grad(helpers.e_lambda_differentiable_loss_fn, has_aux=True)
                    (total_loss, losses), grads = grad_fn(
                        train_state.params,
                        network,
                        traj_batch_mb,
                        advantages_mb,
                        returns_mb,
                        config,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                train_state, traj_batch, advantages, returns, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, returns)
                # Batch along environment axis to preserve temporal trajectory continuity
                minibatches = helpers.shuffle_and_batch_envs(_rng, batch, config["NUM_MINIBATCHES"])

                train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, returns, rng), loss_info

            initial_update_state = (train_state, traj_batch, advantages, returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, initial_update_state, None, config["NUM_EPOCHS"]
            )
            train_state, _, _, _, rng = update_state

            # 4. RUNTIME METRICS
            metric = runtime_metrics.compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=loss_info,
                start_obs=start_obs,
                config=config,
            )

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, 1)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

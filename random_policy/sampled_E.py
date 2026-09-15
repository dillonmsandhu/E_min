# Sampled E-loss (Dirichlet / spectral decomposition) using trajectory sampling
# Evaluates the random policy using on-policy trajectories and Monte-Carlo returns
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics
import core.utils as utils

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "random/sampled_E"

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    next_target: jnp.ndarray
    info: jnp.ndarray

def make_train(base_config):
    # Ensure GAE_LAMBDA defaults to 1.0 for Monte Carlo returns unless specified
    if "GAE_LAMBDA" not in base_config or base_config["GAE_LAMBDA"] == 0.0:
        base_config["GAE_LAMBDA"] = base_config.get("VALUE_LAMBDA", 1.0)
        if base_config["GAE_LAMBDA"] == 0.0:
            base_config["GAE_LAMBDA"] = 1.0

    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = batch_size // base_config["MINIBATCH_SIZE"]
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get('k', 32)
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                value = network.apply(train_state.params, last_obs)
                pi = distrax.Categorical(logits=jnp.zeros((config['NUM_ENVS'], n_actions)))
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs)

                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, 0.0, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # --- MC RETURN / TARGET CALCULATION ---
            # Using GAE with lambda=1.0 gives MC returns G_t
            gae_lambda = config.get("GAE_LAMBDA", 1.0)
            advantages, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)
            
            # Align next targets G_{t+1}
            next_targets = jnp.roll(targets, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
            # If done, next state is terminal (return = 0.0)
            next_targets = (1.0 - traj_batch.done) * next_targets
            traj_batch = traj_batch._replace(next_target=next_targets)

            # LOSS FUNCTION
            def E_loss_fn(params, traj_batch, targets):
                gamma = config["GAMMA"]
                # 1. Current State Predictions & Errors (e_i)
                v_i = network.apply(params, traj_batch.obs)
                e_i = targets - v_i
                
                # 2. Next State Predictions & Errors (e_j)
                v_j = network.apply(params, traj_batch.next_obs)
                # Terminal transitions have e_j = 0
                e_j = (1.0 - traj_batch.done) * (traj_batch.next_target - v_j)
                
                # 3. Magnitude (MC) term: (1 - γ) * E[e_i^2]
                magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)
                
                # 4. Laplacian (Smoothness) term: (γ / 2) * E[(e_i - e_j)^2]
                laplacian_loss = 0.5 * gamma * jnp.mean((e_i - e_j) ** 2)
                
                total_loss = magnitude_loss + laplacian_loss
                return total_loss, {
                    "total_loss": total_loss,
                    "magnitude_loss": magnitude_loss,
                    "laplacian_loss": laplacian_loss,
                    "value_loss": total_loss,
                }

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(E_loss_fn, has_aux=True)
                    
                    (total_loss, losses), grads = grad_fn(
                        train_state.params, traj_batch, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), losses

            initial_update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, _, rng = update_state

            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            # Shared Metrics
            metric.update({k: v.mean() for k, v in losses.items()})
            metric.update({"mean_rew": traj_batch.reward.mean()})
            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=True, light=config.get("LIGHT_METRICS", True)
            )
            metric.update(value_metrics)
            if config.get("LOG_FEATURE_METRICS", False):
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=True,
                ))
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

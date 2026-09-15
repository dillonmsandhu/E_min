# REINFORCE / GRPO-style policy gradient (for intrinsic value)
# uses a timestep dependent variant, based on batch index i.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import distrax
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "fixed_td0"


def td_loss(params, network, traj_batch, config):
    # VALUE LOSS
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    value_pred_next = network.apply(params, traj_batch.next_obs, method=network.value)
    value_pred_next = jax.lax.stop_gradient(value_pred_next)
    
    if "is_timeout" in traj_batch.info:
        true_terminal = traj_batch.done & ~traj_batch.info["is_timeout"]
        bootstrap_mask = 1.0 - true_terminal
    else:
        bootstrap_mask = 1.0 - traj_batch.done
        
    targets = traj_batch.reward + config["GAMMA"] * bootstrap_mask * value_pred_next
    targets = jax.lax.stop_gradient(targets)
    value_loss = helpers.ppo_clipped_v_loss(traj_batch, value_pred, targets, config) # trust region TD.
    total_loss = config["VF_COEF"] * value_loss
    return total_loss

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

    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = batch_size // base_config["MINIBATCH_SIZE"]
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    policy_fn, policy_matrix = helpers.get_evaluation_policies(base_config, evaluator)

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get('k', 32)
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params,)
        
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
                pi = policy_fn(last_obs)
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
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, traj_minibatch):
                    grad_fn = jax.value_and_grad(td_loss, has_aux=False)
                    
                    total_loss, grads = grad_fn(
                        train_state.params, network, traj_minibatch, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    return train_state, total_loss

                train_state, traj_batch, rng = update_state
                rng, _rng = jax.random.split(rng)
                minibatches = helpers.shuffle_and_batch(_rng, traj_batch, config["NUM_MINIBATCHES"])
                
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, rng), total_loss

            initial_update_state = (train_state, traj_batch, rng)
            update_state, total_loss = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, rng = update_state
            # --------- Metrics ---------
            metric = {
                k: v.mean() 
                for k, v in traj_batch.info.items() 
                if k not in ["real_next_obs", "real_next_state"]
            }
            # Shared Metrics
            metric.update(
                {
                    "total_loss": total_loss.mean(),
                    "value_loss": total_loss.mean(),
                    "mean_rew": traj_batch.reward.mean(),
                }
            )
            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn, light=config.get("LIGHT_METRICS", True)
            )
            metric.update(value_metrics)
            if config.get("LOG_FEATURE_METRICS", False):
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn)
                )

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

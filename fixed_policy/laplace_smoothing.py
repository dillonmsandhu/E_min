# REINFORCE / GRPO-style policy gradient (for intrinsic value)
# uses a timestep dependent variant, based on batch index i.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import distrax
import core.bellman_error as bellman_error

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "fixed_lapace"

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

def make_train(config):
    # Load the fixed policy we will evaluate:
    # model saved under ./results/{alg}/{sub_dir}
    model_dir = 'ppo/' + config['MODEL_LOAD_DIR']
    _, out = utils.load_run_data(model_dir, config['ENV_NAME'], 'results') 
    policy_train_state = out['runner_state'][0]
    
    # The saved train state is batched over N_SEEDS (which is 1 by default).
    # We need to extract the parameters for the first seed to remove this extra dimension.
    policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
    get_policy = lambda obs: policy_train_state.apply_fn(policy_params, obs)[0]
    def fixed_policy_fn(obs):
        "The fixed policy this script will sample from."       
        pi, _ = policy_train_state.apply_fn(policy_params, obs)
        return pi

    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    config["NUM_MINIBATCHES"] = batch_size // config["MINIBATCH_SIZE"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    def train(rng):
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
                pi = fixed_policy_fn(last_obs)
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
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, 0, info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # --- ADVANTAGE CALCULATION ---
            advantages, target = helpers.calculate_gae(traj_batch, config["GAMMA"], config["GAE_LAMBDA"], )
            # get the next targets
            next_targets = jnp.roll(target, shift=-1, axis=0)
            next_targets = next_targets.at[-1].set(traj_batch.next_value[-1])
            traj_batch= traj_batch._replace(next_target = next_targets)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info
                    grad_fn = jax.value_and_grad(helpers.v_loss_fn_laplacian_smoothing, has_aux=True)
                    # grad_fn = jax.value_and_grad(helpers.true_laplacian_smoothing, has_aux=True)
                    
                    # 1. Unpack the auxiliary tuple here!
                    (total_loss, losses), grads = grad_fn(
                        train_state.params, network, traj_batch, targets, config
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    # 2. Return them all so they get stacked by the scan
                    return train_state, losses

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch, advantages, targets)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                # loss_info is now a tuple of 4 arrays: (total_loss, value_loss, loss_actor, entropy)
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), losses

            initial_update_state = (train_state, traj_batch, advantages, target, rng)
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
            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, target_policy_fn=get_policy, light=config.get("LIGHT_METRICS", True)
            )
            metric.update(value_metrics)

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

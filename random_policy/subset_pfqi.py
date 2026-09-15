# Samples a set of states without replacement, and performs Partially Fitted Q iteration on them  
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "random/subset/pfqi"

def make_train(config):    
    # The saved train state is batched over N_SEEDS (which is 1 by default).
    # We need to extract the parameters for the first seed to remove this extra dimension.
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"]
    config['NUM_ENVS'] = 1
    config['NUM_STEPS'] = 1
    
    env, env_params = helpers.make_env(config)
    evaluator = helpers.initialize_evaluator(config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    S = evaluator.obs_stack
    n_states = len(evaluator.obs_stack) # also 
    
    # Policy to be evaluated
    def get_random_policy_matrix(obs_stack=None) -> jax.Array:
        """
        Produces a uniform random policy matrix PI of shape (num_total_states, n_actions).
        
        Args:
            n_states: The number of active states in the environment.
            n_actions: The total number of available actions.
        """
        # 1. Create uniform distribution for active states (1/N probability per action)
        pi_active = jnp.ones((n_states, n_actions)) / n_actions
        
        # 2. Create uniform distribution for the single terminal state
        pi_terminal = jnp.ones((1, n_actions)) / n_actions
        
        # 3. Stack them to match your evaluator's S+1 state requirement
        pi = jnp.vstack([pi_active, pi_terminal])
        
        return pi
    
    Pi = get_random_policy_matrix()
    
    # Get the Markov Chain
    
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", Pi, P)
    R_π = jnp.einsum("sa,sam,sam->s", Pi, P, evaluator.R)
    mu = evaluator.compute_stationary_distribution_raw(Pi[:-1, :])[0]
    mu = jnp.append(mu, 0.0)
    
    def train(rng, hparams=None):
        config = utils.merge_hparams(config, hparams) # Used for tuning: overwrite any config with the same key in hparams
        γ = config['GAMMA']
        k = config.get('k', 32)

        # Initialize Network
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        rng, loop_rng = jax.random.split(rng)
        runner_state = (train_state, loop_rng, 1)
        
        def _update_step(runner_state, unused):
            train_state, current_rng, idx = runner_state
            
            # --- 1. SAMPLE BATCH ---
            rng_sample, next_rng = jax.random.split(current_rng)
            
            # mu[:-1] contains probabilities for active states. Normalize to sum to 1.
            p_mu = mu[:-1] / jnp.sum(mu[:-1]) 
            batch_size = config.get('BATCH_SIZE', 32)
            batch_idx = jax.random.choice(
                rng_sample, 
                jnp.arange(n_states), 
                shape=(batch_size,), 
                replace=False 
            )
            
            batch_S = S[batch_idx]
            
            # Extract the true mu probabilities for these specific states
            batch_mu = mu[batch_idx]
            
            # Calculate the unbiased weight scaling factor: |S| / b
            weight_scale = n_states / batch_size
            
            # --- 2. COMPUTE FIXED TARGETS (PFQI Style) ---
            v_all = network.apply(train_state.params, S) 
            v_all = jnp.append(v_all, 0.0) 
            
            # Calculate full targets BEFORE the inner loop, then slice for the batch
            TD_targets_all = R_π + config['GAMMA'] * P_π @ v_all
            fixed_batch_targets = jax.lax.stop_gradient(TD_targets_all[batch_idx])
            
            # --- 3. INNER LOOP: E EPOCHS ON FIXED TARGETS ---
            def inner_td_loss(params):
                v_batch = network.apply(params, batch_S)
                td_errors = v_batch - fixed_batch_targets
                
                # Apply the Horvitz-Thompson unbiased weights
                unbiased_weights = batch_mu * weight_scale
                loss = 0.5 * jnp.sum(unbiased_weights * (td_errors ** 2))
                
                return loss
            
            inner_td_grad = jax.value_and_grad(inner_td_loss)
            
            def inner_td_step(current_train_state, unused):
                loss, grads = inner_td_grad(current_train_state.params)
                current_train_state = current_train_state.apply_gradients(grads=grads)
                return current_train_state, loss

            # Scan applies the gradient update NUM_EPOCHS times using the closed-over fixed targets
            train_state, losses = jax.lax.scan(
                inner_td_step, 
                train_state, 
                None, 
                length=config["NUM_EPOCHS"]
            )
            
            # --- 4. METRICS & LOGGING ---
            metric = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=True, light=config.get("LIGHT_METRICS", True)
            )
            metric.update({"total_loss": losses.mean(), "value_loss": losses.mean()})
            if config["LOG_FEATURE_METRICS"]:
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=True,)
                )
            runner_state = (train_state, next_rng, idx + 1)
            return runner_state, metric
            
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        
        final_train_state, _, final_idx = runner_state
        return {"runner_state": (final_train_state, final_idx), "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

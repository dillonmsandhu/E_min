# Samples a set of states without replacement, and performs TD learning on them. 
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error
from core.feature_metrics import feature_metrics

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "random/subset/E"

def make_train(base_config):    
    # The saved train state is batched over N_SEEDS (which is 1 by default).
    # We need to extract the parameters for the first seed to remove this extra dimension.
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"]
    base_config['NUM_ENVS'] = 1
    base_config['NUM_STEPS'] = 1
    base_config['NUM_EPOCHS'] = 1
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
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
    # construct weighting / laplace matrix
    D = jnp.diag(mu)
    I = jnp.eye(evaluator.num_total_states)
    A = D @ (I - base_config['GAMMA'] * P_π)
    S_mat = 0.5 * (A + A.T) 
    V = evaluator.compute_true_values_raw(Pi)


    def train(rng, hparams=None):
        
        config = utils.merge_hparams(base_config, hparams) # Used for tuning: overwrite any config with the same key in hparams
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
            
            # --- 1. SAMPLE DISTINCT STATES (UNIFORM WoR) ---
            rng_sample, next_rng = jax.random.split(current_rng)
            
            batch_size = config.get('BATCH_SIZE', 32)
            
            # Sample `BATCH_SIZE` distinct states
            batch_idx = jax.random.choice(
                rng_sample, 
                jnp.arange(n_states), 
                shape=(batch_size,), 
                replace=False 
            )
                        
            # Extract the true mu probabilities for these specific states
            batch_mu = mu[batch_idx]
            weight_scale = n_states / batch_size

            def E_loss(params):
                # --- Exact Sampled E Loss ---
                v_all = network.apply(params, S) 
                v_all = jnp.append(v_all, 0.0) 
                e_all = V - v_all
                e_batch = e_all[batch_idx]
                
                # Get the transition probabilities for just the sampled states
                # Shape: (batch_size, n_states)
                P_batch = P_π[batch_idx, :] 
                
                # Compute the squared differences between the batch errors and ALL errors
                # Broadcasting: (batch_size, 1) - (1, n_states) -> (batch_size, n_states)
                e_diff_sq = (e_batch[:, None] - e_all[None, :]) ** 2
                
                # Compute the Laplacian term (weighted average over possible next states)
                # Shape: (batch_size,)
                laplacian_term = 0.5 * config['GAMMA'] * jnp.sum(P_batch * e_diff_sq, axis=1)
                
                # Compute the magnitude term
                # Shape: (batch_size,)
                magnitude_term = (1.0 - config['GAMMA']) * (e_batch ** 2)
                
                # Combine and sum over the batch
                E_loss = jnp.sum(batch_mu * weight_scale * (magnitude_term + laplacian_term))

                return E_loss
            
            td_grad = jax.value_and_grad(E_loss)
            
            def td_step(current_train_state, unused):
                loss, grads = td_grad(current_train_state.params)
                current_train_state = current_train_state.apply_gradients(grads=grads)
                return current_train_state, loss

            train_state, losses = jax.lax.scan(
                td_step, 
                train_state, 
                None, 
                length=config["NUM_EPOCHS"]
            )
            
            # --- 3. METRICS & LOGGING ---
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

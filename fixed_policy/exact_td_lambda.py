# Exact TD Lambda for Evaluation of a pretrained policy
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "fixed/td_lambda_exact"

def make_train(base_config):    
    # The saved train state is batched over N_SEEDS (which is 1 by default).
    # We need to extract the parameters for the first seed to remove this extra dimension.
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"]
    base_config['NUM_ENVS'] = 1
    base_config['NUM_STEPS'] = 1
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    policy_fn, policy_matrix = helpers.get_evaluation_policies(base_config, evaluator)
    

    Pi = policy_matrix

    # Get the Markov Chain
    S = evaluator.obs_stack # does not include terminal state.
    P = evaluator.P # 3d tensor S x A x S' (does include terminal staet)
    P_π = jnp.einsum("sa,sam->sm", Pi, P)
    R_π = jnp.einsum("sa,sam,sam->s", Pi, P, evaluator.R)
    mu = evaluator.compute_stationary_distribution_raw(Pi[:-1, :])[0]
    mu = jnp.append(mu, 0.0)
    
    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams) # For tuning: overwrite config with hparams
        γ = config['GAMMA']
        k = config.get('k', 32)
        λ = config['VALUE_LAMBDA']
        k = config.get('k', 32)
        I = jnp.eye(len(S)+1) # terminal state.
        L = jnp.linalg.inv(I - γ * λ * P_π)

        # Initialize Network
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        runner_state = (train_state, 1)
        
        def T(v):
            return R_π + γ * P_π @ v

        def t_lambda(v):
            return v + L @ (T(v) - v)
        
        def td_loss(params):
            # each update step looks at all observations and produces v_theta(S)            
            v = network.apply(params, S) # 104 states, no terminal
            v = jnp.append(v, 0.0)
            td_errors = v - jax.lax.stop_gradient(t_lambda(v))
            loss = 0.5 * jnp.sum(mu * (td_errors ** 2))
            return loss
        
        td_grad = jax.value_and_grad(td_loss)
    
        def td_step(train_state, unused):
            loss, grads = td_grad(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss
        
        # Main Loop
        def _update_step(runner_state, unused):
            train_state, idx = runner_state
            # 1.  Apply expected update NUM_EPOCHS times
            train_state, loss = jax.lax.scan(td_step, train_state, None, config["NUM_EPOCHS"])
            # 2. Get value metrics and logging
            metric = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn, light=config.get("LIGHT_METRICS", True)
            )
            if config.get("LOG_FEATURE_METRICS", False):
                from core.feature_metrics import feature_metrics
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=False, target_policy_fn=policy_fn)
                )
            metric.update({"total_loss": loss.mean(), "value_loss": loss.mean()})
            runner_state = (train_state, idx + 1)
            return runner_state, metric
            
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

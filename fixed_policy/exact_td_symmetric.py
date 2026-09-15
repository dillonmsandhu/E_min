# Symmetrize the weighting used by TD learning. 
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error

# jax.config.update("jax_enable_x64", True)

SAVE_DIR = "fixed/td_exact_symmetric"

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
    
    # Policy to be evaluated
    # model saved under ./results/{alg}/{sub_dir}
    model_dir = 'ppo/' + config['MODEL_LOAD_DIR']
    _, out = utils.load_run_data(model_dir, config['ENV_NAME'], 'results') 
    policy_train_state = out['runner_state'][0]
    policy_params = jax.tree_util.tree_map(lambda x: x[0], policy_train_state.params)
    get_policy = lambda obs: policy_train_state.apply_fn(policy_params, obs)[0]
    def get_policy_matrix():
        "produces pi(.|S) where S is all states"
        pi_dist, _ = policy_train_state.apply_fn(policy_params, evaluator.obs_stack)
        pi = pi_dist.probs
        terminal_policy = jnp.ones( [1,n_actions], dtype=pi.dtype) / n_actions
        pi = jnp.vstack([pi, terminal_policy])
        return pi
    
    Pi = get_policy_matrix()
    
    # Get the Markov Chain
    ALL_STATES = evaluator.obs_stack
    I = jnp.eye(evaluator.num_total_states)
    P = evaluator.P # 3d tensor S x A x S'
    P_π = jnp.einsum("sa,sam->sm", Pi, P)
    R_π = jnp.einsum("sa,sam,sam->s", Pi, P, evaluator.R)
    mu = evaluator.compute_stationary_distribution_raw(Pi[:-1, :]) # uses the continuing version, where S_T -> S_0
    mu = jnp.append(mu, 0.0)
    D = jnp.diag(mu)
    A = D @ (I - config['GAMMA'] * P_π)
    S = 0.5 * (A + A.T)
    # W = S (DM)^{-1} D = SM^{-1}
    M = I - config['GAMMA'] * P_π
    M_inv = jnp.linalg.solve(M, I)
    W = S @ M_inv
    
    def train(rng):
        k = config.get('k', 32)
        # Initialize Network
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=1, layer_norm=config['LAYER_NORM']
        )
        total_grad_steps = config["NUM_UPDATES"] * config["NUM_EPOCHS"]
        lr_scheduler = optax.linear_schedule(config["LR"], config["LR_END"], total_grad_steps)
        tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adamw(lr_scheduler, 
                weight_decay = config.get('WEIGHT_DECAY', 1e-2),
                eps=config.get('ADAM_EPS', 1e-5)
                ),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        runner_state = (train_state, 1)
        
        def td_loss(params):
            # each update step looks at all observations and produces v_theta(S)            
            # TD minimizes Dδ = Ae = (S + K)e
            # To minimize Ke when only observing δ
            # we need weight matrix W where Wδ = Se
            # Wδ = inv(D) A e -> W = S A^{-1}D = (I-K inv(A)) D
            v = network.apply(params, ALL_STATES) # 104 states, no terminal
            v = jnp.append(v, 0.0)
            TD_targets = R_π + config['GAMMA'] * P_π @ v
            td_errors = v - jax.lax.stop_gradient(TD_targets)
            exact_gradient_vector = W @ td_errors
            loss = jnp.dot(v, jax.lax.stop_gradient(exact_gradient_vector))
            # When JAX calls grad(loss), it returns exactly exact_gradient_vector
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
                evaluator, network, train_state.params, random_policy=False, target_policy_fn=get_policy, light=config.get("LIGHT_METRICS", True)
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

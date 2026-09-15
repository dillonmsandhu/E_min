# see TD(λ) analogue of Tang and Munos's E
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error
from core.policy_metrics import compute_policy_metrics

SAVE_DIR = "ppo/exact_E_lambda"

def network_inference(params, network, S, n_actions):
    pi_dist, v = network.apply(params, S)
    pi = pi_dist.probs
    terminal_policy = jnp.ones([1, n_actions], dtype=pi.dtype) / n_actions
    pi = jnp.vstack([pi, terminal_policy])
    v = jnp.append(v, 0.0)
    return pi, v

def make_train(base_config):
    base_config = base_config.copy()
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"]
    base_config['NUM_ENVS'] = 1
    base_config['NUM_STEPS'] = 1
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    
    S_states = evaluator.obs_stack
    P = evaluator.P 
    I = jnp.eye(evaluator.num_total_states)

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams) # Used for tuning: overwrite any config with the same key in hparams
        γ = config['GAMMA']
        k = config.get('k', 32)

        # Initialize Network
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        runner_state = (train_state, 1)

        def _update_step(runner_state, unused):
            train_state, idx = runner_state

            # 1. Compute Exact Dynamics
            old_pi_dist, old_v = network.apply(train_state.params, S_states)
            old_pi = old_pi_dist.probs
            terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
            old_pi_full = jnp.vstack([old_pi, terminal_policy])
            old_log_pi = jnp.log(old_pi + 1e-8)
            
            old_v_full = jnp.append(old_v, 0.0)

            # Compute stationary distribution and E-loss matrix S
            mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
            mu = jnp.append(mu, 0.0)
            D = jnp.diag(mu)
            
            P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
            A_mat = D @ (I - γ * P_pi)
            λ_val = config['VALUE_LAMBDA']
            L = jnp.linalg.inv(I - γ * λ_val * P_pi)
            AL_mat = A_mat@L
            S_mat = 0.5 * (AL_mat + AL_mat.T)

            # True value function for the current policy (for E-loss target)
            V_true = evaluator.compute_true_values_raw(old_pi_full)

            R_pi = jnp.einsum("sa,sam,sam->s", old_pi_full, P, evaluator.R)
            λ_pi = config.get("GAE_LAMBDA", 0.6)
            A = helpers.compute_exact_advantage(
                P, evaluator.R, P_pi, R_pi, old_v_full, γ, λ_pi
            )
            w = mu[:-1, None] * old_pi
            A = helpers.post_process_advantage(A, config, weights=w)



            def loss_fn(params, network):
                # pi shape (num_states+1, n_actions), v shape (num_states+1,)
                pi, v = network_inference(params, network, S_states, n_actions)
                
                # E-Loss (Bellman error gradient descent loss)
                # (V_true - v)^T @ S @ (V_true - v)
                diff = V_true - v
                value_loss = (diff.T @ S_mat @ diff)

                # Policy Loss
                log_pi = jnp.log(pi[:-1, :] + 1e-8)
                log_pi_sum = jnp.sum(pi[:-1, :] * log_pi, axis=-1)
                entropy = -jnp.sum(mu[:-1] * log_pi_sum) 

                # PPO clip loss
                ratio = jnp.exp(log_pi - old_log_pi)
                pi_old = jnp.exp(old_log_pi)
    
                surr1 = ratio * A
                surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * A
                
                actor_loss = -jnp.sum(mu[:-1, None] * pi_old * jnp.minimum(surr1, surr2))

                total_loss = config.get("VF_COEF", 0.5) * value_loss + actor_loss - entropy * config.get("ENT_COEF", 0.01)

                return total_loss, (value_loss, actor_loss, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            # 3. Apply expected update NUM_EPOCHS times
            def epoch_step(train_state, unused):
                (total_loss, metrics), grads = grad_fn(train_state.params, network)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, metrics

            train_state, epoch_metrics = jax.lax.scan(epoch_step, train_state, None, config["NUM_EPOCHS"])
            
            # Metrics
            value_loss, actor_loss, entropy = epoch_metrics
            metric = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, light=config.get("LIGHT_METRICS", True)
            )
            if config.get("LOG_FEATURE_METRICS", False):
                from core.feature_metrics import feature_metrics
                metric.update(feature_metrics(
                    evaluator, network, train_state.params, random_policy=False,)
                )
            # Policy tracking metrics (TV distance between policies and stationary distributions, state coverage)
            new_pi = network.apply(train_state.params, S_states)[0].probs
            new_mu = evaluator.compute_stationary_distribution_raw(new_pi)[0]
            metric.update(compute_policy_metrics(new_pi, old_pi, new_mu, mu[:-1]))
            metric.update({
                "total_loss": (value_loss + actor_loss - entropy).mean(),
                "value_loss": value_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "entropy": entropy.mean(),
                "v_pred_start": old_v[evaluator.start_idx],
                "Mean_A": A.mean(),
            })
            
            runner_state = (train_state, idx + 1)
            return runner_state, metric

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

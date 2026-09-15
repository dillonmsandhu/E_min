# trains a policy network with PPO where the critic is learned by minimizing
# the exact Bellman error loss using the sampling formulation (Dirichlet / spectral decomposition).
# Includes a scheduled Dirichlet energy coefficient and PPO-style value clipping.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error
from core.policy_metrics import compute_policy_metrics

SAVE_DIR = "ppo/exact_E_sampling_form"

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

            # Compute stationary distribution and transition matrix under current policy
            mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
            mu = jnp.append(mu, 0.0)
            
            P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)

            # True value function for the current policy (for E-loss target)
            V_true = evaluator.compute_true_values_raw(old_pi_full)

            R_pi = jnp.einsum("sa,sam,sam->s", old_pi_full, P, evaluator.R)
            λ_pi = config.get("GAE_LAMBDA", 0.6)
            A = helpers.compute_exact_advantage(
                P, evaluator.R, P_pi, R_pi, old_v_full, γ, λ_pi
            )
            w = mu[:-1, None] * old_pi
            A = helpers.post_process_advantage(A, config, weights=w)


            # Dirichlet energy schedule configuration:
            # Default is 1.0 (recovers exact Bellman error loss identically to exact_E.py).
            # To enable schedule (e.g. 0 -> 1), set DIRICHLET_COEF_START=0.0 in config.
            c_start = config.get("DIRICHLET_COEF_START", 1.0)
            c_end = config.get("DIRICHLET_COEF_END", 1.0)
            schedule_steps = config.get("DIRICHLET_SCHEDULE_UPDATES", config["NUM_UPDATES"])
            progress = jnp.clip((idx - 1.0) / jnp.maximum(schedule_steps - 1.0, 1.0), 0.0, 1.0)
            dirichlet_coef = c_start + (c_end - c_start) * progress

            # Clipping parameters:
            # Default is False (recovers unclipped exact Bellman error loss identically to exact_E.py).
            # To enable PPO-style value clipping, set CLIP_DIRICHLET=True in config.
            vf_clip = config.get("VF_CLIP", 0.2)
            clip_dirichlet = config.get("CLIP_DIRICHLET", False)
            clip_type = config.get("DIRICHLET_CLIP_TYPE", "max")
            clip_magnitude = config.get("VF_CLIP_MAGNITUDE", False)

            def loss_fn(params, network):
                # pi shape (num_states+1, n_actions), v shape (num_states+1,)
                pi, v = network_inference(params, network, S_states, n_actions)
                
                # Sampling formulation of E:
                # e = V_true - v
                # Magnitude term: (1 - gamma) * ||e||_mu^2
                # Dirichlet energy term: (gamma / 2) * sum_{ij} mu_i * P_ij * (e_i - e_j)^2
                e = V_true - v

                # A) Magnitude term
                mag_unclipped = (1.0 - γ) * (e ** 2)
                if clip_magnitude:
                    v_clipped_mag = old_v_full + jnp.clip(v - old_v_full, -vf_clip, vf_clip)
                    e_clipped_mag = V_true - v_clipped_mag
                    mag_clipped = (1.0 - γ) * (e_clipped_mag ** 2)
                    mag_per_state = jnp.maximum(mag_unclipped, mag_clipped)
                else:
                    mag_per_state = mag_unclipped
                magnitude_loss = jnp.sum(mu * mag_per_state)

                # B) Dirichlet energy term with PPO-style clipping
                diff_e = e[:, None] - e[None, :] # shape (S+1, S+1)
                lap_unclipped = 0.5 * γ * jnp.sum(P_pi * (diff_e ** 2), axis=1) # shape (S+1,)

                if clip_dirichlet:
                    v_clipped = old_v_full + jnp.clip(v - old_v_full, -vf_clip, vf_clip)
                    e_clipped = V_true - v_clipped
                    diff_e_clipped = e_clipped[:, None] - e_clipped[None, :]
                    lap_clipped = 0.5 * γ * jnp.sum(P_pi * (diff_e_clipped ** 2), axis=1)

                    if clip_type == "max":
                        lap_per_state = jnp.maximum(lap_unclipped, lap_clipped)
                    else: # "direct"
                        lap_per_state = lap_clipped
                else:
                    lap_per_state = lap_unclipped

                dirichlet_loss = jnp.sum(mu * lap_per_state)

                # Combined Value Loss
                value_loss = magnitude_loss + dirichlet_coef * dirichlet_loss

                # C) Policy Loss
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

                clip_frac = jnp.mean(jnp.abs(v[:-1] - old_v) > vf_clip)

                return total_loss, (value_loss, magnitude_loss, dirichlet_loss, actor_loss, entropy, clip_frac)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            # 3. Apply expected update NUM_EPOCHS times
            def epoch_step(train_state, unused):
                (total_loss, metrics), grads = grad_fn(train_state.params, network)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, metrics

            train_state, epoch_metrics = jax.lax.scan(epoch_step, train_state, None, config["NUM_EPOCHS"])
            
            # Metrics
            value_loss, magnitude_loss, dirichlet_loss, actor_loss, entropy, clip_frac = epoch_metrics
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
                "magnitude_loss": magnitude_loss.mean(),
                "dirichlet_loss": dirichlet_loss.mean(),
                "dirichlet_coef": dirichlet_coef,
                "dirichlet_clip_frac": clip_frac.mean(),
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

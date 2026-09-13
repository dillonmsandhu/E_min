# oracle PPO.
# trains a policy network with PPO where the critic is the true value function at each iteration.
# provides a skyline on PPO with a specific policy net.
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils

SAVE_DIR = "ppo/ground_truth"


def network_inference(params, network, S, n_actions):
    "Maps from network to policy and value vectors in R^S"
    pi_dist, v = network.apply(params, S)
    pi = pi_dist.probs
    terminal_policy = jnp.ones([1, n_actions], dtype=pi.dtype) / n_actions
    pi = jnp.vstack([pi, terminal_policy])
    v = jnp.append(v, 0.0)
    return pi, v

def make_train(base_config):
    base_config = base_config.copy()
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"]
    base_config["NUM_ENVS"] = 1
    base_config["NUM_STEPS"] = 1

    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n

    S = evaluator.obs_stack
    P = evaluator.P

    def train(rng, hparams=None):
        config = utils.merge_hparams(
            base_config, hparams
        )  # Used for tuning: overwrite any config with the same key in hparams
        γ = config["GAMMA"]
        k = config.get("k", 32)

        # Initialize Network
        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config["LAYER_NORM"]
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        runner_state = (train_state, 1)

        def _update_step(runner_state, unused):
            train_state, idx = runner_state

            # 1. Compute Exact Dynamics
            old_pi_dist, _ = network.apply(train_state.params, S)
            old_pi = old_pi_dist.probs
            terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
            old_pi_full = jnp.vstack([old_pi, terminal_policy])
            old_log_pi = jnp.log(old_pi + 1e-8)

            mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
            has_nans = jnp.isnan(mu).any()
            # jax.debug.print(
            #     "Step {idx} | NaN in mu: {has_nans} | Max mu: {max_mu}", 
            #     idx=idx, 
            #     has_nans=has_nans,
            #     max_mu=jnp.max(jnp.nan_to_num(mu))
            # )
            # # ----------
            mu = jnp.append(mu, 0.0)

            V = evaluator.compute_true_values_raw(old_pi_full)

            # 2. Compute Advantages
            P_pi = jnp.einsum("sa,sam->sm", old_pi_full, P)
            R_pi = jnp.einsum("sa,sam,sam->s", old_pi_full, P, evaluator.R)
            A = helpers.compute_exact_advantage(
                P, evaluator.R, P_pi, R_pi, V, γ, 1.0
            )
            w = mu[:-1, None] * old_pi
            A = helpers.post_process_advantage(A, config, weights=w)


            def loss_fn(params, network):
                # A shape is (num_states, num_actions)
                pi, _ = network_inference(params, network, S, n_actions)

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

                total_loss = actor_loss - entropy * config.get("ENT_COEF", 0.01)
                return total_loss, (total_loss, actor_loss, entropy)

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

            # 3. Apply expected update NUM_EPOCHS times
            def epoch_step(train_state, unused):
                (total_loss, metrics), grads = grad_fn(train_state.params, network)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, metrics

            train_state, epoch_metrics = jax.lax.scan(epoch_step, train_state, None, config["NUM_EPOCHS"])

            # Metrics
            total_loss, actor_loss, entropy = epoch_metrics
            metric = {
                "total_loss": total_loss.mean(),
                "entropy": entropy.mean(),
                "V_start": V.squeeze()[evaluator.start_idx],
                "V_mean": V.mean(),
                "Value_Grid": evaluator.get_value_grid(V),
            }
            print(idx)
            runner_state = (train_state, idx + 1)
            return runner_state, metric

        # end update

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from core.runner import run_experiment_main

    run_experiment_main(make_train, SAVE_DIR)

# hybrid version of ppo: exact MC target for value, standard rollout GAE for actor
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
from flax.training.train_state import TrainState
import core.bellman_error as bellman_error
from core.policy_metrics import compute_policy_metrics

SAVE_DIR = "ppo/hybrid_exact_mc"

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def network_inference(params, network, S, n_actions):
    pi_dist, v = network.apply(params, S)
    pi = pi_dist.probs
    terminal_policy = jnp.ones([1, n_actions], dtype=pi.dtype) / n_actions
    pi = jnp.vstack([pi, terminal_policy])
    v = jnp.append(v, 0.0)
    return pi, v

def make_train(base_config):
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = batch_size // base_config["MINIBATCH_SIZE"]
    base_config["NUM_UPDATES"] = base_config["TOTAL_TIMESTEPS"] // batch_size
    
    env, env_params = helpers.make_env(base_config)
    evaluator = helpers.initialize_evaluator(base_config, env, env_params)
    obs_shape = env.observation_space(env_params).shape
    n_actions = env.action_space(env_params).n
    
    S_states = evaluator.obs_stack
    P = evaluator.P 
    I = jnp.eye(evaluator.num_total_states)

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        γ = config['GAMMA']
        k = config.get('k', 32)

        network, network_params = networks.initialize_network(
            rng, obs_shape, env, env_params, k, n_heads=2, layer_norm=config['LAYER_NORM']
        )
        train_state = networks.initialize_flax_train_state(config, network, network_params)
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        runner_state = (train_state, env_state, obsv, rng, 1)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # ==========================================
            # 1. COLLECT TRAJECTORIES
            # ==========================================
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info['real_next_obs']
                next_val = network.apply(train_state.params, true_next_obs, method=network.value)

                clean_info = {k: v for k, v in info.items() if k not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(_env_step, env_step_state, None, config["NUM_STEPS"])

            # --- ADVANTAGE CALCULATION (SAMPLED) ---
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], config["GAE_LAMBDA"])
            advantages = helpers.post_process_advantage(advantages, config)


            # ==========================================
            # 2. EXACT DYNAMICS TARGETS (MC)
            # ==========================================
            old_pi_dist, old_v = network.apply(train_state.params, S_states)
            old_pi = old_pi_dist.probs
            terminal_policy = jnp.ones([1, n_actions], dtype=old_pi.dtype) / n_actions
            old_pi_full = jnp.vstack([old_pi, terminal_policy])
            
            mu = evaluator.compute_stationary_distribution_raw(old_pi)[0]
            mu = jnp.append(mu, 0.0)

            # True value function as the exact MC value net target
            V_true = evaluator.compute_true_values_raw(old_pi_full)


            # ==========================================
            # 3. UPDATE EPOCHS
            # ==========================================
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    obs_mb, action_mb, log_prob_mb, advantages_mb = batch_info

                    def loss_fn(params, network):
                        # A) Sampled Actor Loss
                        pi, _ = network.apply(params, obs_mb)
                        log_prob = pi.log_prob(action_mb)
                        entropy = pi.entropy().mean()
                        ratio = jnp.exp(log_prob - log_prob_mb)
                        surr1 = ratio * advantages_mb
                        surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * advantages_mb
                        actor_loss = -jnp.minimum(surr1, surr2).mean()

                        # B) Exact Critic Loss
                        _, v_full = network_inference(params, network, S_states, n_actions)
                        value_errors = v_full - V_true
                        value_loss = 0.5 * jnp.sum(mu * (value_errors**2))
                        
                        total_loss = actor_loss + config.get("VF_COEF", 0.5) * value_loss - entropy * config.get("ENT_COEF", 0.01)
                        return total_loss, (value_loss, actor_loss, entropy)

                    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                    (total_loss, metrics), grads = grad_fn(train_state.params, network)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, metrics

                train_state, traj_batch, advantages, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch = (traj_batch.obs, traj_batch.action, traj_batch.log_prob, advantages)
                minibatches = helpers.shuffle_and_batch(_rng, batch, config["NUM_MINIBATCHES"])
                
                train_state, epoch_metrics = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, rng), epoch_metrics

            initial_update_state = (train_state, traj_batch, advantages, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, initial_update_state, None, config["NUM_EPOCHS"])
            train_state, _, _, rng = update_state

            # ==========================================
            # 4. METRICS
            # ==========================================
            metric = {k: v.mean() for k, v in traj_batch.info.items()}
            # loss_info metrics have shape (NUM_EPOCHS, NUM_MINIBATCHES)
            value_loss, actor_loss, entropy = loss_info
            
            metric.update({
                "total_loss": (value_loss + actor_loss - entropy).mean(),
                "value_loss": value_loss.mean(),
                "actor_loss": actor_loss.mean(),
                "entropy": entropy.mean(),
                "mean_rew": traj_batch.reward.mean(),
                "v_pred_start": old_v[evaluator.start_idx],
            })

            value_metrics = bellman_error.value_metrics(
                evaluator, network, train_state.params, random_policy=False, light=config.get("LIGHT_METRICS", True)
            )
            metric.update(value_metrics)

            if config.get("LOG_FEATURE_METRICS", False):
                from core.feature_metrics import feature_metrics
                metric.update(feature_metrics(evaluator, network, train_state.params, random_policy=False))
            
            new_pi = network.apply(train_state.params, S_states)[0].probs
            new_mu = evaluator.compute_stationary_distribution_raw(new_pi)[0]
            metric.update(compute_policy_metrics(new_pi, old_pi, new_mu, mu[:-1]))

            runner_state = (train_state, env_state, last_obs, rng, idx + 1)
            return runner_state, metric

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}

    return train

if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

# E_experimental: Decoupled Actor/Critic Optimization, Huber Loss, and Multi-Value Head Consensus
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import core.runtime_metrics as runtime_metrics

SAVE_DIR = "E_experimental"


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


def make_train(base_config):
    base_config = base_config.copy()
    batch_size = base_config["NUM_STEPS"] * base_config["NUM_ENVS"]
    base_config["NUM_MINIBATCHES"] = max(1, batch_size // base_config.get("MINIBATCH_SIZE", batch_size))
    base_config["NUM_UPDATES"] = max(1, base_config["TOTAL_TIMESTEPS"] // batch_size)

    env, env_params = helpers.make_env(base_config)
    obs_shape = env.observation_space(env_params).shape

    def train(rng, hparams=None):
        config = utils.merge_hparams(base_config, hparams)
        k = config.get("k", 32)
        n_value_heads = config.get("NUM_VALUE_HEADS", 1)

        network, network_params = networks.initialize_network(
            rng,
            obs_shape,
            env,
            env_params,
            k,
            n_heads=2,
            layer_norm=config["LAYER_NORM"],
            n_value_heads=n_value_heads,
        )
        train_state = networks.initialize_split_flax_train_state(config, network, network_params)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        start_obs = obsv

        runner_state = (train_state, env_state, obsv, rng, 1)

        critic_epochs = config.get("CRITIC_EPOCHS")
        if critic_epochs is None:
            critic_epochs = config.get("NUM_EPOCHS", 4)

        actor_epochs = config.get("ACTOR_EPOCHS")
        if actor_epochs is None:
            actor_epochs = config.get("NUM_EPOCHS", 4)

        critic_loss_type = config.get("CRITIC_LOSS_TYPE", "mse")
        huber_delta = config.get("HUBER_DELTA", 1.0)
        value_head_agg = config.get("VALUE_HEAD_AGG", "sum")

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                # Returns consensus scalar value (mean across heads) for actor & rollouts
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                    rng_step, env_state, action, env_params
                )
                true_next_obs = info["real_next_obs"].reshape(last_obs.shape)
                _, next_val = network.apply(train_state.params, true_next_obs)

                clean_info = {k: v for k, v in info.items() if k not in ["real_next_obs", "real_next_state"]}
                transition = Transition(
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, 0.0, clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. SEPARATE ADVANTAGE AND VALUE TARGET CALCULATIONS
            gae_lambda = config.get("GAE_LAMBDA", 0.95)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            return_lambda = config.get("RETURN_LAMBDA", 1.0)
            _, targets = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)

            # Align next targets G_{t+1} for adjacent state error calculation
            rolled_targets = jnp.roll(targets, shift=-1, axis=0)
            rolled_targets = rolled_targets.at[-1].set(traj_batch.next_value[-1])

            is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
            true_terminal = traj_batch.done & ~is_timeout

            next_targets = jnp.where(
                true_terminal,
                0.0,
                jnp.where(is_timeout, traj_batch.next_value, rolled_targets),
            )
            traj_batch = traj_batch._replace(next_target=next_targets)

            # 3. SPLIT OPTIMIZATION: CRITIC UPDATE PHASE (E-MINIMIZATION)
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    obs_mb, next_obs_mb, done_mb, is_timeout_mb, next_target_mb, targets_mb = mb

                    def _c_loss(critic_params):
                        full_params = {"params": {**train_state.actor.params, **critic_params}}
                        # Evaluates all k value heads (shape: B, K or B)
                        v_i = network.apply(full_params, obs_mb, method=network.value)
                        v_j = network.apply(full_params, next_obs_mb, method=network.value)
                        
                        if v_j.ndim == 2:
                            done_mask = done_mb[:, None]
                        else:
                            done_mask = done_mb
                        v_j = jnp.where(done_mask, 0.0, v_j)

                        val_loss, mag_loss, lap_loss = helpers.e_experimental_critic_loss(
                            v_i,
                            targets_mb,
                            v_j,
                            next_target_mb,
                            done_mb,
                            config["GAMMA"],
                            is_timeout=is_timeout_mb,
                            loss_type=critic_loss_type,
                            delta=huber_delta,
                            agg=value_head_agg,
                        )
                        scaled_loss = config.get("VF_COEF", 0.5) * val_loss
                        return scaled_loss, (val_loss, mag_loss, lap_loss)

                    grad_fn = jax.value_and_grad(_c_loss, has_aux=True)
                    (_, (val_loss, mag_loss, lap_loss)), grads = grad_fn(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads=grads)
                    return train_state, (val_loss, mag_loss, lap_loss)

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = helpers.shuffle_and_batch(_rng, critic_batch, config["NUM_MINIBATCHES"])
                train_state, c_losses = jax.lax.scan(_critic_minibatch, train_state, minibatches)
                return (train_state, critic_batch, rng), c_losses

            critic_batch = (
                traj_batch.obs,
                traj_batch.next_obs,
                true_terminal,
                is_timeout,
                traj_batch.next_target,
                targets,
            )
            rng, _rng = jax.random.split(rng)
            c_init_state = (train_state, critic_batch, _rng)
            (train_state, _, rng), c_loss_epochs = jax.lax.scan(_critic_epoch, c_init_state, None, critic_epochs)

            # 4. SPLIT OPTIMIZATION: ACTOR UPDATE PHASE (PPO CLIPPED SURROGATE)
            def _actor_epoch(actor_state, unused):
                def _actor_minibatch(train_state, mb):
                    obs_mb, action_mb, log_prob_mb, adv_mb = mb

                    def _a_loss(actor_params):
                        full_params = {"params": {**actor_params, **train_state.critic.params}}
                        pi = network.apply(full_params, obs_mb, method=network.policy)
                        log_prob = pi.log_prob(action_mb)
                        entropy = pi.entropy().mean()
                        ratio = jnp.exp(log_prob - log_prob_mb)
                        adv_norm = helpers.post_process_advantage(adv_mb, config)
                        surr1 = ratio * adv_norm
                        surr2 = (
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"])
                            * adv_norm
                        )
                        actor_loss = -jnp.minimum(surr1, surr2).mean()
                        total_a_loss = (
                            config.get("POLICY_COEFF", 1.0) * actor_loss
                            - config.get("ENT_COEF", 0.01) * entropy
                        )
                        return total_a_loss, (actor_loss, entropy)

                    grad_fn = jax.value_and_grad(_a_loss, has_aux=True)
                    (_, (actor_loss, entropy)), grads = grad_fn(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads=grads)
                    return train_state, (actor_loss, entropy)

                train_state, actor_batch, rng = actor_state
                rng, _rng = jax.random.split(rng)
                minibatches = helpers.shuffle_and_batch(_rng, actor_batch, config["NUM_MINIBATCHES"])
                train_state, a_losses = jax.lax.scan(_actor_minibatch, train_state, minibatches)
                return (train_state, actor_batch, rng), a_losses

            actor_batch = (
                traj_batch.obs,
                traj_batch.action,
                traj_batch.log_prob,
                advantages,
            )
            rng, _rng = jax.random.split(rng)
            a_init_state = (train_state, actor_batch, _rng)
            (train_state, _, rng), a_loss_epochs = jax.lax.scan(_actor_epoch, a_init_state, None, actor_epochs)

            # 5. METRICS & LOSS LOGGING
            val_loss = c_loss_epochs[0].mean()
            mag_loss = c_loss_epochs[1].mean()
            lap_loss = c_loss_epochs[2].mean()
            act_loss = a_loss_epochs[0].mean()
            ent = a_loss_epochs[1].mean()

            loss_info = {
                "total_loss": val_loss + act_loss,
                "value_loss": val_loss,
                "magnitude_loss": mag_loss,
                "laplacian_loss": lap_loss,
                "actor_loss": act_loss,
                "entropy": ent,
            }

            metric = runtime_metrics.compute_runtime_metrics(
                train_state=train_state,
                network=network,
                traj_batch=traj_batch,
                loss_info=loss_info,
                start_obs=start_obs,
                config=config,
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

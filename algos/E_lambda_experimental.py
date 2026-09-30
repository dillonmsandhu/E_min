# E_lambda_experimental: Decoupled Actor/Critic Optimization with Geometric Jump Sampled E(lambda)
from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.utils as utils
import core.runtime_metrics as runtime_metrics

SAVE_DIR = "E_lambda_experimental"


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    next_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def e_lambda_geometric_multihead_critic_loss(
    v_i: jnp.ndarray,
    targets_i: jnp.ndarray,
    v_jump: jnp.ndarray,
    targets_jump: jnp.ndarray,
    valid_jump: jnp.ndarray,
    is_absorbing: jnp.ndarray,
    gamma: float,
    lmbda: float,
    loss_type: str = "mse",
    delta: float = 1.0,
    agg: str = "sum",
):
    """
    Computes multi-head transition-level sampled E(lambda) critic loss with geometric jump pairs.
    Supports single or multi-head value outputs (B or B, K).
    """
    if v_i.ndim == 2:
        targets_i = targets_i.reshape(-1, 1)
        targets_jump = targets_jump.reshape(-1, 1)
        valid_jump = valid_jump.reshape(-1, 1)
        is_absorbing = is_absorbing.reshape(-1, 1)

    e_i = targets_i - v_i
    e_jump = targets_jump - v_jump

    if loss_type == "huber":
        diff_sq = jnp.where(
            is_absorbing,
            helpers.huber_loss(e_i, delta),
            jnp.where(valid_jump, helpers.huber_loss(e_i - e_jump, delta), 0.0),
        )
        mag_sq = helpers.huber_loss(e_i, delta)
    else:
        diff_sq = jnp.where(
            is_absorbing,
            e_i ** 2,
            jnp.where(valid_jump, (e_i - e_jump) ** 2, 0.0),
        )
        mag_sq = e_i ** 2

    gl = gamma * lmbda
    safe_denom = jnp.maximum(1.0 - gl, 1e-8)
    tilde_gamma = (gamma * (1.0 - lmbda)) / safe_denom
    magnitude_weight = (1.0 - gamma) / safe_denom
    dirichlet_weight = 0.5 * tilde_gamma

    if v_i.ndim == 2:
        mag_per_head = magnitude_weight * jnp.mean(mag_sq, axis=0)
        dir_per_head = dirichlet_weight * jnp.mean(diff_sq, axis=0)
        loss_per_head = mag_per_head + dir_per_head

        if agg == "sum":
            val_loss = jnp.sum(loss_per_head)
            mag_loss = jnp.sum(mag_per_head)
            dir_loss = jnp.sum(dir_per_head)
        else:
            val_loss = jnp.mean(loss_per_head)
            mag_loss = jnp.mean(mag_per_head)
            dir_loss = jnp.mean(dir_per_head)
    else:
        mag_loss = magnitude_weight * jnp.mean(mag_sq)
        dir_loss = dirichlet_weight * jnp.mean(diff_sq)
        val_loss = mag_loss + dir_loss

    return val_loss, mag_loss, dir_loss


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
        e_lambda = config.get("E_LAMBDA", config.get("VALUE_LAMBDA", 0.8))

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                # Evaluates policy and consensus value (mean across heads) for rollouts
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
                    done, action, value, next_val, reward, log_prob, last_obs, true_next_obs, clean_info
                )
                return (train_state, env_state, obsv, rng), transition

            env_step_state = (train_state, env_state, last_obs, rng)
            (_, env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, env_step_state, None, config["NUM_STEPS"]
            )

            # 2. POLICY ADVANTAGES & BASELINE RETURNS
            gae_lambda = config.get("GAE_LAMBDA", 0.8)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            return_lambda = config.get("RETURN_LAMBDA", 0.99)
            _, returns = helpers.calculate_gae(traj_batch, config["GAMMA"], return_lambda)

            # Sample geometric lookahead jumps on the rollout buffer
            rng, _rng = jax.random.split(rng)
            obs_jump, targets_jump, valid_jump, is_absorbing = helpers.prepare_geometric_jump_transitions(
                traj_batch, returns, config["GAMMA"], e_lambda, _rng
            )

            # 3. SPLIT OPTIMIZATION: CRITIC UPDATE PHASE (E(lambda) WITH GEOMETRIC JUMPS)
            def _critic_epoch(critic_state, unused):
                def _critic_minibatch(train_state, mb):
                    (
                        obs_mb,
                        returns_mb,
                        obs_jump_mb,
                        targets_jump_mb,
                        valid_jump_mb,
                        is_absorbing_mb,
                    ) = mb

                    def _c_loss(critic_params):
                        full_params = {"params": {**train_state.actor.params, **critic_params}}
                        # Evaluates all k value heads
                        v_i = network.apply(full_params, obs_mb, method=network.value)
                        v_jump = network.apply(full_params, obs_jump_mb, method=network.value)

                        val_loss, mag_loss, dir_loss = e_lambda_geometric_multihead_critic_loss(
                            v_i=v_i,
                            targets_i=returns_mb,
                            v_jump=v_jump,
                            targets_jump=targets_jump_mb,
                            valid_jump=valid_jump_mb,
                            is_absorbing=is_absorbing_mb,
                            gamma=config["GAMMA"],
                            lmbda=e_lambda,
                            loss_type=critic_loss_type,
                            delta=huber_delta,
                            agg=value_head_agg,
                        )
                        scaled_loss = config.get("VF_COEF", 0.5) * val_loss
                        return scaled_loss, (val_loss, mag_loss, dir_loss)

                    grad_fn = jax.value_and_grad(_c_loss, has_aux=True)
                    (_, (val_loss, mag_loss, dir_loss)), grads = grad_fn(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads=grads)
                    return train_state, (val_loss, mag_loss, dir_loss)

                train_state, critic_batch, rng = critic_state
                rng, _rng = jax.random.split(rng)
                minibatches = helpers.shuffle_and_batch(_rng, critic_batch, config["NUM_MINIBATCHES"])
                train_state, c_losses = jax.lax.scan(_critic_minibatch, train_state, minibatches)
                return (train_state, critic_batch, rng), c_losses

            critic_batch = (
                traj_batch.obs,
                returns,
                obs_jump,
                targets_jump,
                valid_jump,
                is_absorbing,
            )
            rng, _rng = jax.random.split(rng)
            c_init_state = (train_state, critic_batch, _rng)
            (train_state, _, rng), c_loss_epochs = jax.lax.scan(
                _critic_epoch, c_init_state, None, critic_epochs
            )

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
                        clip_eps = config.get("CLIP_EPS", 0.2)
                        surr2 = (
                            jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
                        )
                        actor_loss = -jnp.minimum(surr1, surr2).mean()
                        total_actor_loss = (
                            config.get("POLICY_COEFF", 1.0) * actor_loss
                            - config.get("ENT_COEF", 0.01) * entropy
                        )
                        return total_actor_loss, (actor_loss, entropy)

                    grad_fn = jax.value_and_grad(_a_loss, has_aux=True)
                    (_, (act_loss, ent)), grads = grad_fn(train_state.actor.params)
                    train_state = train_state.apply_actor_gradients(grads=grads)
                    return train_state, (act_loss, ent)

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
            (train_state, _, rng), a_loss_epochs = jax.lax.scan(
                _actor_epoch, a_init_state, None, actor_epochs
            )

            # Packaging losses for metrics tracking
            val_loss_epoch, mag_loss_epoch, dir_loss_epoch = c_loss_epochs
            act_loss_epoch, ent_epoch = a_loss_epochs

            loss_info = {
                "value_loss": val_loss_epoch,
                "magnitude_loss": mag_loss_epoch,
                "dirichlet_loss": dir_loss_epoch,
                "actor_loss": act_loss_epoch,
                "entropy": ent_epoch,
                "total_loss": val_loss_epoch + act_loss_epoch,
            }

            # 5. RUNTIME METRICS
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
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    from core.runner import run_experiment_main
    run_experiment_main(make_train, SAVE_DIR)

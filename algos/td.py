"""
td.py

Classic Temporal Difference (TD) Learning with Online Delta Updates:
Unlike PPO which trains against a static, precomputed TD(lambda) target from rollout time
(fitted value iteration), this implementation updates the estimate of delta:
    - In 1-step TD (TD_LAMBDA == 0.0, default): Evaluates v(s') and computes delta = r + gamma * v(s') - v(s)
      at EVERY MINIBATCH gradient step using the CURRENT critic parameters.
    - In TD(lambda) (TD_LAMBDA > 0.0): Re-evaluates v(s) and v(s') across the trajectory with current
      parameters and recomputes the forward delta and lambda-return targets at EVERY CRITIC EPOCH.

Features (identical to E_experimental):
    - Decoupled Actor and Critic Optimization:
        * CRITIC_EPOCHS vs ACTOR_EPOCHS
        * CRITIC_WEIGHT_DECAY vs ACTOR_WEIGHT_DECAY via DualTrainState
    - Multiple Value Heads (NUM_VALUE_HEADS = k):
        * Evaluates and updates k distinct value heads simultaneously.
        * Policy rollouts and advantage estimation use the consensus ensemble mean v_bar(s).
    - Loss Selection:
        * CRITIC_LOSS_TYPE: 'mse' or 'huber' (with HUBER_DELTA).
    - Aggregation Selection:
        * VALUE_HEAD_AGG: 'sum' or 'mean' across value heads.
"""

from core.imports import *
import core.helpers as helpers
import core.networks as networks
import core.runtime_metrics as runtime_metrics
import core.utils as utils

SAVE_DIR = "td"


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


def calculate_recomputed_td_lambda_targets(v_traj, v_next_traj, reward, done, is_timeout, gamma, td_lambda):
    """
    Computes fresh multi-head TD(lambda) targets from trajectory values:
        delta_t = r_t + gamma * bootstrap_mask * v_{t+1} - v_t
        target_t = v_t + sum_{l=0}^inf (gamma * lambda)^l delta_{t+l}
    Shapes:
        v_traj: (T, B, K)
        v_next_traj: (T, B, K)
        reward: (T, B, 1)
        done: (T, B, 1)
        is_timeout: (T, B, 1)
    Returns:
        targets: (T, B, K) with jax.lax.stop_gradient applied
    """
    true_terminal = done & ~is_timeout
    bootstrap_mask = 1.0 - true_terminal.astype(jnp.float32)
    boundary_mask = 1.0 - done.astype(jnp.float32)

    def _get_targets(acc_gae, trans_step):
        v_t, v_next_t, r_t, b_mask, bound_mask = trans_step
        delta = r_t + gamma * b_mask * v_next_t - v_t
        gae = delta + gamma * td_lambda * bound_mask * acc_gae
        target = v_t + gae
        return gae, target

    init_acc = jnp.zeros_like(v_traj[0])
    scan_inputs = (v_traj, v_next_traj, reward, bootstrap_mask, boundary_mask)
    _, targets = jax.lax.scan(_get_targets, init_acc, scan_inputs, reverse=True, unroll=16)
    return jax.lax.stop_gradient(targets)


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
        td_lambda = config.get("TD_LAMBDA")
        if td_lambda is None:
            td_lambda = config.get("VALUE_LAMBDA", 0.0)

        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, rng, idx = runner_state

            # 1. COLLECT TRAJECTORIES
            def _env_step(env_scan_state, unused):
                train_state, env_state, last_obs, rng = env_scan_state

                rng, _rng = jax.random.split(rng)
                # Evaluates consensus scalar value (mean across heads) for actor & rollouts
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

            # 2. ACTOR ADVANTAGE ESTIMATION (GAE on rollout consensus values)
            gae_lambda = config.get("GAE_LAMBDA", 0.95)
            advantages, _ = helpers.calculate_gae(traj_batch, config["GAMMA"], gae_lambda)

            is_timeout = traj_batch.info.get("is_timeout", jnp.zeros_like(traj_batch.done, dtype=bool))
            true_terminal = traj_batch.done & ~is_timeout

            # 3. SPLIT OPTIMIZATION: CRITIC UPDATE PHASE (CLASSIC TD WITH LIVE DELTA)
            if td_lambda == 0.0:
                # --- PURE 1-STEP TD: Recomputes v(s') and delta per MINIBATCH ---
                def _critic_epoch_1step(critic_state, unused):
                    def _critic_minibatch(train_state, mb):
                        obs_mb, next_obs_mb, reward_mb, done_mb, is_timeout_mb = mb

                        def _c_loss(critic_params):
                            full_params = {"params": {**train_state.actor.params, **critic_params}}
                            v_s = network.apply(full_params, obs_mb, method=network.value)
                            v_next = network.apply(full_params, next_obs_mb, method=network.value)

                            if v_s.ndim == 1:
                                v_s = v_s[:, None]
                                v_next = v_next[:, None]

                            t_term = done_mb & ~is_timeout_mb
                            b_mask = (1.0 - t_term.astype(jnp.float32))[:, None]
                            r_exp = reward_mb[:, None]

                            # Live 1-step target updated with CURRENT parameters
                            target = jax.lax.stop_gradient(r_exp + config["GAMMA"] * b_mask * v_next)
                            delta = target - v_s

                            if critic_loss_type == "mse":
                                loss_per_sample = 0.5 * (v_s - target) ** 2
                            else:
                                loss_per_sample = helpers.huber_loss(v_s - target, delta=huber_delta)

                            loss_per_head = jnp.mean(loss_per_sample, axis=0)

                            if value_head_agg == "sum":
                                val_loss = jnp.sum(loss_per_head)
                            else:
                                val_loss = jnp.mean(loss_per_head)

                            scaled_loss = config.get("VF_COEF", 0.5) * val_loss
                            return scaled_loss, (val_loss, jnp.abs(delta).mean())

                        grad_fn = jax.value_and_grad(_c_loss, has_aux=True)
                        (_, (val_loss, delta_mag)), grads = grad_fn(train_state.critic.params)
                        train_state = train_state.apply_critic_gradients(grads=grads)
                        return train_state, (val_loss, delta_mag)

                    train_state, critic_batch, rng = critic_state
                    rng, _rng = jax.random.split(rng)
                    minibatches = helpers.shuffle_and_batch(_rng, critic_batch, config["NUM_MINIBATCHES"])
                    train_state, c_losses = jax.lax.scan(_critic_minibatch, train_state, minibatches)
                    return (train_state, critic_batch, rng), c_losses

                critic_batch = (
                    traj_batch.obs,
                    traj_batch.next_obs,
                    traj_batch.reward,
                    traj_batch.done,
                    is_timeout,
                )
                rng, _rng = jax.random.split(rng)
                c_init_state = (train_state, critic_batch, _rng)
                (train_state, _, rng), c_loss_epochs = jax.lax.scan(
                    _critic_epoch_1step, c_init_state, None, critic_epochs
                )
            else:
                # --- TD(lambda): Fitted Value Iteration (like PPO) vs Recomputing ---
                recompute_targets = config.get("RECOMPUTE_TARGETS_EACH_EPOCH", False)

                def _critic_minibatch_td_lambda(train_state, mb):
                    obs_mb, targets_mb = mb

                    def _c_loss(critic_params):
                        c_full = {"params": {**train_state.actor.params, **critic_params}}
                        v_s = network.apply(c_full, obs_mb, method=network.value)
                        if v_s.ndim == 1:
                            v_s = v_s[:, None]

                        delta = targets_mb - v_s

                        if critic_loss_type == "mse":
                            loss_per_sample = 0.5 * (v_s - targets_mb) ** 2
                        else:
                            loss_per_sample = helpers.huber_loss(v_s - targets_mb, delta=huber_delta)

                        loss_per_head = jnp.mean(loss_per_sample, axis=0)

                        if value_head_agg == "sum":
                            val_loss = jnp.sum(loss_per_head)
                        else:
                            val_loss = jnp.mean(loss_per_head)

                        scaled_loss = config.get("VF_COEF", 0.5) * val_loss
                        return scaled_loss, (val_loss, jnp.abs(delta).mean())

                    grad_fn = jax.value_and_grad(_c_loss, has_aux=True)
                    (_, (val_loss, delta_mag)), grads = grad_fn(train_state.critic.params)
                    train_state = train_state.apply_critic_gradients(grads=grads)
                    return train_state, (val_loss, delta_mag)

                if recompute_targets:
                    # Dynamically recomputes full-trajectory delta and targets per EPOCH
                    def _critic_epoch_td_lambda(critic_state, unused):
                        train_state, rng = critic_state

                        full_params = {"params": {**train_state.actor.params, **train_state.critic.params}}
                        v_traj = network.apply(full_params, traj_batch.obs, method=network.value)
                        v_next_traj = network.apply(full_params, traj_batch.next_obs, method=network.value)

                        if v_traj.ndim == 2:
                            v_traj = v_traj[..., None]
                            v_next_traj = v_next_traj[..., None]

                        r_exp = traj_batch.reward[..., None]
                        d_exp = traj_batch.done[..., None]
                        to_exp = is_timeout[..., None]

                        recomputed_targets = calculate_recomputed_td_lambda_targets(
                            v_traj, v_next_traj, r_exp, d_exp, to_exp, config["GAMMA"], td_lambda
                        )
                        critic_batch = (traj_batch.obs, recomputed_targets)

                        rng, _rng = jax.random.split(rng)
                        minibatches = helpers.shuffle_and_batch(_rng, critic_batch, config["NUM_MINIBATCHES"])
                        train_state, c_losses = jax.lax.scan(_critic_minibatch_td_lambda, train_state, minibatches)
                        return (train_state, rng), c_losses

                    rng, _rng = jax.random.split(rng)
                    c_init_state = (train_state, _rng)
                    (train_state, rng), c_loss_epochs = jax.lax.scan(
                        _critic_epoch_td_lambda, c_init_state, None, critic_epochs
                    )
                else:
                    # Static targets evaluated once at rollout time (standard PPO fitted value iteration, no GAE recomputation)
                    full_params = {"params": {**train_state.actor.params, **train_state.critic.params}}
                    v_traj = network.apply(full_params, traj_batch.obs, method=network.value)
                    v_next_traj = network.apply(full_params, traj_batch.next_obs, method=network.value)

                    if v_traj.ndim == 2:
                        v_traj = v_traj[..., None]
                        v_next_traj = v_next_traj[..., None]

                    r_exp = traj_batch.reward[..., None]
                    d_exp = traj_batch.done[..., None]
                    to_exp = is_timeout[..., None]

                    static_targets = calculate_recomputed_td_lambda_targets(
                        v_traj, v_next_traj, r_exp, d_exp, to_exp, config["GAMMA"], td_lambda
                    )
                    critic_batch = (traj_batch.obs, static_targets)

                    def _critic_epoch_static(critic_state, unused):
                        train_state, critic_batch, rng = critic_state
                        rng, _rng = jax.random.split(rng)
                        minibatches = helpers.shuffle_and_batch(_rng, critic_batch, config["NUM_MINIBATCHES"])
                        train_state, c_losses = jax.lax.scan(_critic_minibatch_td_lambda, train_state, minibatches)
                        return (train_state, critic_batch, rng), c_losses

                    rng, _rng = jax.random.split(rng)
                    c_init_state = (train_state, critic_batch, _rng)
                    (train_state, _, rng), c_loss_epochs = jax.lax.scan(
                        _critic_epoch_static, c_init_state, None, critic_epochs
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
            delta_mag = c_loss_epochs[1].mean()
            act_loss = a_loss_epochs[0].mean()
            ent = a_loss_epochs[1].mean()

            loss_info = {
                "total_loss": val_loss + act_loss,
                "value_loss": val_loss,
                "delta_magnitude": delta_mag,
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

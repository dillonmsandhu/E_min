# helpers.py
# Technical helpers for RL training loops, GAE computation, PPO losses, and environment setup.
import gymnax
from gymnax.environments import spaces
import jax
import jax.numpy as jnp

from envs.log_wrapper import LogWrapper
from envs.wrappers import (
    NormalizeObservationWrapper,
    AddChannelWrapper,
    FlattenObservationWrapper,
    ClipAction,
    TerminalInfoWrapper,
    MountainCarNormalizeWrapper,
    MountainCarSparseRewardWrapper,
)


def make_env(config):
    env_name = config["ENV_NAME"]
    env, env_params = gymnax.make(env_name)

    if env_name == "MountainCar-v0":
        env = MountainCarNormalizeWrapper(env)
        env = MountainCarSparseRewardWrapper(env)
        config["NETWORK_TYPE"] = "mlp"

    env = TerminalInfoWrapper(env)
    env = LogWrapper(env, gamma=config.get("GAMMA", 0.99))

    if isinstance(env.action_space(env_params), spaces.Box):
        action_space = env.action_space(env_params)
        env = ClipAction(env, low=action_space.low, high=action_space.high)

    obs_shape = env.observation_space(env_params).shape
    if len(obs_shape) == 1:
        config["NETWORK_TYPE"] = "mlp"

    if config.get("NETWORK_TYPE", "mlp") == "mlp":
        if len(obs_shape) > 1:
            env = FlattenObservationWrapper(env)
    elif config.get("NETWORK_TYPE", "mlp") == "cnn":
        if len(obs_shape) == 2:
            env = AddChannelWrapper(env)
    if config.get("NORMALIZE_OBS", False):
        env = NormalizeObservationWrapper(env)

    return env, env_params


def post_process_advantage(advantages, config, weights=None):
    """
    Standardizes and clips advantages for PPO policy optimization.
    """
    std_floor = config.get("ADV_STD_FLOOR", 0.1)
    adv_clip = config.get("ADV_CLIP", 3.0)

    if weights is not None:
        if weights.ndim == 1 and advantages.ndim == 2:
            weights = weights[:, None]
        w = weights / jnp.sum(weights)
        mean_adv = jnp.sum(w * advantages)
        var_adv = jnp.sum(w * (advantages - mean_adv) ** 2)
        std_adv = jnp.sqrt(var_adv + 1e-8)
    else:
        mean_adv = jnp.mean(advantages)
        var_adv = jnp.var(advantages)
        std_adv = jnp.sqrt(var_adv + 1e-8)

    if std_floor is not None and std_floor > 0.0:
        denom = jnp.maximum(std_adv, std_floor)
    else:
        denom = std_adv + 1e-8

    adv_norm = (advantages - mean_adv) / denom

    if adv_clip is not None and adv_clip > 0.0:
        adv_norm = jnp.clip(adv_norm, -adv_clip, adv_clip)

    return jax.lax.stop_gradient(adv_norm)


def pi_loss_fn(params, network, traj_batch, gae, config):
    pi = network.apply(params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae = post_process_advantage(gae, config)
    loss_actor1 = ratio * gae
    loss_actor2 = (
        jnp.clip(
            ratio,
            1.0 - config["CLIP_EPS"],
            1.0 + config["CLIP_EPS"],
        )
        * gae
    )
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()
    entropy = pi.entropy().mean()
    return loss_actor, entropy


def ppo_clipped_v_loss(traj_batch, value_pred, targets, config):
    e = config["VF_CLIP"]
    value_pred_clipped = traj_batch.value + (
        value_pred - traj_batch.value
    ).clip(-e, e)
    value_losses = jnp.square(value_pred - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    return 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()


def v_loss_fn(params, network, traj_batch, gae, targets, config):
    value_pred = network.apply(params, traj_batch.obs, method=network.value)
    return ppo_clipped_v_loss(traj_batch, value_pred, targets, config)


def _loss_fn(params, network, traj_batch, gae, targets, config):
    value_loss = v_loss_fn(params, network, traj_batch, gae, targets, config)
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, gae, config)

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )
    return total_loss, (value_loss, loss_actor, entropy)


def e_critic_loss(v_i, targets_i, v_j, targets_j, done, gamma):
    """
    Computes sampled E-loss (magnitude anchor + Dirichlet/Laplacian smoothness).
    - For ongoing transitions and timeouts: smooths e_i against e_j.
    - For true terminal transitions (done=True): absorbing state error is 0,
      so (e_i - e_j)^2 = (e_i - 0)^2 = (r_T - v_T)^2, smoothing v_T directly to reward.
    """
    e_i = targets_i - v_i
    e_j = jnp.where(done, 0.0, targets_j - v_j)

    magnitude_loss = (1.0 - gamma) * jnp.mean(e_i ** 2)
    laplacian_loss = 0.5 * gamma * jnp.mean((e_i - e_j) ** 2)

    value_loss = magnitude_loss + laplacian_loss
    return value_loss, magnitude_loss, laplacian_loss


def e_loss_fn(
    params,
    network,
    obs,
    action,
    log_prob_old,
    next_obs,
    done,
    next_target,
    advantages,
    targets,
    config,
):
    """
    Combined loss for PPO with Sampled E critic.
    """
    # 1. Actor Loss (PPO clipped surrogate)
    pi = network.apply(params, obs, method=network.policy)
    log_prob = pi.log_prob(action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - log_prob_old)

    adv_norm = post_process_advantage(advantages, config)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    # 2. Critic Loss (Sampled E-loss)
    gamma = config.get("GAMMA", 0.99)
    v_i = network.apply(params, obs, method=network.value)
    v_j = network.apply(params, next_obs, method=network.value)
    # Terminal absorbing state has value 0
    v_j = jnp.where(done, 0.0, v_j)

    value_loss, magnitude_loss, laplacian_loss = e_critic_loss(
        v_i, targets, v_j, next_target, done, gamma
    )

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * actor_loss
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )
    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "laplacian_loss": laplacian_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
    }
    return total_loss, losses


def shuffle_and_batch(rng, transitions, n_minibatches):
    def preprocess_transition(x, rng):
        x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs (batch_size), ...
        x = jax.random.permutation(rng, x)  # shuffle the transitions
        x = x.reshape(n_minibatches, -1, *x.shape[1:])  # num_mini_updates, batch_size/num_mini_updates, ...
        return x

    minibatches = jax.tree.map(
        lambda x: preprocess_transition(x, rng), transitions
    )
    return minibatches


def calculate_gae(traj_batch, γ, λ):
    def _get_advantages(gae, transition):
        done = transition.done
        is_timeout = transition.info["is_timeout"]

        # MASK 1: Value Bootstrapping
        true_terminal = done & ~is_timeout
        bootstrap_mask = 1.0 - true_terminal

        # MASK 2: GAE Accumulation (Trajectory Boundary)
        # Sever the GAE chain if the environment reset for ANY reason (terminal or timeout).
        boundary_mask = 1.0 - done

        # 1. Compute TD Error (Safely bootstraps through timeouts)
        delta = (
            transition.reward
            + γ * transition.next_value * bootstrap_mask
            - transition.value
        )

        # 2. Accumulate GAE (Safely breaks at episode resets)
        gae = delta + (γ * λ * boundary_mask * gae)

        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    _, advantages = jax.lax.scan(
        _get_advantages, initial_accs, traj_batch, reverse=True, unroll=16
    )

    return (advantages, advantages + traj_batch.value)


def calculate_e_lambda_targets(
    traj_batch,
    gamma: float,
    lmbda: float,
    return_lambda: float = 1.0,
):
    """
    Computes scalar value regression targets for the symmetrized E(lambda) objective
    using forward and backward error traces:
        y_t = G_t - sg[ 0.5 * gamma * (1 - lambda) * (e_{>t} + e_{<t}) ]

    Boundary and truncation handling:
    - Forward trace (downstream errors): accumulates future errors backwards in time.
      Zeroed on episode boundaries (both true terminals and timeouts/truncations).
    - Backward trace (upstream errors): accumulates past errors forwards in time.
      Wiped to zero at episode starts (when the preceding transition was done).

    Args:
        traj_batch: NamedTuple with fields (done, value, next_value, reward, info, ...)
        gamma: discount factor (e.g. 0.99)
        lmbda: E(lambda) decay parameter in [0, 1]
        return_lambda: lambda for computing baseline return G_t (defaults to 1.0 for Monte Carlo)

    Returns:
        targets: (T, B) scalar value regression targets
        diagnostics: dict with returns, errors, forward_traces, backward_traces, correction
    """
    # 1. Baseline returns G_t and errors e_t = G_t - v_t
    _, returns = calculate_gae(traj_batch, gamma, return_lambda)
    errors = returns - traj_batch.value
    gl = gamma * lmbda

    done_f = traj_batch.done.astype(jnp.float32)

    # 2. Forward Trace: Accumulate future errors backwards in time
    # e_next: step t receives e_{t+1} as immediate lookahead
    e_next = jnp.roll(errors, shift=-1, axis=0).at[-1].set(0.0)

    def _backward_pass(forward_trace, transition):
        done_t, e_nxt = transition
        trace_t = (1.0 - done_t) * (e_nxt + gl * forward_trace)
        return trace_t, trace_t

    init_forward = jnp.zeros_like(errors[0])
    _, forward_traces = jax.lax.scan(
        _backward_pass,
        init_forward,
        (done_f, e_next),
        reverse=True,
    )

    # 3. Backward Trace: Accumulate past errors forwards in time
    # prev_done: step 0 has prev_done = 1.0 (no predecessor in this rollout chunk)
    prev_done = jnp.roll(done_f, shift=1, axis=0).at[0].set(1.0)
    # e_prev: step 0 has e_prev = 0.0; step t receives errors[t-1]
    e_prev = jnp.roll(errors, shift=1, axis=0).at[0].set(0.0)

    def _forward_pass(backward_trace, transition):
        p_done, e_prv = transition
        trace_t = (1.0 - p_done) * (e_prv + gl * backward_trace)
        return trace_t, trace_t

    init_backward = jnp.zeros_like(errors[0])
    _, backward_traces = jax.lax.scan(
        _forward_pass,
        init_backward,
        (prev_done, e_prev),
        reverse=False,
    )

    # 4. Construct Symmetrized E(lambda) scalar regression targets
    coeff = 0.5 * gamma * (1.0 - lmbda)
    smoothing_correction = coeff * (forward_traces + backward_traces)
    targets = returns - jax.lax.stop_gradient(smoothing_correction)

    diagnostics = {
        "returns": returns,
        "errors": errors,
        "forward_traces": forward_traces,
        "backward_traces": backward_traces,
        "correction": smoothing_correction,
    }
    return targets, diagnostics


def e_lambda_differentiable_critic_loss(
    values,
    targets,
    dones,
    gamma: float,
    lmbda: float,
):
    """
    Computes differentiable E(lambda) loss using backward moment traces (Method 2).
    Backpropagates through both values v(s_t) and future values v(s_{t+k+1}).

    Terminal boundary handling:
    - For ongoing transitions (done_t = 0): accumulates lookahead e_{t+1} and discounted future moments.
    - For terminal transitions (done_t = 1): transitions to absorbing state with error e_infty = 0,
      yielding dirichlet_t = (e_t - 0)^2 = e_t^2, exactly matching e_loss_fn.
    - Future accumulation beyond terminal transitions is cleanly severed (w_to_prev = (1 - done_t) * w_t).

    Args:
        values: (T, B) predicted values v(s_t) from network.apply(params, obs)
        targets: (T, B) baseline Monte Carlo returns G_t
        dones: (T, B) bool array of episode termination/truncation
        gamma: discount factor
        lmbda: E(lambda) decay parameter in [0, 1]

    Returns:
        value_loss, magnitude_loss, dirichlet_loss
    """
    errors = targets - values
    gl = gamma * lmbda
    done_f = dones.astype(jnp.float32)

    # e_next is the error at s_{t+1}. For the last step T-1, e_T = 0.
    e_next = jnp.roll(errors, shift=-1, axis=0).at[-1].set(0.0)

    def _moment_step(traces, transition):
        w0_future, w1_future, w2_future = traces
        done_t, e_curr, e_nxt = transition

        valid_next = 1.0 - done_t

        # 1. Moments at step t (lookahead e_{t+1} and discounted future)
        # If done_t = 1: transitions to absorbing state e=0; w0=1, w1=0, w2=0
        w0_t = 1.0 + gl * valid_next * w0_future
        w1_t = valid_next * (e_nxt + gl * w1_future)
        w2_t = valid_next * (e_nxt ** 2 + gl * w2_future)

        # 2. Dirichlet quadratic expansion: w0 * e_t^2 - 2 * w1 * e_t + w2
        # For done_t = 1: dirichlet_t = 1.0 * e_curr^2 = (e_curr - 0)^2, matching e_loss_fn!
        dirichlet_t = w0_t * (e_curr ** 2) - 2.0 * w1_t * e_curr + w2_t

        # 3. Pass traces backward to step t-1 (severed if done_t == 1)
        w0_to_prev = valid_next * w0_t
        w1_to_prev = valid_next * w1_t
        w2_to_prev = valid_next * w2_t

        return (w0_to_prev, w1_to_prev, w2_to_prev), dirichlet_t

    init_traces = (
        jnp.zeros_like(errors[0]),
        jnp.zeros_like(errors[0]),
        jnp.zeros_like(errors[0]),
    )
    _, dirichlet_terms = jax.lax.scan(
        _moment_step,
        init_traces,
        (done_f, errors, e_next),
        reverse=True,
    )

    magnitude_weight = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * gamma * (1.0 - lmbda)

    magnitude_loss = magnitude_weight * jnp.mean(errors ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(dirichlet_terms)
    value_loss = magnitude_loss + dirichlet_loss

    return value_loss, magnitude_loss, dirichlet_loss


def shuffle_and_batch_envs(rng, batch, n_minibatches):
    """
    Shuffles environments (columns) and splits into minibatches,
    preserving full trajectory sequences along the time axis (axis 0).
    """
    sample_leaf = jax.tree.leaves(batch)[0]
    num_envs = sample_leaf.shape[1]
    n_minibatches = max(1, min(n_minibatches, num_envs))
    env_per_mb = num_envs // n_minibatches

    perm = jax.random.permutation(rng, num_envs)

    def _split_leaf(x):
        x_shuffled = jnp.take(x, perm, axis=1)
        x_trimmed = x_shuffled[:, : env_per_mb * n_minibatches]
        reshaped = x_trimmed.reshape(x.shape[0], n_minibatches, env_per_mb, *x.shape[2:])
        return jnp.swapaxes(reshaped, 0, 1)

    return jax.tree.map(_split_leaf, batch)


def e_lambda_differentiable_loss_fn(
    params,
    network,
    traj_batch,
    advantages,
    returns,
    config,
):
    """
    Combined PPO loss with Method 2 differentiable E(lambda) critic loss.
    """
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, advantages, config)

    values = network.apply(params, traj_batch.obs, method=network.value)
    gamma = config.get("GAMMA", 0.99)
    e_lambda = config.get("E_LAMBDA", config.get("VALUE_LAMBDA", 0.8))

    value_loss, magnitude_loss, dirichlet_loss = e_lambda_differentiable_critic_loss(
        values=values,
        targets=returns,
        dones=traj_batch.done,
        gamma=gamma,
        lmbda=e_lambda,
    )

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses


def e_lambda_geometric_critic_loss(
    values,
    targets,
    dones,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
):
    """
    Computes sampled E(lambda) critic loss by sampling lookahead horizon skips
    K ~ Geometric(1 - gamma * lambda) directly from the compound transition matrix
    P_lambda = (1 - gamma * lambda) * sum_{k=0}^infty (gamma * lambda)^k P^{k+1} (Method 3).

    Terminal and episode boundary handling:
    - For ongoing transitions (done_t = 0):
      Samples random lookahead jump K >= 0 from Geometric(1 - gamma * lambda).
      Target index is t' = t + K + 1.
      If jump remains within the rollout chunk (t' < T) and within the same episode
      (no intermediate done between t and t'-1), the Dirichlet term is (e_t - e_{t'})^2.
      Otherwise, the jump is masked out (diff = 0.0).
    - For terminal transitions (done_t = 1):
      Absorbing terminal state error is e_infty = 0, so (e_t - 0)^2 = e_t^2,
      exactly matching e_critic_loss and e_lambda_differentiable_critic_loss.
    - Weighting:
      tilde_gamma = gamma * (1 - lambda) / (1 - gamma * lambda)
      magnitude_loss = (1 - tilde_gamma) * mean(e_t^2)
      dirichlet_loss = 0.5 * tilde_gamma * mean(diff_sq)

    Args:
        values: (T, B) predicted values v(s_t) from network
        targets: (T, B) baseline Monte Carlo returns G_t
        dones: (T, B) bool array of episode termination/truncation
        gamma: discount factor
        lmbda: E(lambda) decay parameter in [0, 1]
        rng: PRNG key for sampling geometric lookahead jumps

    Returns:
        value_loss, magnitude_loss, dirichlet_loss
    """
    T, B = targets.shape[:2]
    errors = targets - values
    gl = gamma * lmbda

    # 1. Sample jump lengths K ~ Geometric(1 - gl) with K >= 0
    # For gl < 1e-6 (e.g. lambda = 0), K = 0 deterministically
    u = jax.random.uniform(rng, shape=(T, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    jumps = jnp.maximum(jumps, 0)

    # 2. Target lookahead index and episode boundary masking
    t_arr = jnp.arange(T)[:, None]
    target_idx = t_arr + jumps + 1
    within_chunk = target_idx < T
    t_clamped = jnp.minimum(target_idx, T - 1)

    # Predecessor index of target_idx is t_arr + jumps.
    # An episode reset occurred between t and target_idx iff dones occurred in [t+1, target_idx-1]
    t_prev = jnp.minimum(t_arr + jumps, T - 1)
    done_cumsum = jnp.cumsum(dones.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    valid_jump = within_chunk & (~has_reset_between)
    e_jump = jnp.take_along_axis(errors, t_clamped, axis=0)

    # 3. Dirichlet term:
    # - Terminal states (done == True): absorbs to e=0, so (e_t - 0)^2 = e_t^2 (always valid)
    # - Ongoing states (done == False): valid jumps evaluate (e_t - e_{t'})^2, invalid jumps 0.0
    diff_sq = jnp.where(
        dones,
        errors ** 2,
        jnp.where(valid_jump, (errors - e_jump) ** 2, 0.0),
    )

    # 4. Weighting
    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    magnitude_loss = magnitude_weight * jnp.mean(errors ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    value_loss = magnitude_loss + dirichlet_loss

    return value_loss, magnitude_loss, dirichlet_loss


def e_lambda_geometric_loss_fn(
    params,
    network,
    traj_batch,
    advantages,
    returns,
    config,
    rng,
):
    """
    Combined PPO loss with Method 3 sampled geometric E(lambda) critic loss.
    """
    loss_actor, entropy = pi_loss_fn(params, network, traj_batch, advantages, config)

    values = network.apply(params, traj_batch.obs, method=network.value)
    gamma = config.get("GAMMA", 0.99)
    e_lambda = config.get("E_LAMBDA", config.get("VALUE_LAMBDA", 0.8))

    value_loss, magnitude_loss, dirichlet_loss = e_lambda_geometric_critic_loss(
        values=values,
        targets=returns,
        dones=traj_batch.done,
        gamma=gamma,
        lmbda=e_lambda,
        rng=rng,
    )

    total_loss = (
        config.get("POLICY_COEFF", 1.0) * loss_actor
        + config.get("VF_COEF", 0.5) * value_loss
        - config.get("ENT_COEF", 0.01) * entropy
    )

    losses = {
        "total_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "actor_loss": loss_actor,
        "entropy": entropy,
    }
    return total_loss, losses




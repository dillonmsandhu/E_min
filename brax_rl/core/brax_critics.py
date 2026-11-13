import jax
import jax.numpy as jnp
from brax_rl.core.brax_utils import scale_rms


def calculate_gae(traj_batch, next_value, gamma: float = 0.99, gae_lambda: float = 0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and value targets.
    Bootstraps through timeouts using true continuation next_value,
    and severs GAE accumulation at episode boundaries (done=True).
    Supports either:
      - 2D next_value of shape (T, B) with exact continuation values (including timeouts)
      - 1D last_val of shape (B,) for backward compatibility
    """
    if next_value.ndim == 1:
        regular_next = jnp.concatenate([traj_batch.value[1:], next_value[None]], axis=0)
        next_val_arr = getattr(traj_batch, "next_value", None)
        if next_val_arr is None:
            next_val_arr = regular_next
    else:
        next_val_arr = next_value

    def _get_advantages(gae, data):
        done, is_timeout, reward, value, next_val = data
        true_terminal = done & ~is_timeout

        bootstrap_mask = 1.0 - true_terminal
        boundary_mask = 1.0 - done

        delta = reward + gamma * next_val * bootstrap_mask - value
        gae = delta + (gamma * gae_lambda * boundary_mask * gae)
        return gae, gae

    initial_accs = jnp.zeros_like(traj_batch.value[0])
    scan_data = (
        traj_batch.done,
        traj_batch.is_timeout,
        traj_batch.reward,
        traj_batch.value,
        next_val_arr,
    )
    _, advantages = jax.lax.scan(
        _get_advantages,
        initial_accs,
        scan_data,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    return advantages, targets


def calculate_targets(traj_batch, next_value, gamma: float = 0.99, target_lambda: float = 1.0):
    """
    Computes value regression targets using generalized lambda-returns.
    target_lambda=1.0 computes empirical Monte Carlo rollout returns.
    """
    _, targets = calculate_gae(traj_batch, next_value, gamma=gamma, gae_lambda=target_lambda)
    return targets

# Aliases
get_return = calculate_targets
calculate_return = calculate_targets


def compute_aux_head_loss(v_aux, aux_targets, aux_coef: float = 1.0):
    """Pure MSE loss training the auxiliary value head toward its (stop-gradient) targets."""
    err = jax.lax.stop_gradient(aux_targets) - v_aux
    return aux_coef * jnp.mean(0.5 * jnp.square(err))


def e_aux_backward_surrogate(e_phi_src, e_phi_dst, v_dst):
    """
    Backward-branch surrogate for the direct error head e_phi(s) ~ V(s) - v_theta(s).
    With d = sg(e_phi_src - e_phi_dst):
        value    = d^2
        gradient = 2 * d * grad v_theta(s_dst)
    """
    sg = jax.lax.stop_gradient
    d = sg(e_phi_src - e_phi_dst)
    return (sg(d - v_dst) + v_dst) ** 2


def get_critic_target_lambda(config: dict) -> float:
    """
    Returns the appropriate lambda to compute value targets based on the critic loss type:
    - E-minimization (e_0, e_lambda): RETURN_LAMBDA (default 1.0) for empirical returns.
    - Fitted PPO / SARSA(lambda): VALUE_LAMBDA (default 0.95).
    - TD(0) / SARSA(0): TD_LAMBDA (default 0.0).
    """
    critic_type = config.get("CRITIC_TYPE", "fitted").lower()
    if critic_type in [
        "e_0", "e_0_q", "e_0_basic", "e_basic", "e_0_biased", "e_biased", "e0", "e",
        "sampled_e", "surgical_v_aux", "surgical",
        "e_lambda", "e_lambda_q", "elambda", "e_geometric",
        "e_inner", "e_bilinear", "e_double", "e_min",
    ]:
        return config["RETURN_LAMBDA"]
    elif critic_type in ["td_0", "td", "td0", "sarsa_0", "msbe", "double_msbe", "mspbe", "double_mspbe"]:
        return config["TD_LAMBDA"]
    elif critic_type in ["mc", "sampled_mc"]:
        return 1.0
    elif critic_type in ["fitted", "ppo", "sarsa", "sarsa_lambda", "td_lambda"]:
        return config.get("VALUE_LAMBDA", config["GAE_LAMBDA"])
    else:
        raise ValueError(f"Unrecognized CRITIC_TYPE '{critic_type}'.")


def ppo_value_loss(value, targets, value_old=None, vf_clip=None):
    """
    Standard PPO value loss with optional value clipping (Fitted Value Iteration MSE).
    Strictly MSE loss (no Huber).
    """
    value_losses = 0.5 * jnp.square(value - targets)
    if vf_clip is not None and value_old is not None:
        value_pred_clipped = value_old + jnp.clip(value - value_old, -vf_clip, vf_clip)
        value_losses_clipped = 0.5 * jnp.square(value_pred_clipped - targets)
        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = value_losses.mean()
    return value_loss


def td_0_value_loss(
    value,
    next_value,
    reward,
    done,
    gamma: float = 0.99,
    vf_clip=None,
    value_old=None,
    value_head_agg: str = "mean",
    truncation=None,
):
    """
    Computes online 1-step TD(0) MSE loss using live next_value:
        target = stop_gradient(reward + gamma * (1 - true_terminal) * next_value)
        loss = 0.5 * (value - target)^2
    At true terminals, value absorbs to 0. At truncations, it safely bootstraps through next_value.
    Strictly MSE loss (no Huber).
    """
    if truncation is not None:
        true_terminal = (done > 0.5) & ~(truncation > 0.5)
    else:
        true_terminal = done > 0.5

    if value.ndim > 1:
        r = reward[..., None]
        d = true_terminal[..., None].astype(jnp.float32)
    else:
        r = reward
        d = true_terminal.astype(jnp.float32)

    target = jax.lax.stop_gradient(r + gamma * (1.0 - d) * next_value)
    delta = target - value

    losses = 0.5 * jnp.square(value - target)

    if vf_clip is not None and value_old is not None:
        value_pred_clipped = value_old + jnp.clip(value - value_old, -vf_clip, vf_clip)
        losses_clipped = 0.5 * jnp.square(value_pred_clipped - target)
        losses = jnp.maximum(losses, losses_clipped)

    if value.ndim > 1:
        loss_per_head = jnp.mean(losses, axis=0)
        val_loss = jnp.sum(loss_per_head) if value_head_agg == "sum" else jnp.mean(loss_per_head)
    else:
        val_loss = jnp.mean(losses)

    return val_loss, delta


# ==============================================================================
# Geometric Lookahead Sampling and Transitions for E-Loss
# ==============================================================================

def prepare_geometric_jump_transitions(
    traj_batch,
    targets,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    last_val: jax.Array = None,
):
    """
    Samples geometric lookahead jumps K ~ Geometric(1 - gamma * lambda) on the rollout buffer
    and extracts jump observation targets, continuation states, and validity masks
    for transition-level minibatching.

    For lambda == 0 (E0), K = 0 deterministically (adjacent state pairs).
    Properly handles Brax true terminals (absorbing) and timeouts (bootstrapped).

    Returns:
        obs_jump: (T, B, *obs_shape) jump target observations
        targets_jump: (T, B) return targets for jump states
        valid_jump: (T, B) bool mask indicating jump does not cross episode reset
        is_absorbing: (T, B) bool mask indicating true terminal absorbing state
    """
    T, B = targets.shape[:2]
    gl = gamma * lmbda

    # 1. Sample lookahead horizon jump K >= 0 from Geometric(1 - gl)
    # For gl < 1e-6 (e.g. lambda = 0), K = 0 deterministically
    u = jax.random.uniform(rng, shape=(T, B))
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    jumps = jnp.maximum(jumps, 0)

    # 2. Clamped target index (T represents continuation state s_T)
    t_arr = jnp.arange(T)[:, None]
    target_idx = t_arr + jumps + 1
    t_clamped = jnp.minimum(target_idx, T)

    # 3. Detect episode resets between t and target
    t_prev = jnp.minimum(t_arr + jumps, T - 1)
    done_cumsum = jnp.cumsum(traj_batch.done.astype(jnp.int32), axis=0)
    has_reset_between = (
        jnp.take_along_axis(done_cumsum, t_prev, axis=0) - done_cumsum
    ) > 0

    # 4. Terminals and Truncations
    if isinstance(traj_batch.info, dict):
        truncation = traj_batch.info.get("truncation", traj_batch.info.get("is_timeout", None))
        if truncation is not None:
            truncation = truncation > 0.5
        else:
            truncation = jnp.zeros_like(traj_batch.done, dtype=bool)
    else:
        truncation = jnp.zeros_like(traj_batch.done, dtype=bool)

    true_terminal = (traj_batch.done > 0.5) & ~truncation
    is_absorbing = true_terminal

    # 5. Extended arrays up to index T (continuation boundary)
    val_T = last_val if last_val is not None else traj_batch.value[-1]
    next_target_T = jnp.where(true_terminal[-1], 0.0, val_T)
    obs_ext = jnp.concatenate([traj_batch.obs, traj_batch.next_obs[-1][None]], axis=0)
    targets_ext = jnp.concatenate([targets, next_target_T[None]], axis=0)

    # 6. Gather jump observations and targets
    idx_expanded = t_clamped.reshape(t_clamped.shape + (1,) * (traj_batch.obs.ndim - 2))
    idx_broadcast = jnp.broadcast_to(idx_expanded, (T, B) + traj_batch.obs.shape[2:])
    obs_jump_gathered = jnp.take_along_axis(obs_ext, idx_broadcast, axis=0)
    targets_jump_gathered = jnp.take_along_axis(targets_ext, t_clamped, axis=0)

    # For truncations with K==0, use real_next_obs
    truncation_mask_obs = truncation.reshape(truncation.shape + (1,) * (traj_batch.obs.ndim - 2))
    obs_jump = jnp.where(truncation_mask_obs, traj_batch.next_obs, obs_jump_gathered)
    targets_jump = targets_jump_gathered

    # 7. Valid jump mask:
    # - true terminal: absorbing (handled via is_absorbing)
    # - truncation: valid iff jumps == 0
    # - ongoing: valid iff no episode reset occurred between t and target
    valid_jump = jnp.where(
        true_terminal,
        False,
        jnp.where(truncation, jumps == 0, ~has_reset_between),
    )

    return obs_jump, targets_jump, valid_jump, is_absorbing


def prepare_critic_transitions(
    critic_type: str,
    traj_batch,
    targets: jax.Array,
    last_val: jax.Array,
    gamma: float,
    e_lambda: float,
    rng: jax.Array,
):
    """
    Attaches jump transition data into traj_batch for E(lambda) critics before PyTree shuffling.
    For fitted, TD, and E(0) critics, returns traj_batch unmodified (targets and masks are first-class fields).
    """
    key = critic_type.lower()
    if key in ["e_lambda", "elambda", "e_geometric"]:
        obs_jump, targets_jump, valid_jump, is_absorbing = prepare_geometric_jump_transitions(
            traj_batch=traj_batch,
            targets=targets,
            gamma=gamma,
            lmbda=e_lambda,
            rng=rng,
            last_val=last_val,
        )
        new_info = {
            **traj_batch.info,
            "obs_jump": obs_jump,
            "targets_jump": targets_jump,
            "valid_jump": valid_jump,
            "is_absorbing": is_absorbing,
        }
        return traj_batch._replace(info=new_info)
    else:
        return traj_batch


def e_lambda_geometric_critic_loss(
    v_i,
    targets_i,
    v_jump,
    targets_jump,
    valid_jump,
    is_absorbing,
    gamma: float,
    lmbda: float,
    value_head_agg: str = "mean",
):
    """
    Computes transition-level sampled E(lambda) critic loss with geometric jump pairs.
    Strictly MSE loss (no Huber).
        e_i = targets_i - v_i
        e_jump = targets_jump - v_jump
        L_mag = (1 - gamma) / (1 - gamma * lambda) * E[e_i^2]
        L_dir = 0.5 * gamma * (1 - lambda) / (1 - gamma * lambda) * E[diff_sq]
    """
    if v_i.ndim > targets_i.ndim:
        targets_i = targets_i[..., None]
        targets_jump = targets_jump[..., None]
        is_absorbing_m = is_absorbing[..., None]
        valid_jump_m = valid_jump[..., None]
    else:
        is_absorbing_m = is_absorbing
        valid_jump_m = valid_jump

    e_i = targets_i - v_i
    e_jump = targets_jump - v_jump

    diff_sq = jnp.where(
        is_absorbing_m,
        e_i ** 2,
        jnp.where(valid_jump_m, (e_i - e_jump) ** 2, 0.0),
    )

    gl = gamma * lmbda
    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    if v_i.ndim > 1:
        mag_per_head = magnitude_weight * jnp.mean(e_i ** 2, axis=0)
        dir_per_head = dirichlet_weight * jnp.mean(diff_sq, axis=0)
        loss_per_head = mag_per_head + dir_per_head
        if value_head_agg == "sum":
            value_loss = jnp.sum(loss_per_head)
            magnitude_loss = jnp.sum(mag_per_head)
            dirichlet_loss = jnp.sum(dir_per_head)
        else:
            value_loss = jnp.mean(loss_per_head)
            magnitude_loss = jnp.mean(mag_per_head)
            dirichlet_loss = jnp.mean(dir_per_head)
    else:
        magnitude_loss = magnitude_weight * jnp.mean(e_i ** 2)
        dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
        value_loss = magnitude_loss + dirichlet_loss

    return value_loss, magnitude_loss, dirichlet_loss


# ==============================================================================
# Unified Critic Loss Functions (Common Signature)
# Signature: (params, network, traj_batch, gae, targets, config, ent_coef=None)
# ==============================================================================

def ppo_loss_fn(params, network, traj_batch, gae, targets, config, ent_coef=None):
    """
    Combined Actor-Critic loss function for standard Brax PPO (Fitted Value Iteration MSE).
    """
    if ent_coef is None:
        ent_coef = config.get("ENT_COEF", 0.001)

    # 1. Forward pass
    pi, value = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)

    # 2. Value loss (fitted MSE regression to GAE targets)
    vf_clip = config.get("VF_CLIP", config.get("CLIP_EPS", None))
    value_loss = ppo_value_loss(
        value=value,
        targets=targets,
        value_old=traj_batch.value,
        vf_clip=vf_clip,
    )

    # 3. Actor loss (PPO clipped surrogate with RMS-scaled advantage)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae_norm = scale_rms(gae)

    clip_eps = config.get("CLIP_EPS", 0.2)
    surr1 = ratio * gae_norm
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    entropy = pi.entropy().mean()

    vf_coef = config.get("NETWORK_CONFIG", {}).get("VF_COEF", config.get("VF_COEF", 0.5))
    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    v_mean = value.mean(-1) if value.ndim > targets.ndim else value
    delta_mag = jnp.abs(targets - v_mean).mean()
    aux_losses = {
        "val_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "delta_mag": delta_mag,
    }
    return total_loss, aux_losses


def td_0_loss_fn(params, network, traj_batch, gae, targets, config, ent_coef=None):
    """
    Combined Actor-Critic loss function for Brax TD(0) with online live delta updates (MSE).
    Accepts unified signature (targets is unused by pure 1-step TD(0)).
    """
    if ent_coef is None:
        ent_coef = config.get("ENT_COEF", 0.001)

    # 1. Forward pass on current observations
    pi, value = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)

    # 2. Forward pass on next observations for live bootstrap target
    next_value = network.apply(params, traj_batch.next_obs, method=network.value)

    # 3. Compute live 1-step TD(0) MSE value loss
    vf_clip = config.get("VF_CLIP", config.get("CLIP_EPS", None))
    gamma = config.get("GAMMA", 0.99)
    value_head_agg = config.get("VALUE_HEAD_AGG", "mean")
    truncation = (
        traj_batch.info.get("truncation", traj_batch.info.get("is_timeout", None))
        if isinstance(traj_batch.info, dict) else None
    )

    value_loss, delta = td_0_value_loss(
        value=value,
        next_value=next_value,
        reward=traj_batch.reward,
        done=traj_batch.done,
        gamma=gamma,
        vf_clip=vf_clip,
        value_old=traj_batch.value,
        value_head_agg=value_head_agg,
        truncation=truncation,
    )

    # 4. Actor loss (PPO clipped surrogate with RMS-scaled advantage)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae_norm = scale_rms(gae)

    clip_eps = config.get("CLIP_EPS", 0.2)
    surr1 = ratio * gae_norm
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    entropy = pi.entropy().mean()

    vf_coef = config.get("NETWORK_CONFIG", {}).get("VF_COEF", config.get("VF_COEF", 0.5))
    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    aux_losses = {
        "val_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "delta_mag": jnp.abs(delta).mean(),
    }
    return total_loss, aux_losses


def e_lambda_geometric_loss_fn(params, network, traj_batch, gae, targets, config, ent_coef=None):
    """
    Combined Actor-Critic loss function for Brax E(lambda) with geometric jumping (MSE only).
    Uses RMS scaling on GAE advantages, transition-level geometric lookahead pairs,
    and returns auxiliary losses as a dictionary.
    """
    if ent_coef is None:
        ent_coef = config.get("ENT_COEF", 0.001)

    # 1. Forward pass on current observations
    pi, v_i = network.apply(params, traj_batch.obs)
    log_prob = pi.log_prob(traj_batch.action)

    # 2. Actor loss (PPO clipped surrogate with RMS-scaled advantage)
    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae_norm = scale_rms(gae)

    clip_eps = config.get("CLIP_EPS", 0.2)
    surr1 = ratio * gae_norm
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    entropy = pi.entropy().mean()

    # 3. Critic Loss (Transition-level Geometric Jump E(lambda))
    obs_jump = traj_batch.info["obs_jump"]
    targets_jump = traj_batch.info["targets_jump"]
    valid_jump = traj_batch.info["valid_jump"]
    is_absorbing = traj_batch.info["is_absorbing"]

    v_jump = network.apply(params, obs_jump, method=network.value)

    gamma = config.get("GAMMA", 0.99)
    e_lambda = config.get("E_LAMBDA", 0.8)
    value_head_agg = config.get("VALUE_HEAD_AGG", "mean")

    value_loss, magnitude_loss, dirichlet_loss = e_lambda_geometric_critic_loss(
        v_i=v_i,
        targets_i=targets,
        v_jump=v_jump,
        targets_jump=targets_jump,
        valid_jump=valid_jump,
        is_absorbing=is_absorbing,
        gamma=gamma,
        lmbda=e_lambda,
        value_head_agg=value_head_agg,
    )

    vf_coef = config.get("NETWORK_CONFIG", {}).get("VF_COEF", config.get("VF_COEF", 0.5))
    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    v_i_mean = v_i.mean(-1) if v_i.ndim > targets.ndim else v_i
    delta_mag = jnp.abs(targets - v_i_mean).mean()

    aux_losses = {
        "val_loss": value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "delta_mag": delta_mag,
    }
    return total_loss, aux_losses


def e_0_loss_fn(params, network, traj_batch, gae, targets, config, ent_coef=None):
    """
    Combined Actor-Critic loss function for Brax E(0) (adjacent-state Dirichlet smoothness, MSE only).
    Special case of E(lambda) with lambda = 0.0.
    """
    config_e0 = {**config, "E_LAMBDA": 0.0}
    return e_lambda_geometric_loss_fn(
        params=params,
        network=network,
        traj_batch=traj_batch,
        gae=gae,
        targets=targets,
        config=config_e0,
        ent_coef=ent_coef,
    )


# ==============================================================================
# Separate (Split) Actor & Critic Loss Functions
# ==============================================================================

def ppo_actor_loss_fn(actor_params, critic_params, network, traj_batch, gae, config):
    """
    Computes PPO clipped surrogate loss with RMS advantage scaling and entropy bonus.
    Differentiates strictly w.r.t. actor_params.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    pi = network.apply(full_params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)
    entropy = pi.entropy().mean()

    ratio = jnp.exp(log_prob - traj_batch.log_prob)
    gae_norm = scale_rms(gae)

    clip_eps = config.get("CLIP_EPS", 0.2)
    surr1 = ratio * gae_norm
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    ent_coef = config.get("ENT_COEF", 0.001)
    total_actor_loss = actor_loss - ent_coef * entropy
    return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}


def ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Fitted Value Iteration MSE critic loss. Differentiates strictly w.r.t. critic_params.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    vf_clip = config.get("VF_CLIP", config.get("CLIP_EPS", None))
    value_loss = ppo_value_loss(value, targets, value_old=traj_batch.value, vf_clip=vf_clip)
    v_mean = value.mean(-1) if value.ndim > targets.ndim else value
    delta_mag = jnp.abs(targets - v_mean).mean()
    return value_loss, {"val_loss": value_loss, "delta_mag": delta_mag}


def td_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Online 1-step TD(0) MSE critic loss. Differentiates strictly w.r.t. critic_params.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    next_value = network.apply(full_params, traj_batch.next_obs, method=network.value)
    vf_clip = config.get("VF_CLIP", config.get("CLIP_EPS", None))
    gamma = config.get("GAMMA", 0.99)
    value_head_agg = config.get("VALUE_HEAD_AGG", "mean")
    truncation = (
        traj_batch.info.get("truncation", traj_batch.info.get("is_timeout", None))
        if isinstance(traj_batch.info, dict) else None
    )
    value_loss, delta = td_0_value_loss(
        value=value,
        next_value=next_value,
        reward=traj_batch.reward,
        done=traj_batch.done,
        gamma=gamma,
        vf_clip=vf_clip,
        value_old=traj_batch.value,
        value_head_agg=value_head_agg,
        truncation=truncation,
    )
    return value_loss, {"val_loss": value_loss, "delta_mag": jnp.abs(delta).mean()}


def e_lambda_geometric_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """
    Geometric jumping E(lambda) MSE critic loss for Brax state-value PPO.
    Dispatches explicitly on config['E_VARIANT']:
      - 'aux': surgical v_aux on backward branch + self-bootstrapped v_aux MSE loss
      - 'e_aux': direct error head e_phi(s) ~ V(s) - v_theta(s) on backward branch
      - 'fwd': forward-only (e_i - stop_gradient(e_jump))^2 without v_aux
      - 'naive': symmetric (e_i - e_jump)^2 without v_aux
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    obs_jump = traj_batch.info["obs_jump"]
    targets_jump = traj_batch.info["targets_jump"]
    valid_jump = traj_batch.info["valid_jump"]
    is_absorbing = traj_batch.info["is_absorbing"]

    gamma = config["GAMMA"]
    lmbda = config["E_LAMBDA"]
    variant = config["E_VARIANT"].lower()
    aux_coef = config.get("AUX_VF_COEF", 1.0)
    vf_coef = config.get("VF_COEF", 0.5)
    gl = gamma * lmbda

    targets_i = targets if targets is not None else traj_batch.targets

    if variant == "aux":
        v_i, v_aux_i = network.apply(full_params, traj_batch.obs, method=network.both)
        v_jump, v_aux_jump = network.apply(full_params, obs_jump, method=network.both)

        e_i = targets_i - v_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        e_aux_i = sg(v_aux_i) - v_i
        e_aux_jump = jnp.where(is_absorbing, 0.0, sg(v_aux_jump) - v_jump)

        inner_diff_sq = (e_i - sg(e_jump)) ** 2 + (sg(e_aux_i) - e_aux_jump) ** 2
        aux_loss = compute_aux_head_loss(v_aux_i, traj_batch.aux_targets, aux_coef=aux_coef)
    elif variant == "e_aux":
        v_i, e_phi_i = network.apply(full_params, traj_batch.obs, method=network.both)
        v_jump, e_phi_jump = network.apply(full_params, obs_jump, method=network.both)

        e_i = targets_i - v_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        e_phi_jump = jnp.where(is_absorbing, 0.0, e_phi_jump)

        inner_diff_sq = (e_i - sg(e_jump)) ** 2 + e_aux_backward_surrogate(e_phi_i, e_phi_jump, v_jump)
        aux_loss = compute_aux_head_loss(e_phi_i, traj_batch.aux_targets - sg(v_i), aux_coef=aux_coef)
    elif variant == "fwd":
        v_i = network.apply(full_params, traj_batch.obs, method=network.value)
        v_jump = network.apply(full_params, obs_jump, method=network.value)

        e_i = targets_i - v_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        inner_diff_sq = (e_i - sg(e_jump)) ** 2
        aux_loss = 0.0
    elif variant == "naive":
        v_i = network.apply(full_params, traj_batch.obs, method=network.value)
        v_jump = network.apply(full_params, obs_jump, method=network.value)

        e_i = targets_i - v_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        inner_diff_sq = (e_i - e_jump) ** 2
        aux_loss = 0.0
    else:
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive')."
        )

    diff_sq = jnp.where(
        is_absorbing,
        e_i ** 2,
        jnp.where(valid_jump, inner_diff_sq, e_i ** 2),
    )

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    magnitude_loss = magnitude_weight * jnp.mean(e_i ** 2)
    dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
    value_loss = magnitude_loss + dirichlet_loss + aux_loss
    scaled_loss = vf_coef * value_loss

    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "aux_mc_loss": aux_loss,
        "delta_mag": jnp.mean(jnp.abs(e_i)),
    }


def e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    1-sample E(0) critic loss without auxiliary critic v_aux for Brax state-value PPO.
    Dispatches explicitly on config['E_VARIANT']:
      - 'fwd': forward-only (e_i - stop_gradient(e_j))^2
      - 'naive': symmetric (e_i - e_j)^2
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    v_i = network.apply(full_params, traj_batch.obs, method=network.value)
    v_j = network.apply(full_params, traj_batch.next_obs, method=network.value)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout > 0.5
    true_terminal = (traj_batch.done > 0.5) & ~is_timeout

    gamma = config["GAMMA"]
    variant = config["E_VARIANT"].lower()
    vf_coef = config.get("VF_COEF", 0.5)

    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    if variant == "fwd":
        inner_diff_sq = (e_i - sg(e_j)) ** 2
    elif variant in ("naive", "aux"):
        inner_diff_sq = (e_i - e_j) ** 2
    else:
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive')."
        )

    weight_mc = 0.5 * (1.0 - gamma)
    weight_dir = 0.5 * gamma
    start_weight = 0.25 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    diff_sq = jnp.where(true_terminal | is_timeout, e_i ** 2, inner_diff_sq)
    laplacian_loss = weight_dir * jnp.mean(diff_sq)
    start_anchor_loss = start_weight * jnp.mean(jnp.where(traj_batch.is_start, e_i ** 2, 0.0))

    value_loss = magnitude_loss + laplacian_loss + start_anchor_loss
    scaled_loss = vf_coef * value_loss

    delta_mag = jnp.abs(targets_i - v_i).mean()
    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "delta_mag": delta_mag,
    }


def e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Surgical auxiliary E(0) critic loss matching E(lambda=0) for Brax state-value PPO.
    Dispatches explicitly on config['E_VARIANT']:
      - 'aux': surgical v_aux on backward branch + self-bootstrapped v_aux MSE loss
      - 'e_aux': direct error head e_phi(s) ~ V(s) - v_theta(s) on backward branch
      - 'fwd' or 'naive': delegates to e_0_basic_critic_loss_fn
    """
    variant = config["E_VARIANT"].lower()
    if variant in ("naive", "fwd"):
        return e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config)
    elif variant not in ("aux", "e_aux"):
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive')."
        )

    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    v_i, aux_head_i = network.apply(full_params, traj_batch.obs, method=network.both)
    v_j, aux_head_j = network.apply(full_params, traj_batch.next_obs, method=network.both)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    aux_targets = traj_batch.aux_targets
    is_timeout = traj_batch.is_timeout > 0.5
    true_terminal = (traj_batch.done > 0.5) & ~is_timeout
    is_start = traj_batch.is_start

    gamma = config["GAMMA"]
    aux_coef = config.get("AUX_VF_COEF", 1.0)
    vf_coef = config.get("VF_COEF", 0.5)

    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    if variant == "aux":
        aux_mc_loss = compute_aux_head_loss(aux_head_i, aux_targets, aux_coef=aux_coef)
        e_tilde_i = sg(aux_head_i) - v_i
        e_tilde_j = jnp.where(true_terminal, 0.0, sg(aux_head_j) - v_j)
        backward_sq = (sg(e_tilde_i) - e_tilde_j) ** 2
    else:  # e_aux
        aux_mc_loss = compute_aux_head_loss(aux_head_i, aux_targets - sg(v_i), aux_coef=aux_coef)
        e_phi_j = jnp.where(true_terminal, 0.0, aux_head_j)
        backward_sq = e_aux_backward_surrogate(aux_head_i, e_phi_j, v_j)

    weight_mc = 0.5 * (1.0 - gamma)
    weight_dir = 0.5 * gamma
    start_weight = 0.25 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    branch_v_i = (e_i - sg(e_j)) ** 2
    branch_v_j = jnp.where(true_terminal, 0.0, backward_sq)
    diff_sq = jnp.where(
        true_terminal | is_timeout,
        e_i ** 2,
        branch_v_i + branch_v_j,
    )
    laplacian_loss = weight_dir * jnp.mean(diff_sq)
    start_anchor_loss = start_weight * jnp.mean(jnp.where(is_start, e_i ** 2, 0.0))

    value_loss = magnitude_loss + laplacian_loss + start_anchor_loss + aux_mc_loss
    scaled_loss = vf_coef * value_loss

    delta_mag = jnp.abs(targets_i - v_i).mean()
    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "aux_mc_loss": aux_mc_loss,
        "delta_mag": delta_mag,
    }


mc_critic_loss_fn = ppo_critic_loss_fn


def double_sample_msbe_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """
    Baird double-sampled Mean Squared Bellman Error (MSBE) critic loss:
        L(theta) = 0.5 * [ sg(delta_1) * delta_2 + sg(delta_2) * delta_1 ]
    where delta_1 and delta_2 are 1-step Bellman errors from the same state s_t
    evaluated along two independent transitions (branch 1 and branch 2).
    In expectation conditional on s_t, E[grad L] = E[delta] * grad E[delta] = grad (0.5 * E[delta]^2).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    gamma = config["GAMMA"]

    v = network.apply(full_params, traj_batch.obs, method=network.value)
    v_next_1 = network.apply(full_params, traj_batch.next_obs, method=network.value)
    v_next_2 = network.apply(full_params, traj_batch.probe_next_obs, method=network.value)

    true_term_1 = traj_batch.done & ~traj_batch.is_timeout
    true_term_2 = traj_batch.probe_done & ~traj_batch.probe_is_timeout

    if v.ndim > traj_batch.reward.ndim:
        reward_1 = traj_batch.reward[..., None]
        reward_2 = traj_batch.probe_reward[..., None]
        true_term_1 = true_term_1[..., None]
        true_term_2 = true_term_2[..., None]
    else:
        reward_1 = traj_batch.reward
        reward_2 = traj_batch.probe_reward

    bootstrap_1 = 1.0 - true_term_1
    bootstrap_2 = 1.0 - true_term_2

    delta_1 = reward_1 + gamma * bootstrap_1 * v_next_1 - v
    delta_2 = reward_2 + gamma * bootstrap_2 * v_next_2 - v

    loss = 0.5 * (sg(delta_1) * delta_2 + sg(delta_2) * delta_1)
    value_loss = jnp.mean(loss)
    diagnostic_msbe = jnp.mean(delta_1 * delta_2)

    return value_loss, {
        "val_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta_1)),
        "delta2_mag": jnp.mean(jnp.abs(delta_2)),
        "diagnostic_msbe": diagnostic_msbe,
    }


def double_sample_e_inner_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """
    Double-sampled Bilinear Energy Minimization critic loss:
        E = 0.5 * <e, bar_delta>_D
    Using branch 1 for empirical error e_hat_1 = targets_1 - v(s_t) (where targets_1 = G_t^(1)),
    and branch 2 (probe) for independent next state s_(t+1)^(2) and delta_2.
    Surrogate loss:
        L(theta) = -0.5 * [ sg(bar_delta) + sg(e_hat_1) ] * v(s_t)
                 + 0.5 * gamma * sg(e_hat_1) * (1 - true_terminal_2) * v(s_(t+1)^(2))
    where bar_delta = 0.5 * (delta_1 + delta_2).
    In expectation conditional on s_t, grad L matches grad E with ZERO double-sampling bias
    and requires no auxiliary value network (v_aux).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    gamma = config["GAMMA"]

    targets_i = targets if targets is not None else traj_batch.targets

    v = network.apply(full_params, traj_batch.obs, method=network.value)
    v_next_1 = network.apply(full_params, traj_batch.next_obs, method=network.value)
    v_next_2 = network.apply(full_params, traj_batch.probe_next_obs, method=network.value)

    true_term_1 = traj_batch.done & ~traj_batch.is_timeout
    true_term_2 = traj_batch.probe_done & ~traj_batch.probe_is_timeout

    if v.ndim > traj_batch.reward.ndim:
        reward_1 = traj_batch.reward[..., None]
        reward_2 = traj_batch.probe_reward[..., None]
        true_term_1 = true_term_1[..., None]
        true_term_2 = true_term_2[..., None]
        targets_i = targets_i[..., None]
    else:
        reward_1 = traj_batch.reward
        reward_2 = traj_batch.probe_reward

    bootstrap_1 = 1.0 - true_term_1
    bootstrap_2 = 1.0 - true_term_2

    delta_1 = reward_1 + gamma * bootstrap_1 * v_next_1 - v
    delta_2 = reward_2 + gamma * bootstrap_2 * v_next_2 - v
    delta_bar = 0.5 * (delta_1 + delta_2)

    e_hat_1 = targets_i - v

    loss = (
        -0.5 * (sg(delta_bar) + sg(e_hat_1)) * v
        + 0.5 * gamma * sg(e_hat_1) * bootstrap_2 * v_next_2
    )

    value_loss = jnp.mean(loss)
    diagnostic_msbe = jnp.mean(delta_1 * delta_2)

    return value_loss, {
        "val_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta_1)),
        "delta2_mag": jnp.mean(jnp.abs(delta_2)),
        "e_mag": jnp.mean(jnp.abs(e_hat_1)),
        "diagnostic_msbe": diagnostic_msbe,
    }


def double_sample_mspbe_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """
    Nonlinear Mean Squared Projected Bellman Error (MSPBE) with 1-step Double Sampling.
    Formally projects the Bellman error onto the tangent space (Jacobian span) of the
    value function manifold, as derived in Maei et al. (NeurIPS 2009):
        grad_theta MSPBE = - E[ (psi(s) - gamma * psi(s'_2)) * (psi(s)^T w*) ]
                           - E[ (delta_2 - psi(s)^T w*) * (nabla_theta^2 V_theta(s) w*) ]
    where:
      - w* is the optimal tangent projection weight vector estimated on Branch 1 (delta_1)
        via conjugate gradient without constructing full Jacobian/Gram matrices.
      - delta_2 and s'_2 are from the conditionally independent Branch 2 (probe), eliminating
        sampling bias on both the tangent projection and the Hessian-vector product.
      - nabla_theta^2 V_theta(s) w* is the exact Hessian-vector product (HVP) computed via
        forward-over-reverse autodiff: d/d theta [ <nabla_theta V_theta(s), w*> ].
    """
    full_params_fn = lambda p: {"params": {**actor_params, **p}}
    sg = jax.lax.stop_gradient
    gamma = config["GAMMA"]
    lambda_ridge = config.get("MSPBE_RIDGE", 1e-4)
    cg_iters = config.get("MSPBE_CG_ITERS", 10)

    obs = traj_batch.obs
    batch_size = obs.shape[0]

    def v_batch(p, x):
        val = network.apply(full_params_fn(p), x, method=network.value)
        if val.ndim > 1:
            val = jnp.squeeze(val, axis=-1)
        return val

    # Continuation values (Branch 1 and Branch 2)
    v_s_old = sg(v_batch(critic_params, obs))
    v_next_1_old = sg(v_batch(critic_params, traj_batch.next_obs))
    v_next_2_old = sg(v_batch(critic_params, traj_batch.probe_next_obs))

    true_term_1 = traj_batch.done & ~traj_batch.is_timeout
    true_term_2 = traj_batch.probe_done & ~traj_batch.probe_is_timeout

    boot_1 = 1.0 - true_term_1
    boot_2 = 1.0 - true_term_2

    reward_1 = traj_batch.reward
    reward_2 = traj_batch.probe_reward
    if reward_1.ndim > 1:
        reward_1 = jnp.squeeze(reward_1, axis=-1)
    if reward_2.ndim > 1:
        reward_2 = jnp.squeeze(reward_2, axis=-1)

    delta_1 = reward_1 + gamma * boot_1 * v_next_1_old - v_s_old
    delta_2 = reward_2 + gamma * boot_2 * v_next_2_old - v_s_old

    # 1. Solve for tangent projection weights w* using Branch 1
    _, vjp_fn = jax.vjp(lambda p: v_batch(p, obs), critic_params)
    b = vjp_fn(delta_1 / batch_size)[0]

    def matvec(w):
        _, jv = jax.jvp(lambda p: v_batch(p, obs), (critic_params,), (w,))
        vj = vjp_fn(jv / batch_size)[0]
        return jax.tree_util.tree_map(lambda v, w_: v + lambda_ridge * w_, vj, w)

    w_init = jax.tree_util.tree_map(jnp.zeros_like, critic_params)
    w_star, _ = jax.scipy.sparse.linalg.cg(matvec, b, x0=w_init, maxiter=cg_iters)
    w_star = sg(w_star)

    # 2. Projected Bellman error and orthogonal residual on Branch 2
    _, delta_proj = jax.jvp(lambda p: v_batch(p, obs), (critic_params,), (w_star,))
    delta_proj = sg(delta_proj)
    r_perp = sg(delta_2 - delta_proj)

    # 3. Live surrogate terms w.r.t critic_params
    v_s_live = v_batch(critic_params, obs)
    v_next_2_live = v_batch(critic_params, traj_batch.probe_next_obs)
    _, jvp_live = jax.jvp(lambda p: v_batch(p, obs), (critic_params,), (w_star,))

    term1 = delta_proj * (gamma * boot_2 * v_next_2_live - v_s_live)
    term2 = - r_perp * jvp_live

    value_loss = jnp.mean(term1 + term2)
    diagnostic_msbe = jnp.mean(delta_1 * delta_2)
    diagnostic_mspbe = jnp.mean(delta_proj ** 2)

    return value_loss, {
        "val_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta_1)),
        "delta2_mag": jnp.mean(jnp.abs(delta_2)),
        "delta_proj_mag": jnp.mean(jnp.abs(delta_proj)),
        "r_perp_mag": jnp.mean(jnp.abs(r_perp)),
        "diagnostic_msbe": diagnostic_msbe,
        "diagnostic_mspbe": diagnostic_mspbe,
    }


SPLIT_CRITIC_LOSS_FNS = {
    "fitted": ppo_critic_loss_fn,
    "ppo": ppo_critic_loss_fn,
    "td": td_0_critic_loss_fn,
    "td_0": td_0_critic_loss_fn,
    "td0": td_0_critic_loss_fn,
    "classic_td": td_0_critic_loss_fn,
    "e_0": e_0_critic_loss_fn,
    "e_0_basic": e_0_basic_critic_loss_fn,
    "e_basic": e_0_basic_critic_loss_fn,
    "e_0_biased": e_0_basic_critic_loss_fn,
    "e_biased": e_0_basic_critic_loss_fn,
    "e0": e_0_critic_loss_fn,
    "e": e_0_critic_loss_fn,
    "sampled_e": e_0_critic_loss_fn,
    "surgical_v_aux": e_0_critic_loss_fn,
    "surgical": e_0_critic_loss_fn,
    "e_lambda": e_lambda_geometric_critic_loss_fn,
    "e_geometric": e_lambda_geometric_critic_loss_fn,
    "elambda": e_lambda_geometric_critic_loss_fn,
    "mc": mc_critic_loss_fn,
    "sampled_mc": mc_critic_loss_fn,
    "msbe": double_sample_msbe_critic_loss_fn,
    "double_msbe": double_sample_msbe_critic_loss_fn,
    "e_inner": double_sample_e_inner_critic_loss_fn,
    "e_bilinear": double_sample_e_inner_critic_loss_fn,
    "e_double": double_sample_e_inner_critic_loss_fn,
    "e_min": double_sample_e_inner_critic_loss_fn,
    "mspbe": double_sample_mspbe_critic_loss_fn,
    "double_mspbe": double_sample_mspbe_critic_loss_fn,
}


CRITIC_LOSS_FNS = {
    "fitted": ppo_loss_fn,
    "ppo": ppo_loss_fn,
    "td": td_0_loss_fn,
    "td_0": td_0_loss_fn,
    "td0": td_0_loss_fn,
    "e_0": e_0_loss_fn,
    "e0": e_0_loss_fn,
    "e": e_0_loss_fn,
    "e_lambda": e_lambda_geometric_loss_fn,
    "e_geometric": e_lambda_geometric_loss_fn,
    "elambda": e_lambda_geometric_loss_fn,
}


def get_critic_loss_fn(critic_type: str, split: bool = True):
    """
    Returns the critic loss function matching the specified critic_type.
    If split=True (default), returns the decoupled critic loss function
    (critic_params, actor_params, network, traj_batch, targets, config).
    """
    key = critic_type.lower()
    table = SPLIT_CRITIC_LOSS_FNS if split else CRITIC_LOSS_FNS
    if key not in table:
        raise ValueError(f"Unknown critic_type '{critic_type}'. Available: {list(table.keys())}")
    return table[key]


# Alias for compatibility
_loss_fn = ppo_loss_fn
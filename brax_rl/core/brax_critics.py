import jax
import jax.numpy as jnp
from brax_rl.core.brax_utils import scale_rms


def calculate_gae(traj_batch, last_val, gamma: float = 0.99, gae_lambda: float = 0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and value targets.
    """
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward,
        )
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    return advantages, targets


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
    Attaches jump transition data (obs_jump, targets_jump, valid_jump, is_absorbing)
    into traj_batch.info for E0 or E(lambda) critics before PyTree shuffling.
    For fitted and TD critics, returns traj_batch unmodified.
    """
    key = critic_type.lower()
    if key in ["e_0", "e0", "e"]:
        obs_jump, targets_jump, valid_jump, is_absorbing = prepare_geometric_jump_transitions(
            traj_batch=traj_batch,
            targets=targets,
            gamma=gamma,
            lmbda=0.0,
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
    elif key in ["e_lambda", "elambda", "e_geometric"]:
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


def e_lambda_geometric_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Geometric jumping E(lambda) MSE critic loss. Differentiates strictly w.r.t. critic_params.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    v_i = network.apply(full_params, traj_batch.obs, method=network.value)
    obs_jump = traj_batch.info["obs_jump"]
    targets_jump = traj_batch.info["targets_jump"]
    valid_jump = traj_batch.info["valid_jump"]
    is_absorbing = traj_batch.info["is_absorbing"]
    v_jump = network.apply(full_params, obs_jump, method=network.value)

    gamma = config.get("GAMMA", 0.99)
    e_lambda = config.get("E_LAMBDA", 0.8)
    value_head_agg = config.get("VALUE_HEAD_AGG", "mean")

    value_loss, mag_loss, dir_loss = e_lambda_geometric_critic_loss(
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
    v_i_mean = v_i.mean(-1) if v_i.ndim > targets.ndim else v_i
    delta_mag = jnp.abs(targets - v_i_mean).mean()
    return value_loss, {
        "val_loss": value_loss,
        "magnitude_loss": mag_loss,
        "dirichlet_loss": dir_loss,
        "delta_mag": delta_mag,
    }


def e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    E(0) adjacent-state Dirichlet MSE critic loss. Differentiates strictly w.r.t. critic_params.
    """
    config_e0 = {**config, "E_LAMBDA": 0.0}
    return e_lambda_geometric_critic_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config_e0
    )


SPLIT_CRITIC_LOSS_FNS = {
    "fitted": ppo_critic_loss_fn,
    "ppo": ppo_critic_loss_fn,
    "td": td_0_critic_loss_fn,
    "td_0": td_0_critic_loss_fn,
    "td0": td_0_critic_loss_fn,
    "e_0": e_0_critic_loss_fn,
    "e0": e_0_critic_loss_fn,
    "e": e_0_critic_loss_fn,
    "e_lambda": e_lambda_geometric_critic_loss_fn,
    "e_geometric": e_lambda_geometric_critic_loss_fn,
    "elambda": e_lambda_geometric_critic_loss_fn,
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
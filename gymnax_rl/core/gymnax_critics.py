import jax
import jax.numpy as jnp
from gymnax_rl.core.gymnax_utils import scale_rms, post_process_advantage


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

get_return = calculate_targets
calculate_return = calculate_targets


def get_critic_target_lambda(config: dict) -> float:
    """
    Returns the appropriate lambda to compute value targets based on CRITIC_TYPE:
    - E-minimization (e_0, e_lambda): RETURN_LAMBDA (default 0.99 / 1.0)
    - TD(0): TD_LAMBDA (default 0.0)
    - Fitted PPO: VALUE_LAMBDA (default GAE_LAMBDA, e.g. 0.9 / 0.95)
    - MC: 1.0
    """
    critic_type = config.get("CRITIC_TYPE", "fitted").lower()
    if critic_type in ["e_0", "e0", "e", "e_0_basic", "e_basic", "e_lambda", "elambda", "e_geometric", "sampled_e", "surgical_v_aux", "surgical", "e_0_q", "e_lambda_q"]:
        return config.get("RETURN_LAMBDA", 1.0)
    elif critic_type in ["td_0", "td", "td0", "classic_td"]:
        return config.get("TD_LAMBDA", 0.0)
    elif critic_type in ["mc", "sampled_mc"]:
        return 1.0
    else:  # fitted / ppo / td_lambda
        return config.get("VALUE_LAMBDA", config.get("GAE_LAMBDA", 0.9))


def huber_loss(x, delta: float = 1.0):
    """Huber loss (Smooth L1) with linear scaling for |x| > delta."""
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    linear = abs_x - quadratic
    return 0.5 * (quadratic ** 2) + delta * linear


def ppo_value_loss(value, targets, value_old=None, vf_clip=None, value_head_agg="sum", loss_type="mse", delta=1.0):
    """
    PPO value loss with optional value clipping and multi-head aggregation.
    """
    if value.ndim > targets.ndim:
        targets_exp = targets[..., None]
    else:
        targets_exp = targets

    if loss_type == "huber":
        err_fn = lambda d: huber_loss(d, delta)
    else:
        err_fn = lambda d: 0.5 * jnp.square(d)

    value_losses = err_fn(value - targets_exp)
    if vf_clip is not None and value_old is not None:
        if value_old.ndim < value.ndim:
            value_old = value_old[..., None]
        value_pred_clipped = value_old + jnp.clip(value - value_old, -vf_clip, vf_clip)
        value_losses_clipped = err_fn(value_pred_clipped - targets_exp)
        max_losses = jnp.maximum(value_losses, value_losses_clipped)
    else:
        max_losses = value_losses

    if value.ndim > 1:
        loss_per_head = jnp.mean(max_losses, axis=0)
        return jnp.sum(loss_per_head) if value_head_agg == "sum" else jnp.mean(loss_per_head)
    return max_losses.mean()


def td_0_value_loss(
    value,
    next_value,
    reward,
    done,
    is_timeout,
    gamma: float = 0.99,
    vf_clip=None,
    value_old=None,
    value_head_agg: str = "mean",
    loss_type: str = "mse",
    delta_huber: float = 1.0,
):
    """
    Computes online 1-step TD(0) loss using live next_value:
        target = stop_gradient(reward + gamma * (1 - true_terminal) * next_value)
    """
    true_terminal = done & ~is_timeout

    if value.ndim > 1:
        if reward.ndim == 1:
            reward = reward[:, None]
        if true_terminal.ndim == 1:
            true_terminal = true_terminal[:, None]

    td_target = jax.lax.stop_gradient(reward + gamma * (1.0 - true_terminal) * next_value)
    delta = td_target - value

    if loss_type == "huber":
        value_losses = huber_loss(delta, delta_huber)
    else:
        value_losses = 0.5 * jnp.square(delta)

    if vf_clip is not None and value_old is not None:
        if value_old.ndim < value.ndim:
            value_old = value_old[:, None]
        value_pred_clipped = value_old + jnp.clip(value - value_old, -vf_clip, vf_clip)
        delta_clipped = td_target - value_pred_clipped
        if loss_type == "huber":
            value_losses_clipped = huber_loss(delta_clipped, delta_huber)
        else:
            value_losses_clipped = 0.5 * jnp.square(delta_clipped)
        value_losses = jnp.maximum(value_losses, value_losses_clipped)

    if value.ndim > 1:
        per_head = jnp.mean(value_losses, axis=0)
        value_loss = jnp.sum(per_head) if value_head_agg == "sum" else jnp.mean(per_head)
    else:
        value_loss = value_losses.mean()

    return value_loss, delta


def prepare_e_lambda_jump_pairs(
    traj_batch,
    returns: jax.Array,
    last_val: jax.Array,
    gamma: float,
    lmbda: float,
    rng: jax.Array,
    steps_since_start: jax.Array,
    jump_mode: str = "full_mc_invalid_jump",
):
    """
    Pairs each source state s_t with its geometric jump destination s_j = s_{min(t+m, T)},
    evaluates boundary conditions in O(1), and packages jump transition attributes.
    Matches bellman_error/ppo/E_lambda_geometric.py.
    """
    T, B = returns.shape[:2]
    gl = gamma * lmbda
    safe_gl = jnp.clip(gl, 1e-8, 1.0 - 1e-8)

    # 1. Sample geometric jump lengths: m = K + 1 where K ~ Geom(1 - gl) for all T steps
    u = jax.random.uniform(rng, shape=(T, B))
    jumps = jnp.where(
        gl < 1e-6,
        jnp.zeros((T, B), dtype=jnp.int32),
        jnp.floor(jnp.log(jnp.clip(1.0 - u, 1e-8, 1.0)) / jnp.log(safe_gl)).astype(jnp.int32),
    )
    m = jnp.maximum(jumps, 0) + 1
    t_arr = jnp.arange(T)[:, None]
    j = jnp.minimum(t_arr + m, T)

    # 2. Episode boundary queries in [t, j - 1]
    is_timeout = traj_batch.is_timeout
    true_terminals = traj_batch.done & ~is_timeout

    term_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(true_terminals.astype(jnp.int32), axis=0)],
        axis=0,
    )
    timeout_cumsum = jnp.concatenate(
        [jnp.zeros((1, B), dtype=jnp.int32), jnp.cumsum(is_timeout.astype(jnp.int32), axis=0)],
        axis=0,
    )

    hit_true_terminal = (jnp.take_along_axis(term_cumsum, j, axis=0) - term_cumsum[:T]) > 0
    hit_timeout = (jnp.take_along_axis(timeout_cumsum, j, axis=0) - timeout_cumsum[:T]) > 0
    hit_horizon = (t_arr + m > T)

    is_valid_jump = (t_arr + m <= T) | hit_true_terminal
    if jump_mode == "drop_invalid_jump":
        critic_mask = is_valid_jump.astype(jnp.float32)
    else:
        critic_mask = jnp.ones((T, B), dtype=jnp.float32)

    # 3. Track steps since start (clearing to 10,000 on timeout)
    def scan_steps(k, step_flags):
        term, timeout = step_flags
        k_next = jnp.where(term, 0, jnp.where(timeout, 10_000, k + 1))
        return k_next, k

    init_k = jnp.full((B,), 10_000, dtype=jnp.int32) if steps_since_start is None else steps_since_start
    final_steps_since_start, all_steps = jax.lax.scan(scan_steps, init_k, (true_terminals, is_timeout))

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    start_trace = jnp.where(
        gl < 1e-6,
        jnp.where(all_steps == 0, 1.0, 0.0),
        safe_gl ** all_steps,
    )
    start_weight = 0.25 * tilde_gamma * start_trace

    # 4. Form extended arrays (length T + 1) for destination lookups at step T
    obs_ext = jnp.concatenate([traj_batch.obs, traj_batch.next_obs[-1][None]], axis=0)
    target_ext = jnp.concatenate([returns, last_val[None]], axis=0)

    # Gather destination states and targets along time axis (axis 0)
    j_expanded = j.reshape(j.shape + (1,) * (obs_ext.ndim - 2))
    j_expanded = jnp.broadcast_to(j_expanded, (T, B) + obs_ext.shape[2:])
    dest_obs = jnp.take_along_axis(obs_ext, j_expanded, axis=0)
    dest_target = jnp.take_along_axis(target_ext, j, axis=0)

    jump_batch = traj_batch._replace(
        dest_obs=dest_obs,
        dest_targets=dest_target,
        hit_true_terminal=hit_true_terminal,
        hit_timeout=hit_timeout,
        hit_horizon=hit_horizon,
        start_weight=start_weight,
        critic_mask=critic_mask,
    )
    return jump_batch, final_steps_since_start


prepare_geometric_jump_transitions = prepare_e_lambda_jump_pairs


def prepare_critic_transitions(
    critic_type: str,
    traj_batch,
    targets: jax.Array,
    last_val: jax.Array,
    gamma: float,
    e_lambda: float,
    rng: jax.Array,
    steps_since_start: jax.Array = None,
    config: dict = None,
):
    """
    Attaches jump transition data for E(lambda) critics before PyTree shuffling.
    For fitted, TD, and E(0) critics, returns traj_batch unmodified.
    """
    key = critic_type.lower()
    if key in ["e_lambda", "elambda", "e_geometric"]:
        cfg = config if config is not None else {}
        jump_mode = cfg.get("E_LAMBDA_JUMP_MODE", "full_mc_invalid_jump")
        if cfg.get("E_LAMBDA_FULL_WEIGHT_MC", cfg.get("FULL_WEIGHT_MC_INVALID", False)):
            jump_mode = "full_mc_invalid_jump"
        return prepare_e_lambda_jump_pairs(
            traj_batch=traj_batch,
            returns=targets,
            last_val=last_val,
            gamma=gamma,
            lmbda=e_lambda,
            rng=rng,
            steps_since_start=steps_since_start,
            jump_mode=jump_mode,
        )
    return traj_batch, steps_since_start


def ppo_actor_loss_fn(actor_params, critic_params, network, traj_batch, gae, config):
    """PPO clipped surrogate actor loss."""
    full_params = {"params": {**actor_params, **critic_params}}
    pi = network.apply(full_params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    adv_norm = post_process_advantage(gae, config)
    surr1 = ratio * adv_norm
    surr2 = (
        jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    )
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    ent_coef = config.get("ENT_COEF", 0.01)
    total_actor_loss = config.get("POLICY_COEFF", 1.0) * actor_loss - ent_coef * entropy
    return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}


def ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Fitted Value Iteration MSE/Huber critic loss."""
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    vf_clip = config.get("VF_CLIP", None)
    loss_type = config.get("CRITIC_LOSS_TYPE", "mse")
    delta = config.get("HUBER_DELTA", 1.0)
    agg = config.get("VALUE_HEAD_AGG", "sum")

    value_loss = ppo_value_loss(
        value=value,
        targets=targets,
        value_old=traj_batch.value,
        vf_clip=vf_clip,
        value_head_agg=agg,
        loss_type=loss_type,
        delta=delta,
    )
    v_mean = value.mean(-1) if value.ndim > targets.ndim else value
    delta_mag = jnp.abs(targets - v_mean).mean()
    scaled_loss = config.get("VF_COEF", 0.5) * value_loss
    return scaled_loss, {"val_loss": value_loss, "delta_mag": delta_mag}


def td_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Online 1-step TD(0) critic loss."""
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    next_value = network.apply(full_params, traj_batch.next_obs, method=network.value)
    vf_clip = config.get("VF_CLIP", None)
    gamma = config.get("GAMMA", 0.99)
    value_head_agg = config.get("VALUE_HEAD_AGG", "sum")
    loss_type = config.get("CRITIC_LOSS_TYPE", "mse")
    delta_huber = config.get("HUBER_DELTA", 1.0)
    value_loss, delta = td_0_value_loss(
        value=value,
        next_value=next_value,
        reward=traj_batch.reward,
        done=traj_batch.done,
        is_timeout=traj_batch.is_timeout,
        gamma=gamma,
        vf_clip=vf_clip,
        value_old=traj_batch.value,
        value_head_agg=value_head_agg,
        loss_type=loss_type,
        delta_huber=delta_huber,
    )
    scaled_loss = config.get("VF_COEF", 0.5) * value_loss
    return scaled_loss, {"val_loss": value_loss, "delta_mag": jnp.abs(delta).mean()}



def compute_aux_head_loss(v_aux, v_aux_old, aux_targets, config, aux_coef=1.0):
    sg = jax.lax.stop_gradient
    targets = sg(aux_targets)
    err = targets - v_aux
    loss_type = config.get("AUX_LOSS_TYPE", config.get("V_AUX_LOSS_TYPE", "mse"))
    delta = config.get("AUX_HUBER_DELTA", config.get("V_AUX_HUBER_DELTA", 10.0))

    def _base_loss(e):
        if loss_type == "huber":
            return huber_loss(e, delta=delta)
        else:
            return 0.5 * (e ** 2)

    l1 = _base_loss(err)
    reg_type = config.get("AUX_REG_TYPE", config.get("V_AUX_REG_TYPE", "l2"))
    clip_eps = config.get("AUX_CLIP_EPS", config.get("V_AUX_CLIP_EPS", 10.0))
    beta = config.get("AUX_TRUST_REGION_COEF", config.get("V_AUX_TRUST_COEF", 0.0))

    if reg_type == "clipping" and v_aux_old is not None:
        v_clipped = sg(v_aux_old) + jnp.clip(v_aux - sg(v_aux_old), -clip_eps, clip_eps)
        err_clipped = targets - v_clipped
        l2 = _base_loss(err_clipped)
        final_element_loss = jnp.maximum(l1, l2)
        return aux_coef * jnp.mean(final_element_loss)
    elif reg_type == "l2" and beta > 0.0 and v_aux_old is not None:
        trust_loss = beta * 0.5 * ((v_aux - sg(v_aux_old)) ** 2)
        return aux_coef * jnp.mean(l1 + trust_loss)
    else:
        return aux_coef * jnp.mean(l1)

def e_lambda_geometric_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """
    Geometric jumping E(lambda) MSE critic loss matching bellman_error/ppo/E_lambda_geometric.py.
    - Uses v_aux for the backwards branch of the Dirichlet loss.
    - Forward branch uses empirical sample returns G_t, G_j.
    - Corrected start-state anchor with geometric trace discount.
    - Full-MC target replacement for invalid jumps (timeouts / buffer overshoots).
    - Trains v_aux towards self-bootstrapped return targets.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    # Evaluate main value head and auxiliary head
    v_start, v_aux_start = network.apply(full_params, traj_batch.obs, method=network.both)
    v_jump, v_aux_jump = network.apply(full_params, traj_batch.dest_obs, method=network.both)

    gamma = config["GAMMA"]
    lmbda = config.get("E_LAMBDA", config.get("VALUE_LAMBDA", 0.8))
    gl = gamma * lmbda

    targets_i = targets if targets is not None else traj_batch.targets
    e_start = targets_i - v_start
    e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)

    e_aux_start = sg(v_aux_start) - v_start
    e_aux_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

    variant = config.get("E_LAMBDA_VARIANT", config.get("E_VARIANT", "aux"))
    if variant in ["naive", "oct8"]:
        inner_diff_sq = (e_start - e_jump) ** 2
    elif variant in ["fwd", "forward", "forward_e"]:
        inner_diff_sq = (e_start - sg(e_jump)) ** 2
    else:
        # Surgical use of v_aux on the backward branch
        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2

    is_invalid = traj_batch.hit_timeout | traj_batch.hit_horizon
    jump_mode = config.get("E_LAMBDA_JUMP_MODE", "full_mc_invalid_jump")
    full_mc = config.get("E_LAMBDA_FULL_WEIGHT_MC", config.get("FULL_WEIGHT_MC_INVALID", True))
    is_full_mc = full_mc or (jump_mode in ["full_mc_invalid_jump", "mc_full_weight", "mc_full", "full_weight_mc"])

    # Full unit-weight MC target replacement for invalid jumps (t + m >= T or timeout)
    invalid_diff_sq = jnp.where(is_full_mc, e_start ** 2, 0.0)
    diff_sq = jnp.where(
        traj_batch.hit_true_terminal,
        e_start ** 2,
        jnp.where(is_invalid, invalid_diff_sq, inner_diff_sq),
    )

    tilde_gamma = (gamma * (1.0 - lmbda)) / jnp.maximum(1.0 - gl, 1e-8)
    magnitude_weight = 0.5 * (1.0 - gamma) / jnp.maximum(1.0 - gl, 1e-8)
    dirichlet_weight = 0.5 * tilde_gamma

    c_mask = traj_batch.critic_mask
    if c_mask is not None:
        c_mask = c_mask.astype(jnp.float32)
        n_crit = jnp.maximum(jnp.sum(c_mask), 1.0)
        magnitude_loss = magnitude_weight * (jnp.sum((e_start ** 2) * c_mask) / n_crit)
        dirichlet_loss = dirichlet_weight * (jnp.sum(diff_sq * c_mask) / n_crit)
        start_loss = jnp.sum(traj_batch.start_weight * (e_start ** 2) * c_mask) / n_crit
    else:
        magnitude_loss = magnitude_weight * jnp.mean(e_start ** 2)
        dirichlet_loss = dirichlet_weight * jnp.mean(diff_sq)
        start_loss = jnp.mean(traj_batch.start_weight * (e_start ** 2))

    value_loss = magnitude_loss + dirichlet_loss + start_loss

    # Auxiliary head loss on self-bootstrapped return targets
    aux_coef = config.get("AUX_VF_COEF", 1.0)
    aux_loss = compute_aux_head_loss(
        v_aux_start, traj_batch.v_aux, traj_batch.aux_targets, config, aux_coef=aux_coef
    )

    total_loss = value_loss + aux_loss
    scaled_loss = config.get("VF_COEF", 0.5) * total_loss

    aux_metrics = {
        "val_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "start_loss": start_loss,
        "aux_mc_loss": aux_loss,
        "delta_mag": jnp.mean(jnp.abs(e_start)),
    }
    return scaled_loss, aux_metrics


def e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Basic / biased 1-sample E(0) critic loss (without auxiliary critic V_aux).
    Smooths v(s_t) directly against v(s_{t+1}) with magnitude anchoring:
      L = (1 - gamma) * E[(targets - v(s_t))^2] + 0.5 * gamma * E[((targets - v(s_t)) - (next_targets - v(s_{t+1})))^2]
    On true terminal states, the absorbing error is 0.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    v_i = network.apply(full_params, traj_batch.obs, method=network.value)
    v_j = network.apply(full_params, traj_batch.next_obs, method=network.value)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout

    gamma = config["GAMMA"]

    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    weight_mc = 1.0 - gamma
    weight_dir = 0.5 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    laplacian_loss = weight_dir * jnp.mean((e_i - e_j) ** 2)

    value_loss = magnitude_loss + laplacian_loss
    scaled_loss = config.get("VF_COEF", 0.5) * value_loss

    delta_mag = jnp.abs(targets_i - v_i).mean()
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


def e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    if not config.get("USES_AUX", config.get("SURGICAL_V_AUX", True)):
        return e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config)
    """
    Surgical auxiliary E(0) critic loss matching bellman_error/ppo/sampled_E.py.
    - Uses v_aux for the backwards branch of the Dirichlet loss.
    - Forward branch uses sample returns G_t, G_{t+1} (noise cancels into exact 1-step TD(0) reward).
    - Corrected start-state anchor 0.5 * gamma * E_{s0}[e(s0)^2] and timeout boundary correction.
    - Trains v_aux towards self-bootstrapped return targets.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    # Evaluate main value head and auxiliary head
    v_i, v_aux_i = network.apply(full_params, traj_batch.obs, method=network.both)
    v_j, v_aux_j = network.apply(full_params, traj_batch.next_obs, method=network.both)

    # First-class attributes on Transition
    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    aux_targets = traj_batch.aux_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout
    is_start = traj_batch.is_start

    gamma = config["GAMMA"]
    aux_coef = config.get("AUX_VF_COEF", 1.0)

    # Empirical MC errors
    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    # Auxiliary head loss and errors
    aux_mc_loss = compute_aux_head_loss(
        v_aux_i, traj_batch.v_aux, aux_targets, config, aux_coef=aux_coef
    )
    e_tilde_i = sg(v_aux_i) - v_i
    e_tilde_j = jnp.where(true_terminal, 0.0, sg(v_aux_j) - v_j)

    weight_mc = 1.0 - gamma
    weight_dir = 0.5 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)

    # Surgical use of v_aux:
    # Forward branch on v_i uses real returns (cancels noise into TD(0)):
    branch_v_i = (e_i - sg(e_j)) ** 2
    # Backward branch on v_j uses v_aux (eliminates return noise & double-sampling bias):
    branch_v_j = jnp.where(true_terminal, 0.0, (sg(e_tilde_i) - e_tilde_j) ** 2)
    laplacian_loss = weight_dir * jnp.mean(branch_v_i + branch_v_j)

    # Start-state anchor loss & timeout boundary correction
    start_anchor_loss = weight_dir * jnp.mean(jnp.where(is_start, e_i ** 2, 0.0))
    timeout_corr_loss = -weight_dir * jnp.mean(jnp.where(is_timeout, e_j ** 2, 0.0))

    value_loss = magnitude_loss + laplacian_loss + start_anchor_loss + timeout_corr_loss + aux_mc_loss
    scaled_loss = config.get("VF_COEF", 0.5) * value_loss

    delta_mag = jnp.abs(targets_i - v_i).mean()
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "timeout_corr_loss": timeout_corr_loss,
        "aux_loss": aux_mc_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


def mc_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Monte Carlo return regression critic loss."""
    return ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config)


SPLIT_CRITIC_LOSS_FNS = {
    "fitted": ppo_critic_loss_fn,
    "ppo": ppo_critic_loss_fn,
    "td": td_0_critic_loss_fn,
    "td_0": td_0_critic_loss_fn,
    "td0": td_0_critic_loss_fn,
    "classic_td": td_0_critic_loss_fn,
    "td_lambda": ppo_critic_loss_fn,
    "sampled_td_lambda": ppo_critic_loss_fn,
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
}


def get_critic_loss_fn(critic_type: str, split: bool = True):
    """Returns the critic loss function matching the specified critic_type."""
    key = critic_type.lower()
    if key not in SPLIT_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown critic_type '{critic_type}'. Available: {list(SPLIT_CRITIC_LOSS_FNS.keys())}")
    return SPLIT_CRITIC_LOSS_FNS[key]


def e_lambda_geometric_critic_loss(*args, **kwargs):
    pass

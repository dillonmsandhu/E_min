import jax
import jax.numpy as jnp
from gymnax_rl.core.gymnax_utils import post_process_advantage


def calculate_gae(traj_batch, next_value, gamma: float, gae_lambda: float):
    """
    Computes Generalized Advantage Estimation (GAE) and value targets.
    Bootstraps through timeouts using true continuation next_value,
    and severs GAE accumulation at episode boundaries (done=True).
    """
    if next_value.ndim == 1:
        regular_next = jnp.concatenate([traj_batch.value[1:], next_value[None]], axis=0)
        next_val_arr = traj_batch.next_value if traj_batch.next_value is not None else regular_next
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


def calculate_targets(traj_batch, next_value, gamma: float, target_lambda: float):
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
    Returns the target lambda for main critic targets based on CRITIC_TYPE.
    Fails fast if CRITIC_TYPE is unrecognized or required lambda key is missing.
    """
    critic_type = config["CRITIC_TYPE"].lower()
    if critic_type in (
        "e_0",
        "e0",
        "e",
        "e_0_basic",
        "e_basic",
        "e_0_biased",
        "e_biased",
        "e_lambda",
        "elambda",
        "e_geometric",
        "sampled_e",
        "surgical_v_aux",
        "surgical",
    ):
        return config["RETURN_LAMBDA"]
    elif critic_type in ("td_0", "td", "td0", "classic_td"):
        return config["TD_LAMBDA"]
    elif critic_type in ("mc", "sampled_mc"):
        return 1.0
    elif critic_type in ("fitted", "ppo", "td_lambda", "sampled_td_lambda"):
        return config["VALUE_LAMBDA"]
    else:
        raise ValueError(f"Unrecognized CRITIC_TYPE '{config['CRITIC_TYPE']}'.")


def ppo_value_loss(value, targets, value_old=None, vf_clip=None, value_head_agg: str = "sum"):
    """PPO MSE value loss with optional value clipping and multi-head aggregation."""
    targets_exp = targets[..., None] if value.ndim > targets.ndim else targets
    value_losses = 0.5 * jnp.square(value - targets_exp)

    if vf_clip is not None and value_old is not None:
        if value_old.ndim < value.ndim:
            value_old = value_old[..., None]
        value_pred_clipped = value_old + jnp.clip(value - value_old, -vf_clip, vf_clip)
        value_losses_clipped = 0.5 * jnp.square(value_pred_clipped - targets_exp)
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
    gamma: float,
    vf_clip=None,
    value_old=None,
    value_head_agg: str = "mean",
):
    """
    Computes online 1-step TD(0) MSE loss using live next_value:
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
    value_losses = 0.5 * jnp.square(delta)

    if vf_clip is not None and value_old is not None:
        if value_old.ndim < value.ndim:
            value_old = value_old[:, None]
        value_pred_clipped = value_old + jnp.clip(value - value_old, -vf_clip, vf_clip)
        delta_clipped = td_target - value_pred_clipped
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
):
    """
    Pairs each source state s_t with its geometric jump destination s_j = s_{min(t+m, T)},
    evaluates boundary conditions in O(1), and packages jump transition attributes.
    Uses full-MC target replacement for invalid jumps (timeouts / buffer overshoots).
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
):
    """
    Attaches jump transition data for E(lambda) critics before PyTree shuffling.
    For fitted, TD, and E(0) critics, returns traj_batch unmodified.
    """
    key = critic_type.lower()
    if key in ("e_lambda", "elambda", "e_geometric"):
        return prepare_e_lambda_jump_pairs(
            traj_batch=traj_batch,
            returns=targets,
            last_val=last_val,
            gamma=gamma,
            lmbda=e_lambda,
            rng=rng,
            steps_since_start=steps_since_start,
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

    total_actor_loss = config["POLICY_COEFF"] * actor_loss - config["ENT_COEF"] * entropy
    return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}


def ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Fitted Value Iteration MSE critic loss."""
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)

    value_loss = ppo_value_loss(
        value=value,
        targets=targets,
        value_old=traj_batch.value,
        vf_clip=config["VF_CLIP"],
        value_head_agg=config["VALUE_HEAD_AGG"],
    )
    v_mean = value.mean(-1) if value.ndim > targets.ndim else value
    delta_mag = jnp.abs(targets - v_mean).mean()
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {"val_loss": value_loss, "delta_mag": delta_mag}


def td_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Online 1-step TD(0) MSE critic loss."""
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    next_value = network.apply(full_params, traj_batch.next_obs, method=network.value)

    value_loss, delta = td_0_value_loss(
        value=value,
        next_value=next_value,
        reward=traj_batch.reward,
        done=traj_batch.done,
        is_timeout=traj_batch.is_timeout,
        gamma=config["GAMMA"],
        vf_clip=config["VF_CLIP"],
        value_old=traj_batch.value,
        value_head_agg=config["VALUE_HEAD_AGG"],
    )
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {"val_loss": value_loss, "delta_mag": jnp.abs(delta).mean()}


def compute_aux_head_loss(v_aux, aux_targets, aux_coef: float = 1.0):
    """Pure MSE loss training the auxiliary head toward its (stop-gradient) targets."""
    err = jax.lax.stop_gradient(aux_targets) - v_aux
    return aux_coef * jnp.mean(0.5 * jnp.square(err))


def e_aux_backward_surrogate(e_phi_src, e_phi_dst, v_dst):
    """
    Backward-branch surrogate for the direct error head e_phi(s) ~ V(s) - v_theta(s).
    With d = sg(e_phi_src - e_phi_dst):
        value    = d^2
        gradient = 2 * d * grad v_theta(s_dst)
    which matches the gradient of (sg(e_src) - e_dst)^2 on the v_theta(s_dst) branch.
    Since e_phi does not depend on theta, no v_theta difference enters the backward
    branch, and an untrained (zero-initialized) e_phi gives exactly Forward-Only.
    """
    sg = jax.lax.stop_gradient
    d = sg(e_phi_src - e_phi_dst)
    return (sg(d - v_dst) + v_dst) ** 2


def e_lambda_geometric_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """
    Geometric jumping E(lambda) MSE critic loss.
    Dispatches explicitly on config['E_VARIANT']:
      - 'aux': surgical v_aux on backward branch + self-bootstrapped v_aux MSE loss
      - 'e_aux': direct error head e_phi(s) ~ V(s) - v_theta(s) on backward branch,
                 e_phi regressed onto G^{lambda_aux}_t - v_theta(s_t)
      - 'fwd': forward-only (e_start - stop_gradient(e_jump))^2 without v_aux
      - 'naive': symmetric (e_start - e_jump)^2 without v_aux
    Uses full-MC target replacement for invalid jumps (timeouts / buffer overshoots).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    gamma = config["GAMMA"]
    lmbda = config["E_LAMBDA"]
    variant = config["E_VARIANT"].lower()
    vf_coef = config["VF_COEF"]
    aux_coef = config["AUX_VF_COEF"]
    gl = gamma * lmbda

    targets_i = targets if targets is not None else traj_batch.targets

    if variant == "aux":
        v_start, v_aux_start = network.apply(full_params, traj_batch.obs, method=network.both)
        v_jump, v_aux_jump = network.apply(full_params, traj_batch.dest_obs, method=network.both)

        e_start = targets_i - v_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)
        e_aux_start = sg(v_aux_start) - v_start
        e_aux_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2
        aux_loss = compute_aux_head_loss(v_aux_start, traj_batch.aux_targets, aux_coef=aux_coef)
    elif variant == "e_aux":
        v_start, e_phi_start = network.apply(full_params, traj_batch.obs, method=network.both)
        v_jump, e_phi_jump = network.apply(full_params, traj_batch.dest_obs, method=network.both)

        e_start = targets_i - v_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)
        e_phi_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, e_phi_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + e_aux_backward_surrogate(e_phi_start, e_phi_jump, v_jump)
        # e_phi regresses onto the empirical error G^{lambda_aux}_t - v_theta(s_t) (live v_theta)
        aux_loss = compute_aux_head_loss(
            e_phi_start, traj_batch.aux_targets - sg(v_start), aux_coef=aux_coef
        )
    elif variant == "fwd":
        v_start = network.apply(full_params, traj_batch.obs, method=network.value)
        v_jump = network.apply(full_params, traj_batch.dest_obs, method=network.value)

        e_start = targets_i - v_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2
        aux_loss = 0.0
    elif variant == "naive":
        v_start = network.apply(full_params, traj_batch.obs, method=network.value)
        v_jump = network.apply(full_params, traj_batch.dest_obs, method=network.value)

        e_start = targets_i - v_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)

        inner_diff_sq = (e_start - e_jump) ** 2
        aux_loss = 0.0
    else:
        raise ValueError(f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive').")

    is_invalid = traj_batch.hit_timeout | traj_batch.hit_horizon
    diff_sq = jnp.where(
        traj_batch.hit_true_terminal | is_invalid,
        e_start ** 2,
        inner_diff_sq,
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
    total_loss = value_loss + aux_loss
    scaled_loss = vf_coef * total_loss

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
    1-sample E(0) critic loss without auxiliary critic v_aux.
    Dispatches explicitly on config['E_VARIANT']:
      - 'fwd': forward-only (e_i - stop_gradient(e_j))^2
      - 'naive' (or 'aux' when CRITIC_TYPE='e_0_basic'): symmetric (e_i - e_j)^2
    On true terminal or timeout states, falls back to full-weight MC anchor e_i^2.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    v_i = network.apply(full_params, traj_batch.obs, method=network.value)
    v_j = network.apply(full_params, traj_batch.next_obs, method=network.value)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout

    gamma = config["GAMMA"]
    variant = config["E_VARIANT"].lower()
    vf_coef = config["VF_COEF"]

    e_i = targets_i - v_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    if variant == "fwd":
        inner_diff_sq = (e_i - sg(e_j)) ** 2
    elif variant in ("naive", "aux"):
        inner_diff_sq = (e_i - e_j) ** 2
    else:
        raise ValueError(f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive').")

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
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "delta_mag": delta_mag,
    }
    return scaled_loss, aux_info


def e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Surgical auxiliary E(0) critic loss matching E(lambda=0).
    Dispatches explicitly on config['E_VARIANT']:
      - 'aux': surgical v_aux on backward branch + self-bootstrapped v_aux MSE loss
      - 'e_aux': direct error head e_phi(s) ~ V(s) - v_theta(s) on backward branch,
                 e_phi regressed onto G^{lambda_aux}_t - v_theta(s_t)
      - 'fwd' or 'naive': delegates to e_0_basic_critic_loss_fn
    """
    variant = config["E_VARIANT"].lower()
    if variant in ("naive", "fwd"):
        return e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config)
    elif variant not in ("aux", "e_aux"):
        raise ValueError(f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive').")

    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    v_i, aux_head_i = network.apply(full_params, traj_batch.obs, method=network.both)
    v_j, aux_head_j = network.apply(full_params, traj_batch.next_obs, method=network.both)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    aux_targets = traj_batch.aux_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout
    is_start = traj_batch.is_start

    gamma = config["GAMMA"]
    aux_coef = config["AUX_VF_COEF"]
    vf_coef = config["VF_COEF"]

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
    aux_info = {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
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

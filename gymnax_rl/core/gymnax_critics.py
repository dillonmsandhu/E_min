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
        "e_0_q",
        "e0",
        "e",
        "e_0_basic",
        "e_basic",
        "e_0_biased",
        "e_biased",
        "e_lambda",
        "e_lambda_q",
        "e_lambda_v_aux",
        "elambda",
        "e_geometric",
        "e_0_v_aux",
        "e_0_bilinear",
        "e_0_dir_marg",
        "bsmc",
        "bsmc_v_aux",
        "bsmc_double_pi",
        "bsmc_bilinear_v_aux",
        "bsmc_dir_marg",
        "bsmc_dir_double",
        "sampled_e",
        "surgical_v_aux",
        "surgical",
        "e_inner",
        "e_bilinear",
        "e_double",
        "e_dirichlet_double",
        "e_dir_double",
        "double_e_dirichlet",
        "e_min",
    ):
        return config["RETURN_LAMBDA"]
    elif critic_type in ("td_0", "td", "td0", "classic_td", "sarsa_0", "msbe", "double_msbe", "mspbe", "double_mspbe", "online_td0", "semi_gradient_td0"):
        return config.get("TD_LAMBDA", 0.0)
    elif critic_type in ("mc", "sampled_mc"):
        return 1.0
    elif critic_type in ("fitted_td0",):
        return 0.0
    elif critic_type in ("fitted_gae",):
        return config.get("VALUE_LAMBDA", config.get("GAE_LAMBDA", 0.95))
    elif critic_type in ("fitted", "ppo", "sarsa", "sarsa_lambda", "td_lambda", "sampled_td_lambda"):
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
        rot_omega = config["ROT_OMEGA"]

        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + rot_omega * (sg(e_aux_start) - e_aux_jump) ** 2
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


def _one_step_e_basic_critic_loss_fn(
    critic_params, actor_params, network, traj_batch, targets, config, weight_mc: float, weight_dir: float
):
    """Shared 1-sample E(0) / BSMC critic loss without auxiliary critic v_aux."""
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


def _one_step_e_critic_loss_fn(
    critic_params, actor_params, network, traj_batch, targets, config, weight_mc: float, weight_dir: float
):
    """Shared 1-step E(0) / BSMC critic loss with surgical v_aux or basic fallback."""
    variant = config["E_VARIANT"].lower()
    if variant in ("naive", "fwd"):
        return _one_step_e_basic_critic_loss_fn(
            critic_params, actor_params, network, traj_batch, targets, config, weight_mc, weight_dir
        )
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

    start_weight = 0.25 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)

    branch_v_i = (e_i - sg(e_j)) ** 2
    branch_v_j = jnp.where(true_terminal, 0.0, backward_sq)
    rot_omega = config["ROT_OMEGA"]
    diff_sq = jnp.where(
        true_terminal | is_timeout,
        e_i ** 2,
        branch_v_i + rot_omega * branch_v_j,
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


def e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """1-sample E(0) critic loss without auxiliary critic v_aux."""
    gamma = config["GAMMA"]
    return _one_step_e_basic_critic_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config,
        weight_mc=0.5 * (1.0 - gamma), weight_dir=0.5 * gamma,
    )


def e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Surgical auxiliary E(0) critic loss matching E(lambda=0)."""
    gamma = config["GAMMA"]
    return _one_step_e_critic_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config,
        weight_mc=0.5 * (1.0 - gamma), weight_dir=0.5 * gamma,
    )


def bsmc_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """1-step Bellman-Smoothed Monte Carlo (BSMC) critic loss with surgical v_aux."""
    gamma = config["GAMMA"]
    bsmc_lambda = config["BSMC_LAMBDA"]
    denom = jnp.maximum(1.0 - gamma * bsmc_lambda, 1e-8)
    weight_mc = 0.5 * (1.0 - gamma) / denom
    weight_dir = 0.5 * (gamma * (1.0 - bsmc_lambda)) / denom
    return _one_step_e_critic_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config,
        weight_mc=weight_mc, weight_dir=weight_dir,
    )


def mc_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """Monte Carlo return regression critic loss."""
    return ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config)


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
    vf_coef = config.get("VF_COEF", 0.5)

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

    scaled_loss = vf_coef * value_loss
    return scaled_loss, {
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
    vf_coef = config.get("VF_COEF", 0.5)

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

    scaled_loss = vf_coef * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta_1)),
        "delta2_mag": jnp.mean(jnp.abs(delta_2)),
        "e_mag": jnp.mean(jnp.abs(e_hat_1)),
        "diagnostic_msbe": diagnostic_msbe,
    }


def _double_sample_dirichlet_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
    weight_mc: float,
    weight_dir: float,
):
    """
    Shared double-sampled Dirichlet E(0) / BSMC critic loss using branch 1
    (obs, next_obs, targets, next_targets) and independent 1-step probe branch 2
    (probe_next_obs) with no v_aux:
        L(theta) = weight_mc * (e_i^(1))^2
                 + 0.5 * weight_dir * (e_i^(1) - sg(e_j^(1)))^2
                 + weight_dir * [ sg(e_i^(1)) * b2 * v(s_j^(2)) - sg(e_j^(1)) * b1 * v(s_j^(1)) ]
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    gamma = config["GAMMA"]
    vf_coef = config.get("VF_COEF", 0.5)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets

    v_i = network.apply(full_params, traj_batch.obs, method=network.value)
    v_j_1 = network.apply(full_params, traj_batch.next_obs, method=network.value)
    v_j_2 = network.apply(full_params, traj_batch.probe_next_obs, method=network.value)

    true_term_1 = traj_batch.done & ~traj_batch.is_timeout
    true_term_2 = traj_batch.probe_done & ~traj_batch.probe_is_timeout

    if v_i.ndim > traj_batch.reward.ndim:
        reward_1 = traj_batch.reward[..., None]
        reward_2 = traj_batch.probe_reward[..., None]
        true_term_1 = true_term_1[..., None]
        true_term_2 = true_term_2[..., None]
        targets_i = targets_i[..., None]
        targets_j = targets_j[..., None]
    else:
        reward_1 = traj_batch.reward
        reward_2 = traj_batch.probe_reward

    bootstrap_1 = 1.0 - true_term_1
    bootstrap_2 = 1.0 - true_term_2

    e_i = targets_i - v_i
    e_j = jnp.where(true_term_1, 0.0, targets_j - v_j_1)

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    forward_dir = jnp.where(
        true_term_1,
        weight_dir * (e_i ** 2),
        0.5 * weight_dir * ((e_i - sg(e_j)) ** 2),
    )
    backward_dir = weight_dir * (
        sg(e_i) * bootstrap_2 * v_j_2 - sg(e_j) * bootstrap_1 * v_j_1
    )
    dirichlet_loss = jnp.mean(forward_dir + backward_dir)

    value_loss = magnitude_loss + dirichlet_loss
    scaled_loss = vf_coef * value_loss

    delta_1 = reward_1 + gamma * bootstrap_1 * v_j_1 - v_i
    delta_2 = reward_2 + gamma * bootstrap_2 * v_j_2 - v_i
    diagnostic_msbe = jnp.mean(delta_1 * delta_2)

    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "delta_mag": jnp.mean(jnp.abs(delta_1)),
        "delta2_mag": jnp.mean(jnp.abs(delta_2)),
        "e_mag": jnp.mean(jnp.abs(e_i)),
        "diagnostic_msbe": diagnostic_msbe,
    }


def double_sample_e_dirichlet_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """Double-sampled Dirichlet E(0) critic loss (0.5 * E(0) scale)."""
    gamma = config["GAMMA"]
    return _double_sample_dirichlet_loss_fn(
        critic_params,
        actor_params,
        network,
        traj_batch,
        targets,
        config,
        weight_mc=0.5 * (1.0 - gamma),
        weight_dir=0.5 * gamma,
    )


def double_sample_bsmc_dirichlet_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
):
    """Double-sampled Dirichlet BSMC critic loss (MC-reweighted 1-step Dirichlet, no v_aux)."""
    gamma = config["GAMMA"]
    bsmc_lambda = config["BSMC_LAMBDA"]
    denom = jnp.maximum(1.0 - gamma * bsmc_lambda, 1e-8)
    weight_mc = 0.5 * (1.0 - gamma) / denom
    weight_dir = 0.5 * (gamma * (1.0 - bsmc_lambda)) / denom
    return _double_sample_dirichlet_loss_fn(
        critic_params,
        actor_params,
        network,
        traj_batch,
        targets,
        config,
        weight_mc=weight_mc,
        weight_dir=weight_dir,
    )


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
    vf_coef = config.get("VF_COEF", 0.5)
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
    term2 = r_perp * jvp_live

    value_loss = jnp.mean(term1 + term2)
    diagnostic_msbe = jnp.mean(delta_1 * delta_2)
    diagnostic_mspbe = jnp.mean(delta_proj ** 2)

    scaled_loss = vf_coef * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta_1)),
        "delta2_mag": jnp.mean(jnp.abs(delta_2)),
        "delta_proj_mag": jnp.mean(jnp.abs(delta_proj)),
        "r_perp_mag": jnp.mean(jnp.abs(r_perp)),
        "diagnostic_msbe": diagnostic_msbe,
        "diagnostic_mspbe": diagnostic_mspbe,
    }


def compute_lambda_trace_rotation_ratio(
    critic_params,
    actor_params,
    network,
    traj_batch,
    gamma: float,
    lmbda: float,
    n_probe_envs: int = 16,
):
    """
    Computes online Rayleigh-quotient rotation statistics (alpha_eff, beta_eff)
    for TD(lambda) / E(lambda) along the unit TD(lambda) semi-gradient direction u
    using 1 JVP + 1 VJP and exact forward/backward lambda-traces over the rollout.

    For lambda = 0, this reduces identically to TD(0) / E(0).
    For lambda = 1, j_fwd = j_bwd = 0 so beta_eff = 0 identically.
    Zero v_aux dependency, zero double-sampling bias.
    """
    from jax.flatten_util import ravel_pytree

    T, B = traj_batch.reward.shape[:2]
    E_p = min(int(n_probe_envs), B)

    obs_p = traj_batch.obs[:, :E_p]          # (T, E_p, *obs_shape)
    next_obs_last = traj_batch.next_obs[-1, :E_p]  # (E_p, *obs_shape)
    rew_p = traj_batch.reward[:, :E_p]       # (T, E_p)
    done_p = traj_batch.done[:, :E_p]        # (T, E_p)
    timeout_p = traj_batch.is_timeout[:, :E_p]
    true_term_p = done_p & ~timeout_p
    cont_p = (~done_p).astype(jnp.float32)   # resets trace at both true_terminal and timeout

    obs_shape = obs_p.shape[2:]
    obs_flat = obs_p.reshape((T * E_p,) + obs_shape)
    next_obs_p = traj_batch.next_obs[:, :E_p]
    next_obs_flat = next_obs_p.reshape((T * E_p,) + obs_shape)

    def v_flat_fn(c_params, x_flat):
        full_p = {"params": {**actor_params, **c_params}}
        v = network.apply(full_p, x_flat, method=network.value)
        return v.mean(-1) if v.ndim > 1 else v

    # 1. Unit TD(lambda) semi-gradient direction u w.r.t. critic_params
    def td_probe_loss(c_params):
        v_s = v_flat_fn(c_params, obs_flat).reshape((T, E_p))
        if lmbda <= 1e-8:
            v_sp = v_flat_fn(c_params, next_obs_flat).reshape((T, E_p))
            target = jax.lax.stop_gradient(rew_p + gamma * (~true_term_p).astype(jnp.float32) * v_sp)
        else:
            # Compute lambda-return targets directly from rollout rewards & current values
            v_last = v_flat_fn(c_params, next_obs_last)
            v_next_reg = jnp.concatenate([v_s[1:], v_last[None]], axis=0)
            v_next_to = v_flat_fn(c_params, next_obs_flat).reshape((T, E_p))
            v_next = jnp.where(timeout_p, v_next_to, v_next_reg)
            deltas = rew_p + gamma * (~true_term_p).astype(jnp.float32) * v_next - v_s

            def _gae_scan(gae, step_in):
                d_t, c_t = step_in
                gae = d_t + gamma * lmbda * c_t * gae
                return gae, gae

            _, advs = jax.lax.scan(_gae_scan, jnp.zeros((E_p,), dtype=jnp.float32), (deltas, cont_p), reverse=True)
            target = jax.lax.stop_gradient(advs + v_s)
        return 0.5 * jnp.mean((v_s - target) ** 2)

    g_td = jax.grad(td_probe_loss)(critic_params)
    g_flat, unravel = ravel_pytree(g_td)
    u_tree = unravel(g_flat / jnp.maximum(jnp.linalg.norm(g_flat), 1e-12))

    # 2. Single JVP over rollout states + final next_obs
    _, j_flat = jax.jvp(lambda cp: v_flat_fn(cp, obs_flat), (critic_params,), (u_tree,))
    _, j_last = jax.jvp(lambda cp: v_flat_fn(cp, next_obs_last), (critic_params,), (u_tree,))
    j = j_flat.reshape((T, E_p))

    # 3. Exact future (backward scan) and past (forward scan) lambda-traces
    #    j_fwd[t] = cont[t] * (gamma * (1 - lmbda) * j[t+1] + gamma * lmbda * j_fwd[t+1])
    #    j_bwd[t] = cont[t-1] * (gamma * (1 - lmbda) * j[t-1] + gamma * lmbda * j_bwd[t-1])
    j_next_arr = jnp.concatenate([j[1:], j_last[None]], axis=0)

    def _fwd_trace_step(trace_next, step_in):
        j_tp1, c_t = step_in
        trace_t = c_t * (gamma * (1.0 - lmbda) * j_tp1 + gamma * lmbda * trace_next)
        return trace_t, trace_t

    _, j_fwd = jax.lax.scan(
        _fwd_trace_step,
        jnp.zeros((E_p,), dtype=jnp.float32),
        (j_next_arr, cont_p),
        reverse=True,
    )

    def _bwd_trace_step(trace_prev, step_in):
        j_t, c_t = step_in
        # trace_prev is j_bwd[t]; compute j_bwd[t+1] for the next step
        trace_next = c_t * (gamma * (1.0 - lmbda) * j_t + gamma * lmbda * trace_prev)
        return trace_next, trace_prev

    _, j_bwd = jax.lax.scan(
        _bwd_trace_step,
        jnp.zeros((E_p,), dtype=jnp.float32),
        (j, cont_p),
        reverse=False,
    )

    # 4. alpha_eff = u^T S_{J,lambda} u; beta_eff = ||K_{J,lambda} u||_2 via 1 VJP
    alpha_eff = jnp.mean(j * j - j * j_fwd)

    _, vjp_fn = jax.vjp(lambda cp: v_flat_fn(cp, obs_flat), critic_params)
    w_k = (0.5 / float(T * E_p)) * (j_bwd - j_fwd).reshape((T * E_p,))
    ku_tree = vjp_fn(w_k)[0]
    ku_flat, _ = ravel_pytree(ku_tree)
    beta_eff = jnp.linalg.norm(ku_flat)
    return alpha_eff, beta_eff


SPLIT_CRITIC_LOSS_FNS = {
    "fitted": ppo_critic_loss_fn,
    "fitted_td0": ppo_critic_loss_fn,
    "fitted_gae": ppo_critic_loss_fn,
    "ppo": ppo_critic_loss_fn,
    "td": td_0_critic_loss_fn,
    "td_0": td_0_critic_loss_fn,
    "td0": td_0_critic_loss_fn,
    "online_td0": td_0_critic_loss_fn,
    "semi_gradient_td0": td_0_critic_loss_fn,
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
    "bsmc": bsmc_critic_loss_fn,
    "bsmc_v_aux": bsmc_critic_loss_fn,
    "mc": mc_critic_loss_fn,
    "sampled_mc": mc_critic_loss_fn,
    "msbe": double_sample_msbe_critic_loss_fn,
    "double_msbe": double_sample_msbe_critic_loss_fn,
    "e_inner": double_sample_e_inner_critic_loss_fn,
    "e_bilinear": double_sample_e_inner_critic_loss_fn,
    "e_double": double_sample_e_inner_critic_loss_fn,
    "e_dirichlet_double": double_sample_e_dirichlet_critic_loss_fn,
    "e_dir_double": double_sample_e_dirichlet_critic_loss_fn,
    "double_e_dirichlet": double_sample_e_dirichlet_critic_loss_fn,
    "bsmc_dir_double": double_sample_bsmc_dirichlet_critic_loss_fn,
    "e_min": double_sample_e_inner_critic_loss_fn,
    "mspbe": double_sample_mspbe_critic_loss_fn,
    "double_mspbe": double_sample_mspbe_critic_loss_fn,
}


def get_critic_loss_fn(critic_type: str, split: bool = True):
    """Returns the critic loss function matching the specified critic_type."""
    key = critic_type.lower()
    if key not in SPLIT_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown critic_type '{critic_type}'. Available: {list(SPLIT_CRITIC_LOSS_FNS.keys())}")
    return SPLIT_CRITIC_LOSS_FNS[key]

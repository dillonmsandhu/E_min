# brax_q_critics.py
# Continuous On-Policy DDPG Actor and Q-Critic Loss Functions for Brax environments:
#   - Actor: Deterministic Policy Gradient (-E_s[Q(s, mu(s))])
#   - Critics:
#       1. Expected SARSA(lambda) / Fitted Q regression
#       2. Online 1-step Expected SARSA(0) / TD(0)
#       3. E(0) State-Action Dirichlet Minimization ('naive', 'fwd', 'aux', 'e_aux')
#       4. E(lambda) Geometric Jump State-Action Dirichlet Minimization ('naive', 'fwd', 'aux', 'e_aux')
from __future__ import annotations
import jax
import jax.numpy as jnp

from brax_rl.core.brax_critics import (
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    prepare_critic_transitions,
)


def compute_aux_head_loss(q_aux, aux_targets, aux_coef: float = 1.0):
    """Pure MSE loss training the auxiliary critic head toward its (stop-gradient) targets."""
    err = jax.lax.stop_gradient(aux_targets) - q_aux
    return aux_coef * jnp.mean(0.5 * jnp.square(err))


def e_aux_backward_surrogate(e_phi_src, e_phi_dst, v_dst):
    """Backward-branch surrogate for direct error head e_phi ~ Q^pi - Q_theta."""
    sg = jax.lax.stop_gradient
    d = sg(e_phi_src - e_phi_dst)
    return (sg(d - v_dst) + v_dst) ** 2


def ddpg_actor_loss_fn(actor_params, critic_params, network, traj_batch, config):
    """
    Deterministic Policy Gradient (DDPG) actor loss for Brax:
        L_actor(phi) = -E_{s_t}[ Q_theta(s_t, mu_phi(s_t)) ] + reg_coef * 0.5 * E[ ||mu_phi(s_t) - mu_old(s_t)||^2 ]
    Differentiates strictly w.r.t. actor_params while critic_params (theta) are held fixed.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    mu = network.apply(full_params, traj_batch.obs, method=network.actor)
    q_pi = network.apply(full_params, traj_batch.obs, mu, method=network.q_value)
    actor_loss = -jnp.mean(q_pi)

    reg_coef = config["ACTION_REG_COEF"]
    action_reg = 0.5 * jnp.mean(jnp.sum(jnp.square(mu - traj_batch.mu), axis=-1))

    pi = network.apply(full_params, traj_batch.obs, method=network.policy)
    entropy = pi.entropy().mean()

    total_actor_loss = actor_loss + reg_coef * action_reg
    return total_actor_loss, {
        "actor_loss": actor_loss,
        "entropy": entropy,
        "q_pi": jnp.mean(q_pi),
    }


def ddpg_sarsa_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Fitted Expected SARSA(lambda) MSE critic loss on Q(s_t, a_t).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_taken = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
    targets_i = targets if targets is not None else traj_batch.targets

    value_loss = 0.5 * jnp.mean(jnp.square(targets_i - q_taken))
    delta_mag = jnp.mean(jnp.abs(targets_i - q_taken))
    return value_loss, {
        "val_loss": value_loss,
        "delta_mag": delta_mag,
    }


def ddpg_td_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Online 1-step Expected SARSA(0) / TD(0) MSE critic loss:
        target = stop_gradient(r_t + gamma * (1 - true_terminal_t) * Q_theta(s_{t+1}, mu_phi(s_{t+1})))
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_taken = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
    next_v = network.apply(full_params, traj_batch.next_obs, method=network.value)

    true_terminal = (traj_batch.done > 0.5) & ~(traj_batch.is_timeout > 0.5)
    gamma = config["GAMMA"]

    td_target = jax.lax.stop_gradient(traj_batch.reward + gamma * (1.0 - true_terminal) * next_v)
    delta = td_target - q_taken
    value_loss = 0.5 * jnp.mean(jnp.square(delta))
    return value_loss, {
        "val_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta)),
    }


def ddpg_e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    1-step E(0) Q-critic loss without auxiliary critic head:
        e_i = G_t - Q_theta(s_t, a_t)
        e_j = G_{t+1} - Q_theta(s_{t+1}, mu_phi(s_{t+1}))
    In deterministic Brax physics, (s_t, a_t) -> s_{t+1} -> mu_phi(s_{t+1}) has zero transition variance,
    so the symmetric ('naive') Dirichlet loss has zero double-sampling bias.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    q_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
    v_j = network.apply(full_params, traj_batch.next_obs, method=network.value)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout > 0.5
    true_terminal = (traj_batch.done > 0.5) & ~is_timeout

    gamma = config["GAMMA"]
    variant = config["E_VARIANT"].lower()

    e_i = targets_i - q_i
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
    scaled_loss = config["VF_COEF"] * value_loss

    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "delta_mag": jnp.mean(jnp.abs(e_i)),
    }


def ddpg_e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    E(0) Q-critic loss supporting 'naive', 'fwd', 'aux', and 'e_aux' variants.
    """
    variant = config["E_VARIANT"].lower()
    if variant in ("naive", "fwd"):
        return ddpg_e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config)
    elif variant not in ("aux", "e_aux"):
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive')."
        )

    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    q_i, aux_head_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.both_q)
    v_j, aux_head_j = network.apply(full_params, traj_batch.next_obs, method=network.both)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    aux_targets = traj_batch.aux_targets
    is_timeout = traj_batch.is_timeout > 0.5
    true_terminal = (traj_batch.done > 0.5) & ~is_timeout
    is_start = traj_batch.is_start

    gamma = config["GAMMA"]
    aux_coef = config["AUX_VF_COEF"]

    e_i = targets_i - q_i
    e_j = jnp.where(true_terminal, 0.0, targets_j - v_j)

    if variant == "aux":
        aux_mc_loss = compute_aux_head_loss(aux_head_i, aux_targets, aux_coef=aux_coef)
        e_tilde_i = sg(aux_head_i) - q_i
        e_tilde_j = jnp.where(true_terminal, 0.0, sg(aux_head_j) - v_j)
        backward_sq = (sg(e_tilde_i) - e_tilde_j) ** 2
    else:  # e_aux
        aux_mc_loss = compute_aux_head_loss(aux_head_i, aux_targets - sg(q_i), aux_coef=aux_coef)
        e_phi_j = jnp.where(true_terminal, 0.0, aux_head_j)
        backward_sq = e_aux_backward_surrogate(aux_head_i, e_phi_j, v_j)

    weight_mc = 0.5 * (1.0 - gamma)
    weight_dir = 0.5 * gamma
    start_weight = 0.25 * gamma

    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    branch_q_i = (e_i - sg(e_j)) ** 2
    branch_v_j = jnp.where(true_terminal, 0.0, backward_sq)
    diff_sq = jnp.where(
        true_terminal | is_timeout,
        e_i ** 2,
        branch_q_i + branch_v_j,
    )
    laplacian_loss = weight_dir * jnp.mean(diff_sq)
    start_anchor_loss = start_weight * jnp.mean(jnp.where(is_start, e_i ** 2, 0.0))

    value_loss = magnitude_loss + laplacian_loss + start_anchor_loss + aux_mc_loss
    scaled_loss = config["VF_COEF"] * value_loss

    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "aux_mc_loss": aux_mc_loss,
        "delta_mag": jnp.mean(jnp.abs(e_i)),
    }


def ddpg_e_lambda_geometric_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config):
    """
    Geometric jumping E(lambda) Q-critic loss for Brax:
        e_i    = G_t - Q_theta(s_t, a_t)
        e_jump = G_j - Q_theta(s_j, mu_phi(s_j))
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
    aux_coef = config["AUX_VF_COEF"]
    gl = gamma * lmbda

    targets_i = targets if targets is not None else traj_batch.targets

    if variant == "aux":
        q_i, q_aux_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.both_q)
        v_jump, v_aux_jump = network.apply(full_params, obs_jump, method=network.both)

        e_i = targets_i - q_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        e_aux_i = sg(q_aux_i) - q_i
        e_aux_jump = jnp.where(is_absorbing, 0.0, sg(v_aux_jump) - v_jump)

        inner_diff_sq = (e_i - sg(e_jump)) ** 2 + (sg(e_aux_i) - e_aux_jump) ** 2
        aux_loss = compute_aux_head_loss(q_aux_i, traj_batch.aux_targets, aux_coef=aux_coef)
    elif variant == "e_aux":
        q_i, e_phi_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.both_q)
        v_jump, e_phi_jump = network.apply(full_params, obs_jump, method=network.both)

        e_i = targets_i - q_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        e_phi_jump = jnp.where(is_absorbing, 0.0, e_phi_jump)

        inner_diff_sq = (e_i - sg(e_jump)) ** 2 + e_aux_backward_surrogate(e_phi_i, e_phi_jump, v_jump)
        aux_loss = compute_aux_head_loss(e_phi_i, traj_batch.aux_targets - sg(q_i), aux_coef=aux_coef)
    elif variant == "fwd":
        q_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
        v_jump = network.apply(full_params, obs_jump, method=network.value)

        e_i = targets_i - q_i
        e_jump = jnp.where(is_absorbing, 0.0, targets_jump - v_jump)
        inner_diff_sq = (e_i - sg(e_jump)) ** 2
        aux_loss = 0.0
    elif variant == "naive":
        q_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
        v_jump = network.apply(full_params, obs_jump, method=network.value)

        e_i = targets_i - q_i
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

    return value_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "aux_mc_loss": aux_loss,
        "delta_mag": jnp.mean(jnp.abs(e_i)),
    }


BRAX_DDPG_CRITIC_LOSS_FNS = {
    "sarsa": ddpg_sarsa_critic_loss_fn,
    "sarsa_lambda": ddpg_sarsa_critic_loss_fn,
    "fitted": ddpg_sarsa_critic_loss_fn,
    "ppo": ddpg_sarsa_critic_loss_fn,
    "td_lambda": ddpg_sarsa_critic_loss_fn,
    "mc": ddpg_sarsa_critic_loss_fn,
    "sampled_mc": ddpg_sarsa_critic_loss_fn,
    "td": ddpg_td_0_critic_loss_fn,
    "td_0": ddpg_td_0_critic_loss_fn,
    "td0": ddpg_td_0_critic_loss_fn,
    "sarsa_0": ddpg_td_0_critic_loss_fn,
    "e_0": ddpg_e_0_critic_loss_fn,
    "e_0_q": ddpg_e_0_critic_loss_fn,
    "e0": ddpg_e_0_critic_loss_fn,
    "e": ddpg_e_0_critic_loss_fn,
    "sampled_e": ddpg_e_0_critic_loss_fn,
    "surgical_v_aux": ddpg_e_0_critic_loss_fn,
    "surgical": ddpg_e_0_critic_loss_fn,
    "e_0_basic": ddpg_e_0_basic_critic_loss_fn,
    "e_basic": ddpg_e_0_basic_critic_loss_fn,
    "e_0_biased": ddpg_e_0_basic_critic_loss_fn,
    "e_biased": ddpg_e_0_basic_critic_loss_fn,
    "e_lambda": ddpg_e_lambda_geometric_critic_loss_fn,
    "e_lambda_q": ddpg_e_lambda_geometric_critic_loss_fn,
    "elambda": ddpg_e_lambda_geometric_critic_loss_fn,
    "e_geometric": ddpg_e_lambda_geometric_critic_loss_fn,
}


def get_brax_ddpg_critic_loss_fn(critic_type: str):
    key = critic_type.lower()
    if key not in BRAX_DDPG_CRITIC_LOSS_FNS:
        raise ValueError(
            f"Unknown Brax DDPG critic_type '{critic_type}'. Available: {list(BRAX_DDPG_CRITIC_LOSS_FNS.keys())}"
        )
    return BRAX_DDPG_CRITIC_LOSS_FNS[key]

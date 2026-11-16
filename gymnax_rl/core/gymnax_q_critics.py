# gymnax_q_critics.py
# Q-function critic and actor losses for:
# 1. Discrete Q-PPO (SARSA(lambda), marginalized bilinear E(0), sampled E(lambda))
# 2. Continuous On-Policy DDPG (Expected SARSA(lambda), Expected SARSA(0), E(0)-Q, E(lambda)-Q + DPG Actor)
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any

from gymnax_rl.core.gymnax_critics import (
    compute_aux_head_loss,
    e_aux_backward_surrogate,
)
from gymnax_rl.algos.sac import sampled_soft_e_loss


def extract_q_taken(q_all: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """Extracts Q(s, a) from discrete Q(s, ·) tensor of shape (..., |A|)."""
    return jnp.take_along_axis(q_all, action[..., None], axis=-1).squeeze(-1)


def _eval_q_taken(full_params, network, obs, action) -> jnp.ndarray:
    """Evaluates Q(s, a) for either discrete (q_values) or continuous (q_value) networks."""
    if hasattr(network, "q_values"):
        q_all = network.apply(full_params, obs, method=network.q_values)
        return extract_q_taken(q_all, action)
    if hasattr(network, "q_value"):
        return network.apply(full_params, obs, action, method=network.q_value)
    raise AttributeError(f"Network {type(network)} must implement 'q_values' or 'q_value'.")


# ==============================================================================
# Q-PPO Critic Losses (Discrete & Continuous)
# ==============================================================================

def sarsa_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Fitted on-policy SARSA(lambda) regression loss for Q(s_t, a_t):
        L = 0.5 * (targets - Q(s_t, a_t))^2
    targets are lambda-returns G_t.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_taken = _eval_q_taken(full_params, network, traj_batch.obs, traj_batch.action)

    targets_i = targets if targets is not None else traj_batch.targets
    loss = 0.5 * jnp.mean((targets_i - q_taken) ** 2)
    delta_mag = jnp.abs(targets_i - q_taken).mean()
    scaled_loss = config["VF_COEF"] * loss
    return scaled_loss, {"val_loss": loss, "delta_mag": delta_mag}


def _q_ppo_bilinear_e0_core(critic_params, actor_params, network, traj_batch, targets, config):
    """Shared bilinear E(0) and MC loss computation for discrete and continuous Gymnax Q-PPO."""
    full_params = {"params": {**actor_params, **critic_params}}
    q_i = _eval_q_taken(full_params, network, traj_batch.obs, traj_batch.action)
    targets_i = jax.lax.stop_gradient(targets if targets is not None else traj_batch.targets)
    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    gamma = config["GAMMA"]

    if hasattr(network, "q_values"):
        q_all_j = network.apply(full_params, traj_batch.next_obs, method=network.q_values)
        pi_j = network.apply(full_params, traj_batch.next_obs, method=network.policy)
        probs_j = jax.lax.stop_gradient(pi_j.probs)
        v_j = jnp.sum(probs_j * q_all_j, axis=-1)
        e_i = targets_i - q_i
        delta = traj_batch.reward + gamma * jnp.where(true_terminal, 0.0, v_j) - q_i
        e0_loss = 0.5 * jnp.mean(e_i * delta)
    elif hasattr(network, "q_value"):
        if traj_batch.next_probe_actions is None:
            raise ValueError("Continuous E(0) Q-critic loss requires traj_batch.next_probe_actions.")
        probe_actions = jax.lax.stop_gradient(traj_batch.next_probe_actions)
        probe_q_all = jax.vmap(
            lambda a: network.apply(full_params, traj_batch.next_obs, a, method=network.q_value),
            in_axes=-2,
            out_axes=-1,
        )(probe_actions)
        v_j = jnp.mean(probe_q_all, axis=-1)
        e0_loss = sampled_soft_e_loss(
            q_taken=q_i,
            probe_q=v_j,
            probe_logp=jnp.zeros_like(v_j),
            reward=traj_batch.reward,
            done=traj_batch.done,
            is_timeout=traj_batch.is_timeout,
            returns=targets_i,
            gamma=gamma,
            alpha=0.0,
        )
        e_i = targets_i - q_i
        delta = traj_batch.reward + gamma * jnp.where(true_terminal, 0.0, v_j) - q_i
    else:
        raise AttributeError(f"Network {type(network)} must implement 'q_values' or 'q_value'.")

    mc_loss = 0.5 * jnp.mean(e_i ** 2)
    return e0_loss, mc_loss, e_i, delta, gamma


def e_0_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Half the bilinear E(0) estimate (discrete exact marginalization / continuous probe actions)."""
    e0_loss, _, e_i, delta, _ = _q_ppo_bilinear_e0_core(
        critic_params, actor_params, network, traj_batch, targets, config
    )
    scaled_loss = config["VF_COEF"] * e0_loss
    return scaled_loss, {
        "val_loss": e0_loss,
        "delta_mag": jnp.abs(e_i).mean(),
        "sampled_value_error_sq": jnp.mean(e_i ** 2),
        "bellman_residual_sq": jnp.mean(delta ** 2),
    }


def bsmc_double_pi_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Bilinear BSMC Q-critic loss using successor marginalization (discrete) or probe actions (continuous)."""
    e0_loss, mc_loss, e_i, delta, gamma = _q_ppo_bilinear_e0_core(
        critic_params, actor_params, network, traj_batch, targets, config
    )
    bsmc_lambda = config["BSMC_LAMBDA"]
    w_extra = ((1.0 - gamma) * bsmc_lambda) / jnp.maximum(1.0 - gamma * bsmc_lambda, 1e-8)
    value_loss = (1.0 - w_extra) * e0_loss + w_extra * mc_loss
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "e0_loss": e0_loss,
        "mc_loss": mc_loss,
        "delta_mag": jnp.abs(e_i).mean(),
        "bellman_residual_sq": jnp.mean(delta ** 2),
    }


def bsmc_bilinear_v_aux_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """1-step bilinear BSMC Q-critic loss debiased via state-only v_aux(s) without double-sampling pi."""
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    q_i = _eval_q_taken(full_params, network, traj_batch.obs, traj_batch.action)
    q_j = _eval_q_taken(full_params, network, traj_batch.next_obs, traj_batch.next_action)
    v_aux_i = network.apply(full_params, traj_batch.obs, method=network.v_aux)
    v_aux_next = network.apply(full_params, traj_batch.next_obs, method=network.v_aux)

    targets_i = sg(targets if targets is not None else traj_batch.targets)
    aux_targets = sg(traj_batch.aux_targets)
    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    gamma = config["GAMMA"]
    aux_coef = config["AUX_VF_COEF"]
    bsmc_lambda = config["BSMC_LAMBDA"]

    e_i = targets_i - q_i
    e_proxy_i = sg(traj_batch.reward + gamma * jnp.where(true_terminal, 0.0, v_aux_next) - q_i)
    td_target_stop = sg(traj_batch.reward + gamma * jnp.where(true_terminal, 0.0, q_j))

    cur_branch = 0.5 * e_i * (td_target_stop - q_i)
    next_branch = 0.5 * gamma * jnp.where(true_terminal, 0.0, e_proxy_i * q_j)
    e0_bilin_vaux = jnp.mean(cur_branch + next_branch)

    mc_loss = 0.5 * jnp.mean(e_i ** 2)
    w_extra = ((1.0 - gamma) * bsmc_lambda) / jnp.maximum(1.0 - gamma * bsmc_lambda, 1e-8)
    aux_mc_loss = compute_aux_head_loss(v_aux_i, aux_targets, aux_coef=aux_coef)

    value_loss = (1.0 - w_extra) * e0_bilin_vaux + w_extra * mc_loss + aux_mc_loss
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "e0_bilin_vaux": e0_bilin_vaux,
        "mc_loss": mc_loss,
        "aux_loss": aux_mc_loss,
        "delta_mag": jnp.abs(e_i).mean(),
    }


def _q_dirichlet_v_aux_loss_fn(
    critic_params, actor_params, network, traj_batch, targets, config, is_jump: bool
) -> tuple:
    """Shared Dirichlet Q-critic loss debiased via state-only v_aux(s) for E(0) and E(lambda)."""
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    gamma = config["GAMMA"]
    aux_coef = config["AUX_VF_COEF"]

    q_i = _eval_q_taken(full_params, network, traj_batch.obs, traj_batch.action)
    v_aux_i = network.apply(full_params, traj_batch.obs, method=network.v_aux)
    v_aux_next_i = network.apply(full_params, traj_batch.next_obs, method=network.v_aux)

    targets_i = sg(targets if targets is not None else traj_batch.targets)
    aux_targets = sg(traj_batch.aux_targets)
    is_timeout = traj_batch.is_timeout
    true_terminal_i = traj_batch.done & ~is_timeout

    if is_jump:
        j_obs = traj_batch.dest_obs
        j_act = traj_batch.dest_action
        j_targets = sg(traj_batch.dest_targets)
        j_rew = traj_batch.dest_reward
        j_next_obs = traj_batch.dest_next_obs
        j_term = traj_batch.dest_true_terminal
        valid_pair = traj_batch.valid_jump
        lmbda = config["E_LAMBDA"]
        denom = jnp.maximum(1.0 - gamma * lmbda, 1e-8)
        weight_mc = 0.5 * (1.0 - gamma) / denom
        weight_dir = 0.5 * (gamma * (1.0 - lmbda)) / denom
        start_weight = 0.0
    else:
        j_obs = traj_batch.next_obs
        j_act = traj_batch.next_action
        j_targets = sg(traj_batch.next_targets)
        j_rew = traj_batch.next_reward
        j_next_obs = traj_batch.next_next_obs
        j_term = traj_batch.next_true_terminal
        valid_pair = (~true_terminal_i) & (~is_timeout) & (~traj_batch.is_horizon)
        weight_mc = 0.5 * (1.0 - gamma)
        weight_dir = 0.5 * gamma
        start_weight = 0.25 * gamma

    q_j = _eval_q_taken(full_params, network, j_obs, j_act)
    v_aux_next_j = network.apply(full_params, j_next_obs, method=network.v_aux)

    e_i = targets_i - q_i
    e_j = jnp.where(true_terminal_i, 0.0, j_targets - q_j)

    q_proxy_i = sg(traj_batch.reward + gamma * jnp.where(true_terminal_i, 0.0, v_aux_next_i))
    q_proxy_j = sg(j_rew + gamma * jnp.where(j_term, 0.0, v_aux_next_j))
    e_aux_i = q_proxy_i - q_i
    e_aux_j = jnp.where(true_terminal_i, 0.0, q_proxy_j - q_j)

    branch_i = (e_i - sg(e_j)) ** 2
    branch_j = jnp.where(true_terminal_i, 0.0, (sg(e_aux_i) - e_aux_j) ** 2)
    diff_sq = jnp.where(valid_pair, branch_i + branch_j, e_i ** 2)

    aux_mc_loss = compute_aux_head_loss(v_aux_i, aux_targets, aux_coef=aux_coef)
    magnitude_loss = weight_mc * jnp.mean(e_i ** 2)
    dirichlet_loss = weight_dir * jnp.mean(diff_sq)
    start_anchor_loss = start_weight * jnp.mean(jnp.where(traj_batch.is_start, e_i ** 2, 0.0))

    value_loss = magnitude_loss + dirichlet_loss + start_anchor_loss + aux_mc_loss
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "aux_loss": aux_mc_loss,
        "delta_mag": jnp.abs(e_i).mean(),
    }


def e_0_v_aux_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """1-step Dirichlet E(0) Q-critic loss debiased via state-only v_aux(s)."""
    return _q_dirichlet_v_aux_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config, is_jump=False
    )


def e_lambda_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Geometric-jump Dirichlet E(lambda) Q-critic loss debiased via state-only v_aux(s)."""
    return _q_dirichlet_v_aux_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config, is_jump=True
    )


Q_CRITIC_LOSS_FNS = {
    "sarsa": sarsa_critic_loss_fn,
    "sarsa_lambda": sarsa_critic_loss_fn,
    "td": sarsa_critic_loss_fn,
    "td_0": sarsa_critic_loss_fn,
    "td_lambda": sarsa_critic_loss_fn,
    "fitted": sarsa_critic_loss_fn,
    "mc": sarsa_critic_loss_fn,
    "e_0": e_0_q_critic_loss_fn,
    "e_0_q": e_0_q_critic_loss_fn,
    "e_0_bilinear": e_0_q_critic_loss_fn,
    "e0": e_0_q_critic_loss_fn,
    "e": e_0_q_critic_loss_fn,
    "e_0_v_aux": e_0_v_aux_q_critic_loss_fn,
    "e_lambda": e_lambda_q_critic_loss_fn,
    "e_lambda_q": e_lambda_q_critic_loss_fn,
    "e_lambda_v_aux": e_lambda_q_critic_loss_fn,
    "elambda": e_lambda_q_critic_loss_fn,
    "e_geometric": e_lambda_q_critic_loss_fn,
    "bsmc_double_pi": bsmc_double_pi_q_critic_loss_fn,
    "bsmc_bilinear_v_aux": bsmc_bilinear_v_aux_q_critic_loss_fn,
}


def get_q_critic_loss_fn(critic_type: str):
    key = critic_type.lower()
    if key not in Q_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown Q-critic type '{critic_type}'. Available: {list(Q_CRITIC_LOSS_FNS.keys())}")
    return Q_CRITIC_LOSS_FNS[key]


# ==============================================================================
# Continuous On-Policy DDPG Actor & Critic Losses
# ==============================================================================

def ddpg_actor_loss_fn(actor_params, critic_params, network, traj_batch, config) -> tuple:
    """
    Deterministic Policy Gradient (DDPG) actor loss:
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


def ddpg_sarsa_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    On-policy Expected SARSA(lambda) / fitted Q regression loss for continuous Q(s_t, a_t):
        L(theta) = 0.5 * E[ (G_t^{Q, lambda} - Q_theta(s_t, a_t))^2 ]
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_taken = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
    targets_i = targets if targets is not None else traj_batch.targets

    value_loss = 0.5 * jnp.mean(jnp.square(targets_i - q_taken))
    delta_mag = jnp.mean(jnp.abs(targets_i - q_taken))
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "value_loss": value_loss,
        "delta_mag": delta_mag,
    }


def ddpg_td_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    Online 1-step Expected SARSA(0) / TD(0) MSE loss using live V(s_{t+1}) = Q(s_{t+1}, mu(s_{t+1})):
        td_target = stop_gradient(r_t + gamma * (1 - true_terminal_t) * Q_theta(s_{t+1}, mu_phi(s_{t+1})))
        L(theta)  = 0.5 * E[ (Q_theta(s_t, a_t) - td_target)^2 ]
    """
    full_params = {"params": {**actor_params, **critic_params}}
    q_taken = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
    next_v = network.apply(full_params, traj_batch.next_obs, method=network.value)

    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    gamma = config["GAMMA"]

    td_target = jax.lax.stop_gradient(traj_batch.reward + gamma * (1.0 - true_terminal) * next_v)
    delta = td_target - q_taken
    value_loss = 0.5 * jnp.mean(jnp.square(delta))
    scaled_loss = config["VF_COEF"] * value_loss
    return scaled_loss, {
        "val_loss": value_loss,
        "value_loss": value_loss,
        "delta_mag": jnp.mean(jnp.abs(delta)),
    }


def ddpg_e_0_basic_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """
    1-step E(0) Q-critic loss without auxiliary critic head:
        e_i = G_t - Q_theta(s_t, a_t)
        e_j = G_{t+1} - Q_theta(s_{t+1}, mu_phi(s_{t+1}))
    In deterministic physics, (s_t, a_t) -> s_{t+1} -> mu_phi(s_{t+1}) is completely deterministic,
    so the symmetric ('naive') Dirichlet loss has zero double-sampling bias.
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient

    q_i = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
    v_j = network.apply(full_params, traj_batch.next_obs, method=network.value)

    targets_i = targets if targets is not None else traj_batch.targets
    targets_j = traj_batch.next_targets
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout

    gamma = config["GAMMA"]
    variant = config["E_VARIANT"].lower()
    vf_coef = config["VF_COEF"]

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
    scaled_loss = vf_coef * value_loss

    delta_mag = jnp.mean(jnp.abs(e_i))
    return scaled_loss, {
        "val_loss": value_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "delta_mag": delta_mag,
    }


def ddpg_e_0_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
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
    is_timeout = traj_batch.is_timeout
    true_terminal = traj_batch.done & ~is_timeout
    is_start = traj_batch.is_start

    gamma = config["GAMMA"]
    aux_coef = config["AUX_VF_COEF"]
    vf_coef = config["VF_COEF"]

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
    scaled_loss = vf_coef * value_loss

    return scaled_loss, {
        "val_loss": value_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": laplacian_loss,
        "start_anchor_loss": start_anchor_loss,
        "aux_loss": aux_mc_loss,
        "delta_mag": jnp.mean(jnp.abs(e_i)),
    }


def ddpg_e_lambda_geometric_critic_loss_fn(
    critic_params,
    actor_params,
    network,
    traj_batch,
    targets,
    config,
) -> tuple:
    """
    Geometric jumping E(lambda) Q-critic loss:
        e_start = G_t - Q_theta(s_t, a_t)
        e_jump  = G_j - Q_theta(s_j, mu_phi(s_j))
    Dispatches on config['E_VARIANT'] in ('aux', 'e_aux', 'fwd', 'naive').
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
        q_start, q_aux_start = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.both_q)
        v_jump, v_aux_jump = network.apply(full_params, traj_batch.dest_obs, method=network.both)

        e_start = targets_i - q_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)
        e_aux_start = sg(q_aux_start) - q_start
        e_aux_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, sg(v_aux_jump) - v_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + (sg(e_aux_start) - e_aux_jump) ** 2
        aux_loss = compute_aux_head_loss(q_aux_start, traj_batch.aux_targets, aux_coef=aux_coef)
    elif variant == "e_aux":
        q_start, e_phi_start = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.both_q)
        v_jump, e_phi_jump = network.apply(full_params, traj_batch.dest_obs, method=network.both)

        e_start = targets_i - q_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)
        e_phi_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, e_phi_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2 + e_aux_backward_surrogate(e_phi_start, e_phi_jump, v_jump)
        aux_loss = compute_aux_head_loss(
            e_phi_start, traj_batch.aux_targets - sg(q_start), aux_coef=aux_coef
        )
    elif variant == "fwd":
        q_start = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
        v_jump = network.apply(full_params, traj_batch.dest_obs, method=network.value)

        e_start = targets_i - q_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)

        inner_diff_sq = (e_start - sg(e_jump)) ** 2
        aux_loss = 0.0
    elif variant == "naive":
        q_start = network.apply(full_params, traj_batch.obs, traj_batch.action, method=network.q_value)
        v_jump = network.apply(full_params, traj_batch.dest_obs, method=network.value)

        e_start = targets_i - q_start
        e_jump = jnp.where(traj_batch.hit_true_terminal, 0.0, traj_batch.dest_targets - v_jump)

        inner_diff_sq = (e_start - e_jump) ** 2
        aux_loss = 0.0
    else:
        raise ValueError(
            f"Unrecognized E_VARIANT '{config['E_VARIANT']}'. Expected one of ('aux', 'e_aux', 'fwd', 'naive')."
        )

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

    return scaled_loss, {
        "val_loss": total_loss,
        "value_loss": value_loss,
        "magnitude_loss": magnitude_loss,
        "dirichlet_loss": dirichlet_loss,
        "start_loss": start_loss,
        "aux_mc_loss": aux_loss,
        "delta_mag": jnp.mean(jnp.abs(e_start)),
    }


DDPG_CRITIC_LOSS_FNS = {
    "sarsa": ddpg_sarsa_critic_loss_fn,
    "sarsa_lambda": ddpg_sarsa_critic_loss_fn,
    "fitted": ddpg_sarsa_critic_loss_fn,
    "ppo": ddpg_sarsa_critic_loss_fn,
    "td_lambda": ddpg_sarsa_critic_loss_fn,
    "sampled_td_lambda": ddpg_sarsa_critic_loss_fn,
    "mc": ddpg_sarsa_critic_loss_fn,
    "sampled_mc": ddpg_sarsa_critic_loss_fn,
    "td_0": ddpg_td_0_critic_loss_fn,
    "td": ddpg_td_0_critic_loss_fn,
    "td0": ddpg_td_0_critic_loss_fn,
    "classic_td": ddpg_td_0_critic_loss_fn,
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


def get_ddpg_critic_loss_fn(critic_type: str):
    key = critic_type.lower()
    if key not in DDPG_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown DDPG critic_type '{critic_type}'. Available: {list(DDPG_CRITIC_LOSS_FNS.keys())}")
    return DDPG_CRITIC_LOSS_FNS[key]

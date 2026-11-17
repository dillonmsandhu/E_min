# jumanji_rl/core/jumanji_critics.py
# Q-function critic and actor losses for discrete Jumanji environments

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from jumanji_rl.core.jumanji_utils import post_process_advantage


def extract_q_taken(q_all: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """Extracts Q(s, a) from discrete Q(s, ·) tensor of shape (..., |A|)."""
    return jnp.take_along_axis(q_all, action[..., None], axis=-1).squeeze(-1)


def compute_aux_head_loss(q_aux: jnp.ndarray, targets: jnp.ndarray, aux_coef: float = 1.0) -> jnp.ndarray:
    """Pure MSE loss training the auxiliary Q-head toward stop-gradient targets."""
    err = jax.lax.stop_gradient(targets) - q_aux
    return aux_coef * jnp.mean(0.5 * jnp.square(err))


def calculate_gae(traj_batch, next_value, gamma: float, gae_lambda: float):
    """
    Computes Generalized Advantage Estimation (GAE) with closed-form marginalized values.
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
    """Computes lambda-return regression targets G_t^lambda."""
    _, targets = calculate_gae(traj_batch, next_value, gamma=gamma, gae_lambda=target_lambda)
    return targets


def get_critic_target_lambda(config: dict) -> float:
    """
    Fail-fast dispatch returning target_lambda according to non-defensive invariants:
      - sarsa / td: config['VALUE_LAMBDA']
      - e_0 / e_lambda / bsmc: config['RETURN_LAMBDA']
    """
    critic_type = config["CRITIC_TYPE"].lower()
    if critic_type in ["sarsa", "sarsa_lambda", "td", "td_0", "td_lambda", "fitted", "mc"]:
        return config["VALUE_LAMBDA"]
    elif critic_type in [
        "e_0",
        "e_0_q",
        "e_0_v_aux",
        "e_0_bilinear",
        "e0",
        "e",
        "e_lambda",
        "e_lambda_q",
        "e_lambda_v_aux",
        "elambda",
        "e_geometric",
        "bsmc",
        "bsmc_v_aux",
        "bsmc_double_pi",
        "bsmc_bilinear_v_aux",
    ]:
        return config["RETURN_LAMBDA"]
    else:
        raise ValueError(f"Unrecognized CRITIC_TYPE '{critic_type}' for target lambda resolution.")


def ppo_actor_loss_fn(actor_params, critic_params, network, traj_batch, gae, config):
    """PPO clipped surrogate actor loss on discrete policy heads."""
    full_params = {"params": {**actor_params, **critic_params}}
    pi = network.apply(full_params, traj_batch.obs, method=network.policy)
    log_prob = pi.log_prob(traj_batch.action)
    entropy = pi.entropy().mean()
    ratio = jnp.exp(log_prob - traj_batch.log_prob)

    adv_norm = post_process_advantage(gae, config)
    surr1 = ratio * adv_norm
    surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * adv_norm
    actor_loss = -jnp.minimum(surr1, surr2).mean()

    total_actor_loss = config["POLICY_COEFF"] * actor_loss - config["ENT_COEF"] * entropy
    return total_actor_loss, {"actor_loss": actor_loss, "entropy": entropy}


def sarsa_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Fitted on-policy SARSA(lambda) regression loss for discrete Q(s_t, a_t)."""
    full_params = {"params": {**actor_params, **critic_params}}
    q_all = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_taken = extract_q_taken(q_all, traj_batch.action)

    targets_i = targets if targets is not None else traj_batch.targets
    loss = 0.5 * jnp.mean((targets_i - q_taken) ** 2)
    delta_mag = jnp.abs(targets_i - q_taken).mean()
    scaled_loss = config["VF_COEF"] * loss
    return scaled_loss, {"val_loss": loss, "delta_mag": delta_mag}


def _q_ppo_bilinear_e0_core(critic_params, actor_params, network, traj_batch, targets, config):
    """Shared bilinear E(0) and MC terms with exact discrete policy marginalization."""
    full_params = {"params": {**actor_params, **critic_params}}
    q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_i = extract_q_taken(q_all_i, traj_batch.action)
    targets_i = jax.lax.stop_gradient(targets if targets is not None else traj_batch.targets)
    true_terminal = traj_batch.done & ~traj_batch.is_timeout
    gamma = config["GAMMA"]

    q_all_j = network.apply(full_params, traj_batch.next_obs, method=network.q_values)
    pi_j = network.apply(full_params, traj_batch.next_obs, method=network.policy)
    probs_j = jax.lax.stop_gradient(pi_j.probs)
    v_j = jnp.sum(probs_j * q_all_j, axis=-1)

    e_i = targets_i - q_i
    delta = traj_batch.reward + gamma * jnp.where(true_terminal, 0.0, v_j) - q_i
    e0_loss = 0.5 * jnp.mean(e_i * delta)
    mc_loss = 0.5 * jnp.mean(e_i ** 2)
    return e0_loss, mc_loss, e_i, delta, gamma


def e_0_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Half the bilinear E(0) estimate with exact discrete policy marginalization."""
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
    """Bilinear BSMC Q-critic loss using exact discrete successor marginalization."""
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
    """1-step Dirichlet BSMC Q-critic loss debiased via state-only v_aux(s)."""
    return bsmc_v_aux_q_critic_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config
    )


def _q_dirichlet_v_aux_loss_fn(
    critic_params, actor_params, network, traj_batch, targets, config, is_jump: bool, is_bsmc: bool = False
) -> tuple:
    """
    Shared Dirichlet Q-critic loss debiased via state-only v_aux(s) for
    1-step E(0) (is_jump=False, is_bsmc=False), 1-step BSMC (is_jump=False, is_bsmc=True),
    and geometric-jump E(lambda) (is_jump=True).
    """
    full_params = {"params": {**actor_params, **critic_params}}
    sg = jax.lax.stop_gradient
    gamma = config["GAMMA"]
    aux_coef = config["AUX_VF_COEF"]

    q_all_i = network.apply(full_params, traj_batch.obs, method=network.q_values)
    q_i = extract_q_taken(q_all_i, traj_batch.action)
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
        if is_bsmc:
            bsmc_lambda = config["BSMC_LAMBDA"]
            denom = jnp.maximum(1.0 - gamma * bsmc_lambda, 1e-8)
            weight_mc = 0.5 * (1.0 - gamma) / denom
            weight_dir = 0.5 * (gamma * (1.0 - bsmc_lambda)) / denom
        else:
            weight_mc = 0.5 * (1.0 - gamma)
            weight_dir = 0.5 * gamma
        start_weight = 0.25 * gamma

    q_all_j = network.apply(full_params, j_obs, method=network.q_values)
    q_j = extract_q_taken(q_all_j, j_act)
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


def bsmc_v_aux_q_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """1-step Dirichlet BSMC Q-critic loss debiased via state-only v_aux(s)."""
    return _q_dirichlet_v_aux_loss_fn(
        critic_params, actor_params, network, traj_batch, targets, config, is_jump=False, is_bsmc=True
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
    "bsmc": bsmc_v_aux_q_critic_loss_fn,
    "bsmc_v_aux": bsmc_v_aux_q_critic_loss_fn,
    "bsmc_double_pi": bsmc_double_pi_q_critic_loss_fn,
    "bsmc_bilinear_v_aux": bsmc_bilinear_v_aux_q_critic_loss_fn,
}


def get_q_critic_loss_fn(critic_type: str):
    key = critic_type.lower()
    if key not in Q_CRITIC_LOSS_FNS:
        raise ValueError(f"Unknown Q-critic type '{critic_type}'. Available: {list(Q_CRITIC_LOSS_FNS.keys())}")
    return Q_CRITIC_LOSS_FNS[key]


get_critic_loss_fn = get_q_critic_loss_fn


def ppo_critic_loss_fn(critic_params, actor_params, network, traj_batch, targets, config) -> tuple:
    """Standard MSE value loss for scalar V(s) Actor-Critic."""
    full_params = {"params": {**actor_params, **critic_params}}
    value = network.apply(full_params, traj_batch.obs, method=network.value)
    targets_i = targets if targets is not None else traj_batch.targets
    loss = 0.5 * jnp.mean((targets_i - value) ** 2)
    delta_mag = jnp.abs(targets_i - value).mean()
    scaled_loss = config["VF_COEF"] * loss
    return scaled_loss, {"val_loss": loss, "delta_mag": delta_mag}

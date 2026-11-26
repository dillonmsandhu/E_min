"""
core/contraction_diagnostics.py

Shared pure-JAX online diagnostics measuring TD's loss of contraction and the
variance / auxiliary-critic cost of E-minimization across Gymnax, Brax, and MJX.

Four canonical diagnostic measures:
  1. diag_rho_c:         Centered gradient aliasing rho_c along the TD probe direction u_c
  2. diag_r_rot_c:       Centered split-half rotation ratio r_{rot,c} = sqrt(max(0, EMA(beta^2))) / max(EMA(alpha), 1e-6)
  3. diag_ret_var_ratio: Return-to-TD variance ratio Var(G_t^{(1)} - v(s_t)) / max(Var(delta_t), 1e-8)
  4. diag_aux_unexp_var: Auxiliary value (or critic) unexplained variance 1 - R^2_{aux}
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


DIAGNOSTIC_KEYS = (
    # Primary 4 (always apples-to-apples across TD, MC, E)
    "diag_rho_c",
    "diag_r_rot_c",
    "diag_env_ret_var",   # Critic-free return variance: Var(G_raw) / E[r^2]
    "diag_v_r2",          # Main critic R^2(v)
    "diag_v_rel_mse",     # Main critic relative MSE: E[(G - v)^2] / E[G^2] (safe when Var(G)->0)
    # Legacy critic-dependent VR + v_aux-only diagnostics (NaN on non-v_aux critics)
    "diag_ret_var_ratio",
    "diag_aux_r2",
    "diag_aux_vs_v",
    # Raw rotation components
    "diag_alpha_eff",
    "diag_beta_eff",
)


def _slice_obs(obs, start: int, end: int):
    return jax.tree_util.tree_map(lambda x: x[:, start:end], obs)


def _slice_obs_last(obs, start: int, end: int):
    return jax.tree_util.tree_map(lambda x: x[-1, start:end], obs)


def _flatten_obs(obs, T: int, E: int):
    return jax.tree_util.tree_map(lambda x: x.reshape((T * E,) + x.shape[2:]), obs)


def resolve_diagnostic_lambda(critic_type: str, config: dict) -> float:
    """Explicit fail-fast dispatch for the operator lambda matching `critic_type`."""
    c_type = critic_type.lower()
    if c_type in ("td_0", "e_0", "e_0_basic", "e_0_v_aux", "e_0_bilinear", "e_0_dir_marg", "e_dirichlet_double", "msbe"):
        return 0.0
    if c_type in ("fitted", "td_lambda", "sarsa_lambda"):
        return float(config["VALUE_LAMBDA"])
    if c_type in ("e_lambda", "e_lambda_v_aux"):
        return float(config["E_LAMBDA"])
    if c_type in ("bsmc", "bsmc_v_aux", "bsmc_double_pi", "bsmc_bilinear_v_aux", "bsmc_dir_marg", "bsmc_dir_double"):
        return float(config["BSMC_LAMBDA"])
    if c_type == "mc":
        return 1.0
    raise ValueError(f"Unrecognized critic_type '{critic_type}' for diagnostic lambda dispatch.")


def compute_contraction_diagnostics(
    critic_params,
    v_apply_fn,
    traj_batch,
    gamma: float,
    lmbda,
    n_probe_envs: int = 16,
    uses_aux: bool = False,
) -> dict[str, jax.Array]:
    """
    Computes online centered contraction-loss and variance diagnostics:
      - alpha_eff:      unbiased u_c^T S_{J_c, lambda} u_c
      - beta2_eff:      unbiased split-half <K_{J_c, lambda}^{(1)} u_c, K_{J_c, lambda}^{(2)} u_c>
      - rho_c:          centered lambda-trace gradient aliasing along u_c
      - ret_var_ratio:  Var(G_t^{(1)} - v(s_t)) / max(Var(delta_t), 1e-8)
      - aux_unexp_var:  1 - R^2_{aux} (for v_aux critics) or 1 - R^2_v (otherwise)

    Args:
        critic_params: Parameters differentiated with respect to by the value function.
        v_apply_fn: Callable `(critic_params, obs_batch) -> values` returning shape `(N,)` or `(N, K)`.
            If multi-head `(N, K)`, averaged over the last axis.
        traj_batch: Transition namedtuple with `obs`, `next_obs`, `reward`, `done`, `is_timeout`,
            `value`, `next_value`, and (when `uses_aux=True`) `v_aux`, `aux_targets`.
        gamma: Discount factor.
        lmbda: Operator lambda (Python float or JAX scalar).
        n_probe_envs: Number of probe environments for JVP/VJP (split in two halves for beta^2).
        uses_aux: Whether `traj_batch.v_aux` and `traj_batch.aux_targets` represent an active v_aux critic.
    """
    lmbda_f = jnp.asarray(lmbda, dtype=jnp.float32)
    gamma_f = jnp.asarray(gamma, dtype=jnp.float32)

    T, B = traj_batch.reward.shape[:2]
    E_p = min(int(n_probe_envs), B)
    half = max(1, E_p // 2)
    E_eval = min(2 * half, B)
    half = E_eval // 2

    # Disjoint envs [E_eval : min(2*E_eval, B)] for u_c when available to eliminate correlation
    if B >= E_eval + 2:
        u_start, u_end = E_eval, min(2 * E_eval, B)
    else:
        u_start, u_end = 0, E_eval
    E_u = u_end - u_start

    obs_u = _slice_obs(traj_batch.obs, u_start, u_end)
    next_obs_u = _slice_obs(traj_batch.next_obs, u_start, u_end)
    next_obs_u_last = _slice_obs_last(traj_batch.next_obs, u_start, u_end)
    rew_u = traj_batch.reward[:, u_start:u_end]
    done_u = traj_batch.done[:, u_start:u_end]
    timeout_u = traj_batch.is_timeout[:, u_start:u_end]
    true_term_u = done_u & ~timeout_u
    cont_u = (~done_u).astype(jnp.float32)
    obs_u_flat = _flatten_obs(obs_u, T, E_u)
    next_obs_u_flat = _flatten_obs(next_obs_u, T, E_u)

    def v_flat_fn(c_params, x_flat):
        v = v_apply_fn(c_params, x_flat)
        return v.mean(-1) if v.ndim > 1 else v

    # 1. Centered unit TD(lambda) semi-gradient direction u_c w.r.t. critic_params
    def td_probe_loss(c_params):
        v_s = v_flat_fn(c_params, obs_u_flat).reshape((T, E_u))
        v_last = v_flat_fn(c_params, next_obs_u_last)
        v_next_reg = jnp.concatenate([v_s[1:], v_last[None]], axis=0)
        v_next_to = v_flat_fn(c_params, next_obs_u_flat).reshape((T, E_u))
        v_next = jnp.where(timeout_u, v_next_to, v_next_reg)
        deltas = rew_u + gamma_f * (~true_term_u).astype(jnp.float32) * v_next - v_s

        def _gae_scan(gae, step_in):
            d_t, c_t = step_in
            gae = d_t + gamma_f * lmbda_f * c_t * gae
            return gae, gae

        _, advs = jax.lax.scan(
            _gae_scan,
            jnp.zeros((E_u,), dtype=jnp.float32),
            (deltas, cont_u),
            reverse=True,
        )
        target = jax.lax.stop_gradient(advs + v_s)
        err = jax.lax.stop_gradient(v_s - target)
        err_c = err - jnp.mean(err)
        return jnp.mean(err_c * v_s)

    g_td = jax.grad(td_probe_loss)(critic_params)
    g_flat, unravel = ravel_pytree(g_td)
    u_tree = unravel(g_flat / jnp.maximum(jnp.linalg.norm(g_flat), 1e-12))

    # 2. Single JVP over evaluation envs [0 : E_eval] + final next_obs, then center j_c = j - E[j]
    obs_p = _slice_obs(traj_batch.obs, 0, E_eval)
    next_obs_last = _slice_obs_last(traj_batch.next_obs, 0, E_eval)
    cont_p = (~traj_batch.done[:, :E_eval]).astype(jnp.float32)
    obs_flat = _flatten_obs(obs_p, T, E_eval)

    _, j_flat = jax.jvp(lambda cp: v_flat_fn(cp, obs_flat), (critic_params,), (u_tree,))
    _, j_last = jax.jvp(lambda cp: v_flat_fn(cp, next_obs_last), (critic_params,), (u_tree,))
    j_raw = j_flat.reshape((T, E_eval))
    j_mean = jnp.mean(j_raw)
    j = j_raw - j_mean
    j_last_c = j_last - j_mean

    # 3. Exact future (backward scan) and past (forward scan) lambda-traces on centered j_c
    j_next_arr = jnp.concatenate([j[1:], j_last_c[None]], axis=0)

    def _fwd_trace_step(trace_next, step_in):
        j_tp1, c_t = step_in
        trace_t = c_t * (gamma_f * (1.0 - lmbda_f) * j_tp1 + gamma_f * lmbda_f * trace_next)
        return trace_t, trace_t

    _, j_fwd = jax.lax.scan(
        _fwd_trace_step,
        jnp.zeros((E_eval,), dtype=jnp.float32),
        (j_next_arr, cont_p),
        reverse=True,
    )

    def _bwd_trace_step(trace_prev, step_in):
        j_t, c_t = step_in
        trace_next = c_t * (gamma_f * (1.0 - lmbda_f) * j_t + gamma_f * lmbda_f * trace_prev)
        return trace_next, trace_prev

    _, j_bwd = jax.lax.scan(
        _bwd_trace_step,
        jnp.zeros((E_eval,), dtype=jnp.float32),
        (j, cont_p),
        reverse=False,
    )

    # 4. Unbiased alpha_eff and centered aliasing rho_c over all E_eval envs
    j_sq_mean = jnp.mean(j * j)
    alpha_eff = j_sq_mean - jnp.mean(j * j_fwd)
    gamma_eff = (gamma_f * (1.0 - lmbda_f)) / jnp.maximum(1.0 - gamma_f * lmbda_f, 1e-8)
    rho_c = jnp.mean(j * j_fwd) / jnp.maximum(gamma_eff * j_sq_mean, 1e-12)

    # 5. Unbiased split-half beta^2 = <K_{J_c}^{(1)} u_c, K_{J_c}^{(2)} u_c>
    w_raw = j_bwd - j_fwd  # (T, E_eval)
    if half >= 1 and E_eval >= 2:
        obs_h1 = _flatten_obs(_slice_obs(obs_p, 0, half), T, half)
        obs_h2 = _flatten_obs(_slice_obs(obs_p, half, 2 * half), T, half)
        w1 = w_raw[:, :half]
        w2 = w_raw[:, half : 2 * half]
        w1_c = (0.5 / float(T * half)) * (w1 - jnp.mean(w1)).reshape((T * half,))
        w2_c = (0.5 / float(T * half)) * (w2 - jnp.mean(w2)).reshape((T * half,))

        _, vjp_1 = jax.vjp(lambda cp: v_flat_fn(cp, obs_h1), critic_params)
        _, vjp_2 = jax.vjp(lambda cp: v_flat_fn(cp, obs_h2), critic_params)
        v1_flat, _ = ravel_pytree(vjp_1(w1_c)[0])
        v2_flat, _ = ravel_pytree(vjp_2(w2_c)[0])
        beta2_eff = jnp.dot(v1_flat, v2_flat)
    else:
        _, vjp_fn = jax.vjp(lambda cp: v_flat_fn(cp, obs_flat), critic_params)
        w_k = (0.5 / float(T * E_eval)) * (w_raw - jnp.mean(w_raw)).reshape((T * E_eval,))
        ku_flat, _ = ravel_pytree(vjp_fn(w_k)[0])
        beta2_eff = jnp.dot(ku_flat, ku_flat)

    # 6. Return-to-TD variance ratio over the full rollout (T, B)
    v_full = traj_batch.value.mean(-1) if traj_batch.value.ndim > 2 else traj_batch.value
    v_next_full = traj_batch.next_value.mean(-1) if traj_batch.next_value.ndim > 2 else traj_batch.next_value
    true_term_full = traj_batch.done & ~traj_batch.is_timeout
    cont_full = (~traj_batch.done).astype(jnp.float32)
    delta_full = traj_batch.reward + gamma_f * (~true_term_full).astype(jnp.float32) * v_next_full - v_full

    def _mc_err_scan(mc_next, step_in):
        d_t, c_t = step_in
        mc_t = d_t + gamma_f * c_t * mc_next
        return mc_t, mc_t

    _, mc_err = jax.lax.scan(
        _mc_err_scan,
        jnp.zeros((B,), dtype=jnp.float32),
        (delta_full, cont_full),
        reverse=True,
    )
    # Second-moment ratio E[(G^{(1)} - v)^2] / E[delta^2]: equals Var(G-v)/Var(delta) once E[delta]~0,
    # while preventing 0/0 blowups on constant-reward envs (e.g. CartPole) at initialization.
    ret_var_ratio = jnp.mean(mc_err ** 2) / jnp.maximum(jnp.mean(delta_full ** 2), 1e-8)

    # 6b. Critic-free environment return variance: Var(G_raw) / max(E[r^2], 1e-8)
    # Zero dependency on the critic v; scale-invariant and safe on constant-reward envs.
    def _raw_ret_scan(g_next, step_in):
        r_t, c_t = step_in
        g_t = r_t + gamma_f * c_t * g_next
        return g_t, g_t

    _, g_raw = jax.lax.scan(
        _raw_ret_scan,
        jnp.zeros((B,), dtype=jnp.float32),
        (traj_batch.reward, cont_full),
        reverse=True,
    )
    env_ret_var = jnp.var(g_raw) / jnp.maximum(jnp.mean(traj_batch.reward ** 2), 1e-8)

    # 7. Main critic accuracy — ALWAYS about the main critic v
    mc_ret = v_full + mc_err
    v_r2 = 1.0 - jnp.mean(mc_err ** 2) / jnp.maximum(jnp.var(mc_ret), 1e-8)
    v_rel_mse = jnp.mean(mc_err ** 2) / jnp.maximum(jnp.mean(mc_ret ** 2), 1e-8)

    # 8. v_aux-only diagnostics (NaN on non-v_aux critics so TD never gets a fake value)
    if uses_aux:
        v_aux_full = traj_batch.v_aux.mean(-1) if traj_batch.v_aux.ndim > 2 else traj_batch.v_aux
        aux_tgt = traj_batch.aux_targets
        mse_aux = jnp.mean((v_aux_full - aux_tgt) ** 2)
        mse_v_on_aux = jnp.mean((v_full - aux_tgt) ** 2)
        aux_r2 = 1.0 - mse_aux / jnp.maximum(jnp.var(aux_tgt), 1e-8)
        # Positive (> 0) iff ||v_aux - V^pi||^2 < ||v - V^pi||^2
        aux_vs_v = 1.0 - mse_aux / jnp.maximum(mse_v_on_aux, 1e-8)
    else:
        aux_r2 = jnp.asarray(jnp.nan, dtype=jnp.float32)
        aux_vs_v = jnp.asarray(jnp.nan, dtype=jnp.float32)

    return {
        "alpha_eff": alpha_eff,
        "beta2_eff": beta2_eff,
        "rho_c": rho_c,
        "env_ret_var": env_ret_var,
        "ret_var_ratio": ret_var_ratio,
        "v_r2": v_r2,
        "v_rel_mse": v_rel_mse,
        "aux_r2": aux_r2,
        "aux_vs_v": aux_vs_v,
    }


def update_diagnostic_ema(
    alpha_ema: jax.Array,
    beta2_ema: jax.Array,
    alpha_eff: jax.Array,
    beta2_eff: jax.Array,
    idx: jax.Array,
    ema_decay: float = 0.95,
):
    """Updates raw signed EMAs for alpha and split-half beta^2 before sqrt(max(0, .))."""
    alpha_ema_new = jnp.where(
        idx == 1, alpha_eff, ema_decay * alpha_ema + (1.0 - ema_decay) * alpha_eff
    )
    beta2_ema_new = jnp.where(
        idx == 1, beta2_eff, ema_decay * beta2_ema + (1.0 - ema_decay) * beta2_eff
    )
    beta_eff = jnp.sqrt(jnp.maximum(beta2_ema_new, 0.0))
    r_rot_c = beta_eff / jnp.maximum(alpha_ema_new, 1e-6)
    return alpha_ema_new, beta2_ema_new, beta_eff, r_rot_c

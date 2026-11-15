"""One-step soft Bellman objectives on paired transitions conditional on (s,a).

The actor stays fixed during critic fitting. Continuation actions are independently
resampled on every minibatch visit. There is no frozen/Polyak target Q network.
MSPBE means a *minibatch tangent-space* projection, regularized by a positive ridge;
it is not an unbiased estimate of the population MSPBE (the inverse covariance
is estimated from the batch). Cross-branch fitting removes conditional target
noise bias for a fixed batch of current state-action inputs and an exact solve.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


CRITIC_ALIASES = {
    "td_0": "td_0", "td0": "td_0", "td": "td_0",
    "online_td0": "td_0", "semi_gradient_td0": "td_0",
    "msbe": "msbe", "double_msbe": "msbe",
    "mspbe": "mspbe", "double_mspbe": "mspbe",
}


def soft_bellman_residuals(critic_params, actor_params, network, batch, config, rng):
    params = {"params": {**actor_params, **critic_params}}
    key1, key2 = jax.random.split(rng)
    action1, logp1 = network.apply(params, batch.next_obs, key1, method=network.sample_action)
    action2, logp2 = network.apply(params, batch.probe_next_obs, key2, method=network.sample_action)
    q = network.apply(params, batch.obs, batch.action, method=network.q_value)
    q1 = network.apply(params, batch.next_obs, action1, method=network.q_value)
    q2 = network.apply(params, batch.probe_next_obs, action2, method=network.q_value)
    terminal1 = batch.done & ~batch.is_timeout
    terminal2 = batch.probe_done & ~batch.probe_is_timeout
    # where also prevents a terminal continuation NaN from polluting the target.
    target1 = batch.reward + config["GAMMA"] * jnp.where(
        terminal1, 0.0, q1 - config["ALPHA"] * logp1)
    target2 = batch.probe_reward + config["GAMMA"] * jnp.where(
        terminal2, 0.0, q2 - config["ALPHA"] * logp2)
    return target1 - q, target2 - q, q


def tangent_mspbe_surrogate(params, q_fn, residual_fn, ridge, cg_iters, cg_tol):
    """Gradient of 0.5 b1^T (C + ridge I)^-1 b2, including dJ/dtheta.

    J = dQ(s,a)/dtheta, C = J^T J / B, bi = J^T delta_i / B.
    Two matrix-free CG solves fit the branches separately. Stopping the solves
    and using the envelope derivative avoids differentiating CG iterations.
    The live JVPs supply the nonlinear Hessian terms, with a positive sign for
    (delta - Jw) d(Jw). They vanish for a linear critic.
    """
    sg = jax.lax.stop_gradient
    d1, d2, _ = residual_fn(params)
    n = d1.size
    _, jvp = jax.linearize(q_fn, params)
    _, vjp = jax.vjp(q_fn, params)

    def matvec(w):
        jtj = vjp(jvp(w) / n)[0]
        return jax.tree_util.tree_map(lambda x, y: x + ridge * y, jtj, w)

    b1, b2 = vjp(sg(d1) / n)[0], vjp(sg(d2) / n)[0]
    w1, _ = cg(matvec, b1, maxiter=cg_iters, tol=cg_tol)
    w2, _ = cg(matvec, b2, maxiter=cg_iters, tol=cg_tol)
    w1, w2 = jax.tree_util.tree_map(sg, (w1, w2))
    _, live_z1 = jax.jvp(q_fn, (params,), (w1,))
    _, live_z2 = jax.jvp(q_fn, (params,), (w2,))
    z1, z2 = sg(live_z1), sg(live_z2)
    surrogate = 0.5 * jnp.mean(
        z2 * d1 + z1 * d2
        + sg(d1 - z1) * live_z2 + sg(d2 - z2) * live_z1)
    # A signed cross estimate: finite batches can yield negative values.
    objective = 0.25 * jnp.mean(d1 * z2 + d2 * z1)
    errors = []
    for w, b in ((w1, b1), (w2, b2)):
        residual = jax.tree_util.tree_map(lambda a, c: a - c, matvec(w), b)
        sqnorm = lambda t: sum(jnp.vdot(x, x).real for x in jax.tree_util.tree_leaves(t))
        errors.append(jnp.sqrt(sqnorm(residual) / (sqnorm(b) + 1e-20)))
    return surrogate, {"diagnostic_mspbe": 2.0 * objective,
                       "cg_relative_residual": jnp.maximum(*errors)}


def sac_critic_loss_fn(critic_params, actor_params, network, batch, config, rng):
    sg = jax.lax.stop_gradient
    residual_fn = lambda p: soft_bellman_residuals(p, actor_params, network, batch, config, rng)
    d1, d2, q = residual_fn(critic_params)
    kind = CRITIC_ALIASES[config["CRITIC_TYPE"].lower()]
    metrics = {"diagnostic_msbe": jnp.mean(d1 * d2),
               "td_error_sq": 0.5 * jnp.mean(d1**2 + d2**2),
               "q_pred": jnp.mean(q)}
    if kind == "td_0":
        # Both branches, equal sample budget; recompute online targets every visit.
        loss = -0.5 * jnp.mean(sg(d1 + d2) * q)
    elif kind == "msbe":
        # Unbiased gradient of half MSBE before data reuse/adaptive fitting.
        loss = 0.5 * jnp.mean(sg(d1) * d2 + sg(d2) * d1)
    else:
        q_fn = lambda p: network.apply(
            {"params": {**actor_params, **p}}, batch.obs, batch.action, method=network.q_value)
        loss, projection_metrics = tangent_mspbe_surrogate(
            critic_params, q_fn, residual_fn, config["MSPBE_RIDGE"],
            config["MSPBE_CG_ITERS"], config["MSPBE_CG_TOL"])
        metrics.update(projection_metrics)
    return loss, {**metrics, "val_loss": loss}


def sac_actor_loss_fn(actor_params, critic_params, network, obs, config, rng):
    params = {"params": {**actor_params, **critic_params}}
    action, logp = network.apply(params, obs, rng, method=network.sample_action)
    q = network.apply(params, obs, action, method=network.q_value)
    loss = jnp.mean(config["ALPHA"] * logp - q)
    return loss, {"actor_loss": loss, "entropy": -jnp.mean(logp), "actor_q": jnp.mean(q)}

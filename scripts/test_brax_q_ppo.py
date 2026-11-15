# test_brax_q_ppo.py
# Unit tests and CPU smoke test for Brax continuous Q-PPO (bilinear E(0), LOO/Hybrid GAE, SARSA).
import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
import numpy as np

from brax_rl.core.brax_config import config as base_config
from brax_rl.core.brax_critics import ppo_actor_loss_fn
from brax_rl.core.brax_network import initialize_split_train_state
from brax_rl.core.brax_q_critics import brax_q_ppo_e_0_critic_loss_fn
from brax_rl.core.brax_q_network import BraxQPPOActorCritic
from brax_rl.algos.q_ppo import BraxQTransition, make_train


def test_brax_q_network_split_and_clipping():
    print("Running test_brax_q_network_split_and_clipping...")
    rng = jax.random.PRNGKey(0)
    obs_dim, act_dim = 5, 3
    net = BraxQPPOActorCritic(action_dim=act_dim, activation="tanh")
    params = net.init(rng, jnp.zeros((obs_dim,)))

    cfg = {
        **base_config,
        "TOTAL_TIMESTEPS": 1024,
        "NUM_ENVS": 4,
        "NUM_STEPS": 16,
        "CRITIC_TYPE": "e_0",
    }
    ts = initialize_split_train_state(cfg, net, params)
    actor_keys = set(ts.actor.params.keys())
    critic_keys = set(ts.critic.params.keys())
    assert actor_keys == {"actor_mean", "log_std"}, f"Unexpected actor keys: {actor_keys}"
    assert critic_keys == {"critic_torso", "q_head"}, f"Unexpected critic keys: {critic_keys}"
    assert actor_keys.isdisjoint(critic_keys)

    obs = jax.random.normal(rng, (8, obs_dim))
    a_unclipped = jnp.full((8, act_dim), 2.5)
    a_clipped = jnp.ones((8, act_dim))
    q_u = net.apply(params, obs, a_unclipped, method=net.q_value)
    q_c = net.apply(params, obs, a_clipped, method=net.q_value)
    assert jnp.allclose(q_u, q_c, atol=1e-6), "q_value must clip actions to [-1, 1]"

    # Actor loss must work cleanly with split params
    pi = net.apply(params, obs, method=net.policy)
    mb = BraxQTransition(
        done=jnp.zeros((8,), dtype=bool),
        action=a_clipped,
        value=jnp.zeros((8,)),
        q_taken=jnp.zeros((8,)),
        reward=jnp.zeros((8,)),
        log_prob=pi.log_prob(a_clipped),
        obs=obs,
        next_obs=obs,
        info={},
        is_start=jnp.zeros((8,), dtype=bool),
        is_timeout=jnp.zeros((8,), dtype=bool),
        actions_k=jnp.zeros((8, 2, act_dim)),
        q_all_k=jnp.zeros((8, 2)),
    )
    (_, aux), grads = jax.value_and_grad(ppo_actor_loss_fn, has_aux=True)(
        ts.actor.params, ts.critic.params, net, mb, jnp.ones((8,)), cfg
    )
    assert set(grads.keys()) == {"actor_mean", "log_std"}
    assert jnp.isfinite(aux["actor_loss"]) and jnp.isfinite(aux["entropy"])
    print("  PASSED")


def test_brax_q_ppo_fail_fast_validation():
    print("Running test_brax_q_ppo_fail_fast_validation...")
    bad_configs = [
        {"ENV_NAME": "inverted_pendulum", "CRITIC_TYPE": "e_0", "K_ACTIONS": 1},
        {"ENV_NAME": "inverted_pendulum", "CRITIC_TYPE": "e_0", "K_ACTIONS": True},
        {"ENV_NAME": "inverted_pendulum", "CRITIC_TYPE": "e_0", "GAE_CONTINUOUS_MODE": "inclusive"},
        {"ENV_NAME": "inverted_pendulum", "CRITIC_TYPE": "e_lambda"},
    ]
    for cfg in bad_configs:
        raised = False
        try:
            make_train(cfg)
        except ValueError:
            raised = True
        assert raised, f"Expected ValueError for config: {cfg}"

    # Missing next_probe_actions in E(0) critic loss must raise ValueError
    net = BraxQPPOActorCritic(action_dim=1)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((4,)))
    ts = initialize_split_train_state(
        {**base_config, "TOTAL_TIMESTEPS": 256, "NUM_ENVS": 4, "NUM_STEPS": 8, "CRITIC_TYPE": "e_0"},
        net,
        params,
    )
    mb = BraxQTransition(
        done=jnp.zeros((4,), dtype=bool),
        action=jnp.zeros((4, 1)),
        value=jnp.zeros((4,)),
        q_taken=jnp.zeros((4,)),
        reward=jnp.zeros((4,)),
        log_prob=jnp.zeros((4,)),
        obs=jnp.zeros((4, 4)),
        next_obs=jnp.zeros((4, 4)),
        info={},
        is_start=jnp.zeros((4,), dtype=bool),
        is_timeout=jnp.zeros((4,), dtype=bool),
        actions_k=jnp.zeros((4, 2, 1)),
        q_all_k=jnp.zeros((4, 2)),
        next_probe_actions=None,
    )
    raised = False
    try:
        brax_q_ppo_e_0_critic_loss_fn(
            ts.critic.params, ts.actor.params, net, mb, jnp.zeros((4,)), {**base_config, "CRITIC_TYPE": "e_0"}
        )
    except ValueError:
        raised = True
    assert raised, "Expected ValueError when next_probe_actions is None"
    print("  PASSED")


def test_brax_q_ppo_e0_unbiasedness_and_timeout_dtypes():
    print("Running test_brax_q_ppo_e0_unbiasedness_and_timeout_dtypes...")
    # Check that sampled bilinear E(0) gradient matches exact analytical E[delta * error] gradient
    # on a continuous 1D action linear-quadratic transition: r(s, a) = a, s' = -s (terminal if done).
    obs_dim, act_dim = 2, 1
    net = BraxQPPOActorCritic(action_dim=act_dim, activation="tanh")
    params = net.init(jax.random.PRNGKey(42), jnp.zeros((obs_dim,)))
    cfg = {**base_config, "TOTAL_TIMESTEPS": 256, "NUM_ENVS": 4, "NUM_STEPS": 8, "CRITIC_TYPE": "e_0"}
    ts = initialize_split_train_state(cfg, net, params)

    N = 16384
    rng = jax.random.PRNGKey(7)
    k1, k2, k3 = jax.random.split(rng, 3)
    obs = jax.random.normal(k1, (N, obs_dim)) * 0.5
    next_obs = jax.random.normal(k2, (N, obs_dim)) * 0.5
    a_cur = jnp.clip(jax.random.normal(k3, (N, act_dim)), -1.0, 1.0)
    a_probe = jnp.clip(jax.random.normal(jax.random.PRNGKey(99), (N, 1, act_dim)), -1.0, 1.0)
    gamma = 0.9
    reward = a_cur.squeeze(-1) + 0.3 * obs[:, 0]
    done_bool = (jnp.arange(N) % 5) == 0
    timeout_bool = (jnp.arange(N) % 10) == 0  # Half of dones are timeouts -> still bootstrap!
    terminal = done_bool & ~timeout_bool
    # Target returns with independent unbiased noisy MC return (mean = exact return)
    true_ret = reward + gamma * jnp.where(terminal, 0.0, 0.5 * next_obs[:, 0])
    noisy_ret = true_ret + jax.random.normal(jax.random.PRNGKey(11), (N,)) * 0.25

    def _loss(cp, done_arr, to_arr):
        mb = BraxQTransition(
            done=done_arr,
            action=a_cur,
            value=jnp.zeros((N,)),
            q_taken=jnp.zeros((N,)),
            reward=reward,
            log_prob=jnp.zeros((N,)),
            obs=obs,
            next_obs=next_obs,
            info={},
            is_start=jnp.zeros((N,), dtype=bool),
            is_timeout=to_arr,
            actions_k=jnp.zeros((N, 2, act_dim)),
            q_all_k=jnp.zeros((N, 2)),
            next_probe_actions=a_probe,
        )
        loss, _ = brax_q_ppo_e_0_critic_loss_fn(
            cp, ts.actor.params, net, mb, noisy_ret, {**cfg, "GAMMA": gamma, "VF_COEF": 1.0}
        )
        return loss

    grad_bool = jax.grad(lambda cp: _loss(cp, done_bool, timeout_bool))(ts.critic.params)
    grad_float = jax.grad(
        lambda cp: _loss(cp, done_bool.astype(jnp.float32), timeout_bool.astype(jnp.float32))
    )(ts.critic.params)

    for g_b, g_f in zip(jax.tree.leaves(grad_bool), jax.tree.leaves(grad_float)):
        assert jnp.allclose(g_b, g_f, atol=1e-6), "Boolean and float masks must yield identical grads"

    # Compare against true noise-free objective 0.5 * mean((true_ret - q) * (r + gamma * mask * q_probe - q))
    def _clean_loss(cp):
        fp = {"params": {**ts.actor.params, **cp}}
        q_i = net.apply(fp, obs, a_cur, method=net.q_value)
        q_j = net.apply(fp, next_obs, a_probe[:, 0, :], method=net.q_value)
        delta = reward + gamma * jnp.where(terminal, 0.0, q_j) - q_i
        err = jax.lax.stop_gradient(true_ret) - q_i
        return 0.5 * jnp.mean(err * delta)

    grad_clean = jax.grad(_clean_loss)(ts.critic.params)
    flat_noisy = jnp.concatenate([x.ravel() for x in jax.tree.leaves(grad_bool)])
    flat_clean = jnp.concatenate([x.ravel() for x in jax.tree.leaves(grad_clean)])
    cos_sim = jnp.dot(flat_noisy, flat_clean) / (jnp.linalg.norm(flat_noisy) * jnp.linalg.norm(flat_clean))
    assert cos_sim > 0.99, f"Expected near-exact unbiased gradient direction, got cos={cos_sim:.4f}"
    print(f"  PASSED (gradient cosine vs exact noise-free bilinear E(0) = {float(cos_sim):.6f})")


def test_brax_q_ppo_smoke_and_training():
    print("Running test_brax_q_ppo_smoke_and_training on Brax inverted_pendulum...")
    configs = [
        ("e_0_loo", {"CRITIC_TYPE": "e_0", "GAE_CONTINUOUS_MODE": "loo", "K_ACTIONS": 2}),
        ("e_0_hybrid", {"CRITIC_TYPE": "e_0", "GAE_CONTINUOUS_MODE": "hybrid", "K_ACTIONS": 2}),
        ("sarsa_loo", {"CRITIC_TYPE": "sarsa", "GAE_CONTINUOUS_MODE": "loo", "K_ACTIONS": 2}),
    ]
    for label, overrides in configs:
        cfg = {
            **base_config,
            "ENV_NAME": "inverted_pendulum",
            "TOTAL_TIMESTEPS": 65536,
            "NUM_ENVS": 64,
            "NUM_STEPS": 32,
            "GAMMA": 0.99,
            "GAE_LAMBDA": 0.95,
            "RETURN_LAMBDA": 1.0,
            "VALUE_LAMBDA": 0.95,
            "VF_COEF": 0.5,
            "NETWORK_CONFIG": {
                **base_config["NETWORK_CONFIG"],
                "LR": 1e-3,
                "NUM_EPOCHS": 4,
                "MINIBATCH_SIZE": 512,
                "LR_SCHEDULE": "linear",
            },
            **overrides,
        }
        train_vjit = jax.jit(jax.vmap(make_train(cfg)))
        rngs = jax.random.split(jax.random.PRNGKey(0), 2)
        outs = jax.block_until_ready(train_vjit(rngs))
        metrics = outs["metrics"]
        returns = np.asarray(metrics["returned_episode_returns"])  # (num_seeds, num_updates)
        val_loss = np.asarray(metrics["val_loss"])
        actor_loss = np.asarray(metrics["actor_loss"])
        assert np.all(np.isfinite(returns)), f"{label}: non-finite returns"
        assert np.all(np.isfinite(val_loss)), f"{label}: non-finite val_loss"
        assert np.all(np.isfinite(actor_loss)), f"{label}: non-finite actor_loss"
        init_ret = float(returns[:, :3].mean())
        final_ret = float(returns[:, -3:].mean())
        assert final_ret > init_ret, f"{label}: expected learning ({init_ret:.2f} -> {final_ret:.2f})"
        print(f"  {label}: init_ret={init_ret:.2f}, final_ret={final_ret:.2f}, final_val_loss={float(val_loss[:, -1].mean()):.4f}")
    print("  PASSED")


if __name__ == "__main__":
    test_brax_q_network_split_and_clipping()
    test_brax_q_ppo_fail_fast_validation()
    test_brax_q_ppo_e0_unbiasedness_and_timeout_dtypes()
    test_brax_q_ppo_smoke_and_training()
    print("\nALL BRAX Q-PPO TESTS PASSED!")

import sys
import os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import jax
import jax.numpy as jnp
import numpy as np
from brax_rl.algos.ppo_double_sample import make_train


def test_critic_variant(critic_type: str):
    print(f"\n=======================================================")
    print(f"Testing ppo_double_sample with CRITIC_TYPE='{critic_type}'")
    print(f"=======================================================")

    config = {
        "ENV_NAME": "inverted_pendulum",
        "TOTAL_TIMESTEPS": 256,
        "NUM_ENVS": 8,
        "NUM_STEPS": 16,
        "GAMMA": 0.99,
        "CRITIC_TYPE": critic_type,
        "GAE_LAMBDA": 0.95,
        "VALUE_LAMBDA": 0.95,
        "RETURN_LAMBDA": 1.0,
        "TD_LAMBDA": 0.0,
        "ACTOR_EPOCHS": 2,
        "CRITIC_EPOCHS": 2,
        "ACTOR_MINIBATCH_SIZE": 64,
        "CRITIC_MINIBATCH_SIZE": 64,
        "ACTOR_LR": 3e-4,
        "CRITIC_LR": 3e-4,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "NETWORK_CONFIG": {
            "LR": 3e-4,
            "NUM_EPOCHS": 2,
            "MINIBATCH_SIZE": 64,
            "NORMALIZE_ENV": True,
            "NORMALIZE_REW": True,
            "LR_SCHEDULE": "linear",
        },
    }

    train_fn = make_train(config)
    rng = jax.random.PRNGKey(42)

    print(f"Compiling and executing 2 update steps for {critic_type}...")
    jit_train = jax.jit(train_fn)
    out = jit_train(rng)
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, out
    )

    metrics = out["metrics"]
    val_loss = np.array(metrics["val_loss"])
    actor_loss = np.array(metrics["actor_loss"])
    diag_msbe = np.array(metrics["diagnostic_msbe"])

    print(f"  val_loss:        {val_loss}")
    print(f"  actor_loss:      {actor_loss}")
    print(f"  diagnostic_msbe: {diag_msbe}")

    assert np.all(np.isfinite(val_loss)), f"val_loss has non-finite values for {critic_type}: {val_loss}"
    assert np.all(np.isfinite(actor_loss)), f"actor_loss has non-finite values for {critic_type}: {actor_loss}"
    assert np.all(np.isfinite(diag_msbe)), f"diagnostic_msbe has non-finite values for {critic_type}: {diag_msbe}"
    print(f"✓ {critic_type} PASSED!")


if __name__ == "__main__":
    for critic in ["fitted", "msbe", "e_inner", "mspbe"]:
        test_critic_variant(critic)
    print("\n All ppo_double_sample tests PASSED successfully!")

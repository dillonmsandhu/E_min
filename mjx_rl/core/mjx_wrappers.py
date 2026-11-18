"""
Wrappers for MuJoCo Playground (MJX) environments.

Adapts MuJoCo Playground MjxEnv instances into the Gymnax-compatible vectorized
interface used across E_min, preserving:
  - `real_next_obs`: true pre-reset observation on episode termination/truncation
  - `is_timeout`: boolean mask distinguishing episode horizon truncation from true termination
  - Support for both flat array observations (DM Control Suite) and dict observations
    (e.g. {'state': ..., 'privileged_state': ...} in locomotion tasks).
"""

from typing import Any, Mapping
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from brax.envs.wrappers import training as brax_training
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src import wrapper as pg_wrapper


class PlaygroundAutoResetWrapper(pg_wrapper.Wrapper):
    """
    AutoReset wrapper for MuJoCo Playground environments (operating on unbatched MjxEnv,
    to be vectorized via VecEnv) that:
      1. Preserves the true pre-reset observation in `state.info['real_next_obs']`.
      2. Exposes `state.info['is_timeout']` from `EpisodeWrapper`'s `state.info['truncation']`.
      3. Resets `data` and `obs` to the cached initial state (or full reset) when `done > 0.5`.
    """

    def __init__(self, env: Any, full_reset: bool = False):
        super().__init__(env)
        self._full_reset = full_reset
        self._info_key = "AutoResetWrapper"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        state = self.env.reset(key)
        state.info[f"{self._info_key}_first_data"] = state.data
        state.info[f"{self._info_key}_first_obs"] = state.obs
        state.info[f"{self._info_key}_rng"] = rng
        state.info["real_next_obs"] = state.obs
        state.info["is_timeout"] = jnp.zeros((), dtype=bool)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, reset_key = jax.random.split(state.info[f"{self._info_key}_rng"])
        if self._full_reset:
            reset_state = self.env.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f"{self._info_key}_first_data"]
            reset_obs = state.info[f"{self._info_key}_first_obs"]

        if "steps" in state.info:
            steps = jnp.where(
                state.done > 0.5,
                jnp.zeros_like(state.info["steps"]),
                state.info["steps"],
            )
            state.info["steps"] = steps

        state = state.replace(done=jnp.zeros_like(state.done))
        next_state = self.env.step(state, action)
        true_next_obs = next_state.obs

        def where_done(x, y):
            d = next_state.done > 0.5
            d = jnp.reshape(d, [1] * len(x.shape))
            return jnp.where(d, x, y)

        data = jax.tree.map(where_done, reset_data, next_state.data)
        obs = jax.tree.map(where_done, reset_obs, next_state.obs)

        next_info = dict(next_state.info)
        next_info[f"{self._info_key}_rng"] = rng
        next_info["real_next_obs"] = true_next_obs
        next_info["is_timeout"] = next_info["truncation"] > 0.5
        return next_state.replace(data=data, obs=obs, info=next_info)


class PlaygroundGymnaxWrapper:
    """
    Converts a MuJoCo Playground environment into a Gymnax-compatible interface.
    """

    def __init__(
        self,
        env_name: str,
        impl: str = "jax",
        episode_length: int = 1000,
        action_repeat: int = 1,
        full_reset: bool = False,
        config_overrides: dict[str, Any] | None = None,
    ):
        overrides = {"impl": impl}
        if config_overrides:
            overrides.update(config_overrides)
        env = registry.load(env_name, config_overrides=overrides)
        env = brax_training.EpisodeWrapper(env, episode_length=episode_length, action_repeat=action_repeat)
        env = PlaygroundAutoResetWrapper(env, full_reset=full_reset)
        self._env = env
        self.action_size = env.action_size
        self.raw_observation_size = env.observation_size
        if isinstance(env.observation_size, Mapping):
            self.observation_size = {
                k: (v,) if isinstance(v, int) else tuple(v)
                for k, v in env.observation_size.items()
            }
        else:
            self.observation_size = (
                (env.observation_size,)
                if isinstance(env.observation_size, int)
                else tuple(env.observation_size)
            )

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        del key, params
        next_state = self._env.step(state, action)
        clean_info = {
            k: v
            for k, v in next_state.info.items()
            if k not in ("AutoResetWrapper_first_data", "AutoResetWrapper_first_obs", "AutoResetWrapper_rng", "rng")
        }
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, clean_info

    def observation_space(self, params=None):
        del params
        if isinstance(self.observation_size, Mapping):
            return spaces.Dict(
                {
                    k: spaces.Box(low=-jnp.inf, high=jnp.inf, shape=shape)
                    for k, shape in self.observation_size.items()
                }
            )
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.observation_size,
        )

    def action_space(self, params=None):
        del params
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_size,),
        )


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low: float = -1.0, high: float = 1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: Any
    var: Any
    count: float
    env_state: environment.EnvState


def _update_welford_leaf(obs_leaf, mean_leaf, var_leaf, count, batch_count):
    batch_mean = jnp.mean(obs_leaf, axis=0)
    batch_var = jnp.var(obs_leaf, axis=0)
    delta = batch_mean - mean_leaf
    tot_count = count + batch_count
    new_mean = mean_leaf + delta * batch_count / tot_count
    m_a = var_leaf * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    return new_mean, new_var


def _normalize_obs_tree(obs, mean, var):
    return jax.tree_util.tree_map(
        lambda o, m, v: (o - m) / jnp.sqrt(v + 1e-8),
        obs,
        mean,
        var,
    )


class NormalizeVecObservation(GymnaxWrapper):
    """
    Running Welford observation normalizer supporting both flat array observations
    and PyTree/dict observations, plus simultaneous normalization of `info['real_next_obs']`.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset_stats(self, state):
        if hasattr(self._env, "reset_stats"):
            new_env_state = self._env.reset_stats(state.env_state)
        else:
            new_env_state = state.env_state
        return NormalizeVecObsEnvState(
            mean=jax.tree_util.tree_map(jnp.zeros_like, state.mean),
            var=jax.tree_util.tree_map(jnp.ones_like, state.var),
            count=1e-4,
            env_state=new_env_state,
        )

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        first_leaf = jax.tree_util.tree_leaves(obs)[0]
        batch_count = first_leaf.shape[0]
        init_mean = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x[0]), obs)
        init_var = jax.tree_util.tree_map(lambda x: jnp.ones_like(x[0]), obs)
        count = 1e-4

        updated = jax.tree_util.tree_map(
            lambda o, m, v: _update_welford_leaf(o, m, v, count, batch_count),
            obs,
            init_mean,
            init_var,
        )
        new_mean = jax.tree_util.tree_map(
            lambda o, m, v: _update_welford_leaf(o, m, v, count, batch_count)[0],
            obs,
            init_mean,
            init_var,
        )
        new_var = jax.tree_util.tree_map(
            lambda o, m, v: _update_welford_leaf(o, m, v, count, batch_count)[1],
            obs,
            init_mean,
            init_var,
        )
        del updated
        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=count + batch_count,
            env_state=env_state,
        )
        return _normalize_obs_tree(obs, state.mean, state.var), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        first_leaf = jax.tree_util.tree_leaves(obs)[0]
        batch_count = first_leaf.shape[0]
        new_mean = jax.tree_util.tree_map(
            lambda o, m, v: _update_welford_leaf(o, m, v, state.count, batch_count)[0],
            obs,
            state.mean,
            state.var,
        )
        new_var = jax.tree_util.tree_map(
            lambda o, m, v: _update_welford_leaf(o, m, v, state.count, batch_count)[1],
            obs,
            state.mean,
            state.var,
        )
        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=state.count + batch_count,
            env_state=env_state,
        )
        norm_obs = _normalize_obs_tree(obs, state.mean, state.var)
        if isinstance(info, dict) and "real_next_obs" in info:
            info["real_next_obs"] = _normalize_obs_tree(
                info["real_next_obs"], state.mean, state.var
            )
        return norm_obs, state, reward, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: jnp.ndarray
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma: float):
        super().__init__(env)
        self.gamma = gamma

    def reset_stats(self, state):
        if hasattr(self._env, "reset_stats"):
            new_env_state = self._env.reset_stats(state.env_state)
        else:
            new_env_state = state.env_state
        return NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=state.return_val,
            env_state=new_env_state,
        )

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        first_leaf = jax.tree_util.tree_leaves(obs)[0]
        batch_count = first_leaf.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        first_leaf = jax.tree_util.tree_leaves(obs)[0]
        batch_count = first_leaf.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info

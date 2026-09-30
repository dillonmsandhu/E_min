# from purejaxrl: https://github.com/luchris429/purejaxrl
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper
from brax import envs
from brax.envs.base import Wrapper as BraxWrapper, State as BraxState
from brax.envs.wrappers.training import EpisodeWrapper


class AutoResetWrapper(BraxWrapper):
    """
    Custom AutoReset wrapper for Brax environments that:
    1. Preserves the true terminal observation in state.info['real_next_obs'] before reset.
    2. Exposes state.info['truncation'] from EpisodeWrapper.
    3. Resets obs and pipeline_state to initial state for the subsequent step.
    """
    def reset(self, rng: jax.Array) -> BraxState:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        state.info["real_next_obs"] = state.obs
        return state

    def step(self, state: BraxState, action: jax.Array) -> BraxState:
        if "steps" in state.info:
            steps = jnp.where(state.done > 0.5, jnp.zeros_like(state.info["steps"]), state.info["steps"])
            state.info["steps"] = steps
        state = state.replace(done=jnp.zeros_like(state.done))

        next_state = self.env.step(state, action)
        true_next_obs = next_state.obs

        def where_done(x, y):
            d = next_state.done > 0.5
            d = jnp.reshape(d, [1] * len(x.shape))
            return jnp.where(d, x, y)

        pipeline_state = jax.tree.map(
            where_done, next_state.info["first_pipeline_state"], next_state.pipeline_state
        )
        obs = where_done(next_state.info["first_obs"], next_state.obs)

        next_state.info["real_next_obs"] = true_next_obs
        return next_state.replace(pipeline_state=pipeline_state, obs=obs)


class BraxGymnaxWrapper:
    """
    Converts a Brax environment into a Gymnax-compatible interface.
    """
    def __init__(self, env_name, backend="positional"):
        backend = "generalized" if env_name in ["swimmer", "humanoid", "humanoidstandup"] else "positional"
        env = envs.get_environment(env_name=env_name, backend=backend)
        env = EpisodeWrapper(env, episode_length=1000, action_repeat=1)
        env = AutoResetWrapper(env)
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        clean_info = {
            k: v for k, v in next_state.info.items()
            if k not in ["first_pipeline_state", "first_obs"]
        }
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, clean_info

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )

class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)

class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState

class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)


    def reset_stats(self, state):
        """
        Resets running mean/var to initial values, but PRESERVES the 
        underlying environment state (physics).
        Recursively calls reset_stats on the inner env if it exists.
        """
        # 1. Recurse: Check if the inner environment also has a reset_stats method
        # (e.g., if this is wrapped around another wrapper)
        if hasattr(self._env, "reset_stats"):
            new_env_state = self._env.reset_stats(state.env_state)
        else:
            new_env_state = state.env_state

        # 2. Reset Own Stats
        return NormalizeVecObsEnvState(
            mean=jnp.zeros_like(state.mean),
            var=jnp.ones_like(state.var),
            count=1e-4,
            env_state=new_env_state # <--- Physics preserved/updated recursively
        )

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        norm_obs = (obs - state.mean) / jnp.sqrt(state.var + 1e-8)
        if isinstance(info, dict) and "real_next_obs" in info:
            info["real_next_obs"] = (info["real_next_obs"] - state.mean) / jnp.sqrt(state.var + 1e-8)
        return norm_obs, state, reward, done, info


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState

class NormalizeVecReward(GymnaxWrapper):

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma
    
    def reset_stats(self, state):
        """
        Resets running mean/var. 
        CRITICAL: We PRESERVE 'return_val' (the current discounted return buffer).
        Why? Because the agent is in the middle of a trajectory. If we zeroed this,
        the running return calculation would break. Keeping it ensures the variance
        updates correctly on the very next step.
        """
        # 1. Recurse
        if hasattr(self._env, "reset_stats"):
            new_env_state = self._env.reset_stats(state.env_state)
        else:
            new_env_state = state.env_state

        # 2. Reset Own Stats
        return NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=state.return_val, # <--- Keep the current return trajectory!
            env_state=new_env_state
        )

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state.env_state, action, params)
        return_val = (state.return_val * self.gamma * (1 - done) + reward)
 
        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info

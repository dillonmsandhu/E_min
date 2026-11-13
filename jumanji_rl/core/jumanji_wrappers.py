# jumanji_rl/core/jumanji_wrappers.py
# Environment adapter and wrappers for Jumanji environments in Pure JAX

from __future__ import annotations
import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Tuple, Dict
from gymnax.environments.environment import Environment, EnvState, EnvParams
from gymnax.environments import spaces

import jumanji
from envs.wrappers import TerminalInfoWrapper
from envs.log_wrapper import LogWrapper


@struct.dataclass
class JumanjiEnvParams(EnvParams):
    max_steps_in_episode: int
    gamma: float = 0.99


@struct.dataclass
class JumanjiEnvState(EnvState):
    jumanji_state: Any
    time: int


class JumanjiEnvironment(Environment):
    """
    Wraps a Jumanji environment into the standard Gymnax interface:
      - reset_env(key, params) -> (obs, state)
      - step_env(key, state, action, params) -> (obs, state, reward, done, info)
      - observation_space(params) -> spaces.Box
      - action_space(params) -> spaces.Discrete
    """

    def __init__(self, env_name: str, **kwargs):
        super().__init__()
        self.raw_env_name = env_name
        clean_name = env_name.replace("jumanji/", "") if env_name.startswith("jumanji/") else env_name
        self.jumanji_name = clean_name
        self._env = jumanji.make(clean_name, **kwargs)
        self.time_limit = int(getattr(self._env, "time_limit", 1000))

        # Probe reset for observation shape and bounds
        probe_state, probe_ts = self._env.reset(jax.random.PRNGKey(0))
        probe_obs = self._extract_obs(probe_ts.observation)
        self._obs_shape = probe_obs.shape

        act_spec = self._env.action_spec
        if not hasattr(act_spec, "num_values"):
            raise ValueError(
                f"JumanjiEnvironment currently supports discrete action spaces with 'num_values', "
                f"got {type(act_spec)} for {clean_name}"
            )
        self._num_actions = int(act_spec.num_values)

    @property
    def default_params(self) -> JumanjiEnvParams:
        return JumanjiEnvParams(max_steps_in_episode=self.time_limit)

    def _extract_obs(self, observation: Any) -> jnp.ndarray:
        """Extracts spatial or vector observation array from Jumanji observation NamedTuple."""
        if hasattr(observation, "grid"):
            return jnp.asarray(observation.grid, dtype=jnp.float32)
        elif hasattr(observation, "board"):
            # e.g., Game2048: shape (4, 4) -> add channel dimension (4, 4, 1)
            board = jnp.asarray(observation.board, dtype=jnp.float32)
            if board.ndim == 2:
                board = board[..., None]
            return board
        elif isinstance(observation, (jnp.ndarray, jax.Array)):
            return jnp.asarray(observation, dtype=jnp.float32)
        else:
            raise TypeError(
                f"Unsupported observation format for {self.jumanji_name}: {type(observation)}. "
                f"Expected attribute 'grid', 'board', or jnp.ndarray."
            )

    def step_env(
        self, key: jax.Array, state: JumanjiEnvState, action: int | jax.Array, params: JumanjiEnvParams
    ) -> Tuple[jnp.ndarray, JumanjiEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        new_jstate, ts = self._env.step(state.jumanji_state, action)
        obs = self._extract_obs(ts.observation)
        reward = jnp.asarray(ts.reward, dtype=jnp.float32)
        done = ts.last()
        new_state = JumanjiEnvState(jumanji_state=new_jstate, time=state.time + 1)
        return obs, new_state, reward, done, {}

    def reset_env(
        self, key: jax.Array, params: JumanjiEnvParams
    ) -> Tuple[jnp.ndarray, JumanjiEnvState]:
        jstate, ts = self._env.reset(key)
        obs = self._extract_obs(ts.observation)
        return obs, JumanjiEnvState(jumanji_state=jstate, time=0)

    def observation_space(self, params: JumanjiEnvParams) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self._obs_shape,
            dtype=jnp.float32,
        )

    def action_space(self, params: JumanjiEnvParams) -> spaces.Discrete:
        return spaces.Discrete(self._num_actions)


def make_env(config: dict) -> Tuple[Any, JumanjiEnvParams]:
    """
    Creates and wraps a Jumanji environment:
    1. Instantiates JumanjiEnvironment adapter
    2. Wraps in TerminalInfoWrapper (to capture real_next_obs and is_timeout)
    3. Wraps in LogWrapper (to track episodic returns and lengths)
    """
    env_name = config["ENV_NAME"]
    env = JumanjiEnvironment(env_name)
    env_params = env.default_params

    # Wrap in TerminalInfoWrapper
    env = TerminalInfoWrapper(env)

    # Wrap in LogWrapper
    gamma = config["GAMMA"]
    env = LogWrapper(env, gamma=gamma)

    return env, env_params

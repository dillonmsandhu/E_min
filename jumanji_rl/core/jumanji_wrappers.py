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
    Supports strongly cyclic single-agent discrete environments:
      - Sokoban-v0 (10x10x4 grid, Discrete(4))
      - Snake-v1 (12x12x5 grid, Discrete(4))
      - Game2048-v1 (4x4x1 board, Discrete(4))
      - Maze-v0 (10x10x3 grid: walls, agent, target, Discrete(4))
      - SlidingTilePuzzle-v0 (5x5x1 board, Discrete(4))
      - RubiksCube-partly-scrambled-v0 (3x3x6 transposed faces, Discrete(18))
      - RubiksCube-v0 (3x3x6 transposed faces, Discrete(18))
    """

    def __init__(self, env_name: str, **kwargs):
        super().__init__()
        self.raw_env_name = env_name
        clean_name = env_name.replace("jumanji/", "") if env_name.startswith("jumanji/") else env_name
        self.jumanji_name = clean_name
        self._env = jumanji.make(clean_name, **kwargs)
        self.time_limit = int(getattr(self._env, "time_limit", 1000))

        # Check action space: either scalar discrete or Rubik's cube [6, 1, 3] -> 18
        act_spec = self._env.action_spec
        if "RubiksCube" in self.jumanji_name:
            self._is_rubiks = True
            self._num_actions = 18  # 6 faces * 1 depth * 3 rotation amounts
        elif hasattr(act_spec, "num_values") and getattr(act_spec.num_values, "shape", ()) == ():
            self._is_rubiks = False
            self._num_actions = int(act_spec.num_values)
        else:
            raise ValueError(
                f"JumanjiEnvironment currently supports discrete action spaces with scalar 'num_values', "
                f"got {type(act_spec)} with shape {getattr(getattr(act_spec, 'num_values', None), 'shape', None)} for {clean_name}"
            )

        # Probe reset for observation shape and bounds
        probe_state, probe_ts = self._env.reset(jax.random.PRNGKey(0))
        probe_obs = self._extract_obs(probe_ts.observation)
        self._obs_shape = probe_obs.shape

    @property
    def default_params(self) -> JumanjiEnvParams:
        return JumanjiEnvParams(max_steps_in_episode=self.time_limit)

    def _extract_obs(self, observation: Any) -> jnp.ndarray:
        """Extracts spatial (H, W, C) observation array from Jumanji observation NamedTuple."""
        # 1. RubiksCube: (6, 3, 3) -> transpose to (3, 3, 6)
        if hasattr(observation, "cube"):
            cube = jnp.asarray(observation.cube, dtype=jnp.float32)
            # Transpose (6, 3, 3) -> (3, 3, 6) for standard CNN (H, W, C)
            if cube.ndim == 3:
                return jnp.transpose(cube, (1, 2, 0))
            return cube

        # 2. Maze: walls (H, W), agent_position, target_position -> 3-channel grid (H, W, 3)
        if hasattr(observation, "walls") and hasattr(observation, "agent_position") and hasattr(observation, "target_position"):
            walls = jnp.asarray(observation.walls, dtype=jnp.float32)
            H, W = walls.shape
            agent_grid = (jnp.arange(H)[:, None] == observation.agent_position.row) & (jnp.arange(W)[None, :] == observation.agent_position.col)
            target_grid = (jnp.arange(H)[:, None] == observation.target_position.row) & (jnp.arange(W)[None, :] == observation.target_position.col)
            return jnp.stack([walls, agent_grid.astype(jnp.float32), target_grid.astype(jnp.float32)], axis=-1)

        # 3. SlidingTilePuzzle: puzzle (5, 5) -> add channel dimension (5, 5, 1)
        if hasattr(observation, "puzzle"):
            puzzle = jnp.asarray(observation.puzzle, dtype=jnp.float32)
            if puzzle.ndim == 2:
                puzzle = puzzle[..., None]
            return puzzle

        # 4. Standard grid (e.g. Sokoban, Snake, Pacman)
        if hasattr(observation, "grid"):
            return jnp.asarray(observation.grid, dtype=jnp.float32)

        # 5. Game2048: board (4, 4) -> (4, 4, 1)
        if hasattr(observation, "board"):
            board = jnp.asarray(observation.board, dtype=jnp.float32)
            if board.ndim == 2:
                board = board[..., None]
            return board

        if isinstance(observation, (jnp.ndarray, jax.Array)):
            return jnp.asarray(observation, dtype=jnp.float32)

        raise TypeError(
            f"Unsupported observation format for {self.jumanji_name}: {type(observation)}. "
            f"Expected attribute 'cube', 'walls', 'puzzle', 'grid', 'board', or jnp.ndarray."
        )

    def _convert_action(self, action: int | jax.Array) -> jnp.ndarray:
        """Converts scalar action to environment-specific format if needed."""
        if self._is_rubiks:
            # scalar in 0..17 -> [face, depth, amount]
            act_int = jnp.asarray(action, dtype=jnp.int32)
            face = act_int // 3
            amount = act_int % 3
            depth = jnp.zeros_like(face)
            return jnp.stack([face, depth, amount], axis=-1)
        return action

    def step_env(
        self, key: jax.Array, state: JumanjiEnvState, action: int | jax.Array, params: JumanjiEnvParams
    ) -> Tuple[jnp.ndarray, JumanjiEnvState, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        env_action = self._convert_action(action)
        new_jstate, ts = self._env.step(state.jumanji_state, env_action)
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

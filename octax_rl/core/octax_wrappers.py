"""Gymnax-compatible wrapper for Octax CHIP-8 environments.

Keeps observations in compact native boolean dtype `(frame_skip, 64, 32)`
so rollout buffers and `real_next_obs` use 1 byte per pixel instead of 4;
casting to `float32` and channel transposition happen only inside the CNN.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple

import flax.struct
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments.spaces import Box, Discrete

from envs.log_wrapper import LogWrapper
from envs.wrappers import TerminalInfoWrapper


def _ensure_flax_dataclass_compat() -> None:
    """Allow octax's `@dataclass(frozen=True)` under flax 0.9.0."""
    if "frozen" not in inspect.signature(flax.struct.dataclass).parameters:
        orig_dc = flax.struct.dataclass

        def _compat_dc(clz=None, **kwargs):
            if clz is None:
                return lambda c: orig_dc(c)
            return orig_dc(clz)

        flax.struct.dataclass = _compat_dc


_ensure_flax_dataclass_compat()

from octax.env import OctaxEnv  # noqa: E402
from octax.environments import create_environment  # noqa: E402


@flax.struct.dataclass
class OctaxEnvParams(EnvParams):
    """Gymnax-compatible parameters for Octax environments."""

    max_steps_in_episode: int = 4500
    sticky_action_prob: float = 0.0


@flax.struct.dataclass
class OctaxWrapperState:
    """Wrapper state exposing `.time` for TerminalInfoWrapper and tracking sticky actions."""

    env_state: Any
    time: jax.Array
    last_action: jax.Array


class OctaxGymnaxWrapper(Environment):
    """Gymnax-compatible wrapper around `octax.env.OctaxEnv`."""

    def __init__(self, octax_env: OctaxEnv, sticky_action_prob: float = 0.0):
        self._octax_env = octax_env
        self._sticky_action_prob = float(sticky_action_prob)

    @property
    def default_params(self) -> OctaxEnvParams:
        return OctaxEnvParams(
            max_steps_in_episode=self._octax_env.max_num_steps_per_episodes,
            sticky_action_prob=self._sticky_action_prob,
        )

    def reset_env(
        self,
        key: jax.Array,
        params: OctaxEnvParams,
    ) -> Tuple[jax.Array, OctaxWrapperState]:
        state, obs, _ = self._octax_env.reset(key)
        wrapped_state = OctaxWrapperState(
            env_state=state,
            time=state.time,
            last_action=jnp.int32(0),
        )
        return obs, wrapped_state

    def step_env(
        self,
        key: jax.Array,
        state: OctaxWrapperState,
        action: jax.Array | int,
        params: OctaxEnvParams,
    ) -> Tuple[jax.Array, OctaxWrapperState, jax.Array, jax.Array, Dict[Any, Any]]:
        action_int = jnp.asarray(action, dtype=jnp.int32)
        if self._sticky_action_prob > 0.0:
            use_sticky = jax.random.bernoulli(key, params.sticky_action_prob)
            effective_action = jnp.where(
                (state.time > 0) & use_sticky,
                state.last_action,
                action_int,
            )
        else:
            effective_action = action_int

        next_state, obs, reward, terminated, truncated, info = self._octax_env.step(
            state.env_state, effective_action
        )
        timeout = truncated | (next_state.time >= params.max_steps_in_episode)
        done = terminated | timeout

        wrapped_state = OctaxWrapperState(
            env_state=next_state,
            time=next_state.time,
            last_action=effective_action,
        )
        return obs, wrapped_state, jnp.asarray(reward, dtype=jnp.float32), done, info

    @property
    def name(self) -> str:
        return f"Octax-{self._octax_env.rom_path.split('/')[-1].split('.')[0]}"

    @property
    def num_actions(self) -> int:
        return self._octax_env.num_actions

    def action_space(self, params: OctaxEnvParams) -> Discrete:
        return Discrete(self.num_actions)

    def observation_space(self, params: OctaxEnvParams) -> Box:
        return Box(
            low=0,
            high=1,
            shape=(self._octax_env.frame_skip, 64, 32),
            dtype=jnp.bool_,
        )


def parse_octax_game_name(env_name: str) -> str:
    """Extracts the Octax game identifier from e.g. 'Octax-pong' or 'octax:pong'."""
    if env_name.startswith("Octax-"):
        return env_name[len("Octax-") :]
    if env_name.startswith("octax:"):
        return env_name[len("octax:") :]
    return env_name


def make_env(config: dict) -> Tuple[Any, OctaxEnvParams]:
    """Constructs and wraps an Octax environment with TerminalInfoWrapper + LogWrapper."""
    game_name = parse_octax_game_name(config["ENV_NAME"])
    octax_env, _ = create_environment(game_name)
    env = OctaxGymnaxWrapper(
        octax_env,
        sticky_action_prob=config["STICKY_ACTION_PROB"],
    )
    env_params = env.default_params
    if config["MAX_STEPS_IN_EPISODE"] is not None:
        env_params = env_params.replace(max_steps_in_episode=int(config["MAX_STEPS_IN_EPISODE"]))

    env = TerminalInfoWrapper(env)
    env = LogWrapper(env, gamma=config["GAMMA"])
    return env, env_params

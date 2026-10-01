"""JAX implementation of Point Robot environment."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    last_action: jax.Array
    last_reward: jax.Array
    pos: jax.Array
    goal: jax.Array
    goals_reached: int
    time: float


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_force: float = 0.1  # Max action (+/-)
    circle_radius: float = 1.0  # Radius of semi-circle
    dense_reward: bool = False  # Distance reward at each timestep
    goal_radius: float = 0.2  # Radius for success
    center_init: bool = False  # Init at [0, 0]. Otherwise sample in radius
    normalize_time: bool = True  # Normalize timestep into [-1, 1]
    max_steps_in_episode: int = 100  # Steps in an episode (constant goal)
    fully_observable: bool = True  # If True (default), includes goal vector in observation (MDP)
    # Potential-based reward shaping parameters
    gamma: float = 0.99  # Discount factor for PBRS
    potential_scale: float = 1.0  # Scale factor for PBRS
    # Stochastic / Noise parameters
    slip_prob: float = 0.0  # Probability of traction slip on each step
    slip_force_scale: float = 0.0  # Fraction of displacement transmitted during slip
    action_noise_std: float = 0.0  # Gaussian noise standard deviation on commanded action
    transition_noise_std: float = 0.0  # Gaussian noise standard deviation on 2D position


class PointRobot(environment.Environment):
    """JAX implementation of 2D Semi-Circle Point Robot environment as in Dorfman et al.
    2021 https://openreview.net/pdf?id=IBdEfhLveS

    Extended with fully observable MDP mode (default), tire/traction slip, transition noise,
    and Potential-Based Reward Shaping (PBRS).
    """

    def __init__(self, fully_observable: bool = True):
        super().__init__()
        self.fully_observable = fully_observable

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """Perform single timestep state transition with optional slip, noise, and PBRS dense reward."""
        key_slip, key_trans, key_act, key_respawn = jax.random.split(key, 4)

        # 1. Action noise & clipping
        noisy_action = action + params.action_noise_std * jax.random.normal(key_act, shape=(2,))
        a = jnp.clip(noisy_action, -params.max_force, params.max_force)

        # 2. Traction slip: with probability slip_prob, displacement is scaled down
        slipped = jax.random.uniform(key_slip, shape=()) < params.slip_prob
        effective_a = jnp.where(slipped, a * params.slip_force_scale, a)

        # 3. Position update with transition noise
        pos = state.pos + effective_a + params.transition_noise_std * jax.random.normal(key_trans, shape=(2,))
        goal_distance = jnp.linalg.norm(state.goal - pos)
        goal_reached = goal_distance <= params.goal_radius
        base_reward = goal_reached * 1.0

        # Potential-Based Reward Shaping (PBRS):
        # Potential Phi(s) = -norm(pos - goal), referenced at 0.0 when goal is reached
        phi_cur = -jnp.linalg.norm(state.goal - state.pos)
        phi_next_raw = -goal_distance
        phi_next = jax.lax.select(goal_reached, 0.0, phi_next_raw)
        shaping = params.gamma * phi_next - phi_cur

        reward = jax.lax.select(
            params.dense_reward,
            base_reward + params.potential_scale * shaping,
            base_reward,
        )

        sampled_pos = sample_agent_position(
            key_respawn, params.circle_radius, params.center_init
        )
        # Sample/set new initial position if goal was reached
        new_pos = jax.lax.select(goal_reached, sampled_pos, pos)
        state = EnvState(
            last_action=action,
            last_reward=reward,
            pos=new_pos,
            goal=state.goal,
            goals_reached=state.goals_reached + goal_reached,
            time=state.time + 1,
        )

        done = self.is_terminal(state, params)
        obs = self.get_obs(state, params)
        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params), "slipped": slipped},
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        # Sample reward function + construct state as concat with timestamp
        key_goal, key_pos = jax.random.split(key)
        angle = jax.random.uniform(key_goal, minval=0, maxval=jnp.pi)
        xs = params.circle_radius * jnp.cos(angle)
        ys = params.circle_radius * jnp.sin(angle)
        goal = jnp.array([xs, ys])
        sampled_pos = sample_agent_position(
            key_pos, params.circle_radius, params.center_init
        )

        state = EnvState(
            last_action=jnp.zeros(2),
            last_reward=jnp.array(0.0),
            pos=sampled_pos,
            goal=goal,
            goals_reached=0,
            time=0.0,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> jax.Array:
        """Construct observation: includes goal vector if fully_observable is True."""
        time_rep = jax.lax.select(
            params.normalize_time, time_normalization(state.time), state.time
        )
        base_obs = jnp.hstack([state.pos, state.last_reward, state.last_action, time_rep])
        if self.fully_observable:
            return jnp.hstack([base_obs, state.goal])
        return base_obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        return state.time >= params.max_steps_in_episode

    @property
    def name(self) -> str:
        """Environment name."""
        return "PointRobot-misc"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        low = jnp.array([-params.max_force, -params.max_force], dtype=jnp.float32)
        high = jnp.array([params.max_force, params.max_force], dtype=jnp.float32)
        return spaces.Box(low, high, (2,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment (8 if fully observable, 6 if POMDP)."""
        dim = 8 if self.fully_observable else 6
        low = jnp.full((dim,), -jnp.finfo(jnp.float32).max, dtype=jnp.float32)
        high = jnp.full((dim,), jnp.finfo(jnp.float32).max, dtype=jnp.float32)
        return spaces.Box(low, high, (dim,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "last_action": spaces.Discrete(self.num_actions),
                "last_reward": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


class PointRobotDiscrete(PointRobot):
    """
    Discrete Point Robot environment with 5 cardinal actions:
        0: Stay  ( 0.0,  0.0)
        1: Right (+max_force, 0.0)
        2: Up    ( 0.0, +max_force)
        3: Left  (-max_force, 0.0)
        4: Down  ( 0.0, -max_force)
    """

    _ACTION_MAP = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=jnp.float32,
    )

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        # Map discrete action [0..4] to continuous 2D displacement
        continuous_action = self._ACTION_MAP[action] * params.max_force
        return super().step_env(key, state, continuous_action, params)

    @property
    def name(self) -> str:
        return "PointRobotDiscrete-misc"

    @property
    def num_actions(self) -> int:
        return 5

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(5)

    def render(self, state: EnvState, params: EnvParams):
        """Small utility for plotting the agent's state."""
        fig, ax = plt.subplots()
        angles = jnp.linspace(0, jnp.pi, 100)
        x, y = jnp.cos(angles), jnp.sin(angles)
        ax.plot(x, y, color="k")
        plt.axis("scaled")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.25, 1.25)
        ax.set_xticks([])
        ax.set_yticks([])

        circle = plt.Circle(
            (state.goal[0], state.goal[1]), radius=params.goal_radius, alpha=0.3
        )
        ax.add_artist(circle)

        circle = plt.Circle(
            (state.pos[0], state.pos[1]), radius=0.05, alpha=1, color="red"
        )
        ax.add_artist(circle)
        return fig, ax


def time_normalization(
    t: float, min_lim: float = -1.0, max_lim: float = 1.0, t_max: int = 100
) -> float:
    """Normalize time integer into range given max time."""
    return (max_lim - min_lim) * t / t_max + min_lim


def sample_agent_position(
    key: jax.Array, circle_radius: float, center_init: bool
) -> jax.Array:
    """Sample a random position in circle (or set position to center)."""
    key_radius, key_angle = jax.random.split(key)
    sampled_radius = jax.random.uniform(key_radius, minval=0, maxval=circle_radius)
    sampled_angle = jax.random.uniform(key_angle, minval=0, maxval=jnp.pi)

    pos = jax.lax.select(
        center_init,
        jnp.zeros(2),
        jnp.array(
            [
                sampled_radius * jnp.cos(sampled_angle),
                sampled_radius * jnp.sin(sampled_angle),
            ]
        ),
    )
    return pos
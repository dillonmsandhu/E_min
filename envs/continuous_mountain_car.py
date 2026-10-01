import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct


@struct.dataclass
class EnvState:
    position: float
    velocity: float
    time: int


@struct.dataclass
class EnvParams:
    min_action: float = -1.0
    max_action: float = 1.0
    min_position: float = -1.2
    max_position: float = 0.6
    max_speed: float = 0.07
    goal_position: float = 0.45
    goal_velocity: float = 0.0
    power: float = 0.0015
    gravity: float = 0.0025
    max_steps_in_episode: int = 999
    # Potential-based reward shaping parameters
    gamma: float = 0.99
    potential_scale: float = 30.0
    # Stochastic / Noise parameters
    slip_prob: float = 0.0  # Probability of tire slip on each step
    slip_force_scale: float = 0.0  # Fraction of force transmitted during slip (0.0 = total loss of traction)
    action_noise_std: float = 0.0  # Additive Gaussian noise standard deviation on commanded action
    transition_noise_std: float = 0.0  # Additive Gaussian noise standard deviation on velocity (terrain roughness)


class MountainCarDenseContinuous(environment.Environment):
    """
    Continuous Mountain Car with Potential-Based Reward Shaping (PBRS) and optional stochasticity.
    
    The potential function is proportional to gravitational potential relative
    to the valley bottom (x ≈ -0.5236, where sin(3x) = -1.0):
        Phi(s) = (sin(3 * position) + 1.0) / 2.0  in [0.0, 1.0]
    
    This ensures:
        - In the starting valley, Phi(s_0) ≈ 0.0.
        - Failing/timeout episodes yield Delta G ≈ 0.0, keeping returns around 0.
        - Successful episodes yield returns around +50 to +90, closely matching
          the sparse environment return distribution.
    """

    def __init__(self):
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def _potential(self, position: chex.Array) -> chex.Array:
        # Normalized height above the valley floor: 0 at valley bottom, ~0.988 at goal
        return 0.5 * (jnp.sin(3.0 * position) + 1.0)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: float,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition with potential-based shaping and optional noise."""
        key_slip, key_trans, key_act = jax.random.split(key, 3)

        # 1. Action noise and clipping
        noisy_action = action + params.action_noise_std * jax.random.normal(key_act, shape=())
        force = jnp.clip(noisy_action, params.min_action, params.max_action)

        # 2. Tire slip (traction loss): with probability slip_prob, wheels lose traction
        slipped = jax.random.uniform(key_slip, shape=()) < params.slip_prob
        effective_force = jnp.where(slipped, force * params.slip_force_scale, force)

        # 3. Update velocity with gravity, effective drive force, and transition noise
        velocity = (
            state.velocity
            + effective_force * params.power
            - jnp.cos(3 * state.position) * params.gravity
            + params.transition_noise_std * jax.random.normal(key_trans, shape=())
        )
        velocity = jnp.clip(velocity, -params.max_speed, params.max_speed)
        position = state.position + velocity
        position = jnp.clip(position, params.min_position, params.max_position)
        velocity = velocity * (
            1 - (position >= params.goal_position) * (velocity < 0)
        )

        is_goal = (position >= params.goal_position) * (
            velocity >= params.goal_velocity
        )
        # Base reward penalizes commanded action effort (-0.1 * action^2) plus goal reward
        base_reward = -0.1 * action ** 2 + 100.0 * is_goal

        # Zero-referenced potential: 0 at bottom, ~0.988 at goal
        phi_cur = self._potential(state.position)
        phi_next_raw = self._potential(position)
        phi_goal = self._potential(params.goal_position)
        phi_next = jnp.where(is_goal, phi_goal, phi_next_raw)

        shaping = params.gamma * phi_next - phi_cur
        reward = (base_reward + params.potential_scale * shaping).squeeze()

        # Update state and evaluate termination
        state = EnvState(position.squeeze(), velocity.squeeze(), state.time + 1)
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": self.discount(state, params), "slipped": slipped},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position in valley."""
        init_state = jax.random.uniform(key, shape=(), minval=-0.6, maxval=-0.4)
        state = EnvState(position=init_state, velocity=0.0, time=0)
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state [position, velocity]."""
        return jnp.array([state.position, state.velocity]).squeeze()

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal (reached goal or timed out)."""
        done_goal = (state.position >= params.goal_position) * (
            state.velocity >= params.goal_velocity
        )
        done_steps = state.time >= params.max_steps_in_episode
        done = jnp.logical_or(done_goal, done_steps)
        return done.squeeze()

    @property
    def name(self) -> str:
        return "MountainCarDenseContinuous-v0"

    @property
    def num_actions(self) -> int:
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        if params is None:
            params = self.default_params
        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(1,),
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )
        return spaces.Box(low, high, shape=(2,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        low = jnp.array(
            [params.min_position, -params.max_speed],
            dtype=jnp.float32,
        )
        high = jnp.array(
            [params.max_position, params.max_speed],
            dtype=jnp.float32,
        )
        return spaces.Dict(
            {
                "position": spaces.Box(low[0], high[0], (), dtype=jnp.float32),
                "velocity": spaces.Box(low[1], high[1], (), dtype=jnp.float32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


class MountainCarDenseDiscrete(MountainCarDenseContinuous):
    """
    Discrete (bang-bang) Mountain Car with identical dynamics and potential-based reward shaping.
    
    Actions:
        0: Full reverse (force = -1.0)
        1: Zero force / coast (force = 0.0)
        2: Full forward (force = +1.0)
        
    Reward & Dynamics:
        Exactly matches MountainCarDenseContinuous:
            force = action - 1.0  in {-1.0, 0.0, 1.0}
            base_reward = -0.1 * force^2 + 100.0 * is_goal
            shaped_reward = base_reward + potential_scale * (gamma * Phi(s') - Phi(s))
    """

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # Map discrete {0, 1, 2} -> continuous force {-1.0, 0.0, 1.0}
        force = (action.astype(jnp.float32) - 1.0).squeeze()
        return super().step_env(key, state, force, params)

    @property
    def name(self) -> str:
        return "MountainCarDenseDiscrete-v0"

    @property
    def num_actions(self) -> int:
        return 3

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(3)

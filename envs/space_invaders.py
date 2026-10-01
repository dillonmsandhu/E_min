"""JAX implementation of Space Invaders MinAtar environment with PBRS/Penalty and Stochastic Noise ablations.

Source:
github.com/kenjyoung/MinAtar/blob/master/minatar/environments/space_invaders.py

ENVIRONMENT DESCRIPTION - 'SpaceInvaders-MinAtar'
- Player controls cannon at bottom of screen and can shoot bullets at aliens.
- Aliens move across screen until one of them hits the edge.
- At this point all move down and switch directions.
- Current alien dir indicated by 2 channels (left/right) - active at position.
- Reward of +1 is given each time alien is shot and alien is removed.
- Aliens will also shoot bullets back at player.
- Alien speed increases when only few of them are left.
- When only one alien is left, it will move at one cell per frame.
- When wave of aliens is cleared, slightly faster new one will spawn.
- Standard Termination occurs when an alien or bullet hits the player.
- Non-Lethal (Fixed Horizon) mode: Player stays alive for max_steps_in_episode,
  taking a -1 penalty per enemy bullet or alien hit rather than terminating.
- Channels are encoded as follows: 'cannon':0, 'alien':1, 'alien_left':2,
  'alien_right':3, 'friendly_bullet':4, 'enemy_bullet':5
- Observation has dimensionality (10, 10, 6)
- Actions are encoded as follows: ['n','l','r','f'] (minimal) or ['n','l','u','r','d','f'] (full)
"""

from typing import Any, Tuple, Optional
import jax
import jax.numpy as jnp
from jax import lax
from flax import struct
import chex
from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """State of the environment."""
    pos: int
    f_bullet_map: chex.Array
    e_bullet_map: chex.Array
    alien_map: chex.Array
    alien_dir: int
    enemy_move_interval: int
    alien_move_timer: int
    alien_shot_timer: int
    ramp_index: int
    shot_timer: int
    ramping: bool
    time: int
    terminal: bool
    last_action: int = 0
    hits_taken: int = 0


@struct.dataclass
class EnvParams(environment.EnvParams):
    shot_cool_down: int = 5
    enemy_move_interval: int = 12
    enemy_shot_interval: int = 10
    max_steps_in_episode: int = 1000
    # Fixed-horizon & penalty ablations
    penalty_on_hit: bool = False       # If True: keep alive for fixed steps, -1 on hit rather than termination
    hit_penalty: float = 1.0           # Penalty deducted on enemy bullet or alien collision
    # Environmental stochasticity
    sticky_action_prob: float = 0.0    # Probability of repeating previous action
    instant_bullets: bool = False      # Hitscan bullets (0 flight delay)


class MinSpaceInvaders(environment.Environment):
    """JAX implementation of Space Invaders MinAtar environment."""

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 6)
        # Full action set: ['n','l','u','r','d','f']
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        # Minimal action set: ['n','l','r','f']
        self.minimal_action_set = jnp.array([0, 1, 3, 5])
        # Set active action set for environment
        if use_minimal_action_set:
            self.action_set = self.minimal_action_set
        else:
            self.action_set = self.full_action_set

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Perform single timestep state transition."""
        key_sticky, key_shoot = jax.random.split(key)

        # 1. Sticky actions: with probability sticky_action_prob, repeat last_action
        is_sticky = jax.random.uniform(key_sticky, shape=()) < params.sticky_action_prob
        effective_action_idx = jax.lax.select(is_sticky, state.last_action, action)
        a = self.action_set[effective_action_idx]

        # 2. Step agent (player move, bullet update, bullet collision check)
        state, bullet_hit = step_agent(a, state, params)

        # 3. Update aliens (movement, border checks, alien-player collision check)
        state, alien_hit = step_aliens(state, params)

        # 4. Shooting & rewards
        state, kill_reward = step_shoot(state, params, a)

        # 5. Reward calculation:
        # When penalty_on_hit is True: deduct hit_penalty for each bullet or alien hit
        # When penalty_on_hit is False: standard kill_reward (and hits cause terminal)
        hit_count = (bullet_hit + alien_hit).astype(jnp.float32)
        reward = jax.lax.select(
            params.penalty_on_hit,
            kill_reward - params.hit_penalty * hit_count,
            kill_reward,
        )

        # Update various timers & evaluate all terminal conditions
        shot_timer = state.shot_timer - (state.shot_timer > 0)
        alien_move_timer = state.alien_move_timer - 1
        alien_shot_timer = state.alien_shot_timer - 1

        # Reset alien map and increase speed if map is cleared
        reset_map_cond = jnp.count_nonzero(state.alien_map) == 0
        ramping_cond = jnp.logical_and(
            state.enemy_move_interval > 6, state.ramping
        )
        reset_ramp_cond = jnp.logical_and(reset_map_cond, ramping_cond)
        enemy_move_interval = state.enemy_move_interval - reset_ramp_cond
        ramp_index = state.ramp_index + reset_ramp_cond
        alien_map = jax.lax.select(
            reset_map_cond, state.alien_map.at[0:4, 2:8].set(1), state.alien_map
        )

        # Check game condition & no. steps for termination condition
        time = state.time + 1
        state = state.replace(time=time)
        done = self.is_terminal(state, params)
        terminal = done
        state = state.replace(
            shot_timer=shot_timer,
            alien_move_timer=alien_move_timer,
            alien_shot_timer=alien_shot_timer,
            enemy_move_interval=enemy_move_interval,
            ramp_index=ramp_index,
            alien_map=alien_map,
            time=time,
            terminal=terminal,
            last_action=effective_action_idx,
            hits_taken=state.hits_taken + hit_count.astype(jnp.int32),
        )

        info = {
            "discount": 1 - done,
            "hits_taken": state.hits_taken,
            "is_sticky": is_sticky,
        }
        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        state = EnvState(
            pos=5,
            f_bullet_map=jnp.zeros((10, 10)),
            e_bullet_map=jnp.zeros((10, 10)),
            alien_map=jnp.zeros((10, 10)).at[0:4, 2:9].set(1),
            alien_dir=-1,
            enemy_move_interval=params.enemy_move_interval,
            alien_move_timer=params.enemy_move_interval,
            alien_shot_timer=params.enemy_shot_interval,
            ramp_index=0,
            shot_timer=0,
            ramping=True,
            time=0,
            terminal=False,
            last_action=0,
            hits_taken=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Return observation from raw state trafo."""
        obs = jnp.zeros((10, 10, 6), dtype=jnp.bool_)
        obs = obs.at[9, state.pos, 0].set(True)
        obs = obs.at[:, :, 1].set(state.alien_map.astype(jnp.bool_))
        left_dir_cond = state.alien_dir < 0
        obs = jax.lax.select(
            left_dir_cond,
            obs.at[:, :, 2].set(state.alien_map.astype(jnp.bool_)),
            obs.at[:, :, 3].set(state.alien_map.astype(jnp.bool_)),
        )
        obs = obs.at[:, :, 4].set(state.f_bullet_map.astype(jnp.bool_))
        obs = obs.at[:, :, 5].set(state.e_bullet_map.astype(jnp.bool_))
        return obs.astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= params.max_steps_in_episode
        done_standard = jnp.logical_or(done_steps, state.terminal)
        return jax.lax.select(params.penalty_on_hit, done_steps, done_standard)

    @property
    def name(self) -> str:
        return "SpaceInvaders-MinAtar"

    @property
    def num_actions(self) -> int:
        return len(self.action_set)

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "pos": spaces.Discrete(10),
                "f_bullet_map": spaces.Box(0, 1, (10, 10)),
                "e_bullet_map": spaces.Box(0, 1, (10, 10)),
                "alien_map": spaces.Box(0, 1, (10, 10)),
                "alien_dir": spaces.Box(-1, 3, ()),
                "enemy_move_interval": spaces.Discrete(params.enemy_move_interval),
                "alien_move_timer": spaces.Discrete(params.enemy_move_interval),
                "alien_shot_timer": spaces.Discrete(params.enemy_shot_interval),
                "ramp_index": spaces.Discrete(2),
                "shot_timer": spaces.Discrete(1000),
                "ramping": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(action: int, state: EnvState, params: EnvParams) -> Tuple[EnvState, jax.Array]:
    """Resolve player action - fire, left, right."""
    fire_cond = jnp.logical_and(action == 5, state.shot_timer == 0)
    left_cond, right_cond = (action == 1), (action == 3)

    # Update position of agent
    pos = jax.lax.select(left_cond, jnp.maximum(0, state.pos - 1), state.pos)
    pos = jax.lax.select(right_cond, jnp.minimum(9, pos + 1), pos)

    # Spawn friendly bullet (unless instant hitscan bullets are enabled)
    spawn_bullet = fire_cond & (~params.instant_bullets)
    f_bullet_map = jax.lax.select(
        spawn_bullet,
        state.f_bullet_map.at[9, pos].set(1),
        state.f_bullet_map,
    )
    shot_timer = jax.lax.select(
        fire_cond, params.shot_cool_down, state.shot_timer
    )

    # Update Friendly Bullets and Enemy Bullets
    f_bullet_map = jnp.roll(f_bullet_map, -1, axis=0)
    f_bullet_map = f_bullet_map.at[9, :].set(0)

    e_bullet_map = jnp.roll(state.e_bullet_map, 1, axis=0)
    e_bullet_map = e_bullet_map.at[0, :].set(0)

    # Check for enemy bullet hit on player cannon at row 9
    bullet_hit = e_bullet_map[9, pos] > 0

    # If penalty_on_hit: clear the bullet so it does not re-hit next step, and do NOT terminate!
    # If standard: mark terminal!
    e_bullet_map = jax.lax.select(
        bullet_hit & params.penalty_on_hit,
        e_bullet_map.at[9, pos].set(0),
        e_bullet_map,
    )
    terminal = jax.lax.select(
        params.penalty_on_hit,
        state.terminal,
        jnp.logical_or(state.terminal, bullet_hit),
    )
    new_state = state.replace(
        pos=pos,
        f_bullet_map=f_bullet_map,
        e_bullet_map=e_bullet_map,
        shot_timer=shot_timer,
        terminal=terminal,
    )
    return new_state, bullet_hit


def step_aliens(state: EnvState, params: EnvParams) -> Tuple[EnvState, jax.Array]:
    """Update aliens - border and collision check."""
    alien_terminal_1 = state.alien_map[9, state.pos] > 0
    alien_move_cond = state.alien_move_timer == 0

    alien_move_timer = jax.lax.select(
        alien_move_cond,
        jnp.minimum(
            jnp.count_nonzero(state.alien_map), state.enemy_move_interval
        ),
        state.alien_move_timer,
    )
    cond1 = jnp.logical_and(
        jnp.sum(state.alien_map[:, 0]) > 0, state.alien_dir < 0
    )
    cond2 = jnp.logical_and(
        jnp.sum(state.alien_map[:, 9]) > 0, state.alien_dir > 0
    )
    alien_border_cond = jnp.logical_and(
        alien_move_cond, jnp.logical_or(cond1, cond2)
    )
    alien_dir = jax.lax.select(
        alien_border_cond, -1 * state.alien_dir, state.alien_dir
    )
    alien_terminal_2 = jnp.logical_and(
        alien_border_cond, jnp.sum(state.alien_map[9, :]) > 0
    )
    alien_map = jax.lax.select(
        alien_move_cond,
        (
            jax.lax.select(
                alien_border_cond,
                jnp.roll(state.alien_map, 1, axis=0),
                jnp.roll(state.alien_map, alien_dir, axis=1),
            )
        ),
        state.alien_map,
    )
    alien_terminal_3 = jnp.logical_and(alien_move_cond, alien_map[9, state.pos] > 0)

    # Jointly evaluate alien collision conditions
    alien_hit = (alien_terminal_1 + alien_terminal_2 + alien_terminal_3) > 0

    # If penalty_on_hit: reset alien formation if aliens reached bottom row so episode continues
    reset_aliens_bottom = alien_border_cond & (jnp.sum(alien_map[9, :]) > 0) & params.penalty_on_hit
    alien_map = jax.lax.select(
        reset_aliens_bottom,
        jnp.zeros((10, 10)).at[0:4, 2:8].set(1),
        alien_map,
    )
    terminal = jax.lax.select(
        params.penalty_on_hit,
        state.terminal,
        jnp.logical_or(state.terminal, alien_hit),
    )
    new_state = state.replace(
        alien_move_timer=alien_move_timer,
        alien_dir=alien_dir,
        alien_map=alien_map,
        terminal=terminal,
    )
    return new_state, alien_hit


def step_shoot(state: EnvState, params: EnvParams, player_action: int) -> Tuple[EnvState, float]:
    """Update aliens - shooting check and calculate rewards."""
    alien_shot_cond = state.alien_shot_timer == 0
    alien_shot_timer = jax.lax.select(
        alien_shot_cond, params.enemy_shot_interval, state.alien_shot_timer
    )

    # Nearest alien shooting target
    alien_exists, loc, idx = get_nearest_alien(state.pos, state.alien_map)
    update_aliens_cond = jnp.logical_and(alien_shot_cond, alien_exists)
    e_bullet_map = jax.lax.select(
        update_aliens_cond,
        state.e_bullet_map.at[loc, idx].set(1),
        state.e_bullet_map,
    )

    # 1. Standard traveling bullet kills
    traveling_kills = jnp.logical_and(
        state.alien_map, state.alien_map == state.f_bullet_map
    )

    # 2. Instant hitscan laser kills (if enabled)
    hitscan_fire = (player_action == 5) & (state.shot_timer == params.shot_cool_down) & params.instant_bullets
    aliens_in_col = state.alien_map[:, state.pos] > 0
    hitscan_hit = hitscan_fire & (jnp.sum(aliens_in_col) > 0)
    lowest_row = jnp.max(jnp.where(aliens_in_col, jnp.arange(10), -1))
    hitscan_kills = jnp.zeros((10, 10), dtype=bool).at[lowest_row, state.pos].set(hitscan_hit)

    all_kills = jnp.logical_or(traveling_kills, hitscan_kills)

    # Reward based on killed aliens
    kill_reward = jnp.sum(all_kills).astype(jnp.float32)

    # Delete killed aliens and consumed friendly bullets
    alien_map = state.alien_map * (1 - all_kills)
    f_bullet_map = state.f_bullet_map * (1 - all_kills)

    return (
        state.replace(
            alien_shot_timer=alien_shot_timer,
            e_bullet_map=e_bullet_map,
            alien_map=alien_map,
            f_bullet_map=f_bullet_map,
        ),
        kill_reward,
    )


def get_nearest_alien(pos: int, alien_map: chex.Array) -> Tuple[int, int, int]:
    """Find alien closest to player in manhattan distance -> shot target."""
    ids = jnp.array([jnp.abs(jnp.array([i for i in range(10)]) - pos)])
    search_order = jnp.argsort(ids).squeeze()
    results_temp = jnp.zeros(3)
    aliens_exist = jnp.sum(alien_map, axis=0) > 0

    counter = 0
    for i in search_order[::-1]:
        locations = alien_map[:, i] * jnp.arange(alien_map[:, i].shape[0])
        aliens_loc = jnp.max(locations)
        results_temp = (
            aliens_exist[i]
            * results_temp.at[:].set(
                jnp.array([aliens_exist[i], aliens_loc, i])
            )
            + (1 - aliens_exist[i]) * results_temp
        )
        counter += 1
    results_temp = jnp.array(results_temp, dtype=int)
    return results_temp[0], results_temp[1], results_temp[2]

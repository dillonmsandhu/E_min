"""Core components for MuJoCo Playground (MJX) reinforcement learning."""

from mjx_rl.core.mjx_config import (
    DEFAULT_DMC_5_ENVS,
    DEFAULT_LOCOMOTION_5_ENVS,
    config as default_config,
    get_dmc_brax_parity_config,
    get_locomotion_brax_parity_config,
)
from mjx_rl.core.mjx_critics import (
    CRITIC_LOSS_FNS,
    calculate_gae,
    calculate_targets,
    get_critic_loss_fn,
    get_critic_target_lambda,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
    ppo_loss_fn,
    ppo_value_loss,
    prepare_critic_transitions,
    td_0_loss_fn,
    td_0_value_loss,
)
from mjx_rl.core.mjx_network import (
    ActorCritic,
    ActorCriticDualCritic,
    BraxTanhNormalDistribution,
    DualTrainState,
    initialize_split_train_state,
)
from mjx_rl.core.mjx_utils import (
    Transition,
    get_lr,
    load_config,
    load_config_dict,
    save_plot,
    save_results,
    scale_rms,
    shuffle_and_batch,
    warmup_and_reset_stats,
)
from mjx_rl.core.mjx_wrappers import (
    ClipAction,
    NormalizeVecObservation,
    NormalizeVecReward,
    PlaygroundAutoResetWrapper,
    PlaygroundGymnaxWrapper,
    VecEnv,
)

# Re-exporting helpers from gymnax_rl.core for backward compatibility
from gymnax_rl.core.gymnax_wrappers import make_env
from gymnax_rl.core.gymnax_utils import shuffle_and_batch, post_process_advantage, Transition, scale_rms
from gymnax_rl.core.gymnax_critics import (
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    prepare_geometric_jump_transitions,
    prepare_critic_transitions,
    huber_loss,
    ppo_value_loss,
    td_0_value_loss,
    e_lambda_geometric_critic_loss,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
    td_0_critic_loss_fn,
    e_0_critic_loss_fn,
    e_lambda_geometric_critic_loss_fn,
    get_critic_loss_fn,
    SPLIT_CRITIC_LOSS_FNS,
)

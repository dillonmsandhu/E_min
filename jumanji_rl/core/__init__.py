# jumanji_rl/core/__init__.py
# Core components for Jumanji RL

from jumanji_rl.core.jumanji_config import config as default_config
from jumanji_rl.core.jumanji_wrappers import (
    JumanjiEnvironment,
    JumanjiEnvParams,
    JumanjiEnvState,
    make_env,
)
from jumanji_rl.core.jumanji_network import (
    DiscretePolicyHead,
    MLP,
    PQN_CNN,
    MLP_Q_AC,
    PQN_Q_AC,
    ActorCritic,
    DualTrainState,
    uses_aux_network,
    initialize_split_train_state,
    initialize_q_actor_critic,
    initialize_actor_critic,
)
from jumanji_rl.core.jumanji_critics import (
    extract_q_taken,
    compute_aux_head_loss,
    calculate_gae,
    calculate_targets,
    get_critic_target_lambda,
    ppo_actor_loss_fn,
    ppo_critic_loss_fn,
    sarsa_critic_loss_fn,
    e_0_q_critic_loss_fn,
    e_lambda_q_critic_loss_fn,
    get_critic_loss_fn,
    get_q_critic_loss_fn,
    Q_CRITIC_LOSS_FNS,
)
from jumanji_rl.core.jumanji_utils import (
    Transition,
    QTransition,
    prepare_q_jump_transitions,
    shuffle_and_batch,
    scale_rms,
    post_process_advantage,
    compute_runtime_metrics,
)

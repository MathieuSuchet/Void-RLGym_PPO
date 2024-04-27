import pickle

import rlgym_sim
from rlgym_ppo import Learner
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.reward_functions.common_rewards import EventReward, VelocityBallToGoalReward, \
    LiuDistancePlayerToBallReward, FaceBallReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

import wandb
from logger import Logger
from rlgym1_assets.action_parsers.action_parsers import WandbActionParser, LookupAction
from rlgym1_assets.rewards.rewards import LoggerCombinedReward, FlipResetReward
from rlgym1_assets.state_mutators.state_mutators import DefaultState, ShotState, DynamicScoredReplaySetter
from rlgym1_assets.terminal_conditions.multi_condition import MultiLoggedCondition
from rlgym1_assets.wandb_loggers.ball_loggers import get_all_ball_loggers
from rlgym1_assets.wandb_loggers.global_loggers import get_all_global_loggers
from rlgym1_assets.wandb_loggers.player_loggers import get_all_player_loggers

TICK_RATE = 1. / 120.
tick_skip = 8

n_proc = 10
ts_per_iteration = 100_000
timestep_limit = ts_per_iteration * 10_000
ppo_batch_size = ts_per_iteration // 2
n_epochs = 10
ppo_minibatch_size = ppo_batch_size // n_epochs

# Aech's rlgym_v2 example's approximation
min_inference_size = max(1, int(round(n_proc * 0.9)))

spawn_opponents = True
blue_count = 3
orange_count = 3 if spawn_opponents else 0

logger = Logger(
        *get_all_global_loggers(),
        *get_all_ball_loggers(),
        *get_all_player_loggers()
    )

action_parser = WandbActionParser(LookupAction())
obs_builder = AdvancedObs()
reward_fn = LoggerCombinedReward(
    EventReward(
        goal=1,
        concede=-1,
        touch=.01,
        shot=.1,
        save=.1
    ),
    (
        VelocityBallToGoalReward(),
        .1
    ),
    (
        LiuDistancePlayerToBallReward(),
        .1
    ),
    (
        FaceBallReward(),
        .1
    ),
    (
        FlipResetReward(),
        .1
    )
)

total_timeout = 2
termination_conditions = [MultiLoggedCondition(
    # GoalCondition(),
    TimeoutCondition(int(total_timeout / TICK_RATE)),
    BallTouchedCondition()
)]

rendered = False
continue_run = False  # If you did a run already, and you are continuing (make sure to give the run's id)


def create_env():

    with open("tmp/replay_setter", "rb") as f:
        dynamic_replay_setter = pickle.load(f)

    state_mutator = WeightedSampleSetter(
        state_setters=(
            DefaultState(),
            ShotState(),
            dynamic_replay_setter,
        ),
        weights=(1, 1, 1)
    )

    rlgym_env = rlgym_sim.make(
        state_setter=state_mutator,
        action_parser=action_parser,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        terminal_conditions=termination_conditions,
        tick_skip=tick_skip,
        team_size=blue_count,
        spawn_opponents=spawn_opponents,
    )

    return rlgym_env


if __name__ == "__main__":

    dynamic_replay_setter = DynamicScoredReplaySetter(
                "replays/states_scores_duels.npz",
                "replays/states_scores_doubles.npz",
                "replays/states_scores_standard.npz"
            )

    with open("tmp/replay_setter", "wb") as f:
        pickle.dump(dynamic_replay_setter, f)

    config = {
        "project": "void",
        "entity": "madaos",
        "name": "void"
    }

    if continue_run:
        run_id = "ldokyvda"
        wandb_run = wandb.init(
            entity=config["entity"],
            name=config["name"],
            project=config["project"],
            resume=True,
            id=run_id)
    else:
        wandb_run = None

    agent = Learner(
        env_create_function=create_env,

        ts_per_iteration=ts_per_iteration,
        timestep_limit=timestep_limit,
        ppo_batch_size=ppo_batch_size,
        ppo_minibatch_size=ppo_minibatch_size,


        ppo_epochs=n_epochs,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),

        # checkpoint_load_folder="data/rl_model/-1712845896901723900/28001784",
        checkpoints_save_folder="data/rl_model/",
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=logger,

        log_to_wandb=True,
        wandb_run_name=config['name'],
        wandb_group_name=config['entity'],
        wandb_project_name=config['project'],
        load_wandb=continue_run,
        wandb_run=wandb_run,

        render=rendered,
        render_delay=(120/8) if rendered else 0,

        device="cuda"
    )

    dynamic_replay_setter.load_replays()

    agent.learn()

import pickle

import rlgym_sim
from rlgym_ppo import Learner
from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder

import wandb
from logger import Logger
from rlgym1_assets.action_parsers.action_parsers import WandbActionParser, LookupAction
from rlgym1_assets.rewards.NewRewards import PlayerVelocityReward, KickoffReward_MMR, DribbleReward
from rlgym1_assets.rewards.rewards import LoggerCombinedReward, BumpReward, GoalScoreSpeed, \
    SaveBoostReward, BoostPickupReward
from rlgym1_assets.state_mutators.state_mutators import DefaultState, DynamicScoredReplaySetter, \
    TeamSizeSetter
from rlgym1_assets.terminal_conditions.multi_condition import MultiLoggedCondition


tick_skip = 8
STEP_TIME = tick_skip / 120.


n_proc = 55
ts_per_iteration = 1_000_000
timestep_limit = ts_per_iteration * 10_000
ppo_batch_size = 200_000
n_epochs = 3
ppo_minibatch_size = ppo_batch_size // 10

# Aech's rlgym_v2 example's approximation
min_inference_size = max(1, int(round(n_proc * 0.9)))

spawn_opponents = True
blue_count = 3
orange_count = 3 if spawn_opponents else 0

logger = Logger(
    # *get_all_global_loggers(),
    # *get_all_ball_loggers(),
    # *get_all_player_loggers()
)

action_parser = WandbActionParser(LookupAction())
obs_builder = AdvancedObsPadder()
reward_fn = LoggerCombinedReward(
    EventReward(
        goal=1,
        team_goal=1.2,
        concede=-1,
        touch=0.5,
        shot=0.6,
        save=0.7,
        demo=0.8,
        boost_pickup=0.5
    ),
    (BumpReward(), 0.001),
    (PlayerVelocityReward(), 0.001),
    (GoalScoreSpeed(), 30),
    (KickoffReward_MMR(), 30),
    (SaveBoostReward(), 5),
    (BoostPickupReward(), 100),
    (DribbleReward(), 0.03)
)

total_timeout = 2
termination_conditions = [MultiLoggedCondition(
    GoalScoredCondition(),
    TimeoutCondition(int(total_timeout / STEP_TIME)),
)]

rendered = False
continue_run = False  # If you did a run already, and you are continuing (make sure to give the run's id)


def create_env():
    with open("tmp/replay_setter", "rb") as f:
        dynamic_replay_setter = pickle.load(f)

    state_mutator = TeamSizeSetter(
        gm_probs=(0.3, 0.4, 0.4),
        setters=(
            DefaultState(),
            dynamic_replay_setter
        ),
        weights=(1, 1)
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
        "project": "Void",
        "entity": "madaos",
        "name": "Void"
    }

    if continue_run:
        run_id = "rck3n25q"
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
        exp_buffer_size=ppo_batch_size,

        ppo_epochs=n_epochs,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),

        # checkpoint_load_folder="data/rl_model/-1714574662835816800/32001932",
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
        render_delay=(120 / 8) if rendered else 0,

        device="cuda"
    )

    dynamic_replay_setter.load_replays()

    agent.learn()

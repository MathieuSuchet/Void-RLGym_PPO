from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction
from rlgym.rocket_league.done_conditions import TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.done_conditions.goal_condition import GoalCondition
from rlgym.rocket_league.game.game_engine import GameEngine
from rlgym.rocket_league.obs_builders.default_obs import DefaultObs
from rlgym.rocket_league.sim import RLViserRenderer
from rlgym.rocket_league.sim.rocketsim_engine import RocketSimEngine
from rlgym.rocket_league.state_mutators import MutatorSequence, KickoffMutator, FixedTeamSizeMutator
from rlgym_ppo import Learner
from rlgym_ppo.util import RLGymV2GymWrapper

import wandb
from done_conditions import LoggedAnyCondition
from logger import Logger
from rewards import LoggerCombinedReward, VelBallToGoalReward, LiuDistancePlayerToBallReward, EventReward
from wandb_loggers import BallVelocityLogger, BallHeightLogger, TouchLogger, GoalLogger, PlayerVelocityLogger

TICK_RATE = 1. / 120.
tick_skip = 8

n_proc = 1
ts_per_iteration = 10_000
timestep_limit = ts_per_iteration * 100
ppo_batch_size = ts_per_iteration
n_epochs = 10
ppo_minibatch_size = ppo_batch_size // n_epochs

# Aech's rlgym_v2 example's approximation
min_inference_size = max(1, int(round(n_proc * 0.9)))

blue_count = orange_count = 3

state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_count, orange_count), KickoffMutator())
action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
obs_builder = DefaultObs()
reward_fn = LoggerCombinedReward(
    EventReward(
        goal_w=100,
        concede_w=-100,
        touch_w=10
    ),
    (
        VelBallToGoalReward(),
        2
    ),
    (
        LiuDistancePlayerToBallReward(),
        2
    )
)

total_timeout = 3
termination_conditions = LoggedAnyCondition(
    GoalCondition(),
    TimeoutCondition(total_timeout / TICK_RATE),
    name="Terminations"
)

no_touch_timeout = 1
truncation_conditions = LoggedAnyCondition(
    NoTouchTimeoutCondition(no_touch_timeout / TICK_RATE),
    name="Truncations"
)

simulator = True  # Since the plugin is not implemented, game engine is just empty (Leave it to True)
rendered = False  # Make sure you got rlviser in the folder
continue_run = False  # If you did a run already, and you are continuing (make sure to give the run's id)


def create_env():
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        action_parser=action_parser,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        termination_cond=termination_conditions,
        truncation_cond=truncation_conditions,

        transition_engine=RocketSimEngine() if simulator else GameEngine(),
        renderer=RLViserRenderer() if rendered else None
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    config = {
        "project": "rlgym_ppo_tests",
        "entity": "madaos",
        "name": "logger_run"
    }

    if continue_run:
        run_id = "id_to_continue"  # TODO: change the id to match the last wandb run's id
        wandb_run = wandb.init(
            entity=config["entity"],
            name=config["name"],
            project=config["project"],
            resume=True,
            id=run_id)
    else:
        wandb_run = None

    logger = Logger(
        BallHeightLogger(),
        TouchLogger(),
        GoalLogger(),
        BallVelocityLogger(),
        PlayerVelocityLogger()
    )

    agent = Learner(
        env_create_function=create_env,

        ts_per_iteration=ts_per_iteration,
        timestep_limit=timestep_limit,
        ppo_batch_size=ppo_batch_size,
        ppo_minibatch_size=ppo_minibatch_size,
        ppo_epochs=n_epochs,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),

        # checkpoint_load_folder="data/rl_model/-1711460520183028600/100002/",
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

    agent.learn()

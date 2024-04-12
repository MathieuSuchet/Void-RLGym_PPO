import time
from itertools import chain

import rlviser_py
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.game import GameEngine
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.sim import RocketSimEngine, RLViserRenderer
from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
from rlgym_ppo import Learner
from rlgym_ppo.ppo import PPOLearner
from rlgym_ppo.util import RLGymV2GymWrapper

from action_parsers import WandbActionParser
from done_conditions import LoggedAnyCondition
from rewards import LoggerCombinedReward, EventReward, VelBallToGoalReward, LiuDistancePlayerToBallReward, \
    FaceBallReward
from state_mutators import WeightedStateMutator

TICK_RATE = 1. / 120.
tick_skip = 8

blue_count = orange_count = 3

state_mutator = MutatorSequence(FixedTeamSizeMutator(blue_count, orange_count), WeightedStateMutator(KickoffMutator()))
action_parser = WandbActionParser(RepeatAction(LookupTableAction(), repeats=tick_skip))
obs_builder = DefaultObs()
reward_fn = LoggerCombinedReward(
    EventReward(
        goal_w=1,
        concede_w=-1,
        touch_w=.01,
        shot_w=.1,
        save_w=.1
    ),
    (
        VelBallToGoalReward(),
        .1
    ),
    (
        LiuDistancePlayerToBallReward(),
        .1
    ),
    (
        FaceBallReward(),
        .1
    )
)

total_timeout = 3
termination_conditions = LoggedAnyCondition(
    GoalCondition(),
    TimeoutCondition(total_timeout / TICK_RATE),
    # BallTouchedCondition(),
    name="Terminations"
)

no_touch_timeout = 1
truncation_conditions = LoggedAnyCondition(
    NoTouchTimeoutCondition(no_touch_timeout / TICK_RATE),
    name="Truncations"
)

simulator = True  # Since the plugin is not implemented, game engine is just empty (Leave it to True)
deterministic = False

model_to_load = "data/rl_model/-1712845896901723900/28001784"


def create_env():
    rlgym_env = RLGym(
        state_mutator=state_mutator,
        action_parser=action_parser,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        termination_cond=termination_conditions,
        truncation_cond=truncation_conditions,

        transition_engine=RocketSimEngine() if simulator else GameEngine(),
        renderer=RLViserRenderer()
    )

    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":

    agent = PPOLearner(
            obs_builder.get_obs_space('blue-0'),
            action_parser.get_action_space('blue-0'),
            device="cuda",
            batch_size=10_000,
            mini_batch_size=1_000,
            n_epochs=10,
            continuous_var_range=(0.0, 1.0),
            policy_type=0,
            policy_layer_sizes=(256, 256, 256),
            critic_layer_sizes=(256, 256, 256),
            policy_lr=3e-4,
            critic_lr=3e-4,
            clip_range=0.2,
            ent_coef=0.005)

    agent.load_from(model_to_load)

    env = create_env()

    while True:
        obs_dict = env.reset()
        steps = 0
        t0 = time.time()
        while True:
            env.render()
            time.sleep(6 / 120)

            actions, _ = agent.policy.get_action(obs_dict, deterministic)
            actions = actions.detach().numpy().reshape((actions.shape[0], 1))

            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)

            steps += 1

            if any(chain(terminated_dict.values(), truncated_dict.values())):
                break

import time

from rlgym import make
from rlgym.gamelaunch import LaunchPreference
from rlgym_ppo.ppo import PPOLearner
from rlgym_sim.utils.obs_builders import AdvancedObs
from rlgym_sim.utils.reward_functions.common_rewards import VelocityBallToGoalReward, EventReward, \
    LiuDistancePlayerToBallReward, FaceBallReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

from rlgym1_assets.action_parsers.action_parsers import WandbActionParser, LookupAction
from rlgym1_assets.rewards.rewards import LoggerCombinedReward
from rlgym1_assets.state_mutators.state_mutators import DefaultState, ShotState
from rlgym1_assets.terminal_conditions.multi_condition import MultiLoggedCondition

TICK_RATE = 1. / 120.
tick_skip = 8

blue_count = orange_count = 3

spawn_opponents = True
blue_count = 3
orange_count = 3 if spawn_opponents else False

state_mutator = WeightedSampleSetter(
    state_setters=(
        DefaultState(),
        ShotState()
    ),
    weights=(1, 1)
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
    )
)

total_timeout = 2
termination_conditions = [MultiLoggedCondition(
    # GoalCondition(),
    TimeoutCondition(int(total_timeout / TICK_RATE)),
    BallTouchedCondition()
)]
deterministic = False

model_to_load = "data/rl_model/-1712845896901723900/28001784"


def create_env():
    rlgym_env = make(
        state_setter=state_mutator,
        action_parser=action_parser,
        obs_builder=obs_builder,
        reward_fn=reward_fn,
        terminal_conditions=termination_conditions,
        tick_skip=tick_skip,
        team_size=blue_count,
        spawn_opponents=spawn_opponents,
        game_speed=1,
        launch_preference=LaunchPreference.STEAM)

    return rlgym_env


if __name__ == "__main__":

    agent = PPOLearner(
            231,
            90,
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
        obs = env.reset()
        steps = 0
        t0 = time.time()
        while True:
            actions, _ = agent.policy.get_action(obs, deterministic)
            actions = actions.detach().numpy().reshape((actions.shape, 1))

            obs, reward, terminated, info = env.step(actions)

            steps += 1

            if terminated:
                obs = env.reset()

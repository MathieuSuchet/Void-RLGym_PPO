from rlgym import make
from rlgym.gamelaunch import LaunchPreference
from rlgym_ppo.ppo import PPOLearner
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_sim.utils.reward_functions.common_rewards import VelocityBallToGoalReward, EventReward, \
    LiuDistancePlayerToBallReward, FaceBallReward, ConstantReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition, BallTouchedCondition

from rlgym1_assets.action_parsers.action_parsers import WandbActionParser, LookupAction
from rlgym1_assets.rewards.rewards import LoggerCombinedReward
from rlgym1_assets.state_mutators.state_mutators import DefaultState, ShotState, TeamSizeSetter, \
    DynamicScoredReplaySetter
from rlgym1_assets.terminal_conditions.multi_condition import MultiLoggedCondition

tick_skip = 8
STEP_TIME = tick_skip / 120.



spawn_opponents = True
blue_count = 3
orange_count = 3 if spawn_opponents else False

dynamic_replay = DynamicScoredReplaySetter(
                "replays/states_scores_duels.npz",
                "replays/states_scores_doubles.npz",
                "replays/states_scores_standard.npz"
)
dynamic_replay.load_replays()
state_mutator = TeamSizeSetter(
    setters=(
        DefaultState(),
        dynamic_replay
    ),
    weights=(1, 1),
    gm_probs=(0.3, 0.4, 0.3)
)
action_parser = WandbActionParser(LookupAction())
obs_builder = AdvancedObsPadder()
reward_fn = LoggerCombinedReward(ConstantReward())

total_timeout = 2
termination_conditions = [MultiLoggedCondition(
    GoalScoredCondition(),
    TimeoutCondition(int(total_timeout / STEP_TIME)),
)]
deterministic = False

model_to_load = "data/rl_model/-1714578008627305700/152007804"


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
        launch_preference=LaunchPreference.EPIC
    )

    return rlgym_env


if __name__ == "__main__":

    agent = PPOLearner(
            obs_space_size=237,
            act_space_size=action_parser.get_action_space().n,
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
        while True:
            actions, _ = agent.policy.get_action(obs, deterministic)
            actions = actions.detach().numpy().reshape((*actions.shape, 1))

            obs, reward, terminated, info = env.step(actions)

            if terminated:
                obs = env.reset()
# region ========================= Imports =========================
import os
import random
import time

import numpy as np
import torch
from rlgym.gamelaunch import LaunchPreference
from rlgym_ppo.ppo import PPOLearner
from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import EventReward
from rlgym_sim.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder

from rlgym1_assets.action_parsers.action_parsers import WandbActionParser, LookupAction
from rlgym1_assets.rewards.NewRewards import PlayerVelocityReward, KickoffReward_MMR, SaveBoostReward, \
    BoostPickupReward, DribbleReward
from rlgym1_assets.rewards.rewards import BumpReward, GoalScoreSpeed
from rlgym1_assets.state_mutators.state_mutators import DefaultState, TeamSizeSetter, \
    DynamicScoredReplaySetter
from rlgym1_assets.terminal_conditions.multi_condition import MultiLoggedCondition
from utils import get_latest_model_path, live_log

# endregion
# region ========================= Environment settings ============================
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
# dynamic_replay.load_replays()
state_mutator = TeamSizeSetter(
    setters=(
        DefaultState(),
        # dynamic_replay
    ),
    weights=(1,),
    gm_probs=(0.3, 0.4, 0.3)
)

rewards_functions = (
    EventReward(
        goal=1,
        team_goal=1.2,
        concede=-1,
        touch=0.75,
        shot=0.6,
        save=0.7,
        demo=0.8,
        boost_pickup=0.5
    ),
    BumpReward(),
    PlayerVelocityReward(),
    GoalScoreSpeed(),
    KickoffReward_MMR(),
    SaveBoostReward(),
    BoostPickupReward(),
    DribbleReward()
)

rewards_weights = (
    0.5, 0.01, 0.0007, 30, 20, 0.0005, 0.01, 0.01
)
reward_fn = CombinedReward(
    reward_functions=rewards_functions,
    reward_weights=rewards_weights
)

total_timeout = 20
termination_conditions = [MultiLoggedCondition(
    GoalScoredCondition(),

    TimeoutCondition(int(total_timeout / STEP_TIME)),
)]
# endregion
# region ========================= Model Settings =============================

action_parser = WandbActionParser(LookupAction())
obs_builder = AdvancedObsPadder()

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

# endregion
# region ========================= Live instance Settings =============================
deterministic = True

model_to_load = get_latest_model_path("data/rl_model")

minutes_before_update = 15
seconds_before_update = 0
time_before_update = minutes_before_update * 60 + seconds_before_update
current_time = 0
last_ep_reward = 0

# endregion
# region ========================= Live instance functions =========================

def create_env(sim: bool = True):
    if sim:
        import rlgym_sim
        rlgym_env = rlgym_sim.make(
            state_setter=state_mutator,
            action_parser=action_parser,
            obs_builder=obs_builder,
            reward_fn=reward_fn,
            terminal_conditions=termination_conditions,
            tick_skip=tick_skip,
            team_size=blue_count,
            spawn_opponents=spawn_opponents,
            # game_speed=1,
            # launch_preference=LaunchPreference.EPIC
        )
    else:
        import rlgym
        rlgym_env = rlgym.make(
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


def model_reload():
    global agent, current_time
    latest_model_path = get_latest_model_path("data/rl_model")
    agent.load_from(latest_model_path)
    live_log("Model reloaded")
    current_time = time.time()


def playstyle_switch():
    global deterministic

    deterministic = random.uniform(0, 1) > 0.5


def print_live_state():
    ttu = time_before_update - (time.time() - current_time)
    os.system('cls')
    live_log(" === State report === ")
    live_log(f" Mode : {'Deterministic' if deterministic else 'Stochastic'}")
    live_log(f" Model reload in {int(ttu // 60):02d}:{int(ttu % 60):02d}")
    live_log(f" Last episode reward (Average per player) : {last_ep_reward:.6f}")

# endregion
# region ========================= Main loop =========================

if __name__ == "__main__":

    agent.load_from(model_to_load)

    env = create_env(sim=True)
    current_time = time.time()
    refresh_time = current_time
    rewards = []

    while True:
        playstyle_switch()
        obs = env.reset()
        if len(rewards) > 0:
            last_ep_reward = sum(rewards) / len(rewards)
        rewards = []
        terminated = False
        while not terminated:
            if time.time() - refresh_time >= 1:
                refresh_time = time.time()
                print_live_state()

            with torch.no_grad():
                actions = np.array([agent.policy.get_action(o, deterministic)[0] for o in obs])
                actions = actions.reshape((*actions.shape, 1))

            obs, reward, terminated, info = env.step(actions)
            rewards.append(sum(reward) / len(reward))

            if time.time() - current_time >= time_before_update:
                model_reload()

            time.sleep(STEP_TIME)

# endregion
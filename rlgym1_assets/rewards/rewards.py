import math
from typing import Tuple, Union

import numpy as np
from numpy import ndarray
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import *
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward

from logger import _add, Logger
from rlgym1_assets.wandb_loggers.global_loggers import FlipResetLogger
from rlgym1_assets.wandb_loggers.player_loggers import TouchForceLogger, PlayerSupersonicTimeLogger


class LoggerCombinedReward(RewardFunction):

    def __init__(self, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]):
        reward_fns = []
        weights = []

        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            reward_fns.append(r)
            weights.append(w)

        self.reward_fns = tuple(reward_fns)
        self.weights = tuple(weights)
        self.logger = Logger()

        self.reward_logs = []

    def reset(self, initial_state: GameState):
        for r in self.reward_fns:
            r.reset(initial_state)

        if len(self.reward_logs) == 0:
            return

        self.reward_logs = np.sum(self.reward_logs, axis=0)
        data = {"Rewards": {}}
        for i, reward_fn in enumerate(self.reward_fns):
            data["Rewards"].setdefault(reward_fn.__class__.__name__, {
                "value": float(self.reward_logs[i]),
                "nb_episodes": 1
            })
        self.logger.add_result(data, _add)
        self.reward_logs = []

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_fns
        ]

        self.reward_logs.append([rew * w for rew, w in zip(rewards, self.weights)])

        return float(np.dot(self.weights, rewards))

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_fns
        ]

        self.reward_logs.append([rew * w for rew, w in zip(rewards, self.weights)])

        return float(np.dot(self.weights, rewards))


class EnhancedEventReward(RewardFunction):

    def __init__(
            self,
            goal_w: float,
            team_goal_w: float,
            concede_w: float,
            touch_w: float,
            shot_w: float,
            save_w: float,
            demo_w: float,
            flip_reset_w: float,
    ):
        """
        Rewards when events occur
        :param goal_w: Weight of goal event
        :param team_goal_w: Weight of team goal event
        :param concede_w: Weight of goal concede event
        :param touch_w: Weight of touch event
        :param shot_w: Weight of shot event
        :param save_w: Weight of save event
        :param demo_w: Weight of demo event
        :param flip_reset_w: Weight of demo event
        """
        self.weights = np.array([goal_w, team_goal_w, concede_w, touch_w, shot_w, save_w, demo_w,
                                 flip_reset_w])

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return np.dot(np.array(self._extract_values(player, state)), self.weights)

    def _extract_values(self, player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes])


class TouchForceReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def __init__(self, force_threshold: float = 1):
        """
        Rewards hard touches
        :param force_threshold: Minimum threshold to trigger (kph)
        """
        self.force_threshold = force_threshold
        self.touch_force_logger = TouchForceLogger(standalone_mode=True)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        metric = self.touch_force_logger.collect_metrics(state, player.car_id)
        return metric / BALL_MAX_SPEED if metric * 0.036 > self.force_threshold else 0


class SupersonicReward(RewardFunction):
    def __init__(self):
        self.logger = PlayerSupersonicTimeLogger(standalone_mode=True)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        metric = self.logger.collect_metrics(state, player.car_id)
        return 1 if metric > 0 else 0


class FlipResetReward(RewardFunction):
    def __init__(self):
        self.flip_reset_logger = FlipResetLogger(standalone_mode=True)

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        has_reset = self.flip_reset_logger.collect_metrics(state, player.car_id)
        return int(has_reset)  # 0 = False, 1 = True


class BumpReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_team = player.team_num  # 0 = Blue; 1 = Orange

        for agent in state.players:
            if agent.team_num == player_team:
                rel_pos = player.car_data.position - agent.car_data.position
                dist_to_agent = np.linalg.norm(rel_pos)

                if 250 > dist_to_agent > 1:
                    # print(f"Bumped.  Distance to Agent: {np.round(distToAgent,5)}")
                    return -2
                return 0
            elif agent.team_num != player_team:
                rel_pos = player.car_data.position - agent.car_data.position
                dist_to_agent = np.linalg.norm(rel_pos)

                if 170 > dist_to_agent > 1:
                    # print(f"Opponent Bumped.  Distance to Agent: {np.round(distToAgent,5)}")
                    return 1
            return 0
        return 0


class TestReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.car_id == 1:
            print(state.ball.position.item(2))
        return 0


class DistToBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        distance = 1 / math.sqrt((state.ball.position.item(0) - player.car_data.position.item(0)) ** 2 +
                                 (state.ball.position.item(1) - player.car_data.position.item(1)) ** 2 +
                                 (state.ball.position.item(2) - player.car_data.position.item(2)) ** 2)

        return distance * float(100)


class ClosestDistToBallReward(RewardFunction):
    def __init__(self):
        self._inner_touch_reward = TouchHeightReward()

    # ENCOURAGE CONTROLLING THE BALL AND KEEPING THE BALL ON THE TOP OF THE CAR
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        touch_height = self._inner_touch_reward.get_reward(player, state, previous_action)

        agent = None
        for p in state.players:
            if agent is None:
                agent = p
            else:
                if np.linalg.norm(p.car_data.position - state.ball.position) < np.linalg.norm(
                        agent.car_data.position - state.ball.position):
                    agent = p

        if agent is player:
            distance = 1 / math.sqrt((state.ball.position.item(0) - player.car_data.position.item(0)) ** 2 +
                                     (state.ball.position.item(1) - player.car_data.position.item(1)) ** 2 +
                                     (state.ball.position.item(2) - player.car_data.position.item(2)) ** 2)
            reward = distance * float(100)

        reward += touch_height
        return reward


class DistancePlayerToBall(RewardFunction):
    def __init__(self):
        self.data = []

    def reset(self, initial_state: GameState):
        if len(self.data) != 0:
            self.data.clear()
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        self.data.append(dist)

        return -(dist ** 2) / 6_000_000 + 0.5


class GoalScoreSpeed(RewardFunction):
    def __init__(self):
        self.prev_position = 0
        self.prev_speed = 1

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.prev_position = state.ball.position.item(1)
        self.prev_speed = np.linalg.norm(state.ball.linear_velocity)
        if self.prev_speed < 1:
            self.prev_speed = 1
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # print("final reward")
        # print(np.linalg.norm(state.ball.linear_velocity))
        if player.team_num == BLUE_TEAM:
            if self.prev_position > 4000:
                uu_reward = self.prev_speed / BALL_MAX_SPEED
                kmh_reward = 215 * uu_reward
                reward = max(0, 0.5 * np.log2(kmh_reward / 10) - 0.6)
                # print(f"Blue reward: {reward}")
                return reward  # blue scored
        if player.team_num == ORANGE_TEAM:
            if self.prev_position < -4000:
                uu_reward = self.prev_speed / BALL_MAX_SPEED
                kmh_reward = 215 * uu_reward
                reward = max(0, 0.5 * np.log2(kmh_reward / 10) - 0.6)
                # print(f"Orange reward: {reward}")
                return reward  # orange scored
        return 0


class EpisodeLengthReward(RewardFunction):
    def __init__(self):
        self.nb_steps_since_reset = 0

    def reset(self, initial_state: GameState):
        self.nb_steps_since_reset = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.nb_steps_since_reset += 1

        return - self.nb_steps_since_reset ** 2 / 25_000 + 1

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)


# region KickOff

# class LeftKickoffReward(RewardFunction):
class KickoffReward(RewardFunction):

    def reset(self, initial_state: GameState):
        return 0

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: ndarray
    ) -> float:
        reward = 0
        vel_dir_reward = VelocityPlayerToBallReward()
        if state.ball.position.item(0) == 0:
            reward += vel_dir_reward.get_reward(player, state, previous_action) * 10
        if state.ball.position[1] > 500:
            # print("KICKOFF WON BY BLUE")
            if player.team_num == BLUE_TEAM:
                reward += 10
            else:
                reward -= 10

            return reward
        if state.ball.position[1] < -500:
            # print("KICKOFF WON BY ORANGE")
            if player.team_num == ORANGE_TEAM:
                reward += 10
            else:
                reward -= 10
            return reward
        return reward


class KickoffRewardMMR(RewardFunction):
    def __init__(self):
        self.kickoff_reward = KickoffReward()
        self.blue_team = []
        self.orange_team = []
        self.last_pos_saved = False
        self.is_kickoff = False
        self.is_kickoff_done = False
        self.closest_blue_agent = None
        self.closest_orange_agent = None

    def reset(self, initial_state: GameState):
        self.last_pos_saved = False
        self.is_kickoff = False
        self.is_kickoff_done = False
        self.closest_blue_agent = None
        self.closest_orange_agent = None
        return 0

    def get_players(self, players):
        for p in players:
            if p.team_num == BLUE_TEAM:
                self.blue_team.append(p)
            else:
                self.orange_team.append(p)

    def get_closest_blue(self, ball_pos):
        agent = None
        for p in self.blue_team:
            if agent is None:
                agent = p
            else:
                if np.linalg.norm(p.car_data.position - ball_pos) < np.linalg.norm(agent.car_data.position - ball_pos):
                    agent = p
        return agent

    def get_closest_orange(self, ball_pos):
        agent = None
        for p in self.orange_team:
            if agent is None:
                agent = p
            else:
                if np.linalg.norm(p.car_data.position - ball_pos) < np.linalg.norm(agent.car_data.position - ball_pos):
                    agent = p
        return agent

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:

        if len(self.blue_team) == 0:
            self.get_players(state.players)

        if self.closest_blue_agent is None:
            self.closest_blue_agent = self.get_closest_blue(state.ball.position)

        if self.closest_orange_agent is None:
            self.closest_orange_agent = self.get_closest_orange(state.ball.position)

        # Is KICKOFF
        if state.ball.position[0] == 0 and state.ball.position[1] == 0 and not self.last_pos_saved:
            self.last_pos_saved = True

        if self.last_pos_saved and not self.is_kickoff and state.ball.position[0] == 0 and state.ball.position[1] == 0:
            self.is_kickoff = True

        if self.is_kickoff and not self.is_kickoff_done:
            if state.ball.position[1] < -500 or state.ball.position[1] > 500:
                self.is_kickoff_done = True
            if player.car_id == self.closest_orange_agent.car_id or player.car_id == self.closest_blue_agent.car_id:
                return self.kickoff_reward.get_reward(player, state, previous_action)
        return 0


# endregion

# region Boost
class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[2] > 230 or (state.ball.position[0] == 0 and state.ball.position[1] == 0):
            return 0
        else:
            try:
                reward = (0.1 * math.log2(player.boost_amount - 0.06) + 0.4) / 10
            except ValueError:
                reward = -1
            return reward


class BoostPickupReward(RewardFunction):
    def __init__(self):
        self.prev_boost = 0

    def reset(self, initial_state: GameState):
        self.prev_boost = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        reward = 0
        if self.prev_boost < player.boost_amount:
            if player.boost_amount - self.prev_boost < 15:
                reward = 0.8
            else:
                reward = 1
            self.prev_boost = player.boost_amount

        return reward


class BoostReward(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            try:
                reward = (0.1 * math.log2(player.boost_amount - 0.06) + 0.4) / 10
            except ValueError:
                reward = 0
            return reward
        if player.team_num == ORANGE_TEAM:
            try:
                reward = (0.1 * math.log2(player.boost_amount - 0.06) + 0.4) / 10
            except ValueError:
                reward = -0.06
            return reward
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


# endregion


class AerialReward(RewardFunction):
    def __init__(self):
        self.in_air_reward = 0
        self.last_touch_pos = None  # we use ball cause yes
        self.lost_jump = False
        self.reward = 0
        self.dist_to_ball = DistToBallReward()
        self.on_wall = False

    def reset(self, initial_state: GameState):
        self.last_touch_pos = None
        self.lost_jump = False
        self.reward = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: ndarray) -> float:
        # starting dribble reward

        if 180 > state.ball.position[2] > 130 and player.ball_touched:
            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            if ball_speed < 2000:
                self.reward += 2

        if state.ball.position[2] > 230:  # 230
            if player.ball_touched:

                # aerial touch
                self.reward = + 1
                self.last_touch_pos = state.ball.position

                # for power flicks while dribbling
                if 400 > state.ball.position[2] < 220:
                    ball_speed = np.linalg.norm(state.ball.linear_velocity)
                    ball_speed_reward = (ball_speed / BALL_MAX_SPEED)
                    self.reward += ball_speed_reward
                    # print(f"height_reward: {height_reward}")

                # if no jump we reward ball power
                if not player.has_jump:
                    if not self.lost_jump:
                        self.lost_jump = True

                    ball_speed = np.linalg.norm(state.ball.linear_velocity)
                    ball_speed_reward = (ball_speed / BALL_MAX_SPEED) / 10
                    self.reward += ball_speed_reward
                    # print(f"BallSpeedReward: {ball_speed_reward}")

            elif player.car_data.position.item(2) > 300:
                dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
                dist = (dist ** 2) / 6_000_000 + 0.5
                if dist < 0.6:
                    height_reward = state.ball.position[2] / CEILING_Z
                    self.reward += height_reward
                    dist_to_ball_reward = self.dist_to_ball.get_reward(player, state, previous_action) * 10
                    self.reward += dist_to_ball_reward
            return self.reward

        else:
            self.reward = 0
            if self.lost_jump:
                self.lost_jump = False
            return self.reward


class WallReward(RewardFunction):

    def __init__(self) -> None:
        self.dist_to_ball = DistToBallReward()
        self.on_wall = False
        self.lost_jump = False
        self.reward = 0

    def reset(self, initial_state: GameState):
        self.on_wall = False
        self.lost_jump = False
        self.reward = 0
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # reward for distance to ball + wall encouragement + after wall near the ball reward

        if not player.has_jump:
            if not self.lost_jump:
                self.lost_jump = True

        dist_to_ball_reward = self.dist_to_ball.get_reward(player, state, previous_action) * 0.04

        # starting encouragement
        if state.ball.position.item(0) < -4000 or state.ball.position.item(0) > 4000 or (
                state.ball.position.item(1) > 5020 and 900 < state.ball.position.item(0) > -900) or (
                state.ball.position.item(1) < -5020 and 900 < state.ball.position.item(0) > -900):
            self.reward += dist_to_ball_reward * 2

        if player.car_data.position.item(0) < -4000 or player.car_data.position.item(0) > 4000 or (
                player.car_data.position.item(1) > 5020 and 900 < player.car_data.position.item(0) > -900) or (
                player.car_data.position.item(1) < -5020 and 900 < player.car_data.position.item(0) > -900):

            addup = (player.car_data.position.item(2) / CEILING_Z) * 2
            dist_to_ball_reward * (10 + addup)
            if not self.on_wall:
                self.on_wall = True
            else:
                dist_to_ball_reward *= 10

        if player.has_flip and self.lost_jump:
            if (player.car_data.position.item(0) < -4000 or player.car_data.position.item(
                    0) > 4000) and state.ball.position.item(2) > 500 and self.on_wall:
                dist_to_ball_reward *= 20

        self.reward += dist_to_ball_reward

        if not player.has_jump:
            if not self.lost_jump:
                self.lost_jump = True

            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            ball_speed_reward = ball_speed / BALL_MAX_SPEED
            self.reward += ball_speed_reward / 10

        if player.ball_touched:
            self.reward += 5

        if player.has_jump:
            self.lost_jump = False
        return self.reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class TouchHeightReward(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:

        if player.ball_touched and state.ball.position.item(2) > BALL_RADIUS:
            player_height = player.car_data.position
            calc = player_height.item(2) / CEILING_Z * 100
            if calc < 0:
                return 0
            else:
                reward = np.sqrt(calc)
                if np.isnan(reward) or reward < 0:
                    return 0
                else:
                    # print("Ball touched: " + str(reward))
                    return reward

        return 0

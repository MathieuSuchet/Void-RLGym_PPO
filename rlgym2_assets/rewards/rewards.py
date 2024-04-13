from typing import List, Dict, Any, Union, Tuple

import numpy as np
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS

import math_gym
from logger import Logger, _add


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

    def reset(self, initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        for reward_fn in self.reward_fns:
            reward_fn.reset(initial_state, shared_info)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        combined_rewards = {agent: 0. for agent in agents}
        self.reward_logs.append([])
        for reward_fn, weight in zip(self.reward_fns, self.weights):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            self.reward_logs[-1].append(np.mean(np.multiply(list(rewards.values()), weight)))

            for agent, reward in rewards.items():
                combined_rewards[agent] += reward * weight

        if any(is_truncated.values()) or any(is_terminated.values()):
            self.reward_logs = np.mean(self.reward_logs, axis=0)
            data = {"Rewards": {}}
            for i, reward_fn in enumerate(self.reward_fns):
                data["Rewards"].setdefault(reward_fn.__class__.__name__, {
                    "value": float(self.reward_logs[i]),
                    "nb_episodes": 1
                })
            self.logger.add_result(data, _add)
            self.reward_logs = []

        return combined_rewards


class VelocityReward(RewardFunction):
    def __init__(self, negative: bool = False):
        super().__init__()
        self.negative = negative

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {agent: 0. for agent in agents}
        for agent, car in state.cars.items():
            rewards[agent] = np.linalg.norm(car.physics.linear_velocity) / CAR_MAX_SPEED * (1 - 2 * self.negative)

        return rewards


class ConstantReward(RewardFunction):
    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        return {agent: 1 for agent in agents}


class VelBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {agent: 0. for agent in agents}
        for agent in agents:
            player = state.cars[agent]

            if player.team_num == BLUE_TEAM and not self.own_goal \
                    or player.team_num == ORANGE_TEAM and self.own_goal:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)

            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            if self.use_scalar_projection:
                # Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
                # Max value should be max_speed / ball_radius = 2300 / 94 = 24.5
                # Used to guide the agent towards the ball
                inv_t = math_gym.scalar_projection(vel, pos_diff)
                rewards[agent] = inv_t
            else:
                # Regular component velocity
                norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                norm_vel = vel / BALL_MAX_SPEED
                rewards[agent] = float(np.dot(norm_pos_diff, norm_vel))
        return rewards


class LiuDistancePlayerToBallReward(RewardFunction):

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {agent: 0 for agent in agents}
        for agent in agents:
            dist = np.linalg.norm(state.cars[agent].physics.position - state.ball.position) - BALL_RADIUS
            rewards[agent] = np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

        return rewards


class EventReward(RewardFunction):
    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def __init__(self, goal_w: float, concede_w: float, touch_w: float, shot_w: float, save_w: float):
        """
        Rewards when events occur
        :param goal_w: Weight of goal event
        :param concede_w: Weight of goal concede event
        :param touch_w: Weight of touch event
        """
        self.goal_w = goal_w
        self.concede_w = concede_w
        self.touch_w = touch_w
        self.shot_w = shot_w
        self.save_w = save_w
        self.weights = np.array([self.goal_w, self.concede_w, self.touch_w, self.shot_w, self.save_w])

    def _extract_values(self, agent: AgentID, state: GameState):
        goal = 1 if state.scoring_team == state.cars[agent].team_num else 0
        concede = -1 if state.scoring_team != state.cars[agent].team_num and state.scoring_team is not None else 0
        touch = state.cars[agent].ball_touches
        shot = state.cars[agent].shots
        save = state.cars[agent].saves

        return [goal, concede, touch, shot, save]

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        return {agent: np.dot(np.array(self._extract_values(agent, state)), self.weights) for agent in agents}


class FaceBallReward(RewardFunction):
    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {agent: 0. for agent in agents}

        for agent in agents:
            player = state.cars[agent]

            pos_diff = state.ball.position - player.physics.position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            rewards[agent] = float(np.dot(player.physics.forward, norm_pos_diff))

        return rewards

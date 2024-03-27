from typing import List, Dict, Any, Union, Tuple

import numpy as np
from rlgym.api import RewardFunction, AgentID, StateType, RewardType
from rlgym.rocket_league.api import GameState

from logger import Logger, _replace


class LoggerCombinedReward(RewardFunction):
    def __init__(self, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]], logger: Logger):
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
        self.logger = logger
        self.n_steps = 0

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

        self.n_steps += len(agents)

        if any(is_truncated.values()) or any(is_terminated.values()):
            self.reward_logs = np.mean(self.reward_logs, axis=0)
            data = {"Rewards": {}}
            for i, reward_fn in enumerate(self.reward_fns):
                data["Rewards"].setdefault(reward_fn.__class__.__name__, float(self.reward_logs[i]))
            self.logger.add_result(data, _replace)
            self.reward_logs = []
            self.n_steps = 0

        return combined_rewards


class VelocityReward(RewardFunction):
    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        rewards = {agent: 0. for agent in agents}
        for agent, car in state.cars.items():
            rewards[agent] = np.sum(car.physics.linear_velocity)

        return rewards


class ConstantReward(RewardFunction):
    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: StateType, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, RewardType]:
        return {agent: 1 for agent in agents}
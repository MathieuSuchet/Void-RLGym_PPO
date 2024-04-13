"""
Contains all loggers related to global stats
"""
from typing import List

import numpy as np
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState

from rlgym1_assets.wandb_loggers.logger_utils import _is_goal_scored


def get_all_global_loggers():
    return [
        GoalLogger(),
        TouchLogger(),
        GoalVelocityLogger(),
        TouchHeightLogger(),
        ShotLogger(),
        SaveLogger()
    ]


class WandbMetricsLogger(MetricsLogger):
    """
    A logger that contains metrics and which logs to wandb
    """

    @property
    def metrics(self) -> List[str]:
        """
        All the metrics names that will be uploaded to wandb
        :return: The metrics names
        """
        return []

    def compute_data(self, metrics):
        return np.mean(metrics)


class GoalLogger(WandbMetricsLogger):
    """
    Logs :\n
    The goal rate
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/goal_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        goal_rate = _is_goal_scored(game_state.ball)
        return np.array([goal_rate])


class TouchLogger(WandbMetricsLogger):
    """
    Logs :\n
    The mean touch of all the agents
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/touch_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        touch_rate = np.mean([int(car.ball_touched) for car in game_state.players])
        return np.array([touch_rate])


class GoalVelocityLogger(WandbMetricsLogger):
    """
    Logs :\n
    The velocity if scored else 0
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/avg_goal_vel"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([np.linalg.norm(game_state.ball.linear_velocity) if _is_goal_scored(game_state.ball) else 0])

    def compute_data(self, metrics: np.array):
        metrics = metrics[np.nonzero(metrics)]
        return np.mean(metrics) if metrics.size != 0 else 0


class TouchHeightLogger(WandbMetricsLogger):
    """
    Logs :\n
    The height of touches if touched else 0
    """

    def __init__(self):
        self.last_touched = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/avg_touch_height"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        touch_heights = np.zeros((len(game_state.players)))
        for i, agent in enumerate(game_state.players):
            if agent in self.last_touched.keys():
                self.last_touched[agent] = 0
            else:
                self.last_touched.setdefault(agent, 0)

            agent_touches = int(game_state.players[i].ball_touched)

            if agent_touches > self.last_touched[agent]:
                self.last_touched[agent] = agent_touches
                touch_heights[i] = game_state.players[i].car_data.position[2]

        touch_heights = touch_heights[touch_heights.nonzero()]
        if touch_heights.size == 0:
            return np.array([0])
        return np.array([np.mean(touch_heights)])

    def compute_data(self, metrics: np.array):
        self.last_touched.clear()
        metrics = metrics[np.nonzero(metrics)]
        return np.mean(metrics) if metrics.size != 0 else 0


class ShotLogger(WandbMetricsLogger):
    def __init__(self):
        self.shots = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/shot_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        for agent in game_state.players:
            if agent.car_id not in self.shots.keys():
                self.shots.setdefault(agent.car_id, 0)

        result = np.array(
            [np.sum([1 if car.match_shots > self.shots[car.car_id] else 0 for car in game_state.players])])
        for agent in game_state.players:
            self.shots[agent.car_id] = agent.match_shots

        return result


class SaveLogger(WandbMetricsLogger):
    def __init__(self):
        self.saves = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/save_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        for agent in game_state.players:
            if agent.car_id not in self.saves.keys():
                self.saves.setdefault(agent.car_id, 0)

        result = np.array(
            [np.sum([1 if car.match_saves > self.saves[car.car_id] else 0 for car in game_state.players])])
        for agent in game_state.players:
            self.saves[agent.car_id] = agent.match_shots

        return result

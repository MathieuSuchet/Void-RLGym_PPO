"""
Contains all loggers related to global stats
"""
from typing import List

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger


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
        goal_rate = game_state.goal_scored
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
        touch_rate = np.mean([car.ball_touches for car in game_state.cars.values()])
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
        return np.array([np.linalg.norm(game_state.ball.linear_velocity) if game_state.goal_scored else 0])

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
        touch_heights = np.zeros((len(game_state.cars.keys())))
        for i, agent in enumerate(game_state.cars.keys()):
            if agent in self.last_touched.keys():
                self.last_touched[agent] = 0
            else:
                self.last_touched.setdefault(agent, 0)

            agent_touches = game_state.cars[agent].ball_touches

            if agent_touches > self.last_touched[agent]:
                self.last_touched[agent] = agent_touches
                touch_heights[i] = game_state.cars[agent].physics.position[2]

        touch_heights = touch_heights[touch_heights.nonzero()]
        if touch_heights.size == 0:
            return np.array([0])
        return np.array([np.mean(touch_heights)])

    def compute_data(self, metrics: np.array):
        self.last_touched.clear()
        metrics = metrics[np.nonzero(metrics)]
        return np.mean(metrics) if metrics.size != 0 else 0


class ShotLogger(WandbMetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return ["stats/shot_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([np.sum([car.shots for car in game_state.cars.values()])])


class SaveLogger(WandbMetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return ["stats/save_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([np.sum([car.saves for car in game_state.cars.values()])])

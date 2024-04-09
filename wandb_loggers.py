from typing import List

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger


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


class BallHeightLogger(WandbMetricsLogger):
    """
    Logs :\n
    The ball's height
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/ball/ball_height"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([game_state.ball.position[2]])


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


class PlayerVelocityLogger(WandbMetricsLogger):
    """
    Logs :\n
    The mean of all player's linear velocity's magnitude\n
    The mean of all player's angular velocity's magnitude
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_lin_vel", "stats/player/avg_ang_vel"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_cars = len(game_state.cars.values())
        lin_vel = np.zeros((n_cars, 3))
        ang_vel = np.zeros((n_cars, 3))

        for i in range(n_cars):
            car = list(game_state.cars.values())[i]
            lin_vel[i] = car.physics.linear_velocity
            ang_vel[i] = car.physics.angular_velocity

        lin_vel, ang_vel = np.mean(lin_vel, axis=1), np.mean(ang_vel, axis=1)
        lin_vel, ang_vel = np.linalg.norm(lin_vel), np.linalg.norm(ang_vel)

        return np.array([lin_vel, ang_vel])


class BallVelocityLogger(WandbMetricsLogger):
    """
    Logs :\n
    The ball's linear velocity's magnitude\n
    The ball's angular velocity's magnitude
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/ball/avg_lin_vel", "stats/ball/avg_ang_vel"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([
            np.linalg.norm(game_state.ball.linear_velocity),
            np.linalg.norm(game_state.ball.angular_velocity)
        ])


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

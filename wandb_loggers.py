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

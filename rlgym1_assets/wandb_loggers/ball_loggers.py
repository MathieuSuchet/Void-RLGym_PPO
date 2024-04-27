"""
Contains all loggers related to the ball
"""
from typing import List

import numpy as np
from rlgym_sim.utils.gamestates import GameState

from rlgym1_assets.wandb_loggers.global_loggers import WandbMetricsLogger


def get_all_ball_loggers():
    """
    Get all the loggers related to the ball
    :return: All the loggers related to the ball
    """
    return [
        BallHeightLogger(),
        BallVelocityLogger(),
        BallAccelerationLogger()
    ]


class BallHeightLogger(WandbMetricsLogger):
    """
    Logs :\n
    The ball's height
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/ball/ball_height"]

    def get_data_for(self, car_id: int, game_state: GameState):
        return game_state.ball.position[2]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([self.get_data_for(-1, game_state)])


class BallVelocityLogger(WandbMetricsLogger):
    """
    Logs :\n
    The ball's linear velocity's magnitude\n
    The ball's angular velocity's magnitude
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/ball/avg_lin_vel", "stats/ball/avg_ang_vel"]

    def get_data_for(self, car_id: int, game_state: GameState):
        return [
            np.linalg.norm(game_state.ball.linear_velocity),
            np.linalg.norm(game_state.ball.angular_velocity)
        ]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array(self.get_data_for(-1, game_state))


class BallAccelerationLogger(WandbMetricsLogger):
    """
    Logs :\n
    The ball's linear acceleration (on touch)\n
    The ball's angular acceleration (on touch)
    """
    def __init__(self, standalone_mode: bool = False):
        super().__init__(standalone_mode)
        self.last_lin_vel = np.array([0, 0, 0])
        self.last_ang_vel = np.array([0, 0, 0])

    @property
    def metrics(self) -> List[str]:
        return ["stats/ball/avg_lin_accel", "stats/ball/avg_ang_accel"]

    def get_data_for(self, car_id: int, game_state: GameState):
        return (
            np.linalg.norm(game_state.ball.linear_velocity - self.last_lin_vel),
            np.linalg.norm(game_state.ball.angular_velocity - self.last_ang_vel)
        ) \
            if any([p.ball_touched for p in game_state.players]) \
            else (0, 0)

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        lin_accel, ang_accel = self.get_data_for(-1, game_state)

        self.last_lin_vel = game_state.ball.linear_velocity
        self.last_ang_vel = game_state.ball.angular_velocity

        return np.array([lin_accel, ang_accel])

from typing import List

import numpy as np
from rlgym_sim.utils.gamestates import GameState

from rlgym1_assets.wandb_loggers.global_loggers import WandbMetricsLogger


def get_all_ball_loggers():
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

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([game_state.ball.position[2]])


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


class BallAccelerationLogger(WandbMetricsLogger):
    def __init__(self):
        self.last_lin_vel = np.array([0, 0, 0])
        self.last_ang_vel = np.array([0, 0, 0])

    @property
    def metrics(self) -> List[str]:
        return ["stats/ball/avg_lin_accel", "stats/ball/avg_ang_accel"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        lin_accel = ang_accel = 0

        for i in range(len(game_state.players)):
            if game_state.players[i].ball_touched:
                lin_accel = np.linalg.norm(game_state.ball.linear_velocity - self.last_lin_vel)
                ang_accel = np.linalg.norm(game_state.ball.angular_velocity - self.last_ang_vel)

        self.last_lin_vel = game_state.ball.linear_velocity
        self.last_ang_vel = game_state.ball.angular_velocity

        return np.array([lin_accel, ang_accel])

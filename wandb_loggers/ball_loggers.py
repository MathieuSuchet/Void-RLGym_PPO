from typing import List

import numpy as np
from rlgym.rocket_league.api import GameState
from wandb_loggers.global_loggers import WandbMetricsLogger


def get_all_ball_loggers():
    return [BallHeightLogger(), BallVelocityLogger()]


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

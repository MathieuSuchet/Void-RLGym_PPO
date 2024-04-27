"""
Contains all loggers related to global stats
"""
from typing import List, Any

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
        SaveLogger(),
        FlipResetLogger()
    ]


class WandbMetricsLogger(MetricsLogger):
    """
    A logger that contains metrics and which logs to wandb
    """
    def __init__(self, standalone_mode: bool = False):
        self.standalone = standalone_mode

    @property
    def metrics(self) -> List[str]:
        """
        All the metrics names that will be uploaded to wandb
        :return: The metrics names
        """
        return []

    @staticmethod
    def _get_player_data(car_id: int, game_state: GameState):
        for player in game_state.players:
            if player.car_id == car_id:
                return player
        return None

    def get_data_for(self, car_id: int, game_state: GameState) -> Any:
        pass

    def collect_metrics(self, game_state: GameState, car_id: int = -1) -> np.ndarray:
        if self.standalone:
            return self._standalone_metrics(game_state, car_id)
        return super().collect_metrics(game_state)

    def compute_data(self, metrics):
        return np.mean(metrics)

    def _standalone_metrics(self, game_state, car_id):
        metric = self.get_data_for(car_id, game_state)
        self._update_self(car_id, game_state, metric)
        return metric

    def _update_self(self, car_id: int, game_state: GameState, metric: Any):
        pass


class GoalLogger(WandbMetricsLogger):
    """
    Logs :\n
    The goal rate
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/goal_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        goal_rate = _is_goal_scored(game_state)
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

    def get_data_for(self, car_id: int, game_state: GameState):
        return np.array([np.linalg.norm(game_state.ball.linear_velocity) if _is_goal_scored(game_state) else 0])

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return self.get_data_for(-1, game_state)

    def compute_data(self, metrics: np.array):
        metrics = metrics[np.nonzero(metrics)]
        return np.mean(metrics) if metrics.size != 0 else 0


class TouchHeightLogger(WandbMetricsLogger):
    """
    Logs :\n
    The height of touches if touched else 0
    """

    @property
    def metrics(self) -> List[str]:
        return ["stats/avg_touch_height"]

    def get_data_for(self, car_id: int, game_state: GameState):
        player = WandbMetricsLogger._get_player_data(car_id, game_state)
        return player.car_data.position[2] if player.ball_touched else 0

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        touch_heights = np.zeros((len(game_state.players)))
        for i, player in enumerate(game_state.players):
            touch_heights[i] = self.get_data_for(player.car_id, game_state)

        touch_heights = touch_heights[touch_heights.nonzero()]
        if touch_heights.size == 0:
            return np.array([0])
        return np.array([np.mean(touch_heights)])

    def compute_data(self, metrics: np.array):
        metrics = metrics[np.nonzero(metrics)]
        return np.mean(metrics) if metrics.size != 0 else 0


class ShotLogger(WandbMetricsLogger):
    def __init__(self, standalone_mode: bool = False):
        super().__init__(standalone_mode)
        self.shots = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/shot_rate"]

    def get_data_for(self, car_id: int, game_state: GameState):
        car = WandbMetricsLogger._get_player_data(car_id, game_state)
        return 1 if car.match_shots > self.shots[car.car_id] else 0

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        for agent in game_state.players:
            if agent.car_id not in self.shots.keys():
                self.shots.setdefault(agent.car_id, 0)

        result = np.array(
            [np.sum([self.get_data_for(car.car_id, game_state) for car in game_state.players])])
        for agent in game_state.players:
            self.shots[agent.car_id] = agent.match_shots

        return result


class SaveLogger(WandbMetricsLogger):
    def __init__(self, standalone_mode: bool = False):
        super().__init__(standalone_mode)
        self.saves = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/save_rate"]

    def get_data_for(self, car_id: int, game_state: GameState):
        car = WandbMetricsLogger._get_player_data(car_id, game_state)
        return 1 if car.match_saves > self.saves[car.car_id] else 0

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        for agent in game_state.players:
            if agent.car_id not in self.saves.keys():
                self.saves.setdefault(agent.car_id, 0)

        result = np.array(
            [np.sum([self.get_data_for(car.car_id, game_state) for car in game_state.players])])
        for agent in game_state.players:
            self.saves[agent.car_id] = agent.match_shots

        return result


class FlipResetLogger(WandbMetricsLogger):
    BALL_DIST_THRESHOLD: int = 170
    DIRECTION_SIMILARITY_THRESHOLD: float = 0.7

    def __init__(self, standalone_mode: bool = False):
        super().__init__(standalone_mode)
        self.has_resets = {}

    def _update_self(self, car_id: int, game_state: GameState, metric):
        if car_id not in self.has_resets.keys():
            self.has_resets.setdefault(car_id, metric)
        else:
            self.has_resets[car_id] = metric

    def get_data_for(self, car_id: int, game_state: GameState):
        if car_id not in self.has_resets.keys():
            self.has_resets.setdefault(car_id, False)

        car = WandbMetricsLogger._get_player_data(car_id, game_state)
        # Up direction comparison
        ball_distance = car.car_data.position - game_state.ball.position
        ball_dir = ball_distance / np.linalg.norm(ball_distance)

        # Similarity check
        similarity = ball_dir.dot(-car.car_data.up())
        if similarity < FlipResetLogger.DIRECTION_SIMILARITY_THRESHOLD:
            return False

        # Distance to ball
        if np.linalg.norm(ball_distance) > FlipResetLogger.BALL_DIST_THRESHOLD:
            return False

        has_reset = (
                car.on_ground
                and not car.has_jump
                and car.has_flip)

        return not self.has_resets[car_id] and has_reset

    def _update_self(self, car_id: int, game_state: GameState, metric: Any):
        self.has_resets[car_id] = metric

    @property
    def metrics(self) -> List[str]:
        return ["stats/flip_resets"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_resets = 0

        for car in game_state.players:
            if car.car_id not in self.has_resets:
                self.has_resets.setdefault(car.car_id, False)

            has_reset = self.get_data_for(car.car_id, game_state)

            n_resets += int(has_reset)
            self.has_resets[car.car_id] = has_reset

        return np.array([n_resets])

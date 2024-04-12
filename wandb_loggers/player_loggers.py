from typing import List

import numpy as np
from rlgym.rocket_league.api import GameState

from wandb_loggers.global_loggers import WandbMetricsLogger
from wandb_loggers.logger_utils import _is_on_wall, _is_on_ceiling


def get_all_player_loggers(wall_width_tolerance: float = 100., wall_height_tolerance: float = 100.) -> List[WandbMetricsLogger]:
    return [
        PlayerVelocityLogger(),
        PlayerHeightLogger(),
        PlayerBoostLogger(),
        PlayerFlipTimeLogger(),
        PlayerWallTimeLogger(wall_width_tolerance, wall_height_tolerance),
        PlayerCeilingTimeLogger(wall_width_tolerance, wall_height_tolerance),
        PlayerWallHeightLogger(wall_width_tolerance, wall_height_tolerance),
        PlayerRelDistToBallLogger(),
        PlayerRelVelToBallLogger()
    ]


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


class PlayerHeightLogger(WandbMetricsLogger):
    """
    Logs :\n
    Player's average height
    """
    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_height"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_cars = len(game_state.cars.keys())
        heights = np.zeros((n_cars,))

        for i, car in enumerate(game_state.cars.values()):
            heights[i] = car.physics.position[2]

        return np.array([np.mean(heights)])


class PlayerBoostLogger(WandbMetricsLogger):
    """
    Logs :\n
    Player's average boost
    """
    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_boost_amount"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_cars = len(game_state.cars.keys())
        boosts = np.zeros((n_cars,))

        for i, car in enumerate(game_state.cars.values()):
            boosts[i] = car.boost_amount

        return np.array([np.mean(boosts)])

    def compute_data(self, metrics):
        metrics *= 100
        return np.mean(metrics)


class PlayerFlipTimeLogger(WandbMetricsLogger):
    """
    Logs :\n
    Average time before flipping/double jumping
    """
    def __init__(self):
        self.time_between_jump_and_flip = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_flip_time"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        times = np.zeros((len(game_state.cars.keys()),))
        for agent in game_state.cars.keys():
            if agent not in self.time_between_jump_and_flip.keys():
                self.time_between_jump_and_flip.setdefault(agent, 0)

        for i, (agent, car) in enumerate(game_state.cars.items()):
            if car.has_jumped and not (car.has_flipped or car.has_double_jumped):
                self.time_between_jump_and_flip[agent] += 1
            else:
                times[i] = self.time_between_jump_and_flip[agent]
                self.time_between_jump_and_flip[agent] = 0

        times = times[np.nonzero(times)]
        if times.size == 0:
            return np.array([0])
        return np.array([np.mean(times)])

    def compute_data(self, metrics):
        self.time_between_jump_and_flip.clear()

        metrics = metrics[np.nonzero(metrics)]
        return np.mean(metrics)


class PlayerWallTimeLogger(WandbMetricsLogger):
    """
    Logs :\n
    Average time on wall
    """

    def __init__(self, wall_width_tolerance: float = 100, wall_height_tolerance: float = 100):
        self.wall_width_tolerance = wall_width_tolerance
        self.wall_height_tolerance = wall_height_tolerance
        self.time_between_wall_and_other = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_wall_time"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        times = np.zeros((len(game_state.cars.keys()),))
        for agent in game_state.cars.keys():
            if agent not in self.time_between_wall_and_other.keys():
                self.time_between_wall_and_other.setdefault(agent, 0)

        for i, (agent, car) in enumerate(game_state.cars.items()):
            if _is_on_wall(car, wall_width_tolerance=self.wall_width_tolerance,
                           wall_height_tolerance=self.wall_height_tolerance):
                self.time_between_wall_and_other[agent] += 1
            else:
                times[i] = self.time_between_wall_and_other[agent]
                self.time_between_wall_and_other[agent] = 0

        times = times[np.nonzero(times)]
        if times.size == 0:
            return np.array([0])
        return np.array([np.mean(times)])

    def compute_data(self, metrics):
        self.time_between_wall_and_other.clear()
        metrics = metrics[np.nonzero(metrics)]
        if metrics.size == 0:
            return 0
        return np.mean(metrics)


class PlayerWallHeightLogger(WandbMetricsLogger):
    """
    Logs :\n
    Average player height (when on wall)
    """
    def __init__(self, wall_width_tolerance: float = 100, wall_height_tolerance: float = 100):
        self.wall_width_tolerance = wall_width_tolerance
        self.wall_height_tolerance = wall_height_tolerance

    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_wall_height"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_cars = len(game_state.cars.keys())
        heights = np.zeros((n_cars,))

        for i, car in enumerate(game_state.cars.values()):
            if _is_on_wall(car, wall_width_tolerance=self.wall_width_tolerance,
                           wall_height_tolerance=self.wall_height_tolerance):
                heights[i] = car.physics.position[2]

        heights = heights[np.nonzero(heights)]
        if heights.size == 0:
            return np.array([0])
        return np.array([np.mean(heights)])


class PlayerCeilingTimeLogger(WandbMetricsLogger):
    """
    Logs :\n
    Average ceiling time
    """
    def __init__(self, wall_width_tolerance: float = 100, wall_height_tolerance: float = 100):
        self.wall_width_tolerance = wall_width_tolerance
        self.wall_height_tolerance = wall_height_tolerance
        self.time_between_ceil_and_other = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_ceil_time"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        times = np.zeros((len(game_state.cars.keys()),))
        for agent in game_state.cars.keys():
            if agent not in self.time_between_ceil_and_other.keys():
                self.time_between_ceil_and_other.setdefault(agent, 0)

        for i, (agent, car) in enumerate(game_state.cars.items()):
            if _is_on_ceiling(car, wall_width_tolerance=self.wall_width_tolerance,
                              wall_height_tolerance=self.wall_height_tolerance):
                self.time_between_ceil_and_other[agent] += 1
            else:
                times[i] = self.time_between_ceil_and_other[agent]
                self.time_between_ceil_and_other[agent] = 0

        times = times[np.nonzero(times)]
        if times.size == 0:
            return np.array([0])
        return np.array([np.mean(times)])

    def compute_data(self, metrics):
        self.time_between_ceil_and_other.clear()
        metrics = metrics[np.nonzero(metrics)]
        if metrics.size == 0:
            return 0
        return np.mean(metrics)


class PlayerRelDistToBallLogger(WandbMetricsLogger):
    """
    Logs :\n
    Average relative distance to ball
    """
    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_rel_dist_to_ball"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_cars = len(game_state.cars.keys())
        rel_dists = np.zeros((n_cars, ))
        ball = game_state.ball

        for i, agent in enumerate(game_state.cars.values()):
            rel_dists[i] = np.linalg.norm(ball.position - agent.physics.position)

        return np.array([np.mean(rel_dists)])


class PlayerRelVelToBallLogger(WandbMetricsLogger):
    """
    Logs :\n
    Average relative velocity to ball
    """
    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_rel_vel_to_ball"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        n_cars = len(game_state.cars.keys())
        rel_vel = np.zeros((n_cars,))
        ball = game_state.ball

        for i, agent in enumerate(game_state.cars.values()):
            rel_dist = ball.position - agent.physics.position
            player_vel = agent.physics.linear_velocity
            ball_vel = ball.linear_velocity
            rel_vel[i] = np.dot(player_vel - ball_vel, rel_dist / np.linalg.norm(rel_dist))

        return np.array([np.mean(rel_vel)])

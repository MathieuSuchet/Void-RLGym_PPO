from math import sqrt
from typing import List

import numpy as np
from rlgym.rocket_league.api import GameState, Car
from rlgym.rocket_league.common_values import SIDE_WALL_X, CEILING_Z, BACK_WALL_Y

from wandb_loggers.global_loggers import WandbMetricsLogger


def get_all_player_loggers() -> List[WandbMetricsLogger]:
    return [PlayerVelocityLogger(), PlayerHeightLogger(), PlayerBoostLogger(), PlayerFlipTimeLogger()]


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


X_AT_ZERO = 8064


class PlayerWallTimeLogger(WandbMetricsLogger):

    def __init__(self, wall_width_tolerance: float = 100, wall_height_tolerance: float = 100):
        self.wall_width_tolerance = wall_width_tolerance
        self.wall_height_tolerance = wall_height_tolerance
        self.time_between_wall_and_other = {}

    @property
    def metrics(self) -> List[str]:
        return ["stats/player/avg_wall_time"]

    def _is_on_wall(self, car: Car) -> bool:
        on_flat_wall = (
                car.on_ground
                # Side wall comparison
                and SIDE_WALL_X - self.wall_width_tolerance
                < abs(car.physics.position[0])
                < SIDE_WALL_X + self.wall_width_tolerance
                # Back wall comparison
                and BACK_WALL_Y - self.wall_width_tolerance
                < abs(car.physics.position[1])
                < BACK_WALL_Y + self.wall_width_tolerance
                # Ceiling/Ground comparison
                and self.wall_height_tolerance
                < car.physics.position[2]
                < CEILING_Z - self.wall_height_tolerance
        )

        if on_flat_wall:
            return True

        is_on_corner = False

        for a in (-1, 1):
            if is_on_corner:
                break

            for b in (-1, 1):
                if (car.physics.position[1] - self.wall_width_tolerance
                        < a * car.physics.position[0] + (X_AT_ZERO * b)
                        < car.physics.position[1] + self.wall_width_tolerance):
                    # On wall
                    is_on_corner = True
                    break

        return is_on_corner

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        times = np.zeros((len(game_state.cars.keys()),))
        for agent in game_state.cars.keys():
            if agent not in self.time_between_wall_and_other.keys():
                self.time_between_wall_and_other.setdefault(agent, 0)

        for i, (agent, car) in enumerate(game_state.cars.items()):
            if self._is_on_wall(car):
                self.time_between_wall_and_other[agent] += 1
            else:
                times[i] = self.time_between_wall_and_other[agent]
                self.time_between_wall_and_other[agent] = 0

        times = times[np.nonzero(times)]
        if times.size == 0:
            return np.array([0])
        return np.array([np.mean(times)])

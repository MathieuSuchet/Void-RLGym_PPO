import random
from typing import Dict, Any, Union, Tuple

import numpy as np
from rlgym.api import StateMutator, StateType
from rlgym.rocket_league.common_values import BALL_MAX_SPEED, BALL_RADIUS, CAR_MAX_ANG_VEL, CAR_MAX_SPEED, SIDE_WALL_X, \
    BACK_WALL_Y, CEILING_Z, BLUE_TEAM, ORANGE_TEAM

from math_gym import rand_vec3

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi


class RandomStateMutator(StateMutator):
    def apply(self, state: StateType, shared_info: Dict[str, Any]) -> None:
        state.ball.position = np.array(
            [np.random.uniform(-LIM_X, LIM_X),
             np.random.uniform(-LIM_Y, LIM_Y),
             np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z)]
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state.ball.linear_velocity = vel

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state.ball.angular_velocity = ang_vel

        for car in state.cars.values():
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.physics.position = car_pos
            else:  # Fallback on fully random
                car.physics.position = np.array([
                    np.random.uniform(-LIM_X, LIM_X),
                    np.random.uniform(-LIM_Y, LIM_Y),
                    np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z)],
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.physics.linear_velocity = vel

            car.physics.euler_angles = np.array([
                np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                np.random.uniform(-YAW_LIM, YAW_LIM),
                np.random.triangular(-ROLL_LIM, 0, ROLL_LIM)]
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.physics.angular_velocity = ang_vel
            car.boost_amount = np.random.uniform(0, 1)


class ShotMutator(StateMutator):
    def apply(self, state: StateType, shared_info: Dict[str, Any]) -> None:
        for car in state.cars.values():
            if car.team_num == BLUE_TEAM:
                car.physics.position = np.array([
                    random.uniform(-4096, 4096),
                    random.uniform(0, 3000),
                    17]
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.physics.linear_velocity = vel

                car.physics.euler_angles = np.array([
                    np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    np.random.uniform(-YAW_LIM, YAW_LIM),
                    np.random.triangular(-ROLL_LIM, 0, ROLL_LIM)]
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.physics.angular_velocity = ang_vel
                car.boost_amount = np.random.uniform(0, 1)

                state.ball.position = np.array([
                    np.random.uniform(max(car.physics.position.item(0) - 1000, -LIM_X),
                                      min(car.physics.position.item(0) + 1000, LIM_X)),
                    np.random.uniform(car.physics.position.item(1) + 1000, car.physics.position.item(1) + 100),
                    np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2)]
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state.ball.linear_velocity = vel

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state.ball.angular_velocity = ang_vel

            if car.team_num == ORANGE_TEAM:
                car.physics.position = np.array([
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17]
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.physics.linear_velocity = vel

                car.physics.euler_angles = np.array([
                    np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    np.random.uniform(-YAW_LIM, YAW_LIM),
                    np.random.triangular(-ROLL_LIM, 0, ROLL_LIM)],
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.physics.angular_velocity = ang_vel
                car.boost_amount = np.random.uniform(0, 1)


class WeightedStateMutator(StateMutator):
    def __init__(self, *mutators: Union[StateMutator, Tuple[StateMutator, float]]):
        mutators_l = []
        weights = []

        for v in mutators:
            if isinstance(v, tuple):
                mut, w = v
            else:
                mut, w = v, 1

            mutators_l.append(mut)
            weights.append(w)

        self.mutators = tuple(mutators_l)
        self.weights = tuple(weights)

    def apply(self, state: StateType, shared_info: Dict[str, Any]) -> None:
        selected_mutator = random.choices(self.mutators, self.weights)[0]
        selected_mutator.apply(state, shared_info)

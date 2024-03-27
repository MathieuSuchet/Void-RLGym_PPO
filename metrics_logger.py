import numpy as np
from rlgym.api import DoneCondition
from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger
from wandb.sdk.wandb_run import Run


class TouchLogger(MetricsLogger):
    def __init__(self, terminal_condition: DoneCondition):
        self.terminal_cond = terminal_condition
        self.n_steps = 0

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        if any(self.terminal_cond.is_done(list(game_state.cars.keys()), game_state, shared_info={}).values()):
            touch_rate = np.mean([car.ball_touches for car in game_state.cars.values()])

            self.n_steps = 0
            print("Episode ended")
            return np.array([touch_rate])

        self.n_steps += 1
        return np.array([None])

    def _report_metrics(self, collected_metrics, wandb_run: Run, cumulative_timesteps):
        print(np.sum(np.nan_to_num(collected_metrics)))
        wandb_run.log(
            data={"stats/touches": np.sum(
                np.nan_to_num(collected_metrics)
            )},
            step=cumulative_timesteps)


class BallHeightLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([game_state.ball.position[2]])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        wandb_run.log(
            data={"stats/ball_height": np.mean(
                np.nan_to_num(collected_metrics)
            )},
            step=cumulative_timesteps)


class MultiLogger(MetricsLogger):
    def __init__(self, *loggers: MetricsLogger):
        self.loggers = loggers

    def collect_metrics(self, game_state: GameState) -> np.ndarray:
        for logger in self.loggers:
            logger.collect_metrics(game_state)

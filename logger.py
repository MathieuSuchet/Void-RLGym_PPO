import json
import os
from typing import Dict, Any, Callable, List

import filelock
import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger
from wandb.sdk.wandb_run import Run


def _add(x: Any, y: Any):
    return x + y


def _minus(x: Any, y: Any):
    return x - y


def _times(x: Any, y: Any):
    return x * y


def _pow(x: Any, y: Any):
    return pow(x, y)


def _replace(x: Any, y: Any):
    return x


def _init_empty_file(filepath):
    with open(filepath, "w") as f:
        json.dump({}, f)


class WandbMetricsLogger(MetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return []


class BallHeightLogger(WandbMetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return ["stats/ball_height"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([game_state.ball.position[2]])


class TouchLogger(WandbMetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return ["stats/touch_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        touch_rate = np.mean([car.ball_touches for car in game_state.cars.values()])
        return np.array([touch_rate])


class GoalLogger(WandbMetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return ["stats/goal_rate"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        goal_rate = game_state.goal_scored
        return np.array([goal_rate])


class Logger(MetricsLogger):
    def __init__(self, *loggers: WandbMetricsLogger):
        self.file_path = "./logging/logging.json"
        if "logging" not in os.listdir():
            os.makedirs("logging")
        _init_empty_file(self.file_path)
        self.loggers: Dict[WandbMetricsLogger, Any] = {logger: [] for logger in loggers}

    def add_logger(self, logger: WandbMetricsLogger):
        if logger in self.loggers.keys():
            print("Logger", logger.__class__.__name__, "already exists in the logger list")
        else:
            self.loggers.setdefault(logger, [])

    def log(self, metrics_data, wandb_run: Run, step: int):
        data = self.get_results().copy()
        self._compute_data(data)
        self._print_data(data)
        new_data = {}

        self._format_to_wandb(data, new_data)
        self._handle_metrics(metrics_data, new_data)

        wandb_run.log(new_data, step)

        # Clear loggers to only register current rollout's metrics
        for logger in self.loggers.keys():
            self.loggers[logger].clear()

    def _format_to_wandb(self, data: Dict[str, Any], new_data: Dict, trace: str = ""):
        for k, v in data.items():
            if isinstance(v, Dict):
                self._format_to_wandb(v, new_data, trace + k + "/")
            else:
                new_data.setdefault(trace + k, v)

    def save_results(self, data: Dict[str, Any]):
        with filelock.FileLock("logging/f_lock.lock"):
            with open(self.file_path, "w") as f:
                json.dump(data, f)

    def get_results(self) -> Dict[str, Any]:
        with filelock.FileLock("logging/f_lock.lock"):
            with open(self.file_path, "r") as f:
                data = json.load(f)
                return data

    def _map_result(self, data: Dict[str, Any], in_file: Dict[str, Any], func: Callable[[Any, Any], Any]):
        for k, v in in_file.items():
            if isinstance(v, Dict):
                if k not in data.keys():
                    data.setdefault(k, {})
                self._map_result(data[k], v, func)
            else:
                if k in in_file.keys():
                    if k in data.keys():
                        data[k] = func(data[k], in_file[k])
                    else:
                        data.setdefault(k, in_file[k])
                else:
                    data.setdefault(k, in_file[k])

    def add_result(self, data: Dict[str, Any], func_merge: Callable[[Any, Any], Any] = _add):
        data_cp = data.copy()

        if os.path.exists(self.file_path):
            in_file = self.get_results()
            self._map_result(data_cp, in_file, func_merge)

        self.save_results(data_cp)

    def _report_metrics(self, collected_metrics, wandb_run: Run, cumulative_timesteps):
        self.log(collected_metrics, wandb_run, cumulative_timesteps)
        _init_empty_file(self.file_path)

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        data = []
        for logger in self.loggers.keys():
            data.extend(logger.collect_metrics(game_state))

        return np.array(data)

    def _compute_data(self, data: Dict[str, Any]):
        if "Rewards" in data.keys():
            for r in data["Rewards"].keys():
                rew = data["Rewards"][r]
                result = rew["value"] / rew["nb_episodes"]
                data["Rewards"][r] = result

    def _print_data(self, data, space_len: int = 0):
        for k, v in data.items():
            if space_len == 0:
                print("=" * 70)
            print("--" * space_len, f" {k}:{'': <10}", end=" ")
            if isinstance(v, Dict):
                print()
                self._print_data(v, space_len + 1)
            else:
                print(f"{v: >{20 + (len(max(data.keys(), key=lambda y: len(y))) - len(k))}.4f}")

    def _handle_metrics(self, metrics_data, logging_data: Dict[str, Any]):
        i = 0
        metrics_start = []
        metrics_end = []

        metrics_data = np.array(metrics_data)
        while i < metrics_data.shape[1]:
            len_shape = int(metrics_data[0][i][0])
            shape_metric = int(metrics_data[0][i + len_shape][0]) if len_shape > 0 else 1
            metrics_start.append(i + len_shape + 1)
            metrics_end.append(i + len_shape + shape_metric + 1)
            i += len_shape + shape_metric + 1

        for i, metrics_flags in enumerate(zip(metrics_start, metrics_end)):
            start, end = metrics_flags
            self.loggers[list(self.loggers.keys())[i]].append(np.mean(np.squeeze(metrics_data[:, start:end])))

        for logger in self.loggers.keys():
            for i, m in enumerate(logger.metrics):
                logging_data.setdefault(m, self.loggers[logger][i])

import json
import os
from typing import Dict, Any, Callable, List

import filelock
import numpy as np
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState
from wandb.sdk.wandb_run import Run


def _add(x: Any, y: Any):
    return x + y


def _replace(x: Any, y: Any):
    return x


def _init_empty_file(filepath):
    with open(filepath, "w") as f:
        json.dump({}, f)


class BallHeightLogger(MetricsLogger):
    @property
    def metrics(self) -> List[str]:
        return ["stats/ball_height"]

    def _collect_metrics(self, game_state: GameState) -> np.ndarray:
        return np.array([game_state.ball.position[2]])

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        wandb_run.log(
            data={"stats/ball_height": np.mean(
                np.nan_to_num(collected_metrics)
            )},
            step=cumulative_timesteps)


class Logger(MetricsLogger):
    def __init__(self):
        self.file_path = "./logging/logging.json"
        if "logging" not in os.listdir():
            os.makedirs("logging")
            _init_empty_file(self.file_path)
        else:
            if not os.path.exists(self.file_path):
                _init_empty_file(self.file_path)
        self.loggers = {}

    def add_logger(self, logger: MetricsLogger):
        self.loggers.setdefault(logger, [])

    def log(self, wandb_run: Run, step: int):
        data = self.get_results().copy()
        new_data = {}
        self._format_to_wandb(data, new_data)
        wandb_run.log(data=new_data, step=step)

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
                self._map_result(data[k], v, func)
            else:
                if k in data.keys():
                    data[k] = func(data[k], in_file[k])
                else:
                    data.setdefault(k, in_file[k])

    def add_result(self, data: Dict[str, Any], func_merge: Callable[[Any, Any], Any] = _add):
        data_cp = data.copy()

        if os.path.exists(self.file_path):
            in_file = self.get_results()
            self._map_result(data_cp, in_file, func_merge)

        self.save_results(data_cp)

    def report_metrics(self, collected_metrics, wandb_run: Run, cumulative_timesteps):
        self.log(wandb_run, cumulative_timesteps)
        _init_empty_file(self.file_path)

    def collect_metrics(self, game_state: GameState) -> np.ndarray:
        data = []
        for logger in self.loggers.keys():
            data.append(logger.collect_metrics(game_state))

        return np.array(data)

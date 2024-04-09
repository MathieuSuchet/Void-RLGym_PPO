import json
import os
from typing import Dict, Any, Callable

import filelock
import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym_ppo.util import MetricsLogger
from wandb.sdk.wandb_run import Run

from wandb_loggers import WandbMetricsLogger


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
        unformat_metrics = self._handle_metrics(metrics_data, new_data)
        self._print_data(unformat_metrics)

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

    def format_from_wandb(self, key: str, value: Any) -> Dict[str, Any]:
        all_attr = key.split("/")
        data = {}
        temp_ref = data

        len_attrs = 0
        while len_attrs < len(all_attr) - 1:
            temp_ref.setdefault(all_attr[len_attrs], {})
            temp_ref = temp_ref[all_attr[len_attrs]]
            len_attrs += 1

            if len_attrs == len(all_attr) - 1:
                temp_ref[all_attr[-1]] = value

        return data

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
            data.append(logger.collect_metrics(game_state))

        return data

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
        metrics_start = []
        metrics_end = []

        metrics_dict = {}

        for i, metric in enumerate(metrics_data[0]):
            metric_len = 0
            local_metric_start = []
            local_metric_end = []
            while metric_len < metric.shape[0]:
                len_shape_metrics = int(metric[metric_len])
                shape_metrics = metric[metric_len:metric_len + len_shape_metrics]

                is_metric_atomic = len(shape_metrics) == 0

                metric_len += len_shape_metrics
                metric_len += 1 if is_metric_atomic else len(shape_metrics)

                local_metric_start.append(metric_len)
                local_metric_end.append(metric_len + (int(shape_metrics[0]) if not is_metric_atomic else 1))

                metric_len += shape_metrics[0] if not is_metric_atomic else 1

            metrics_start.append(local_metric_start)
            metrics_end.append(local_metric_end)

        for i, metrics_flags in enumerate(zip(metrics_start, metrics_end)):
            local_starts, local_ends = metrics_flags
            for j in range(len(local_starts)):
                start, end = local_starts[j], local_ends[j]

                logger = list(self.loggers.keys())[i]

                self.loggers[list(self.loggers.keys())[i]].append(
                     logger.compute_data(np.array([metric_data[i][start:end] for metric_data in metrics_data])))

        for logger in self.loggers.keys():
            for i, m in enumerate(logger.metrics):
                logging_data.setdefault(m, self.loggers[logger][i])
                metrics_dict = Logger.merge(self.format_from_wandb(m, self.loggers[logger][i]), metrics_dict)

        return metrics_dict

    @staticmethod
    def merge(source, destination):
        for key, value in source.items():
            if isinstance(value, dict):
                # get node or create one
                node = destination.setdefault(key, {})
                Logger.merge(value, node)
            else:
                destination[key] = value

        return destination
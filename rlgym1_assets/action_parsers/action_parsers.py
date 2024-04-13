from typing import Dict, Any, List

import numpy as np
from rlgym_sim.utils import ActionParser
from rlgym_sim.utils.gamestates import GameState

from logger import Logger, _replace

from gym.spaces import Space, Discrete
import gym


class WandbActionParser(ActionParser):
    def __init__(self, action_parser: ActionParser, log_frequency: int = 15_000):
        super().__init__()
        self.parser = action_parser
        self.logger = Logger()
        self.bins: Dict[str, List[float]] = {}
        self._init_bins()
        self._n_steps = 0
        self._threshold = log_frequency

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        actions = self.parser.parse_actions(actions, state)

        for i, act in enumerate(actions):
            act = np.array(act)
            for j in range(act.shape[0]):
                if len(act.shape) > 1:
                    self.bins[list(self.bins.keys())[j]].append(np.mean(act[:, j]))
                else:
                    self.bins[list(self.bins.keys())[j]].append(float(act[j]))

        self._n_steps += 1

        if self._n_steps >= self._threshold:
            self._reset()
            self._n_steps = 0

        return actions

    def get_action_space(self) -> Space:
        return self.parser.get_action_space()

    def _init_bins(self):
        self.bins.setdefault("throttle", [])
        self.bins.setdefault("steer", [])
        self.bins.setdefault("pitch", [])
        self.bins.setdefault("yaw", [])
        self.bins.setdefault("roll", [])
        self.bins.setdefault("jump", [])
        self.bins.setdefault("boost", [])
        self.bins.setdefault("handbrake", [])

    def _reset(self):
        if len(self.bins["throttle"]) != 0:
            for key in self.bins.keys():
                self.bins[key] = np.mean(self.bins[key])

            self.logger.add_result({"actions": self.bins}, func_merge=_replace)

        self.bins.clear()
        self._init_bins()


class LookupAction(ActionParser):
    def __init__(self, bins=None):
        super().__init__()
        if bins is None:
            self.bins = [(-1, 0, 1)] * 5
        elif isinstance(bins[0], (float, int)):
            self.bins = [bins] * 5
        else:
            assert len(bins) == 5, "Need bins for throttle, steer, pitch, yaw and roll"
            self.bins = bins
        self._lookup_table = self.make_lookup_table(self.bins)

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        returns = []
        for a in actions:
            returns.append(self._lookup_table[int(a)])
        return np.array(returns)

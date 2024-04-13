from typing import Dict, Any, List

import numpy as np
from rlgym.api import ActionParser, AgentID, ActionType, StateType, EngineActionType, SpaceType

from logger import Logger, _replace


class WandbActionParser(ActionParser):
    def __init__(self, action_parser: ActionParser):
        self.parser = action_parser
        self.logger = Logger()
        self.bins: Dict[str, List[float]] = {}
        self._init_bins()

    def _init_bins(self):
        self.bins.setdefault("throttle", [])
        self.bins.setdefault("steer", [])
        self.bins.setdefault("pitch", [])
        self.bins.setdefault("yaw", [])
        self.bins.setdefault("roll", [])
        self.bins.setdefault("jump", [])
        self.bins.setdefault("boost", [])
        self.bins.setdefault("handbrake", [])

    def get_action_space(self, agent: AgentID) -> SpaceType:
        return self.parser.get_action_space(agent)

    def reset(self, initial_state: StateType, shared_info: Dict[str, Any]) -> None:
        self.parser.reset(initial_state, shared_info)

        if len(self.bins["throttle"]) != 0:
            for key in self.bins.keys():
                self.bins[key] = np.mean(self.bins[key])

            self.logger.add_result({"actions": self.bins}, func_merge=_replace)

        self.bins.clear()
        self._init_bins()

    def parse_actions(self, actions: Dict[AgentID, ActionType], state: StateType, shared_info: Dict[str, Any]) -> Dict[
        AgentID, EngineActionType]:
        actions = self.parser.parse_actions(actions, state, shared_info)

        for i, (agent, act) in enumerate(actions.items()):
            act = np.array(act)
            for j in range(act.shape[0]):
                if len(act.shape) > 1:
                    self.bins[list(self.bins.keys())[j]].append(np.mean(act[:, j]))
                else:
                    self.bins[list(self.bins.keys())[j]].append(float(act[j]))

        return actions

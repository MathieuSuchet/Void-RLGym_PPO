from typing import List, Dict, Any

from rlgym.api import AgentID, StateType
from rlgym.rocket_league.done_conditions import TimeoutCondition
from rlgym.rocket_league.done_conditions.no_touch_condition import NoTouchTimeoutCondition

from logger import Logger, _add


class LoggedNoTouchTimeoutCondition(NoTouchTimeoutCondition):
    def __init__(self, timeout: float):
        super().__init__(timeout)
        self.logger = Logger()

    def is_done(self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        result = super().is_done(agents, state, shared_info)
        if any(result.values()):
            self.logger.add_result({"DoneConditions": {
                "NoTouchTimeout": 1
            }}, func_merge=_add)
        return result


class LoggedTimeoutCondition(TimeoutCondition):
    def __init__(self, timeout: float):
        super().__init__(timeout)
        self.logger = Logger()

    def is_done(self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        result = super().is_done(agents, state, shared_info)
        if any(result.values()):
            self.logger.add_result({"DoneConditions": {
                "Timeout": 1
            }}, func_merge=_add)
        return result

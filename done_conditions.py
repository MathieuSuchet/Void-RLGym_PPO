from typing import List, Dict, Any

from rlgym.api import AgentID, StateType, DoneCondition
from rlgym.rocket_league.done_conditions import AnyCondition

from logger import Logger


class LoggedAnyCondition(AnyCondition):
    def __init__(self, *conditions: DoneCondition, name: str = "DoneCondition"):
        super().__init__(*conditions)
        self.logger = Logger()
        self.name = name

    def is_done(self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        data = {}
        log = False
        for c in self.conditions:

            result = any(c.is_done(agents, state, shared_info).values())
            if result:
                print(c.__class__)
            data.setdefault(c.__class__.__name__, int(result))
            if result and not log:
                log = True
        if log:
            self.logger.add_result({"DoneConditions": {
                self.name: data
            }})
        return super().is_done(agents, state, shared_info)
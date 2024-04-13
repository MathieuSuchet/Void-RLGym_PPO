from rlgym_sim.utils import TerminalCondition
from rlgym_sim.utils.gamestates import GameState

from logger import Logger


class MultiLoggedCondition(TerminalCondition):
    def __init__(self, *conditions: TerminalCondition):
        super().__init__()
        self.logger = Logger()
        self.conditions = conditions

    def reset(self, initial_state: GameState):
        for c in self.conditions:
            c.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        results = [c.is_terminal(current_state) for c in self.conditions]
        if any(results):
            data_to_log = {"terminal_conditions": {}}
            for result, condition in zip(results, self.conditions):
                if result:
                    data_to_log["terminal_conditions"].setdefault(condition.__class__.__name__, 1)

            self.logger.add_result(data_to_log)
            return True
        return False

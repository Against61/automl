from __future__ import annotations

from dataclasses import dataclass

from orchestrator.domain.state_machine import RunStateMachine
from orchestrator.persistence.schemas import RunStatus


@dataclass(slots=True)
class RunAggregate:
    run_id: str
    status: RunStatus

    def apply(self, action: str) -> RunStatus:
        self.status = RunStateMachine.transition(self.status, action)
        return self.status


from __future__ import annotations

from orchestrator.application.dto import HandleControlCommand
from orchestrator.application.use_cases.process_run_tick import ProcessRunTickUseCase


class HandleControlEventUseCase:
    def __init__(self, process_uc: ProcessRunTickUseCase):
        self.process_uc = process_uc

    async def execute(self, command: HandleControlCommand) -> bool:
        return await self.process_uc.handle_control_event(command.payload)


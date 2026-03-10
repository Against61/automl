from __future__ import annotations

from orchestrator.application.dto import SubmitTaskCommand
from orchestrator.application.use_cases.process_run_tick import ProcessRunTickUseCase


class SubmitTaskUseCase:
    def __init__(self, process_uc: ProcessRunTickUseCase):
        self.process_uc = process_uc

    async def execute(self, command: SubmitTaskCommand) -> str | None:
        return await self.process_uc.submit_task_event(command.payload)


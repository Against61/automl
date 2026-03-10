from __future__ import annotations

from orchestrator.application.dto import FinalizeRunCommand
from orchestrator.application.use_cases.process_run_tick import ProcessRunTickUseCase
from orchestrator.persistence.schemas import RunStatus


class FinalizeRunUseCase:
    def __init__(self, process_uc: ProcessRunTickUseCase):
        self.process_uc = process_uc

    async def execute(self, command: FinalizeRunCommand) -> None:
        if command.final_status == RunStatus.CANCELLED:
            await self.process_uc.finalize_cancelled(command.run_id)


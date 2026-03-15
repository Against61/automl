from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.application.dto import (
    FinalizeRunCommand,
    HandleControlCommand,
    ProcessRunTickCommand,
    SubmitTaskCommand,
)
from orchestrator.application.use_cases.finalize_run import FinalizeRunUseCase
from orchestrator.application.use_cases.handle_control import HandleControlEventUseCase
from orchestrator.application.use_cases.process_run_tick import ProcessRunTickUseCase
from orchestrator.application.use_cases.submit_task import SubmitTaskUseCase
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.execution.codex_runner import CodexRunner
from orchestrator.execution.policy import PolicyEngine
from orchestrator.execution.verifier import Verifier
from orchestrator.persistence.db import Database
from orchestrator.planning.planner import Planner
from orchestrator.planning.ralph import RalphBacklogService
from orchestrator.runtime.bus import InMemoryEventBus, RedisEventBus
from orchestrator.config import Settings
from orchestrator.persistence.schemas import RunStatus

EventBus = RedisEventBus | InMemoryEventBus


class SessionManager:
    """
    Compatibility facade.

    Public API stays stable while orchestration logic lives in application use-cases.
    """

    def __init__(
        self,
        settings: Settings,
        db: Database,
        bus: EventBus,
        planner: Planner,
        policy_engine: PolicyEngine,
        ralph_backlog: RalphBacklogService,
        codex_runner: CodexRunner,
        verifier: Verifier,
        artifact_publisher: ArtifactPublisher,
    ) -> None:
        self._process_run_uc = ProcessRunTickUseCase(
            settings=settings,
            db=db,
            bus=bus,
            planner=planner,
            policy_engine=policy_engine,
            ralph_backlog=ralph_backlog,
            codex_runner=codex_runner,
            verifier=verifier,
            artifact_publisher=artifact_publisher,
        )
        self._submit_task_uc = SubmitTaskUseCase(self._process_run_uc)
        self._handle_control_uc = HandleControlEventUseCase(self._process_run_uc)
        self._finalize_uc = FinalizeRunUseCase(self._process_run_uc)

    async def submit_task_event(self, payload: dict[str, Any]) -> str | None:
        return await self._submit_task_uc.execute(SubmitTaskCommand(payload=payload))

    async def handle_control_event(self, payload: dict[str, Any]) -> bool:
        return await self._handle_control_uc.execute(HandleControlCommand(payload=payload))

    async def finalize_cancelled(self, run_id: str) -> None:
        await self._finalize_uc.execute(
            FinalizeRunCommand(
                run_id=run_id,
                final_status=RunStatus.CANCELLED,
                workspace_path=Path("."),
            )
        )

    async def process_run(self, run_id: str) -> None:
        await self._process_run_uc.process_run(ProcessRunTickCommand(run_id=run_id).run_id)

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.config import Settings
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import RunRecord, RunStatus

StageOutcome = Literal["unhandled", "continue", "return"]


class RunContextStage:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        bus: Any,
        workspace_snapshot_service: WorkspaceSnapshotService,
        set_status: Callable[[str, RunStatus, str | None, RunRecord | None], Awaitable[None]],
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.workspace_snapshot_service = workspace_snapshot_service
        self.set_status = set_status

    async def handle_waiting_status(self, *, run_id: str, run: RunRecord) -> StageOutcome:
        if run.status == RunStatus.WAITING_APPROVAL:
            if run.approved_at is not None:
                await self.set_status(run_id, RunStatus.CONTEXT_READY, None, None)
                await self.bus.publish_internal("run.approved", {"run_id": run_id})
                return "continue"
            return "return"
        if run.status == RunStatus.WAITING_PLAN_REVIEW:
            if run.approved_at is not None or self.settings.auto_approve_in_pilot:
                await self.set_status(run_id, RunStatus.CONTEXT_READY, "plan review approved", None)
                await self.bus.publish_internal(
                    "run.plan_review_approved",
                    {"run_id": run_id},
                )
                return "continue"
            return "return"
        return "unhandled"

    async def handle_received(
        self,
        *,
        run_id: str,
        run: RunRecord,
        workspace_path: Path,
    ) -> StageOutcome:
        if run.status != RunStatus.RECEIVED:
            return "unhandled"
        acquired = await self.db.acquire_workspace_lock(run.workspace_id, run_id)
        if not acquired:
            await self.set_status(
                run_id,
                RunStatus.RECEIVED,
                "waiting for workspace lock",
                None,
            )
            return "return"
        self.workspace_snapshot_service.refresh(workspace_path)
        await self.db.set_context(run_id, [])
        await self.set_status(run_id, RunStatus.CONTEXT_READY, None, None)
        await self.bus.publish_internal("run.context_ready", {"run_id": run_id})
        return "continue"

    async def handle_plan_ready(self, *, run_id: str, run: RunRecord) -> StageOutcome:
        if run.status != RunStatus.PLAN_READY:
            return "unhandled"
        await self.set_status(run_id, RunStatus.EXECUTING, None, None)
        return "continue"

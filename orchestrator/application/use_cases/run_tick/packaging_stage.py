from __future__ import annotations

import logging
from pathlib import Path
from typing import Awaitable, Callable, Literal

from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import RunRecord, RunResultSummary, RunStatus

logger = logging.getLogger(__name__)

StageOutcome = Literal["unhandled", "continue", "return"]


class RunPackagingStage:
    def __init__(
        self,
        *,
        db: Database,
        artifact_publisher: ArtifactPublisher,
        ralph_service: RalphScenarioService,
        set_status: Callable[[str, RunStatus, str | None, RunRecord | None], Awaitable[None]],
        publish_result: Callable[[str, str, str, RunResultSummary, dict[str, str]], Awaitable[None]],
    ) -> None:
        self.db = db
        self.artifact_publisher = artifact_publisher
        self.ralph_service = ralph_service
        self.set_status = set_status
        self.publish_result = publish_result

    async def handle_packaging(
        self,
        *,
        run_id: str,
        run: RunRecord,
        task: dict[str, str],
        workspace_path: Path,
    ) -> StageOutcome:
        if run.status != RunStatus.PACKAGING:
            return "unhandled"

        planned_steps = len((run.plan_json or {}).get("steps", []))
        executed_steps = await self.db.count_attempted_steps(run_id)
        verification = "passed"
        if run.verification_json and run.verification_json.get("status") != "passed":
            verification = "failed"
        final_status = RunStatus.COMPLETED if verification == "passed" else RunStatus.FAILED
        summary = RunResultSummary(
            planned_steps=planned_steps,
            executed_steps=executed_steps,
            verification=verification,
        )
        artifacts = await self.artifact_publisher.package(
            run_id=run_id,
            task_id=run.task_id,
            workspace_path=workspace_path,
            status=final_status.value.lower(),
            summary=summary.model_dump(mode="json"),
        )
        await self.publish_result(
            run_id=run_id,
            task_id=run.task_id,
            status=final_status.value.lower(),
            summary=summary,
            artifacts=artifacts,
        )
        if final_status == RunStatus.COMPLETED:
            try:
                await self.ralph_service.maybe_handle_completion(
                    run=run,
                    task=task,
                    summary=summary,
                    workspace_path=workspace_path,
                )
            except Exception as exc:
                logger.exception("ralph completion hook failed for run %s: %s", run_id, exc)
        await self.set_status(run_id, final_status, None, None)
        await self.db.release_workspace_lock(run.workspace_id, run_id)
        return "return"

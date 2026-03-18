from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from orchestrator.application.use_cases.run_tick.planning_context import PlanningContextService
from orchestrator.domain.errors import InvalidTransitionError
from orchestrator.domain.state_machine import RunStateMachine
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.execution.codex_runner import CodexRunner
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import (
    ControlEvent,
    RunRecord,
    RunResultArtifacts,
    RunResultEvent,
    RunResultSummary,
    RunStatus,
    TaskSubmittedEvent,
)
from orchestrator.config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_REQUIREMENT_CONSTRAINTS: tuple[tuple[str, str], ...] = (
    ("SUPERVISION_SOURCE:", "SUPERVISION_SOURCE: explicit_dataset_annotations_only"),
    ("ACCEPTANCE_METRIC_SOURCE:", "ACCEPTANCE_METRIC_SOURCE: explicit_dataset_annotations_only"),
    ("TARGET_INVENTION_ALLOWED:", "TARGET_INVENTION_ALLOWED: false"),
    ("TASK_REFORMULATION_ALLOWED:", "TASK_REFORMULATION_ALLOWED: false"),
    ("PSEUDOLABELS_ALLOWED:", "PSEUDOLABELS_ALLOWED: false"),
    ("ACCEPTANCE_SPLIT:", "ACCEPTANCE_SPLIT: heldout_or_disjoint_eval_only"),
    ("SMOKE_ACCEPTANCE_ALLOWED:", "SMOKE_ACCEPTANCE_ALLOWED: false"),
    ("COMPUTE_DEVICE:", "COMPUTE_DEVICE: mps"),
    ("MAX_QUALITY_RETRIES:", "MAX_QUALITY_RETRIES: 10"),
)


class RunCoordinatorSupportService:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        bus: Any,
        codex_runner: CodexRunner,
        artifact_publisher: ArtifactPublisher,
        planning_context_service: PlanningContextService,
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.codex_runner = codex_runner
        self.artifact_publisher = artifact_publisher
        self.planning_context_service = planning_context_service

    async def set_status(
        self,
        run_id: str,
        target_status: RunStatus,
        error_message: str | None = None,
        run: RunRecord | None = None,
    ) -> None:
        current = run or await self.db.get_run(run_id)
        from_status = current.status if current else None
        if from_status and from_status != target_status:
            action = RunStateMachine.infer_action(from_status, target_status)
            if action != "direct_set":
                try:
                    validated = RunStateMachine.transition(from_status, action)
                    if validated != target_status:
                        logger.warning(
                            "state-machine mismatch for run %s: %s --%s--> %s (expected %s)",
                            run_id,
                            from_status.value,
                            action,
                            validated.value,
                            target_status.value,
                        )
                except InvalidTransitionError:
                    logger.warning(
                        "state-machine validation failed for run %s: %s -> %s",
                        run_id,
                        from_status.value,
                        target_status.value,
                    )
        await self.db.update_run_status(run_id, target_status, error_message)
        if from_status and from_status != target_status:
            await self.bus.publish_internal(
                "domain.run.transitioned",
                {
                    "run_id": run_id,
                    "from_status": from_status.value,
                    "to_status": target_status.value,
                },
            )

    def workspace_dir(self, workspace_id: str) -> Path:
        path = self.settings.workspace_root / workspace_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _with_default_requirement_constraints(self, event: TaskSubmittedEvent) -> TaskSubmittedEvent:
        constraints = [str(item).strip() for item in event.payload.constraints if str(item).strip()]
        existing_prefixes = {item.split(":", 1)[0].strip().upper() + ":" for item in constraints if ":" in item}
        for prefix, value in _DEFAULT_REQUIREMENT_CONSTRAINTS:
            if prefix.upper() in existing_prefixes:
                continue
            constraints.append(value)
        event.payload.constraints = constraints
        return event

    async def submit_task_event(self, payload: dict[str, Any]) -> str | None:
        event = self._with_default_requirement_constraints(TaskSubmittedEvent.model_validate(payload))
        normalized_payload = event.model_dump(mode="json")
        inserted = await self.db.record_stream_event(
            event_id=str(event.event_id),
            stream=self.settings.stream_tasks,
            event_type=event.event_type,
            payload_json=json.dumps(normalized_payload, ensure_ascii=True),
        )
        if not inserted:
            logger.info("duplicate task event ignored: %s", event.event_id)
            return None
        await self.db.upsert_task(event)
        run_signature = self.planning_context_service.build_task_goal_signature(event.payload.model_dump(mode="json"))
        run_id = await self.db.create_or_get_run(
            task_id=str(event.task_id),
            workspace_id=event.workspace_id,
            priority=event.priority,
            goal_signature=run_signature,
        )
        await self.bus.publish_internal(
            "run.created",
            {"run_id": run_id, "task_id": str(event.task_id), "workspace_id": event.workspace_id},
        )
        return run_id

    async def handle_control_event(self, payload: dict[str, Any]) -> bool:
        event = ControlEvent.model_validate(payload)
        inserted = await self.db.record_stream_event(
            event_id=str(event.event_id),
            stream=self.settings.stream_control,
            event_type=event.event_type,
            payload_json=json.dumps(payload, ensure_ascii=True),
            run_id=str(event.payload.run_id),
        )
        if not inserted:
            return False

        run_id = str(event.payload.run_id)
        if event.event_type == "run.approve":
            changed = await self.db.set_approved(run_id)
            if changed:
                await self.bus.publish_internal("run.approved", {"run_id": run_id})
            return changed

        if event.event_type == "run.cancel":
            changed = await self.db.cancel_run(run_id, reason=event.payload.reason)
            if changed:
                await self.codex_runner.cancel_run(run_id)
                await self.finalize_cancelled(run_id)
            return changed

        if event.event_type == "run.retry":
            changed = await self.db.reset_run_for_retry(run_id)
            if changed:
                await self.bus.publish_internal("run.retry", {"run_id": run_id})
            return changed
        return False

    async def finalize_cancelled(self, run_id: str) -> None:
        run = await self.db.get_run(run_id)
        if not run:
            return
        summary = RunResultSummary(
            planned_steps=len((run.plan_json or {}).get("steps", [])),
            executed_steps=await self.db.count_attempted_steps(run_id),
            verification="failed",
        )
        workspace_path = self.workspace_dir(run.workspace_id)
        artifacts = await self.artifact_publisher.package(
            run_id=run_id,
            task_id=run.task_id,
            workspace_path=workspace_path,
            status="cancelled",
            summary=summary.model_dump(mode="json"),
        )
        await self.publish_result(run_id, run.task_id, "cancelled", summary, artifacts)
        await self.db.release_workspace_lock(run.workspace_id, run_id)

    async def schedule_replan(
        self,
        *,
        run_id: str,
        target_status: RunStatus,
        error_message: str,
    ) -> int:
        cycle = await self.db.advance_execution_cycle(run_id)
        await self.set_status(run_id, target_status, error_message)
        return cycle

    async def publish_result(
        self,
        run_id: str,
        task_id: str,
        status: str,
        summary: RunResultSummary,
        artifacts: dict[str, str],
    ) -> None:
        event = RunResultEvent(
            event_id=uuid4(),
            run_id=run_id,
            task_id=task_id,
            status=status,
            artifacts=RunResultArtifacts(**artifacts),
            summary=summary,
        )
        await self.bus.publish_result(event.model_dump(mode="json"))

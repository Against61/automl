from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from orchestrator.application.services.plan_contract_service import PlanContractService
from orchestrator.application.services.quality_gate_service import QualityGateService
from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.application.services.improvement_strategy_service import ImprovementStrategyService
from orchestrator.application.services.recovery_service import MissingFileRecoveryService
from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.domain.errors import InvalidTransitionError
from orchestrator.domain.state_machine import RunStateMachine
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.runtime.bus import InMemoryEventBus, RedisEventBus
from orchestrator.execution.codex_runner import CodexRunner, StepExecutionResult
from orchestrator.config import Settings
from orchestrator.persistence.db import Database, normalize_verification_payload
from orchestrator.planning.planner import PlanInput, Planner, PlannerError
from orchestrator.execution.policy import PolicyEngine
from orchestrator.planning.ralph import RalphBacklogError, RalphBacklogService
from orchestrator.persistence.schemas import (
    ArtifactKind,
    ControlEvent,
    PlannerStep,
    PlannerPlan,
    RetrievedContext,
    RunResultSummary,
    RunRecord,
    RunResultArtifacts,
    RunResultEvent,
    RunStatus,
    StepIOResult,
    StepIntent,
    TaskSubmittedEvent,
)
from orchestrator.execution.verifier import Verifier

logger = logging.getLogger(__name__)

EventBus = RedisEventBus | InMemoryEventBus


class ProcessRunTickUseCase:
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
        self.settings = settings
        self.db = db
        self.bus = bus
        self.planner = planner
        self.policy_engine = policy_engine
        self.codex_runner = codex_runner
        self.verifier = verifier
        self.artifact_publisher = artifact_publisher
        self._run_mutex: dict[str, Any] = {}
        self.plan_contract_service = PlanContractService(
            plan_review_enabled=settings.plan_review_enabled,
            strictness=settings.contract_strictness,
        )
        self.recovery_service = MissingFileRecoveryService()
        quality_gate_service = QualityGateService(ralph_backlog=ralph_backlog)
        self.improvement_strategy_service = ImprovementStrategyService(quality_gate_service=quality_gate_service)
        self.workspace_snapshot_service = WorkspaceSnapshotService()
        self.ralph_service = RalphScenarioService(
            settings=settings,
            backlog=ralph_backlog,
            planner=planner,
            db=db,
            bus=bus,
            quality_gate_service=quality_gate_service,
        )

    async def _set_status(
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

    def _workspace_dir(self, workspace_id: str) -> Path:
        path = self.settings.workspace_root / workspace_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def submit_task_event(self, payload: dict[str, Any]) -> str | None:
        event = TaskSubmittedEvent.model_validate(payload)
        inserted = await self.db.record_stream_event(
            event_id=str(event.event_id),
            stream=self.settings.stream_tasks,
            event_type=event.event_type,
            payload_json=json.dumps(payload, ensure_ascii=True),
        )
        if not inserted:
            logger.info("duplicate task event ignored: %s", event.event_id)
            return None
        await self.db.upsert_task(event)
        run_signature = self._build_task_goal_signature(event.payload.model_dump(mode="json"))
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
        task = await self.db.get_task(run.task_id)
        summary = RunResultSummary(
            planned_steps=len((run.plan_json or {}).get("steps", [])),
            executed_steps=await self.db.count_attempted_steps(run_id),
            verification="failed",
        )
        workspace_path = self._workspace_dir(run.workspace_id)
        artifacts = await self.artifact_publisher.package(
            run_id=run_id,
            task_id=run.task_id,
            workspace_path=workspace_path,
            status="cancelled",
            summary=summary.model_dump(mode="json"),
        )
        await self._publish_result(run_id, run.task_id, "cancelled", summary, artifacts)
        await self.db.release_workspace_lock(run.workspace_id, run_id)

    async def process_run(self, run_id: str) -> None:
        mutex = self._run_mutex.setdefault(run_id, asyncio.Lock())
        async with mutex:
            await self._process_run_impl(run_id)

    def _needs_manual_approval(self, run_approved_at: datetime | None, requires_approval: bool) -> bool:
        return bool(requires_approval and run_approved_at is None and not self.settings.auto_approve_in_pilot)

    def _evaluate_plan_contract(
        self,
        step: Any,
        workspace_path: Path,
        result: Any,
    ) -> tuple[bool, str]:
        return self.plan_contract_service.evaluate(
            step=step,
            workspace_path=workspace_path,
            result=result,
        )

    async def _attempt_missing_python_file_recover(
        self,
        run_id: str,
        step: PlannerStep,
        step_index: int,
        workspace_path: Path,
        result: StepExecutionResult,
        run_path: Path,
    ) -> tuple[PlannerStep, StepExecutionResult] | None:
        decision = self.recovery_service.detect_missing_python_file(result.stderr_text)
        missing_path = decision.missing_path
        if not missing_path:
            return None

        candidates = self.recovery_service.find_python_file_candidates(missing_path, workspace_path)
        candidate_summary = (
            ", ".join(str(path) for path in candidates)
            if candidates
            else "none"
        )
        await self.db.insert_run_step(
            run_id=run_id,
            step_id=f"{step.id}-fs-scan",
            step_title=f"Filesystem scan for missing `{missing_path}`",
            step_index=step_index,
            action="read",
            command=f"scan filesystem for python file `{missing_path}`",
            status="completed",
            stdout_text=(
                f"summary: python file missing for step execution\n"
                f"missing: {missing_path}\n"
                f"discovered candidates: {candidate_summary}"
            ),
            stderr_text="",
            duration_ms=0,
        )

        if len(candidates) != 1:
            return None

        replacement = str(candidates[0])
        repaired_step = self.recovery_service.replace_missing_file_in_step(
            step=step,
            expected_missing=missing_path,
            replacement=replacement,
        )
        if repaired_step is None:
            return None

        repaired_result = await self.codex_runner.execute_step(
            run_id=run_id,
            step=repaired_step,
            workspace_path=workspace_path,
            run_path=run_path,
        )
        if repaired_result.status == "completed":
            repaired_result.summary = (
                f"{result.summary}; recovered missing file path via filesystem scan: "
                f"{missing_path} -> {replacement}"
            )
        else:
            repaired_result.summary = (
                f"{result.summary}; retry with recovered path failed: {replacement}"
            )

        return repaired_step, repaired_result

    def _extract_missing_python_file_path(self, text: str) -> str | None:
        decision = self.recovery_service.detect_missing_python_file(text)
        return decision.missing_path

    def _is_quality_threshold_soft_failure(
        self,
        *,
        step: PlannerStep,
        result: StepExecutionResult,
        workspace_path: Path,
    ) -> bool:
        if result.status == "completed":
            return False
        if step.step_intent != StepIntent.run_training:
            return False

        output = f"{result.stderr_text}\n{result.stdout_text}".lower()
        markers = (
            "target accuracy threshold not reached",
            "accuracy threshold not reached",
            "target metric threshold not reached",
            "required metric not reached",
            "did not reach target accuracy",
        )
        if not any(marker in output for marker in markers):
            return False

        for spec in step.expected_artifacts:
            if spec.kind != ArtifactKind.metrics or not spec.path:
                continue
            target = self._resolve_expected_artifact_target(spec.path, workspace_path)
            if target is None:
                continue
            try:
                if target.exists() and target.stat().st_size > 0:
                    return True
            except OSError:
                continue

        for fallback in (workspace_path / "metrics.md", workspace_path / "metrics.json"):
            try:
                if fallback.exists() and fallback.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    def _resolve_expected_artifact_target(self, raw_path: str, workspace_path: Path) -> Path | None:
        cleaned = raw_path.strip().strip("`\"'")
        if not cleaned:
            return None
        normalized = cleaned.replace("\\", "/").strip("/")
        parts = [part for part in normalized.split("/") if part and part != "."]
        if not parts:
            return None
        if parts[0] == "workspace":
            parts = parts[1:]
        if parts and parts[0] == workspace_path.name:
            parts = parts[1:]
        if not parts:
            return None
        try:
            return (workspace_path / "/".join(parts)).resolve()
        except OSError:
            return workspace_path / "/".join(parts)

    def _stepio_result_sort_key(self, path: Path) -> tuple[int, str]:
        name = path.name
        match = re.search(r"\.step_result(?:\.(\d+))?\.json$", name)
        if not match:
            return (0, name)
        suffix = match.group(1)
        index = 1 if suffix is None else int(suffix)
        return (index, name)

    def _load_stepio_result(self, path: Path) -> StepIOResult | None:
        try:
            return StepIOResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("unable to parse stepio artifact: %s", path)
            return None

    def _read_text_if_exists(self, path_value: str | None) -> str:
        if not path_value:
            return ""
        try:
            return Path(path_value).read_text(encoding="utf-8")
        except OSError:
            return ""

    def _infer_reconciled_action(self, payload: StepIOResult) -> str:
        if payload.operation == "verify_metrics":
            return "verify"
        command = str(payload.command or "").strip().lower()
        if command.startswith("codex "):
            return "codex"
        if command:
            return "shell"
        return "read"

    async def _reconcile_stepio_artifacts(
        self,
        *,
        run: RunRecord,
        run_path: Path,
    ) -> dict[str, StepIOResult]:
        if not run_path.exists():
            return {}

        stepio_files = sorted(run_path.glob("*.step_result*.json"), key=self._stepio_result_sort_key)
        if not stepio_files:
            return {}

        plan_index_by_id: dict[str, int] = {}
        plan_title_by_id: dict[str, str] = {}
        if run.plan_json:
            try:
                current_plan = PlannerPlan.model_validate(run.plan_json)
                plan_index_by_id = {step.id: index for index, step in enumerate(current_plan.steps)}
                plan_title_by_id = {step.id: step.title for step in current_plan.steps}
            except Exception:
                plan_index_by_id = {}
                plan_title_by_id = {}

        existing_rows = await self.db.list_run_steps(run.run_id)
        existing_counts: dict[str, int] = {}
        for row in existing_rows:
            step_id = str(row.get("step_id") or "").strip()
            if not step_id:
                continue
            existing_counts[step_id] = existing_counts.get(step_id, 0) + 1

        grouped: dict[str, list[tuple[Path, StepIOResult]]] = {}
        latest_by_step_id: dict[str, StepIOResult] = {}
        inserted = 0

        for path in stepio_files:
            payload = self._load_stepio_result(path)
            if payload is None:
                continue
            grouped.setdefault(payload.step_id, []).append((path, payload))
            latest_by_step_id[payload.step_id] = payload

        for step_id, entries in grouped.items():
            entries.sort(key=lambda item: self._stepio_result_sort_key(item[0]))
            already_recorded = existing_counts.get(step_id, 0)
            for path, payload in entries[already_recorded:]:
                stdout_text = self._read_text_if_exists(payload.stdout_path)
                stderr_text = self._read_text_if_exists(payload.stderr_path)
                prefixed_stdout = f"summary: {payload.summary}"
                if stdout_text:
                    prefixed_stdout = f"{prefixed_stdout}\n{stdout_text}".strip()
                created_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
                await self.db.insert_run_step(
                    run_id=run.run_id,
                    step_id=payload.step_id,
                    step_title=plan_title_by_id.get(payload.step_id) or payload.step_id,
                    step_index=plan_index_by_id.get(payload.step_id, 0),
                    action=self._infer_reconciled_action(payload),
                    command=payload.command,
                    status="completed" if payload.status in {"completed", "auto_repaired"} else "failed",
                    stdout_text=prefixed_stdout,
                    stderr_text=stderr_text,
                    duration_ms=payload.duration_ms,
                    created_at=created_at,
                )
                inserted += 1

        if inserted:
            await self.bus.publish_internal(
                "run.stepio_reconciled",
                {
                    "run_id": run.run_id,
                    "inserted_steps": inserted,
                },
            )
        return latest_by_step_id

    async def _sync_run_progress_from_stepio(
        self,
        *,
        run: RunRecord,
        latest_by_step_id: dict[str, StepIOResult],
    ) -> bool:
        if not run.plan_json:
            return False
        try:
            plan = PlannerPlan.model_validate(run.plan_json)
        except Exception:
            return False
        if not plan.steps:
            return False

        completed_prefix = 0
        for step in plan.steps:
            payload = latest_by_step_id.get(step.id)
            if payload is None or payload.status not in {"completed", "auto_repaired"}:
                break
            completed_prefix += 1

        changed = False
        if completed_prefix > run.next_step_index:
            await self.db.set_next_step_index(run.run_id, completed_prefix)
            changed = True

        if run.status == RunStatus.EXECUTING and completed_prefix >= len(plan.steps):
            await self._set_status(run.run_id, RunStatus.VERIFYING)
            await self.bus.publish_internal(
                "run.progress_recovered",
                {
                    "run_id": run.run_id,
                    "next_step_index": completed_prefix,
                    "transition": "EXECUTING->VERIFYING",
                },
            )
            return True

        if changed:
            await self.bus.publish_internal(
                "run.progress_recovered",
                {
                    "run_id": run.run_id,
                    "next_step_index": completed_prefix,
                },
            )
        return changed

    async def _process_run_impl(self, run_id: str) -> None:
        while True:
            run = await self.db.get_run(run_id)
            if run is None:
                return
            if run.status in RunStatus.terminal():
                await self.db.release_workspace_lock(run.workspace_id, run_id)
                return

            task = await self.db.get_task(run.task_id)
            if not task:
                await self._set_status(run_id, RunStatus.FAILED, "task not found")
                continue

            task_signature = self._build_task_signature_from_record(task)
            if task_signature is not None and run.goal_signature != task_signature:
                await self.db.reset_run_for_goal_change(run_id, goal_signature=task_signature)
                await self.bus.publish_internal(
                    "run.goal_signature_mismatch",
                    {
                        "run_id": run_id,
                        "task_id": run.task_id,
                        "previous_signature": run.goal_signature,
                        "new_signature": task_signature,
                    },
                )
                logger.info(
                    "run %s reset due goal signature change: %s -> %s",
                    run_id,
                    run.goal_signature,
                    task_signature,
                )
                continue

            if run.status == RunStatus.WAITING_APPROVAL:
                if run.approved_at is not None:
                    await self._set_status(run_id, RunStatus.CONTEXT_READY)
                    await self.bus.publish_internal("run.approved", {"run_id": run_id})
                    continue
                return
            if run.status == RunStatus.WAITING_PLAN_REVIEW:
                if run.approved_at is not None or self.settings.auto_approve_in_pilot:
                    await self._set_status(run_id, RunStatus.CONTEXT_READY, "plan review approved")
                    await self.bus.publish_internal(
                        "run.plan_review_approved",
                        {"run_id": run_id},
                    )
                    continue
                return

            workspace_path = self._workspace_dir(run.workspace_id)
            run_path = self.artifact_publisher.run_dir(run_id)
            latest_stepio = await self._reconcile_stepio_artifacts(run=run, run_path=run_path)
            if latest_stepio:
                refreshed_run = await self.db.get_run(run_id)
                if refreshed_run is not None:
                    run = refreshed_run
                    if await self._sync_run_progress_from_stepio(run=run, latest_by_step_id=latest_stepio):
                        continue

            try:
                if run.status == RunStatus.RECEIVED:
                    acquired = await self.db.acquire_workspace_lock(run.workspace_id, run_id)
                    if not acquired:
                        await self._set_status(
                            run_id,
                            RunStatus.RECEIVED,
                            "waiting for workspace lock",
                        )
                        return
                    contexts: list[RetrievedContext] = []
                    self.workspace_snapshot_service.refresh(workspace_path)
                    await self.db.set_context(run_id, [item.model_dump(mode="json") for item in contexts])
                    await self._set_status(run_id, RunStatus.CONTEXT_READY)
                    await self.bus.publish_internal("run.context_ready", {"run_id": run_id})
                    continue

                if run.status == RunStatus.CONTEXT_READY:
                    contexts = self._coerce_run_contexts(run.context_json)
                    last_failed_step = await self.db.get_latest_failed_step(run_id)
                    previous_verification = self._latest_verification_snapshot(run.verification_json)
                    workspace_snapshot_summary = self.workspace_snapshot_service.get_prompt_summary(workspace_path)
                    experiment_history = await self._list_experiment_history(run=run, task=task)
                    experiment_history_summary = self._format_experiment_history_summary(experiment_history)
                    missing_quality_reason = self.ralph_service.ralph_quality_requirement_missing_reason(
                        task=task,
                        workspace_path=workspace_path,
                    )
                    if missing_quality_reason:
                        await self._set_status(
                            run_id,
                            RunStatus.WAITING_PLAN_REVIEW,
                            missing_quality_reason,
                        )
                        await self.bus.publish_internal(
                            "run.waiting_plan_review",
                            {
                                "run_id": run_id,
                                "reason": missing_quality_reason,
                            },
                        )
                        return
                    plan = await self.ralph_service.maybe_build_plan(
                        task=task,
                        run=run,
                        contexts=contexts,
                        workspace_path=workspace_path,
                        last_failed_step=last_failed_step,
                        previous_verification=previous_verification,
                        experiment_history_summary=experiment_history_summary,
                    )
                    if plan is None:
                        plan_input = self._build_plan_input(
                            task=task,
                            run=run,
                            workspace_id=run.workspace_id,
                            contexts=contexts,
                            workspace_snapshot_summary=workspace_snapshot_summary,
                            experiment_history_summary=experiment_history_summary,
                            last_failed_step=last_failed_step,
                            previous_verification=previous_verification,
                        )
                        previous_plan: PlannerPlan | None = None
                        if run.plan_json:
                            try:
                                previous_plan = PlannerPlan.model_validate(run.plan_json)
                            except Exception:
                                previous_plan = None
                        should_replan = bool(
                            run.error_message
                            and self.settings.execution_failure_replan_enabled
                            and previous_plan is not None
                        )
                        if should_replan and previous_plan is not None:
                            plan = await self.planner.replan(
                                plan_input,
                                failure_reason=run.error_message or "",
                                previous_plan=previous_plan,
                            )
                            await self.bus.publish_internal(
                                "run.replanned",
                                {
                                    "run_id": run_id,
                                    "reason": run.error_message or "",
                                },
                            )
                        else:
                            plan = await self.planner.build_plan(plan_input)
                    plan = self._attach_selected_skills_to_plan(
                        plan=plan,
                        previous_verification=previous_verification,
                        experiment_history=experiment_history,
                    )
                    if len(plan.steps) > self.settings.max_run_steps:
                        await self._set_status(
                            run_id,
                            RunStatus.FAILED,
                            f"plan exceeds max_run_steps={self.settings.max_run_steps}",
                        )
                        continue
                    await self.db.set_plan(run_id, plan.model_dump(mode="json"))
                    if plan.planner_sanitization:
                        await self.bus.publish_internal(
                            "planner.payload_sanitized",
                            {
                                "run_id": run_id,
                                "changes": [item.model_dump(mode="json") for item in plan.planner_sanitization],
                            },
                        )
                    requires_approval, decisions = self.policy_engine.plan_requires_approval(plan.steps, workspace_path)
                    denied = any(decision.decision == "DENY" for decision in decisions)
                    for decision in decisions:
                        await self.db.add_policy_decision(
                            run_id=run_id,
                            layer=decision.layer,
                            subject=decision.subject,
                            decision=decision.decision,
                            reason=decision.reason,
                        )
                    if denied:
                        await self._set_status(run_id, RunStatus.FAILED, "policy denied plan")
                        continue
                    if self._needs_manual_approval(run.approved_at, requires_approval):
                        await self._set_status(run_id, RunStatus.WAITING_APPROVAL)
                        await self.bus.publish_internal("run.waiting_approval", {"run_id": run_id})
                        return
                    await self._set_status(run_id, RunStatus.PLAN_READY)
                    await self.bus.publish_internal("run.plan_ready", {"run_id": run_id})
                    continue

                if run.status == RunStatus.PLAN_READY:
                    await self._set_status(run_id, RunStatus.EXECUTING)
                    continue

                if run.status == RunStatus.EXECUTING:
                    if not run.plan_json:
                        await self._set_status(run_id, RunStatus.FAILED, "missing plan")
                        continue
                    plan = PlannerPlan.model_validate(run.plan_json)
                    steps = plan.steps
                    contexts = self._coerce_run_contexts(run.context_json)
                    for idx in range(run.next_step_index, len(steps)):
                        current = await self.db.get_run(run_id)
                        if current is None:
                            return
                        if current.status == RunStatus.CANCELLED:
                            await self.finalize_cancelled(run_id)
                            return
                        step = steps[idx]
                        decisions = self.policy_engine.evaluate_step(step, workspace_path)
                        denied = any(d.decision == "DENY" for d in decisions)
                        requires_approval = any(d.decision == "REQUIRE_APPROVAL" for d in decisions)
                        for decision in decisions:
                            await self.db.add_policy_decision(
                                run_id=run_id,
                                layer=decision.layer,
                                subject=decision.subject,
                                decision=decision.decision,
                                reason=decision.reason,
                            )
                        if denied:
                            await self._set_status(run_id, RunStatus.FAILED, "policy denied step")
                            break
                        if self._needs_manual_approval(current.approved_at, requires_approval):
                            await self._set_status(run_id, RunStatus.WAITING_APPROVAL)
                            await self.bus.publish_internal("run.waiting_approval", {"run_id": run_id, "step_id": step.id})
                            return

                        result = await self.codex_runner.execute_step(
                            run_id=run_id,
                            step=step,
                            workspace_path=workspace_path,
                            run_path=run_path,
                        )
                        self.workspace_snapshot_service.refresh(
                            workspace_path,
                            step=step,
                            result=result,
                        )
                        missing_recovery = await self._attempt_missing_python_file_recover(
                            run_id=run_id,
                            step=step,
                            step_index=idx,
                            workspace_path=workspace_path,
                            run_path=run_path,
                            result=result,
                        )
                        if missing_recovery is not None:
                            step, result = missing_recovery

                        if self._is_quality_threshold_soft_failure(
                            step=step,
                            result=result,
                            workspace_path=workspace_path,
                        ):
                            result.status = "completed"
                            result.exit_code = 0
                            result.is_infra_error = False
                            result.errors = []
                            result.summary = (
                                f"{result.summary}; quality threshold not reached in training step "
                                "and deferred to quality gate"
                            )

                        plan_contract_ok, plan_contract_reason = self._evaluate_plan_contract(
                            step=step,
                            workspace_path=workspace_path,
                            result=result,
                        )
                        if not plan_contract_ok and result.status == "completed":
                            result_summary = (
                                f"{result.summary}; plan contract check failed: {plan_contract_reason}"
                            )
                        else:
                            result_summary = result.summary
                        await self.db.insert_run_step(
                            run_id=run_id,
                            step_id=step.id,
                            step_title=step.title,
                            step_index=idx,
                            action=step.action,
                            command=result.command,
                            status="completed" if result.status == "completed" and plan_contract_ok else "failed",
                            stdout_text=f"summary: {result_summary}\n{result.stdout_text}".strip(),
                            stderr_text=result.stderr_text,
                            duration_ms=result.duration_ms,
                        )

                        post_step_run = await self.db.get_run(run_id)
                        if post_step_run and post_step_run.status == RunStatus.CANCELLED:
                            await self.finalize_cancelled(run_id)
                            return

                        if result.status == "completed":
                            if not plan_contract_ok:
                                if self.settings.execution_failure_replan_enabled:
                                    replan_attempt = await self.db.increment_stage_attempt(
                                        run_id,
                                        "PLAN_REVIEW_REPLAN",
                                    )
                                    if replan_attempt <= self.settings.max_execution_replans:
                                        await self._set_status(
                                            run_id,
                                            RunStatus.CONTEXT_READY,
                                            f"plan contract failed: {plan_contract_reason}",
                                        )
                                        await self.bus.publish_internal(
                                            "run.replan_scheduled",
                                            {
                                                "run_id": run_id,
                                                "stage": "PLAN_REVIEW_REPLAN",
                                                "attempt": replan_attempt,
                                                "max_attempts": self.settings.max_execution_replans,
                                                "step_id": step.id,
                                                "reason": plan_contract_reason,
                                            },
                                        )
                                        return

                                if self.settings.plan_review_manual_fallback:
                                    await self._set_status(
                                        run_id,
                                        RunStatus.WAITING_PLAN_REVIEW,
                                        f"plan contract failed: {plan_contract_reason}",
                                    )
                                    await self.bus.publish_internal(
                                        "run.waiting_plan_review",
                                        {
                                            "run_id": run_id,
                                            "step_id": step.id,
                                            "reason": plan_contract_reason,
                                        },
                                    )
                                    return
                                await self._set_status(
                                    run_id,
                                    RunStatus.FAILED,
                                    f"plan contract failed: {plan_contract_reason}",
                                )
                                break

                            await self.db.set_next_step_index(run_id, idx + 1)
                            switch_to_context_ready, reason = await self.ralph_service.maybe_handle_successful_step(
                                task=task,
                                step_id=step.id,
                                workspace_path=workspace_path,
                            )
                            if switch_to_context_ready:
                                await self._set_status(
                                    run_id,
                                    RunStatus.CONTEXT_READY,
                                    reason,
                                )
                                return
                            continue

                        missing_file = self._extract_missing_python_file_path(result.stderr_text)
                        if missing_file:
                            replan_attempt = await self.db.increment_stage_attempt(
                                run_id,
                                "MISSING_FILE_REPLAN",
                            )
                            if replan_attempt <= 1:
                                await self._set_status(
                                    run_id,
                                    RunStatus.CONTEXT_READY,
                                    f"python file missing: {missing_file} (auto recovery attempt {replan_attempt})",
                                )
                                await self.bus.publish_internal(
                                    "run.replan_scheduled",
                                    {
                                        "run_id": run_id,
                                        "stage": "MISSING_FILE_REPLAN",
                                        "attempt": replan_attempt,
                                        "max_attempts": 1,
                                        "step_id": step.id,
                                        "missing_file": missing_file,
                                    },
                                )
                                break

                        if result.is_infra_error:
                            attempts = await self.db.increment_stage_attempt(run_id, "EXECUTING")
                            if attempts <= 2:
                                await self.bus.publish_internal("run.retry_scheduled", {"run_id": run_id, "stage": "EXECUTING", "attempt": attempts})
                                return
                            await self._set_status(run_id, RunStatus.FAILED, f"infra failure after retries: {result.stderr_text}")
                            break

                        if self.settings.execution_failure_replan_enabled:
                            replan_attempt = await self.db.increment_stage_attempt(run_id, "EXECUTION_REPLAN")
                            if replan_attempt <= self.settings.max_execution_replans:
                                fail_reason = self._format_execution_failure(step.id, result)
                                await self._set_status(
                                    run_id,
                                    RunStatus.CONTEXT_READY,
                                    fail_reason,
                                )
                                await self.bus.publish_internal(
                                    "run.replan_scheduled",
                                    {
                                        "run_id": run_id,
                                        "attempt": replan_attempt,
                                        "max_attempts": self.settings.max_execution_replans,
                                        "step_id": step.id,
                                    },
                                )
                                return

                        await self._set_status(run_id, RunStatus.FAILED, self._format_execution_failure(step.id, result))
                        break
                    else:
                        await self._set_status(run_id, RunStatus.VERIFYING)
                    continue

                if run.status == RunStatus.VERIFYING:
                    verification = await self.verifier.run(workspace_path)
                    hyperparameter_context = await self._build_hyperparameter_context(run_id)
                    previous_verification_snapshot = self._latest_verification_snapshot(run.verification_json)
                    experiment_history = await self._list_experiment_history(run=run, task=task)
                    verification_payload: dict[str, Any] = {
                        "status": verification.status,
                        "commands": verification.commands,
                        "metrics": verification.metrics,
                        **hyperparameter_context,
                    }
                    quality_gate_skip_reason = self._quality_gate_skip_reason(run=run, verification=verification)
                    quality_gate = None
                    if quality_gate_skip_reason:
                        verification_payload["quality_gate"] = {
                            "status": "skipped",
                            "reason": quality_gate_skip_reason,
                        }
                    else:
                        quality_gate = await self.ralph_service.maybe_evaluate_quality_gate(
                            task=task,
                            workspace_path=workspace_path,
                            verification=verification,
                            previous_verification=previous_verification_snapshot,
                        )
                    if quality_gate is not None:
                        quality_ok, quality_reason = quality_gate
                        verification_payload["quality_gate"] = {
                            "status": "passed" if quality_ok else "failed",
                            "reason": quality_reason,
                        }
                        if not quality_ok:
                            verification_payload["improvement_strategy"] = (
                                self.improvement_strategy_service.build_for_quality_failure(
                                    run_id=run_id,
                                    task=task,
                                    workspace_path=workspace_path,
                                    verification=verification,
                                    previous_verification=previous_verification_snapshot,
                                    quality_reason=quality_reason,
                                    experiment_history=experiment_history,
                                )
                            )

                    normalized_verification = self._persist_verification_artifacts(
                        run_id=run_id,
                        workspace_path=workspace_path,
                        verification_payload=verification_payload,
                        previous_verification=run.verification_json if isinstance(run.verification_json, dict) else None,
                    )
                    verification_payload = normalized_verification
                    experiment_attempt_payload = self._experiment_attempt_payload_from_verification(
                        run=run,
                        task=task,
                        verification=verification_payload,
                    )
                    if experiment_attempt_payload is not None:
                        self._append_experiment_history_artifact(
                            workspace_path=workspace_path,
                            payload=experiment_attempt_payload,
                        )

                    refreshed_after_verification: RunRecord | None = None
                    try:
                        await self.db.set_verification(
                            run_id,
                            verification_payload,
                        )
                        refreshed_after_verification = await self.db.get_run(run_id)
                    except Exception as exc:
                        logger.exception(
                            "verification persistence to sqlite failed for run %s; file artifacts were preserved: %s",
                            run_id,
                            exc,
                        )
                        await self.bus.publish_internal(
                            "run.verification_persist_failed",
                            {
                                "run_id": run_id,
                                "reason": str(exc),
                            },
                        )
                    if refreshed_after_verification is not None:
                        await self._record_experiment_attempt(
                            run=refreshed_after_verification,
                            task=task,
                            workspace_path=workspace_path,
                        )

                    if quality_gate is not None:
                        quality_ok, quality_reason = quality_gate
                        if not quality_ok:
                            max_replans = self._quality_replan_limit_for_task(task)
                            replan_attempt = await self.db.increment_stage_attempt(run_id, "QUALITY_REPLAN")
                            if replan_attempt <= max_replans:
                                # Quality-gate replans must execute a fresh plan from step 0.
                                # Otherwise we can get stuck re-verifying stale artifacts only.
                                await self.db.set_next_step_index(run_id, 0)
                                await self._set_status(
                                    run_id,
                                    RunStatus.CONTEXT_READY,
                                    f"quality gate failed: {quality_reason}",
                                )
                                await self.bus.publish_internal(
                                    "run.replan_scheduled",
                                    {
                                        "run_id": run_id,
                                        "stage": "QUALITY_REPLAN",
                                        "attempt": replan_attempt,
                                        "max_attempts": max_replans,
                                        "reason": quality_reason,
                                    },
                                )
                                return
                            await self._set_status(
                                run_id,
                                RunStatus.FAILED,
                                f"quality gate failed after {replan_attempt} attempt(s): {quality_reason}",
                            )
                            return

                    await self._set_status(run_id, RunStatus.PACKAGING)
                    continue

                if run.status == RunStatus.PACKAGING:
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
                    await self._publish_result(
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
                    await self._set_status(run_id, final_status)
                    await self.db.release_workspace_lock(run.workspace_id, run_id)
                    return
            except RalphBacklogError as exc:
                await self._set_status(run_id, RunStatus.FAILED, f"ralph backlog error: {exc}")
            except PlannerError as exc:
                await self._set_status(run_id, RunStatus.FAILED, f"planner error: {exc}")
            except Exception as exc:
                await self._set_status(run_id, RunStatus.FAILED, f"unexpected error: {exc}")

    async def _publish_result(
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

    def _build_plan_input(
        self,
        task: dict[str, Any],
        run: RunRecord,
        workspace_id: str,
        contexts: list[RetrievedContext],
        workspace_snapshot_summary: str | None,
        experiment_history_summary: str | None,
        last_failed_step: dict[str, Any] | None,
        previous_verification: dict[str, Any] | None = None,
    ) -> PlanInput:
        return PlanInput(
            goal=task["goal"],
            constraints=json.loads(task["constraints_json"]),
            contexts=contexts,
            workspace_id=workspace_id,
            workspace_snapshot_summary=workspace_snapshot_summary,
            experiment_history_summary=experiment_history_summary,
            previous_error=run.error_message,
            last_failed_step=last_failed_step,
            previous_verification=previous_verification,
        )

    def _latest_verification_snapshot(self, verification_json: dict[str, Any] | None) -> dict[str, Any] | None:
        if not verification_json:
            return None
        snapshot = dict(verification_json)
        history_raw = verification_json.get("history")
        compact_history: list[dict[str, Any]] = []
        if isinstance(history_raw, list):
            for entry in history_raw[-8:]:
                if not isinstance(entry, dict):
                    continue
                compact_history.append(
                    {
                        "attempt": entry.get("attempt"),
                        "status": entry.get("status"),
                        "metrics": entry.get("metrics", {}),
                        "latest_hyperparameters": entry.get("latest_hyperparameters", {}),
                        "hyperparameter_attempts": (entry.get("hyperparameter_attempts") or [])[-4:],
                    }
                )
        if compact_history:
            snapshot["attempt_history"] = compact_history
        snapshot.pop("history", None)
        return snapshot

    async def _list_experiment_history(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
    ) -> list[dict[str, Any]]:
        goal_signature = run.goal_signature or self._build_task_signature_from_record(task)
        if not goal_signature:
            return []
        return await self.db.list_experiment_attempts(
            workspace_id=run.workspace_id,
            goal_signature=goal_signature,
            limit=self.settings.experiment_history_context_limit,
        )

    def _format_experiment_history_summary(self, attempts: list[dict[str, Any]]) -> str:
        if not attempts:
            return "none"
        lines: list[str] = []
        for item in attempts[-8:]:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id") or "n/a")
            attempt = item.get("attempt")
            quality_status = str(item.get("quality_status") or "n/a")
            quality_reason = str(item.get("quality_reason") or "").strip()
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            hyperparameters = item.get("hyperparameters") if isinstance(item.get("hyperparameters"), dict) else {}
            skill_paths = item.get("skill_paths") if isinstance(item.get("skill_paths"), list) else []
            chosen = None
            strategy = item.get("strategy") if isinstance(item.get("strategy"), dict) else {}
            if strategy:
                chosen = (
                    strategy.get("chosen_intervention", {}).get("id")
                    if isinstance(strategy.get("chosen_intervention"), dict)
                    else strategy.get("chosen_intervention_id")
                )
            metric_preview = ", ".join(
                f"{key}={metrics[key]}"
                for key in list(metrics.keys())[:3]
            ) or "no metrics"
            hp_preview = ", ".join(f"{key}={value}" for key, value in list(hyperparameters.items())[:4]) or "no hyperparameters"
            skill_preview = ", ".join(str(item) for item in skill_paths[:3]) or "no skills"
            line = (
                f"- run={run_id} attempt={attempt} quality={quality_status} "
                f"chosen={chosen or 'n/a'} metrics=[{metric_preview}] "
                f"hyperparameters=[{hp_preview}] skills=[{skill_preview}]"
            )
            if quality_reason:
                line += f" reason={quality_reason}"
            lines.append(line)
        return "\n".join(lines) if lines else "none"

    def _selected_skill_paths_from_verification(self, verification: dict[str, Any] | None) -> list[str]:
        if not isinstance(verification, dict):
            return []
        strategy = verification.get("improvement_strategy")
        if not isinstance(strategy, dict):
            return []
        chosen = strategy.get("chosen_intervention")
        if not isinstance(chosen, dict):
            return []
        values = chosen.get("skill_paths")
        if not isinstance(values, list):
            return []
        skill_paths: list[str] = []
        for item in values:
            value = str(item).strip()
            if value and value not in skill_paths:
                skill_paths.append(value)
        return skill_paths[:3]

    def _attach_selected_skills_to_plan(
        self,
        *,
        plan: PlannerPlan,
        previous_verification: dict[str, Any] | None,
        experiment_history: list[dict[str, Any]] | None = None,
    ) -> PlannerPlan:
        selected_skill_paths = self._selected_skill_paths_from_verification(previous_verification)
        if not selected_skill_paths:
            selected_skill_paths = self._selected_skill_paths_from_experiment_history(experiment_history or [])
        if not selected_skill_paths:
            return plan
        attached = False
        for step in plan.steps:
            if step.action != "codex":
                continue
            if step.skill_paths:
                attached = True
                continue
            step.skill_paths = list(selected_skill_paths)
            attached = True
        return plan if attached else plan

    def _selected_skill_paths_from_experiment_history(self, attempts: list[dict[str, Any]]) -> list[str]:
        for item in reversed(attempts):
            if not isinstance(item, dict):
                continue
            strategy = item.get("strategy")
            if isinstance(strategy, dict):
                chosen = strategy.get("chosen_intervention")
                if isinstance(chosen, dict):
                    values = chosen.get("skill_paths")
                    if isinstance(values, list):
                        paths = [str(value).strip() for value in values if str(value).strip()]
                        if paths:
                            return paths[:3]
            values = item.get("skill_paths")
            if isinstance(values, list):
                paths = [str(value).strip() for value in values if str(value).strip()]
                if paths:
                    return paths[:3]
        return []

    def _persist_verification_artifacts(
        self,
        *,
        run_id: str,
        workspace_path: Path,
        verification_payload: dict[str, Any],
        previous_verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = normalize_verification_payload(verification_payload, previous_verification)
        run_path = self.settings.runs_root / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        latest_path = run_path / "verification.latest.json"
        attempt = normalized.get("attempt")
        attempt_path = (
            run_path / f"verification.attempt_{int(attempt)}.json"
            if isinstance(attempt, int)
            else None
        )
        try:
            latest_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2), encoding="utf-8")
            if attempt_path is not None:
                attempt_path.write_text(json.dumps(normalized, ensure_ascii=True, indent=2), encoding="utf-8")
        except OSError:
            logger.warning("failed to persist verification artifact for run %s", run_id)
        strategy = normalized.get("improvement_strategy")
        if isinstance(strategy, dict):
            strategy_copy = dict(strategy)
            strategy_copy.setdefault("run_id", run_id)
            strategy_path = run_path / "improvement_strategy.latest.json"
            try:
                strategy_path.write_text(json.dumps(strategy_copy, ensure_ascii=True, indent=2), encoding="utf-8")
            except OSError:
                logger.warning("failed to persist improvement strategy artifact for run %s", run_id)
        return normalized

    def _experiment_attempt_payload_from_verification(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any] | None:
        goal_signature = run.goal_signature or self._build_task_signature_from_record(task)
        if not goal_signature:
            return None
        attempt = verification.get("latest_attempt") or verification.get("attempt")
        try:
            run_attempt = int(attempt)
        except (TypeError, ValueError):
            return None
        metrics = verification.get("metrics") if isinstance(verification.get("metrics"), dict) else {}
        latest_hyperparameters = (
            verification.get("latest_hyperparameters")
            if isinstance(verification.get("latest_hyperparameters"), dict)
            else {}
        )
        quality_gate = verification.get("quality_gate") if isinstance(verification.get("quality_gate"), dict) else {}
        improvement_strategy = (
            verification.get("improvement_strategy")
            if isinstance(verification.get("improvement_strategy"), dict)
            else None
        )
        selected_skill_paths = self._selected_skill_paths_from_verification(verification)
        return {
            "ts": datetime.now().astimezone().isoformat(),
            "workspace_id": run.workspace_id,
            "goal_signature": goal_signature,
            "run_id": run.run_id,
            "task_id": run.task_id,
            "attempt": run_attempt,
            "verification_status": str(verification.get("status") or ""),
            "quality_status": str(quality_gate.get("status") or ""),
            "quality_reason": str(quality_gate.get("reason") or ""),
            "metrics": metrics,
            "hyperparameters": latest_hyperparameters,
            "skill_paths": selected_skill_paths,
            "chosen_intervention_id": (
                improvement_strategy.get("chosen_intervention_id")
                if isinstance(improvement_strategy, dict)
                else None
            ),
            "strategy": improvement_strategy,
        }

    async def _record_experiment_attempt(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
        workspace_path: Path,
    ) -> None:
        verification = run.verification_json if isinstance(run.verification_json, dict) else None
        if not verification:
            return
        payload = self._experiment_attempt_payload_from_verification(
            run=run,
            task=task,
            verification=verification,
        )
        if payload is None:
            return
        await self.db.record_experiment_attempt(
            workspace_id=run.workspace_id,
            goal_signature=str(payload["goal_signature"]),
            run_id=run.run_id,
            task_id=run.task_id,
            run_attempt=int(payload["attempt"]),
            verification_status=str(payload["verification_status"]),
            quality_status=str(payload["quality_status"]),
            quality_reason=str(payload["quality_reason"]),
            metrics=dict(payload["metrics"]),
            hyperparameters=dict(payload["hyperparameters"]),
            strategy=payload.get("strategy"),
            skill_paths=list(payload["skill_paths"]),
        )

    def _append_experiment_history_artifact(self, *, workspace_path: Path, payload: dict[str, Any]) -> None:
        history_path = workspace_path / "knowledge" / "experiments" / "experiment_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            logger.warning("failed to append experiment history artifact for workspace %s", workspace_path)

    def _quality_gate_skip_reason(self, run: RunRecord, verification: VerificationResult) -> str | None:
        metrics = verification.metrics or {}
        if metrics.get("planning_only_report_detected") is True:
            return "quality gate skipped: planning-only artifacts are not evaluated against target metrics"

        plan = run.plan_json or {}
        steps_raw = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps_raw, list) or not steps_raw:
            return None

        if any(self._plan_step_runs_training(step) for step in steps_raw if isinstance(step, dict)):
            return None

        text_parts: list[str] = [str(plan.get("summary", ""))]
        for step in steps_raw:
            if not isinstance(step, dict):
                continue
            text_parts.extend(
                [
                    str(step.get("title", "")),
                    str(step.get("instruction", "")),
                    str(step.get("codex_prompt", "")),
                    str(step.get("stop_condition", "")),
                ]
            )
        normalized = " ".join(" ".join(text_parts).lower().split())
        planning_markers = (
            "planning-only",
            "planning only",
            "planning artifact",
            "no training is allowed",
            "no model execution",
            "training is deferred",
        )
        if any(marker in normalized for marker in planning_markers):
            return "quality gate skipped: current plan is planning-only"
        return None

    def _plan_step_runs_training(self, step: dict[str, Any]) -> bool:
        action = str(step.get("action", "")).strip().lower()
        step_intent = str(step.get("step_intent", "")).strip().lower()
        operation = str(step.get("operation", "")).strip().lower()
        if step_intent == StepIntent.run_training.value or operation == StepIntent.run_training.value:
            return True
        if action == "shell":
            command = str(step.get("command", "")).lower()
            commands = " ".join(str(item) for item in (step.get("commands") or [] if isinstance(step.get("commands"), list) else []))
            combined = f"{command} {commands}".lower()
            if any(token in combined for token in ("python ", "python3 ", "torchrun", "accelerate launch", "--epochs")):
                return True
        return False

    async def _build_hyperparameter_context(self, run_id: str) -> dict[str, Any]:
        try:
            steps = await self.db.list_run_steps(run_id)
        except Exception:
            return {}

        attempts: list[dict[str, Any]] = []
        last_signature: str | None = None
        for row in steps:
            command = str(row.get("command") or "").strip()
            if not command:
                continue
            if self._is_llm_or_agent_command(command):
                continue
            hyperparameters = self._extract_hyperparameters_from_command(command)
            if not hyperparameters:
                continue
            signature = json.dumps(hyperparameters, ensure_ascii=True, sort_keys=True)
            if signature == last_signature:
                continue
            last_signature = signature
            attempts.append(
                {
                    "step_id": str(row.get("step_id") or ""),
                    "step_index": int(row.get("step_index") or 0),
                    "status": str(row.get("status") or ""),
                    "command": command,
                    "hyperparameters": hyperparameters,
                }
            )

        if not attempts:
            return {}
        return {
            "latest_hyperparameters": attempts[-1]["hyperparameters"],
            "hyperparameter_attempts": attempts[-10:],
        }

    @staticmethod
    def _extract_hyperparameters_from_command(command: str) -> dict[str, Any]:
        if not command:
            return {}
        if ProcessRunTickUseCase._is_llm_or_agent_command(command):
            return {}
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        aliases = {
            "epochs": "epochs",
            "epoch": "epochs",
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "learning-rate": "learning_rate",
            "batch_size": "batch_size",
            "batch-size": "batch_size",
            "bs": "batch_size",
            "test_batch_size": "test_batch_size",
            "test-batch-size": "test_batch_size",
            "optimizer": "optimizer",
            "optim": "optimizer",
            "weight_decay": "weight_decay",
            "weight-decay": "weight_decay",
            "wd": "weight_decay",
            "momentum": "momentum",
            "dropout": "dropout",
            "model": "model",
            "model_name": "model",
            "seed": "seed",
            "workers": "workers",
            "num_workers": "workers",
            "num-workers": "workers",
            "img_size": "img_size",
            "image_size": "img_size",
            "image-size": "img_size",
        }
        params: dict[str, Any] = {}

        def normalize_key(raw: str) -> str | None:
            key = raw.strip().lstrip("-").replace(".", "_").replace(" ", "_").lower()
            return aliases.get(key)

        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--"):
                flag = token[2:]
                value: str | None = None
                if "=" in flag:
                    key_raw, value = flag.split("=", 1)
                else:
                    key_raw = flag
                    if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                        value = tokens[idx + 1]
                        idx += 1
                    else:
                        value = "true"
                normalized = normalize_key(key_raw)
                if normalized:
                    parsed = ProcessRunTickUseCase._coerce_hyperparameter_value(value)
                    if normalized == "model" and ProcessRunTickUseCase._looks_like_llm_model_name(parsed):
                        idx += 1
                        continue
                    params[normalized] = parsed
            elif "=" in token and not token.startswith("-"):
                key_raw, value = token.split("=", 1)
                normalized = normalize_key(key_raw)
                if normalized:
                    parsed = ProcessRunTickUseCase._coerce_hyperparameter_value(value)
                    if normalized == "model" and ProcessRunTickUseCase._looks_like_llm_model_name(parsed):
                        idx += 1
                        continue
                    params[normalized] = parsed
            idx += 1
        return params

    @staticmethod
    def _is_llm_or_agent_command(command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if not tokens:
            return False
        first = Path(tokens[0]).name.lower()
        if first in {"codex", "claude", "chatgpt"}:
            return True
        if first.startswith("codex-"):
            return True
        joined = " ".join(tokens[:4]).lower()
        if "codex exec" in joined:
            return True
        return False

    @staticmethod
    def _looks_like_llm_model_name(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        raw = value.strip().lower()
        if not raw:
            return False
        if raw.startswith(("gpt-", "o1", "o3", "o4", "codex", "claude", "gemini")):
            return True
        if raw.endswith("-codex") or raw.endswith("-spark"):
            return True
        return False

    @staticmethod
    def _coerce_hyperparameter_value(value: str | None) -> Any:
        if value is None:
            return None
        raw = str(value).strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"[-+]?\d+", raw):
            try:
                return int(raw)
            except ValueError:
                return raw
        if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", raw) or re.fullmatch(
            r"[-+]?\d+(?:[eE][-+]?\d+)", raw
        ):
            try:
                return float(raw)
            except ValueError:
                return raw
        return raw

    def _coerce_run_contexts(self, context_json: list[dict[str, Any]] | None) -> list[RetrievedContext]:
        contexts_raw = context_json or []
        validated: list[RetrievedContext] = []
        for item in contexts_raw:
            if not isinstance(item, dict):
                continue
            validated.append(
                RetrievedContext.model_validate(
                    {
                        "snippet": item.get("snippet", ""),
                        "document_path": item.get("document_path", ""),
                        "page_number": int(item.get("page_number", 1)),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                )
            )
        return validated

    def _build_task_goal_signature(self, payload_dict: dict[str, Any]) -> str:
        normalized = {
            "goal": str(payload_dict.get("goal", "")).strip(),
            "constraints": sorted(str(item) for item in payload_dict.get("constraints", [])),
            "execution_mode": str(payload_dict.get("execution_mode", "plan_execute")),
            "pdf_scope": sorted(str(item) for item in payload_dict.get("pdf_scope", [])),
        }
        serialized = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_task_signature_from_record(self, task: dict[str, Any]) -> str | None:
        payload_data = self._extract_task_payload(task)
        try:
            return self._build_task_goal_signature(payload_data)
        except Exception:
            return None

    def _extract_task_payload(self, task: dict[str, Any]) -> dict[str, Any]:
        def _safe_list(value: str) -> list[Any]:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []

        constraints: list[str] = []
        for raw in _safe_list(task.get("constraints_json", "[]")):
            value = str(raw).strip()
            if value:
                constraints.append(value)
        pdf_scope: list[str] = []
        for raw in _safe_list(task.get("pdf_scope_json", "[]")):
            value = str(raw).strip()
            if value:
                pdf_scope.append(value)
        execution_mode = "plan_execute"
        try:
            payload = json.loads(task.get("payload_json", "{}") or "{}")
            if isinstance(payload, dict):
                inner = payload.get("payload")
                if isinstance(inner, dict):
                    payload = inner
            mode = None
            if isinstance(payload, dict):
                mode = payload.get("execution_mode")
            if isinstance(mode, str) and mode.strip():
                execution_mode = mode.strip().lower()
        except json.JSONDecodeError:
            pass
        return {
            "goal": str(task.get("goal", "")).strip(),
            "constraints": constraints,
            "execution_mode": execution_mode,
            "pdf_scope": pdf_scope,
        }

    def _quality_replan_limit_for_task(self, task: dict[str, Any]) -> int:
        default_limit = int(self.settings.quality_replan_limit)
        try:
            constraints = json.loads(task.get("constraints_json", "[]"))
        except json.JSONDecodeError:
            return default_limit
        if not isinstance(constraints, list):
            return default_limit
        pattern = re.compile(r"(?i)^max_quality_retries\s*:\s*(\d+)\s*$")
        for item in constraints:
            value = str(item).strip()
            match = pattern.match(value)
            if not match:
                continue
            try:
                parsed = int(match.group(1))
            except ValueError:
                continue
            return max(0, min(parsed, 20))
        return default_limit

    def _format_execution_failure(self, step_id: str, result: Any) -> str:
        stderr = (result.stderr_text or "").strip()
        stderr_compact = " ".join(stderr.split())
        if len(stderr_compact) > 600:
            stderr_compact = f"{stderr_compact[:600]}..."
        return f"execution failed at step '{step_id}': {stderr_compact or 'unknown error'}"

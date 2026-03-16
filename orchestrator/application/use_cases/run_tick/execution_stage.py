from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.application.use_cases.run_tick.execution_guards import ExecutionGuardService
from orchestrator.config import Settings
from orchestrator.execution.codex_runner import CodexRunner
from orchestrator.execution.policy import PolicyEngine
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import PlannerPlan, RunRecord, RunStatus

StageOutcome = Literal["unhandled", "continue", "return"]


class RunExecutionStage:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        bus: Any,
        policy_engine: PolicyEngine,
        codex_runner: CodexRunner,
        execution_guard_service: ExecutionGuardService,
        workspace_snapshot_service: WorkspaceSnapshotService,
        ralph_service: RalphScenarioService,
        set_status: Callable[[str, RunStatus, str | None, RunRecord | None], Awaitable[None]],
        schedule_replan: Callable[..., Awaitable[int]],
        finalize_cancelled: Callable[[str], Awaitable[None]],
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.policy_engine = policy_engine
        self.codex_runner = codex_runner
        self.execution_guard_service = execution_guard_service
        self.workspace_snapshot_service = workspace_snapshot_service
        self.ralph_service = ralph_service
        self.set_status = set_status
        self.schedule_replan = schedule_replan
        self.finalize_cancelled = finalize_cancelled

    async def handle_executing(
        self,
        *,
        run_id: str,
        run: RunRecord,
        task: dict[str, Any],
        workspace_path: Path,
        run_path: Path,
    ) -> StageOutcome:
        if run.status != RunStatus.EXECUTING:
            return "unhandled"
        if not run.plan_json:
            await self.set_status(run_id, RunStatus.FAILED, "missing plan", None)
            return "continue"

        plan = PlannerPlan.model_validate(run.plan_json)
        steps = plan.steps
        for idx in range(run.next_step_index, len(steps)):
            current = await self.db.get_run(run_id)
            if current is None:
                return "return"
            if current.status == RunStatus.CANCELLED:
                await self.finalize_cancelled(run_id)
                return "return"
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
                await self.set_status(run_id, RunStatus.FAILED, "policy denied step", None)
                return "continue"
            if self.execution_guard_service.needs_manual_approval(current.approved_at, requires_approval):
                await self.set_status(run_id, RunStatus.WAITING_APPROVAL, None, None)
                await self.bus.publish_internal("run.waiting_approval", {"run_id": run_id, "step_id": step.id})
                return "return"

            synthetic_guard = self.execution_guard_service.synthetic_real_dataset_smoke_guard_reason(
                task=task,
                workspace_path=workspace_path,
                step=step,
            )
            if synthetic_guard:
                await self.db.insert_run_step(
                    run_id=run_id,
                    step_id=step.id,
                    step_title=step.title,
                    step_index=idx,
                    action=step.action,
                    command=step.command,
                    status="failed",
                    stdout_text=f"summary: blocked before execution\n{synthetic_guard}",
                    stderr_text="",
                    duration_ms=0,
                )
                if self.settings.execution_failure_replan_enabled:
                    replan_attempt = await self.db.increment_stage_attempt(run_id, "EXECUTION_REPLAN")
                    if replan_attempt <= self.settings.max_execution_replans:
                        cycle = await self.schedule_replan(
                            run_id=run_id,
                            target_status=RunStatus.CONTEXT_READY,
                            error_message=synthetic_guard,
                        )
                        await self.bus.publish_internal(
                            "run.replan_scheduled",
                            {
                                "run_id": run_id,
                                "attempt": replan_attempt,
                                "max_attempts": self.settings.max_execution_replans,
                                "step_id": step.id,
                                "reason": synthetic_guard,
                                "execution_cycle": cycle,
                            },
                        )
                        return "return"
                await self.set_status(run_id, RunStatus.FAILED, synthetic_guard, None)
                return "continue"

            preflight_dependency_guard = self.execution_guard_service.preflight_dependency_block_reason(
                workspace_path=workspace_path,
                step=step,
            )
            if preflight_dependency_guard:
                await self.db.insert_run_step(
                    run_id=run_id,
                    step_id=step.id,
                    step_title=step.title,
                    step_index=idx,
                    action=step.action,
                    command=step.command,
                    status="failed",
                    stdout_text=f"summary: blocked before execution\n{preflight_dependency_guard}",
                    stderr_text="",
                    duration_ms=0,
                )
                await self.set_status(run_id, RunStatus.FAILED, preflight_dependency_guard, None)
                return "continue"

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
            missing_recovery = await self.execution_guard_service.attempt_missing_python_file_recover(
                run_id=run_id,
                step=step,
                step_index=idx,
                workspace_path=workspace_path,
                run_path=run_path,
                result=result,
            )
            if missing_recovery is not None:
                step, result = missing_recovery

            if self.execution_guard_service.is_quality_threshold_soft_failure(
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

            plan_contract_ok, plan_contract_reason = self.execution_guard_service.evaluate_plan_contract(
                step=step,
                workspace_path=workspace_path,
                result=result,
            )
            if not plan_contract_ok and result.status == "completed":
                result_summary = f"{result.summary}; plan contract check failed: {plan_contract_reason}"
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
                return "return"

            dependency_issue = self.execution_guard_service.structured_dependency_failure_reason(
                workspace_path=workspace_path,
                step=step,
                result=result,
            )
            if dependency_issue:
                await self.set_status(run_id, RunStatus.FAILED, dependency_issue, None)
                return "continue"

            foreign_metrics_path_issue = self.execution_guard_service.foreign_run_metrics_path_reason(
                run_id=run_id,
                step=step,
                result=result,
            )
            if foreign_metrics_path_issue:
                await self.set_status(run_id, RunStatus.FAILED, foreign_metrics_path_issue, None)
                return "continue"

            if result.status == "completed":
                if not plan_contract_ok:
                    if self.settings.execution_failure_replan_enabled:
                        replan_attempt = await self.db.increment_stage_attempt(
                            run_id,
                            "PLAN_REVIEW_REPLAN",
                        )
                        if replan_attempt <= self.settings.max_execution_replans:
                            cycle = await self.schedule_replan(
                                run_id=run_id,
                                target_status=RunStatus.CONTEXT_READY,
                                error_message=f"plan contract failed: {plan_contract_reason}",
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
                                    "execution_cycle": cycle,
                                },
                            )
                            return "return"

                    if self.settings.plan_review_manual_fallback:
                        await self.set_status(
                            run_id,
                            RunStatus.WAITING_PLAN_REVIEW,
                            f"plan contract failed: {plan_contract_reason}",
                            None,
                        )
                        await self.bus.publish_internal(
                            "run.waiting_plan_review",
                            {
                                "run_id": run_id,
                                "step_id": step.id,
                                "reason": plan_contract_reason,
                            },
                        )
                        return "return"
                    await self.set_status(
                        run_id,
                        RunStatus.FAILED,
                        f"plan contract failed: {plan_contract_reason}",
                        None,
                    )
                    return "continue"

                await self.db.set_next_step_index(run_id, idx + 1)
                switch_to_context_ready, reason = await self.ralph_service.maybe_handle_successful_step(
                    task=task,
                    step_id=step.id,
                    workspace_path=workspace_path,
                )
                if switch_to_context_ready:
                    cycle = await self.schedule_replan(
                        run_id=run_id,
                        target_status=RunStatus.CONTEXT_READY,
                        error_message=reason,
                    )
                    await self.bus.publish_internal(
                        "run.story_cycle_advanced",
                        {
                            "run_id": run_id,
                            "step_id": step.id,
                            "reason": reason,
                            "execution_cycle": cycle,
                        },
                    )
                    return "return"
                continue

            missing_file = self.execution_guard_service.extract_missing_python_file_path(result.stderr_text)
            if missing_file:
                replan_attempt = await self.db.increment_stage_attempt(
                    run_id,
                    "MISSING_FILE_REPLAN",
                )
                if replan_attempt <= 1:
                    cycle = await self.schedule_replan(
                        run_id=run_id,
                        target_status=RunStatus.CONTEXT_READY,
                        error_message=f"python file missing: {missing_file} (auto recovery attempt {replan_attempt})",
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
                            "execution_cycle": cycle,
                        },
                    )
                    return "continue"

            if result.is_infra_error:
                attempts = await self.db.increment_stage_attempt(run_id, "EXECUTING")
                if attempts <= 2:
                    await self.bus.publish_internal(
                        "run.retry_scheduled",
                        {"run_id": run_id, "stage": "EXECUTING", "attempt": attempts},
                    )
                    return "return"
                await self.set_status(run_id, RunStatus.FAILED, f"infra failure after retries: {result.stderr_text}", None)
                return "continue"

            if self.settings.execution_failure_replan_enabled:
                replan_attempt = await self.db.increment_stage_attempt(run_id, "EXECUTION_REPLAN")
                if replan_attempt <= self.settings.max_execution_replans:
                    fail_reason = self.execution_guard_service.format_execution_failure(step.id, result)
                    cycle = await self.schedule_replan(
                        run_id=run_id,
                        target_status=RunStatus.CONTEXT_READY,
                        error_message=fail_reason,
                    )
                    await self.bus.publish_internal(
                        "run.replan_scheduled",
                        {
                            "run_id": run_id,
                            "attempt": replan_attempt,
                            "max_attempts": self.settings.max_execution_replans,
                            "step_id": step.id,
                            "execution_cycle": cycle,
                        },
                    )
                    return "return"

            await self.set_status(
                run_id,
                RunStatus.FAILED,
                self.execution_guard_service.format_execution_failure(step.id, result),
                None,
            )
            return "continue"

        await self.set_status(run_id, RunStatus.VERIFYING, None, None)
        return "continue"

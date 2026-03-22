from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.application.services.baseline_research_service import BaselineResearchService
from orchestrator.application.services.micro_training_policy_service import MicroTrainingPolicyService
from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.application.use_cases.run_tick.execution_guards import ExecutionGuardService
from orchestrator.application.use_cases.run_tick.planning_context import PlanningContextService
from orchestrator.config import Settings
from orchestrator.execution.policy import PolicyEngine
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import RunRecord, RunStatus
from orchestrator.planning.planner import Planner
from orchestrator.persistence.schemas import PlannerPlan

StageOutcome = Literal["unhandled", "continue", "return"]


class RunPlanningStage:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        bus: Any,
        planner: Planner,
        policy_engine: PolicyEngine,
        ralph_service: RalphScenarioService,
        baseline_research_service: BaselineResearchService,
        micro_training_policy_service: MicroTrainingPolicyService,
        planning_context_service: PlanningContextService,
        execution_guard_service: ExecutionGuardService,
        workspace_snapshot_service: WorkspaceSnapshotService,
        set_status: Callable[[str, RunStatus, str | None, RunRecord | None], Awaitable[None]],
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.planner = planner
        self.policy_engine = policy_engine
        self.ralph_service = ralph_service
        self.baseline_research_service = baseline_research_service
        self.micro_training_policy_service = micro_training_policy_service
        self.planning_context_service = planning_context_service
        self.execution_guard_service = execution_guard_service
        self.workspace_snapshot_service = workspace_snapshot_service
        self.set_status = set_status

    async def handle_context_ready(
        self,
        *,
        run_id: str,
        run: RunRecord,
        task: dict[str, Any],
        workspace_path: Path,
    ) -> StageOutcome:
        if run.status != RunStatus.CONTEXT_READY:
            return "unhandled"

        contexts = self.planning_context_service.coerce_run_contexts(run.context_json)
        last_failed_step = await self.db.get_latest_failed_step(run_id)
        previous_verification = self.planning_context_service.latest_verification_snapshot(run.verification_json)
        workspace_snapshot_summary = self.workspace_snapshot_service.get_prompt_summary(workspace_path)
        experiment_history = await self.planning_context_service.list_experiment_history(run=run, task=task)
        experiment_history_summary = self.planning_context_service.format_experiment_history_summary(
            experiment_history
        )
        experiment_memory_summary = self.planning_context_service.build_experiment_memory_summary(experiment_history)
        baseline_research_summary = await self.baseline_research_service.build_summary(
            task=task,
            workspace_path=workspace_path,
            experiment_history=experiment_history,
            previous_verification=previous_verification,
        )
        micro_training_policy = self.micro_training_policy_service.build_from_previous_verification(
            task=task,
            workspace_path=workspace_path,
            previous_verification=previous_verification,
        )

        missing_quality_reason = self.ralph_service.ralph_quality_requirement_missing_reason(
            task=task,
            workspace_path=workspace_path,
        )
        if missing_quality_reason:
            await self.set_status(
                run_id,
                RunStatus.WAITING_PLAN_REVIEW,
                missing_quality_reason,
                None,
            )
            await self.bus.publish_internal(
                "run.waiting_plan_review",
                {
                    "run_id": run_id,
                    "reason": missing_quality_reason,
                },
            )
            return "return"

        plan = await self.ralph_service.maybe_build_plan(
            task=task,
            run=run,
            contexts=contexts,
            workspace_path=workspace_path,
            last_failed_step=last_failed_step,
            previous_verification=previous_verification,
            experiment_history_summary=experiment_history_summary,
            experiment_memory_summary=experiment_memory_summary,
            baseline_research_summary=baseline_research_summary,
        )
        if plan is None:
            plan_input = self.planning_context_service.build_plan_input(
                task=task,
                run=run,
                workspace_id=run.workspace_id,
                contexts=contexts,
                workspace_snapshot_summary=workspace_snapshot_summary,
                experiment_history_summary=experiment_history_summary,
                experiment_memory_summary=experiment_memory_summary,
                baseline_research_summary=baseline_research_summary,
                last_failed_step=last_failed_step,
                previous_verification=previous_verification,
                micro_training_policy=micro_training_policy,
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

        plan = self.planning_context_service.attach_selected_skills_to_plan(
            plan=plan,
            previous_verification=previous_verification,
            experiment_history=experiment_history,
        )
        quality_plan_issue = self.execution_guard_service.plan_quality_execution_issue(
            task=task,
            workspace_path=workspace_path,
            plan=plan,
        )
        if quality_plan_issue:
            await self.db.set_plan(run_id, plan.model_dump(mode="json"))
            if self.settings.plan_review_manual_fallback:
                await self.set_status(
                    run_id,
                    RunStatus.WAITING_PLAN_REVIEW,
                    quality_plan_issue,
                    None,
                )
                await self.bus.publish_internal(
                    "run.waiting_plan_review",
                    {
                        "run_id": run_id,
                        "reason": quality_plan_issue,
                    },
                )
                return "return"
            await self.set_status(run_id, RunStatus.FAILED, quality_plan_issue, None)
            return "return"

        if len(plan.steps) > self.settings.max_run_steps:
            await self.set_status(
                run_id,
                RunStatus.FAILED,
                f"plan exceeds max_run_steps={self.settings.max_run_steps}",
                None,
            )
            return "continue"

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
            await self.set_status(run_id, RunStatus.FAILED, "policy denied plan", None)
            return "continue"

        if self.execution_guard_service.needs_manual_approval(run.approved_at, requires_approval):
            await self.set_status(run_id, RunStatus.WAITING_APPROVAL, None, None)
            await self.bus.publish_internal("run.waiting_approval", {"run_id": run_id})
            return "return"

        await self.set_status(run_id, RunStatus.PLAN_READY, None, None)
        await self.bus.publish_internal("run.plan_ready", {"run_id": run_id})
        return "continue"

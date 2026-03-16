from __future__ import annotations

import asyncio
import logging
from typing import Any

from orchestrator.application.use_cases.run_tick import (
    ExecutionGuardService,
    HyperparameterService,
    PlanningContextService,
    RunCoordinatorSupportService,
    RunContextStage,
    RunExecutionStage,
    RunPackagingStage,
    RunPlanningStage,
    RunVerificationStage,
    StepioRecoveryService,
    VerificationFlowService,
)
from orchestrator.application.services.plan_contract_service import PlanContractService
from orchestrator.application.services.metric_interpretation_service import CodexMetricInterpreter
from orchestrator.application.services.quality_gate_service import QualityGateService
from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.application.services.improvement_strategy_service import ImprovementStrategyService
from orchestrator.application.services.micro_training_policy_service import MicroTrainingPolicyService
from orchestrator.application.services.recovery_service import MissingFileRecoveryService
from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.runtime.bus import InMemoryEventBus, RedisEventBus
from orchestrator.execution.codex_runner import CodexRunner
from orchestrator.config import Settings
from orchestrator.persistence.db import Database
from orchestrator.planning.planner import Planner, PlannerError
from orchestrator.execution.policy import PolicyEngine
from orchestrator.planning.ralph import RalphBacklogError, RalphBacklogService
from orchestrator.persistence.schemas import RunStatus
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
        self.db = db
        self._run_mutex: dict[str, Any] = {}
        plan_contract_service = PlanContractService(
            plan_review_enabled=settings.plan_review_enabled,
            strictness=settings.contract_strictness,
        )
        recovery_service = MissingFileRecoveryService()
        hyperparameter_service = HyperparameterService(db)
        self.planning_context_service = PlanningContextService(db, settings.experiment_history_context_limit)
        self.coordinator_support_service = RunCoordinatorSupportService(
            settings=settings,
            db=db,
            bus=bus,
            codex_runner=codex_runner,
            artifact_publisher=artifact_publisher,
            planning_context_service=self.planning_context_service,
        )
        verification_flow_service = VerificationFlowService(
            db=db,
            runs_root=settings.runs_root,
            quality_replan_limit=settings.quality_replan_limit,
            planning_context_service=self.planning_context_service,
        )
        quality_gate_service = QualityGateService(
            ralph_backlog=ralph_backlog,
            settings=settings,
            metric_interpreter=CodexMetricInterpreter(settings),
        )
        improvement_strategy_service = ImprovementStrategyService(quality_gate_service=quality_gate_service)
        micro_training_policy_service = MicroTrainingPolicyService(quality_gate_service=quality_gate_service)
        workspace_snapshot_service = WorkspaceSnapshotService()
        ralph_service = RalphScenarioService(
            settings=settings,
            backlog=ralph_backlog,
            planner=planner,
            db=db,
            bus=bus,
            quality_gate_service=quality_gate_service,
        )
        execution_guard_service = ExecutionGuardService(
            db=db,
            auto_approve_in_pilot=settings.auto_approve_in_pilot,
            plan_contract_service=plan_contract_service,
            recovery_service=recovery_service,
            codex_runner=codex_runner,
            ralph_service=ralph_service,
        )
        self.ralph_service = ralph_service
        self.execution_guard_service = execution_guard_service
        self.stepio_recovery_service = StepioRecoveryService(
            db=db,
            bus=bus,
            set_status=self.coordinator_support_service.set_status,
        )
        self.context_stage = RunContextStage(
            settings=settings,
            db=db,
            bus=bus,
            workspace_snapshot_service=workspace_snapshot_service,
            set_status=self.coordinator_support_service.set_status,
        )
        self.planning_stage = RunPlanningStage(
            settings=settings,
            db=db,
            bus=bus,
            planner=planner,
            policy_engine=policy_engine,
            ralph_service=ralph_service,
            micro_training_policy_service=micro_training_policy_service,
            planning_context_service=self.planning_context_service,
            execution_guard_service=execution_guard_service,
            workspace_snapshot_service=workspace_snapshot_service,
            set_status=self.coordinator_support_service.set_status,
        )
        self.execution_stage = RunExecutionStage(
            settings=settings,
            db=db,
            bus=bus,
            policy_engine=policy_engine,
            codex_runner=codex_runner,
            execution_guard_service=execution_guard_service,
            workspace_snapshot_service=workspace_snapshot_service,
            ralph_service=ralph_service,
            set_status=self.coordinator_support_service.set_status,
            schedule_replan=self.coordinator_support_service.schedule_replan,
            finalize_cancelled=self.coordinator_support_service.finalize_cancelled,
        )
        self.verification_stage = RunVerificationStage(
            settings=settings,
            db=db,
            bus=bus,
            verifier=verifier,
            hyperparameter_service=hyperparameter_service,
            planning_context_service=self.planning_context_service,
            verification_flow_service=verification_flow_service,
            execution_guard_service=execution_guard_service,
            improvement_strategy_service=improvement_strategy_service,
            micro_training_policy_service=micro_training_policy_service,
            ralph_service=ralph_service,
            set_status=self.coordinator_support_service.set_status,
            schedule_replan=self.coordinator_support_service.schedule_replan,
        )
        self.verification_flow_service = verification_flow_service
        self.packaging_stage = RunPackagingStage(
            db=db,
            artifact_publisher=artifact_publisher,
            ralph_service=ralph_service,
            set_status=self.coordinator_support_service.set_status,
            publish_result=self.coordinator_support_service.publish_result,
        )

    async def submit_task_event(self, payload: dict[str, Any]) -> str | None:
        return await self.coordinator_support_service.submit_task_event(payload)

    async def handle_control_event(self, payload: dict[str, Any]) -> bool:
        return await self.coordinator_support_service.handle_control_event(payload)

    async def finalize_cancelled(self, run_id: str) -> None:
        await self.coordinator_support_service.finalize_cancelled(run_id)

    async def process_run(self, run_id: str) -> None:
        mutex = self._run_mutex.setdefault(run_id, asyncio.Lock())
        async with mutex:
            await self._process_run_impl(run_id)

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
                await self.coordinator_support_service.set_status(run_id, RunStatus.FAILED, "task not found")
                continue

            task_signature = self.planning_context_service.build_task_signature_from_record(task)
            if task_signature is not None and run.goal_signature != task_signature:
                await self.db.reset_run_for_goal_change(run_id, goal_signature=task_signature)
                await self.coordinator_support_service.bus.publish_internal(
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

            waiting_outcome = await self.context_stage.handle_waiting_status(run_id=run_id, run=run)
            if waiting_outcome == "continue":
                continue
            if waiting_outcome == "return":
                return

            workspace_path = self.coordinator_support_service.workspace_dir(run.workspace_id)
            run_path = self.coordinator_support_service.artifact_publisher.run_dir(run_id)
            latest_stepio = await self.stepio_recovery_service.reconcile_stepio_artifacts(run=run, run_path=run_path)
            if latest_stepio:
                refreshed_run = await self.db.get_run(run_id)
                if refreshed_run is not None:
                    run = refreshed_run
                    if await self.stepio_recovery_service.sync_run_progress_from_stepio(
                        run=run,
                        latest_by_step_id=latest_stepio,
                    ):
                        continue

            try:
                received_outcome = await self.context_stage.handle_received(
                    run_id=run_id,
                    run=run,
                    workspace_path=workspace_path,
                )
                if received_outcome == "continue":
                    continue
                if received_outcome == "return":
                    return

                planning_outcome = await self.planning_stage.handle_context_ready(
                    run_id=run_id,
                    run=run,
                    task=task,
                    workspace_path=workspace_path,
                )
                if planning_outcome == "continue":
                    continue
                if planning_outcome == "return":
                    return

                plan_ready_outcome = await self.context_stage.handle_plan_ready(run_id=run_id, run=run)
                if plan_ready_outcome == "continue":
                    continue
                if plan_ready_outcome == "return":
                    return

                execution_outcome = await self.execution_stage.handle_executing(
                    run_id=run_id,
                    run=run,
                    task=task,
                    workspace_path=workspace_path,
                    run_path=run_path,
                )
                if execution_outcome == "continue":
                    continue
                if execution_outcome == "return":
                    return

                verification_outcome = await self.verification_stage.handle_verifying(
                    run_id=run_id,
                    run=run,
                    task=task,
                    workspace_path=workspace_path,
                )
                if verification_outcome == "continue":
                    continue
                if verification_outcome == "return":
                    return

                packaging_outcome = await self.packaging_stage.handle_packaging(
                    run_id=run_id,
                    run=run,
                    task=task,
                    workspace_path=workspace_path,
                )
                if packaging_outcome == "continue":
                    continue
                if packaging_outcome == "return":
                    return
            except RalphBacklogError as exc:
                await self.coordinator_support_service.set_status(run_id, RunStatus.FAILED, f"ralph backlog error: {exc}")
            except PlannerError as exc:
                await self.coordinator_support_service.set_status(run_id, RunStatus.FAILED, f"planner error: {exc}")
            except Exception as exc:
                await self.coordinator_support_service.set_status(run_id, RunStatus.FAILED, f"unexpected error: {exc}")

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from orchestrator.application.services.improvement_strategy_service import ImprovementStrategyService
from orchestrator.application.services.proxy_metric_service import ProxyMetricService
from orchestrator.application.services.proxy_continuation_service import ProxyContinuationService
from orchestrator.application.services.budget_tier_service import BudgetTierService
from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.application.use_cases.run_tick.execution_guards import ExecutionGuardService
from orchestrator.application.use_cases.run_tick.hyperparameters import HyperparameterService
from orchestrator.application.use_cases.run_tick.planning_context import PlanningContextService
from orchestrator.application.use_cases.run_tick.verification_flow import VerificationFlowService
from orchestrator.config import Settings
from orchestrator.execution.verifier import Verifier
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import RunRecord, RunStatus

logger = logging.getLogger(__name__)

StageOutcome = Literal["unhandled", "continue", "return"]


class RunVerificationStage:
    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        bus: Any,
        verifier: Verifier,
        hyperparameter_service: HyperparameterService,
        planning_context_service: PlanningContextService,
        verification_flow_service: VerificationFlowService,
        execution_guard_service: ExecutionGuardService,
        improvement_strategy_service: ImprovementStrategyService,
        proxy_metric_service: ProxyMetricService,
        budget_tier_service: BudgetTierService,
        proxy_continuation_service: ProxyContinuationService,
        ralph_service: RalphScenarioService,
        set_status: Callable[[str, RunStatus, str | None, RunRecord | None], Awaitable[None]],
        schedule_replan: Callable[..., Awaitable[int]],
    ) -> None:
        self.settings = settings
        self.db = db
        self.bus = bus
        self.verifier = verifier
        self.hyperparameter_service = hyperparameter_service
        self.planning_context_service = planning_context_service
        self.verification_flow_service = verification_flow_service
        self.execution_guard_service = execution_guard_service
        self.improvement_strategy_service = improvement_strategy_service
        self.proxy_metric_service = proxy_metric_service
        self.budget_tier_service = budget_tier_service
        self.proxy_continuation_service = proxy_continuation_service
        self.ralph_service = ralph_service
        self.set_status = set_status
        self.schedule_replan = schedule_replan

    async def handle_verifying(
        self,
        *,
        run_id: str,
        run: RunRecord,
        task: dict[str, Any],
        workspace_path: Path,
    ) -> StageOutcome:
        if run.status != RunStatus.VERIFYING:
            return "unhandled"

        verification = await self.verifier.run(
            workspace_path,
            task=task,
            story_id=self.ralph_service.extract_story_id(task),
        )
        hyperparameter_context = await self.hyperparameter_service.build_context(run_id)
        previous_verification_snapshot = self.planning_context_service.latest_verification_snapshot(run.verification_json)
        experiment_history = await self.planning_context_service.list_experiment_history(run=run, task=task)
        verification_payload: dict[str, Any] = {
            "status": verification.status,
            "commands": verification.commands,
            "metrics": verification.metrics,
            **hyperparameter_context,
        }
        if isinstance(verification.details, dict) and verification.details:
            verification_payload.update(verification.details)
        proxy_metric = self.proxy_metric_service.build_proxy_metric(
            task=task,
            metrics=verification.metrics,
            previous_verification=previous_verification_snapshot,
            experiment_history=experiment_history,
        )
        if proxy_metric is not None:
            verification.details["proxy_metric"] = proxy_metric
            verification_payload["proxy_metric"] = proxy_metric
        budget_tier = self.budget_tier_service.current_training_tier(run=run, task=task)
        if budget_tier is not None:
            verification.details["budget_tier"] = budget_tier
            verification_payload["budget_tier"] = budget_tier
        quality_gate_skip_reason = self.execution_guard_service.quality_gate_skip_reason(
            run=run,
            verification=verification,
        )
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
            self.ralph_service.quality_gate_service.attach_search_metric_progress(
                task=task,
                workspace_path=workspace_path,
                verification=verification,
                previous_verification=previous_verification_snapshot,
                experiment_history=experiment_history,
                story_id=self.ralph_service.extract_story_id(task),
            )
            verification_payload["quality_gate"] = {
                "status": "passed" if quality_ok else "failed",
                "reason": quality_reason,
            }
            if not quality_ok:
                verification_payload["improvement_strategy"] = self.improvement_strategy_service.build_for_quality_failure(
                    run_id=run_id,
                    task=task,
                    workspace_path=workspace_path,
                    verification=verification,
                    previous_verification=previous_verification_snapshot,
                    quality_reason=quality_reason,
                    experiment_history=experiment_history,
                )

        terminal_skip_reason = self.verification_flow_service.planning_only_terminal_skip_reason(verification_payload)
        if terminal_skip_reason:
            verification_payload["quality_gate"] = {
                "status": "skipped",
                "reason": terminal_skip_reason,
            }
            quality_gate = None
        proxy_decision = None
        if quality_gate is not None:
            quality_ok, _ = quality_gate
            if not quality_ok:
                proxy_decision = self.proxy_continuation_service.decide(
                    run=run,
                    task=task,
                    proxy_metric=proxy_metric,
                )
                if proxy_decision is not None:
                    verification.details["proxy_decision"] = proxy_decision
                    verification_payload["proxy_decision"] = proxy_decision

        normalized_verification = self.verification_flow_service.persist_verification_artifacts(
            run_id=run_id,
            workspace_path=workspace_path,
            verification_payload=verification_payload,
            previous_verification=run.verification_json if isinstance(run.verification_json, dict) else None,
        )
        verification_payload = normalized_verification
        experiment_attempt_payload = self.verification_flow_service.experiment_attempt_payload_from_verification(
            run=run,
            task=task,
            verification=verification_payload,
        )
        if experiment_attempt_payload is not None:
            self.verification_flow_service.append_experiment_history_artifact(
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
            await self.verification_flow_service.record_experiment_attempt(
                run=refreshed_after_verification,
                task=task,
            )

        if quality_gate is not None:
            quality_ok, quality_reason = quality_gate
            if not quality_ok:
                if isinstance(proxy_decision, dict):
                    replan_attempt = await self.db.increment_stage_attempt(run_id, "PROXY_REPLAN")
                    max_replans = self.verification_flow_service.quality_replan_limit_for_task(task)
                    target_tier = str(proxy_decision.get("to_tier") or "micro").strip() or "micro"
                    if replan_attempt <= max_replans:
                        await self.db.set_run_budget_tier(run_id, target_tier)
                        cycle = await self.schedule_replan(
                            run_id=run_id,
                            target_status=RunStatus.CONTEXT_READY,
                            error_message=f"proxy {proxy_decision.get('decision')}: {proxy_decision.get('reason')}",
                        )
                        await self.bus.publish_internal(
                            f"run.proxy_{proxy_decision.get('decision')}",
                            {
                                "run_id": run_id,
                                "stage": "PROXY_REPLAN",
                                "attempt": replan_attempt,
                                "max_attempts": max_replans,
                                "from_tier": proxy_decision.get("from_tier"),
                                "to_tier": target_tier,
                                "proxy_gain": proxy_decision.get("proxy_gain"),
                                "threshold": proxy_decision.get("threshold"),
                                "reason": proxy_decision.get("reason"),
                                "execution_cycle": cycle,
                            },
                        )
                        return "return"
                    await self.set_status(
                        run_id,
                        RunStatus.FAILED,
                        (
                            f"proxy continuation exhausted after {replan_attempt} attempt(s): "
                            f"{proxy_decision.get('reason')}; last result: {quality_reason}"
                        ),
                        None,
                    )
                    return "return"
                quality_replan_issue = self.execution_guard_service.quality_replan_block_reason(
                    run=run,
                    task=task,
                    workspace_path=workspace_path,
                )
                if quality_replan_issue:
                    await self.set_status(
                        run_id,
                        RunStatus.FAILED,
                        f"quality gate failed without executable training plan: {quality_replan_issue}; last result: {quality_reason}",
                        None,
                    )
                    return "return"
                max_replans = self.verification_flow_service.quality_replan_limit_for_task(task)
                replan_attempt = await self.db.increment_stage_attempt(run_id, "QUALITY_REPLAN")
                if replan_attempt <= max_replans:
                    cycle = await self.schedule_replan(
                        run_id=run_id,
                        target_status=RunStatus.CONTEXT_READY,
                        error_message=f"quality gate failed: {quality_reason}",
                    )
                    await self.bus.publish_internal(
                        "run.replan_scheduled",
                        {
                            "run_id": run_id,
                            "stage": "QUALITY_REPLAN",
                            "attempt": replan_attempt,
                            "max_attempts": max_replans,
                            "reason": quality_reason,
                            "execution_cycle": cycle,
                        },
                    )
                    return "return"
                await self.set_status(
                    run_id,
                    RunStatus.FAILED,
                    f"quality gate failed after {replan_attempt} attempt(s): {quality_reason}",
                    None,
                )
                return "return"

        await self.set_status(run_id, RunStatus.PACKAGING, None, None)
        return "continue"

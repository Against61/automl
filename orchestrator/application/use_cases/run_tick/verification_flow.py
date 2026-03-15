from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.application.use_cases.run_tick.planning_context import PlanningContextService
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import RunRecord
from orchestrator.persistence.verification_payloads import normalize_verification_payload

logger = logging.getLogger(__name__)


class VerificationFlowService:
    def __init__(
        self,
        db: Database,
        runs_root: Path,
        quality_replan_limit: int,
        planning_context_service: PlanningContextService,
    ) -> None:
        self.db = db
        self.runs_root = runs_root
        self.quality_replan_limit = quality_replan_limit
        self.planning_context_service = planning_context_service

    def persist_verification_artifacts(
        self,
        *,
        run_id: str,
        workspace_path: Path,
        verification_payload: dict[str, Any],
        previous_verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del workspace_path
        normalized = normalize_verification_payload(verification_payload, previous_verification)
        run_path = self.runs_root / run_id
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

    def experiment_attempt_payload_from_verification(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
        verification: dict[str, Any],
    ) -> dict[str, Any] | None:
        goal_signature = run.goal_signature or self.planning_context_service.build_task_signature_from_record(task)
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
        final_metric = verification.get("final_metric") if isinstance(verification.get("final_metric"), dict) else {}
        budget_tier = verification.get("budget_tier") if isinstance(verification.get("budget_tier"), dict) else {}
        proxy_metric = verification.get("proxy_metric") if isinstance(verification.get("proxy_metric"), dict) else {}
        search_metric = verification.get("search_metric") if isinstance(verification.get("search_metric"), dict) else {}
        quality_gate = verification.get("quality_gate") if isinstance(verification.get("quality_gate"), dict) else {}
        improvement_strategy = self.planning_context_service.compact_strategy_summary(
            verification.get("improvement_strategy")
        )
        selected_skill_paths = self.planning_context_service.selected_skill_paths_from_verification(verification)
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
            "final_metric": final_metric,
            "budget_tier": budget_tier,
            "proxy_metric": proxy_metric,
            "search_metric": search_metric,
            "hyperparameters": latest_hyperparameters,
            "skill_paths": selected_skill_paths,
            "chosen_intervention_id": (
                improvement_strategy.get("chosen_intervention_id")
                if isinstance(improvement_strategy, dict)
                else None
            ),
            "strategy": improvement_strategy,
        }

    async def record_experiment_attempt(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
    ) -> None:
        verification = run.verification_json if isinstance(run.verification_json, dict) else None
        if not verification:
            return
        payload = self.experiment_attempt_payload_from_verification(
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
            final_metric=dict(payload.get("final_metric") or {}),
            budget_tier=dict(payload.get("budget_tier") or {}),
            proxy_metric=dict(payload.get("proxy_metric") or {}),
            search_metric=dict(payload.get("search_metric") or {}),
            hyperparameters=dict(payload["hyperparameters"]),
            strategy=payload.get("strategy"),
            skill_paths=list(payload["skill_paths"]),
        )

    @staticmethod
    def append_experiment_history_artifact(*, workspace_path: Path, payload: dict[str, Any]) -> None:
        history_path = workspace_path / "knowledge" / "experiments" / "experiment_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except OSError:
            logger.warning("failed to append experiment history artifact for workspace %s", workspace_path)

    @staticmethod
    def planning_only_terminal_skip_reason(verification_payload: dict[str, Any]) -> str | None:
        metrics = verification_payload.get("metrics") if isinstance(verification_payload.get("metrics"), dict) else {}
        if metrics.get("planning_only_report_detected") is True:
            return "quality gate skipped: planning-only artifacts are not evaluated against target metrics"
        if metrics.get("training_deferred") is True and metrics.get("dataset_parse_ok") is True:
            return "quality gate skipped: planning-only artifacts are not evaluated against target metrics"
        quality_gate = verification_payload.get("quality_gate")
        if isinstance(quality_gate, dict) and str(quality_gate.get("status") or "").strip().lower() == "skipped":
            reason = str(quality_gate.get("reason") or "").strip()
            return reason or "quality gate skipped: planning-only artifacts are not evaluated against target metrics"
        return None

    def quality_replan_limit_for_task(self, task: dict[str, Any]) -> int:
        default_limit = int(self.quality_replan_limit)
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

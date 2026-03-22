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
        quality_gate = verification.get("quality_gate") if isinstance(verification.get("quality_gate"), dict) else {}
        improvement_strategy = self.planning_context_service.compact_strategy_summary(
            verification.get("improvement_strategy")
        )
        selected_skill_paths = self.planning_context_service.selected_skill_paths_from_verification(verification)
        recipe_snapshot = self.recipe_snapshot_from_verification(verification)
        recipe_diff = self.recipe_diff_from_verification(
            current_verification=verification,
            previous_verification=run.verification_json if isinstance(run.verification_json, dict) else None,
        )
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
            "recipe_snapshot": recipe_snapshot,
            "recipe_diff": recipe_diff,
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
            hyperparameters=dict(payload["hyperparameters"]),
            recipe_snapshot=payload.get("recipe_snapshot") if isinstance(payload.get("recipe_snapshot"), dict) else None,
            recipe_diff=payload.get("recipe_diff") if isinstance(payload.get("recipe_diff"), dict) else None,
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

    @staticmethod
    def recipe_snapshot_from_verification(verification: dict[str, Any]) -> dict[str, Any]:
        latest_hyperparameters = (
            verification.get("latest_hyperparameters")
            if isinstance(verification.get("latest_hyperparameters"), dict)
            else {}
        )
        snapshot: dict[str, Any] = {}
        for key in ("epochs", "learning_rate", "batch_size", "optimizer", "weight_decay", "dropout", "model"):
            if key in latest_hyperparameters:
                snapshot[key] = latest_hyperparameters.get(key)
        strategy = verification.get("improvement_strategy")
        if isinstance(strategy, dict):
            chosen_id = (
                strategy.get("chosen_intervention_id")
                or (
                    strategy.get("chosen_intervention", {}).get("id")
                    if isinstance(strategy.get("chosen_intervention"), dict)
                    else None
                )
            )
            if chosen_id:
                snapshot["intervention"] = chosen_id
            chosen = strategy.get("chosen_intervention")
            if isinstance(chosen, dict):
                skill_paths = chosen.get("skill_paths")
                if isinstance(skill_paths, list) and skill_paths:
                    snapshot["skill_paths"] = [str(item).strip() for item in skill_paths[:3] if str(item).strip()]
        micro_training_policy = verification.get("micro_training_policy")
        if isinstance(micro_training_policy, dict):
            phase = str(micro_training_policy.get("phase") or "").strip()
            if phase:
                snapshot["micro_training_phase"] = phase
            next_epochs = micro_training_policy.get("next_epochs")
            if next_epochs is not None:
                snapshot["next_epochs"] = next_epochs
        return snapshot

    def recipe_diff_from_verification(
        self,
        *,
        current_verification: dict[str, Any],
        previous_verification: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        current_snapshot = self.recipe_snapshot_from_verification(current_verification)
        previous_snapshot = self.recipe_snapshot_from_verification(previous_verification or {}) if previous_verification else {}
        changes: dict[str, dict[str, Any]] = {}
        for key in sorted(set(previous_snapshot.keys()) | set(current_snapshot.keys())):
            before = previous_snapshot.get(key)
            after = current_snapshot.get(key)
            if before == after:
                continue
            changes[key] = {"before": before, "after": after}
        if not changes:
            return None
        return {
            "changed_keys": list(changes.keys()),
            "changes": changes,
        }

from __future__ import annotations

from pathlib import Path
from typing import Any

from orchestrator.application.services.evaluation_contract_service import EvaluationContractService
from orchestrator.application.services.task_intent_service import TaskIntentService


class IntentValidationSupport:
    def __init__(
        self,
        task_intent_service: TaskIntentService | None = None,
        evaluation_contract_service: EvaluationContractService | None = None,
    ) -> None:
        self.task_intent_service = task_intent_service or TaskIntentService()
        self.evaluation_contract_service = evaluation_contract_service or EvaluationContractService()

    def build_intent_validation_details(
        self,
        *,
        metrics: dict[str, float | int | str | bool],
        task: dict[str, Any] | None,
        workspace_path: Path,
        story_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(task, dict):
            return {}
        contract = self.evaluation_contract_service.load_from_task(task)
        intent = self.task_intent_service.infer_from_task(
            task=task,
            workspace_path=workspace_path,
        )
        details: dict[str, Any] = {
            "task_intent": {
                "task_family": intent.task_family,
                "metric_family": intent.metric_family,
                "primary_metric_key": intent.primary_metric_key,
                "preferred_metrics": list(intent.preferred_metrics),
                "real_dataset_smoke_required": intent.requires_real_dataset_smoke,
                "evidence": list(intent.evidence),
            }
        }
        if contract is not None:
            details["evaluation_contract"] = self.evaluation_contract_service.serialize(contract)
        reported_primary = str(metrics.get("primary_metric_key") or "").strip().lower()
        reported_family = str(metrics.get("task_family") or "").strip().lower()
        expected_primary = contract.primary_metric_key if contract is not None else intent.primary_metric_key
        expected_family = contract.task_family if contract is not None else intent.task_family

        validation_status = "skipped"
        validation_reason = "metrics artifact does not declare primary_metric_key"
        drift_detected = False

        if reported_primary and expected_primary:
            if reported_primary != expected_primary.lower():
                validation_status = "failed"
                validation_reason = (
                    f"metrics artifact primary_metric_key='{reported_primary}' does not match "
                    f"{'evaluation contract' if contract is not None else 'inferred primary metric'} "
                    f"'{expected_primary}'"
                )
                drift_detected = True
            else:
                validation_status = "passed"
                validation_reason = (
                    "metrics artifact primary_metric_key matches evaluation contract"
                    if contract is not None
                    else "metrics artifact primary_metric_key matches inferred intent"
                )

        if not drift_detected and reported_family and expected_family != "generic":
            if reported_family != expected_family.lower():
                validation_status = "failed"
                validation_reason = (
                    f"metrics artifact task_family='{reported_family}' does not match "
                    f"{'evaluation contract' if contract is not None else 'inferred task family'} "
                    f"'{expected_family}'"
                )
                drift_detected = True
            elif validation_status == "skipped":
                validation_status = "passed"
                validation_reason = (
                    "metrics artifact task_family matches evaluation contract"
                    if contract is not None
                    else "metrics artifact task_family matches inferred intent"
                )

        metrics["metric_intent_drift_detected"] = drift_detected
        details["intent_validation"] = {
            "status": validation_status,
            "reason": validation_reason,
            "reported_primary_metric_key": reported_primary or None,
            "inferred_primary_metric_key": intent.primary_metric_key,
            "expected_primary_metric_key": expected_primary,
            "reported_task_family": reported_family or None,
            "inferred_task_family": intent.task_family,
            "expected_task_family": expected_family,
            "story_id": story_id,
        }
        return details

from __future__ import annotations

from typing import Any

from orchestrator.application.services.quality_gate_service import QualityGateService
from orchestrator.persistence.verification_payloads import compact_verification_history_entry


class MicroTrainingPolicyService:
    EPOCH_LADDER = (1, 2, 5)

    def __init__(self, quality_gate_service: QualityGateService, growth_epsilon: float = 1e-9) -> None:
        self.quality_gate_service = quality_gate_service
        self.growth_epsilon = growth_epsilon

    def build_from_previous_verification(
        self,
        *,
        task: dict[str, Any],
        workspace_path,
        previous_verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return self._build_policy_from_snapshot(
            task=task,
            workspace_path=workspace_path,
            snapshot=previous_verification,
        )

    def build_from_current_attempt(
        self,
        *,
        task: dict[str, Any],
        workspace_path,
        current_verification: dict[str, Any],
        previous_verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        if isinstance(previous_verification, dict):
            previous_history = previous_verification.get("history")
            if isinstance(previous_history, list):
                history.extend(entry for entry in previous_history if isinstance(entry, dict))
            compact_previous = compact_verification_history_entry(previous_verification)
            if compact_previous:
                history.append(compact_previous)
        merged_snapshot = dict(current_verification)
        if history:
            merged_snapshot["history"] = history
        return self._build_policy_from_snapshot(
            task=task,
            workspace_path=workspace_path,
            snapshot=merged_snapshot,
        )

    def _build_policy_from_snapshot(
        self,
        *,
        task: dict[str, Any],
        workspace_path,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        requirement = self.quality_gate_service.extract_requirement(
            task=task,
            workspace_path=workspace_path,
        ) or {}
        metric_key = str(requirement.get("metric_key") or "").strip()
        attempts = self._ordered_attempts(snapshot)
        if not attempts:
            return {
                "enabled": True,
                "ladder_epochs": list(self.EPOCH_LADDER),
                "phase": "baseline",
                "next_epochs": self.EPOCH_LADDER[0],
                "reason": "start with a 1-epoch baseline for the current training strategy",
                "metric_key": metric_key or None,
                "force_strategy_reset": False,
            }

        latest = attempts[-1]
        latest_epochs = self._extract_epochs(latest)
        latest_metrics = latest.get("metrics") if isinstance(latest.get("metrics"), dict) else {}
        latest_metric_value = (
            self.quality_gate_service.select_metric_value(latest_metrics, metric_key)
            if metric_key
            else None
        )
        quality_gate = latest.get("quality_gate") if isinstance(latest.get("quality_gate"), dict) else {}
        quality_status = str(quality_gate.get("status") or "").strip().lower()

        policy: dict[str, Any] = {
            "enabled": True,
            "ladder_epochs": list(self.EPOCH_LADDER),
            "metric_key": metric_key or None,
            "current_epochs": latest_epochs,
            "current_metric_value": latest_metric_value,
            "quality_status": quality_status or None,
            "force_strategy_reset": False,
        }

        if quality_status == "passed":
            policy.update(
                {
                    "phase": "target_met",
                    "next_epochs": latest_epochs,
                    "reason": "target metric already passed; no further micro-training escalation is needed",
                }
            )
            return policy

        if latest_epochs <= self.EPOCH_LADDER[0]:
            policy.update(
                {
                    "phase": "growth_check",
                    "next_epochs": self.EPOCH_LADDER[1],
                    "reason": "baseline metric captured at 1 epoch; repeat the same strategy for 2 epochs and compare growth",
                }
            )
            return policy

        if latest_epochs <= self.EPOCH_LADDER[1]:
            baseline_attempt = self._find_latest_attempt_with_epochs(
                attempts=attempts[:-1],
                expected_epochs=self.EPOCH_LADDER[0],
            )
            baseline_metric = self._metric_value_from_attempt(baseline_attempt, metric_key)
            metric_growth = None
            if latest_metric_value is not None and baseline_metric is not None:
                metric_growth = latest_metric_value - baseline_metric
            policy["baseline_metric_value"] = baseline_metric
            policy["metric_growth"] = metric_growth
            if metric_growth is not None and metric_growth > self.growth_epsilon:
                policy.update(
                    {
                        "phase": "expand",
                        "next_epochs": self.EPOCH_LADDER[2],
                        "reason": "metric improved from 1 to 2 epochs; promote the same strategy to a 5-epoch run",
                    }
                )
                return policy

            policy.update(
                {
                    "phase": "reset_strategy",
                    "next_epochs": self.EPOCH_LADDER[0],
                    "reason": "metric did not improve from 1 to 2 epochs; switch strategy and restart from a 1-epoch baseline",
                    "force_strategy_reset": True,
                }
            )
            return policy

        policy.update(
            {
                "phase": "reset_strategy",
                "next_epochs": self.EPOCH_LADDER[0],
                "reason": "5-epoch micro-training did not reach the target metric; switch strategy and restart from a 1-epoch baseline",
                "force_strategy_reset": True,
            }
        )
        return policy

    @staticmethod
    def _ordered_attempts(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(snapshot, dict):
            return []
        attempts: list[dict[str, Any]] = []
        history = snapshot.get("history")
        if not isinstance(history, list):
            history = snapshot.get("attempt_history")
        if isinstance(history, list):
            attempts.extend(entry for entry in history if isinstance(entry, dict))
        attempts.append(snapshot)
        return attempts

    def _metric_value_from_attempt(self, attempt: dict[str, Any] | None, metric_key: str) -> float | None:
        if not isinstance(attempt, dict) or not metric_key:
            return None
        metrics = attempt.get("metrics")
        if not isinstance(metrics, dict):
            return None
        return self.quality_gate_service.select_metric_value(metrics, metric_key)

    @staticmethod
    def _find_latest_attempt_with_epochs(
        *,
        attempts: list[dict[str, Any]],
        expected_epochs: int,
    ) -> dict[str, Any] | None:
        for attempt in reversed(attempts):
            if MicroTrainingPolicyService._extract_epochs(attempt) == expected_epochs:
                return attempt
        return None

    @staticmethod
    def _extract_epochs(snapshot: dict[str, Any] | None) -> int:
        if not isinstance(snapshot, dict):
            return 1
        latest_hyperparameters = (
            snapshot.get("latest_hyperparameters")
            if isinstance(snapshot.get("latest_hyperparameters"), dict)
            else {}
        )
        raw = latest_hyperparameters.get("epochs")
        parsed = MicroTrainingPolicyService._to_int(raw)
        if parsed is not None and parsed > 0:
            return parsed

        attempts = snapshot.get("hyperparameter_attempts")
        if isinstance(attempts, list):
            for item in reversed(attempts):
                if not isinstance(item, dict):
                    continue
                hyperparameters = item.get("hyperparameters")
                if not isinstance(hyperparameters, dict):
                    continue
                parsed = MicroTrainingPolicyService._to_int(hyperparameters.get("epochs"))
                if parsed is not None and parsed > 0:
                    return parsed

        metrics = snapshot.get("metrics")
        if isinstance(metrics, dict):
            parsed = MicroTrainingPolicyService._to_int(metrics.get("epochs_trained"))
            if parsed is not None and parsed > 0:
                return parsed
        return 1

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed

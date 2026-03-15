from __future__ import annotations

from typing import Any

from orchestrator.application.services.evaluation_contract_service import EvaluationContractService
from orchestrator.application.services.final_metric_service import FinalMetricService
from orchestrator.application.services.metric_utility_service import MetricUtilityService


class ProxyMetricService:
    def __init__(
        self,
        *,
        evaluation_contract_service: EvaluationContractService | None = None,
        final_metric_service: FinalMetricService | None = None,
        metric_utility_service: MetricUtilityService | None = None,
    ) -> None:
        self.evaluation_contract_service = evaluation_contract_service or EvaluationContractService()
        self.metric_utility_service = metric_utility_service or MetricUtilityService()
        self.final_metric_service = final_metric_service or FinalMetricService(self.metric_utility_service)

    def build_proxy_metric(
        self,
        *,
        task: dict[str, Any],
        metrics: dict[str, Any],
        previous_verification: dict[str, Any] | None = None,
        experiment_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        contract = self.evaluation_contract_service.load_from_task(task)
        if contract is None:
            return None
        resolved = self._resolve_proxy_value(contract=contract, metrics=metrics)
        if resolved is None:
            return None
        metric_key, value = resolved
        prior_values = self._history_proxy_values(
            experiment_history=experiment_history or [],
            previous_verification=previous_verification,
        )
        baseline_value = prior_values[0] if prior_values else value
        effective_train_seconds = self.metric_utility_service.coerce_float(metrics.get("effective_train_seconds"))
        if contract.proxy_direction == "min":
            best_value_so_far = min(prior_values) if prior_values else value
            proxy_gain = (baseline_value - value) / (abs(baseline_value) + 1e-8)
            delta_best = best_value_so_far - value
        else:
            best_value_so_far = max(prior_values) if prior_values else value
            proxy_gain = (value - baseline_value) / (abs(baseline_value) + 1e-8)
            delta_best = value - best_value_so_far
        return {
            "kind": contract.proxy_metric_kind,
            "metric_key": metric_key,
            "direction": contract.proxy_direction,
            "value": value,
            "baseline_value": baseline_value,
            "best_value_so_far": best_value_so_far,
            "delta_best": delta_best,
            "proxy_gain": proxy_gain,
            "effective_train_seconds": effective_train_seconds,
            "gain_per_budget": self.metric_utility_service.gain_per_budget(proxy_gain, effective_train_seconds),
            "micro_split_id": contract.micro_split_id,
            "macro_split_id": contract.macro_split_id,
            "history_count": len(prior_values),
        }

    def _resolve_proxy_value(
        self,
        *,
        contract,
        metrics: dict[str, Any],
    ) -> tuple[str, float] | None:
        if contract.proxy_metric_kind == "micro_loss":
            for key in (
                "micro_val_loss",
                "micro_eval_loss",
                "micro_loss",
                "val_loss",
                "eval_loss",
                "validation_loss",
                "loss",
                "test_loss",
                "train_loss",
            ):
                value = self.metric_utility_service.coerce_float(metrics.get(key))
                if value is not None:
                    return key, value
            return None

        value = self.final_metric_service.select_metric_value(metrics, contract.primary_metric_key)
        if value is None:
            return None
        return contract.primary_metric_key, value

    def _history_proxy_values(
        self,
        *,
        experiment_history: list[dict[str, Any]],
        previous_verification: dict[str, Any] | None,
    ) -> list[float]:
        values: list[float] = []
        for item in experiment_history:
            if not isinstance(item, dict):
                continue
            proxy_metric = item.get("proxy_metric") if isinstance(item.get("proxy_metric"), dict) else None
            if not isinstance(proxy_metric, dict):
                continue
            value = self.metric_utility_service.coerce_float(proxy_metric.get("value"))
            if value is not None:
                values.append(value)
        if values or not isinstance(previous_verification, dict):
            return values
        history_entries = previous_verification.get("attempt_history")
        if not isinstance(history_entries, list):
            return values
        for entry in history_entries:
            if not isinstance(entry, dict):
                continue
            proxy_metric = entry.get("proxy_metric") if isinstance(entry.get("proxy_metric"), dict) else None
            if not isinstance(proxy_metric, dict):
                continue
            value = self.metric_utility_service.coerce_float(proxy_metric.get("value"))
            if value is not None:
                values.append(value)
        return values

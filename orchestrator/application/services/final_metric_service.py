from __future__ import annotations

from typing import Any

from orchestrator.application.services.metric_utility_service import MetricUtilityService
from orchestrator.execution.metric_parsing import normalize_metric_key


class FinalMetricService:
    def __init__(self, metric_utility_service: MetricUtilityService | None = None) -> None:
        self.metric_utility_service = metric_utility_service or MetricUtilityService()

    def normalize_metric_key(self, raw: str) -> str:
        value = normalize_metric_key(str(raw))
        aliases = {
            "acc": "accuracy",
            "acc1": "accuracy",
            "test_acc": "test_accuracy",
            "train_acc": "train_accuracy",
            "val_acc": "val_accuracy",
            "val_accuracy": "val_accuracy",
            "eval_acc": "eval_accuracy",
            "evaluation_acc": "eval_accuracy",
            "held_out_accuracy": "eval_accuracy",
            "held_out_acc": "eval_accuracy",
            "held_out_eval_accuracy": "eval_accuracy",
            "held_out_test_accuracy": "test_accuracy",
            "out_evaluation_accuracy": "eval_accuracy",
            "test_loss": "test_loss",
            "train_loss": "train_loss",
            "loss": "loss",
            "m_iou": "miou",
            "mean_intersection_over_union": "mean_iou",
            "mean_intersection_union": "mean_iou",
            "jaccard": "iou",
            "jaccard_index": "iou",
            "jaccard_score": "iou",
        }
        return aliases.get(value, value)

    def resolve_metric(
        self,
        metrics: dict[str, Any],
        metric_key: str,
    ) -> tuple[float | None, dict[str, Any] | None]:
        raw_value, resolved_key, mode = self._pick_metric_value(metrics, metric_key)
        if raw_value is None:
            return None, None
        value = self.metric_utility_service.coerce_float(raw_value)
        if value is None:
            return None, None
        resolution = {
            "required_metric_key": self.normalize_metric_key(metric_key),
            "resolved_metric_key": resolved_key or self.normalize_metric_key(metric_key),
            "resolved_value": value,
            "mode": mode or "exact",
            "confidence": "high" if mode in {"exact", "preferred_alias"} else "medium",
            "reason": "",
        }
        return value, resolution

    def build_final_metric(
        self,
        *,
        metric_key: str,
        resolution: dict[str, Any] | None,
        metric_value: float,
        operator: str,
        target_value: float,
        unit: str,
    ) -> dict[str, Any]:
        normalized_metric_key = self.normalize_metric_key(metric_key)
        scale = self.metric_utility_service.scale_for_unit(unit)
        target_utility = self.metric_utility_service.target_to_utility(target_value, unit)
        metric_utility = self.metric_utility_service.to_utility(metric_value, scale)
        return {
            "metric_key": normalized_metric_key,
            "resolved_metric_key": (resolution or {}).get("resolved_metric_key") or normalized_metric_key,
            "raw_value": metric_value,
            "utility": metric_utility,
            "scale": scale,
            "operator": operator,
            "target_raw": target_value,
            "target_utility": target_utility,
            "unit": unit,
        }

    def build_search_metric(
        self,
        *,
        metric_key: str,
        resolution: dict[str, Any] | None,
        final_metric: dict[str, Any],
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_metric_key = self.normalize_metric_key(metric_key)
        effective_train_seconds = self.metric_utility_service.coerce_float((metrics or {}).get("effective_train_seconds"))
        return {
            "metric_key": normalized_metric_key,
            "resolved_metric_key": (resolution or {}).get("resolved_metric_key") or normalized_metric_key,
            "scale": final_metric.get("scale"),
            "utility": final_metric.get("utility"),
            "target_utility": final_metric.get("target_utility"),
            "effective_train_seconds": effective_train_seconds,
        }

    def passes_requirement(
        self,
        *,
        final_metric: dict[str, Any],
    ) -> bool:
        metric_utility = self.metric_utility_service.coerce_float(final_metric.get("utility"))
        target_utility = self.metric_utility_service.coerce_float(final_metric.get("target_utility"))
        operator = str(final_metric.get("operator") or ">=").strip()
        if metric_utility is None or target_utility is None:
            return False
        if operator == ">=":
            return metric_utility >= target_utility
        if operator == ">":
            return metric_utility > target_utility
        if operator == "<=":
            return metric_utility <= target_utility
        if operator == "<":
            return metric_utility < target_utility
        if operator in {"==", "="}:
            return abs(metric_utility - target_utility) <= 1e-9
        if operator == "!=":
            return abs(metric_utility - target_utility) > 1e-9
        return False

    def select_metric_value(self, metrics: dict[str, Any], metric_key: str) -> float | None:
        value, _ = self.resolve_metric(metrics, metric_key)
        return value

    def select_metric_utility(self, metrics: dict[str, Any], metric_key: str, unit: str | None) -> float | None:
        value = self.select_metric_value(metrics, metric_key)
        if value is None:
            return None
        scale = self.metric_utility_service.scale_for_unit(unit)
        return self.metric_utility_service.to_utility(value, scale)

    def _pick_metric_value(self, metrics: dict[str, Any], metric_key: str) -> tuple[Any | None, str | None, str | None]:
        normalized = {self.normalize_metric_key(key): metrics[key] for key in metrics}
        target = self.normalize_metric_key(metric_key)

        if target in normalized:
            return normalized[target], target, "exact"

        for candidate in self._preferred_metric_aliases(target):
            if candidate in normalized:
                mode = "alias"
                if candidate != target and candidate.startswith(("val_", "validation_", "test_", "eval_")):
                    mode = "preferred_alias"
                return normalized[candidate], candidate, mode

        partial_matches: list[tuple[str, Any]] = []
        for key, value in normalized.items():
            if target in key or key in target:
                partial_matches.append((key, value))

        if partial_matches:
            partial_matches.sort(key=lambda item: self._metric_match_rank(item[0], target))
            return partial_matches[0][1], partial_matches[0][0], "partial"

        return None, None, None

    def _preferred_metric_aliases(self, target: str) -> list[str]:
        if not target:
            return []
        if target == "iou":
            return [
                "eval_mean_iou",
                "test_mean_iou",
                "val_mean_iou",
                "validation_mean_iou",
                "mean_iou",
                "miou",
                "eval_iou",
                "test_iou",
                "val_iou",
                "jaccard",
                "iou",
                "train_mean_iou",
                "train_iou",
            ]
        if target.startswith(("train_", "val_", "validation_", "test_", "eval_")):
            return []
        return [
            f"val_{target}",
            f"validation_{target}",
            f"test_{target}",
            f"eval_{target}",
            target,
            f"train_{target}",
        ]

    def _metric_match_rank(self, candidate: str, target: str) -> tuple[int, int, str]:
        if candidate == target:
            return (0, 0, candidate)
        preferred = self._preferred_metric_aliases(target)
        if candidate in preferred:
            return (1, preferred.index(candidate), candidate)
        if candidate.startswith(("val_", "validation_", "test_", "eval_")):
            return (2, len(candidate), candidate)
        if candidate.startswith("train_"):
            return (4, len(candidate), candidate)
        return (3, len(candidate), candidate)

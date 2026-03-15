from __future__ import annotations

import json
import re
from typing import Any

from orchestrator.application.services.metric_interpretation_service import CodexMetricInterpreter
from orchestrator.application.services.task_intent_service import TaskIntent, TaskIntentService
from orchestrator.config import Settings
from orchestrator.execution.metric_parsing import normalize_metric_key
from orchestrator.execution.verifier import VerificationResult
from orchestrator.planning.ralph import RalphBacklogService


class QualityGateService:
    def __init__(
        self,
        ralph_backlog: RalphBacklogService,
        settings: Settings | None = None,
        metric_interpreter: CodexMetricInterpreter | None = None,
        task_intent_service: TaskIntentService | None = None,
    ):
        self.ralph_backlog = ralph_backlog
        self.settings = settings
        self.metric_interpreter = metric_interpreter
        self.task_intent_service = task_intent_service or TaskIntentService()

    async def evaluate(
        self,
        task: dict[str, Any],
        workspace_path,
        verification: VerificationResult,
        story_id: str | None = None,
    ) -> tuple[bool, str]:
        requirement = self.extract_requirement(
            task=task,
            workspace_path=workspace_path,
            story_id=story_id,
        )
        if self._has_split_leakage(verification.metrics or {}):
            return False, "quality gate failed: evaluation split reuses training examples"
        blocked_reason = self._non_production_report_reason(verification.metrics or {})
        if blocked_reason:
            return False, blocked_reason
        if not requirement:
            return True, "no explicit ralph quality requirement"

        metrics = dict(verification.metrics or {})
        metric_key = requirement.get("metric_key", "")
        target = requirement.get("target")
        operator = (requirement.get("operator") or ">=").strip()
        unit = requirement.get("unit") or "ratio"
        if not metric_key or target is None:
            return False, "quality requirement is malformed: missing metric or target"

        metric_value, resolution = self._resolve_metric(metrics, metric_key)
        if metric_value is None:
            interpreted = await self._interpret_missing_metric(
                required_metric_key=metric_key,
                metrics=metrics,
                workspace_path=workspace_path,
            )
            if interpreted is not None:
                metric_value = interpreted.resolved_value
                metrics.setdefault(metric_key, interpreted.resolved_value)
                metrics.setdefault(f"{metric_key}_resolved_from", interpreted.resolved_metric_key)
                metrics.setdefault(f"{metric_key}_resolution_confidence", interpreted.confidence)
                if interpreted.reason:
                    metrics.setdefault(f"{metric_key}_resolution_reason", interpreted.reason)
                verification.metrics = metrics
                resolution = {
                    "required_metric_key": metric_key,
                    "resolved_metric_key": interpreted.resolved_metric_key,
                    "resolved_value": interpreted.resolved_value,
                    "mode": "semantic_fallback",
                    "confidence": interpreted.confidence,
                    "reason": interpreted.reason,
                }
        if metric_value is None:
            return False, f"metric '{metric_key}' not found in verification output"
        if resolution:
            verification.details["metric_resolution"] = resolution

        target_value = self._to_float(target)
        if target_value is None:
            return False, f"unable to parse required target '{target}'"
        if unit in {"%", "percent", "pct"} and target_value is not None:
            target_value = target_value / 100.0
            metric_value = self._normalize_metric_value_for_unit(metric_value, unit)

        check = False
        if operator == ">=":
            check = metric_value >= target_value
        elif operator == ">":
            check = metric_value > target_value
        elif operator == "<=":
            check = metric_value <= target_value
        elif operator == "<":
            check = metric_value < target_value
        elif operator in {"==", "="}:
            check = abs(metric_value - target_value) <= 1e-9
        elif operator == "!=":
            check = abs(metric_value - target_value) > 1e-9

        if check:
            return True, "quality gate passed"
        return (
            False,
            (
                f"quality gate failed: {metric_key}={metric_value:.6g} {operator} "
                f"{target_value:.6g} required (unit={unit})"
            ),
        )

    def extract_requirement(
        self,
        task: dict[str, Any],
        workspace_path,
        story_id: str | None = None,
    ) -> dict[str, Any] | None:
        constraints = [str(item).strip() for item in json.loads(task["constraints_json"])]
        requirement = self._parse_quality_requirement_from_constraints(constraints)
        if requirement:
            return self._normalize_requirement_for_task(
                requirement=requirement,
                task=task,
                workspace_path=workspace_path,
                story_id=story_id,
            )

        story = None
        if story_id:
            prd = self.ralph_backlog.load_prd(workspace_path)
            story = self.ralph_backlog.pick_by_id(prd, story_id=story_id)
        if story and getattr(story, "acceptance_criteria", None):
            for item in story.acceptance_criteria:
                if isinstance(item, str):
                    parsed = self._parse_quality_requirement_from_text(item)
                    if parsed:
                        return self._normalize_requirement_for_task(
                            requirement=parsed,
                            task=task,
                            workspace_path=workspace_path,
                            story_id=story_id,
                        )

        raw_goal = str(task.get("goal", ""))
        parsed = self._parse_quality_requirement_from_text(raw_goal)
        if parsed:
            return self._normalize_requirement_for_task(
                requirement=parsed,
                task=task,
                workspace_path=workspace_path,
                story_id=story_id,
            )
        return None

    def _parse_quality_requirement_from_constraints(self, constraints: list[str]) -> dict[str, Any] | None:
        quality_re = re.compile(
            r"(?i)^(?:required[_\s-]*metric|ralph_required_metric)\s*[:=]\s*(.+)$"
        )
        for item in constraints:
            value = str(item).strip()
            if not value:
                continue
            match = quality_re.match(value)
            if not match:
                continue
            parsed = self._parse_quality_requirement_from_text(match.group(1))
            if parsed:
                return parsed
        return None

    def _parse_quality_requirement_from_text(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        pattern = re.compile(
            r"(?P<metric>[a-zA-Z][a-zA-Z0-9_\s\-\./%]*?)\s*(?P<op>>=|<=|!=|==|>|<|=)\s*(?P<value>[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)\s*(?P<unit>%|percent|pct|ratio)?",
            re.IGNORECASE,
        )

        match = pattern.search(text)
        if not match:
            return None
        metric = match.group("metric").strip()
        op = match.group("op")
        value = match.group("value")
        unit = (match.group("unit") or "").lower().strip()
        if not metric or value is None:
            return None
        return {
            "metric_key": self._normalize_metric_key(metric),
            "operator": op,
            "target": value,
            "unit": unit or "ratio",
        }

    def _normalize_metric_key(self, raw: str) -> str:
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

    def infer_task_intent(
        self,
        *,
        task: dict[str, Any],
        workspace_path,
        story_id: str | None = None,
    ) -> TaskIntent:
        extra_texts: list[str] = []
        if story_id:
            try:
                prd = self.ralph_backlog.load_prd(workspace_path)
                story = self.ralph_backlog.pick_by_id(prd, story_id=story_id)
            except Exception:
                story = None
            if story is not None:
                extra_texts.extend(
                    [
                        story.story_id,
                        story.title,
                        story.description,
                        *list(story.acceptance_criteria or []),
                    ]
                )
        return self.task_intent_service.infer_from_task(
            task=task,
            workspace_path=workspace_path,
            extra_texts=extra_texts,
        )

    def _normalize_requirement_for_task(
        self,
        *,
        requirement: dict[str, Any],
        task: dict[str, Any],
        workspace_path,
        story_id: str | None = None,
    ) -> dict[str, Any]:
        normalized = dict(requirement)
        intent = self.infer_task_intent(
            task=task,
            workspace_path=workspace_path,
            story_id=story_id,
        )
        metric_key = self._normalize_metric_key(normalized.get("metric_key", ""))
        if not intent.primary_metric_key:
            return normalized

        generic_keys = {"metric", "score", "quality"}
        if not metric_key or metric_key in generic_keys:
            normalized["metric_key"] = intent.primary_metric_key
            return normalized

        if not intent.supports_metric(metric_key):
            normalized["metric_key"] = intent.primary_metric_key
        return normalized

    def _pick_metric_value(self, metrics: dict[str, Any], metric_key: str) -> tuple[Any | None, str | None, str | None]:
        normalized = {self._normalize_metric_key(k): metrics[k] for k in metrics}
        target = self._normalize_metric_key(metric_key)

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

    def _resolve_metric(self, metrics: dict[str, Any], metric_key: str) -> tuple[float | None, dict[str, Any] | None]:
        raw_value, resolved_key, mode = self._pick_metric_value(metrics, metric_key)
        if raw_value is None:
            return None, None
        value = self._to_float(raw_value)
        if value is None:
            return None, None
        resolution = {
            "required_metric_key": self._normalize_metric_key(metric_key),
            "resolved_metric_key": resolved_key or self._normalize_metric_key(metric_key),
            "resolved_value": value,
            "mode": mode or "exact",
            "confidence": "high" if mode in {"exact", "preferred_alias"} else "medium",
            "reason": "",
        }
        return value, resolution

    def select_metric_value(self, metrics: dict[str, Any], metric_key: str) -> float | None:
        value, _ = self._resolve_metric(metrics, metric_key)
        return value

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

    async def _interpret_missing_metric(
        self,
        *,
        required_metric_key: str,
        metrics: dict[str, Any],
        workspace_path,
    ):
        if self.metric_interpreter is None:
            return None
        try:
            return await self.metric_interpreter.resolve_metric(
                required_metric_key=required_metric_key,
                metrics=metrics,
                workspace_path=workspace_path,
            )
        except Exception:
            return None

    def _to_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", ".")
            if "%" in cleaned:
                cleaned = cleaned.replace("%", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    def _normalize_metric_value_for_unit(self, value: float, unit: str) -> float:
        normalized_unit = str(unit or "").strip().lower()
        if normalized_unit not in {"%", "percent", "pct"}:
            return value
        absolute = abs(value)
        if 1.0 < absolute <= 100.0:
            return value / 100.0
        return value

    def _has_split_leakage(self, metrics: dict[str, Any]) -> bool:
        leakage_keys = (
            "split_leakage_detected",
            "data_leakage_detected",
            "evaluation_overlap_detected",
            "train_eval_overlap_detected",
        )
        for key in leakage_keys:
            value = metrics.get(key)
            if value is True:
                return True
            if isinstance(value, str) and value.strip().lower() in {"true", "yes", "1", "pass_with_overlap"}:
                return True
        split_integrity = metrics.get("split_integrity_passed")
        if split_integrity is False:
            return True
        if isinstance(split_integrity, str) and split_integrity.strip().lower() in {"false", "no", "0"}:
            return True
        return False

    def _non_production_report_reason(self, metrics: dict[str, Any]) -> str | None:
        if metrics.get("metric_intent_drift_detected") is True:
            return "quality gate failed: metrics artifact intent does not match inferred task intent"
        if metrics.get("reference_evaluation_fixture_detected") is True:
            return "quality gate failed: reference_evaluation_fixture is not a valid production evaluation artifact"
        if metrics.get("oracle_predictions_detected") is True:
            return "quality gate failed: oracle predictions report is not a valid production evaluation artifact"
        if metrics.get("planning_only_report_detected") is True:
            return "quality gate failed: planning-only report is not a valid production evaluation artifact"
        if metrics.get("non_production_report_detected") is True:
            return "quality gate failed: non-production evaluation artifact cannot satisfy target metric"
        return None

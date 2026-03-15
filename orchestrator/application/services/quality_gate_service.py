from __future__ import annotations

import json
import re
from typing import Any

from orchestrator.application.services.metric_interpretation_service import CodexMetricInterpreter
from orchestrator.application.services.metric_utility_service import MetricUtilityService
from orchestrator.application.services.evaluation_contract_service import EvaluationContractService
from orchestrator.application.services.final_metric_service import FinalMetricService
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
        metric_utility_service: MetricUtilityService | None = None,
        evaluation_contract_service: EvaluationContractService | None = None,
        final_metric_service: FinalMetricService | None = None,
    ):
        self.ralph_backlog = ralph_backlog
        self.settings = settings
        self.metric_interpreter = metric_interpreter
        self.task_intent_service = task_intent_service or TaskIntentService()
        self.metric_utility_service = metric_utility_service or MetricUtilityService()
        self.evaluation_contract_service = evaluation_contract_service or EvaluationContractService()
        self.final_metric_service = final_metric_service or FinalMetricService(self.metric_utility_service)

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

        metric_value, resolution = self.final_metric_service.resolve_metric(metrics, metric_key)
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
        final_metric = self.final_metric_service.build_final_metric(
            metric_key=metric_key,
            resolution=resolution,
            metric_value=metric_value,
            operator=operator,
            target_value=target_value,
            unit=unit,
        )
        search_metric = self.final_metric_service.build_search_metric(
            metric_key=metric_key,
            resolution=resolution,
            final_metric=final_metric,
            metrics=metrics,
        )
        verification.details["final_metric"] = final_metric
        verification.details["search_metric"] = search_metric

        check = self.final_metric_service.passes_requirement(final_metric=final_metric)

        if check:
            return True, "quality gate passed"
        return (
            False,
            (
                f"quality gate failed: {metric_key}={float(final_metric['utility']):.6g} {operator} "
                f"{float(final_metric['target_utility']):.6g} required (unit={unit})"
            ),
        )

    def extract_requirement(
        self,
        task: dict[str, Any],
        workspace_path,
        story_id: str | None = None,
    ) -> dict[str, Any] | None:
        contract_requirement = self._requirement_from_contract(task)
        if contract_requirement:
            return contract_requirement
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
        return self.final_metric_service.normalize_metric_key(raw)

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
        contract = self.evaluation_contract_service.load_from_task(task)
        if contract is not None:
            normalized["metric_key"] = contract.primary_metric_key
            normalized["unit"] = "percent" if contract.primary_scale == "percent" else "ratio"
            if contract.target_value is not None and normalized.get("target") in {None, ""}:
                normalized["target"] = contract.target_value
            normalized.setdefault("operator", ">=" if contract.primary_direction == "max" else "<=")
            return normalized
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

    def select_metric_value(self, metrics: dict[str, Any], metric_key: str) -> float | None:
        return self.final_metric_service.select_metric_value(metrics, metric_key)

    def select_metric_utility(self, metrics: dict[str, Any], metric_key: str, unit: str | None) -> float | None:
        return self.final_metric_service.select_metric_utility(metrics, metric_key, unit)

    def attach_search_metric_progress(
        self,
        *,
        task: dict[str, Any],
        workspace_path,
        verification: VerificationResult,
        previous_verification: dict[str, Any] | None = None,
        experiment_history: list[dict[str, Any]] | None = None,
        story_id: str | None = None,
    ) -> dict[str, Any] | None:
        final_metric = verification.details.get("final_metric") if isinstance(verification.details, dict) else None
        if not isinstance(final_metric, dict):
            return None
        current_utility = self.metric_utility_service.coerce_float(final_metric.get("utility"))
        if current_utility is None:
            return None
        requirement = self.extract_requirement(
            task=task,
            workspace_path=workspace_path,
            story_id=story_id,
        ) or {}
        metric_key = str(final_metric.get("metric_key") or requirement.get("metric_key") or "").strip()
        unit = str(final_metric.get("unit") or requirement.get("unit") or "ratio").strip() or "ratio"
        prior_utilities = self._history_utilities(
            experiment_history=experiment_history or [],
            previous_verification=previous_verification,
            metric_key=metric_key,
            unit=unit,
        )
        target_utility = self.metric_utility_service.coerce_float(final_metric.get("target_utility"))
        progress = self.metric_utility_service.progress(
            utility=current_utility,
            prior_utilities=prior_utilities,
            target_utility=target_utility,
        )
        search_metric = dict(verification.details.get("search_metric") or {})
        search_metric.update(progress)
        effective_train_seconds = self.metric_utility_service.coerce_float(search_metric.get("effective_train_seconds"))
        search_metric["gain_per_budget"] = self.metric_utility_service.gain_per_budget(
            self.metric_utility_service.coerce_float(search_metric.get("delta_best")),
            effective_train_seconds,
        )
        verification.details["search_metric"] = search_metric
        return search_metric

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
        return self.metric_utility_service.coerce_float(value)

    def _normalize_metric_value_for_unit(self, value: float, unit: str) -> float:
        scale = self.metric_utility_service.scale_for_unit(unit)
        return self.metric_utility_service.to_utility(value, scale)

    def _history_utilities(
        self,
        *,
        experiment_history: list[dict[str, Any]],
        previous_verification: dict[str, Any] | None,
        metric_key: str,
        unit: str,
    ) -> list[float]:
        utilities: list[float] = []
        for item in experiment_history:
            if not isinstance(item, dict):
                continue
            search_metric = item.get("search_metric") if isinstance(item.get("search_metric"), dict) else None
            if isinstance(search_metric, dict):
                utility = self.metric_utility_service.coerce_float(search_metric.get("utility"))
                if utility is not None:
                    utilities.append(utility)
                    continue
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            utility = self.select_metric_utility(metrics, metric_key, unit)
            if utility is not None:
                utilities.append(utility)
        if utilities or not isinstance(previous_verification, dict):
            return utilities
        history_entries = previous_verification.get("attempt_history")
        if not isinstance(history_entries, list):
            return utilities
        for entry in history_entries:
            if not isinstance(entry, dict):
                continue
            search_metric = entry.get("search_metric") if isinstance(entry.get("search_metric"), dict) else None
            if isinstance(search_metric, dict):
                utility = self.metric_utility_service.coerce_float(search_metric.get("utility"))
                if utility is not None:
                    utilities.append(utility)
                    continue
            metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            utility = self.select_metric_utility(metrics, metric_key, unit)
            if utility is not None:
                utilities.append(utility)
        return utilities

    def _requirement_from_contract(self, task: dict[str, Any]) -> dict[str, Any] | None:
        contract = self.evaluation_contract_service.load_from_task(task)
        if contract is None or contract.target_value is None:
            return None
        return {
            "metric_key": contract.primary_metric_key,
            "operator": ">=" if contract.primary_direction == "max" else "<=",
            "target": contract.target_value,
            "unit": "percent" if contract.primary_scale == "percent" else "ratio",
        }

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

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from orchestrator.application.services.task_intent_service import TaskIntent, TaskIntentService
from orchestrator.domain.evaluation_contract import BudgetTier, EvaluationContract
from orchestrator.execution.metric_parsing import normalize_metric_key


class EvaluationContractService:
    _QUALITY_RE = re.compile(
        r"(?i)(?:required[_\s-]*metric|ralph_required_metric)\s*[:=]\s*(?P<metric>[a-zA-Z][a-zA-Z0-9_\s\-\./%]*?)\s*(?P<op>>=|<=|!=|==|>|<|=)\s*(?P<value>[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)\s*(?P<unit>%|percent|pct|ratio)?"
    )

    def __init__(self, task_intent_service: TaskIntentService | None = None) -> None:
        self.task_intent_service = task_intent_service or TaskIntentService()

    def build_from_payload(
        self,
        *,
        goal: str,
        constraints: list[str],
        workspace_path: Path | None = None,
        extra_texts: Iterable[str] | None = None,
    ) -> EvaluationContract:
        intent = self.task_intent_service.infer(
            goal=goal,
            constraints=constraints,
            workspace_path=workspace_path,
            extra_texts=extra_texts,
        )
        requirement = self._extract_requirement([goal, *constraints, *(extra_texts or [])])
        return self.build_from_intent(intent=intent, requirement=requirement)

    def build_from_task(
        self,
        task: dict[str, Any],
        *,
        workspace_path: Path | None = None,
        extra_texts: Iterable[str] | None = None,
    ) -> EvaluationContract:
        goal = str(task.get("goal", "")).strip()
        constraints = self._task_constraints(task)
        return self.build_from_payload(
            goal=goal,
            constraints=constraints,
            workspace_path=workspace_path,
            extra_texts=extra_texts,
        )

    def build_from_intent(
        self,
        *,
        intent: TaskIntent,
        requirement: dict[str, Any] | None = None,
    ) -> EvaluationContract:
        primary_metric_key = normalize_metric_key(
            str((requirement or {}).get("metric_key") or intent.primary_metric_key or "quality_metric")
        ) or "quality_metric"
        primary_direction = self._direction_for_metric(primary_metric_key)
        primary_scale = self._scale_for_requirement(requirement)
        return EvaluationContract(
            task_family=intent.task_family,
            primary_metric_key=primary_metric_key,
            primary_direction=primary_direction,
            primary_scale=primary_scale,
            proxy_metric_kind=self._proxy_metric_kind(intent.task_family),
            proxy_direction="min",
            target_value=self._to_float((requirement or {}).get("target")),
            micro_split_id="micro/frozen/v1",
            macro_split_id="validation/main",
            min_effect_size=0.01,
            budget_tiers=self._budget_tiers(intent.requires_real_dataset_smoke),
        )

    def serialize(self, contract: EvaluationContract | dict[str, Any]) -> dict[str, Any]:
        if isinstance(contract, dict):
            parsed = self.deserialize(contract)
        else:
            parsed = contract
        return {
            "task_family": parsed.task_family,
            "primary_metric_key": parsed.primary_metric_key,
            "primary_direction": parsed.primary_direction,
            "primary_scale": parsed.primary_scale,
            "proxy_metric_kind": parsed.proxy_metric_kind,
            "proxy_direction": parsed.proxy_direction,
            "target_value": parsed.target_value,
            "micro_split_id": parsed.micro_split_id,
            "macro_split_id": parsed.macro_split_id,
            "min_effect_size": parsed.min_effect_size,
            "budget_tiers": [
                {
                    "name": tier.name,
                    "max_effective_train_seconds": tier.max_effective_train_seconds,
                    "max_epochs": tier.max_epochs,
                    "max_steps": tier.max_steps,
                    "requires_real_dataset": tier.requires_real_dataset,
                }
                for tier in parsed.budget_tiers
            ],
        }

    def deserialize(self, payload: dict[str, Any] | None) -> EvaluationContract | None:
        if not isinstance(payload, dict):
            return None
        budget_tiers: list[BudgetTier] = []
        for raw in payload.get("budget_tiers") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if name not in {"smoke", "micro", "short", "full"}:
                continue
            budget_tiers.append(
                BudgetTier(
                    name=name,  # type: ignore[arg-type]
                    max_effective_train_seconds=int(raw.get("max_effective_train_seconds") or 0),
                    max_epochs=self._to_int_or_none(raw.get("max_epochs")),
                    max_steps=self._to_int_or_none(raw.get("max_steps")),
                    requires_real_dataset=bool(raw.get("requires_real_dataset")),
                )
            )
        if not budget_tiers:
            budget_tiers = list(self._budget_tiers(False))
        task_family = str(payload.get("task_family") or "generic").strip() or "generic"
        primary_metric_key = (
            normalize_metric_key(str(payload.get("primary_metric_key") or "quality_metric")) or "quality_metric"
        )
        primary_direction = str(payload.get("primary_direction") or self._direction_for_metric(primary_metric_key))
        if primary_direction not in {"max", "min"}:
            primary_direction = self._direction_for_metric(primary_metric_key)
        primary_scale = str(payload.get("primary_scale") or "ratio")
        if primary_scale not in {"ratio", "percent"}:
            primary_scale = "ratio"
        proxy_metric_kind = str(payload.get("proxy_metric_kind") or self._proxy_metric_kind(task_family))
        if proxy_metric_kind not in {"micro_loss", "micro_primary"}:
            proxy_metric_kind = self._proxy_metric_kind(task_family)
        proxy_direction = str(payload.get("proxy_direction") or "min")
        if proxy_direction not in {"max", "min"}:
            proxy_direction = "min"
        return EvaluationContract(
            task_family=task_family,
            primary_metric_key=primary_metric_key,
            primary_direction=primary_direction,  # type: ignore[arg-type]
            primary_scale=primary_scale,  # type: ignore[arg-type]
            proxy_metric_kind=proxy_metric_kind,  # type: ignore[arg-type]
            proxy_direction=proxy_direction,  # type: ignore[arg-type]
            target_value=self._to_float(payload.get("target_value")),
            micro_split_id=str(payload.get("micro_split_id") or "micro/frozen/v1"),
            macro_split_id=str(payload.get("macro_split_id") or "validation/main"),
            min_effect_size=float(payload.get("min_effect_size") or 0.01),
            budget_tiers=tuple(budget_tiers),
        )

    def load_from_task(self, task: dict[str, Any]) -> EvaluationContract | None:
        raw = task.get("evaluation_contract_json")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return self.deserialize(payload)
        if isinstance(raw, dict):
            return self.deserialize(raw)
        return None

    def load_from_run(self, run: Any) -> EvaluationContract | None:
        payload = getattr(run, "evaluation_contract_json", None)
        if isinstance(payload, dict):
            return self.deserialize(payload)
        return None

    def to_prompt_summary(self, contract: EvaluationContract | dict[str, Any] | None) -> str:
        parsed = contract if isinstance(contract, EvaluationContract) else self.deserialize(contract)
        if parsed is None:
            return "none"
        budget_block = ", ".join(
            f"{tier.name}:{tier.max_effective_train_seconds}s"
            for tier in parsed.budget_tiers
        ) or "none"
        return (
            f"- task_family: {parsed.task_family}\n"
            f"- primary_metric_key: {parsed.primary_metric_key}\n"
            f"- primary_direction: {parsed.primary_direction}\n"
            f"- primary_scale: {parsed.primary_scale}\n"
            f"- proxy_metric_kind: {parsed.proxy_metric_kind}\n"
            f"- proxy_direction: {parsed.proxy_direction}\n"
            f"- target_value: {parsed.target_value if parsed.target_value is not None else 'not set'}\n"
            f"- micro_split_id: {parsed.micro_split_id}\n"
            f"- macro_split_id: {parsed.macro_split_id}\n"
            f"- min_effect_size: {parsed.min_effect_size}\n"
            f"- budget_tiers: {budget_block}"
        )

    @staticmethod
    def _task_constraints(task: dict[str, Any]) -> list[str]:
        raw = task.get("constraints_json")
        if not isinstance(raw, str):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [str(item).strip() for item in parsed if str(item).strip()]

    def _extract_requirement(self, texts: Iterable[str]) -> dict[str, Any] | None:
        for text in texts:
            raw = str(text or "").strip()
            if not raw:
                continue
            match = self._QUALITY_RE.search(raw)
            if not match:
                continue
            metric = normalize_metric_key(match.group("metric"))
            if not metric:
                continue
            unit = (match.group("unit") or "").lower().strip()
            return {
                "metric_key": metric,
                "operator": match.group("op"),
                "target": match.group("value"),
                "unit": unit or "ratio",
            }
        return None

    @staticmethod
    def _direction_for_metric(metric_key: str) -> str:
        lowered = normalize_metric_key(metric_key)
        if lowered in {"loss", "val_loss", "train_loss", "eval_loss", "micro_loss", "rmse", "mae", "mse", "mape"}:
            return "min"
        return "max"

    @staticmethod
    def _scale_for_requirement(requirement: dict[str, Any] | None) -> str:
        unit = str((requirement or {}).get("unit") or "ratio").lower().strip()
        if unit in {"%", "percent", "pct"}:
            return "percent"
        return "ratio"

    @staticmethod
    def _proxy_metric_kind(task_family: str) -> str:
        if task_family in {"classification", "segmentation", "detection", "regression"}:
            return "micro_loss"
        return "micro_primary"

    @staticmethod
    def _budget_tiers(requires_real_dataset: bool) -> tuple[BudgetTier, ...]:
        return (
            BudgetTier(
                name="smoke",
                max_effective_train_seconds=15,
                max_epochs=1,
                max_steps=2,
                requires_real_dataset=requires_real_dataset,
            ),
            BudgetTier(
                name="micro",
                max_effective_train_seconds=90,
                max_epochs=2,
                max_steps=64,
                requires_real_dataset=requires_real_dataset,
            ),
            BudgetTier(
                name="short",
                max_effective_train_seconds=300,
                max_epochs=5,
                max_steps=256,
                requires_real_dataset=requires_real_dataset,
            ),
            BudgetTier(
                name="full",
                max_effective_train_seconds=1800,
                max_epochs=None,
                max_steps=None,
                requires_real_dataset=requires_real_dataset,
            ),
        )

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int_or_none(value: Any) -> int | None:
        try:
            if value in {None, ""}:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

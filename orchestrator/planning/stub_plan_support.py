from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestrator.application.services.task_intent_service import TaskIntent

if TYPE_CHECKING:
    from orchestrator.planning.planner import PlanInput


@dataclass(frozen=True, slots=True)
class StubPlanIntent:
    task_family: str
    primary_metric_key: str | None
    preferred_metrics: tuple[str, ...]
    real_dataset_smoke_required: bool
    raw_required_metric_key: str | None
    required_metric_key: str | None


def should_use_codex_preflight(payload: PlanInput) -> bool:
    return payload.previous_verification is None


def requires_explicit_training_shell(payload: PlanInput) -> bool:
    goal = payload.goal.lower()
    training_markers = (
        "train",
        "training",
        "smoke test",
        "smoke-test",
        "обучи",
        "обучи",
        "обучение",
        "сегментатор",
        "классификатор",
        "детектор",
    )
    if any(marker in goal for marker in training_markers):
        return True
    return any("required_metric" in str(item).strip().lower() for item in payload.constraints)


def is_ralph_preparatory_story(payload: PlanInput) -> bool:
    constraint_blob = "\n".join(str(item).strip().lower() for item in payload.constraints if str(item).strip())
    if "ralph_story_id:" not in constraint_blob:
        return False
    goal = payload.goal.lower()
    preparatory_markers = (
        "определить тип",
        "тип датасета",
        "целевые классы",
        "список классов",
        "dataset type",
        "target classes",
        "class list",
        "class names",
        "annotation schema",
        "analyze the structure",
        "проанализировать структуру",
        "аннотацион",
        "categories",
        "segmentation",
    )
    preparatory_hits = sum(1 for marker in preparatory_markers if marker in goal)
    return preparatory_hits >= 2


def constraint_value(constraints: list[str], prefix: str) -> str | None:
    for raw in constraints:
        value = str(raw).strip()
        if value.startswith(prefix):
            return value[len(prefix) :].strip() or None
    return None


def required_metric_key(constraints: list[str]) -> str | None:
    for raw in constraints:
        value = str(raw).strip()
        if not value:
            continue
        upper = value.upper()
        if "REQUIRED_METRIC" not in upper:
            continue
        metric_part = value.split(":", 1)[-1].strip() if ":" in value else value
        token = metric_part.split()[0].strip().strip("`\"'(),")
        if token:
            return token.lower()
    return None


def task_family(payload: PlanInput) -> str | None:
    return constraint_value(payload.constraints, "TASK_FAMILY:")


def primary_metric_key(payload: PlanInput) -> str | None:
    return constraint_value(payload.constraints, "PRIMARY_METRIC_KEY:")


def preferred_metrics(payload: PlanInput) -> tuple[str, ...]:
    raw = constraint_value(payload.constraints, "PREFERRED_METRICS:")
    if not raw:
        return ()
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(dict.fromkeys(items))


def real_dataset_smoke_required(payload: PlanInput) -> bool:
    raw = constraint_value(payload.constraints, "REAL_DATASET_SMOKE_REQUIRED:")
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes"}


def resolve_stub_plan_intent(payload: PlanInput, inferred_intent: TaskIntent) -> StubPlanIntent:
    resolved_task_family = task_family(payload) or inferred_intent.task_family
    resolved_primary_metric_key = primary_metric_key(payload) or inferred_intent.primary_metric_key
    resolved_preferred_metrics = preferred_metrics(payload) or inferred_intent.preferred_metrics
    resolved_real_dataset_smoke_required = real_dataset_smoke_required(payload)
    if not resolved_real_dataset_smoke_required:
        resolved_real_dataset_smoke_required = inferred_intent.requires_real_dataset_smoke
    raw_required = required_metric_key(payload.constraints)
    resolved_required = raw_required or resolved_primary_metric_key
    if (
        resolved_task_family != "generic"
        and resolved_primary_metric_key
        and raw_required
        and resolved_preferred_metrics
        and raw_required not in resolved_preferred_metrics
    ):
        resolved_required = resolved_primary_metric_key
    return StubPlanIntent(
        task_family=resolved_task_family,
        primary_metric_key=resolved_primary_metric_key,
        preferred_metrics=tuple(resolved_preferred_metrics),
        real_dataset_smoke_required=resolved_real_dataset_smoke_required,
        raw_required_metric_key=raw_required,
        required_metric_key=resolved_required,
    )

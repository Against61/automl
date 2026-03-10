from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class QualityRequirement:
    metric_key: str
    metric_min: float


@dataclass(slots=True)
class QualityEvaluation:
    passed: bool
    actual_value: float | None
    reason: str


def normalize_metric_key(raw: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", raw.strip().lower()).strip("_")


def parse_quality_requirement_from_text(text: str) -> QualityRequirement | None:
    lowered = text.lower()
    if "accuracy" not in lowered and "acc" not in lowered:
        return None
    match = re.search(r"(?:accuracy|acc)\s*(?:>=|>|=|at least)?\s*(\d+(?:\.\d+)?)\s*%?", lowered)
    if not match:
        return None
    value = float(match.group(1))
    if value > 1.0:
        value = value / 100.0
    return QualityRequirement(metric_key="accuracy", metric_min=value)


def pick_metric_value(metrics: dict[str, Any], metric_key: str) -> float | None:
    key = normalize_metric_key(metric_key)
    aliases = {key, f"test_{key}", f"val_{key}", f"final_{key}"}
    for candidate in aliases:
        if candidate not in metrics:
            continue
        value = to_float(metrics[candidate])
        if value is not None:
            return value
    return None


def evaluate_quality(metrics: dict[str, Any], requirement: QualityRequirement) -> QualityEvaluation:
    value = pick_metric_value(metrics, requirement.metric_key)
    if value is None:
        return QualityEvaluation(
            passed=False,
            actual_value=None,
            reason=f"missing metric '{requirement.metric_key}'",
        )
    if value >= requirement.metric_min:
        return QualityEvaluation(
            passed=True,
            actual_value=value,
            reason=f"{requirement.metric_key}={value:.4f} >= {requirement.metric_min:.4f}",
        )
    return QualityEvaluation(
        passed=False,
        actual_value=value,
        reason=f"{requirement.metric_key}={value:.4f} < {requirement.metric_min:.4f}",
    )


def to_float(value: Any) -> float | None:
    try:
        if isinstance(value, str) and value.strip().endswith("%"):
            return float(value.strip().rstrip("%")) / 100.0
        parsed = float(value)
        if parsed > 1.0 and isinstance(value, str) and "%" in value:
            return parsed / 100.0
        return parsed
    except (TypeError, ValueError):
        return None


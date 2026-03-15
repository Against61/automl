from __future__ import annotations

from typing import Any


def compact_strategy_payload(strategy: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(strategy, dict):
        return None
    diagnosis = strategy.get("diagnosis") if isinstance(strategy.get("diagnosis"), dict) else {}
    objective = strategy.get("objective") if isinstance(strategy.get("objective"), dict) else {}
    chosen = strategy.get("chosen_intervention") if isinstance(strategy.get("chosen_intervention"), dict) else {}
    return {
        "kind": strategy.get("kind"),
        "run_id": strategy.get("run_id"),
        "quality_reason": strategy.get("quality_reason"),
        "diagnosis": {
            "pattern": diagnosis.get("pattern"),
            "confidence": diagnosis.get("confidence"),
        },
        "objective": {
            "metric_key": objective.get("metric_key"),
            "target": objective.get("target"),
            "current_value": objective.get("current_value"),
            "gap": objective.get("gap"),
            "unit": objective.get("unit"),
        },
        "chosen_intervention_id": strategy.get("chosen_intervention_id") or chosen.get("id"),
        "chosen_intervention": {
            "id": chosen.get("id"),
            "type": chosen.get("type"),
            "description": chosen.get("description"),
            "actions": list(chosen.get("actions") or [])[:5] if isinstance(chosen.get("actions"), list) else [],
            "skill_paths": list(chosen.get("skill_paths") or [])[:4]
            if isinstance(chosen.get("skill_paths"), list)
            else [],
            "skill_names": list(chosen.get("skill_names") or [])[:4]
            if isinstance(chosen.get("skill_names"), list)
            else [],
        },
        "recommended_skills": list(strategy.get("recommended_skills") or [])[:6]
        if isinstance(strategy.get("recommended_skills"), list)
        else [],
        "planner_directives": list(strategy.get("planner_directives") or [])[:4]
        if isinstance(strategy.get("planner_directives"), list)
        else [],
    }


def compact_verification_history_entry(payload: dict[str, Any]) -> dict[str, Any]:
    entry = {
        key: value
        for key, value in payload.items()
        if key
        in {
            "status",
            "commands",
            "metrics",
            "latest_hyperparameters",
            "hyperparameter_attempts",
            "quality_gate",
            "attempt",
            "latest_attempt",
        }
    }
    strategy = compact_strategy_payload(payload.get("improvement_strategy"))
    if strategy:
        entry["improvement_strategy"] = strategy
    if isinstance(entry.get("commands"), list):
        entry["commands"] = entry["commands"][:8]
    if isinstance(entry.get("hyperparameter_attempts"), list):
        entry["hyperparameter_attempts"] = [
            item for item in entry["hyperparameter_attempts"][-4:] if isinstance(item, dict)
        ]
    return entry


def normalize_verification_payload(
    current_payload: dict[str, Any],
    previous_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(current_payload or {})
    history: list[dict[str, Any]] = []
    if previous_payload:
        previous_history = previous_payload.get("history")
        if isinstance(previous_history, list):
            history = [entry for entry in previous_history if isinstance(entry, dict)]
        legacy_entry = compact_verification_history_entry(previous_payload)
        if legacy_entry:
            history.append(legacy_entry)

    previous_attempt = 0
    if previous_payload:
        raw_attempt = previous_payload.get("attempt")
        if raw_attempt is not None:
            try:
                previous_attempt = int(raw_attempt)
            except (TypeError, ValueError):
                previous_attempt = 0
        elif previous_payload.get("latest_attempt") is not None:
            try:
                previous_attempt = int(previous_payload.get("latest_attempt"))
            except (TypeError, ValueError):
                previous_attempt = 0
    attempt = max(len(history) + 1, previous_attempt + 1)
    normalized["attempt"] = attempt
    normalized["latest_attempt"] = attempt
    normalized["history"] = history
    return normalized

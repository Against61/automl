from __future__ import annotations

from typing import Any

_ALLOWED_RETRY_REASONS = {
    "infra_error",
    "missing_file",
    "arg_error",
    "contract_error",
    "execution_error",
}

_ALLOWED_STEP_TYPES = {"change", "check"}
_ALLOWED_OPERATIONS = {
    "inspect_workspace",
    "edit_code",
    "run_training",
    "verify_metrics",
    "research",
    "general",
}
_ALLOWED_STEP_INTENTS = {
    "create_file",
    "modify_file",
    "run_training",
    "verify_metrics",
    "general",
}
_ALLOWED_ACTIONS = {"codex", "shell", "read", "verify"}


def normalize_retry_reason(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in _ALLOWED_RETRY_REASONS:
        return raw

    aliases = {
        "quality_gate": "execution_error",
        "quality_threshold_not_met": "execution_error",
        "quality_metric_not_met": "execution_error",
        "quality_failure": "execution_error",
        "validation_failed": "execution_error",
        "timeout": "infra_error",
        "process_timeout": "infra_error",
        "missing_artifact": "missing_file",
        "artifact_missing": "missing_file",
        "bad_args": "arg_error",
        "invalid_arguments": "arg_error",
        "plan_contract_failed": "contract_error",
    }
    if raw in aliases:
        return aliases[raw]

    if "quality" in raw or "metric" in raw:
        return "execution_error"
    if "timeout" in raw or "infra" in raw:
        return "infra_error"
    if "missing" in raw and ("file" in raw or "artifact" in raw or "path" in raw):
        return "missing_file"
    if "arg" in raw:
        return "arg_error"
    if "contract" in raw:
        return "contract_error"
    return "execution_error"


def normalize_step_type(step: dict[str, Any]) -> str | None:
    raw = str(step.get("step_type") or "").strip().lower()
    if not raw:
        return None
    if raw in _ALLOWED_STEP_TYPES:
        return raw

    action = str(step.get("action") or "").strip().lower()
    operation = str(step.get("operation") or "").strip().lower()
    step_intent = str(step.get("step_intent") or "").strip().lower()
    title_blob = " ".join(
        str(step.get(key) or "")
        for key in ("title", "instruction", "codex_prompt", "stop_condition")
    ).lower()

    if raw in {"verify", "verify_metrics", "read", "inspect_workspace", "inspection"}:
        return "check"
    if raw in {"edit", "edit_code", "modify_file", "create_file", "change"}:
        return "change"
    if raw in {"run_training", "train", "training", "shell"}:
        return "check" if action in {"shell", "verify", "read"} else "change"

    if action in {"read", "verify"}:
        return "check"
    if action == "shell" and (operation == "run_training" or step_intent == "run_training" or "train" in title_blob):
        return "check"
    if action == "codex":
        return "change"
    return "check" if operation in {"inspect_workspace", "verify_metrics"} else "change"


def normalize_step_operation(step: dict[str, Any]) -> str | None:
    raw = str(step.get("operation") or "").strip().lower()
    if not raw:
        return None
    if raw in _ALLOWED_OPERATIONS:
        return raw

    action = str(step.get("action") or "").strip().lower()
    step_intent = str(step.get("step_intent") or "").strip().lower()
    title_blob = " ".join(
        str(step.get(key) or "")
        for key in ("title", "instruction", "codex_prompt", "stop_condition", "command")
    ).lower()

    aliases = {
        "verify": "verify_metrics",
        "verification": "verify_metrics",
        "metrics": "verify_metrics",
        "inspect": "inspect_workspace",
        "read": "inspect_workspace",
        "ls": "inspect_workspace",
        "find": "inspect_workspace",
        "grep": "inspect_workspace",
        "rg": "inspect_workspace",
        "edit": "edit_code",
        "codex": "edit_code",
        "shell": "run_training" if step_intent == "run_training" or "train" in title_blob else "inspect_workspace",
        "train": "run_training",
        "training": "run_training",
    }
    if raw in aliases:
        return aliases[raw]

    if step_intent in _ALLOWED_OPERATIONS:
        return step_intent
    if action == "verify":
        return "verify_metrics"
    if action == "read":
        return "inspect_workspace"
    if action == "shell":
        return "run_training" if "train" in title_blob or "--epochs" in title_blob else "inspect_workspace"
    if action == "codex":
        return "edit_code"
    return "general"


def normalize_step_intent(step: dict[str, Any]) -> str | None:
    raw = str(step.get("step_intent") or "").strip().lower()
    if not raw:
        return None
    if raw in _ALLOWED_STEP_INTENTS:
        return raw

    operation = str(step.get("operation") or "").strip().lower()
    action = str(step.get("action") or "").strip().lower()
    title_blob = " ".join(
        str(step.get(key) or "")
        for key in ("title", "instruction", "codex_prompt", "stop_condition", "command")
    ).lower()

    aliases = {
        "edit_code": "modify_file",
        "edit": "modify_file",
        "create": "create_file",
        "create_code": "create_file",
        "modify": "modify_file",
        "verify": "verify_metrics",
        "verification": "verify_metrics",
        "metrics": "verify_metrics",
        "train": "run_training",
        "training": "run_training",
        "shell": "run_training" if "train" in title_blob else "general",
    }
    if raw in aliases:
        return aliases[raw]

    if operation == "run_training" or ("train" in title_blob and action == "shell"):
        return "run_training"
    if operation == "verify_metrics" or "metric" in title_blob or "verify" in title_blob:
        return "verify_metrics"
    if action == "codex":
        return "modify_file"
    return "general"


def normalize_action(step: dict[str, Any]) -> str | None:
    raw = str(step.get("action") or "").strip().lower()
    if not raw:
        return None
    if raw in _ALLOWED_ACTIONS:
        return raw

    operation = str(step.get("operation") or "").strip().lower()
    step_intent = str(step.get("step_intent") or "").strip().lower()
    if raw in {"inspect_workspace", "read"}:
        return "read"
    if raw in {"verify", "verify_metrics"}:
        return "verify"
    if raw in {"run_training", "shell"}:
        return "shell"
    if raw in {"edit_code", "change", "modify_file", "create_file", "codex"}:
        return "codex"
    if operation == "verify_metrics":
        return "verify"
    if operation == "inspect_workspace":
        return "read"
    if operation == "run_training" or step_intent == "run_training":
        return "shell"
    return "codex"


def sanitize_planner_payload(
    payload: Any,
    *,
    collect_changes: bool = False,
) -> Any | tuple[Any, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return (payload, []) if collect_changes else payload
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return (payload, []) if collect_changes else payload
    changes: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        normalized_step_type = normalize_step_type(step)
        if normalized_step_type and normalized_step_type != str(step.get("step_type") or "").strip().lower():
            original = str(step.get("step_type") or "").strip() or "missing"
            step["step_type"] = normalized_step_type
            changes.append(
                {
                    "step_id": str(step.get("id") or "").strip() or None,
                    "field": "step_type",
                    "original": original,
                    "normalized": normalized_step_type,
                }
            )
        normalized_step_intent = normalize_step_intent(step)
        if normalized_step_intent and normalized_step_intent != str(step.get("step_intent") or "").strip().lower():
            original = str(step.get("step_intent") or "").strip() or "missing"
            step["step_intent"] = normalized_step_intent
            changes.append(
                {
                    "step_id": str(step.get("id") or "").strip() or None,
                    "field": "step_intent",
                    "original": original,
                    "normalized": normalized_step_intent,
                }
            )
        normalized_operation = normalize_step_operation(step)
        if normalized_operation and normalized_operation != str(step.get("operation") or "").strip().lower():
            original = str(step.get("operation") or "").strip() or "missing"
            step["operation"] = normalized_operation
            changes.append(
                {
                    "step_id": str(step.get("id") or "").strip() or None,
                    "field": "operation",
                    "original": original,
                    "normalized": normalized_operation,
                }
            )
        normalized_action = normalize_action(step)
        if normalized_action and normalized_action != str(step.get("action") or "").strip().lower():
            original = str(step.get("action") or "").strip() or "missing"
            step["action"] = normalized_action
            changes.append(
                {
                    "step_id": str(step.get("id") or "").strip() or None,
                    "field": "action",
                    "original": original,
                    "normalized": normalized_action,
                }
            )
        step_intent = str(step.get("step_intent") or "").strip().lower()
        operation = str(step.get("operation") or "").strip().lower()
        commands = step.get("commands")
        command = step.get("command")
        has_commands = isinstance(commands, list) and any(str(item).strip() for item in commands)
        if not has_commands and isinstance(command, str) and command.strip():
            has_commands = True
            if not isinstance(commands, list):
                step["commands"] = [command.strip()]
            elif command.strip() not in step["commands"]:
                step["commands"].append(command.strip())
        if step_intent == "run_training" or operation == "run_training":
            if has_commands and step.get("action") != "shell":
                original_action = str(step.get("action") or "").strip() or "codex"
                step["action"] = "shell"
                changes.append(
                    {
                        "step_id": str(step.get("id") or "").strip() or None,
                        "field": "action",
                        "original": original_action,
                        "normalized": "shell",
                    }
                )
        retry_policy = step.get("retry_policy")
        if not isinstance(retry_policy, dict):
            continue
        raw_reasons = retry_policy.get("on")
        if not isinstance(raw_reasons, list):
            continue
        normalized: list[str] = []
        for item in raw_reasons:
            original = str(item or "").strip().lower()
            reason = normalize_retry_reason(item)
            if reason and reason not in normalized:
                normalized.append(reason)
                if reason != original and original:
                    changes.append(
                        {
                            "step_id": str(step.get("id") or "").strip() or None,
                            "field": "retry_policy.on",
                            "original": original,
                            "normalized": reason,
                        }
                    )
        retry_policy["on"] = normalized
    if collect_changes:
        return payload, changes
    return payload

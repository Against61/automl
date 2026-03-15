from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import streamlit as st
from orchestrator.application.services.task_intent_service import TaskIntentService


TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
_TASK_INTENT = TaskIntentService()


def _api_call(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method=method.upper())
    try:
        with request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {raw}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {path} failed: {exc.reason}") from exc


def _line_split(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return default
    raw = value.strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _load_verification_fallback(run_id: str) -> dict[str, Any] | None:
    if not run_id:
        return None
    path = Path("workspace") / "runs" / run_id / "verification.latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _verification_for_display(run_payload: dict[str, Any]) -> dict[str, Any] | None:
    verification = run_payload.get("verification_json")
    if isinstance(verification, dict):
        return verification
    run_id = str(run_payload.get("run_id") or "").strip()
    return _load_verification_fallback(run_id)


def _structured_metrics_payload(run_payload: dict[str, Any], task_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    run_id = str(run_payload.get("run_id") or "").strip()
    verification = _verification_for_display(run_payload)
    if isinstance(verification, dict):
        metrics = verification.get("metrics")
        if isinstance(metrics, dict) and metrics:
            return metrics, "verification.metrics"

    run_metrics_path = Path("workspace") / "runs" / run_id / "metrics.json"
    run_metrics = _load_json_if_exists(run_metrics_path)
    if isinstance(run_metrics, dict):
        if isinstance(run_metrics.get("metrics"), dict):
            return run_metrics["metrics"], str(run_metrics_path)
        return run_metrics, str(run_metrics_path)

    workspace_id = str(task_payload.get("workspace_id") or "").strip()
    if workspace_id:
        workspace_root = Path("workspace") / workspace_id
        workspace_candidates = [
            workspace_root / "metrics.json",
            workspace_root / "results.json",
        ]
        for path in workspace_candidates:
            payload = _load_json_if_exists(path)
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get("metrics"), dict):
                return payload["metrics"], str(path)
            return payload, str(path)
    return None, None


def _task_constraints(task_payload: dict[str, Any]) -> list[str]:
    direct = task_payload.get("constraints")
    if isinstance(direct, list):
        return [str(item).strip() for item in direct if str(item).strip()]
    encoded = _safe_json_loads(task_payload.get("constraints_json"), [])
    if isinstance(encoded, list):
        return [str(item).strip() for item in encoded if str(item).strip()]
    payload = task_payload.get("payload_json")
    if isinstance(payload, dict):
        inner = payload.get("payload", {}).get("constraints")
        if isinstance(inner, list):
            return [str(item).strip() for item in inner if str(item).strip()]
    return []


def _extract_constraint_value(constraints: list[str], *prefixes: str) -> str:
    normalized_prefixes = tuple(prefix.lower() for prefix in prefixes)
    for item in constraints:
        raw = str(item).strip()
        lower = raw.lower()
        for prefix in normalized_prefixes:
            if lower.startswith(prefix):
                return raw.split(":", 1)[1].strip() if ":" in raw else raw
    return ""


def _extract_quality_target(constraints: list[str]) -> str:
    return _extract_constraint_value(constraints, "ralph_required_metric:", "required_metric:")


def _format_metric_display(value: Any, unit: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value or "n/a")
    normalized_unit = str(unit or "").strip().lower()
    if normalized_unit in {"%", "percent", "pct"}:
        percent_value = value * 100.0 if abs(value) <= 1.0 else value
        return f"{percent_value:.2f}%"
    return f"{value:.4f}"


def _parse_metric_from_quality_reason(reason: str, metric_key: str) -> float | None:
    if not reason or not metric_key:
        return None
    match = re.search(
        rf"(?i)\b{re.escape(str(metric_key).strip())}\s*=\s*(?P<value>[-+]?\d+(?:\.\d+)?)",
        reason,
    )
    if not match:
        return None
    try:
        return float(match.group("value"))
    except ValueError:
        return None


def _extract_primary_goal(task_payload: dict[str, Any], constraints: list[str]) -> str:
    primary = _extract_constraint_value(constraints, "primary_user_goal:")
    if primary:
        return primary
    return str(task_payload.get("goal") or "").strip()


def _task_intent_payload(run_payload: dict[str, Any], task_payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    verification = _verification_for_display(run_payload)
    if isinstance(verification, dict):
        intent = verification.get("task_intent")
        if isinstance(intent, dict) and intent:
            return intent, "verification.task_intent"

    workspace_id = str(task_payload.get("workspace_id") or run_payload.get("workspace_id") or "").strip()
    workspace_path = Path("workspace") / workspace_id if workspace_id else None
    intent = _TASK_INTENT.infer_from_task(
        task=task_payload,
        workspace_path=workspace_path if workspace_path and workspace_path.exists() else None,
    )
    return {
        "task_family": intent.task_family,
        "metric_family": intent.metric_family,
        "primary_metric_key": intent.primary_metric_key,
        "preferred_metrics": list(intent.preferred_metrics),
        "real_dataset_smoke_required": intent.requires_real_dataset_smoke,
        "evidence": list(intent.evidence),
    }, "inferred"


def _prettify_identifier(value: str) -> str:
    cleaned = re.sub(r"[_-]+", " ", str(value or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title() if cleaned else "N/A"


def _compact_text(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit - 3]}..."


def _parse_skill_context(command: str | None) -> tuple[bool, list[str]]:
    if not command:
        return False, []
    marker = "skill-context:"
    if marker not in command:
        return False, []
    suffix = command.split(marker, 1)[1].strip()
    if suffix.startswith("|") or suffix.startswith("]"):
        return False, []
    suffix = suffix.rstrip("]").strip()
    # expected suffix: "N file(s): path1, path2"
    if ": " in suffix:
        _, files_part = suffix.split(": ", 1)
    else:
        files_part = suffix
    sources = [item.strip() for item in files_part.split(",") if item.strip()]
    return True, sources


def _looks_like_training_step(step: dict[str, Any]) -> tuple[bool, str]:
    candidates = [
        str(step.get("title", "")),
        str(step.get("stop_condition", "")),
        str(step.get("codex_prompt", "")),
        str(step.get("instruction", "")),
    ]
    text = " ".join([value.lower() for value in candidates if value]).replace("_", " ")
    markers = (
        "train",
        "training",
        "fit(",
        "epoch",
        "model",
        "dataloader",
        "dataset",
        "batch size",
        "learning rate",
        "optimizer",
        "smoke test",
        "accuracy",
        "loss",
        "tensorflow",
        "pytorch",
        "torch",
        "cnn",
    )
    for marker in markers:
        if marker in text:
            return True, marker
    return False, ""


def _init_state() -> None:
    st.session_state.setdefault("run_ids", [])
    st.session_state.setdefault("selected_run_id", "")
    st.session_state.setdefault("event_feed", {})
    st.session_state.setdefault("last_status", {})
    st.session_state.setdefault("seen_steps", {})
    st.session_state.setdefault("shown_training", {})
    st.session_state.setdefault("seen_plan_sanitization", {})


def _append_feed(run_id: str, message: str) -> None:
    event_feed: dict[str, list[str]] = st.session_state["event_feed"]
    event_feed.setdefault(run_id, [])
    timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    event_feed[run_id].append(f"[{timestamp}] {message}")
    event_feed[run_id] = event_feed[run_id][-200:]


def _register_run(run_id: str) -> None:
    _remember_run(run_id, select=True)


def _remember_run(run_id: str, *, select: bool = False) -> None:
    run_ids: list[str] = st.session_state["run_ids"]
    if run_id not in run_ids:
        run_ids.insert(0, run_id)
    if select or not st.session_state.get("selected_run_id"):
        st.session_state["selected_run_id"] = run_id
    st.session_state["seen_steps"].setdefault(run_id, 0)
    st.session_state["last_status"].setdefault(run_id, "")
    st.session_state["event_feed"].setdefault(run_id, [])
    st.session_state["shown_training"].setdefault(run_id, "")


def _sync_known_runs(base_url: str) -> None:
    active_statuses = [
        "RECEIVED",
        "CONTEXT_READY",
        "PLAN_READY",
        "EXECUTING",
        "VERIFYING",
        "PACKAGING",
        "WAITING_APPROVAL",
        "WAITING_PLAN_REVIEW",
    ]
    query = "&".join(f"status={item}" for item in active_statuses)
    payload = _api_call(base_url, "GET", f"/runs?{query}&limit=100")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return
    for item in runs:
        if not isinstance(item, dict):
            continue
        run_id = str(item.get("run_id") or "").strip()
        if run_id:
            _remember_run(run_id, select=False)


def _submit_event(
    base_url: str,
    workspace_id: str,
    goal: str,
    constraints: list[str],
    priority: str,
    execution_mode: str,
    required_metric_key: str,
    required_metric_min: float | None,
    max_quality_retries: int,
) -> None:
    payload = {
        "workspace_id": workspace_id,
        "goal": goal,
        "constraints": constraints,
        "priority": priority,
        "execution_mode": execution_mode,
        "required_metric_key": required_metric_key,
        "required_metric_min": required_metric_min,
        "max_quality_retries": max_quality_retries,
    }
    response = _api_call(base_url, "POST", "/event", payload)
    run_id = str(response["run_id"])
    _register_run(run_id)
    _append_feed(run_id, "event submitted")
    st.success(f"Run created: {run_id}")


def _control_action(base_url: str, run_id: str, action: str, reason: str = "") -> None:
    if action == "approve":
        _api_call(base_url, "POST", "/control/approve", {"run_id": run_id})
        _append_feed(run_id, "approve requested")
    elif action == "cancel":
        _api_call(base_url, "POST", "/control/cancel", {"run_id": run_id, "reason": reason or None})
        _append_feed(run_id, f"cancel requested ({reason or 'no reason'})")
    else:
        raise ValueError(f"unknown action: {action}")


def _render_plan(plan_payload: dict[str, Any] | None) -> None:
    if not plan_payload:
        st.info("No plan generated yet.")
        return

    st.subheader("Execution Plan")
    st.write(f"**Version:** `{plan_payload.get('version', 'n/a')}`")
    st.write(f"**Summary:** {plan_payload.get('summary', '')}")
    sanitization = plan_payload.get("planner_sanitization")
    if isinstance(sanitization, list) and sanitization:
        st.warning(f"Planner output was auto-normalized in {len(sanitization)} place(s).")
        with st.expander("Planner sanitization details", expanded=False):
            for item in sanitization:
                if not isinstance(item, dict):
                    continue
                step_id = str(item.get("step_id") or "n/a")
                field = str(item.get("field") or "unknown")
                original = str(item.get("original") or "").strip()
                normalized = str(item.get("normalized") or "").strip()
                st.write(f"- step `{step_id}`: `{field}` -> `{original}` => `{normalized}`")
    steps = plan_payload.get("steps", [])
    if not steps:
        st.info("Plan has no steps.")
        return

    for step in steps:
        title = str(step.get("title", "step"))
        step_id = step.get("id", "n/a")
        action = step.get("action", "codex")
        with st.expander(f"{step_id}: {title}", expanded=False):
            cols = st.columns(3)
            cols[0].caption(f"Action: `{action}`")
            cols[1].caption(f"Type: `{step.get('step_type', 'change')}`")
            cols[2].caption(f"Risk: `{step.get('risk_level', 'low')}`")
            command = step.get("command")
            if command:
                st.code(command)
            commands = step.get("commands") or []
            if commands:
                st.markdown("**Commands:**")
                for command_item in commands:
                    st.code(str(command_item))
            st.markdown(f"**Stop condition:** {step.get('stop_condition', 'n/a')}")
            expected = step.get("expected_artifacts") or []
            if expected:
                st.markdown("**Expected artifacts:**")
                for item in expected:
                    st.write(f"- {item}")
            codex_prompt = step.get("codex_prompt")
            if codex_prompt:
                with st.expander("Codex prompt", expanded=False):
                    st.code(str(codex_prompt), language="text")
            instruction = step.get("instruction")
            if instruction and (not codex_prompt or instruction != codex_prompt):
                with st.expander("Instruction", expanded=False):
                    st.code(str(instruction), language="text")


def _extract_latest_improvement_strategy(run_payload: dict[str, Any]) -> dict[str, Any] | None:
    verification = _verification_for_display(run_payload)
    if not verification:
        return None

    strategy = verification.get("improvement_strategy")
    if isinstance(strategy, dict):
        return _normalize_strategy_for_display(strategy)

    history = verification.get("history")
    if not isinstance(history, list):
        return None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("improvement_strategy")
        if isinstance(candidate, dict):
            return _normalize_strategy_for_display(candidate)
    return None


def _missing_improvement_strategy_reason(run_payload: dict[str, Any]) -> tuple[str, str]:
    verification = _verification_for_display(run_payload)
    run_status = str(run_payload.get("status") or "").strip().upper()
    if not verification:
        if run_status in {"RECEIVED", "CONTEXT_READY", "PLAN_READY", "EXECUTING"}:
            return "info", "No improvement strategy yet: run has not reached verification."
        return "warning", "No improvement strategy available: verification payload is missing."

    quality_gate = verification.get("quality_gate") if isinstance(verification.get("quality_gate"), dict) else {}
    quality_status = str(quality_gate.get("status") or "").strip().lower()
    quality_reason = str(quality_gate.get("reason") or "").strip()

    if quality_status == "passed":
        detail = quality_reason or "quality gate passed"
        return "info", f"No improvement strategy: {detail}."
    if quality_status == "skipped":
        detail = quality_reason or "quality gate skipped"
        return "info", f"No improvement strategy: {detail}."
    if quality_status == "failed":
        detail = quality_reason or "quality gate failed"
        return "warning", f"Improvement strategy expected but missing: {detail}."

    if run_status in {"RECEIVED", "CONTEXT_READY", "PLAN_READY", "EXECUTING", "VERIFYING"}:
        return "info", "No improvement strategy yet: verification is still in progress."
    return "warning", "No improvement strategy available for this run."


def _normalize_strategy_for_display(strategy: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(strategy)
    objective = normalized.get("objective")
    if not isinstance(objective, dict):
        return normalized

    updated_objective = dict(objective)
    metric_key = str(updated_objective.get("metric_key") or "").strip()
    unit = str(updated_objective.get("unit") or "").strip().lower()
    current_value = updated_objective.get("current_value")
    repaired_value = _parse_metric_from_quality_reason(
        str(normalized.get("quality_reason") or ""),
        metric_key,
    )
    if (
        isinstance(current_value, (int, float))
        and repaired_value is not None
        and unit in {"%", "percent", "pct"}
        and abs(current_value) > 1.5
        and abs(repaired_value) <= 1.0
    ):
        updated_objective["current_value"] = repaired_value
        target = updated_objective.get("target")
        if isinstance(target, (int, float)):
            updated_objective["gap"] = target - repaired_value
        normalized["objective"] = updated_objective
    return normalized


def _derive_epic_context(
    run_payload: dict[str, Any],
    task_payload: dict[str, Any],
    plan_payload: dict[str, Any],
) -> dict[str, Any]:
    constraints = _task_constraints(task_payload)
    strategy = _extract_latest_improvement_strategy(run_payload) or {}
    verification = _verification_for_display(run_payload) or {}
    chosen = strategy.get("chosen_intervention") if isinstance(strategy.get("chosen_intervention"), dict) else {}
    objective = strategy.get("objective") if isinstance(strategy.get("objective"), dict) else {}
    history = strategy.get("history") if isinstance(strategy.get("history"), dict) else {}

    story_id = _extract_constraint_value(constraints, "ralph_story_id:")
    quality_target = _extract_quality_target(constraints)
    primary_goal = _extract_primary_goal(task_payload, constraints)
    plan_summary = str(plan_payload.get("summary") or "").strip()

    epic_id = str(chosen.get("id") or story_id or run_payload.get("run_id") or "epic").strip()
    epic_title = (
        str(chosen.get("description") or "").strip()
        or (f"RALPH story {story_id}" if story_id else "")
        or plan_summary
        or primary_goal
        or "Current experiment cycle"
    )
    hypothesis = (
        str(chosen.get("description") or "").strip()
        or strategy.get("quality_reason")
        or (strategy.get("planner_directives") or [None])[0]
        or primary_goal
        or "No explicit hypothesis yet"
    )
    current_value = objective.get("current_value")
    target_value = objective.get("target")
    attempt_number = verification.get("latest_attempt") or verification.get("attempt") or history.get("attempt_number") or 1

    return {
        "epic_id": epic_id,
        "epic_tag": _prettify_identifier(epic_id),
        "epic_title": epic_title,
        "hypothesis": hypothesis,
        "quality_target": quality_target or "not set",
        "current_metric_key": objective.get("metric_key"),
        "current_metric_value": current_value,
        "current_metric_unit": objective.get("unit"),
        "target_metric_value": target_value,
        "target_metric_unit": objective.get("unit"),
        "attempt_number": attempt_number,
        "strategy": strategy,
        "primary_goal": primary_goal,
    }


def _latest_attempts_by_step(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for step in steps:
        step_id = str(step.get("step_id") or "").strip()
        if not step_id:
            continue
        latest[step_id] = step
    return latest


def _annotate_step_cycles(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    cycle_index = 0
    prev_index: int | None = None
    prev_step_id: str | None = None
    for raw in steps:
        step = dict(raw)
        try:
            step_index = int(step.get("step_index", 0))
        except (TypeError, ValueError):
            step_index = 0
        step_id = str(step.get("step_id") or "").strip()
        if prev_index is not None:
            if step_index == 0 and (prev_index != 0 or (prev_step_id and prev_step_id != step_id)):
                cycle_index += 1
            elif step_index < prev_index:
                cycle_index += 1
        step["_plan_cycle"] = cycle_index
        annotated.append(step)
        prev_index = step_index
        prev_step_id = step_id or prev_step_id
    return annotated


def _infer_running_cycle(steps: list[dict[str, Any]], next_step_index: int) -> int:
    if not steps:
        return 0
    last = steps[-1]
    last_cycle = int(last.get("_plan_cycle", 0))
    try:
        last_index = int(last.get("step_index", 0))
    except (TypeError, ValueError):
        last_index = 0
    if next_step_index == 0 and last.get("status") in {"completed", "failed", "timeout"}:
        if last_index != 0:
            return last_cycle + 1
    return last_cycle


def _build_backlog_cards(epic: dict[str, Any]) -> list[dict[str, Any]]:
    strategy = epic.get("strategy") if isinstance(epic.get("strategy"), dict) else {}
    chosen_id = str(strategy.get("chosen_intervention_id") or "").strip()
    chosen = strategy.get("chosen_intervention") if isinstance(strategy.get("chosen_intervention"), dict) else {}
    directives = strategy.get("planner_directives") if isinstance(strategy.get("planner_directives"), list) else []
    interventions = strategy.get("candidate_interventions") if isinstance(strategy.get("candidate_interventions"), list) else []

    cards: list[dict[str, Any]] = []
    if chosen:
        instruction_lines = [str(item).strip() for item in chosen.get("actions") or [] if str(item).strip()]
        cards.append(
            {
                "id": f"next-cycle-{chosen.get('id', 'chosen')}",
                "title": f"Next cycle: {str(chosen.get('id') or 'chosen intervention').strip()}",
                "status": "next_cycle",
                "action": chosen.get("type", "strategy"),
                "step_type": "planning",
                "step_intent": "general",
                "instruction": "\n".join(f"- {line}" for line in instruction_lines) or str(chosen.get("description") or ""),
                "codex_prompt": "\n".join(str(item) for item in directives) if directives else "",
                "stop_condition": "next cycle is selected and translated into a new execution plan",
                "risk_level": chosen.get("cost_level", "medium"),
                "meta": {
                    "epic_tag": epic.get("epic_tag", "N/A"),
                    "tag": "Next Cycle",
                    "subtitle": _compact_text(chosen.get("description") or epic.get("hypothesis")),
                },
            }
        )

    for item in interventions:
        if not isinstance(item, dict):
            continue
        intervention_id = str(item.get("id") or "").strip()
        if intervention_id and intervention_id == chosen_id:
            continue
        actions = [str(action).strip() for action in item.get("actions") or [] if str(action).strip()]
        cards.append(
            {
                "id": f"backlog-{intervention_id or len(cards)}",
                "title": _prettify_identifier(intervention_id or "candidate intervention"),
                "status": "backlog",
                "action": item.get("type", "strategy"),
                "step_type": "planning",
                "step_intent": "general",
                "instruction": "\n".join(f"- {action}" for action in actions) or str(item.get("description") or ""),
                "codex_prompt": "\n".join(str(item) for item in directives) if directives else "",
                "stop_condition": "candidate intervention is promoted into the next experiment cycle",
                "risk_level": item.get("cost_level", "medium"),
                "meta": {
                    "epic_tag": epic.get("epic_tag", "N/A"),
                    "tag": "Backlog",
                    "subtitle": _compact_text(item.get("expected_gain") or item.get("description")),
                },
            }
        )
    return cards


def _build_kanban_model(
    *,
    run_payload: dict[str, Any],
    task_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    status: str,
    running_step: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    resolve_step_title,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    plan_steps = [step for step in (plan_payload.get("steps") or []) if isinstance(step, dict)]
    next_step_index = int(run_payload.get("next_step_index", 0))
    latest_attempts = _latest_attempts_by_step(steps)
    epic = _derive_epic_context(run_payload=run_payload, task_payload=task_payload, plan_payload=plan_payload)

    columns: dict[str, list[dict[str, Any]]] = {"backlog": [], "todo": [], "doing": [], "done": []}
    columns["backlog"] = _build_backlog_cards(epic)

    if running_step is not None:
        columns["doing"].append(
            {
                "id": str(running_step.get("id", "current-step")),
                "title": str(running_step.get("title") or "Current step"),
                "status": "in_progress",
                "action": running_step.get("action", "codex"),
                "step_type": running_step.get("step_type", "change"),
                "step_intent": running_step.get("step_intent", "general"),
                "instruction": running_step.get("instruction") or "",
                "codex_prompt": running_step.get("codex_prompt") or "",
                "stop_condition": running_step.get("stop_condition") or "step completed without errors",
                "risk_level": running_step.get("risk_level", "low"),
                "command": running_step.get("command") or " && ".join(running_step.get("commands") or []),
                "meta": {
                    "epic_tag": epic.get("epic_tag", "N/A"),
                    "tag": status,
                    "subtitle": f"Current plan step #{next_step_index}",
                },
            }
        )
    elif status not in TERMINAL_STATUSES:
        columns["doing"].append(
            {
                "id": f"runtime-{status.lower()}",
                "title": f"Runtime stage: {status}",
                "status": status.lower(),
                "action": "runtime",
                "step_type": "check",
                "step_intent": "general",
                "instruction": f"Run is currently in stage {status}. Execution plan will resume when this stage completes.",
                "codex_prompt": "",
                "stop_condition": f"runtime leaves {status}",
                "risk_level": "low",
                "meta": {
                    "epic_tag": epic.get("epic_tag", "N/A"),
                    "tag": "Runtime",
                    "subtitle": _compact_text(plan_payload.get("summary") or epic.get("hypothesis")),
                },
            }
        )

    for idx, plan_step in enumerate(plan_steps):
        step_id = str(plan_step.get("id") or "").strip()
        latest = latest_attempts.get(step_id)
        title = str(plan_step.get("title") or step_id or f"step-{idx}")
        card = {
            "id": step_id or f"plan-{idx}",
            "title": title,
            "status": latest.get("status") if latest else "planned",
            "action": plan_step.get("action", "codex"),
            "step_type": plan_step.get("step_type", "change"),
            "step_intent": plan_step.get("step_intent", "general"),
            "instruction": plan_step.get("instruction") or "",
            "codex_prompt": plan_step.get("codex_prompt") or "",
            "stop_condition": plan_step.get("stop_condition") or "step completed without errors",
            "risk_level": plan_step.get("risk_level", "low"),
            "command": plan_step.get("command") or " && ".join(plan_step.get("commands") or []),
            "stdout_text": latest.get("stdout_text") if latest else "",
            "stderr_text": latest.get("stderr_text") if latest else "",
            "meta": {
                "epic_tag": epic.get("epic_tag", "N/A"),
                "tag": f"Step #{idx}",
                "subtitle": resolve_step_title(latest) if latest else "",
            },
        }

        if running_step is not None and step_id == str(running_step.get("id") or "").strip():
            continue

        if latest and str(latest.get("status")) == "completed":
            columns["done"].append(card)
            continue

        if latest and str(latest.get("status")) in {"failed", "timeout"} and idx <= next_step_index:
            columns["doing"].append(card)
            continue

        if idx >= next_step_index:
            columns["todo"].append(card)

    return epic, columns


def _render_kanban_card(card: dict[str, Any], column_key: str) -> None:
    meta = card.get("meta") if isinstance(card.get("meta"), dict) else {}
    title = str(card.get("title") or "Untitled task")
    subtitle = str(meta.get("subtitle") or "").strip()
    epic_tag = str(meta.get("epic_tag") or "N/A")
    stage_tag = str(meta.get("tag") or column_key.title())
    action = str(card.get("action") or "n/a")
    intent = str(card.get("step_intent") or "general")
    risk = str(card.get("risk_level") or "low")

    st.markdown(
        (
            "<div class='kanban-card'>"
            f"<div class='kanban-tags'><span class='kanban-chip epic'>{epic_tag}</span>"
            f"<span class='kanban-chip stage'>{stage_tag}</span></div>"
            f"<div class='kanban-title'>{title}</div>"
            f"<div class='kanban-meta'>action: {action} | intent: {intent} | risk: {risk}</div>"
            + (f"<div class='kanban-subtitle'>{subtitle}</div>" if subtitle else "")
            + "</div>"
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Open card", expanded=False):
        command = str(card.get("command") or "").strip()
        if command:
            st.code(command)
        st.markdown(f"**Stop condition:** {card.get('stop_condition', 'n/a')}")
        instruction = str(card.get("instruction") or "").strip()
        if instruction:
            st.markdown("**Instruction**")
            st.code(instruction, language="text")
        prompt = str(card.get("codex_prompt") or "").strip()
        if prompt:
            st.markdown("**Prompt**")
            st.code(prompt, language="text")
        stdout_text = str(card.get("stdout_text") or "").strip()
        stderr_text = str(card.get("stderr_text") or "").strip()
        if stdout_text:
            st.text_area("stdout", value=stdout_text, height=120, disabled=True, key=f"{column_key}-{card['id']}-stdout")
        if stderr_text:
            st.text_area("stderr", value=stderr_text, height=120, disabled=True, key=f"{column_key}-{card['id']}-stderr")


def _render_kanban_view(
    *,
    run_payload: dict[str, Any],
    task_payload: dict[str, Any],
    plan_payload: dict[str, Any],
    status: str,
    running_step: dict[str, Any] | None,
    steps: list[dict[str, Any]],
    resolve_step_title,
) -> None:
    epic, columns = _build_kanban_model(
        run_payload=run_payload,
        task_payload=task_payload,
        plan_payload=plan_payload,
        status=status,
        running_step=running_step,
        steps=steps,
        resolve_step_title=resolve_step_title,
    )

    st.subheader("Experiment Epic")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Epic", epic.get("epic_tag", "N/A"))
    m2.metric("Cycle", str(epic.get("attempt_number", "n/a")))
    m3.metric("Quality Target", str(epic.get("quality_target", "not set")))
    current_metric_key = epic.get("current_metric_key")
    current_metric_value = epic.get("current_metric_value")
    current_metric_unit = epic.get("current_metric_unit")
    current_metric_label = str(current_metric_key or "Current Metric")
    m4.metric(current_metric_label, _format_metric_display(current_metric_value, current_metric_unit))

    st.markdown(
        (
            "<div class='epic-banner'>"
            f"<div class='epic-title'>{epic.get('epic_title', 'Current experiment')}</div>"
            f"<div class='epic-hypothesis'><strong>Main hypothesis:</strong> {_compact_text(epic.get('hypothesis'), 320)}</div>"
            f"<div class='epic-goal'><strong>Primary goal:</strong> {_compact_text(epic.get('primary_goal'), 320)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    board_cols = st.columns(4)
    board_order = [
        ("backlog", "Backlog"),
        ("todo", "Todo"),
        ("doing", "Doing"),
        ("done", "Done"),
    ]
    for ui_col, (key, title) in zip(board_cols, board_order):
        with ui_col:
            st.markdown(f"### {title}")
            cards = columns.get(key, [])
            if not cards:
                st.caption("No cards.")
                continue
            for card in cards:
                _render_kanban_card(card, key)


def _render_improvement_strategy(run_payload: dict[str, Any]) -> None:
    strategy = _extract_latest_improvement_strategy(run_payload)
    if not strategy:
        level, message = _missing_improvement_strategy_reason(run_payload)
        if level == "warning":
            st.warning(message)
        else:
            st.info(message)
        return

    st.subheader("Improvement Strategy")
    objective = strategy.get("objective") if isinstance(strategy.get("objective"), dict) else {}
    diagnosis = strategy.get("diagnosis") if isinstance(strategy.get("diagnosis"), dict) else {}
    chosen = strategy.get("chosen_intervention") if isinstance(strategy.get("chosen_intervention"), dict) else {}
    history = strategy.get("history") if isinstance(strategy.get("history"), dict) else {}

    c1, c2, c3 = st.columns(3)
    c1.metric("Metric", str(objective.get("metric_key", "n/a")))
    current_value = objective.get("current_value")
    unit = objective.get("unit")
    c2.metric("Current", _format_metric_display(current_value, unit))
    target_value = objective.get("target")
    c3.metric("Target", _format_metric_display(target_value, unit))

    st.caption(
        f"Diagnosis: `{diagnosis.get('pattern', 'n/a')}` "
        f"(confidence: `{diagnosis.get('confidence', 'n/a')}`)"
    )
    if strategy.get("quality_reason"):
        st.caption(f"Quality gate reason: {strategy['quality_reason']}")

    with st.expander("Chosen Intervention", expanded=True):
        st.write(f"**ID:** `{chosen.get('id', 'n/a')}`")
        st.write(f"**Type:** `{chosen.get('type', 'n/a')}`")
        if chosen.get("description"):
            st.write(str(chosen["description"]))
        if chosen.get("actions"):
            st.write("**Actions:**")
            for action in chosen["actions"]:
                st.write(f"- {action}")
        skill_paths = [str(item).strip() for item in (chosen.get("skill_paths") or []) if str(item).strip()]
        if skill_paths:
            st.write("**Skills to apply:**")
            for path in skill_paths:
                st.write(f"- `{path}`")
        st.write(f"**Expected gain:** {chosen.get('expected_gain', 'n/a')}")
        st.write(f"**Cost level:** `{chosen.get('cost_level', 'n/a')}`")

    latest_hparams = history.get("latest_hyperparameters")
    if isinstance(latest_hparams, dict) and latest_hparams:
        with st.expander("Latest Hyperparameters", expanded=False):
            st.json(latest_hparams)

    attempts = history.get("hyperparameter_attempts")
    if isinstance(attempts, list) and attempts:
        with st.expander("Hyperparameter Attempts", expanded=False):
            st.json(attempts[-6:])

    experiment_attempts = history.get("experiment_attempts")
    if isinstance(experiment_attempts, list) and experiment_attempts:
        with st.expander("Recent Experiment Attempts", expanded=False):
            st.json(experiment_attempts[-6:])

    directives = strategy.get("planner_directives")
    if isinstance(directives, list) and directives:
        with st.expander("Planner Directives", expanded=False):
            for line in directives:
                st.write(f"- {line}")

    artifact_path = strategy.get("artifact_path")
    if artifact_path:
        st.caption(f"Strategy artifact: `{artifact_path}`")


def _render_task_intent(run_payload: dict[str, Any], task_payload: dict[str, Any]) -> None:
    intent, source = _task_intent_payload(run_payload, task_payload)
    if not isinstance(intent, dict) or not intent:
        return
    verification = _verification_for_display(run_payload) or {}

    st.subheader("Task Intent")
    if source:
        st.caption(f"Source: `{source}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task Family", _prettify_identifier(str(intent.get("task_family") or "")))
    c2.metric("Metric Family", _prettify_identifier(str(intent.get("metric_family") or "")))
    c3.metric("Primary Metric", str(intent.get("primary_metric_key") or "n/a"))
    smoke_required = intent.get("real_dataset_smoke_required")
    c4.metric("Real Dataset Smoke", "yes" if smoke_required is True else "no" if smoke_required is False else "n/a")

    preferred = intent.get("preferred_metrics")
    if isinstance(preferred, list) and preferred:
        st.caption(f"Preferred metrics: {', '.join(str(item) for item in preferred if str(item).strip())}")

    evidence = intent.get("evidence")
    if isinstance(evidence, list) and evidence:
        with st.expander("Intent Evidence", expanded=False):
            for item in evidence:
                st.write(f"- {item}")

    validation = verification.get("intent_validation") if isinstance(verification, dict) else None
    if isinstance(validation, dict) and validation:
        status = str(validation.get("status") or "n/a")
        reason = str(validation.get("reason") or "").strip()
        if status == "failed":
            st.warning(f"Intent validation failed: {reason}")
        elif status == "passed":
            st.caption(f"Intent validation passed: {reason}")
        elif status == "skipped":
            st.caption(f"Intent validation skipped: {reason}")


def _render_structured_metrics(run_payload: dict[str, Any], task_payload: dict[str, Any]) -> None:
    metrics_payload, source = _structured_metrics_payload(run_payload, task_payload)
    if not isinstance(metrics_payload, dict) or not metrics_payload:
        st.info("Structured metrics are not available yet. Preferred artifact: `metrics.json`.")
        return

    metric_aliases: list[tuple[str, str]] = [
        ("eval_accuracy", "Eval Accuracy"),
        ("test_accuracy", "Test Accuracy"),
        ("val_accuracy", "Validation Accuracy"),
        ("train_accuracy", "Train Accuracy"),
        ("loss", "Loss"),
        ("eval_loss", "Eval Loss"),
        ("test_loss", "Test Loss"),
        ("threshold_met", "Threshold Met"),
        ("split_integrity_passed", "Split Integrity"),
    ]
    shown: set[str] = set()

    st.subheader("Structured Metrics")
    if source:
        st.caption(f"Source: `{source}`")

    metric_cols = st.columns(4)
    card_index = 0
    for key, label in metric_aliases:
        if key not in metrics_payload:
            continue
        shown.add(key)
        value = metrics_payload.get(key)
        display = _format_metric_display(value, "%" if "accuracy" in key else None) if isinstance(value, (int, float)) else str(value)
        metric_cols[card_index % 4].metric(label, display)
        card_index += 1

    remaining = {key: value for key, value in metrics_payload.items() if key not in shown}
    if remaining:
        with st.expander("All structured metrics", expanded=False):
            st.json(remaining)


def _render_metric_resolution(run_payload: dict[str, Any]) -> None:
    verification = _verification_for_display(run_payload)
    if not isinstance(verification, dict):
        return
    resolution = verification.get("metric_resolution")
    if not isinstance(resolution, dict) or not resolution:
        return

    st.subheader("Metric Resolution")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required", str(resolution.get("required_metric_key") or "n/a"))
    c2.metric("Resolved From", str(resolution.get("resolved_metric_key") or "n/a"))
    c3.metric("Mode", str(resolution.get("mode") or "n/a"))
    c4.metric("Confidence", str(resolution.get("confidence") or "n/a"))
    resolved_value = resolution.get("resolved_value")
    if resolved_value is not None:
        st.caption(f"Resolved value: `{resolved_value}`")
    reason = str(resolution.get("reason") or "").strip()
    if reason:
        st.caption(reason)


def _render_run_stream(base_url: str, run_id: str) -> None:
    run_payload = _api_call(base_url, "GET", f"/runs/{run_id}")
    status_payload = _api_call(base_url, "GET", f"/status/{run_id}")
    artifacts_payload = _api_call(base_url, "GET", f"/artifacts/{run_id}")
    task_payload = _api_call(base_url, "GET", f"/tasks/{run_payload.get('task_id')}")

    status = str(status_payload.get("status", "UNKNOWN"))
    plan_payload = run_payload.get("plan_json") or {}
    plan_steps = plan_payload.get("steps", [])
    plan_step_by_id: dict[str, dict[str, Any]] = {}
    for item in plan_steps:
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("id", "")).strip()
        if step_id:
            plan_step_by_id[step_id] = item

    def resolve_step_title(step_payload: dict[str, Any]) -> str:
        stored_title = str(step_payload.get("step_title") or "").strip()
        if stored_title:
            return stored_title
        step_id = str(step_payload.get("step_id", "")).strip()
        if step_id and step_id in plan_step_by_id:
            return str(plan_step_by_id[step_id].get("title", "")).strip()
        try:
            idx = int(step_payload.get("step_index"))
        except (TypeError, ValueError):
            return ""
        if 0 <= idx < len(plan_steps) and isinstance(plan_steps[idx], dict):
            return str(plan_steps[idx].get("title", "")).strip()
        return ""
    next_step_index = int(run_payload.get("next_step_index", 0))
    steps = _annotate_step_cycles(run_payload.get("steps", []))
    running_step: dict[str, Any] | None = None
    if status == "EXECUTING" and plan_steps:
        if 0 <= next_step_index < len(plan_steps):
            running_step = plan_steps[next_step_index]
            running_step = dict(running_step)
            running_step["_plan_cycle"] = _infer_running_cycle(steps, next_step_index)

    last_status: dict[str, str] = st.session_state["last_status"]
    if last_status.get(run_id) != status:
        _append_feed(run_id, f"status -> {status}")
        last_status[run_id] = status

    sanitization = plan_payload.get("planner_sanitization")
    if isinstance(sanitization, list) and sanitization:
        signature = json.dumps(sanitization, ensure_ascii=True, sort_keys=True)
        seen_plan_sanitization: dict[str, str] = st.session_state["seen_plan_sanitization"]
        if seen_plan_sanitization.get(run_id) != signature:
            _append_feed(run_id, f"planner output auto-normalized ({len(sanitization)} change(s))")
            seen_plan_sanitization[run_id] = signature

    seen_steps: dict[str, int] = st.session_state["seen_steps"]
    prev_seen = int(seen_steps.get(run_id, 0))
    if len(steps) > prev_seen:
        new_steps = steps[prev_seen:]
        for offset, step in enumerate(new_steps, start=1):
            step_index = step.get("step_index")
            plan_cycle = int(step.get("_plan_cycle", 0))
            step_status = step.get("status")
            step_action = step.get("action")
            step_id = step.get("step_id", "n/a")
            step_title = resolve_step_title(step)
            attempt_no = prev_seen + offset
            used_skills, _ = _parse_skill_context(step.get("command"))
            skill_tag = " • skills applied" if used_skills else ""
            summary = (step.get("stdout_text") or "").splitlines()[:1]
            summary_text = summary[0] if summary else ""
            if step_action == "codex" and "epoch" in (step.get("stdout_text") or "").lower():
                summary_text = f"{summary_text} [training output detected]" if summary_text else "training output detected"
            _append_feed(
                run_id,
                f"step attempt #{attempt_no} (cycle #{plan_cycle}, step #{step_index}, id={step_id}, title={step_title or 'n/a'}) "
                f"{step_action} -> {step_status}{skill_tag}"
                + (f" | {summary_text[:140]}" if summary_text else ""),
            )
        seen_steps[run_id] = len(steps)

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", status)
    col2.metric("Planned Steps", len((run_payload.get("plan_json") or {}).get("steps", [])))
    col3.metric("Executed Attempts", len(steps))

    if running_step:
        is_training, marker = _looks_like_training_step(running_step)
        if is_training:
            st.warning(f"Training step in progress: `{running_step.get('id', 'n/a')}` (`{running_step.get('title', '')}`)")
            st.caption(f"Detected training marker: `{marker}`")
            st.caption("Model training is currently the active plan step.")
            st.caption("Training steps use an idle-timeout watchdog, not the default hard wall-clock timeout.")
            training_key = f"{running_step.get('id', '')}:{running_step.get('title', '')}"
            shown_training = st.session_state.get("shown_training", {})
            if shown_training.get(run_id) != training_key:
                _append_feed(run_id, f"training started in step: {running_step.get('id', 'n/a')} ({running_step.get('title', '')})")
                shown_training[run_id] = training_key
                st.session_state["shown_training"] = shown_training

    if status_payload.get("error_message"):
        st.error(status_payload["error_message"])

    task_goal = task_payload.get("goal")
    if task_goal:
        st.caption(f"Goal: {task_goal}")
    st.caption(f"Task id: {run_payload.get('task_id')}")

    timeline_tab, kanban_tab = st.tabs(["Timeline", "Kanban"])
    with timeline_tab:
        _render_plan(run_payload.get("plan_json"))
        _render_improvement_strategy(run_payload)
        _render_task_intent(run_payload, task_payload)
        _render_structured_metrics(run_payload, task_payload)
        _render_metric_resolution(run_payload)

        st.subheader("Live Event Feed")
        st.caption("Feed shows the full chronological history of attempts across replans/cycles.")
        feed_lines = st.session_state["event_feed"].get(run_id, [])
        st.code("\n".join(feed_lines[-120:]) if feed_lines else "No events yet.", language="text")

        st.subheader("Steps")
        display_steps = list(steps)
        if running_step is not None:
            running_commands = running_step.get("commands") or []
            running_command = running_step.get("command")
            if not running_command and running_commands:
                running_command = " && ".join(str(item) for item in running_commands if str(item).strip())
            display_steps.insert(
                0,
                {
                    "step_index": next_step_index,
                    "step_id": running_step.get("id"),
                    "action": running_step.get("action", "codex"),
                    "status": "in_progress",
                    "command": running_command,
                    "stdout_text": "",
                    "stderr_text": "",
                    "created_at": "in_progress",
                    "_plan_cycle": int(running_step.get("_plan_cycle", 0)),
                    "_synthetic_running": True,
                    "_step_title": str(running_step.get("title", "")).strip(),
                },
            )

        if not display_steps:
            st.info("No steps recorded yet.")
        for idx, step in enumerate(display_steps, start=1):
            is_running = bool(step.get("_synthetic_running"))
            step_title = str(step.get("_step_title") or "").strip() or resolve_step_title(step)
            plan_cycle = int(step.get("_plan_cycle", 0))
            if is_running:
                title = (
                    f"CURRENT | cycle #{plan_cycle} | step #{step.get('step_index')} | "
                    f"{step.get('step_id')} | {step_title or 'n/a'} | {step.get('action')} [in_progress]"
                )
            else:
                title = (
                    f"#{idx} attempt | cycle #{plan_cycle} | step #{step.get('step_index')} | "
                    f"{step.get('step_id')} | {step_title or 'n/a'} | {step.get('action')} [{step.get('status')}]"
                )
            with st.expander(title, expanded=False):
                base_key = f"{run_id}-{step.get('step_index')}-{idx}-{step.get('created_at', '')}-{int(is_running)}"
                if step_title:
                    st.caption(f"Step title: {step_title}")
                if is_running:
                    st.info("Step is currently executing.")
                else:
                    used_skills, skill_sources = _parse_skill_context(step.get("command"))
                    st.markdown(f"**Skills context:** {'used' if used_skills else 'not used'}")
                    if used_skills and skill_sources:
                        st.caption("Sources:")
                        for source in skill_sources:
                            st.code(source)
                st.write(f"Command: `{step.get('command')}`")
                if step.get("stdout_text"):
                    st.text_area("stdout", value=step["stdout_text"], height=140, disabled=True, key=f"stdout-{base_key}")
                if step.get("stderr_text"):
                    st.text_area("stderr", value=step["stderr_text"], height=120, disabled=True, key=f"stderr-{base_key}")

        st.subheader("Artifacts")
        artifacts = artifacts_payload.get("artifacts", [])
        if not artifacts:
            st.info("No artifacts yet.")
        for item in artifacts:
            st.write(f"- `{item.get('kind')}` -> `{item.get('path')}`")

    with kanban_tab:
        st.caption("Kanban shows only the current plan. Historical attempts and previous replans stay in Timeline.")
        _render_kanban_view(
            run_payload=run_payload,
            task_payload=task_payload,
            plan_payload=plan_payload,
            status=status,
            running_step=running_step,
            steps=steps,
            resolve_step_title=resolve_step_title,
        )


def main() -> None:
    st.set_page_config(page_title="Agent UI", layout="wide")
    _init_state()
    st.markdown(
        """
        <style>
        .kanban-card {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-left: 4px solid #0b6e4f;
            border-radius: 12px;
            padding: 0.85rem 0.95rem;
            margin: 0.35rem 0 0.55rem 0;
            background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(240,244,247,0.96));
        }
        .kanban-tags {
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-bottom: 0.45rem;
        }
        .kanban-chip {
            display: inline-block;
            padding: 0.1rem 0.45rem;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .kanban-chip.epic {
            background: #dff3ea;
            color: #0b6e4f;
        }
        .kanban-chip.stage {
            background: #f1e7c9;
            color: #7a5a00;
        }
        .kanban-title {
            font-size: 1rem;
            font-weight: 700;
            color: #18212b;
            margin-bottom: 0.15rem;
        }
        .kanban-meta {
            font-size: 0.78rem;
            color: #506070;
            margin-bottom: 0.3rem;
        }
        .kanban-subtitle {
            font-size: 0.84rem;
            color: #293846;
        }
        .epic-banner {
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.2rem 0 1rem 0;
            background: linear-gradient(135deg, #f7f3e8 0%, #e6efe8 100%);
            border: 1px solid rgba(73, 80, 87, 0.18);
        }
        .epic-title {
            font-size: 1.1rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
            color: #1b2430;
        }
        .epic-hypothesis, .epic-goal {
            font-size: 0.9rem;
            color: #2f3d4b;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Agent Control UI")

    with st.sidebar:
        st.header("Connection")
        base_url = st.text_input("API Base URL", value=os.environ.get("AGENT_API_BASE_URL", "http://localhost:8080"))
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_seconds = st.slider("Refresh Interval (sec)", min_value=1, max_value=15, value=2)
        if st.button("Ping /healthz"):
            try:
                health = _api_call(base_url, "GET", "/healthz")
                st.success(f"health: {health.get('status')}")
            except Exception as exc:
                st.error(str(exc))

    try:
        _sync_known_runs(base_url)
    except Exception:
        pass

    st.subheader("Submit Task")
    with st.form("submit-event-form", clear_on_submit=False):
        workspace_id = st.text_input("workspace_id", value="demo")
        goal = st.text_area("goal", value="Train a simple MNIST baseline and report metrics")
        constraints_raw = st.text_area(
            "constraints (one per line)",
            value="Use PyTorch.\nRun quick smoke test first.\nWrite structured metrics to metrics.json.\nMarkdown report optional.",
        )
        priority = st.selectbox("priority", options=["high", "normal", "low"], index=1)
        execution_mode_options = ["plan_execute", "ralph_story"]
        execution_mode = st.selectbox(
            "execution_mode",
            options=execution_mode_options,
            index=1,
        )
        required_metric_key = st.text_input("required_metric_key", value="accuracy")
        required_metric_min = st.text_input("required_metric_min (blank = no threshold)", value="")
        max_quality_retries = st.number_input(
            "max_quality_retries",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )
        submit_clicked = st.form_submit_button("Create Event")
        if submit_clicked:
            try:
                metric_min = None if not required_metric_min.strip() else float(required_metric_min)
                _submit_event(
                    base_url=base_url,
                    workspace_id=workspace_id,
                    goal=goal,
                    constraints=_line_split(constraints_raw),
                    priority=priority,
                    execution_mode=execution_mode,
                    required_metric_key=required_metric_key,
                    required_metric_min=metric_min,
                    max_quality_retries=int(max_quality_retries),
                )
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Run Monitor")
    col_a, col_b = st.columns([3, 2])
    run_ids = st.session_state["run_ids"]
    with col_a:
        if run_ids:
            selected = st.selectbox("Known Runs", options=run_ids, index=0)
            st.session_state["selected_run_id"] = selected
        else:
            st.info("No runs yet. Submit an event first.")
    with col_b:
        manual_run_id = st.text_input("or enter run_id")
        if st.button("Load run_id") and manual_run_id.strip():
            _register_run(manual_run_id.strip())

    run_id = st.session_state.get("selected_run_id", "")
    if run_id:
        ctl1, ctl2, ctl3 = st.columns([1, 1, 3])
        with ctl1:
            if st.button("Approve"):
                try:
                    _control_action(base_url, run_id, "approve")
                except Exception as exc:
                    st.error(str(exc))
        with ctl2:
            cancel_reason = st.text_input("Cancel reason", value="user requested stop")
            if st.button("Cancel"):
                try:
                    _control_action(base_url, run_id, "cancel", reason=cancel_reason)
                except Exception as exc:
                    st.error(str(exc))
        try:
            _render_run_stream(base_url, run_id)
        except Exception as exc:
            st.error(str(exc))

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()


if __name__ == "__main__":
    main()

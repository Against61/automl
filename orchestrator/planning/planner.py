from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from orchestrator.config import Settings
from orchestrator.persistence.schemas import PlannerPlan, RetrievedContext

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


class PlannerError(RuntimeError):
    pass


def _normalize_retry_reason(value: Any) -> str | None:
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


def _normalize_step_type(step: dict[str, Any]) -> str | None:
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


def _normalize_step_operation(step: dict[str, Any]) -> str | None:
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


def _normalize_step_intent(step: dict[str, Any]) -> str | None:
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


def _normalize_action(step: dict[str, Any]) -> str | None:
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


def _sanitize_planner_payload(
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
        normalized_step_type = _normalize_step_type(step)
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
        normalized_step_intent = _normalize_step_intent(step)
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
        normalized_operation = _normalize_step_operation(step)
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
        normalized_action = _normalize_action(step)
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
            reason = _normalize_retry_reason(item)
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


@dataclass(slots=True)
class PlanInput:
    goal: str
    constraints: list[str]
    contexts: list[RetrievedContext]
    workspace_id: str
    workspace_snapshot_summary: str | None = None
    experiment_history_summary: str | None = None
    previous_error: str | None = None
    last_failed_step: dict[str, Any] | None = None
    previous_verification: dict[str, Any] | None = None


class Planner:
    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        raise NotImplementedError

    async def replan(
        self,
        payload: PlanInput,
        failure_reason: str,
        previous_plan: PlannerPlan,
    ) -> PlannerPlan:
        # Default fallback keeps behavior backward-compatible for planners
        # that only implement build_plan.
        return await self.build_plan(payload)


class CodexOnlyPlanner(Planner):
    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        context_lines: list[str] = []
        for idx, item in enumerate(payload.contexts[:6], start=1):
            snippet = " ".join(item.snippet.split())
            if len(snippet) > 600:
                snippet = f"{snippet[:600]}..."
            context_lines.append(
                f"{idx}. {item.document_path}:p{item.page_number} (score={item.confidence:.3f}) -> {snippet}"
            )

        constraints_block = "\n".join(f"- {c}" for c in payload.constraints if c.strip()) or "- none"
        context_block = "\n".join(context_lines) if context_lines else "- no retrieved context"
        workspace_snapshot_block = payload.workspace_snapshot_summary or "none"
        experiment_history_block = payload.experiment_history_summary or "none"
        error_block = payload.previous_error or "none"
        previous_verification_block = payload.previous_verification or {}
        if isinstance(previous_verification_block, dict) and previous_verification_block:
            previous_verification_block = json.dumps(previous_verification_block, ensure_ascii=True, indent=2)
        else:
            previous_verification_block = "none"
        failed_step_block = (
            json.dumps(payload.last_failed_step, ensure_ascii=True, indent=2)
            if payload.last_failed_step
            else "none"
        )

        codex_prompt = (
            "You are the sole execution engine for this run. "
            "Complete the task end-to-end in the current workspace.\n\n"
            f"Goal:\n{payload.goal}\n\n"
            f"Constraints:\n{constraints_block}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Workspace snapshot:\n{workspace_snapshot_block}\n\n"
            f"Experiment history:\n{experiment_history_block}\n\n"
            f"Previous execution error:\n{error_block}\n\n"
            f"Previous verification:\n{previous_verification_block}\n\n"
            f"Last failed step snapshot:\n{failed_step_block}\n\n"
            "Execution contract:\n"
            "- Apply required code/file changes directly in workspace.\n"
            "- Run minimal validation commands needed to prove the goal is met.\n"
            "- For any training or smoke test, evaluation must use a disjoint split and must not reuse the exact training subset.\n"
            "- A train-subset overfit check may be used only as a diagnostic and never as the reported acceptance metric.\n"
            "- Any metrics artifact used for acceptance must explicitly state which split was used for evaluation.\n"
            "- Do not keep long-running training inside a Codex step; if training execution is needed, use an explicit shell command step.\n"
            "- If a command fails, fix root cause and retry before finishing.\n"
            "- Use experiment history to avoid repeating failed recipes or hyperparameter regimes.\n"
            "- Keep output concise and implementation deterministic.\n"
        )

        return PlannerPlan(
            version="1.0",
            summary="Codex-only synthetic plan",
            steps=[
                {
                    "id": "codex-main",
                    "title": "Execute task with Codex only",
                    "step_type": "change",
                    "step_intent": "general",
                    "commands": [],
                    "codex_prompt": codex_prompt,
                    "expected_artifacts": [
                        {"path": None, "kind": "generic", "must_exist": False},
                    ],
                    "stop_condition": "goal implemented and validated",
                    "action": "codex",
                    "instruction": codex_prompt,
                    "command": None,
                    "risk_level": "medium",
                }
            ],
        )

    async def replan(
        self,
        payload: PlanInput,
        failure_reason: str,
        previous_plan: PlannerPlan,
    ) -> PlannerPlan:
        return await self.build_plan(payload)


class OpenAIPlanner(Planner):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise PlannerError("OPENAI_API_KEY is required for OpenAI planner")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        return await self._build_plan(payload=payload)

    async def replan(
        self,
        payload: PlanInput,
        failure_reason: str,
        previous_plan: PlannerPlan,
    ) -> PlannerPlan:
        return await self._build_plan(payload=payload)

    async def _build_plan(self, payload: PlanInput) -> PlannerPlan:
        context_block = []
        for item in payload.contexts:
            context_block.append(
                {
                    "document_path": item.document_path,
                    "page_number": item.page_number,
                    "confidence": item.confidence,
                    "snippet": item.snippet[:800],
                }
            )

        prompt = {
            "goal": payload.goal,
            "constraints": payload.constraints,
            "workspace_id": payload.workspace_id,
            "retrieved_context": context_block,
            "workspace_snapshot_summary": payload.workspace_snapshot_summary,
            "experiment_history": payload.experiment_history_summary,
            "previous_error": payload.previous_error,
            "previous_verification": payload.previous_verification,
            "last_failed_step": payload.last_failed_step,
            "output_schema": {
                "version": "1.0",
                "summary": "string",
                "steps": [
                    {
                        "id": "string unique",
                        "title": "string",
                        "step_type": "change|check",
                        "operation": "inspect_workspace|edit_code|run_training|verify_metrics|research|general",
                        "step_intent": "create_file|modify_file|run_training|verify_metrics|general",
                        "inputs": {
                            "files": ["string path"],
                            "commands": ["string shell command"],
                            "params": {"key": "value"},
                        },
                        "commands": ["string shell command"],
                        "codex_prompt": "string|null",
                        "expected_outputs": {
                            "artifacts": [
                                {
                                    "path": "string|null",
                                    "kind": "file|report|metrics|checkpoint|generic",
                                    "must_exist": "bool",
                                    "must_be_nonempty": "bool",
                                    "metric_keys": ["string"],
                                }
                            ],
                            "metrics_required": ["string"],
                            "stop_condition": "string",
                        },
                        "expected_artifacts": [
                            {
                                "path": "string|null",
                                "kind": "file|report|metrics|checkpoint|generic",
                                "must_exist": "bool",
                                "must_be_nonempty": "bool",
                                "metric_keys": ["string"],
                            }
                        ],
                        "stop_condition": "string",
                        "policy": {
                            "risk": "low|medium|high",
                            "approval_required": "bool",
                        },
                        "retry_policy": {
                            "max_retries": "int",
                            "on": ["infra_error|missing_file|arg_error|contract_error|execution_error"],
                        },
                        "action": "codex|shell|read|verify",
                        "instruction": "string",
                        "command": "string|null",
                        "risk_level": "low|medium|high",
                    }
                ],
            },
        }

        prompt_payload = json.dumps(prompt, ensure_ascii=True)
        completion = await self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                    "You are a deterministic planning engine for an event-driven software agent. "
                    "Output strict JSON that matches the requested schema. "
                    "For each step, include explicit stop_condition and expected_artifacts. "
                    "Also include StepIO fields (operation, inputs, expected_outputs, policy, retry_policy). "
                    "Use structured expected_artifacts objects instead of plain strings. "
                    "For each artifact include path/kind/must_exist/must_be_nonempty and metric_keys when relevant. "
                    "If previous_error, previous_verification, or last_failed_step is present, adapt the plan to fix that root cause first. "
                    "Use experiment_history to avoid repeating failed recipes across previous runs of the same experiment. "
                    "If previous_verification has unmet quality thresholds, prioritize changing hyperparameters, model size, "
                    "batch size, optimizer, or data split strategy. "
                    "Use previous_verification.latest_hyperparameters and previous_verification.attempt_history "
                    "to avoid repeating the same failed hyperparameter sets. "
                    "If previous_verification.improvement_strategy exists, implement its chosen_intervention first. "
                    "Prefer step_type=change for modifications and step_type=check for verification. "
                    "Use action='codex' as the default for change steps. "
                        "Allowed retry_policy.on values are exactly: infra_error, missing_file, arg_error, contract_error, execution_error. "
                        "Do not invent any other retry categories. "
                        "For any training or smoke-test plan, evaluation must be on a disjoint split; never evaluate on the same subset used for training. "
                        "A train-subset overfit check is allowed only as a diagnostic and never as the metric used to pass acceptance criteria or quality gates. "
                        "Any metrics/report artifact used for acceptance must explicitly record the evaluation split or split-integrity note. "
                        "Any long-running training execution must be emitted as action='shell' with an explicit command. "
                        "Use action='codex' for preparing scripts/configs, not for owning the training process itself. "
                        "Use shell only for read-only status/inspection commands (git status, git diff, ls, pwd, cat, rg, grep, find) "
                        "and for explicit long-running training commands. "
                        "If a step is editing, generation, file creation, or package install, set action='codex'. "
                        "Across non-read/verify steps, prefer a majority codex plan; practical default is that at least 70% are codex actions."
                    ),
                },
                {"role": "user", "content": prompt_payload},
            ],
            temperature=0.1,
        )
        content = completion.choices[0].message.content
        if not content:
            raise PlannerError("empty planner response")
        try:
            parsed, sanitization = _sanitize_planner_payload(
                json.loads(content),
                collect_changes=True,
            )
            if isinstance(parsed, dict) and sanitization:
                parsed["planner_sanitization"] = sanitization
            return PlannerPlan.model_validate(parsed)
        except Exception as exc:
            raise PlannerError(f"invalid planner output: {exc}") from exc


class StubPlanner(Planner):
    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        goal = payload.goal.strip() or "user goal not provided"
        constraints_block = "\n".join(f"- {c}" for c in payload.constraints if c.strip()) or "- none"
        context_lines: list[str] = []
        for idx, item in enumerate(payload.contexts[:3], start=1):
            snippet = " ".join(item.snippet.split())
            if len(snippet) > 500:
                snippet = f"{snippet[:500]}..."
            context_lines.append(f"{idx}. {item.document_path}:p{item.page_number} (score={item.confidence:.3f}) -> {snippet}")
        context_block = "\n".join(context_lines) if context_lines else "- no retrieved context"
        workspace_snapshot_block = payload.workspace_snapshot_summary or "none"
        experiment_history_block = payload.experiment_history_summary or "none"
        previous_error = payload.previous_error or "none"
        previous_verification_block = payload.previous_verification or {}
        if isinstance(previous_verification_block, dict) and previous_verification_block:
            previous_verification_block = json.dumps(previous_verification_block, ensure_ascii=True, indent=2)
        else:
            previous_verification_block = "none"
        failed_step_block = (
            json.dumps(payload.last_failed_step, ensure_ascii=True, indent=2)
            if payload.last_failed_step
            else "none"
        )

        base_prompt = (
            f"Task goal:\n{goal}\n\n"
            f"Constraints:\n{constraints_block}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Workspace snapshot:\n{workspace_snapshot_block}\n\n"
            f"Experiment history:\n{experiment_history_block}\n\n"
            f"Previous execution error:\n{previous_error}\n\n"
            f"Previous verification:\n{previous_verification_block}\n\n"
            f"Last failed step snapshot:\n{failed_step_block}\n\n"
            "Implement the task exactly as described above and keep changes confined to this goal.\n"
            "For any training or smoke-test logic, keep training and evaluation splits disjoint.\n"
            "A train-subset overfit check can only be a diagnostic and cannot satisfy the task's acceptance metric.\n"
            "Any metrics artifact used to pass the task must state the evaluation split explicitly.\n"
            "If the task requires actual model training, plan it as an explicit shell command step rather than running it inside Codex.\n"
            "Use experiment history to avoid repeating already failed strategies.\n"
        )

        step2_prompt = base_prompt + "Create or modify code files required to satisfy this goal."
        if payload.previous_error:
            step2_prompt = (
                f"{base_prompt}Previous execution error:\n{payload.previous_error}\n"
                "Fix this root cause before proceeding."
            )
        steps = [
            {
                "id": "step-1",
                "title": "Capture baseline state",
                "step_type": "check",
                "step_intent": "general",
                "commands": ["git status --porcelain", "git diff --stat"],
                "codex_prompt": None,
                "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                "stop_condition": "state snapshot collected",
                "action": "shell",
                "instruction": "Collect short repository state snapshot",
                "command": None,
                "risk_level": "low",
            },
            {
                "id": "step-2",
                "title": "Execute via Codex",
                "step_type": "change",
                "step_intent": "modify_file",
                "commands": [],
                "codex_prompt": step2_prompt,
                "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                "stop_condition": "requested change is implemented without command errors",
                "action": "codex",
                "instruction": step2_prompt,
                "command": None,
                "risk_level": "medium",
            },
            {
                "id": "step-3",
                "title": "Run task checks",
                "step_type": "check",
                "step_intent": "verify_metrics",
                "commands": [],
                "codex_prompt": (
                    f"{base_prompt}Run the minimal checks you need for this goal (tests, smoke checks, quality metrics). "
                    "Return concise command results and artifacts."
                ),
                "expected_artifacts": [{"path": None, "kind": "metrics", "must_exist": False}],
                "stop_condition": "checks run and summarized",
                "action": "codex",
                "instruction": f"{base_prompt}Run smoke tests and any direct validation required by the goal, then summarize results.",
                "command": None,
                "risk_level": "low",
            },
            {
                "id": "step-4",
                "title": "Validate workspace state",
                "step_type": "check",
                "step_intent": "general",
                "commands": ["git status --porcelain", "git diff --stat"],
                "codex_prompt": None,
                "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                "stop_condition": "validation commands succeeded",
                "action": "shell",
                "instruction": "Collect post-change snapshot",
                "command": None,
                "risk_level": "low",
            },
        ]
        return PlannerPlan(version="1.0", summary="Stub plan", steps=steps)

    async def replan(
        self,
        payload: PlanInput,
        failure_reason: str,
        previous_plan: PlannerPlan,
    ) -> PlannerPlan:
        return await self.build_plan(payload)


def make_planner(settings: Settings) -> Planner:
    mode = (settings.planner_mode or "").strip().lower()
    if mode == "codex_only":
        return CodexOnlyPlanner()
    if settings.llm_provider == "openai":
        try:
            return OpenAIPlanner(settings)
        except PlannerError:
            return StubPlanner()
    if settings.llm_provider == "stub":
        return StubPlanner()
    return StubPlanner()

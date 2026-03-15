from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from orchestrator.application.services.prompt_content_service import PromptContentService
from orchestrator.application.services.task_intent_service import TaskIntentService
from orchestrator.planning.planner_sanitizer import sanitize_planner_payload as _sanitize_planner_payload
from orchestrator.planning.stub_plan_support import (
    is_ralph_preparatory_story,
    requires_explicit_training_shell,
    resolve_stub_plan_intent,
    should_use_codex_preflight,
)
from orchestrator.persistence.schemas import PlannerPlan, RetrievedContext
_PROMPT_CONTENT = PromptContentService()
_TASK_INTENT = TaskIntentService()


class PlannerError(RuntimeError):
    pass


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
            previous_verification_block = json.dumps(
                _PROMPT_CONTENT.compact_json_for_prompt(previous_verification_block),
                ensure_ascii=True,
                indent=2,
            )
        else:
            previous_verification_block = "none"
        failed_step_block = (
            json.dumps(
                _PROMPT_CONTENT.compact_json_for_prompt(payload.last_failed_step),
                ensure_ascii=True,
                indent=2,
            )
            if payload.last_failed_step
            else "none"
        )
        workspace_snapshot_block = _PROMPT_CONTENT.compact_text_for_prompt(
            workspace_snapshot_block,
            label="workspace_snapshot",
        ) or "none"
        experiment_history_block = _PROMPT_CONTENT.compact_text_for_prompt(
            experiment_history_block,
            label="experiment_history",
            focus_terms=["accuracy", "iou", "loss", "quality", "chosen", "hyperparameters"],
        ) or "none"
        error_block = _PROMPT_CONTENT.compact_text_for_prompt(
            error_block,
            label="previous_error",
            focus_terms=["error", "traceback", "module", "missing", "timeout"],
        ) or "none"

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
            "- All shell paths are relative to the current workspace root. Do not prefix commands or file paths with the workspace_id directory name.\n"
            "- For any training or smoke test, evaluation must use a disjoint split and must not reuse the exact training subset.\n"
            "- A train-subset overfit check may be used only as a diagnostic and never as the reported acceptance metric.\n"
            "- Any metrics artifact used for acceptance must explicitly state which split was used for evaluation.\n"
            "- Any training or smoke-test script must write structured metrics to metrics.json with canonical keys such as train_accuracy, eval_accuracy, test_accuracy, loss, threshold_met, and split_integrity_passed.\n"
            "- A markdown report is optional and secondary; quality decisions should be derivable from metrics.json.\n"
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
        workspace_snapshot_block = _PROMPT_CONTENT.compact_text_for_prompt(
            payload.workspace_snapshot_summary,
            label="workspace_snapshot",
        ) or "none"
        experiment_history_block = _PROMPT_CONTENT.compact_text_for_prompt(
            payload.experiment_history_summary,
            label="experiment_history",
            focus_terms=["accuracy", "iou", "loss", "quality", "chosen", "hyperparameters"],
        ) or "none"
        previous_error = _PROMPT_CONTENT.compact_text_for_prompt(
            payload.previous_error,
            label="previous_error",
            focus_terms=["error", "traceback", "module", "missing", "timeout"],
        ) or "none"
        previous_verification_block = payload.previous_verification or {}
        if isinstance(previous_verification_block, dict) and previous_verification_block:
            previous_verification_block = json.dumps(
                _PROMPT_CONTENT.compact_json_for_prompt(previous_verification_block),
                ensure_ascii=True,
                indent=2,
            )
        else:
            previous_verification_block = "none"
        failed_step_block = (
            json.dumps(
                _PROMPT_CONTENT.compact_json_for_prompt(payload.last_failed_step),
                ensure_ascii=True,
                indent=2,
            )
            if payload.last_failed_step
            else "none"
        )
        inferred_intent = _TASK_INTENT.infer(
            goal=payload.goal,
            constraints=payload.constraints,
        )
        stub_intent = resolve_stub_plan_intent(payload, inferred_intent)
        task_family = stub_intent.task_family
        primary_metric_key = stub_intent.primary_metric_key
        preferred_metrics = stub_intent.preferred_metrics
        real_dataset_smoke_required = stub_intent.real_dataset_smoke_required
        intent_block = (
            f"- task_family: {task_family}\n"
            f"- primary_metric_key: {primary_metric_key or 'not set'}\n"
            f"- preferred_metrics: {', '.join(preferred_metrics) or 'none'}\n"
            f"- real_dataset_smoke_required: {'true' if real_dataset_smoke_required else 'false'}"
        )

        base_prompt = (
            f"Task goal:\n{goal}\n\n"
            f"Constraints:\n{constraints_block}\n\n"
            f"Inferred task intent:\n{intent_block}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Workspace snapshot:\n{workspace_snapshot_block}\n\n"
            f"Experiment history:\n{experiment_history_block}\n\n"
            f"Previous execution error:\n{previous_error}\n\n"
            f"Previous verification:\n{previous_verification_block}\n\n"
            f"Last failed step snapshot:\n{failed_step_block}\n\n"
            "Implement the task exactly as described above and keep changes confined to this goal.\n"
            "All shell commands and file paths must be relative to the current workspace root; do not prepend the workspace_id directory name.\n"
            "For any training or smoke-test logic, keep training and evaluation splits disjoint.\n"
            "A train-subset overfit check can only be a diagnostic and cannot satisfy the task's acceptance metric.\n"
            "Any metrics artifact used to pass the task must state the evaluation split explicitly.\n"
            "For any new training or smoke-test script, write structured metrics to metrics.json with canonical keys like train_accuracy, eval_accuracy, test_accuracy, loss, threshold_met, and split_integrity_passed.\n"
            "Markdown metrics are optional and secondary.\n"
            "If the task requires actual model training, plan it as an explicit shell command step rather than running it inside Codex.\n"
            "Use experiment history to avoid repeating already failed strategies.\n"
        )

        step2_prompt = base_prompt + "Create or modify code files required to satisfy this goal."
        if payload.previous_error:
            step2_prompt = (
                f"{base_prompt}Previous execution error:\n{payload.previous_error}\n"
                "Fix this root cause before proceeding."
            )
        required_metric_key = stub_intent.required_metric_key
        if requires_explicit_training_shell(payload):
            if is_ralph_preparatory_story(payload):
                step2_prompt = (
                    f"{step2_prompt}\n\n"
                    "This is a preparatory RALPH story. Do not implement real model training in this step.\n"
                    "Create or update a lightweight executable entrypoint at `run_task.py` in the workspace root.\n"
                    "Running `python run_task.py` must only inspect the dataset and produce a planning-only report.\n"
                    "Do not depend on PyTorch or other heavy ML frameworks for this preparatory story.\n"
                    "Inspect dataset structure, annotation schema, split layout, and class/category names.\n"
                    "Write structured output to `metrics.json` with keys such as "
                    "`planning_only_report_detected`, `training_deferred`, `task_family`, `dataset_type`, "
                    "`class_names`, `class_count`, and `split_integrity_passed`.\n"
                    f"Include `candidate_metric_keys` and `suggested_primary_metric` aligned with: {', '.join(preferred_metrics) or 'the inferred metric family'}.\n"
                    "If possible, also record schema hints such as images/annotations/categories/segmentation or bbox/class columns.\n"
                )
                steps = [
                    {
                        "id": "step-1",
                        "title": "Capture baseline state",
                        "step_type": "check",
                        "step_intent": "general",
                        "commands": [
                            "pwd",
                            "ls -la",
                            "find . -maxdepth 3 -type f | sort | sed -n '1,200p'",
                        ],
                        "codex_prompt": None,
                        "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                        "stop_condition": "state snapshot collected",
                        "action": "shell",
                        "instruction": "Collect short workspace state snapshot",
                        "command": None,
                        "risk_level": "low",
                    },
                    {
                        "id": "step-2",
                        "title": "Prepare preparatory dataset analysis script",
                        "step_type": "change",
                        "step_intent": "modify_file",
                        "commands": [],
                        "codex_prompt": step2_prompt,
                        "expected_artifacts": [
                            {"path": "run_task.py", "kind": "file", "must_exist": True, "must_be_nonempty": True},
                        ],
                        "stop_condition": "run_task.py exists and performs preparatory dataset analysis only",
                        "action": "codex",
                        "instruction": step2_prompt,
                        "command": None,
                        "risk_level": "medium",
                    },
                    {
                        "id": "step-3",
                        "title": "Run preparatory dataset analysis",
                        "step_type": "check",
                        "step_intent": "general",
                        "commands": [],
                        "codex_prompt": None,
                        "expected_artifacts": [
                            {"path": "metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True},
                        ],
                        "stop_condition": "python run_task.py completes and writes a planning-only metrics.json report",
                        "action": "shell",
                        "instruction": "Execute the preparatory dataset analysis entrypoint; training is deferred.",
                        "command": "python run_task.py",
                        "risk_level": "low",
                    },
                    {
                        "id": "step-4",
                        "title": "Verify planning-only report",
                        "step_type": "check",
                        "step_intent": "verify_metrics",
                        "commands": [],
                        "codex_prompt": None,
                        "expected_artifacts": [
                            {"path": "metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True},
                        ],
                        "stop_condition": "metrics.json is present and declares planning_only_report_detected=true",
                        "action": "verify",
                        "instruction": "Verify the preparatory planning-only report artifact. Training is deferred.",
                        "command": None,
                        "risk_level": "low",
                    },
                ]
                return PlannerPlan(
                    version="1.0",
                    summary="Stub Ralph preparatory plan (training is deferred)",
                    steps=steps,
                )
            if real_dataset_smoke_required:
                step2_prompt = (
                    f"{step2_prompt}\n\n"
                    f"This is a {task_family} task that requires real-dataset evidence.\n"
                    "Do not use synthetic datasets, random tensors, toy point clouds, or generated samples.\n"
                    "The smoke step must operate on a real subset of the dataset already present in the workspace.\n"
                    "Use a small real train subset and a disjoint real eval subset from the dataset on disk.\n"
                    f"For acceptance-oriented metrics, report one of: {', '.join(preferred_metrics) or 'the preferred metric family'} on the real held-out split.\n"
                    "Metrics outside the preferred family may be logged only as secondary diagnostics unless explicitly required.\n"
                )
            step2_prompt = (
                f"{step2_prompt}\n\n"
                "Create or update a stable executable entrypoint at `run_task.py` in the workspace root.\n"
                "The script must support a fast `--preflight` mode for the very first baseline cycle.\n"
                "Running `python run_task.py --preflight --metrics-path preflight_metrics.json` must only validate dataset parsing, split construction, and a minimal executable smoke path before the real baseline.\n"
                "Running `python run_task.py --metrics-path metrics.json` must execute the minimal training/evaluation flow required by the task.\n"
                "The script must write structured metrics to JSON and keep evaluation disjoint from training.\n"
                "When checking overlap across different annotation json files, compare stable image identity such as `file_name` or normalized image path rather than raw COCO image ids alone.\n"
                "Write `task_family`, `primary_metric_key`, and `evaluation_split` into `metrics.json`.\n"
                "Codex may run only short diagnostic commands while preparing the script; do not run the full baseline training inside the Codex step.\n"
                "Prefer reusing existing dataset paths and code if they already exist in the workspace.\n"
            )
            if should_use_codex_preflight(payload):
                preflight_instruction = (
                    f"{base_prompt}\n\n"
                    "Use this step as a short Codex-owned preflight/debug loop before shell training.\n"
                    "Run only short diagnostic commands such as `python -m py_compile ...`, dataset parsing checks, "
                    "split construction checks, and `python run_task.py --preflight --metrics-path preflight_metrics.json`.\n"
                    "If preflight fails, fix the script and retry inside this Codex step until preflight succeeds.\n"
                    "Do not run the full baseline training here. Full training remains a separate shell step.\n"
                    "Finish only when `preflight_metrics.json` exists and the preflight path passes.\n"
                )
                preflight_step = {
                    "id": "step-3",
                    "title": "Run Codex preflight debug loop",
                    "step_type": "change",
                    "step_intent": "modify_file",
                    "commands": [],
                    "codex_prompt": preflight_instruction,
                    "expected_artifacts": [
                        {"path": "preflight_metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True},
                    ],
                    "stop_condition": "preflight_metrics.json exists and the short preflight path passes",
                    "action": "codex",
                    "instruction": preflight_instruction,
                    "command": None,
                    "risk_level": "medium",
                }
            else:
                preflight_step = {
                    "id": "step-3",
                    "title": "Run preflight validation",
                    "step_type": "check",
                    "step_intent": "general",
                    "commands": [],
                    "codex_prompt": None,
                    "expected_artifacts": [
                        {"path": "preflight_metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True},
                    ],
                    "stop_condition": "python run_task.py --preflight completes and writes preflight_metrics.json",
                    "action": "shell",
                    "instruction": "Execute the short preflight path before the real baseline training step",
                    "command": "python run_task.py --preflight --metrics-path preflight_metrics.json",
                    "risk_level": "low",
                }
            verify_artifact: dict[str, Any] = {
                "path": "metrics.json",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
            }
            if required_metric_key:
                verify_artifact["metric_keys"] = [required_metric_key]
            steps = [
                {
                    "id": "step-1",
                    "title": "Capture baseline state",
                    "step_type": "check",
                    "step_intent": "general",
                    "commands": [
                        "pwd",
                        "ls -la",
                        "find . -maxdepth 3 -type f | sort | sed -n '1,200p'",
                    ],
                    "codex_prompt": None,
                    "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                    "stop_condition": "state snapshot collected",
                    "action": "shell",
                    "instruction": "Collect short workspace state snapshot",
                    "command": None,
                    "risk_level": "low",
                },
                {
                    "id": "step-2",
                    "title": "Prepare executable task script",
                    "step_type": "change",
                    "step_intent": "modify_file",
                    "commands": [],
                    "codex_prompt": step2_prompt,
                    "expected_artifacts": [
                        {"path": "run_task.py", "kind": "file", "must_exist": True, "must_be_nonempty": True},
                    ],
                    "stop_condition": "run_task.py exists and reflects the requested experiment flow",
                    "action": "codex",
                    "instruction": step2_prompt,
                    "command": None,
                    "risk_level": "medium",
                },
                preflight_step,
                {
                    "id": "step-4",
                    "title": "Run training or smoke evaluation",
                    "step_type": "check",
                    "step_intent": "run_training",
                    "commands": [],
                    "codex_prompt": None,
                    "expected_artifacts": [
                        {"path": "metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True},
                    ],
                    "stop_condition": "python run_task.py completes and writes metrics.json",
                    "action": "shell",
                    "instruction": "Execute the prepared training/evaluation entrypoint after preflight succeeds",
                    "command": "python run_task.py --metrics-path metrics.json",
                    "risk_level": "medium",
                },
                {
                    "id": "step-5",
                    "title": "Verify structured metrics",
                    "step_type": "check",
                    "step_intent": "verify_metrics",
                    "commands": [],
                    "codex_prompt": None,
                    "expected_artifacts": [verify_artifact],
                    "stop_condition": "metrics.json is present and contains the required evaluation metric",
                    "action": "verify",
                    "instruction": "Verify the structured metrics artifact produced by the training step.",
                    "command": None,
                    "risk_level": "low",
                },
            ]
            return PlannerPlan(version="1.0", summary="Stub training plan", steps=steps)
        steps = [
            {
                "id": "step-1",
                "title": "Capture baseline state",
                "step_type": "check",
                "step_intent": "general",
                "commands": [
                    "pwd",
                    "ls -la",
                    "find . -maxdepth 3 -type f | sort | sed -n '1,200p'",
                ],
                "codex_prompt": None,
                "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                "stop_condition": "state snapshot collected",
                "action": "shell",
                "instruction": "Collect short workspace state snapshot",
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
                "expected_artifacts": [{"path": "metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True}],
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
                "commands": [
                    "ls -la",
                    "find . -maxdepth 3 -type f | sort | sed -n '1,200p'",
                ],
                "codex_prompt": None,
                "expected_artifacts": [{"path": None, "kind": "generic", "must_exist": False}],
                "stop_condition": "validation commands succeeded",
                "action": "shell",
                "instruction": "Collect post-change workspace snapshot",
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


def make_planner() -> Planner:
    return CodexOnlyPlanner()

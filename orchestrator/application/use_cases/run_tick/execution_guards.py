from __future__ import annotations

import json
import logging
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.application.services.plan_contract_service import PlanContractService
from orchestrator.application.services.ralph_service import RalphScenarioService
from orchestrator.application.services.recovery_service import MissingFileRecoveryService
from orchestrator.execution.codex_runner import CodexRunner, StepExecutionResult
from orchestrator.execution.verifier import VerificationResult
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import (
    ArtifactKind,
    PlannerPlan,
    PlannerStep,
    RunRecord,
    StepIntent,
)

logger = logging.getLogger(__name__)
_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


class ExecutionGuardService:
    def __init__(
        self,
        *,
        db: Database,
        auto_approve_in_pilot: bool,
        plan_contract_service: PlanContractService,
        recovery_service: MissingFileRecoveryService,
        codex_runner: CodexRunner,
        ralph_service: RalphScenarioService,
    ) -> None:
        self.db = db
        self.auto_approve_in_pilot = auto_approve_in_pilot
        self.plan_contract_service = plan_contract_service
        self.recovery_service = recovery_service
        self.codex_runner = codex_runner
        self.ralph_service = ralph_service

    def needs_manual_approval(self, run_approved_at: datetime | None, requires_approval: bool) -> bool:
        return bool(requires_approval and run_approved_at is None and not self.auto_approve_in_pilot)

    def evaluate_plan_contract(
        self,
        *,
        step: Any,
        workspace_path: Path,
        result: Any,
    ) -> tuple[bool, str]:
        return self.plan_contract_service.evaluate(
            step=step,
            workspace_path=workspace_path,
            result=result,
        )

    async def attempt_missing_python_file_recover(
        self,
        *,
        run_id: str,
        step: PlannerStep,
        step_index: int,
        workspace_path: Path,
        result: StepExecutionResult,
        run_path: Path,
    ) -> tuple[PlannerStep, StepExecutionResult] | None:
        decision = self.recovery_service.detect_missing_python_file(result.stderr_text)
        missing_path = decision.missing_path
        if not missing_path:
            return None

        candidates = self.recovery_service.find_python_file_candidates(missing_path, workspace_path)
        candidate_summary = ", ".join(str(path) for path in candidates) if candidates else "none"
        await self.db.insert_run_step(
            run_id=run_id,
            step_id=f"{step.id}-fs-scan",
            step_title=f"Filesystem scan for missing `{missing_path}`",
            step_index=step_index,
            action="read",
            command=f"scan filesystem for python file `{missing_path}`",
            status="completed",
            stdout_text=(
                f"summary: python file missing for step execution\n"
                f"missing: {missing_path}\n"
                f"discovered candidates: {candidate_summary}"
            ),
            stderr_text="",
            duration_ms=0,
        )

        if len(candidates) != 1:
            return None

        replacement = str(candidates[0])
        repaired_step = self.recovery_service.replace_missing_file_in_step(
            step=step,
            expected_missing=missing_path,
            replacement=replacement,
        )
        if repaired_step is None:
            return None

        repaired_result = await self.codex_runner.execute_step(
            run_id=run_id,
            step=repaired_step,
            workspace_path=workspace_path,
            run_path=run_path,
        )
        if repaired_result.status == "completed":
            repaired_result.summary = (
                f"{result.summary}; recovered missing file path via filesystem scan: "
                f"{missing_path} -> {replacement}"
            )
        else:
            repaired_result.summary = f"{result.summary}; retry with recovered path failed: {replacement}"

        return repaired_step, repaired_result

    def extract_missing_python_file_path(self, text: str) -> str | None:
        decision = self.recovery_service.detect_missing_python_file(text)
        return decision.missing_path

    def is_quality_threshold_soft_failure(
        self,
        *,
        step: PlannerStep,
        result: StepExecutionResult,
        workspace_path: Path,
    ) -> bool:
        if result.status == "completed":
            return False
        if step.step_intent != StepIntent.run_training:
            return False

        output = f"{result.stderr_text}\n{result.stdout_text}".lower()
        markers = (
            "target accuracy threshold not reached",
            "accuracy threshold not reached",
            "target metric threshold not reached",
            "required metric not reached",
            "did not reach target accuracy",
        )
        if not any(marker in output for marker in markers):
            return False

        for spec in step.expected_artifacts:
            if spec.kind != ArtifactKind.metrics or not spec.path:
                continue
            target = self.resolve_expected_artifact_target(spec.path, workspace_path)
            if target is None:
                continue
            try:
                if target.exists() and target.stat().st_size > 0:
                    return True
            except OSError:
                continue

        for fallback in list(workspace_path.rglob("*metrics*.json")) + list(workspace_path.rglob("*metrics*.md")):
            try:
                if fallback.exists() and fallback.stat().st_size > 0:
                    return True
            except OSError:
                continue
        return False

    @staticmethod
    def resolve_expected_artifact_target(raw_path: str, workspace_path: Path) -> Path | None:
        cleaned = raw_path.strip().strip("`\"'")
        if not cleaned:
            return None
        normalized = cleaned.replace("\\", "/").strip("/")
        parts = [part for part in normalized.split("/") if part and part != "."]
        if not parts:
            return None
        if parts[0] == "workspace":
            parts = parts[1:]
        if parts and parts[0] == workspace_path.name:
            parts = parts[1:]
        if not parts:
            return None
        try:
            return (workspace_path / "/".join(parts)).resolve()
        except OSError:
            return workspace_path / "/".join(parts)

    def quality_gate_skip_reason(self, *, run: RunRecord, verification: VerificationResult) -> str | None:
        metrics = verification.metrics or {}
        if metrics.get("planning_only_report_detected") is True:
            return "quality gate skipped: planning-only artifacts are not evaluated against target metrics"

        plan = run.plan_json or {}
        steps_raw = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps_raw, list) or not steps_raw:
            return None

        if any(self.plan_step_runs_training(step) for step in steps_raw if isinstance(step, dict)):
            return None

        text_parts: list[str] = [str(plan.get("summary", ""))]
        for step in steps_raw:
            if not isinstance(step, dict):
                continue
            text_parts.extend(
                [
                    str(step.get("title", "")),
                    str(step.get("instruction", "")),
                    str(step.get("codex_prompt", "")),
                    str(step.get("stop_condition", "")),
                ]
            )
        normalized = " ".join(" ".join(text_parts).lower().split())
        planning_markers = (
            "planning-only",
            "planning only",
            "planning artifact",
            "no training is allowed",
            "no model execution",
            "training is deferred",
        )
        if any(marker in normalized for marker in planning_markers):
            return "quality gate skipped: current plan is planning-only"
        return None

    def synthetic_real_dataset_smoke_guard_reason(
        self,
        *,
        task: dict[str, Any],
        workspace_path: Path,
        step: PlannerStep,
    ) -> str | None:
        if step.action != "shell" or step.step_intent != StepIntent.run_training:
            return None
        intent = self.ralph_service.quality_gate_service.infer_task_intent(
            task=task,
            workspace_path=workspace_path,
            story_id=self.ralph_service.extract_story_id(task),
        )
        if not intent.requires_real_dataset_smoke:
            return None
        script_path = self.extract_python_script_target(step, workspace_path)
        if script_path is None or not script_path.is_file():
            return None
        try:
            content = script_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        normalized = " ".join(content.lower().split())
        synthetic_markers = (
            "torch.randn(",
            "_build_synthetic",
            "synthetic",
            "toy point",
            "by construction",
        )
        if any(marker in normalized for marker in synthetic_markers):
            return (
                f"{intent.task_family} smoke step uses synthetic data instead of a real dataset subset; "
                "rewrite the script to train/evaluate on a small real subset from the workspace dataset "
                f"and report one of the preferred metrics: {', '.join(intent.preferred_metrics) or 'task-specific metrics'}"
            )
        return None

    def preflight_dependency_block_reason(
        self,
        *,
        workspace_path: Path,
        step: PlannerStep,
    ) -> str | None:
        if step.action != "shell" or step.step_intent != StepIntent.run_training:
            return None
        reason = self._structured_dependency_issue_reason([workspace_path / "preflight_metrics.json"])
        if not reason:
            return None
        return (
            f"preflight dependency recovery required before training step: {reason}; "
            "environment change required before rerun"
        )

    def structured_dependency_failure_reason(
        self,
        *,
        workspace_path: Path,
        step: PlannerStep,
        result: StepExecutionResult,
    ) -> str | None:
        if result.status == "completed":
            return None
        if step.action != "shell" or step.step_intent != StepIntent.run_training:
            return None
        candidate_paths: list[Path] = [workspace_path / "metrics.json", workspace_path / "preflight_metrics.json"]
        for spec in step.expected_artifacts:
            if spec.kind != ArtifactKind.metrics or not spec.path:
                continue
            target = self.resolve_expected_artifact_target(spec.path, workspace_path)
            if target is not None:
                candidate_paths.append(target)
        reason = self._structured_dependency_issue_reason(candidate_paths)
        if not reason:
            return None
        return f"dependency recovery required after training step: {reason}; environment change required before rerun"

    def _structured_dependency_issue_reason(self, candidate_paths: list[Path]) -> str | None:
        seen: set[Path] = set()
        for path in candidate_paths:
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = self._read_json_mapping(resolved)
            if payload is None:
                continue
            issue = self._dependency_issue_from_payload(payload)
            if issue:
                return f"{issue} (source: {resolved.name})"
        return None

    @staticmethod
    def _read_json_mapping(path: Path) -> dict[str, Any] | None:
        try:
            if not path.exists() or not path.is_file():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _dependency_issue_from_payload(payload: dict[str, Any]) -> str | None:
        error_text = str(payload.get("error") or "").strip()
        mode = str(payload.get("mode") or "").strip().lower()
        smoke_torch_available = payload.get("smoke_torch_available")
        if smoke_torch_available is False:
            return error_text or "PyTorch is not available in the current environment"
        if mode == "smoke_torch_missing" or mode.endswith("_torch_missing"):
            return error_text or "PyTorch is not available in the current environment"
        match = _MISSING_MODULE_RE.search(error_text)
        if match:
            module_name = match.group(1).strip()
            return f"python module '{module_name}' is not installed in the current environment"
        lowered = error_text.lower()
        if "required for smoke execution" in lowered or "not installed in this environment" in lowered:
            return error_text
        return None

    @staticmethod
    def extract_python_script_target(step: PlannerStep, workspace_path: Path) -> Path | None:
        command = step.command or (step.commands[0] if step.commands else "")
        if not command:
            return None
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None
        binary = Path(tokens[0]).name.lower()
        if binary not in {"python", "python3"}:
            return None
        idx = 1
        while idx < len(tokens):
            token = tokens[idx]
            if token in {"-u", "-B", "-E", "-s", "-S", "-O", "-OO", "-q", "-I", "-P"}:
                idx += 1
                continue
            if token in {"-X", "-W"}:
                idx += 2
                continue
            if token.startswith("-"):
                idx += 1
                continue
            if token == "-m":
                return None
            candidate = Path(token)
            if candidate.suffix != ".py":
                return None
            return candidate if candidate.is_absolute() else (workspace_path / candidate)
        return None

    def plan_quality_execution_issue(
        self,
        *,
        task: dict[str, Any],
        workspace_path: Path,
        plan: PlannerPlan,
    ) -> str | None:
        requirement = self.ralph_service.quality_gate_service.extract_requirement(
            task=task,
            workspace_path=workspace_path,
        )
        if not requirement:
            return None
        if self.is_ralph_bootstrap_only_plan(task=task, plan=plan):
            return None
        if self.is_ralph_preparatory_story_plan(task=task, plan=plan):
            return None
        if self.plan_has_explicit_shell_training(plan):
            return None
        summary = str(plan.summary or "").strip()
        if summary.lower() == "stub plan":
            return (
                "quality-targeted task received a Stub plan without an explicit shell run_training step; "
                "manual review required before execution"
            )
        return (
            "quality-targeted task plan has no explicit shell run_training step; "
            "current plan would only inspect or re-verify artifacts"
        )

    def is_ralph_bootstrap_only_plan(self, *, task: dict[str, Any], plan: PlannerPlan) -> bool:
        if not self.ralph_service.is_ralph_task(task):
            return False
        if not plan.steps:
            return False
        return all(self.ralph_service.is_prd_bootstrap_step(step.id) for step in plan.steps)

    def is_ralph_preparatory_story_plan(self, *, task: dict[str, Any], plan: PlannerPlan) -> bool:
        if not self.ralph_service.is_ralph_task(task):
            return False
        summary = str(plan.summary or "").strip().lower()
        if "preparatory" in summary and "training is deferred" in summary:
            return True
        text_parts = [summary]
        for step in plan.steps:
            text_parts.extend(
                [
                    str(step.title or "").lower(),
                    str(step.instruction or "").lower(),
                    str(step.stop_condition or "").lower(),
                ]
            )
        normalized = " ".join(" ".join(text_parts).split())
        return "planning-only" in normalized or "training is deferred" in normalized

    def quality_replan_block_reason(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
        workspace_path: Path,
    ) -> str | None:
        plan_json = run.plan_json if isinstance(run.plan_json, dict) else {}
        try:
            plan = PlannerPlan.model_validate(plan_json)
        except Exception:
            return None
        return self.plan_quality_execution_issue(
            task=task,
            workspace_path=workspace_path,
            plan=plan,
        )

    @staticmethod
    def plan_has_explicit_shell_training(plan: PlannerPlan) -> bool:
        for step in plan.steps:
            if step.step_intent != StepIntent.run_training:
                continue
            if step.action != "shell":
                continue
            commands = [str(item).strip() for item in step.commands if str(item).strip()]
            if step.command and step.command not in commands:
                commands.append(step.command)
            if commands:
                return True
        return False

    @staticmethod
    def plan_step_runs_training(step: dict[str, Any]) -> bool:
        action = str(step.get("action", "")).strip().lower()
        step_intent = str(step.get("step_intent", "")).strip().lower()
        operation = str(step.get("operation", "")).strip().lower()
        if step_intent == StepIntent.run_training.value or operation == StepIntent.run_training.value:
            return True
        if action == "shell":
            command = str(step.get("command", "")).lower()
            commands = " ".join(
                str(item) for item in (step.get("commands") or [] if isinstance(step.get("commands"), list) else [])
            )
            combined = f"{command} {commands}".lower()
            if any(token in combined for token in ("python ", "python3 ", "torchrun", "accelerate launch", "--epochs")):
                return True
        return False

    @staticmethod
    def format_execution_failure(step_id: str, result: Any) -> str:
        stderr = (result.stderr_text or "").strip()
        stderr_compact = " ".join(stderr.split())
        if len(stderr_compact) > 600:
            stderr_compact = f"{stderr_compact[:600]}..."
        return f"execution failed at step '{step_id}': {stderr_compact or 'unknown error'}"

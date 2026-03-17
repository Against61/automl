from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path
from orchestrator.application.services.prompt_content_service import PromptContentService
from orchestrator.execution.command_sequence import CommandSequenceSupport
from orchestrator.config import Settings
from orchestrator.execution.command_recovery import CommandRecoverySupport
from orchestrator.execution.codex_cli import normalize_codex_command
from orchestrator.execution.metric_parsing import extract_metrics_from_text
from orchestrator.execution.codex_prompting import CodexPromptSupport
from orchestrator.execution.runner_models import StepExecutionResult
from orchestrator.execution.shell_command_normalizer import ShellCommandNormalizer
from orchestrator.execution.stepio_support import StepIOSupport
from orchestrator.execution.subprocess_support import SubprocessExecutionSupport
from orchestrator.persistence.schemas import (
    ArtifactSpec,
    PlannerStep,
    StepIntent,
)


class CodexRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.prompt_content_service = PromptContentService()
        self._codex_soft_failure_markers = (
            "i can’t run file operations here due to a sandbox restriction",
            "i can't run file operations here due to a sandbox restriction",
            "i'm unable to run file operations in this environment",
            "not inside a trusted directory and --skip-git-repo-check was not specified",
            "sandbox(landlockrestrict)",
        )
        self.prompt_support = CodexPromptSupport(
            prompt_content_service=self.prompt_content_service,
            soft_failure_markers=self._codex_soft_failure_markers,
        )
        self.shell_command_normalizer = ShellCommandNormalizer()
        self._running: dict[str, asyncio.subprocess.Process] = {}
        self._shell_bin = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        self.subprocess_support = SubprocessExecutionSupport(
            settings=self.settings,
            shell_bin=self._shell_bin,
            sanitized_env=self._sanitized_env,
            truncate=self._truncate,
            normalize_expected_path=self._normalize_expected_path,
            extract_missing_python_file_path=self._extract_missing_python_file_path,
            running=self._running,
        )
        self.stepio_support = StepIOSupport(
            normalize_expected_path=self._normalize_expected_path,
            normalize_step_intent=self._normalize_step_intent,
        )
        self._missing_python_file_re = re.compile(
            r"python[^:]*:\s+can't open file ['\"]([^'\"]+)['\"]:\s+\[Errno\s+2\]\s+No such file or directory",
            flags=re.IGNORECASE,
        )
        self._python_missing_module_map: dict[str, str] = {
            "torch": "torch torchvision",
            "torchvision": "torch torchvision",
            "yaml": "pyyaml",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "PIL": "pillow",
        }
        self.command_recovery = CommandRecoverySupport(
            settings=self.settings,
            shell_primary_binary=self._shell_primary_binary,
            run_command=self._run_command,
            python_missing_module_map=self._python_missing_module_map,
        )
        self.command_sequence = CommandSequenceSupport(
            run_command=self._run_command,
            attempt_missing_module_recovery=self._attempt_missing_module_recovery,
            attempt_argparse_recovery=self._attempt_argparse_recovery,
            is_non_fatal_search_failure=self._is_non_fatal_search_failure,
            git_observation=self._git_observation,
            merge_output_paths_into_files_changed=self._merge_output_paths_into_files_changed,
            write_json_log=self._write_json_log,
        )

    def _sanitized_env(self) -> dict[str, str]:
        env = {"PATH": os.environ.get("PATH", "")}
        if os.environ.get("HOME"):
            env["HOME"] = os.environ["HOME"]
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.settings.codex_use_openai_api_key and self.settings.openai_api_key:
            env["OPENAI_API_KEY"] = self.settings.openai_api_key
        passthrough_vars = (
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "CODEX_HOME",
            "CODEX_CONFIG_HOME",
            "CODEX_API_KEY",
            "CODEX_MODEL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        )
        for key in passthrough_vars:
            value = os.environ.get(key)
            if value:
                env[key] = value
        if self.settings.codex_use_openai_api_key:
            value = os.environ.get("OPENAI_API_KEY")
            if value:
                env["OPENAI_API_KEY"] = value
        if not env.get("CODEX_HOME") and env.get("HOME"):
            env["CODEX_HOME"] = f"{env['HOME']}/.codex"
        if not env.get("CODEX_CONFIG_HOME") and env.get("CODEX_HOME"):
            env["CODEX_CONFIG_HOME"] = env["CODEX_HOME"]
        return env

    def _truncate(self, text: str) -> str:
        if len(text.encode("utf-8")) <= self.settings.max_stdio_bytes:
            return text
        encoded = text.encode("utf-8")[: self.settings.max_stdio_bytes]
        return encoded.decode("utf-8", errors="ignore")

    async def cancel_run(self, run_id: str) -> None:
        process = self._running.get(run_id)
        if process and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()

    async def execute_step(
        self,
        run_id: str,
        step: PlannerStep,
        workspace_path: Path,
        run_path: Path,
    ) -> StepExecutionResult:
        if step.action == "read":
            result = StepExecutionResult(
                status="completed",
                exit_code=0,
                summary="read/check step acknowledged",
                stdout_text="read step acknowledged",
                stderr_text="",
                duration_ms=0,
                command=None,
            )
            self._write_stepio_result(run_id, step, result, workspace_path, run_path)
            return result
        if step.action == "verify":
            result = StepExecutionResult(
                status="completed",
                exit_code=0,
                summary="verify step delegated to verifier",
                stdout_text="verify step delegated to verifier",
                stderr_text="",
                duration_ms=0,
                command=None,
            )
            self._write_stepio_result(run_id, step, result, workspace_path, run_path)
            return result
        if step.action == "shell" or step.commands:
            commands = list(step.commands)
            if step.command and step.command not in commands:
                commands.append(step.command)
            if not commands:
                result = StepExecutionResult(
                    status="failed",
                    exit_code=2,
                    summary="shell step missing commands",
                    stdout_text="",
                    stderr_text="shell step missing commands",
                    duration_ms=0,
                    command=None,
                    errors=["shell step missing commands"],
                    is_infra_error=False,
                )
                self._write_stepio_result(run_id, step, result, workspace_path, run_path)
                return result
            normalized_commands = [
                self._prepare_shell_command(
                    run_id=run_id,
                    command=command,
                    workspace_path=workspace_path,
                )
                for command in commands
            ]
            result = await self._run_commands(run_id, step, normalized_commands, workspace_path, run_path)
            self._write_stepio_result(run_id, step, result, workspace_path, run_path)
            return result
        result = await self._run_codex(run_id, step, workspace_path, run_path)
        self._write_stepio_result(run_id, step, result, workspace_path, run_path)
        return result

    async def _run_codex(
        self,
        run_id: str,
        step: PlannerStep,
        workspace_path: Path,
        run_path: Path,
    ) -> StepExecutionResult:
        prompt_path = run_path / f"{step.id}.prompt.txt"
        base_prompt = step.codex_prompt or step.instruction or ""
        prompt_with_snapshot = self._inject_workspace_snapshot(base_prompt=base_prompt, workspace_path=workspace_path)
        prompt_with_notes = self._inject_learning_notes(base_prompt=prompt_with_snapshot, workspace_path=workspace_path)
        prompt_with_skills, used_skill_paths = self._inject_skill_context(
            base_prompt=prompt_with_notes,
            workspace_path=workspace_path,
            step=step,
        )
        enforced_prompt = self._with_contract_tail(step=step, base_prompt=prompt_with_skills, workspace_path=workspace_path)
        prompt_path.write_text(enforced_prompt, encoding="utf-8")
        base_cmd = self._normalize_codex_command(shlex.split(self.settings.codex_cli_cmd))
        # Generic invocation contract: if command contains "{prompt_file}" placeholder, replace it.
        if any("{prompt_file}" in part for part in base_cmd):
            cmd = [part.replace("{prompt_file}", prompt_path.as_posix()) for part in base_cmd]
            command = " ".join(shlex.quote(x) for x in cmd)
            result = await self._run_command(
                run_id=run_id,
                step_id=step.id,
                command=command,
                workspace_path=workspace_path,
                run_path=run_path,
                stdin_payload=None,
            )
        else:
            command = " ".join(base_cmd)
            result = await self._run_raw(
                run_id=run_id,
                step_id=step.id,
                argv=base_cmd,
                workspace_path=workspace_path,
                run_path=run_path,
                stdin_payload=enforced_prompt,
                command=command,
            )
        result = self._auto_materialize_expected_file(
            step=step,
            result=result,
            workspace_path=workspace_path,
        )
        if used_skill_paths:
            marker = f"skill-context: {len(used_skill_paths)} file(s): {', '.join(used_skill_paths)}"
            result.command = f"{result.command} [{marker}]" if result.command else marker
        if result.status == "completed":
            self._append_learning_note(
                workspace_path=workspace_path,
                step=step,
                result=result,
            )
        return result

    def _with_contract_tail(self, step: PlannerStep, base_prompt: str, workspace_path: Path) -> str:
        expected_paths = self._expected_artifact_paths(step.expected_artifacts, workspace_path=workspace_path)
        lines: list[str] = [
            "",
            "",
            "Execution hard contract:",
            "- Apply file operations directly in workspace; do not return only code snippets in stdout.",
            "- Ensure expected artifacts are created/updated on disk before finishing.",
        ]
        if expected_paths:
            lines.append("Expected output files:")
            lines.extend(f"- {path}" for path in expected_paths)
        intent = self._normalize_step_intent(step.step_intent)
        if intent == StepIntent.run_training:
            lines.append("- This is a training step: run training and persist metrics artifacts.")
        elif intent == StepIntent.verify_metrics:
            lines.append("- This is a metrics verification step: report concrete metric keys/values.")
        elif intent in {StepIntent.create_file, StepIntent.modify_file}:
            lines.append("- Persist created/updated file content to disk in this workspace.")
            lines.append("- Do not start long-running model training inside this Codex step; leave training execution to a separate shell step.")
        return base_prompt + "\n".join(lines)

    def _expected_artifact_paths(self, artifacts: list[ArtifactSpec], workspace_path: Path) -> list[str]:
        return self.prompt_support.expected_artifact_paths(artifacts, workspace_path)

    def _normalize_expected_path(self, raw_path: str, workspace_path: Path) -> str | None:
        return self.prompt_support.normalize_expected_path(raw_path, workspace_path)

    def _normalize_step_intent(self, value: StepIntent | str) -> StepIntent:
        if isinstance(value, StepIntent):
            return value
        try:
            return StepIntent(str(value))
        except ValueError:
            return StepIntent.general

    def _auto_materialize_expected_file(
        self,
        step: PlannerStep,
        result: StepExecutionResult,
        workspace_path: Path,
    ) -> StepExecutionResult:
        if result.status != "completed":
            return result
        expected_paths = self._expected_artifact_paths(step.expected_artifacts, workspace_path=workspace_path)
        if len(expected_paths) != 1:
            return result
        relative_target = expected_paths[0]
        target = (workspace_path / relative_target).resolve()
        workspace_root = workspace_path.resolve()
        try:
            if not target.is_relative_to(workspace_root):
                return result
        except ValueError:
            return result
        if target.exists():
            return result

        code_block = self._extract_first_code_block(result.stdout_text)
        if not code_block:
            return result
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code_block, encoding="utf-8")
        if relative_target not in result.files_changed:
            result.files_changed.append(relative_target)
        result.auto_repaired = True
        result.summary = f"{result.summary}; AUTO_REPAIRED materialized {relative_target} from codex stdout"
        return result

    def _extract_first_code_block(self, text: str) -> str | None:
        return self.prompt_support.extract_first_code_block(text)

    def _inject_workspace_snapshot(self, base_prompt: str, workspace_path: Path) -> str:
        return self.prompt_support.inject_workspace_snapshot(base_prompt, workspace_path)

    def _inject_learning_notes(self, base_prompt: str, workspace_path: Path) -> str:
        return self.prompt_support.inject_learning_notes(base_prompt, workspace_path)

    def _inject_skill_context(
        self,
        *,
        base_prompt: str,
        workspace_path: Path,
        step: PlannerStep,
    ) -> tuple[str, list[str]]:
        return self.prompt_support.inject_skill_context(
            base_prompt=base_prompt,
            workspace_path=workspace_path,
            skill_paths=list(step.skill_paths),
        )

    def _resolve_skill_path(self, raw_path: str, workspace_path: Path) -> Path | None:
        return self.prompt_support.resolve_skill_path(raw_path, workspace_path)

    def _append_learning_note(
        self,
        *,
        workspace_path: Path,
        step: PlannerStep,
        result: StepExecutionResult,
    ) -> None:
        self.prompt_support.append_learning_note(
            workspace_path=workspace_path,
            step_id=step.id,
            step_action=step.action,
            status=result.status,
            summary=result.summary,
            stdout_text=result.stdout_text,
            stderr_text=result.stderr_text,
        )

    def _result_has_codex_soft_failure(self, result: StepExecutionResult) -> bool:
        return self.prompt_support.result_has_codex_soft_failure(result.stdout_text, result.stderr_text)

    def _normalize_codex_command(self, base_cmd: list[str]) -> list[str]:
        return normalize_codex_command(base_cmd, model=self.settings.codex_model)

    async def _run_commands(
        self,
        run_id: str,
        step: PlannerStep,
        commands: list[str],
        workspace_path: Path,
        run_path: Path,
    ) -> StepExecutionResult:
        return await self.command_sequence.run_commands(
            run_id=run_id,
            step=step,
            commands=commands,
            workspace_path=workspace_path,
            run_path=run_path,
        )

    async def _attempt_missing_module_recovery(
        self,
        run_id: str,
        step_id: str,
        command: str,
        base_step_idx: int,
        workspace_path: Path,
        run_path: Path,
        failed_result: StepExecutionResult,
    ) -> tuple[StepExecutionResult, StepExecutionResult] | None:
        return await self.command_recovery.attempt_missing_module_recovery(
            run_id=run_id,
            step_id=step_id,
            command=command,
            base_step_idx=base_step_idx,
            workspace_path=workspace_path,
            run_path=run_path,
            failed_result=failed_result,
        )

    async def _attempt_argparse_recovery(
        self,
        run_id: str,
        step_id: str,
        command: str,
        base_step_idx: int,
        workspace_path: Path,
        run_path: Path,
        failed_result: StepExecutionResult,
    ) -> list[tuple[str, StepExecutionResult]]:
        return await self.command_recovery.attempt_argparse_recovery(
            run_id=run_id,
            step_id=step_id,
            command=command,
            base_step_idx=base_step_idx,
            workspace_path=workspace_path,
            run_path=run_path,
            failed_result=failed_result,
        )

    def _looks_like_python_command(self, command: str) -> bool:
        return self.command_recovery.looks_like_python_command(command)

    def _missing_module_name(self, stderr_text: str) -> str | None:
        return self.command_recovery.missing_module_name(stderr_text)

    def _missing_module_name_from_result(self, result: StepExecutionResult) -> str | None:
        return self.command_recovery.missing_module_name_from_result(result)

    def _is_local_module_reference(self, module_name: str, workspace_path: Path) -> bool:
        return self.command_recovery.is_local_module_reference(module_name, workspace_path)

    async def _run_command(
        self,
        run_id: str,
        step_id: str,
        command: str,
        workspace_path: Path,
        run_path: Path,
        stdin_payload: str | None,
        step: PlannerStep | None = None,
    ) -> StepExecutionResult:
        command = self._prepare_shell_command(
            run_id=run_id,
            command=command,
            workspace_path=workspace_path,
        )
        command = self._normalize_python_runtime_command(command, workspace_path)
        return await self._run_raw(
            run_id=run_id,
            step_id=step_id,
            argv=[self._shell_bin, "-lc", command],
            workspace_path=workspace_path,
            run_path=run_path,
            stdin_payload=stdin_payload,
            command=command,
            step=step,
        )

    def _is_non_fatal_search_failure(self, *, step: PlannerStep, command: str, result: StepExecutionResult) -> bool:
        return self.command_recovery.is_non_fatal_search_failure(
            step=step,
            command=command,
            result=result,
        )

    def _shell_primary_binary(self, command: str) -> str:
        return self.shell_command_normalizer.shell_primary_binary(command)

    def _normalize_python_runtime_command(self, command: str, workspace_path: Path) -> str:
        return self.shell_command_normalizer.normalize_python_runtime_command(command, workspace_path)

    def _sanitize_shell_command(self, command: str, workspace_path: Path) -> str:
        return self.shell_command_normalizer.sanitize_shell_command(command, workspace_path)

    def _prepare_shell_command(self, *, run_id: str, command: str, workspace_path: Path) -> str:
        sanitized = self._sanitize_shell_command(command, workspace_path)
        return self.shell_command_normalizer.rewrite_run_scoped_metrics_paths(
            sanitized,
            run_id=run_id,
        )

    async def _run_raw(
        self,
        run_id: str,
        step_id: str,
        argv: list[str],
        workspace_path: Path,
        run_path: Path,
        stdin_payload: str | None,
        command: str | None = None,
        step: PlannerStep | None = None,
        ) -> StepExecutionResult:
        return await self.subprocess_support.run_raw(
            run_id=run_id,
            step_id=step_id,
            argv=argv,
            workspace_path=workspace_path,
            run_path=run_path,
            stdin_payload=stdin_payload,
            command=command,
            step=step,
        )

    def _extract_missing_python_file_path(self, text: str) -> str | None:
        match = self._missing_python_file_re.search(text or "")
        if not match:
            return None
        return match.group(1).strip()

    async def _git_observation(self, workspace_path: Path) -> tuple[str, list[str]]:
        return await self.subprocess_support.git_observation(workspace_path)

    def _merge_output_paths_into_files_changed(
        self,
        *,
        files_changed: list[str],
        stdout_text: str,
        stderr_text: str,
        workspace_path: Path,
    ) -> list[str]:
        return self.subprocess_support.merge_output_paths_into_files_changed(
            files_changed=files_changed,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            workspace_path=workspace_path,
        )

    def _write_json_log(self, run_path: Path, step_id: str, payload: dict[str, Any]) -> str:
        return self.stepio_support.write_json_log(run_path, step_id, payload)

    def _write_stepio_result(
        self,
        run_id: str,
        step: PlannerStep,
        result: StepExecutionResult,
        workspace_path: Path,
        run_path: Path,
    ) -> str:
        return self.stepio_support.write_stepio_result(
            run_id=run_id,
            step=step,
            result=result,
            workspace_path=workspace_path,
            run_path=run_path,
        )

    def _extract_metrics_from_text(self, text: str) -> dict[str, float | int | str | bool]:
        return extract_metrics_from_text(text)

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.config import Settings
from orchestrator.execution.metric_parsing import extract_metrics_from_text
from orchestrator.persistence.schemas import (
    ArtifactKind,
    ArtifactSpec,
    PlannerStep,
    StepArtifactManifest,
    StepIOResult,
    StepIntent,
)


@dataclass(slots=True)
class StepExecutionResult:
    status: str
    exit_code: int
    summary: str
    stdout_text: str
    stderr_text: str
    duration_ms: int
    command: str | None
    diff_stats: str = ""
    test_report: str = ""
    files_changed: list[str] = None  # type: ignore[assignment]
    errors: list[str] = None  # type: ignore[assignment]
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_path: str | None = None
    is_infra_error: bool = False
    auto_repaired: bool = False
    missing_artifact: str | None = None

    def __post_init__(self) -> None:
        if self.files_changed is None:
            self.files_changed = []
        if self.errors is None:
            self.errors = []


@dataclass(slots=True)
class TimeoutProfile:
    hard_timeout_sec: int | None
    idle_timeout_sec: int | None
    max_wall_clock_sec: int | None
    label: str = "default"


class _IdleProcessTimeout(RuntimeError):
    pass


class CodexRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._running: dict[str, asyncio.subprocess.Process] = {}
        self._shell_bin = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        self._missing_module_re = re.compile(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]")
        self._missing_python_file_re = re.compile(
            r"python[^:]*:\s+can't open file ['\"]([^'\"]+)['\"]:\s+\[Errno\s+2\]\s+No such file or directory",
            flags=re.IGNORECASE,
        )
        self._argparse_unknown_re = re.compile(r"unrecognized arguments:\s*(.+)", flags=re.IGNORECASE)
        self._codex_soft_failure_markers = (
            "i can’t run file operations here due to a sandbox restriction",
            "i can't run file operations here due to a sandbox restriction",
            "i'm unable to run file operations in this environment",
            "not inside a trusted directory and --skip-git-repo-check was not specified",
            "sandbox(landlockrestrict)",
        )
        self._python_missing_module_map: dict[str, str] = {
            "torch": "torch torchvision",
            "torchvision": "torch torchvision",
            "yaml": "pyyaml",
            "cv2": "opencv-python",
            "sklearn": "scikit-learn",
            "PIL": "pillow",
        }

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
                self._sanitize_shell_command(command, workspace_path) for command in commands
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
        paths: list[str] = []
        for spec in artifacts:
            if not spec.path or not spec.must_exist:
                continue
            normalized = self._normalize_expected_path(spec.path, workspace_path)
            if normalized and normalized not in paths:
                paths.append(normalized)
        return paths

    def _normalize_expected_path(self, raw_path: str, workspace_path: Path) -> str | None:
        cleaned = raw_path.strip().strip("`\"'")
        if not cleaned:
            return None
        cleaned = cleaned.replace("\\", "/").strip("/")
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if not parts:
            return None
        if parts[0] == "workspace":
            parts = parts[1:]
        workspace_name = workspace_path.name
        if parts and parts[0] == workspace_name:
            parts = parts[1:]
        if not parts:
            return None
        return "/".join(parts)

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
        if not text:
            return None
        matches = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", text, flags=re.DOTALL)
        for block in matches:
            candidate = block.strip("\n")
            if len(candidate) < 4:
                continue
            return candidate + "\n"
        return None

    def _learning_notes_path(self, workspace_path: Path) -> Path:
        return workspace_path / "knowledge" / "codex_learning_notes.md"

    def _workspace_snapshot_markdown_path(self, workspace_path: Path) -> Path:
        return workspace_path / ".agent" / "workspace_snapshot.md"

    def _load_learning_notes(self, workspace_path: Path) -> str:
        path = self._learning_notes_path(workspace_path)
        if not path.exists():
            return ""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return ""
        if not content:
            return ""
        lines = content.splitlines()
        return "\n".join(lines[-60:])

    def _load_workspace_snapshot_summary(self, workspace_path: Path) -> str:
        path = self._workspace_snapshot_markdown_path(workspace_path)
        if not path.exists():
            return ""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return ""
        if not content:
            return ""
        lines = content.splitlines()
        return "\n".join(lines[:80])

    def _inject_workspace_snapshot(self, base_prompt: str, workspace_path: Path) -> str:
        snapshot = self._load_workspace_snapshot_summary(workspace_path)
        if not snapshot:
            return base_prompt
        header = "Current workspace snapshot (authoritative paths):"
        return f"{header}\n\n{snapshot}\n\nTask prompt:\n{base_prompt}"

    def _inject_learning_notes(self, base_prompt: str, workspace_path: Path) -> str:
        notes = self._load_learning_notes(workspace_path)
        if not notes:
            return base_prompt
        header = "Persistent execution notes (learned from previous codex runs):"
        return f"{header}\n\n{notes}\n\nTask prompt:\n{base_prompt}"

    def _inject_skill_context(
        self,
        *,
        base_prompt: str,
        workspace_path: Path,
        step: PlannerStep,
    ) -> tuple[str, list[str]]:
        skill_paths = [str(item).strip() for item in step.skill_paths if str(item).strip()]
        if not skill_paths:
            return base_prompt, []
        loaded_chunks: list[str] = []
        used_paths: list[str] = []
        for raw_path in skill_paths[:3]:
            resolved = self._resolve_skill_path(raw_path, workspace_path)
            if resolved is None or not resolved.exists():
                continue
            try:
                content = resolved.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError:
                continue
            if not content:
                continue
            excerpt = "\n".join(content.splitlines()[:80]).strip()
            if not excerpt:
                continue
            loaded_chunks.append(f"### {resolved.parent.name} ({raw_path})\n{excerpt}")
            used_paths.append(raw_path)
        if not loaded_chunks:
            return base_prompt, []
        header = "Selected skill context (apply these instructions before coding):"
        return f"{header}\n\n" + "\n\n".join(loaded_chunks) + f"\n\nTask prompt:\n{base_prompt}", used_paths

    def _resolve_skill_path(self, raw_path: str, workspace_path: Path) -> Path | None:
        cleaned = raw_path.strip().strip("`\"'")
        if not cleaned:
            return None
        candidate = Path(cleaned)
        if candidate.is_absolute():
            return candidate
        workspace_candidate = workspace_path / cleaned
        if workspace_candidate.exists():
            return workspace_candidate
        if cleaned.startswith("knowledge/"):
            direct_workspace = workspace_path / cleaned
            if direct_workspace.exists():
                return direct_workspace
        codex_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
        codex_candidate = codex_home / cleaned
        if codex_candidate.exists():
            return codex_candidate
        if not cleaned.endswith("SKILL.md"):
            workspace_matches = sorted((workspace_path / "knowledge" / "skills").glob(f"**/{cleaned}/SKILL.md"))
            if workspace_matches:
                return workspace_matches[0]
            codex_matches = sorted((codex_home / "skills").glob(f"**/{cleaned}/SKILL.md"))
            if codex_matches:
                return codex_matches[0]
        return None

    def _append_learning_note(
        self,
        *,
        workspace_path: Path,
        step: PlannerStep,
        result: StepExecutionResult,
    ) -> None:
        notes_path = self._learning_notes_path(workspace_path)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        summary = result.summary.replace("\n", " ").strip()
        if len(summary) > 240:
            summary = f"{summary[:240]}..."
        marker = " soft_failure=true" if self._result_has_codex_soft_failure(result) else ""
        line = (
            f"- ts={timestamp} step={step.id} action={step.action} status={result.status}"
            f"{marker} summary={summary}"
        )
        with notes_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def _result_has_codex_soft_failure(self, result: StepExecutionResult) -> bool:
        text = f"{result.stdout_text}\n{result.stderr_text}".lower()
        return any(marker in text for marker in self._codex_soft_failure_markers)

    def _normalize_codex_command(self, base_cmd: list[str]) -> list[str]:
        if not base_cmd:
            return ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"]
        binary = os.path.basename(base_cmd[0])
        if binary != "codex":
            return base_cmd

        known_subcommands = {"exec", "login", "logout", "completion", "mcp", "sandbox", "help"}
        tail = base_cmd[1:]
        if not tail:
            return [base_cmd[0], "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"]

        subcommand = None
        for token in tail:
            if token.startswith("-"):
                continue
            subcommand = token
            break

        if subcommand in known_subcommands:
            if subcommand == "exec":
                normalized = list(base_cmd)
                model = self.settings.codex_model.strip()
                has_model_flag = any(
                    token in {"-m", "--model"} or token.startswith("--model=")
                    for token in normalized
                )
                has_danger_flag = "--dangerously-bypass-approvals-and-sandbox" in normalized
                has_full_auto = "--full-auto" in normalized
                has_sandbox_flag = (
                    "--sandbox" in normalized
                    or "-s" in normalized
                    or any(token.startswith("--sandbox=") for token in normalized)
                )
                if model and not has_model_flag:
                    normalized.extend(["--model", model])
                if has_full_auto and not has_danger_flag and not has_sandbox_flag:
                    normalized = [token for token in normalized if token != "--full-auto"]
                    normalized.append("--dangerously-bypass-approvals-and-sandbox")
                elif not has_full_auto and not has_danger_flag and not has_sandbox_flag:
                    normalized.append("--dangerously-bypass-approvals-and-sandbox")
                if "--skip-git-repo-check" not in normalized:
                    normalized.append("--skip-git-repo-check")
                return normalized
            return base_cmd

        # Force non-interactive mode for orchestrator runs.
        normalized = [base_cmd[0], "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", *tail]
        model = self.settings.codex_model.strip()
        has_model_flag = any(
            token in {"-m", "--model"} or token.startswith("--model=")
            for token in normalized
        )
        if model and not has_model_flag:
            normalized.extend(["--model", model])
        return normalized

    async def _run_commands(
        self,
        run_id: str,
        step: PlannerStep,
        commands: list[str],
        workspace_path: Path,
        run_path: Path,
    ) -> StepExecutionResult:
        started = time.monotonic()
        step_id = step.id
        chunks: list[dict[str, Any]] = []
        for idx, command in enumerate(commands):
            command_step_id = step_id if len(commands) == 1 else f"{step_id}-{idx+1}"
            result = await self._run_command(
                run_id=run_id,
                step_id=command_step_id,
                command=command,
                workspace_path=workspace_path,
                run_path=run_path,
                stdin_payload=None,
                step=step,
            )
            if result.status != "completed":
                recovered = await self._attempt_missing_module_recovery(
                    run_id=run_id,
                    step_id=step_id,
                    command=command,
                    base_step_idx=idx + 1,
                    workspace_path=workspace_path,
                    run_path=run_path,
                    failed_result=result,
                )
                if recovered is not None:
                    install_result, retry_result = recovered
                    chunks.append(
                        {
                            "command": install_result.command,
                            "status": install_result.status,
                            "exit_code": install_result.exit_code,
                            "stdout": install_result.stdout_text,
                            "stderr": install_result.stderr_text,
                            "duration_ms": install_result.duration_ms,
                        }
                    )
                    chunks.append(
                        {
                            "command": f"{command} (retry)",
                            "status": retry_result.status,
                            "exit_code": retry_result.exit_code,
                            "stdout": retry_result.stdout_text,
                            "stderr": retry_result.stderr_text,
                            "duration_ms": retry_result.duration_ms,
                        }
                    )
                    result = retry_result

            if result.status != "completed":
                argparse_recovery_attempts = await self._attempt_argparse_recovery(
                    run_id=run_id,
                    step_id=step_id,
                    command=command,
                    base_step_idx=idx + 1,
                    workspace_path=workspace_path,
                    run_path=run_path,
                    failed_result=result,
                )
                if argparse_recovery_attempts:
                    for label, recovered_result in argparse_recovery_attempts:
                        chunks.append(
                            {
                                "command": label,
                                "status": recovered_result.status,
                                "exit_code": recovered_result.exit_code,
                                "stdout": recovered_result.stdout_text,
                                "stderr": recovered_result.stderr_text,
                                "duration_ms": recovered_result.duration_ms,
                            }
                        )
                    result = argparse_recovery_attempts[-1][1]

            if self._is_non_fatal_search_failure(step=step, command=command, result=result):
                result = StepExecutionResult(
                    status="completed",
                    exit_code=0,
                    summary="search command returned no matches (accepted for check step)",
                    stdout_text=result.stdout_text,
                    stderr_text=result.stderr_text,
                    duration_ms=result.duration_ms,
                    command=result.command or command,
                    diff_stats=result.diff_stats,
                    files_changed=list(result.files_changed),
                    stdout_path=result.stdout_path,
                    stderr_path=result.stderr_path,
                    log_path=result.log_path,
                    is_infra_error=False,
                    missing_artifact=None,
                )

            chunks.append(
                {
                    "command": command,
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "stdout": result.stdout_text,
                    "stderr": result.stderr_text,
                    "duration_ms": result.duration_ms,
                }
            )
            if result.status != "completed":
                total_ms = int((time.monotonic() - started) * 1000)
                stderr_text = "\n".join(chunk["stderr"] for chunk in chunks if chunk["stderr"]).strip()
                if not stderr_text:
                    stderr_text = f"exit code {result.exit_code} with no stderr output"
                failure_summary = result.summary if result.status == "timeout" else f"command failed: {command}"
                return StepExecutionResult(
                    status=result.status,
                    exit_code=result.exit_code,
                    summary=failure_summary,
                    stdout_text="\n".join(chunk["stdout"] for chunk in chunks if chunk["stdout"]),
                    stderr_text=stderr_text,
                    duration_ms=total_ms,
                    command=" && ".join(commands),
                    errors=[failure_summary],
                    is_infra_error=result.is_infra_error,
                    missing_artifact=result.missing_artifact,
                )

        diff_stats, files_changed = await self._git_observation(workspace_path)
        files_changed = self._merge_output_paths_into_files_changed(
            files_changed=files_changed,
            stdout_text="\n".join(chunk["stdout"] for chunk in chunks if chunk["stdout"]),
            stderr_text="\n".join(chunk["stderr"] for chunk in chunks if chunk["stderr"]),
            workspace_path=workspace_path,
        )
        total_ms = int((time.monotonic() - started) * 1000)
        summary = f"executed {len(commands)} shell command(s) successfully"
        stdout_path = result.stdout_path if len(commands) == 1 else None
        stderr_path = result.stderr_path if len(commands) == 1 else None
        return StepExecutionResult(
            status="completed",
            exit_code=0,
            summary=summary,
            stdout_text="\n".join(chunk["stdout"] for chunk in chunks if chunk["stdout"]),
            stderr_text="\n".join(chunk["stderr"] for chunk in chunks if chunk["stderr"]),
            duration_ms=total_ms,
            command=" && ".join(commands),
            diff_stats=diff_stats,
            files_changed=files_changed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_path=self._write_json_log(run_path, step_id, {"commands": chunks, "summary": summary}),
            is_infra_error=False,
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
        if not self.settings.auto_install_missing_python_modules:
            return None
        if failed_result.status == "completed":
            return None
        if not self._looks_like_python_command(command):
            return None

        module_name = self._missing_module_name(failed_result.stderr_text)
        if not module_name:
            return None
        if self._is_local_module_reference(module_name, workspace_path):
            return None

        pip_packages = self._package_for_module(module_name)
        python_bin = "python3" if command.strip().startswith("python3") else "python"
        install_cmd = f"{python_bin} -m pip install {pip_packages}"
        install_result = await self._run_command(
            run_id=run_id,
            step_id=f"{step_id}-{base_step_idx}-autoinstall",
            command=install_cmd,
            workspace_path=workspace_path,
            run_path=run_path,
            stdin_payload=None,
        )
        if install_result.status != "completed":
            return install_result, failed_result

        retry_result = await self._run_command(
            run_id=run_id,
            step_id=f"{step_id}-{base_step_idx}-retry",
            command=command,
            workspace_path=workspace_path,
            run_path=run_path,
            stdin_payload=None,
        )
        if retry_result.status == "completed":
            retry_result.summary = (
                f"command completed after auto-install of missing module '{module_name}' ({pip_packages})"
            )
        return install_result, retry_result

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
        if failed_result.status == "completed":
            return []
        if not self._looks_like_python_command(command):
            return []

        unknown_args = self._extract_unknown_args(failed_result.stderr_text)
        if not unknown_args:
            return []

        attempts: list[tuple[str, StepExecutionResult]] = []

        normalized_cmd = self._replace_underscored_flags(command, unknown_args)
        if normalized_cmd and normalized_cmd != command:
            normalized_result = await self._run_command(
                run_id=run_id,
                step_id=f"{step_id}-{base_step_idx}-argfix",
                command=normalized_cmd,
                workspace_path=workspace_path,
                run_path=run_path,
                stdin_payload=None,
            )
            attempts.append((f"{normalized_cmd} (argfix)", normalized_result))
            if normalized_result.status == "completed":
                normalized_result.summary = "command completed after argparse flag normalization"
                return attempts

        pruned_cmd = self._remove_unknown_args(command, unknown_args)
        if pruned_cmd and pruned_cmd not in {command, normalized_cmd}:
            pruned_result = await self._run_command(
                run_id=run_id,
                step_id=f"{step_id}-{base_step_idx}-argdrop",
                command=pruned_cmd,
                workspace_path=workspace_path,
                run_path=run_path,
                stdin_payload=None,
            )
            attempts.append((f"{pruned_cmd} (argdrop)", pruned_result))
            if pruned_result.status == "completed":
                pruned_result.summary = "command completed after removing unsupported argparse options"
        return attempts

    def _looks_like_python_command(self, command: str) -> bool:
        stripped = command.strip()
        return stripped.startswith("python ") or stripped.startswith("python3 ")

    def _missing_module_name(self, stderr_text: str) -> str | None:
        match = self._missing_module_re.search(stderr_text or "")
        if not match:
            return None
        return match.group(1)

    def _is_local_module_reference(self, module_name: str, workspace_path: Path) -> bool:
        root = (module_name or "").split(".", 1)[0].strip()
        if not root:
            return False
        if root in {"src", "app", "workspace"}:
            return True
        candidates = [
            workspace_path / root,
            workspace_path / f"{root}.py",
            workspace_path / "src" / root,
            workspace_path / "src" / f"{root}.py",
        ]
        return any(path.exists() for path in candidates)

    def _package_for_module(self, module_name: str) -> str:
        package = self._python_missing_module_map.get(module_name)
        if package:
            return package
        if "." in module_name:
            root = module_name.split(".", 1)[0]
            return self._python_missing_module_map.get(root, root)
        return module_name

    def _extract_unknown_args(self, stderr_text: str) -> list[str]:
        match = self._argparse_unknown_re.search(stderr_text or "")
        if not match:
            return []
        payload = match.group(1).strip()
        try:
            tokens = shlex.split(payload)
        except ValueError:
            tokens = payload.split()
        return [token for token in tokens if token.startswith("-")]

    def _replace_underscored_flags(self, command: str, unknown_args: list[str]) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        unknown = set(unknown_args)
        changed = False
        replaced: list[str] = []
        for token in tokens:
            if token in unknown and token.startswith("--") and "_" in token:
                replaced.append(token.replace("_", "-"))
                changed = True
            else:
                replaced.append(token)
        if not changed:
            return None
        return " ".join(shlex.quote(token) for token in replaced)

    def _remove_unknown_args(self, command: str, unknown_args: list[str]) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        unknown = set(unknown_args)
        filtered = [token for token in tokens if token not in unknown]
        if len(filtered) == len(tokens):
            return None
        return " ".join(shlex.quote(token) for token in filtered)

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
        command = self._sanitize_shell_command(command, workspace_path)
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
        if result.status == "completed":
            return False
        if result.exit_code != 1:
            return False
        if step.step_type != "check":
            return False
        search_bin = self._shell_primary_binary(command)
        if search_bin not in {"rg", "grep"}:
            return False
        stderr = (result.stderr_text or "").strip().lower()
        if stderr and "no matches found" not in stderr:
            return False
        return True

    def _shell_primary_binary(self, command: str) -> str:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token):
                continue
            return Path(token).name.lower()
        return ""

    def _normalize_python_runtime_command(self, command: str, workspace_path: Path) -> str:
        if not command:
            return command
        if any(op in command for op in ("&&", "||", ";", "|")):
            return command
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command
        if not tokens:
            return command

        py_idx = self._python_token_index(tokens)
        if py_idx is None:
            return command

        self._normalize_smoke_flag(tokens, py_idx)
        script_idx = py_idx + 1
        if script_idx >= len(tokens):
            return " ".join(shlex.quote(token) for token in tokens)
        script_token = tokens[script_idx]
        if script_token.startswith("-"):
            return " ".join(shlex.quote(token) for token in tokens)
        if script_token == "-m":
            return " ".join(shlex.quote(token) for token in tokens)
        if not script_token.endswith(".py"):
            return " ".join(shlex.quote(token) for token in tokens)

        resolved = self._resolve_python_script_token(script_token, workspace_path)
        if resolved:
            tokens[script_idx] = resolved
        self._maybe_switch_script_to_module(
            tokens=tokens,
            script_idx=script_idx,
            workspace_path=workspace_path,
        )
        return " ".join(shlex.quote(token) for token in tokens)

    def _maybe_switch_script_to_module(
        self,
        *,
        tokens: list[str],
        script_idx: int,
        workspace_path: Path,
    ) -> None:
        if script_idx >= len(tokens):
            return
        script_token = tokens[script_idx]
        if script_token.startswith("-"):
            return
        script_path = Path(script_token)
        if script_path.is_absolute():
            try:
                rel_path = script_path.resolve().relative_to(workspace_path.resolve())
            except ValueError:
                return
        else:
            rel_path = script_path

        if rel_path.suffix != ".py" or len(rel_path.parts) < 2:
            return
        module_root = rel_path.parts[0]
        module_name = ".".join(rel_path.with_suffix("").parts)
        source_path = workspace_path / rel_path
        if not source_path.exists():
            return
        if not self._script_uses_module_root(source_path, module_root):
            return

        # Run as module to keep workspace root on sys.path.
        tokens[script_idx : script_idx + 1] = ["-m", module_name]

    def _script_uses_module_root(self, script_path: Path, module_root: str) -> bool:
        pattern = re.compile(rf"(^|\n)\s*(from|import)\s+{re.escape(module_root)}(\.|\\b)")
        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return bool(pattern.search(text))

    def _python_token_index(self, tokens: list[str]) -> int | None:
        for idx, token in enumerate(tokens):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token):
                continue
            binary = Path(token).name.lower()
            if binary in {"python", "python3"}:
                return idx
            return None
        return None

    def _normalize_smoke_flag(self, tokens: list[str], python_idx: int) -> None:
        if "--mode" in tokens:
            return
        normalized: list[str] = []
        replaced = False
        for idx, token in enumerate(tokens):
            if idx <= python_idx:
                normalized.append(token)
                continue
            if token in {"--smoke_test", "--smoke-test"}:
                if not replaced:
                    normalized.extend(["--mode", "smoke"])
                    replaced = True
                continue
            normalized.append(token)
        if replaced:
            tokens[:] = normalized

    def _resolve_python_script_token(self, script_token: str, workspace_path: Path) -> str | None:
        script_path = Path(script_token)
        if script_path.is_absolute():
            if script_path.exists():
                return script_token
        else:
            candidate = (workspace_path / script_path).resolve()
            if candidate.exists():
                return script_token

        fallback = self._find_python_script_candidate(script_path.name, workspace_path)
        if fallback is None:
            return None
        try:
            return fallback.relative_to(workspace_path.resolve()).as_posix()
        except ValueError:
            return fallback.as_posix()

    def _find_python_script_candidate(self, filename: str, workspace_path: Path) -> Path | None:
        root = workspace_path.resolve()
        exact_matches = [
            path for path in root.rglob(filename)
            if path.is_file() and path.suffix == ".py"
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            exact_matches.sort(key=lambda p: len(p.as_posix()))
            return exact_matches[0]

        target_stem = re.sub(r"[^a-z0-9]", "", Path(filename).stem.lower())
        if not target_stem:
            return None
        fuzzy: list[Path] = []
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            stem = re.sub(r"[^a-z0-9]", "", path.stem.lower())
            if stem == target_stem:
                fuzzy.append(path)
        if len(fuzzy) == 1:
            return fuzzy[0]
        if len(fuzzy) > 1:
            fuzzy.sort(key=lambda p: len(p.as_posix()))
            return fuzzy[0]
        return None

    def _sanitize_shell_command(self, command: str, workspace_path: Path) -> str:
        if not command:
            return command

        workspace_name = workspace_path.name
        workspace_abs = workspace_path.resolve().as_posix().rstrip("/")
        if not workspace_name:
            return command

        sanitized = command
        # Collapse duplicated workspace prefixes that are commonly produced by mixed planner prompts.
        # Keep results within the current cwd (workspace_path).
        sanitized = sanitized.replace(f"{workspace_abs}/", "./")
        sanitized = sanitized.replace(f"{workspace_abs}", ".")

        # Convert explicit workspace root references to the current working directory.
        workspace_root_token = re.escape(workspace_name)
        sanitized = re.sub(
            rf"(^|[\s\"'\(\[])(?:\./)?workspace/{workspace_root_token}(?=\s|/|$|[;&|)])",
            lambda m: f"{m.group(1)}.",
            sanitized,
        )
        return sanitized

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
        start = time.monotonic()
        stdout_path_obj = run_path / f"{step_id}.stdout.log"
        stderr_path_obj = run_path / f"{step_id}.stderr.log"
        stdout_path_obj.write_text("", encoding="utf-8")
        stderr_path_obj.write_text("", encoding="utf-8")
        timeout_profile = self._timeout_profile_for_step(step=step, command=command or " ".join(argv))
        last_output_at = start
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace_path.as_posix(),
            env=self._sanitized_env(),
            stdin=asyncio.subprocess.PIPE if stdin_payload is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._running[run_id] = process
        stdout_chunks = bytearray()
        stderr_chunks = bytearray()

        async def _pump_stream(
            stream: asyncio.StreamReader | None,
            target_path: Path,
            buffer: bytearray,
        ) -> None:
            nonlocal last_output_at
            if stream is None:
                return
            with target_path.open("ab") as handle:
                while True:
                    chunk = await stream.read(4096)
                    if not chunk:
                        break
                    buffer.extend(chunk)
                    last_output_at = time.monotonic()
                    handle.write(chunk)
                    handle.flush()

        stdout_task = asyncio.create_task(_pump_stream(process.stdout, stdout_path_obj, stdout_chunks))
        stderr_task = asyncio.create_task(_pump_stream(process.stderr, stderr_path_obj, stderr_chunks))
        try:
            if stdin_payload and process.stdin is not None:
                process.stdin.write(stdin_payload.encode("utf-8"))
                await process.stdin.drain()
                process.stdin.close()
            if timeout_profile.hard_timeout_sec is not None:
                await asyncio.wait_for(process.wait(), timeout=timeout_profile.hard_timeout_sec)
            else:
                await self._wait_with_idle_watchdog(
                    process=process,
                    start_time=start,
                    timeout_profile=timeout_profile,
                    last_output_at_ref=lambda: last_output_at,
                )
            await asyncio.gather(stdout_task, stderr_task)
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout_text = self._truncate(stdout_chunks.decode("utf-8", errors="ignore"))
            stderr_text = self._truncate(stderr_chunks.decode("utf-8", errors="ignore"))
            status = "completed" if process.returncode == 0 else "failed"
            diff_stats, files_changed = await self._git_observation(workspace_path)
            files_changed = self._merge_output_paths_into_files_changed(
                files_changed=files_changed,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                workspace_path=workspace_path,
            )
            summary = "command completed" if status == "completed" else "command failed"
            missing_artifact = None
            if status != "completed":
                missing_artifact = self._extract_missing_python_file_path(stderr_text)
                if missing_artifact:
                    stderr_text = f"MISSING_FILE:{missing_artifact}\n{stderr_text}"
                    stderr_path_obj.write_text(stderr_text, encoding="utf-8")
            return StepExecutionResult(
                status=status,
                exit_code=int(process.returncode or 0),
                summary=summary,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                duration_ms=duration_ms,
                command=command or " ".join(argv),
                diff_stats=diff_stats,
                files_changed=files_changed,
                stdout_path=stdout_path_obj.as_posix(),
                stderr_path=stderr_path_obj.as_posix(),
                errors=[] if status == "completed" else [stderr_text[:300] or "command failed"],
                is_infra_error=False,
                missing_artifact=missing_artifact,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout_text = self._truncate(stdout_chunks.decode("utf-8", errors="ignore"))
            stderr_text = "process timeout"
            stderr_path_obj.write_text(stderr_text, encoding="utf-8")
            return StepExecutionResult(
                status="timeout",
                exit_code=124,
                summary="process timeout",
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                duration_ms=duration_ms,
                command=command or " ".join(argv),
                stdout_path=stdout_path_obj.as_posix(),
                stderr_path=stderr_path_obj.as_posix(),
                errors=["process timeout"],
                is_infra_error=True,
                missing_artifact=None,
            )
        except _IdleProcessTimeout as exc:
            process.kill()
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout_text = self._truncate(stdout_chunks.decode("utf-8", errors="ignore"))
            stderr_text = str(exc)
            stderr_path_obj.write_text(stderr_text, encoding="utf-8")
            return StepExecutionResult(
                status="timeout",
                exit_code=124,
                summary=stderr_text,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                duration_ms=duration_ms,
                command=command or " ".join(argv),
                stdout_path=stdout_path_obj.as_posix(),
                stderr_path=stderr_path_obj.as_posix(),
                errors=[stderr_text],
                is_infra_error=True,
                missing_artifact=None,
            )
        except Exception as exc:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            duration_ms = int((time.monotonic() - start) * 1000)
            stderr_text = f"runner exception: {exc}"
            stderr_path_obj.write_text(stderr_text, encoding="utf-8")
            return StepExecutionResult(
                status="failed",
                exit_code=1,
                summary="runner exception",
                stdout_text=self._truncate(stdout_chunks.decode("utf-8", errors="ignore")),
                stderr_text=stderr_text,
                duration_ms=duration_ms,
                command=command or " ".join(argv),
                stdout_path=stdout_path_obj.as_posix(),
                stderr_path=stderr_path_obj.as_posix(),
                errors=[stderr_text],
                is_infra_error=True,
                missing_artifact=None,
            )
        finally:
            self._running.pop(run_id, None)

    async def _wait_with_idle_watchdog(
        self,
        *,
        process: asyncio.subprocess.Process,
        start_time: float,
        timeout_profile: TimeoutProfile,
        last_output_at_ref: Any,
    ) -> None:
        while True:
            if process.returncode is not None:
                await process.wait()
                return
            await asyncio.sleep(1.0)
            now = time.monotonic()
            max_wall_clock = timeout_profile.max_wall_clock_sec or 0
            if max_wall_clock > 0 and (now - start_time) > max_wall_clock:
                raise _IdleProcessTimeout(
                    f"process max wall-clock timeout ({max_wall_clock}s) [{timeout_profile.label}]"
                )
            idle_timeout = timeout_profile.idle_timeout_sec or 0
            if idle_timeout > 0 and (now - float(last_output_at_ref())) > idle_timeout:
                raise _IdleProcessTimeout(
                    f"process idle timeout ({idle_timeout}s without output) [{timeout_profile.label}]"
                )

    def _timeout_profile_for_step(self, *, step: PlannerStep | None, command: str) -> TimeoutProfile:
        if self._is_long_running_training_step(step=step, command=command):
            return TimeoutProfile(
                hard_timeout_sec=None,
                idle_timeout_sec=self.settings.training_idle_timeout_sec or None,
                max_wall_clock_sec=self.settings.training_max_wall_clock_sec or None,
                label="training-idle-watchdog",
            )
        return TimeoutProfile(
            hard_timeout_sec=self.settings.codex_step_timeout_sec,
            idle_timeout_sec=None,
            max_wall_clock_sec=None,
            label="hard-timeout",
        )

    def _is_long_running_training_step(self, *, step: PlannerStep | None, command: str) -> bool:
        if step is None or step.action != "shell":
            return False
        step_intent = getattr(step.step_intent, "value", step.step_intent)
        operation = getattr(step.operation, "value", step.operation)
        if step_intent == StepIntent.run_training.value or operation == StepIntent.run_training.value:
            return True
        blob = " ".join(
            [
                str(step.title or ""),
                str(step.instruction or ""),
                str(command or ""),
            ]
        ).lower()
        markers = (" training", "train ", "--epochs", "epoch", "torchrun", "accelerate launch", " fit(")
        return any(marker in blob for marker in markers)

    def _extract_missing_python_file_path(self, text: str) -> str | None:
        match = self._missing_python_file_re.search(text or "")
        if not match:
            return None
        return match.group(1).strip()

    async def _git_observation(self, workspace_path: Path) -> tuple[str, list[str]]:
        diff_stats = ""
        files_changed: list[str] = []
        diff_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            workspace_path.as_posix(),
            "diff",
            "--stat",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await diff_proc.communicate()
        if diff_proc.returncode == 0:
            diff_stats = self._truncate(stdout.decode("utf-8", errors="ignore")).strip()

        status_proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            workspace_path.as_posix(),
            "status",
            "--porcelain",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        status_out, _ = await status_proc.communicate()
        if status_proc.returncode == 0:
            lines = status_out.decode("utf-8", errors="ignore").splitlines()
            for line in lines:
                if len(line) > 3:
                    files_changed.append(line[3:].strip())
        return diff_stats, files_changed

    def _merge_output_paths_into_files_changed(
        self,
        *,
        files_changed: list[str],
        stdout_text: str,
        stderr_text: str,
        workspace_path: Path,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for raw in files_changed:
            normalized = self._normalize_expected_path(raw, workspace_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)

        for normalized in self._extract_output_paths_from_text(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            workspace_path=workspace_path,
        ):
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    def _extract_output_paths_from_text(
        self,
        *,
        stdout_text: str,
        stderr_text: str,
        workspace_path: Path,
    ) -> list[str]:
        text = "\n".join([stdout_text or "", stderr_text or ""])
        matches = re.findall(
            r"(?im)^\s*([A-Za-z_][A-Za-z0-9_]*(?:_path|_file|_dir))\s*=\s*([^\n]+?)\s*$",
            text,
        )
        results: list[str] = []
        seen: set[str] = set()
        for _, raw_path in matches:
            candidate = str(raw_path).strip().strip("`\"'")
            if not candidate:
                continue
            candidate = candidate.split()[0].strip()
            normalized = self._normalize_expected_path(candidate, workspace_path)
            if normalized and normalized not in seen:
                seen.add(normalized)
                results.append(normalized)
                continue
            path = Path(candidate)
            if path.is_absolute():
                try:
                    rel = path.resolve().relative_to(workspace_path.resolve()).as_posix()
                except ValueError:
                    continue
                if rel not in seen:
                    seen.add(rel)
                    results.append(rel)
        return results

    def _write_stream_logs(self, run_path: Path, step_id: str, stdout_text: str, stderr_text: str) -> tuple[str, str]:
        stdout_path = run_path / f"{step_id}.stdout.log"
        stderr_path = run_path / f"{step_id}.stderr.log"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        return stdout_path.as_posix(), stderr_path.as_posix()

    def _write_json_log(self, run_path: Path, step_id: str, payload: dict[str, Any]) -> str:
        log_path = run_path / f"{step_id}.executor.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return log_path.as_posix()

    def _write_stepio_result(
        self,
        run_id: str,
        step: PlannerStep,
        result: StepExecutionResult,
        workspace_path: Path,
        run_path: Path,
    ) -> str:
        step_status: str
        if result.auto_repaired and result.status == "completed":
            step_status = "auto_repaired"
        elif result.status == "timeout":
            step_status = "timeout"
        elif result.status == "completed":
            step_status = "completed"
        else:
            step_status = "failed"

        artifacts = self._collect_stepio_artifacts(step, result, workspace_path)
        payload = StepIOResult(
            run_id=run_id,
            step_id=step.id,
            status=step_status,
            error_code=self._stepio_error_code(result),
            summary=result.summary,
            operation=step.operation,
            intent="check" if step.step_type == "check" else "change",
            inputs=step.inputs,
            expected_outputs=step.expected_outputs,
            artifacts_produced=artifacts,
            metrics=self._extract_stepio_metrics(step=step, result=result),
            hyperparameters=self._extract_hyperparameters_from_command(result.command),
            duration_ms=result.duration_ms,
            command=result.command,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            log_path=result.log_path,
        )
        output_path = self._next_stepio_output_path(run_path, step.id)
        output_path.write_text(json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2), encoding="utf-8")
        return output_path.as_posix()

    def _next_stepio_output_path(self, run_path: Path, step_id: str) -> Path:
        primary = run_path / f"{step_id}.step_result.json"
        if not primary.exists():
            return primary
        index = 2
        while True:
            candidate = run_path / f"{step_id}.step_result.{index}.json"
            if not candidate.exists():
                return candidate
            index += 1

    def _stepio_error_code(self, result: StepExecutionResult) -> str:
        if result.status == "completed":
            return "none"
        if result.status == "timeout" or result.is_infra_error:
            return "infra_error"
        if result.missing_artifact:
            return "missing_file"
        stderr = (result.stderr_text or "").lower()
        if "unrecognized arguments" in stderr:
            return "arg_error"
        return "execution_error"

    def _collect_stepio_artifacts(
        self,
        step: PlannerStep,
        result: StepExecutionResult,
        workspace_path: Path,
    ) -> list[StepArtifactManifest]:
        produced: list[StepArtifactManifest] = []
        expected_kind_by_path: dict[str, ArtifactKind] = {}
        for spec in (step.expected_outputs.artifacts or []) + (step.expected_artifacts or []):
            if not spec.path:
                continue
            normalized = self._normalize_expected_path(spec.path, workspace_path)
            if not normalized:
                continue
            expected_kind_by_path[normalized] = spec.kind
        candidate_paths: list[str] = []
        for rel in result.files_changed or []:
            normalized = self._normalize_expected_path(rel, workspace_path)
            if normalized and normalized not in candidate_paths:
                candidate_paths.append(normalized)
        for rel in expected_kind_by_path.keys():
            if rel not in candidate_paths:
                candidate_paths.append(rel)

        for rel in candidate_paths:
            abs_path = (workspace_path / rel).resolve()
            workspace_root = workspace_path.resolve()
            try:
                if not abs_path.is_relative_to(workspace_root):
                    continue
            except ValueError:
                continue
            exists = abs_path.exists()
            size: int | None = None
            sha256: str | None = None
            if exists and abs_path.is_file():
                try:
                    size = abs_path.stat().st_size
                    sha256 = self._sha256_file(abs_path)
                except OSError:
                    size = None
                    sha256 = None
            produced.append(
                StepArtifactManifest(
                    path=rel,
                    kind=expected_kind_by_path.get(rel, ArtifactKind.file),
                    exists=exists,
                    size=size,
                    sha256=sha256,
                )
            )
        return produced

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _extract_metrics_from_text(self, text: str) -> dict[str, float | int | str | bool]:
        return extract_metrics_from_text(text)

    def _extract_stepio_metrics(
        self,
        *,
        step: PlannerStep,
        result: StepExecutionResult,
    ) -> dict[str, float | int | str | bool]:
        intent = self._normalize_step_intent(step.step_intent)
        if step.action == "codex" and intent in {
            StepIntent.create_file,
            StepIntent.modify_file,
            StepIntent.general,
        }:
            return {}
        return self._extract_metrics_from_text(f"{result.stdout_text}\n{result.stderr_text}")

    def _extract_hyperparameters_from_command(self, command: str | None) -> dict[str, Any]:
        if not command:
            return {}
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if not tokens:
            return {}
        if Path(tokens[0]).name.lower() == "codex":
            return {}
        aliases = {
            "epochs": "epochs",
            "epoch": "epochs",
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "learning-rate": "learning_rate",
            "batch_size": "batch_size",
            "batch-size": "batch_size",
            "bs": "batch_size",
            "optimizer": "optimizer",
            "optim": "optimizer",
            "weight_decay": "weight_decay",
            "weight-decay": "weight_decay",
            "wd": "weight_decay",
            "momentum": "momentum",
            "dropout": "dropout",
            "seed": "seed",
            "workers": "workers",
            "num_workers": "workers",
            "num-workers": "workers",
            "mode": "mode",
        }
        parsed: dict[str, Any] = {}
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--"):
                key_raw = token[2:]
                value: str | None = None
                if "=" in key_raw:
                    key_raw, value = key_raw.split("=", 1)
                elif idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                    value = tokens[idx + 1]
                    idx += 1
                else:
                    value = "true"
                normalized = aliases.get(key_raw.strip().lower())
                if normalized:
                    parsed[normalized] = self._coerce_value(value)
            idx += 1
        return parsed

    def _coerce_value(self, value: str | None) -> Any:
        if value is None:
            return None
        raw = str(value).strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"[-+]?\d+", raw):
            try:
                return int(raw)
            except ValueError:
                return raw
        if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", raw) or re.fullmatch(
            r"[-+]?\d+(?:[eE][-+]?\d+)", raw
        ):
            try:
                return float(raw)
            except ValueError:
                return raw
        return raw

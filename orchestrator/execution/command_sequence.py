from __future__ import annotations

import time
from pathlib import Path
from typing import Awaitable, Callable

from orchestrator.execution.runner_models import StepExecutionResult
from orchestrator.persistence.schemas import PlannerStep


class CommandSequenceSupport:
    def __init__(
        self,
        *,
        run_command: Callable[..., Awaitable[StepExecutionResult]],
        attempt_missing_module_recovery: Callable[..., Awaitable[tuple[StepExecutionResult, StepExecutionResult] | None]],
        attempt_argparse_recovery: Callable[..., Awaitable[list[tuple[str, StepExecutionResult]]]],
        is_non_fatal_search_failure: Callable[..., bool],
        git_observation: Callable[[Path], Awaitable[tuple[str, list[str]]]],
        merge_output_paths_into_files_changed: Callable[..., list[str]],
        write_json_log: Callable[[Path, str, dict], str],
    ) -> None:
        self.run_command = run_command
        self.attempt_missing_module_recovery = attempt_missing_module_recovery
        self.attempt_argparse_recovery = attempt_argparse_recovery
        self.is_non_fatal_search_failure = is_non_fatal_search_failure
        self.git_observation = git_observation
        self.merge_output_paths_into_files_changed = merge_output_paths_into_files_changed
        self.write_json_log = write_json_log

    async def run_commands(
        self,
        *,
        run_id: str,
        step: PlannerStep,
        commands: list[str],
        workspace_path: Path,
        run_path: Path,
    ) -> StepExecutionResult:
        started = time.monotonic()
        step_id = step.id
        chunks: list[dict] = []

        for idx, command in enumerate(commands):
            command_step_id = step_id if len(commands) == 1 else f"{step_id}-{idx+1}"
            result = await self.run_command(
                run_id=run_id,
                step_id=command_step_id,
                command=command,
                workspace_path=workspace_path,
                run_path=run_path,
                stdin_payload=None,
                step=step,
            )
            if result.status != "completed":
                recovered = await self.attempt_missing_module_recovery(
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
                argparse_recovery_attempts = await self.attempt_argparse_recovery(
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

            if self.is_non_fatal_search_failure(step=step, command=command, result=result):
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

        diff_stats, files_changed = await self.git_observation(workspace_path)
        stdout_text = "\n".join(chunk["stdout"] for chunk in chunks if chunk["stdout"])
        stderr_text = "\n".join(chunk["stderr"] for chunk in chunks if chunk["stderr"])
        files_changed = self.merge_output_paths_into_files_changed(
            files_changed=files_changed,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
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
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            duration_ms=total_ms,
            command=" && ".join(commands),
            diff_stats=diff_stats,
            files_changed=files_changed,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            log_path=self.write_json_log(run_path, step_id, {"commands": chunks, "summary": summary}),
            is_infra_error=False,
        )

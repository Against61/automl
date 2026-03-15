from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Callable

from orchestrator.config import Settings
from orchestrator.execution.runner_models import IdleProcessTimeout, StepExecutionResult, TimeoutProfile
from orchestrator.persistence.schemas import PlannerStep, StepIntent


class SubprocessExecutionSupport:
    def __init__(
        self,
        *,
        settings: Settings,
        shell_bin: str,
        sanitized_env: Callable[[], dict[str, str]],
        truncate: Callable[[str], str],
        normalize_expected_path: Callable[[str, Path], str | None],
        extract_missing_python_file_path: Callable[[str], str | None],
        running: dict[str, asyncio.subprocess.Process],
    ) -> None:
        self.settings = settings
        self.shell_bin = shell_bin
        self.sanitized_env = sanitized_env
        self.truncate = truncate
        self.normalize_expected_path = normalize_expected_path
        self.extract_missing_python_file_path = extract_missing_python_file_path
        self.running = running

    async def run_raw(
        self,
        *,
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
        timeout_profile = self.timeout_profile_for_step(step=step, command=command or " ".join(argv))
        last_output_at = start
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace_path.as_posix(),
            env=self.sanitized_env(),
            stdin=asyncio.subprocess.PIPE if stdin_payload is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.running[run_id] = process
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
                await self.wait_with_idle_watchdog(
                    process=process,
                    start_time=start,
                    timeout_profile=timeout_profile,
                    last_output_at_ref=lambda: last_output_at,
                )
            await asyncio.gather(stdout_task, stderr_task)
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout_text = self.truncate(stdout_chunks.decode("utf-8", errors="ignore"))
            stderr_text = self.truncate(stderr_chunks.decode("utf-8", errors="ignore"))
            status = "completed" if process.returncode == 0 else "failed"
            diff_stats, files_changed = await self.git_observation(workspace_path)
            files_changed = self.merge_output_paths_into_files_changed(
                files_changed=files_changed,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                workspace_path=workspace_path,
            )
            summary = "command completed" if status == "completed" else "command failed"
            missing_artifact = None
            if status != "completed":
                missing_artifact = self.extract_missing_python_file_path(stderr_text)
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
            stdout_text = self.truncate(stdout_chunks.decode("utf-8", errors="ignore"))
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
        except IdleProcessTimeout as exc:
            process.kill()
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            duration_ms = int((time.monotonic() - start) * 1000)
            stdout_text = self.truncate(stdout_chunks.decode("utf-8", errors="ignore"))
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
                stdout_text=self.truncate(stdout_chunks.decode("utf-8", errors="ignore")),
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
            self.running.pop(run_id, None)

    async def wait_with_idle_watchdog(
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
                raise IdleProcessTimeout(
                    f"process max wall-clock timeout ({max_wall_clock}s) [{timeout_profile.label}]"
                )
            idle_timeout = timeout_profile.idle_timeout_sec or 0
            if idle_timeout > 0 and (now - float(last_output_at_ref())) > idle_timeout:
                raise IdleProcessTimeout(
                    f"process idle timeout ({idle_timeout}s without output) [{timeout_profile.label}]"
                )

    def timeout_profile_for_step(self, *, step: PlannerStep | None, command: str) -> TimeoutProfile:
        if self.is_long_running_training_step(step=step, command=command):
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

    def is_long_running_training_step(self, *, step: PlannerStep | None, command: str) -> bool:
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

    async def git_observation(self, workspace_path: Path) -> tuple[str, list[str]]:
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
            diff_stats = self.truncate(stdout.decode("utf-8", errors="ignore")).strip()

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

    def merge_output_paths_into_files_changed(
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
            normalized = self.normalize_expected_path(raw, workspace_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)

        for normalized in self.extract_output_paths_from_text(
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            workspace_path=workspace_path,
        ):
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
        return merged

    def extract_output_paths_from_text(
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
            normalized = self.normalize_expected_path(candidate, workspace_path)
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

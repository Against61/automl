from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Awaitable, Callable

from orchestrator.config import Settings
from orchestrator.execution.runner_models import StepExecutionResult
from orchestrator.persistence.schemas import PlannerStep


class CommandRecoverySupport:
    def __init__(
        self,
        *,
        settings: Settings,
        shell_primary_binary: Callable[[str], str],
        run_command: Callable[..., Awaitable[StepExecutionResult]],
        python_missing_module_map: dict[str, str],
    ) -> None:
        self.settings = settings
        self.shell_primary_binary = shell_primary_binary
        self.run_command = run_command
        self.python_missing_module_map = python_missing_module_map
        self.missing_module_re = re.compile(r"ModuleNotFoundError:\s+No module named ['\"]([^'\"]+)['\"]")
        self.no_module_named_re = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
        self.argparse_unknown_re = re.compile(r"unrecognized arguments:\s*(.+)", flags=re.IGNORECASE)

    async def attempt_missing_module_recovery(
        self,
        *,
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
        if not self.looks_like_python_command(command):
            return None

        module_name = self.missing_module_name_from_result(failed_result)
        if not module_name:
            return None
        if self.is_local_module_reference(module_name, workspace_path):
            return None

        pip_packages = self.package_for_module(module_name)
        python_bin = "python3" if command.strip().startswith("python3") else "python"
        install_cmd = f"{python_bin} -m pip install {pip_packages}"
        install_result = await self.run_command(
            run_id=run_id,
            step_id=f"{step_id}-{base_step_idx}-autoinstall",
            command=install_cmd,
            workspace_path=workspace_path,
            run_path=run_path,
            stdin_payload=None,
        )
        if install_result.status != "completed":
            return install_result, failed_result

        retry_result = await self.run_command(
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

    async def attempt_argparse_recovery(
        self,
        *,
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
        if not self.looks_like_python_command(command):
            return []

        unknown_args = self.extract_unknown_args(failed_result.stderr_text)
        if not unknown_args:
            return []

        attempts: list[tuple[str, StepExecutionResult]] = []

        normalized_cmd = self.replace_underscored_flags(command, unknown_args)
        if normalized_cmd and normalized_cmd != command:
            normalized_result = await self.run_command(
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

        pruned_cmd = self.remove_unknown_args(command, unknown_args)
        if pruned_cmd and pruned_cmd not in {command, normalized_cmd}:
            pruned_result = await self.run_command(
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

    def looks_like_python_command(self, command: str) -> bool:
        stripped = command.strip()
        return stripped.startswith("python ") or stripped.startswith("python3 ")

    def missing_module_name(self, stderr_text: str) -> str | None:
        text = stderr_text or ""
        match = self.missing_module_re.search(text)
        if not match:
            match = self.no_module_named_re.search(text)
        if not match:
            return None
        return match.group(1)

    def missing_module_name_from_result(self, result: StepExecutionResult) -> str | None:
        for text in (result.stderr_text, result.stdout_text):
            module_name = self.missing_module_name(text)
            if module_name:
                return module_name
        return None

    def is_local_module_reference(self, module_name: str, workspace_path: Path) -> bool:
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

    def package_for_module(self, module_name: str) -> str:
        package = self.python_missing_module_map.get(module_name)
        if package:
            return package
        if "." in module_name:
            root = module_name.split(".", 1)[0]
            return self.python_missing_module_map.get(root, root)
        return module_name

    def extract_unknown_args(self, stderr_text: str) -> list[str]:
        match = self.argparse_unknown_re.search(stderr_text or "")
        if not match:
            return []
        payload = match.group(1).strip()
        try:
            tokens = shlex.split(payload)
        except ValueError:
            tokens = payload.split()
        return [token for token in tokens if token.startswith("-")]

    def replace_underscored_flags(self, command: str, unknown_args: list[str]) -> str | None:
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

    def remove_unknown_args(self, command: str, unknown_args: list[str]) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        unknown = set(unknown_args)
        filtered = [token for token in tokens if token not in unknown]
        if len(filtered) == len(tokens):
            return None
        return " ".join(shlex.quote(token) for token in filtered)

    def is_non_fatal_search_failure(self, *, step: PlannerStep, command: str, result: StepExecutionResult) -> bool:
        if result.status == "completed":
            return False
        if result.exit_code != 1:
            return False
        if step.step_type != "check":
            return False
        search_bin = self.shell_primary_binary(command)
        if search_bin not in {"rg", "grep"}:
            return False
        stderr = (result.stderr_text or "").strip().lower()
        if stderr and "no matches found" not in stderr:
            return False
        return True

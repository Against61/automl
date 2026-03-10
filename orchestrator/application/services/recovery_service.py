from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from orchestrator.persistence.schemas import PlannerStep

_MISSING_PYTHON_FILE_RE = re.compile(
    r"python[^:]*:\s+can't open file ['\"]([^'\"]+)['\"]:\s+\[Errno\s+2\]\s+No such file or directory",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class MissingFileRecoveryDecision:
    detected: bool
    missing_path: str | None


class MissingFileRecoveryService:
    def detect_missing_python_file(self, stderr_text: str) -> MissingFileRecoveryDecision:
        match = _MISSING_PYTHON_FILE_RE.search(stderr_text or "")
        if not match:
            return MissingFileRecoveryDecision(detected=False, missing_path=None)
        return MissingFileRecoveryDecision(detected=True, missing_path=match.group(1).strip())

    def find_python_file_candidates(self, missing_path: str, workspace_path: Path) -> list[Path]:
        if not missing_path:
            return []

        expected = Path(missing_path)
        expected_name = expected.name
        if not expected_name:
            return []

        candidates: list[Path] = []
        seen: set[str] = set()
        direct_path = expected if expected.is_absolute() else (workspace_path / expected)
        direct_candidates: list[Path] = [direct_path]
        if not workspace_path in direct_path.parents and not direct_path.is_relative_to(
            workspace_path
        ):
            direct_candidates.append(workspace_path / expected_name)

        for candidate in direct_candidates:
            if candidate.is_file():
                key = candidate.resolve().as_posix()
                if key not in seen:
                    candidates.append(candidate)
                    seen.add(key)

        for match in workspace_path.rglob(expected_name):
            if not match.is_file():
                continue
            key = match.resolve().as_posix()
            if key in seen:
                continue
            candidates.append(match)
            seen.add(key)

        return candidates

    def replace_missing_file_in_step(
        self,
        step: PlannerStep,
        expected_missing: str,
        replacement: str,
    ) -> PlannerStep | None:
        repaired = step.model_copy(deep=True)
        changed = False

        if repaired.command:
            updated_command = self._replace_missing_file_in_command(
                repaired.command,
                expected_missing,
                replacement,
            )
            if updated_command:
                repaired.command = updated_command
                changed = True

        updated_commands: list[str] = []
        for command in repaired.commands:
            updated = self._replace_missing_file_in_command(
                command,
                expected_missing,
                replacement,
            )
            updated_commands.append(updated or command)
            if updated:
                changed = True
        repaired.commands = updated_commands

        if not changed:
            return None
        return repaired

    def _replace_missing_file_in_command(self, command: str, expected_missing: str, replacement: str) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None

        for idx, token in enumerate(tokens):
            if idx == 0:
                continue
            if self._is_python_file_token_match(token, expected_missing):
                tokens[idx] = replacement
                return " ".join(shlex.quote(token) for token in tokens)
        return None

    def _is_python_file_token_match(self, token: str, expected_missing: str) -> bool:
        if not token or not expected_missing:
            return False

        expected_name = Path(expected_missing).name
        if not expected_name:
            return False

        token_clean = token.strip().strip("\"'")
        if token_clean == expected_missing:
            return True
        if token_clean == expected_name:
            return True
        if token_clean.endswith(f"/{expected_name}") or token_clean.endswith(f"\\\\{expected_name}"):
            return True
        return False


def detect_missing_python_file(stderr_text: str) -> MissingFileRecoveryDecision:
    return MissingFileRecoveryService().detect_missing_python_file(stderr_text)


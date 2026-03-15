from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orchestrator.config import Settings
from orchestrator.execution.intent_validation import IntentValidationSupport
from orchestrator.execution.metric_parsing import extract_metrics_from_text
from orchestrator.execution.workspace_metrics_reader import WorkspaceMetricsReader


@dataclass(slots=True)
class VerificationResult:
    status: str
    passed: bool
    commands: list[dict[str, str | int | bool]]
    metrics: dict[str, float | int | str | bool]
    details: dict[str, Any] = field(default_factory=dict)


class Verifier:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.workspace_metrics_reader = WorkspaceMetricsReader()
        self.intent_validation_support = IntentValidationSupport()

    async def run(
        self,
        workspace_path: Path,
        *,
        task: dict[str, Any] | None = None,
        story_id: str | None = None,
    ) -> VerificationResult:
        commands = self.settings.verify_command_list
        if not commands:
            metrics = self._read_workspace_metrics(workspace_path)
            return VerificationResult(
                status="passed",
                passed=True,
                commands=[],
                metrics=metrics,
                details=self._build_intent_validation_details(
                    metrics=metrics,
                    task=task,
                    workspace_path=workspace_path,
                    story_id=story_id,
                ),
            )

        summary: list[dict[str, str | int | bool]] = []
        metrics: dict[str, float | int | str | bool] = {}

        for command in commands:
            start = time.monotonic()
            process = await asyncio.create_subprocess_exec(
                "/bin/sh",
                "-lc",
                command,
                cwd=workspace_path.as_posix(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            duration_ms = int((time.monotonic() - start) * 1000)
            status_ok = process.returncode == 0

            stdout_text = stdout.decode("utf-8", errors="ignore")[:2000]
            stderr_text = stderr.decode("utf-8", errors="ignore")[:2000]
            summary.append(
                {
                    "command": command,
                    "ok": status_ok,
                    "duration_ms": duration_ms,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                }
            )

            metrics.update(self._extract_metrics_from_text(f"{stdout_text}\n{stderr_text}"))
            workspace_metrics = self._read_workspace_metrics(workspace_path)
            metrics.update(workspace_metrics)

            if not status_ok:
                return VerificationResult(
                    status="failed",
                    passed=False,
                    commands=summary,
                    metrics=metrics,
                    details=self._build_intent_validation_details(
                        metrics=metrics,
                        task=task,
                        workspace_path=workspace_path,
                        story_id=story_id,
                    ),
                )

        return VerificationResult(
            status="passed",
            passed=True,
            commands=summary,
            metrics=metrics,
            details=self._build_intent_validation_details(
                metrics=metrics,
                task=task,
                workspace_path=workspace_path,
                story_id=story_id,
            ),
        )

    def _extract_metrics_from_text(self, text: str) -> dict[str, float | int | str | bool]:
        return extract_metrics_from_text(text)

    def _read_workspace_metrics(self, workspace_path: Path) -> dict[str, float | int | str | bool]:
        return self.workspace_metrics_reader.read_workspace_metrics(workspace_path)

    def _extract_metrics_from_markdown_table(self, text: str) -> dict[str, float | int | str | bool]:
        return self.workspace_metrics_reader.extract_metrics_from_markdown_table(text)

    def _looks_like_key_value_metric_table(self, header_cells: list[str]) -> bool:
        return self.workspace_metrics_reader.looks_like_key_value_metric_table(header_cells)

    def _parse_numeric_cell(self, raw: str) -> float | int | None:
        return self.workspace_metrics_reader.parse_numeric_cell(raw)

    def _read_json_metrics(self, path: Path) -> dict[str, float | int | str | bool]:
        return self.workspace_metrics_reader.read_json_metrics(path)

    def _extract_json_metrics_payload(self, raw: object) -> dict[str, float | int | str | bool]:
        return self.workspace_metrics_reader.extract_json_metrics_payload(raw)

    def _collect_json_metrics_recursive(
        self,
        raw: object,
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        self.workspace_metrics_reader.collect_json_metrics_recursive(raw, metrics)

    def _flatten_metric_mapping(
        self,
        parent_key: str,
        mapping: dict[str, object],
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        self.workspace_metrics_reader.flatten_metric_mapping(parent_key, mapping, metrics)

    def _extract_split_flags_from_mapping(
        self,
        mapping: dict[str, object],
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        self.workspace_metrics_reader.extract_split_flags_from_mapping(mapping, metrics)

    def _extract_report_flags_from_mapping(
        self,
        mapping: dict[str, object],
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        self.workspace_metrics_reader.extract_report_flags_from_mapping(mapping, metrics)

    def _build_intent_validation_details(
        self,
        *,
        metrics: dict[str, float | int | str | bool],
        task: dict[str, Any] | None,
        workspace_path: Path,
        story_id: str | None = None,
    ) -> dict[str, Any]:
        return self.intent_validation_support.build_intent_validation_details(
            metrics=metrics,
            task=task,
            workspace_path=workspace_path,
            story_id=story_id,
        )

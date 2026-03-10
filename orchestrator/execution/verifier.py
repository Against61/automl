from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.execution.metric_parsing import (
    extract_metrics_from_text,
    extract_numeric_metrics,
    extract_report_context_flags,
    extract_split_integrity_flags,
    normalize_metric_key,
)


@dataclass(slots=True)
class VerificationResult:
    status: str
    passed: bool
    commands: list[dict[str, str | int | bool]]
    metrics: dict[str, float | int | str | bool]


class Verifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def run(self, workspace_path: Path) -> VerificationResult:
        commands = self.settings.verify_command_list
        if not commands:
            return VerificationResult(
                status="passed",
                passed=True,
                commands=[],
                metrics=self._read_workspace_metrics(workspace_path),
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
                )

        return VerificationResult(
            status="passed",
            passed=True,
            commands=summary,
            metrics=metrics,
        )

    def _extract_metrics_from_text(self, text: str) -> dict[str, float | int | str | bool]:
        return extract_metrics_from_text(text)

    def _read_workspace_metrics(self, workspace_path: Path) -> dict[str, float | int | str | bool]:
        metrics: dict[str, float | int | str | bool] = {}
        candidates: list[Path] = [
            workspace_path / "metrics.md",
            workspace_path / "results.json",
            workspace_path / "metrics.json",
        ]
        # Best-effort discovery: many runs persist metrics under custom markdown/json names.
        for pattern in ("*metrics*.md", "*metrics*.markdown", "*metrics*.json"):
            for path in sorted(workspace_path.glob(pattern)):
                if path not in candidates:
                    candidates.append(path)

        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                metrics.update(self._read_json_metrics(path))
            else:
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                metrics.update(self._extract_metrics_from_text(text))
                metrics.update(self._extract_metrics_from_markdown_table(text))
                metrics.update(extract_split_integrity_flags(text))
                metrics.update(extract_report_context_flags(text))

        return metrics

    def _extract_metrics_from_markdown_table(self, text: str) -> dict[str, float | int | str | bool]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        metrics: dict[str, float | int | str | bool] = {}
        idx = 0
        while idx + 2 < len(lines):
            header = lines[idx]
            sep = lines[idx + 1]
            if not (header.startswith("|") and sep.startswith("|")):
                idx += 1
                continue
            if "---" not in sep:
                idx += 1
                continue

            header_cells = [cell.strip().lower().replace(" ", "_") for cell in header.strip("|").split("|")]
            data_rows: list[list[str]] = []
            row_idx = idx + 2
            while row_idx < len(lines) and lines[row_idx].startswith("|"):
                cells = [cell.strip() for cell in lines[row_idx].strip("|").split("|")]
                if len(cells) == len(header_cells):
                    data_rows.append(cells)
                row_idx += 1

            if data_rows:
                if self._looks_like_key_value_metric_table(header_cells):
                    name_idx = 0
                    value_idx = 1
                    for row in data_rows:
                        metric_name = normalize_metric_key(row[name_idx])
                        if not metric_name:
                            continue
                        parsed = self._parse_numeric_cell(row[value_idx])
                        if parsed is None:
                            continue
                        metrics[metric_name] = parsed
                else:
                    last_row = data_rows[-1]
                    for key, raw in zip(header_cells, last_row):
                        parsed = self._parse_numeric_cell(raw)
                        if parsed is None:
                            continue
                        metrics[key] = parsed

            idx = row_idx
        return metrics

    def _looks_like_key_value_metric_table(self, header_cells: list[str]) -> bool:
        if len(header_cells) < 2:
            return False
        first = header_cells[0]
        second = header_cells[1]
        return first in {"metric", "name", "measure", "key"} and second in {"value", "score", "metric_value"}

    def _parse_numeric_cell(self, raw: str) -> float | int | None:
        normalized = raw.replace("**", "").replace("__", "").strip()
        is_percent = "%" in normalized
        normalized = normalized.replace("%", "").strip()
        if not normalized:
            return None
        if not re.match(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$", normalized):
            return None
        value = float(normalized)
        if is_percent and value > 1.0:
            value = value / 100.0
        if value.is_integer():
            return int(value)
        return value

    def _read_json_metrics(self, path: Path) -> dict[str, float | int | str | bool]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return extract_numeric_metrics(raw)

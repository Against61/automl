from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.application.services.prompt_content_service import PromptContentService
from orchestrator.config import Settings
from orchestrator.execution.codex_cli import normalize_codex_command
from orchestrator.execution.metric_parsing import normalize_metric_key


@dataclass(slots=True)
class MetricInterpretation:
    resolved_metric_key: str
    resolved_value: float
    confidence: str
    reason: str


class CodexMetricInterpreter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.prompt_content_service = PromptContentService()

    async def resolve_metric(
        self,
        *,
        required_metric_key: str,
        metrics: dict[str, Any],
        workspace_path: Path,
    ) -> MetricInterpretation | None:
        if not self.settings.codex_cli_cmd.strip():
            return None

        prompt = self._build_prompt(
            required_metric_key=required_metric_key,
            metrics=metrics,
            workspace_path=workspace_path,
        )
        argv = normalize_codex_command(
            shlex.split(self.settings.codex_cli_cmd),
            model=self.settings.codex_model,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=workspace_path.as_posix(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._sanitized_env(),
            )
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=min(self.settings.codex_step_timeout_sec, 90),
            )
        except (OSError, asyncio.TimeoutError):
            return None

        if process.returncode != 0:
            return None
        return self._parse_response(stdout.decode("utf-8", errors="ignore"))

    def _build_prompt(
        self,
        *,
        required_metric_key: str,
        metrics: dict[str, Any],
        workspace_path: Path,
    ) -> str:
        compact_metrics = {
            normalize_metric_key(str(key)): value
            for key, value in list(metrics.items())[:80]
            if isinstance(key, str)
        }
        artifact_excerpt = self._collect_artifact_excerpt(
            workspace_path,
            required_metric_key=required_metric_key,
        )
        return "\n".join(
            [
                "You are resolving metric naming only.",
                "Return JSON only, no prose.",
                'Schema: {"resolved_metric_key":"<key>","resolved_value":0.0,"confidence":"high|medium|low","reason":"<short>"}',
                'If unresolved, return "{}".',
                "",
                f"Required metric key: {normalize_metric_key(required_metric_key)}",
                "Prefer held-out/test/eval/val metrics over train metrics.",
                "For segmentation, IoU, mIoU, mean_iou, and Jaccard may be equivalent if they describe the same held-out metric.",
                "",
                "Normalized metrics already extracted:",
                json.dumps(compact_metrics, ensure_ascii=True, indent=2),
                "",
                "Metric artifact excerpts:",
                artifact_excerpt or "[none]",
            ]
        )

    def _collect_artifact_excerpt(self, workspace_path: Path, *, required_metric_key: str) -> str:
        candidates: list[Path] = []
        for raw in (
            workspace_path / "metrics.json",
            workspace_path / "results.json",
            workspace_path / "metrics.md",
        ):
            if raw.exists():
                candidates.append(raw)
        for pattern in ("*metrics*.json", "*metrics*.md", "*metrics*.markdown", "*report*.md"):
            for path in sorted(workspace_path.rglob(pattern)):
                if path.is_file() and path not in candidates:
                    candidates.append(path)

        chunks: list[str] = []
        focus_terms = [
            required_metric_key,
            "accuracy",
            "iou",
            "mean_iou",
            "loss",
            "threshold",
            "split",
            "train",
            "eval",
            "test",
            "val",
        ]
        for path in candidates[:3]:
            text = self.prompt_content_service.render_file_for_prompt(
                path,
                purpose="metrics_artifact",
                focus_terms=focus_terms,
            )
            text = text.strip()
            if not text:
                continue
            chunks.append(f"### {path.relative_to(workspace_path).as_posix()}\n{text}")
        return "\n\n".join(chunks)

    def _parse_response(self, text: str) -> MetricInterpretation | None:
        payload = text.strip()
        if not payload:
            return None
        if payload.startswith("```"):
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", payload, flags=re.DOTALL)
            if match:
                payload = match.group(1)
        if not payload.startswith("{"):
            match = re.search(r"(\{.*\})", payload, flags=re.DOTALL)
            if match:
                payload = match.group(1)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        key = normalize_metric_key(str(data.get("resolved_metric_key") or ""))
        value = data.get("resolved_value")
        if not key or not isinstance(value, int | float):
            return None
        confidence = str(data.get("confidence") or "low").strip().lower()
        reason = str(data.get("reason") or "").strip()
        return MetricInterpretation(
            resolved_metric_key=key,
            resolved_value=float(value),
            confidence=confidence or "low",
            reason=reason,
        )

    def _sanitized_env(self) -> dict[str, str]:
        env = {"PATH": os.environ.get("PATH", "")}
        if os.environ.get("HOME"):
            env["HOME"] = os.environ["HOME"]
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORG_ID",
            "CODEX_HOME",
            "CODEX_CONFIG_HOME",
            "CODEX_API_KEY",
            "CODEX_MODEL",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        ):
            value = os.environ.get(key)
            if value:
                env[key] = value
        if not env.get("CODEX_HOME") and env.get("HOME"):
            env["CODEX_HOME"] = f"{env['HOME']}/.codex"
        if not env.get("CODEX_CONFIG_HOME") and env.get("CODEX_HOME"):
            env["CODEX_CONFIG_HOME"] = env["CODEX_HOME"]
        return env

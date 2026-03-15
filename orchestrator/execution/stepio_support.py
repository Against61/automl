from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable

from orchestrator.execution.metric_parsing import extract_metrics_from_text
from orchestrator.execution.runner_models import StepExecutionResult
from orchestrator.persistence.schemas import ArtifactKind, PlannerStep, StepArtifactManifest, StepIOResult, StepIntent


class StepIOSupport:
    def __init__(
        self,
        *,
        normalize_expected_path: Callable[[str, Path], str | None],
        normalize_step_intent: Callable[[StepIntent | str], StepIntent],
    ) -> None:
        self.normalize_expected_path = normalize_expected_path
        self.normalize_step_intent = normalize_step_intent

    def write_stream_logs(self, run_path: Path, step_id: str, stdout_text: str, stderr_text: str) -> tuple[str, str]:
        stdout_path = run_path / f"{step_id}.stdout.log"
        stderr_path = run_path / f"{step_id}.stderr.log"
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        return stdout_path.as_posix(), stderr_path.as_posix()

    def write_json_log(self, run_path: Path, step_id: str, payload: dict[str, Any]) -> str:
        log_path = run_path / f"{step_id}.executor.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return log_path.as_posix()

    def write_stepio_result(
        self,
        *,
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

        artifacts = self.collect_stepio_artifacts(step, result, workspace_path)
        payload = StepIOResult(
            run_id=run_id,
            step_id=step.id,
            status=step_status,
            error_code=self.stepio_error_code(result),
            summary=result.summary,
            operation=step.operation,
            intent="check" if step.step_type == "check" else "change",
            inputs=step.inputs,
            expected_outputs=step.expected_outputs,
            artifacts_produced=artifacts,
            metrics=self.extract_stepio_metrics(step=step, result=result),
            hyperparameters=self.extract_hyperparameters_from_command(result.command),
            duration_ms=result.duration_ms,
            command=result.command,
            stdout_path=result.stdout_path,
            stderr_path=result.stderr_path,
            log_path=result.log_path,
        )
        output_path = self.next_stepio_output_path(run_path, step.id)
        output_path.write_text(json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2), encoding="utf-8")
        return output_path.as_posix()

    def next_stepio_output_path(self, run_path: Path, step_id: str) -> Path:
        primary = run_path / f"{step_id}.step_result.json"
        if not primary.exists():
            return primary
        index = 2
        while True:
            candidate = run_path / f"{step_id}.step_result.{index}.json"
            if not candidate.exists():
                return candidate
            index += 1

    def stepio_error_code(self, result: StepExecutionResult) -> str:
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

    def collect_stepio_artifacts(
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
            normalized = self.normalize_expected_path(spec.path, workspace_path)
            if not normalized:
                continue
            expected_kind_by_path[normalized] = spec.kind
        candidate_paths: list[str] = []
        for rel in result.files_changed or []:
            normalized = self.normalize_expected_path(rel, workspace_path)
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
                    sha256 = self.sha256_file(abs_path)
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

    def sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def extract_stepio_metrics(
        self,
        *,
        step: PlannerStep,
        result: StepExecutionResult,
    ) -> dict[str, float | int | str | bool]:
        intent = self.normalize_step_intent(step.step_intent)
        if step.action == "codex" and intent in {
            StepIntent.create_file,
            StepIntent.modify_file,
            StepIntent.general,
        }:
            return {}
        return extract_metrics_from_text(f"{result.stdout_text}\n{result.stderr_text}")

    def extract_hyperparameters_from_command(self, command: str | None) -> dict[str, Any]:
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
                    parsed[normalized] = self.coerce_value(value)
            idx += 1
        return parsed

    def coerce_value(self, value: str | None) -> Any:
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

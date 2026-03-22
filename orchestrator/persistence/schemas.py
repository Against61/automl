from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Priority(str, Enum):
    high = "high"
    normal = "normal"
    low = "low"


class RunStatus(str, Enum):
    RECEIVED = "RECEIVED"
    CONTEXT_READY = "CONTEXT_READY"
    PLAN_READY = "PLAN_READY"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    WAITING_PLAN_REVIEW = "WAITING_PLAN_REVIEW"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    PACKAGING = "PACKAGING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @classmethod
    def terminal(cls) -> set["RunStatus"]:
        return {cls.COMPLETED, cls.FAILED, cls.CANCELLED}


class TaskPayload(BaseModel):
    goal: str = Field(min_length=3)
    constraints: list[str] = Field(default_factory=list)
    pdf_scope: list[str] = Field(default_factory=list)
    execution_mode: Literal["plan_execute", "ralph_story"] = "plan_execute"


class TaskSubmittedEvent(BaseModel):
    event_id: UUID
    event_type: Literal["task.submitted"]
    schema_version: Literal["1.0"]
    timestamp: datetime = Field(default_factory=utc_now)
    task_id: UUID
    workspace_id: str = Field(min_length=1)
    priority: Priority = Priority.normal
    payload: TaskPayload


class RunControlPayload(BaseModel):
    run_id: UUID
    reason: str | None = None


class ControlEvent(BaseModel):
    event_id: UUID
    event_type: Literal["run.approve", "run.cancel", "run.retry"]
    schema_version: Literal["1.0"] = "1.0"
    timestamp: datetime = Field(default_factory=utc_now)
    payload: RunControlPayload


class ArtifactKind(str, Enum):
    file = "file"
    report = "report"
    metrics = "metrics"
    checkpoint = "checkpoint"
    generic = "generic"


class StepIntent(str, Enum):
    create_file = "create_file"
    modify_file = "modify_file"
    run_training = "run_training"
    verify_metrics = "verify_metrics"
    general = "general"


class StepOperation(str, Enum):
    inspect_workspace = "inspect_workspace"
    edit_code = "edit_code"
    run_training = "run_training"
    verify_metrics = "verify_metrics"
    research = "research"
    general = "general"


class ArtifactSpec(BaseModel):
    path: str | None = None
    kind: ArtifactKind = ArtifactKind.file
    must_exist: bool = True
    must_be_nonempty: bool = False
    metric_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize(self) -> "ArtifactSpec":
        if self.path is not None:
            cleaned = self.path.strip().strip("`\"'")
            self.path = cleaned or None
        self.metric_keys = [str(item).strip() for item in self.metric_keys if str(item).strip()]
        return self


class StepIOInputs(BaseModel):
    files: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize(self) -> "StepIOInputs":
        self.files = [str(item).strip() for item in self.files if str(item).strip()]
        self.commands = [str(item).strip() for item in self.commands if str(item).strip()]
        return self


class StepIOExpectedOutputs(BaseModel):
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    metrics_required: list[str] = Field(default_factory=list)
    stop_condition: str = ""

    @model_validator(mode="after")
    def normalize(self) -> "StepIOExpectedOutputs":
        self.metrics_required = [str(item).strip() for item in self.metrics_required if str(item).strip()]
        return self


class StepPolicy(BaseModel):
    risk: Literal["low", "medium", "high"] = "low"
    approval_required: bool = False


class StepRetryPolicy(BaseModel):
    max_retries: int = Field(default=0, ge=0, le=10)
    on: list[Literal["infra_error", "missing_file", "arg_error", "contract_error", "execution_error"]] = Field(
        default_factory=list
    )


class StepArtifactManifest(BaseModel):
    path: str
    kind: ArtifactKind = ArtifactKind.file
    exists: bool = True
    size: int | None = None
    sha256: str | None = None


class StepIOResult(BaseModel):
    version: Literal["stepio.v1"] = "stepio.v1"
    run_id: str
    step_id: str
    status: Literal["completed", "failed", "timeout", "auto_repaired"]
    error_code: Literal["none", "infra_error", "missing_file", "arg_error", "contract_error", "execution_error"]
    summary: str
    operation: StepOperation = StepOperation.general
    intent: Literal["check", "change"] = "change"
    inputs: StepIOInputs = Field(default_factory=StepIOInputs)
    expected_outputs: StepIOExpectedOutputs = Field(default_factory=StepIOExpectedOutputs)
    artifacts_produced: list[StepArtifactManifest] = Field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    command: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_path: str | None = None


class PlannerStep(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    step_type: Literal["change", "check"] = "change"
    operation: StepOperation = StepOperation.general
    commands: list[str] = Field(default_factory=list)
    codex_prompt: str | None = None
    expected_artifacts: list[ArtifactSpec] = Field(default_factory=list)
    inputs: StepIOInputs = Field(default_factory=StepIOInputs)
    expected_outputs: StepIOExpectedOutputs = Field(default_factory=StepIOExpectedOutputs)
    policy: StepPolicy = Field(default_factory=StepPolicy)
    retry_policy: StepRetryPolicy = Field(default_factory=StepRetryPolicy)
    step_intent: StepIntent = StepIntent.general
    skill_paths: list[str] = Field(default_factory=list)
    stop_condition: str = ""
    action: Literal["codex", "shell", "read", "verify"] = "codex"
    instruction: str = ""
    command: str | None = None
    risk_level: Literal["low", "medium", "high"] = "low"

    @field_validator("expected_artifacts", mode="before")
    @classmethod
    def coerce_expected_artifacts(
        cls, value: list[ArtifactSpec | dict[str, Any] | str] | ArtifactSpec | dict[str, Any] | str | None
    ) -> list[dict[str, Any]]:
        if value is None:
            return []
        raw_items = value if isinstance(value, list) else [value]
        normalized: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, ArtifactSpec):
                normalized.append(item.model_dump(mode="json"))
                continue
            if isinstance(item, dict):
                normalized.append(item)
                continue
            if isinstance(item, str):
                normalized.append(cls._artifact_spec_from_text(item).model_dump(mode="json"))
                continue
        return normalized

    @model_validator(mode="after")
    def normalize_contract(self) -> "PlannerStep":
        if self.action in {"read", "verify"}:
            self.step_type = "check"
        if self.command and not self.commands:
            self.commands = [self.command]
        if self.action == "shell" and self.command and self.command not in self.commands:
            self.commands.append(self.command)
        if self.commands:
            deduped: list[str] = []
            seen: set[str] = set()
            for cmd in self.commands:
                normalized = cmd.strip()
                if not normalized or normalized in seen:
                    continue
                deduped.append(normalized)
                seen.add(normalized)
            self.commands = deduped
        self.skill_paths = [str(item).strip() for item in self.skill_paths if str(item).strip()]
        if not self.codex_prompt and self.instruction:
            self.codex_prompt = self.instruction
        if not self.instruction and self.codex_prompt:
            self.instruction = self.codex_prompt
        if not self.instruction:
            self.instruction = self.title
        if not self.stop_condition:
            self.stop_condition = "step completed without errors"
        if self.step_intent == StepIntent.general:
            self.step_intent = self._infer_step_intent()
        if self.operation == StepOperation.general:
            self.operation = self._infer_operation()
        self._ensure_default_metrics_artifact()
        if self.step_intent == StepIntent.run_training:
            if self.command and self.command not in self.commands:
                self.commands.append(self.command)
            if self.commands:
                self.action = "shell"
                if not self.command:
                    self.command = self.commands[0]
            elif self.action == "codex":
                raise ValueError(
                    "run_training steps must declare explicit shell commands; "
                    "codex may prepare training code but must not own training execution"
                )
        self.policy.risk = self.risk_level
        if not self.inputs.commands:
            self.inputs.commands = list(self.commands)
        if self.command and self.command not in self.inputs.commands:
            self.inputs.commands.append(self.command)
        if not self.expected_outputs.artifacts and self.expected_artifacts:
            self.expected_outputs.artifacts = list(self.expected_artifacts)
        if not self.expected_outputs.metrics_required:
            metric_keys: list[str] = []
            for artifact in self.expected_outputs.artifacts:
                for key in artifact.metric_keys:
                    normalized = str(key).strip()
                    if normalized and normalized not in metric_keys:
                        metric_keys.append(normalized)
            self.expected_outputs.metrics_required = metric_keys
        if not self.expected_outputs.stop_condition:
            self.expected_outputs.stop_condition = self.stop_condition
        return self

    def _ensure_default_metrics_artifact(self) -> None:
        if self.step_intent not in {StepIntent.run_training, StepIntent.verify_metrics}:
            return

        metric_keys = [str(item).strip() for item in self.expected_outputs.metrics_required if str(item).strip()]
        metric_spec: ArtifactSpec | None = None
        for artifact in self.expected_artifacts:
            if artifact.kind == ArtifactKind.metrics:
                metric_spec = artifact
                break

        if metric_spec is None:
            metric_spec = ArtifactSpec(
                path="metrics.json",
                kind=ArtifactKind.metrics,
                must_exist=True,
                must_be_nonempty=True,
                metric_keys=metric_keys,
            )
            self.expected_artifacts.append(metric_spec)
        else:
            if not metric_spec.path:
                metric_spec.path = "metrics.json"
            metric_spec.must_exist = True
            metric_spec.must_be_nonempty = True
            if metric_keys:
                seen_metric_keys = set(metric_spec.metric_keys)
                for key in metric_keys:
                    if key not in seen_metric_keys:
                        metric_spec.metric_keys.append(key)
                        seen_metric_keys.add(key)

        if not self.expected_outputs.artifacts:
            self.expected_outputs.artifacts = list(self.expected_artifacts)
        else:
            output_metric_spec: ArtifactSpec | None = None
            for artifact in self.expected_outputs.artifacts:
                if artifact.kind == ArtifactKind.metrics:
                    output_metric_spec = artifact
                    break
            if output_metric_spec is None:
                self.expected_outputs.artifacts.append(metric_spec.model_copy(deep=True))
            else:
                if not output_metric_spec.path:
                    output_metric_spec.path = "metrics.json"
                output_metric_spec.must_exist = True
                output_metric_spec.must_be_nonempty = True
                if metric_keys:
                    seen_metric_keys = set(output_metric_spec.metric_keys)
                    for key in metric_keys:
                        if key not in seen_metric_keys:
                            output_metric_spec.metric_keys.append(key)
                            seen_metric_keys.add(key)

    @staticmethod
    def _artifact_spec_from_text(text: str) -> ArtifactSpec:
        raw = text.strip()
        lowered = raw.lower()
        path = PlannerStep._extract_path_from_text(raw)
        kind = ArtifactKind.file
        if "metric" in lowered:
            kind = ArtifactKind.metrics
        elif "report" in lowered:
            kind = ArtifactKind.report
        elif "checkpoint" in lowered or raw.endswith(".ckpt") or raw.endswith(".pth") or raw.endswith(".pt"):
            kind = ArtifactKind.checkpoint
        if not path:
            return ArtifactSpec(path=None, kind=ArtifactKind.generic, must_exist=False)
        return ArtifactSpec(path=path, kind=kind, must_exist=True)

    @staticmethod
    def _extract_path_from_text(text: str) -> str | None:
        candidates: list[str] = []
        stripped = text.strip().strip("`\"'")
        if stripped:
            candidates.append(stripped)
        tokens = [token.strip("`\"'[],(){}") for token in stripped.split()]
        candidates.extend(token for token in tokens if token)
        filename_with_ext = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*\.[A-Za-z0-9]{1,10}$")
        hidden_file = re.compile(r"^\.[A-Za-z0-9_.-]+$")
        for token in candidates:
            cleaned = token.rstrip(".,;:")
            if not cleaned:
                continue
            has_whitespace = any(ch.isspace() for ch in cleaned)
            if ("/" in cleaned or "\\" in cleaned) and not has_whitespace:
                return cleaned
            if hidden_file.match(cleaned) and not has_whitespace:
                return cleaned
            if filename_with_ext.match(cleaned) and not has_whitespace:
                return cleaned
        return None

    def _infer_step_intent(self) -> StepIntent:
        title = f"{self.title} {self.instruction} {self.stop_condition}".lower()
        artifact_kinds = {artifact.kind for artifact in self.expected_artifacts}
        has_explicit_execution = bool(self.command or self.commands or self.action == "shell")
        if has_explicit_execution and ("train" in title or "epoch" in title or StepIntent.run_training.value in title):
            return StepIntent.run_training
        if "verify" in title or "metric" in title or StepIntent.verify_metrics.value in title:
            return StepIntent.verify_metrics
        if self.action == "codex" and any(kind in {ArtifactKind.file, ArtifactKind.report, ArtifactKind.metrics, ArtifactKind.checkpoint} for kind in artifact_kinds):
            has_existing_hint = ("update" in title) or ("modify" in title)
            return StepIntent.modify_file if has_existing_hint else StepIntent.create_file
        if self.action == "shell" and "metric" in title:
            return StepIntent.verify_metrics
        return StepIntent.general

    def _infer_operation(self) -> StepOperation:
        if self.step_intent == StepIntent.run_training:
            return StepOperation.run_training
        if self.step_intent == StepIntent.verify_metrics:
            return StepOperation.verify_metrics
        if self.action in {"read", "verify"}:
            return StepOperation.inspect_workspace
        if self.action == "shell" and self.step_type == "check":
            return StepOperation.inspect_workspace
        if self.action == "codex":
            return StepOperation.edit_code
        return StepOperation.general


class PlannerPlan(BaseModel):
    version: Literal["1.0"] = "1.0"
    summary: str = Field(min_length=1)
    steps: list[PlannerStep] = Field(min_length=1)
    planner_sanitization: list["PlannerSanitizationEntry"] = Field(default_factory=list)
    experiment_memory_summary: str | None = None
    baseline_research_summary: str | None = None

    @model_validator(mode="after")
    def step_ids_unique(self) -> "PlannerPlan":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        return self


class PlannerSanitizationEntry(BaseModel):
    step_id: str | None = None
    field: str = Field(min_length=1)
    original: str = Field(min_length=1)
    normalized: str = Field(min_length=1)


class RetrievedContext(BaseModel):
    snippet: str
    document_path: str
    page_number: int
    confidence: float


class RunResultSummary(BaseModel):
    planned_steps: int
    executed_steps: int
    verification: Literal["passed", "failed"]


class RunResultArtifacts(BaseModel):
    patch_bundle_path: str
    report_json_path: str


class RunResultEvent(BaseModel):
    event_id: UUID
    event_type: Literal["run.result"] = "run.result"
    schema_version: Literal["1.0"] = "1.0"
    timestamp: datetime = Field(default_factory=utc_now)
    run_id: UUID
    task_id: UUID
    status: Literal["completed", "failed", "cancelled"]
    artifacts: RunResultArtifacts
    summary: RunResultSummary


class RunRecord(BaseModel):
    run_id: str
    task_id: str
    workspace_id: str
    priority: Priority
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    execution_cycle: int = 0
    cycle_started_at: datetime | None = None
    attempts_by_stage: dict[str, int] = Field(default_factory=dict)
    next_step_index: int = 0
    plan_json: dict[str, Any] | None = None
    context_json: list[dict[str, Any]] | None = None
    verification_json: dict[str, Any] | None = None
    error_message: str | None = None
    approved_at: datetime | None = None
    goal_signature: str | None = None


class ApproveRequest(BaseModel):
    run_id: UUID


class CancelRequest(BaseModel):
    run_id: UUID
    reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    timestamp: datetime = Field(default_factory=utc_now)


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    details: dict[str, Any]


class EventCreateRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    goal: str = Field(min_length=3)
    constraints: list[str] = Field(default_factory=list)
    pdf_scope: list[str] = Field(default_factory=list)
    priority: Priority = Priority.normal
    execution_mode: Literal["plan_execute", "ralph_story"] = "plan_execute"
    required_metric_key: str | None = None
    required_metric_min: float | None = None
    max_quality_retries: int | None = Field(default=None, ge=0, le=20)

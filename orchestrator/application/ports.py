from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from orchestrator.execution.codex_runner import StepExecutionResult
from orchestrator.execution.policy import PolicyDecision
from orchestrator.execution.verifier import VerificationResult
from orchestrator.persistence.schemas import PlannerPlan, PlannerStep, RunRecord, RunStatus
from orchestrator.planning.planner import PlanInput


class RunRepositoryPort(Protocol):
    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def update_run_status(self, run_id: str, status: RunStatus, error_message: str | None = None) -> None: ...
    async def set_plan(self, run_id: str, plan: dict[str, Any]) -> None: ...
    async def set_context(self, run_id: str, context: list[dict[str, Any]]) -> None: ...
    async def add_run_step(
        self,
        run_id: str,
        step_id: str,
        step_index: int,
        action: str,
        command: str | None,
        status: str,
        stdout_text: str,
        stderr_text: str,
        duration_ms: int,
    ) -> None: ...
    async def acquire_workspace_lock(self, workspace_id: str, run_id: str) -> bool: ...
    async def release_workspace_lock(self, workspace_id: str, run_id: str) -> None: ...


class EventBusPort(Protocol):
    async def publish_internal(self, event_type: str, payload: dict[str, Any]) -> str: ...
    async def publish_result(self, payload: dict[str, Any]) -> str: ...


class PlannerPort(Protocol):
    async def build_plan(self, payload: PlanInput) -> PlannerPlan: ...


class ExecutorPort(Protocol):
    async def execute_step(
        self,
        run_id: str,
        step: PlannerStep,
        workspace_path: Path,
        run_path: Path,
    ) -> StepExecutionResult: ...
    async def cancel_run(self, run_id: str) -> None: ...


class PolicyPort(Protocol):
    def evaluate_step(self, step: PlannerStep, workspace_root: Path) -> list[PolicyDecision]: ...
    def plan_requires_approval(self, steps: list[PlannerStep], workspace_root: Path) -> tuple[bool, list[PolicyDecision]]: ...


class VerifierPort(Protocol):
    async def run(self, workspace_path: Path) -> VerificationResult: ...


class ArtifactPublisherPort(Protocol):
    async def package(
        self,
        run_id: str,
        task_id: str,
        workspace_path: Path,
        status: str,
        summary: dict[str, Any],
    ) -> dict[str, str]: ...

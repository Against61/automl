from __future__ import annotations

import asyncio
import json
import os
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.runtime.bus import InMemoryEventBus
from orchestrator.execution.codex_runner import CodexRunner, StepExecutionResult
from orchestrator.config import Settings
from orchestrator.persistence.db import Database
from orchestrator.planning.planner import CodexOnlyPlanner, PlanInput, Planner, PlannerError, StubPlanner, make_planner
from orchestrator.execution.policy import PolicyEngine
from orchestrator.planning.ralph import RalphBacklogService
from orchestrator.persistence.schemas import Priority, PlannerPlan, PlannerStep, RunStatus, StepIOResult, TaskPayload, TaskSubmittedEvent
from orchestrator.runtime.session import SessionManager
from orchestrator.execution.verifier import Verifier, VerificationResult
from orchestrator.planning.planner import _sanitize_planner_payload
from streamlit_app import _extract_latest_improvement_strategy


class StaticPlanner(Planner):
    def __init__(self, plan: PlannerPlan):
        self.plan = plan

    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        return self.plan

    async def replan(self, payload: PlanInput, failure_reason: str, previous_plan: PlannerPlan) -> PlannerPlan:
        return self.plan


class FailingPlanner(Planner):
    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        raise PlannerError("planner unavailable")


class MissingFileReplanPlanner(Planner):
    def __init__(self) -> None:
        self.replanned = False

    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        return PlannerPlan(
            version="1.0",
            summary="run missing file command and recover",
            steps=[
                {
                    "id": "s1",
                    "title": "run missing file",
                    "action": "shell",
                    "instruction": "run missing script",
                    "command": "python missing_smoke_test.py",
                    "risk_level": "low",
                },
                {
                    "id": "s2",
                    "title": "post-check",
                    "action": "read",
                    "instruction": "should not reach without replan",
                    "risk_level": "low",
                },
            ],
        )

    async def replan(self, payload: PlanInput, failure_reason: str, previous_plan: PlannerPlan) -> PlannerPlan:
        self.replanned = True
        return PlannerPlan(
            version="1.0",
            summary=f"recovered from: {failure_reason[:120]}",
            steps=[
                {
                    "id": "r1",
                    "title": "recover by skipping shell",
                    "action": "read",
                    "instruction": "skip shell step after missing artifact",
                    "risk_level": "low",
                }
            ],
        )


class QualityReplanPlanner(Planner):
    def __init__(self) -> None:
        self.replanned = False

    async def build_plan(self, payload: PlanInput) -> PlannerPlan:
        return PlannerPlan(
            version="1.0",
            summary="write low quality metric",
            steps=[
                {
                    "id": "q1",
                    "title": "emit baseline metric",
                    "action": "shell",
                    "step_intent": "run_training",
                    "instruction": "write low val accuracy",
                    "command": "printf 'val_accuracy: 0.80\\n' > metrics.md",
                    "risk_level": "low",
                }
            ],
        )

    async def replan(self, payload: PlanInput, failure_reason: str, previous_plan: PlannerPlan) -> PlannerPlan:
        self.replanned = True
        return PlannerPlan(
            version="1.0",
            summary=f"improve metric after: {failure_reason[:120]}",
            steps=[
                {
                    "id": "q2",
                    "title": "emit improved metric",
                    "action": "shell",
                    "step_intent": "run_training",
                    "instruction": "write improved val accuracy",
                    "command": "printf 'val_accuracy: 0.96\\n' > metrics.md",
                    "risk_level": "low",
                }
            ],
        )


async def create_session(
    tmp_path: Path,
    planner: Planner | None = None,
    timeout_sec: int = 2,
) -> tuple[SessionManager, Database, InMemoryEventBus, None, Settings]:
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_cli_cmd="cat",
        codex_step_timeout_sec=timeout_sec,
        verify_commands="",
        poll_interval_sec=0.5,
    )
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)

    db = Database(settings.sqlite_path)
    await db.connect()
    bus = InMemoryEventBus()
    await bus.initialize()
    session = SessionManager(
        settings=settings,
        db=db,
        bus=bus,
        planner=planner or StubPlanner(),
        policy_engine=PolicyEngine(settings),
        ralph_backlog=RalphBacklogService(
            prd_file_name=settings.ralph_prd_file_name,
            progress_file_name=settings.ralph_progress_file_name,
        ),
        codex_runner=CodexRunner(settings),
        verifier=Verifier(settings),
        artifact_publisher=ArtifactPublisher(db, settings.runs_root),
    )
    return session, db, bus, None, settings


def test_codex_command_normalization_drops_full_auto(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_cli_cmd="codex exec --full-auto --skip-git-repo-check",
    )
    runner = CodexRunner(settings)
    normalized = runner._normalize_codex_command(shlex.split(settings.codex_cli_cmd))
    assert "--full-auto" not in normalized
    assert "--dangerously-bypass-approvals-and-sandbox" in normalized


def test_make_planner_returns_codex_only_planner():
    assert isinstance(make_planner(), CodexOnlyPlanner)


@pytest.mark.asyncio
async def test_database_lists_runs_by_status(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    db = Database(settings.sqlite_path)
    await db.connect()
    try:
        task_event = TaskSubmittedEvent(
            event_id=uuid4(),
            event_type="task.submitted",
            schema_version="1.0",
            task_id=uuid4(),
            workspace_id="demo",
            priority=Priority.normal,
            payload=TaskPayload(goal="demo goal", constraints=[], pdf_scope=[]),
        )
        await db.upsert_task(task_event)
        run_id = await db.create_or_get_run(
            task_id=str(task_event.task_id),
            workspace_id="demo",
            priority=Priority.normal,
        )
        await db.update_run_status(run_id, RunStatus.EXECUTING)

        runs = await db.list_runs(statuses=[RunStatus.EXECUTING])

        assert [run.run_id for run in runs] == [run_id]
        assert runs[0].status == RunStatus.EXECUTING
    finally:
        await db.close()


def test_codex_soft_failure_detection(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    runner = CodexRunner(settings)
    result = runner._result_has_codex_soft_failure(
        StepExecutionResult(
            status="completed",
            exit_code=0,
            summary="command completed",
            stdout_text="I can’t run file operations here due to a sandbox restriction.",
            stderr_text="",
            duration_ms=1,
            command="codex exec",
        )
    )
    assert result is True


def test_check_step_search_no_matches_is_non_fatal(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    runner = CodexRunner(settings)
    step = PlannerStep(
        id="check-rg",
        title="inspect codebase",
        action="shell",
        step_type="check",
        instruction="inspect files",
        command="rg does-not-exist",
        risk_level="low",
    )
    result = StepExecutionResult(
        status="failed",
        exit_code=1,
        summary="command failed",
        stdout_text="",
        stderr_text="",
        duration_ms=10,
        command="rg does-not-exist",
    )
    assert runner._is_non_fatal_search_failure(step=step, command="rg does-not-exist", result=result) is True


def test_python_runtime_command_repairs_script_path_and_smoke_flag(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "train_fashion_mnist.py").write_text("print('ok')\n", encoding="utf-8")

    runner = CodexRunner(settings)
    normalized = runner._normalize_python_runtime_command(
        "python train_fashionmnist.py --smoke_test",
        workspace,
    )
    assert "train_fashion_mnist.py" in normalized
    assert "--mode" in normalized
    assert "smoke" in normalized


def test_shell_command_normalizes_workspace_name_prefix_inside_workspace_root(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace" / "demo",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)

    runner = CodexRunner(settings)
    command = (
        "ls -la demo && "
        "find demo -maxdepth 2 -type f && "
        "python demo/scripts/run.py && "
        "cat demo/prd.json"
    )
    normalized = runner._sanitize_shell_command(command, workspace)

    assert "ls -la ." in normalized
    assert "find . -maxdepth 2 -type f" in normalized
    assert "python ./scripts/run.py" in normalized
    assert "cat ./prd.json" in normalized
    assert " demo/" not in normalized


def test_shell_command_rewrites_missing_relative_file_argument_to_existing_workspace_file(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace" / "demo",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    (workspace / "ralph").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "smoke_fashionmnist.py").write_text("print('ok')\n", encoding="utf-8")
    (workspace / "ralph" / "metrics.py").write_text("print('ok')\n", encoding="utf-8")

    runner = CodexRunner(settings)
    normalized = runner._sanitize_shell_command(
        "python -m py_compile scripts/metrics.py scripts/smoke_fashionmnist.py",
        workspace,
    )

    assert "python -m py_compile" in normalized
    assert "ralph/metrics.py" in normalized
    assert "scripts/smoke_fashionmnist.py" in normalized
    assert "scripts/metrics.py" not in normalized


@pytest.mark.asyncio
async def test_set_approved_preserves_waiting_plan_review_status(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    db = Database(settings.sqlite_path)
    await db.connect()
    event = TaskSubmittedEvent(
        event_id=str(uuid4()),
        event_type="task.submitted",
        schema_version="1.0",
        task_id=str(uuid4()),
        workspace_id="demo",
        priority=Priority.normal,
        payload=TaskPayload(goal="demo"),
    )
    await db.upsert_task(event)
    run_id = await db.create_or_get_run(
        task_id=str(event.task_id),
        workspace_id=event.workspace_id,
        priority=event.priority,
    )
    await db.update_run_status(run_id, RunStatus.WAITING_PLAN_REVIEW, "needs review")

    changed = await db.set_approved(run_id)
    assert changed is True

    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.WAITING_PLAN_REVIEW
    assert run.approved_at is not None

    await db.close()


def test_python_runtime_command_switches_to_module_for_src_layout(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "train.py").write_text(
        "from src.data_loader import get_loader\nprint('ok')\n",
        encoding="utf-8",
    )

    runner = CodexRunner(settings)
    normalized = runner._normalize_python_runtime_command(
        "python src/train.py --epochs 1",
        workspace,
    )
    assert "python -m src.train" in normalized
    assert "--epochs 1" in normalized


def test_python_runtime_command_adds_pythonpath_for_sibling_workspace_imports(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    (workspace / "ralph").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "smoke_fashionmnist.py").write_text(
        "from ralph.metrics import write_json_atomic\nprint('ok')\n",
        encoding="utf-8",
    )
    (workspace / "ralph" / "metrics.py").write_text("def write_json_atomic(*args, **kwargs):\n    return None\n", encoding="utf-8")

    runner = CodexRunner(settings)
    normalized = runner._normalize_python_runtime_command(
        "python -u scripts/smoke_fashionmnist.py --epochs 1",
        workspace,
    )
    assert normalized.startswith("PYTHONPATH=. python -u scripts/smoke_fashionmnist.py")


def test_local_module_reference_is_not_auto_installed(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    runner = CodexRunner(settings)
    assert runner._is_local_module_reference("src", workspace) is True
    assert runner._is_local_module_reference("src.data_loader", workspace) is True
    assert runner._is_local_module_reference("torch", workspace) is False


def test_streamlit_strategy_display_falls_back_to_run_verification_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    run_id = "run-ui-fallback"
    run_dir = tmp_path / "workspace" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "verification.latest.json").write_text(
        json.dumps(
            {
                "attempt": 2,
                "improvement_strategy": {
                    "objective": {"metric_key": "accuracy", "current_value": 0.88, "target": 0.95, "unit": "ratio"},
                    "chosen_intervention_id": "targeted_finetune",
                    "chosen_intervention": {"id": "targeted_finetune", "type": "fine_tuning"},
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    strategy = _extract_latest_improvement_strategy({"run_id": run_id, "verification_json": None})
    assert strategy is not None
    assert strategy["chosen_intervention_id"] == "targeted_finetune"


@pytest.mark.asyncio
async def test_runner_writes_stepio_result_file(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    run_path = settings.runs_root / "run-1"
    workspace.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)

    runner = CodexRunner(settings)
    step = PlannerStep(
        id="step-shell-metrics",
        title="emit metrics",
        action="shell",
        step_type="check",
        command="python -c \"print('accuracy: 0.95')\"",
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-1",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "completed"
    stepio_path = run_path / "step-shell-metrics.step_result.json"
    assert stepio_path.exists()
    payload = json.loads(stepio_path.read_text(encoding="utf-8"))
    assert payload["version"] == "stepio.v1"
    assert payload["step_id"] == "step-shell-metrics"
    assert payload["status"] == "completed"
    assert "accuracy" in payload["metrics"]


@pytest.mark.asyncio
async def test_training_shell_step_uses_idle_watchdog_instead_of_hard_timeout(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_step_timeout_sec=1,
        training_idle_timeout_sec=3,
        training_max_wall_clock_sec=0,
    )
    workspace = settings.workspace_root
    run_path = settings.runs_root / "run-training-heartbeat"
    workspace.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)

    runner = CodexRunner(settings)
    step = PlannerStep(
        id="train-heartbeat",
        title="run training heartbeat",
        action="shell",
        step_intent="run_training",
        command="python -c \"import time; [print(f'epoch={i}', flush=True) or time.sleep(0.6) for i in range(3)]\"",
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-training-heartbeat",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "completed"
    assert result.duration_ms >= 1500
    assert "epoch=0" in result.stdout_text


@pytest.mark.asyncio
async def test_training_shell_step_times_out_on_idle_without_output(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_step_timeout_sec=5,
        training_idle_timeout_sec=1,
        training_max_wall_clock_sec=0,
    )
    workspace = settings.workspace_root
    run_path = settings.runs_root / "run-training-idle-timeout"
    workspace.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)

    runner = CodexRunner(settings)
    step = PlannerStep(
        id="train-idle",
        title="run silent training",
        action="shell",
        step_intent="run_training",
        command="python -c \"import time; time.sleep(2.2)\"",
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-training-idle-timeout",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "timeout"
    assert "idle timeout" in result.summary
    assert "training-idle-watchdog" in result.stderr_text


def test_runner_detects_missing_module_from_stdout_payload(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    runner = CodexRunner(settings)
    result = StepExecutionResult(
        status="failed",
        exit_code=1,
        summary="command failed: python run_task.py",
        stdout_text="PyTorch import failed: No module named 'torch'\n{\"status\":\"FAIL\"}",
        stderr_text="",
        duration_ms=10,
        command="python run_task.py",
    )

    assert runner._missing_module_name_from_result(result) == "torch"


def test_runner_ignores_stale_metrics_for_codex_edit_steps(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    run_path = settings.runs_root / "run-codex-edit"
    workspace.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)

    runner = CodexRunner(settings)
    step = PlannerStep(
        id="step-codex-edit",
        title="update report writer",
        action="codex",
        instruction="modify code only",
        step_intent="modify_file",
        risk_level="low",
    )
    result = StepExecutionResult(
        status="completed",
        exit_code=0,
        summary="command completed",
        stdout_text="No training was started.",
        stderr_text="Held-out evaluation accuracy: 84.20%\nTest samples available but not used for reported accuracy: 10000",
        duration_ms=1,
        command="codex exec --model gpt-5.4",
    )
    runner._write_stepio_result(
        run_id="run-codex-edit",
        step=step,
        result=result,
        workspace_path=workspace,
        run_path=run_path,
    )

    payload = json.loads((run_path / "step-codex-edit.step_result.json").read_text(encoding="utf-8"))
    assert payload["metrics"] == {}


def test_policy_network_block_does_not_match_substrings(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        block_network_commands=True,
    )
    engine = PolicyEngine(settings)
    step = PlannerStep(
        id="safe-encoding",
        title="safe python command",
        action="shell",
        instruction="run safe inline python",
        command=(
            "python -c \"from pathlib import Path; "
            "Path('metrics.md').write_text('accuracy: 0.91\\\\n', encoding='utf-8')\""
        ),
        risk_level="low",
    )
    decisions = engine.evaluate_step(step, settings.workspace_root)
    assert not any(
        decision.decision == "DENY" and "network command blocked" in decision.reason
        for decision in decisions
    )


def test_policy_denies_over_budget_training_step_based_on_script_defaults(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "train_fashionmnist.py").write_text(
        "\n".join(
            [
                "import argparse",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--epochs', type=int, default=12)",
                "parser.add_argument('--trial-count', type=int, default=3)",
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=workspace_root,
        pdf_root=workspace_root / "knowledge" / "pdfs",
        runs_root=workspace_root / "runs",
        cpu_training_budget_epoch_trials_limit=20,
    )
    engine = PolicyEngine(settings)
    step = PlannerStep(
        id="budgeted-train",
        title="run heavy training",
        action="shell",
        instruction="run training script",
        step_intent="run_training",
        command="python train_fashionmnist.py --epochs 10",
        risk_level="low",
    )

    decisions = engine.evaluate_step(step, workspace_root)

    assert any(
        decision.decision == "DENY"
        and "training budget exceeds cpu guardrail" in decision.reason
        and "epochs(10) * trial_count(3) = 30 > 20" in decision.reason
        for decision in decisions
    )


def test_policy_allows_training_step_within_budget(tmp_path: Path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "train_fashionmnist.py").write_text(
        "\n".join(
            [
                "import argparse",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--epochs', type=int, default=12)",
                "parser.add_argument('--trial-count', type=int, default=1)",
            ]
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=workspace_root,
        pdf_root=workspace_root / "knowledge" / "pdfs",
        runs_root=workspace_root / "runs",
        cpu_training_budget_epoch_trials_limit=20,
    )
    engine = PolicyEngine(settings)
    step = PlannerStep(
        id="budgeted-train-ok",
        title="run bounded training",
        action="shell",
        instruction="run training script",
        step_intent="run_training",
        command="python train_fashionmnist.py --epochs 10",
        risk_level="low",
    )

    decisions = engine.evaluate_step(step, workspace_root)

    assert not any(
        decision.decision == "DENY" and "training budget exceeds cpu guardrail" in decision.reason
        for decision in decisions
    )


@pytest.mark.asyncio
async def test_process_run_reconciles_stepio_results_and_completes_run(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="two-step reconciled plan",
        steps=[
            {
                "id": "s1",
                "title": "prepare",
                "action": "read",
                "instruction": "already done",
                "risk_level": "low",
            },
            {
                "id": "s2",
                "title": "verify",
                "action": "read",
                "instruction": "already done",
                "risk_level": "low",
            },
        ],
    )
    session, db, _, _, settings = await create_session(tmp_path, planner=StaticPlanner(plan))
    event = TaskSubmittedEvent(
        event_id=uuid4(),
        event_type="task.submitted",
        schema_version="1.0",
        task_id=uuid4(),
        workspace_id="demo",
        priority=Priority.normal,
        payload=TaskPayload(goal="reconcile stepio artifacts", constraints=[]),
    )
    run_id = await session.submit_task_event(event.model_dump(mode="json"))
    assert run_id is not None

    workspace_path = settings.workspace_root / "demo"
    workspace_path.mkdir(parents=True, exist_ok=True)
    run_path = settings.runs_root / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    await db.set_context(run_id, [])
    await db.set_plan(run_id, plan.model_dump(mode="json"))
    await db.update_run_status(run_id, RunStatus.EXECUTING)

    for step_id in ("s1", "s2"):
        payload = StepIOResult(
            run_id=run_id,
            step_id=step_id,
            status="completed",
            error_code="none",
            summary="recovered from stepio",
            operation="general",
            intent="check",
            command=None,
            metrics={},
            duration_ms=0,
        )
        (run_path / f"{step_id}.step_result.json").write_text(
            json.dumps(payload.model_dump(mode="json"), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    await session.process_run(run_id)

    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert await db.count_attempted_steps(run_id) == 2


@pytest.mark.asyncio
async def test_reconcile_stepio_ignores_stale_artifacts_from_previous_execution_cycle(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="single-step recovery plan",
        steps=[
            {
                "id": "s1",
                "title": "prepare",
                "action": "read",
                "instruction": "recover current cycle only",
                "risk_level": "low",
            },
            {
                "id": "s2",
                "title": "next step",
                "action": "read",
                "instruction": "keep run in progress",
                "risk_level": "low",
            },
        ],
    )
    session, db, _, _, settings = await create_session(tmp_path, planner=StaticPlanner(plan))
    event = TaskSubmittedEvent(
        event_id=uuid4(),
        event_type="task.submitted",
        schema_version="1.0",
        task_id=uuid4(),
        workspace_id="demo",
        priority=Priority.normal,
        payload=TaskPayload(goal="reconcile only current execution cycle", constraints=[]),
    )
    run_id = await session.submit_task_event(event.model_dump(mode="json"))
    assert run_id is not None

    workspace_path = settings.workspace_root / "demo"
    workspace_path.mkdir(parents=True, exist_ok=True)
    run_path = settings.runs_root / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    await db.set_context(run_id, [])
    await db.set_plan(run_id, plan.model_dump(mode="json"))
    await db.update_run_status(run_id, RunStatus.EXECUTING)

    old_payload = StepIOResult(
        run_id=run_id,
        step_id="s1",
        status="completed",
        error_code="none",
        summary="old cycle artifact",
        operation="general",
        intent="check",
        command=None,
        metrics={},
        duration_ms=0,
    )
    old_path = run_path / "s1.step_result.json"
    old_path.write_text(json.dumps(old_payload.model_dump(mode="json"), ensure_ascii=True, indent=2), encoding="utf-8")
    old_created_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    os.utime(old_path, (old_created_at.timestamp(), old_created_at.timestamp()))
    await db.insert_run_step(
        run_id=run_id,
        step_id="s1",
        step_title="prepare",
        step_index=0,
        action="read",
        command=None,
        status="completed",
        stdout_text="summary: old cycle artifact",
        stderr_text="",
        duration_ms=0,
        created_at=old_created_at.isoformat(),
    )

    next_cycle = await db.advance_execution_cycle(run_id)
    assert next_cycle == 1
    run = await db.get_run(run_id)
    assert run is not None
    assert run.execution_cycle == 1
    assert run.cycle_started_at is not None

    new_payload = StepIOResult(
        run_id=run_id,
        step_id="s1",
        status="completed",
        error_code="none",
        summary="current cycle artifact",
        operation="general",
        intent="check",
        command=None,
        metrics={},
        duration_ms=0,
    )
    new_path = run_path / "s1.step_result.2.json"
    new_path.write_text(json.dumps(new_payload.model_dump(mode="json"), ensure_ascii=True, indent=2), encoding="utf-8")

    latest = await session._process_run_uc.stepio_recovery_service.reconcile_stepio_artifacts(
        run=run,
        run_path=run_path,
    )
    assert latest["s1"].summary == "current cycle artifact"

    refreshed = await db.get_run(run_id)
    assert refreshed is not None
    changed = await session._process_run_uc.stepio_recovery_service.sync_run_progress_from_stepio(
        run=refreshed,
        latest_by_step_id=latest,
    )
    assert changed is True

    final_run = await db.get_run(run_id)
    assert final_run is not None
    assert final_run.next_step_index == 1

    steps = await db.list_run_steps(run_id)
    s1_steps = [step for step in steps if step["step_id"] == "s1"]
    assert len(s1_steps) == 2
    assert any("current cycle artifact" in (step["stdout_text"] or "") for step in s1_steps)
    await db.close()


def test_codex_env_does_not_forward_api_key_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        openai_api_key="from-settings",
        codex_use_openai_api_key=False,
    )
    runner = CodexRunner(settings)
    env = runner._sanitized_env()
    assert "OPENAI_API_KEY" not in env


def test_runner_metric_extraction_ignores_non_metric_operational_lines(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    runner = CodexRunner(settings)
    metrics = runner._extract_metrics_from_text(
        "\n".join(
            [
                "session id: 019c71b2-60af-7993-93d0-62b77afb90df",
                "workdir: /app/workspace/demo",
                "Epoch 1/1 | train_loss=0.3412 | val_accuracy=92.58%",
            ]
        )
    )
    assert "session_id" not in metrics
    assert "workdir" not in metrics
    assert metrics["train_loss"] == pytest.approx(0.3412, rel=1e-6)
    assert metrics["val_accuracy"] == pytest.approx(0.9258, rel=1e-6)


@pytest.mark.asyncio
async def test_stub_planner_uses_goal_not_mnist_template():
    planner = StubPlanner()
    payload = PlanInput(
        goal="Train a YOLOv8 detector for barcodes in the demo folder.",
        constraints=["Use Python, no GPU assumptions."],
        contexts=[],
        workspace_id="ws",
    )
    plan = await planner.build_plan(payload)

    assert any("Train a YOLOv8 detector for barcodes" in (step.codex_prompt or step.instruction) for step in plan.steps)
    assert all("MNIST" not in (step.codex_prompt or step.instruction) for step in plan.steps)
    assert any("disjoint" in (step.codex_prompt or step.instruction).lower() for step in plan.steps)
    assert any("overfit check" in (step.codex_prompt or step.instruction).lower() for step in plan.steps)
    assert plan.summary != ""


@pytest.mark.asyncio
async def test_stub_planner_is_repo_agnostic():
    planner = StubPlanner()
    payload = PlanInput(
        goal="Inspect workspace and prepare a training script.",
        constraints=[],
        contexts=[],
        workspace_id="ws",
    )
    plan = await planner.build_plan(payload)

    shell_commands = [command for step in plan.steps if step.action == "shell" for command in step.commands]

    assert shell_commands
    assert all("git status --porcelain" not in command for command in shell_commands)
    assert all("git diff --stat" not in command for command in shell_commands)
    assert "pwd" in shell_commands
    assert any(command.startswith("ls -la") for command in shell_commands)
    assert any(command.startswith("find . -maxdepth 3 -type f") for command in shell_commands)


@pytest.mark.asyncio
async def test_stub_planner_quality_target_emits_shell_training_step():
    planner = StubPlanner()
    payload = PlanInput(
        goal="обучи сегментатор на датасете в workspace",
        constraints=["RALPH_REQUIRED_METRIC: iou >= 95%"],
        contexts=[],
        workspace_id="ws",
    )
    plan = await planner.build_plan(payload)

    assert plan.summary == "Stub training plan"
    preflight_step = next(step for step in plan.steps if step.id == "step-3")
    training_step = next(step for step in plan.steps if step.id == "step-4")
    assert any(step.step_intent.value == "run_training" and step.action == "shell" for step in plan.steps)
    assert preflight_step.action == "codex"
    assert "preflight/debug loop" in (preflight_step.instruction or "")
    assert (training_step.command or "") == "python run_task.py --metrics-path metrics.json"
    assert any(step.step_intent.value == "verify_metrics" and step.action == "verify" for step in plan.steps)


@pytest.mark.asyncio
async def test_stub_planner_routes_first_preflight_through_codex_by_default():
    planner = StubPlanner()
    payload = PlanInput(
        goal="обучи сегментатор на датасете в workspace",
        constraints=["RALPH_REQUIRED_METRIC: iou >= 95%"],
        contexts=[],
        workspace_id="ws",
    )

    plan = await planner.build_plan(payload)

    preflight_step = next(step for step in plan.steps if step.id == "step-3")
    training_step = next(step for step in plan.steps if step.id == "step-4")

    assert preflight_step.action == "codex"
    assert "Codex-owned preflight/debug loop" in (preflight_step.instruction or "")
    assert training_step.action == "shell"
    assert (training_step.command or "") == "python run_task.py --metrics-path metrics.json"


@pytest.mark.asyncio
async def test_stub_planner_ralph_preparatory_story_emits_analysis_plan():
    planner = StubPlanner()
    payload = PlanInput(
        goal=(
            "Implement RALPH story US-001: Определить тип и целевые классы датасета\n\n"
            "Story description:\n"
            "Проанализировать структуру папки coco-segmentation и аннотационных файлов, "
            "убедиться, что это COCO instance segmentation, и зафиксировать список классов.\n\n"
            "Acceptance criteria:\n"
            "- Подтверждено присутствие fields images, annotations, categories, segmentation.\n"
            "- Составлен точный список классов.\n\n"
            "Primary user goal:\n"
            "обучи сегментатор"
        ),
        constraints=[
            "RALPH_STORY_ID: US-001",
            "RALPH_REQUIRED_METRIC: iou >= 95%",
        ],
        contexts=[],
        workspace_id="ws",
    )
    plan = await planner.build_plan(payload)

    assert plan.summary == "Stub Ralph preparatory plan (training is deferred)"
    assert any(step.action == "shell" and (step.command or "") == "python run_task.py" for step in plan.steps)
    codex_step = next(step for step in plan.steps if step.action == "codex")
    assert "Do not depend on PyTorch" in (codex_step.instruction or "")
    assert "planning_only_report_detected" in (codex_step.instruction or "")


@pytest.mark.asyncio
async def test_stub_planner_segmentation_training_prompt_forbids_synthetic_smoke():
    planner = StubPlanner()
    payload = PlanInput(
        goal="обучи сегментатор на coco-segmentation и отчитай IoU",
        constraints=[
            "RALPH_REQUIRED_METRIC: iou >= 95%",
            "TASK_FAMILY: segmentation",
            "PRIMARY_METRIC_KEY: iou",
            "PREFERRED_METRICS: iou, mean_iou, dice",
            "REAL_DATASET_SMOKE_REQUIRED: true",
        ],
        contexts=[],
        workspace_id="ws",
    )
    plan = await planner.build_plan(payload)

    codex_step = next(step for step in plan.steps if step.action == "codex")
    assert "requires real-dataset evidence" in (codex_step.instruction or "")
    assert "real subset of the dataset" in (codex_step.instruction or "")
    assert "iou, mean_iou, dice" in (codex_step.instruction or "")
    assert "--preflight" in (codex_step.instruction or "")
    assert "file_name" in (codex_step.instruction or "")


def test_planner_payload_sanitizes_retry_policy_reasons():
    payload = {
        "version": "1.0",
        "summary": "demo",
        "steps": [
            {
                "id": "s1",
                "title": "demo step",
                "action": "codex",
                "instruction": "do work",
                "retry_policy": {
                    "max_retries": 2,
                    "on": ["quality_gate", "quality_threshold_not_met", "timeout", "missing_artifact"],
                },
            }
        ],
    }
    sanitized = _sanitize_planner_payload(payload)
    assert sanitized["steps"][0]["retry_policy"]["on"] == [
        "execution_error",
        "infra_error",
        "missing_file",
    ]


def test_planner_payload_collects_sanitization_details():
    payload = {
        "version": "1.0",
        "summary": "demo",
        "steps": [
            {
                "id": "s1",
                "title": "demo step",
                "action": "codex",
                "instruction": "do work",
                "retry_policy": {
                    "max_retries": 2,
                    "on": ["quality_threshold_not_met", "timeout"],
                },
            }
        ],
    }
    sanitized, changes = _sanitize_planner_payload(payload, collect_changes=True)
    assert sanitized["steps"][0]["retry_policy"]["on"] == [
        "execution_error",
        "infra_error",
    ]
    assert changes == [
        {
            "step_id": "s1",
            "field": "retry_policy.on",
            "original": "quality_threshold_not_met",
            "normalized": "execution_error",
        },
        {
            "step_id": "s1",
            "field": "retry_policy.on",
            "original": "timeout",
            "normalized": "infra_error",
        },
    ]


def test_planner_payload_normalizes_run_training_action_to_shell():
    payload = {
        "version": "1.0",
        "summary": "demo",
        "steps": [
            {
                "id": "s1",
                "title": "run training",
                "action": "codex",
                "step_intent": "run_training",
                "instruction": "execute training",
                "command": "python train.py --epochs 5",
                "commands": ["python train.py --epochs 5"],
            }
        ],
    }
    sanitized, changes = _sanitize_planner_payload(payload, collect_changes=True)
    assert sanitized["steps"][0]["action"] == "shell"
    assert any(change["field"] == "action" and change["normalized"] == "shell" for change in changes)


def test_planner_payload_normalizes_confused_step_schema_fields():
    payload = {
        "version": "1.0",
        "summary": "demo",
        "steps": [
            {
                "id": "s-train",
                "title": "run smoke test training",
                "step_type": "run_training",
                "operation": "shell",
                "action": "shell",
                "step_intent": "run_training",
                "instruction": "train for 3 epochs",
                "command": "python train.py --epochs 3",
            },
            {
                "id": "s-verify",
                "title": "verify metrics",
                "step_type": "check",
                "operation": "verify",
                "action": "verify",
                "step_intent": "verify_metrics",
                "instruction": "verify accuracy from metrics.md",
            },
        ],
    }
    sanitized, changes = _sanitize_planner_payload(payload, collect_changes=True)

    assert sanitized["steps"][0]["step_type"] == "check"
    assert sanitized["steps"][0]["operation"] == "run_training"
    assert sanitized["steps"][0]["action"] == "shell"
    assert sanitized["steps"][1]["step_type"] == "check"
    assert sanitized["steps"][1]["operation"] == "verify_metrics"
    assert sanitized["steps"][1]["action"] == "verify"
    assert {"field": "step_type", "original": "run_training", "normalized": "check"} in [
        {k: change[k] for k in ("field", "original", "normalized")}
        for change in changes
    ]
    assert {"field": "operation", "original": "shell", "normalized": "run_training"} in [
        {k: change[k] for k in ("field", "original", "normalized")}
        for change in changes
    ]
    assert {"field": "operation", "original": "verify", "normalized": "verify_metrics"} in [
        {k: change[k] for k in ("field", "original", "normalized")}
        for change in changes
    ]
    PlannerPlan.model_validate(sanitized)


def test_planner_step_requires_explicit_command_for_run_training():
    with pytest.raises(ValueError, match="run_training steps must declare explicit shell commands"):
        PlannerStep(
            id="train-1",
            title="run training",
            action="codex",
            step_intent="run_training",
            instruction="train the model to target accuracy",
            risk_level="medium",
        )


def test_planner_step_run_training_with_command_becomes_shell():
    step = PlannerStep(
        id="train-2",
        title="run training",
        action="codex",
        step_intent="run_training",
        instruction="execute training script",
        command="python train.py --epochs 5",
        risk_level="medium",
    )
    assert step.action == "shell"
    assert step.command == "python train.py --epochs 5"
    assert step.commands == ["python train.py --epochs 5"]


@pytest.mark.asyncio
async def test_process_run_publishes_planner_sanitization_event(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="sanitized plan",
        planner_sanitization=[
            {
                "step_id": "s1",
                "field": "retry_policy.on",
                "original": "quality_threshold_not_met",
                "normalized": "execution_error",
            }
        ],
        steps=[
            {
                "id": "s1",
                "title": "noop read step",
                "action": "read",
                "instruction": "inspect state",
                "risk_level": "low",
            }
        ],
    )
    session, db, bus, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan))
    run_id = await session.submit_task_event(task_event(goal="publish sanitization event"))
    assert run_id

    await session.process_run(run_id)

    assert any(
        item.get("event_type") == "planner.payload_sanitized" and item.get("run_id") == run_id
        for item in bus.internal
    )
    run = await db.get_run(run_id)
    assert run is not None
    assert run.plan_json is not None
    assert run.plan_json.get("planner_sanitization")
    await db.close()


@pytest.mark.asyncio
async def test_ralph_prd_bootstrap_plan_has_explicit_file_edit_intent(tmp_path: Path):
    session, _, _, _, _ = await create_session(tmp_path)
    workspace_path = tmp_path / "workspace" / "demo"
    workspace_path.mkdir(parents=True, exist_ok=True)

    task = {
        "goal": "Train a detector",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 95%"]),
        "payload_json": json.dumps({"payload": {"execution_mode": "ralph_story"}}),
    }
    run = type(
        "Run",
        (),
        {
            "run_id": "run-ralph-bootstrap",
            "error_message": None,
            "plan_json": None,
            "verification_json": None,
        },
    )()
    plan = await session._process_run_uc.ralph_service.build_prd_bootstrap_plan(
        task=task,
        run=run,
        contexts=[],
        workspace_path=workspace_path,
        last_failed_step=None,
        previous_verification=None,
    )

    step = plan.steps[0]
    assert step.id == "ralph-prd-bootstrap"
    assert step.step_intent.value == "modify_file"
    assert step.operation.value == "edit_code"


@pytest.mark.asyncio
async def test_quality_gate_skip_reason_for_planning_only_plan(tmp_path: Path):
    session, _, _, _, _ = await create_session(tmp_path)
    run = type(
        "Run",
        (),
        {
            "plan_json": {
                "summary": "Planning-only artifact definition for detector workflow",
                "steps": [
                    {
                        "id": "s1",
                        "title": "Create planning artifact",
                        "action": "codex",
                        "step_intent": "modify_file",
                        "instruction": "No training is allowed in this step.",
                    }
                ],
            }
        },
    )()
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={},
    )

    reason = session._process_run_uc.execution_guard_service.quality_gate_skip_reason(
        run=run,
        verification=verification,
    )
    assert reason == "quality gate skipped: current plan is planning-only"


@pytest.mark.asyncio
async def test_quality_gate_skip_reason_does_not_skip_real_training_plan(tmp_path: Path):
    session, _, _, _, _ = await create_session(tmp_path)
    run = type(
        "Run",
        (),
        {
            "plan_json": {
                "summary": "Train baseline model",
                "steps": [
                    {
                        "id": "train-1",
                        "title": "Run training",
                        "action": "shell",
                        "step_intent": "run_training",
                        "command": "python train.py --epochs 5",
                    }
                ],
            }
        },
    )()
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={"planning_only_report_detected": False},
    )

    reason = session._process_run_uc.execution_guard_service.quality_gate_skip_reason(
        run=run,
        verification=verification,
    )
    assert reason is None


@pytest.mark.asyncio
async def test_planning_only_terminal_skip_reason_overrides_failed_quality_gate(tmp_path: Path):
    session, _, _, _, _ = await create_session(tmp_path)
    reason = session._process_run_uc.verification_stage.verification_flow_service.planning_only_terminal_skip_reason(
        {
            "metrics": {
                "planning_only_report_detected": True,
                "training_deferred": True,
                "dataset_parse_ok": True,
            },
            "quality_gate": {
                "status": "failed",
                "reason": "metric 'iou' not found in verification output",
            },
        }
    )
    assert reason == "quality gate skipped: planning-only artifacts are not evaluated against target metrics"


@pytest.mark.asyncio
async def test_synthetic_real_dataset_smoke_guard_blocks_generated_samples_script(tmp_path: Path):
    session, _, _, _, settings = await create_session(tmp_path)
    workspace = settings.workspace_root / "demo"
    workspace.mkdir(parents=True, exist_ok=True)
    script = workspace / "run_task.py"
    script.write_text(
        "\n".join(
            [
                "import torch",
                "x = torch.randn(16, 2)",
                "print(x)",
            ]
        ),
        encoding="utf-8",
    )
    step = PlannerStep(
        id="train-seg",
        title="run segmentation smoke",
        action="shell",
        step_intent="run_training",
        command="python run_task.py",
        risk_level="low",
    )

    reason = session._process_run_uc.execution_guard_service.synthetic_real_dataset_smoke_guard_reason(
        task={
            "goal": "Train segmentation model on coco-segmentation",
            "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
        },
        workspace_path=workspace,
        step=step,
    )

    assert reason is not None
    assert "synthetic data" in reason


@pytest.mark.asyncio
async def test_preflight_dependency_block_reason_detects_missing_torch_metrics(tmp_path: Path):
    session, db, _, _, settings = await create_session(tmp_path)
    workspace = settings.workspace_root / "demo"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "preflight_metrics.json").write_text(
        json.dumps(
            {
                "mode": "preflight",
                "smoke_torch_available": False,
                "error": "PyTorch is required for smoke execution: No module named 'torch'",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    step = PlannerStep(
        id="train-seg",
        title="run segmentation smoke",
        action="shell",
        step_intent="run_training",
        command="python run_task.py --metrics-path metrics.json",
        risk_level="low",
    )

    reason = session._process_run_uc.execution_guard_service.preflight_dependency_block_reason(
        workspace_path=workspace,
        step=step,
    )

    assert reason is not None
    assert "dependency recovery required" in reason
    assert "torch" in reason.lower()
    await db.close()


@pytest.mark.asyncio
async def test_structured_dependency_failure_reason_detects_missing_torch_from_metrics_payload(tmp_path: Path):
    session, db, _, _, settings = await create_session(tmp_path)
    workspace = settings.workspace_root / "demo"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.json").write_text(
        json.dumps(
            {
                "mode": "smoke_torch_missing",
                "status": "failed",
                "error": "PyTorch is required for smoke execution: No module named 'torch'",
                "quality_gate_applies": False,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    step = PlannerStep(
        id="train-seg",
        title="run segmentation smoke",
        action="shell",
        step_intent="run_training",
        command="python run_task.py --metrics-path metrics.json",
        expected_artifacts=[{"path": "metrics.json", "kind": "metrics", "must_exist": True}],
        risk_level="low",
    )
    result = StepExecutionResult(
        status="failed",
        exit_code=1,
        summary="command failed: python run_task.py --metrics-path metrics.json",
        stdout_text="",
        stderr_text="",
        duration_ms=5,
        command="python run_task.py --metrics-path metrics.json",
    )

    reason = session._process_run_uc.execution_guard_service.structured_dependency_failure_reason(
        workspace_path=workspace,
        step=step,
        result=result,
    )

    assert reason is not None
    assert "dependency recovery required" in reason
    assert "metrics.json" in reason
    await db.close()


@pytest.mark.asyncio
async def test_stub_planner_uses_task_intent_constraints_for_detection_metrics():
    planner = StubPlanner()
    payload = PlanInput(
        goal="обучи модель на наборе изображений",
        constraints=[
            "RALPH_REQUIRED_METRIC: accuracy >= 95%",
            "TASK_FAMILY: detection",
            "PRIMARY_METRIC_KEY: map50",
            "PREFERRED_METRICS: map50, map50_95, precision, recall",
            "REAL_DATASET_SMOKE_REQUIRED: true",
        ],
        contexts=[],
        workspace_id="ws",
    )
    plan = await planner.build_plan(payload)

    verify_step = next(step for step in plan.steps if step.step_intent.value == "verify_metrics")
    codex_step = next(step for step in plan.steps if step.action == "codex")

    assert verify_step.expected_artifacts[0].metric_keys == ["map50"]
    assert "task_family: detection" in (codex_step.instruction or "")
    assert "map50, map50_95, precision, recall" in (codex_step.instruction or "")
    assert any((step.command or "") == "python run_task.py --preflight --metrics-path preflight_metrics.json" for step in plan.steps)


@pytest.mark.asyncio
async def test_missing_python_file_detected_as_missing_artifact(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    runner = CodexRunner(settings)

    workspace = settings.workspace_root
    run_path = workspace / "runs"
    run_path.mkdir(parents=True, exist_ok=True)
    step = PlannerStep(
        id="missing-py",
        title="missing file",
        action="shell",
        instruction="run missing smoke test",
        command="python smoke_test_fashionmnist.py",
        risk_level="low",
    )

    result = await runner.execute_step(
        run_id="run-missing-file",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )

    assert result.status == "failed"
    assert result.exit_code == 2
    assert result.missing_artifact is not None
    assert "MISSING_FILE:" in result.stderr_text
    assert result.is_infra_error is False


@pytest.mark.asyncio
async def test_db_reconnect_repairs_corrupted_database(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    db = Database(settings.sqlite_path)
    await db.connect()
    await db.close()

    settings.sqlite_path.write_bytes(b"not-a-sqlite-database")
    reparing_db = Database(settings.sqlite_path)
    await reparing_db.connect()
    event_task_id = str(uuid4())
    await reparing_db.upsert_task(
        TaskSubmittedEvent(
            event_id=uuid4(),
            event_type="task.submitted",
            schema_version="1.0",
            task_id=event_task_id,
            workspace_id="ws",
            priority=Priority.normal,
            payload=TaskPayload(
                goal="reconnect test",
                constraints=[],
                pdf_scope=[],
            ),
        )
    )

    run_id = await reparing_db.create_or_get_run(
        task_id=event_task_id,
        workspace_id="ws",
        priority=Priority.normal,
    )
    assert run_id
    await reparing_db.close()

    assert any(".corrupt" in name.name for name in settings.sqlite_path.parent.glob("orchestrator.db.*"))


@pytest.mark.asyncio
async def test_repair_database_file_archives_sqlite_sidecars(tmp_path: Path):
    sqlite_path = tmp_path / "orchestrator.db"
    sqlite_path.write_bytes(b"main-corruption")
    sqlite_path.with_name("orchestrator.db-wal").write_bytes(b"wal-corruption")
    sqlite_path.with_name("orchestrator.db-shm").write_bytes(b"shm-corruption")
    sqlite_path.with_name("orchestrator.db-journal").write_bytes(b"journal-corruption")

    db = Database(sqlite_path)
    await db._repair_database_file()

    assert not sqlite_path.exists()
    assert not sqlite_path.with_name("orchestrator.db-wal").exists()
    assert not sqlite_path.with_name("orchestrator.db-shm").exists()
    assert not sqlite_path.with_name("orchestrator.db-journal").exists()
    assert any(path.name.startswith("orchestrator.db.") and path.name.endswith(".corrupt") for path in tmp_path.iterdir())
    assert any(path.name.startswith("orchestrator.db-wal.") and path.name.endswith(".corrupt") for path in tmp_path.iterdir())
    assert any(path.name.startswith("orchestrator.db-shm.") and path.name.endswith(".corrupt") for path in tmp_path.iterdir())
    assert any(path.name.startswith("orchestrator.db-journal.") and path.name.endswith(".corrupt") for path in tmp_path.iterdir())


@pytest.mark.asyncio
async def test_rehydrate_terminal_run_from_result_artifact(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run-123"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "run_id": "run-123",
        "task_id": "task-123",
        "status": "completed",
        "summary": {"planned_steps": 2, "executed_steps": 2, "verification": "passed"},
        "steps": [
            {
                "step_id": "step-1",
                "step_index": 0,
                "action": "codex",
                "command": "python train.py --epochs 5",
                "status": "completed",
                "stdout_text": "accuracy: 0.91",
                "stderr_text": "",
                "duration_ms": 1000,
                "created_at": "2026-03-08T10:00:00+00:00",
            }
        ],
        "artifacts": [
            {
                "kind": "report_json",
                "path": str(run_dir / "legacy-report.json"),
                "created_at": "2026-03-08T10:00:01+00:00",
            }
        ],
        "recovery": {
            "workspace_id": "demo",
            "priority": "high",
            "created_at": "2026-03-08T09:59:00+00:00",
            "updated_at": "2026-03-08T10:00:02+00:00",
            "error_message": None,
            "plan_json": {"version": "1.0", "summary": "recovered plan", "steps": []},
            "context_json": [],
            "verification_json": {"status": "passed", "metrics": {"accuracy": 0.91}},
            "task": {
                "goal": "Recovered training run",
                "constraints": ["Use PyTorch"],
                "pdf_scope": [],
                "payload_json": {
                    "payload": {
                        "goal": "Recovered training run",
                        "constraints": ["Use PyTorch"],
                        "pdf_scope": [],
                        "execution_mode": "plan_execute",
                    }
                },
                "workspace_id": "demo",
                "priority": "high",
                "created_at": "2026-03-08T09:58:00+00:00",
                "updated_at": "2026-03-08T09:58:30+00:00",
            },
            "packaged_artifacts": {
                "patch_bundle_path": str(run_dir / "patches.tar.gz"),
                "report_json_path": str(run_dir / "result.json"),
            },
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    stats = await db.rehydrate_terminal_runs_from_artifacts(runs_root)
    assert stats["restored_tasks"] == 1
    assert stats["restored_runs"] == 1
    assert stats["restored_steps"] == 1
    assert stats["restored_artifacts"] >= 3

    task = await db.get_task("task-123")
    assert task is not None
    assert task["goal"] == "Recovered training run"
    assert task["workspace_id"] == "demo"

    run = await db.get_run("run-123")
    assert run is not None
    assert run.status == RunStatus.COMPLETED
    assert run.workspace_id == "demo"
    assert run.priority == Priority.high
    assert run.verification_json is not None
    assert run.verification_json["metrics"]["accuracy"] == 0.91

    steps = await db.list_run_steps("run-123")
    assert len(steps) == 1
    assert steps[0]["step_id"] == "step-1"

    artifacts = await db.get_artifacts("run-123")
    artifact_kinds = {item["kind"] for item in artifacts}
    assert "report_json" in artifact_kinds
    assert "patch_bundle" in artifact_kinds

    repeat_stats = await db.rehydrate_terminal_runs_from_artifacts(runs_root)
    assert repeat_stats["restored_runs"] == 0
    assert repeat_stats["skipped_existing_runs"] == 1
    await db.close()


@pytest.mark.asyncio
async def test_list_run_steps_returns_chronological_attempts_and_titles(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()

    await db.insert_run_step(
        run_id="run-chron",
        step_id="step-a",
        step_title="First logical step",
        step_index=0,
        action="codex",
        command="echo a",
        status="completed",
        stdout_text="a",
        stderr_text="",
        duration_ms=1,
        created_at="2026-03-09T10:00:00+00:00",
    )
    await db.insert_run_step(
        run_id="run-chron",
        step_id="step-b",
        step_title="Second logical step",
        step_index=2,
        action="shell",
        command="echo b",
        status="failed",
        stdout_text="b",
        stderr_text="",
        duration_ms=1,
        created_at="2026-03-09T10:00:02+00:00",
    )
    await db.insert_run_step(
        run_id="run-chron",
        step_id="step-c",
        step_title="Replan first step",
        step_index=0,
        action="codex",
        command="echo c",
        status="completed",
        stdout_text="c",
        stderr_text="",
        duration_ms=1,
        created_at="2026-03-09T10:00:03+00:00",
    )

    steps = await db.list_run_steps("run-chron")
    assert [step["step_id"] for step in steps] == ["step-a", "step-b", "step-c"]
    assert [step["step_title"] for step in steps] == [
        "First logical step",
        "Second logical step",
        "Replan first step",
    ]
    await db.close()


@pytest.mark.asyncio
async def test_missing_python_file_triggers_replan_once(tmp_path: Path):
    planner = MissingFileReplanPlanner()
    session, db, _, _, _ = await create_session(tmp_path, planner=planner)
    run_id = await session.submit_task_event(task_event(goal="Run missing file workflow"))
    assert run_id

    await session.process_run(run_id)
    run_after_failure = await db.get_run(run_id)
    assert run_after_failure is not None
    assert planner.replanned
    assert run_after_failure.status in {RunStatus.COMPLETED, RunStatus.PLAN_READY}
    assert run_after_failure.error_message is None
    steps = await db.list_run_steps(run_id)
    assert any(step["step_id"] == "r1" for step in steps)
    assert any("MISSING_FILE" in (step["stderr_text"] or "") for step in steps)
    await db.close()


@pytest.mark.asyncio
async def test_run_training_threshold_miss_is_deferred_to_quality_gate(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="training threshold miss should not hard-fail execution",
        steps=[
            {
                "id": "s1",
                "title": "run training",
                "action": "shell",
                "step_intent": "run_training",
                "instruction": "simulate training below target threshold",
                "command": (
                    "python -c \"from pathlib import Path; "
                    "Path('metrics.md').write_text('accuracy: 0.91\\\\n', encoding='utf-8'); "
                    "import sys; "
                    "print('RuntimeError: Target accuracy threshold not reached', file=sys.stderr); "
                    "sys.exit(1)\""
                ),
                "expected_artifacts": [
                    {"path": "metrics.md", "kind": "metrics", "must_exist": True, "must_be_nonempty": True}
                ],
                "risk_level": "low",
            },
            {
                "id": "s2",
                "title": "final check",
                "action": "read",
                "instruction": "continue after threshold miss",
                "risk_level": "low",
            },
        ],
    )
    session, db, _, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan))
    run_id = await session.submit_task_event(task_event(goal="training threshold miss flow"))
    assert run_id

    await session.process_run(run_id)
    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED

    steps = await db.list_run_steps(run_id)
    s1 = next(step for step in steps if step["step_id"] == "s1")
    assert s1["status"] == "completed"
    assert "deferred to quality gate" in (s1["stdout_text"] or "")
    await db.close()


@pytest.mark.asyncio
async def test_quality_replan_restarts_plan_from_first_step(tmp_path: Path):
    planner = QualityReplanPlanner()
    session, db, _, _, _ = await create_session(tmp_path, planner=planner)
    event = task_event(goal="Train and satisfy val accuracy target")
    event["payload"]["constraints"] = [
        "RALPH_REQUIRED_METRIC: val_accuracy >= 95%",
        "MAX_QUALITY_RETRIES: 1",
    ]
    run_id = await session.submit_task_event(event)
    assert run_id

    await session.process_run(run_id)
    mid = await db.get_run(run_id)
    assert mid is not None
    assert mid.status == RunStatus.CONTEXT_READY
    assert mid.attempts_by_stage.get("QUALITY_REPLAN") == 1
    assert mid.execution_cycle == 1
    assert mid.cycle_started_at is not None

    await session.process_run(run_id)
    done = await db.get_run(run_id)
    assert done is not None
    assert done.status == RunStatus.COMPLETED
    assert planner.replanned is True

    steps = await db.list_run_steps(run_id)
    assert any(step["step_id"] == "q1" for step in steps)
    assert any(step["step_id"] == "q2" for step in steps)
    await db.close()


@pytest.mark.asyncio
async def test_preflight_dependency_block_fails_run_without_execution_replan(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="single training step blocked by preflight dependency signal",
        steps=[
            {
                "id": "train-1",
                "title": "run training",
                "action": "shell",
                "step_intent": "run_training",
                "instruction": "attempt smoke training",
                "command": "python -c \"from pathlib import Path; Path('touched.txt').write_text('ran', encoding='utf-8')\"",
                "expected_artifacts": [
                    {"path": "metrics.json", "kind": "metrics", "must_exist": True, "must_be_nonempty": True}
                ],
                "risk_level": "low",
            }
        ],
    )
    session, db, _, _, settings = await create_session(tmp_path, planner=StaticPlanner(plan))
    run_id = await session.submit_task_event(task_event(workspace_id="demo", goal="run blocked training"))
    assert run_id

    workspace = settings.workspace_root / "demo"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "preflight_metrics.json").write_text(
        json.dumps(
            {
                "mode": "preflight",
                "smoke_torch_available": False,
                "error": "PyTorch is required for smoke execution: No module named 'torch'",
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    await session.process_run(run_id)
    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.FAILED
    assert "preflight dependency recovery required before training step" in (run.error_message or "")
    assert run.attempts_by_stage.get("EXECUTION_REPLAN", 0) == 0
    assert not (workspace / "touched.txt").exists()

    steps = await db.list_run_steps(run_id)
    train_step = next(step for step in steps if step["step_id"] == "train-1")
    assert train_step["status"] == "failed"
    assert "blocked before execution" in (train_step["stdout_text"] or "")
    await db.close()


@pytest.mark.asyncio
async def test_quality_target_plan_without_shell_training_goes_to_plan_review(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="Stub plan",
        steps=[
            {
                "id": "s1",
                "title": "inspect",
                "action": "shell",
                "instruction": "inspect workspace",
                "commands": ["pwd"],
                "risk_level": "low",
            },
            {
                "id": "s2",
                "title": "verify metrics via codex",
                "action": "codex",
                "step_intent": "verify_metrics",
                "instruction": "check metrics",
                "risk_level": "low",
            },
        ],
    )
    session, db, _, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan))
    event = task_event(goal="Train segmentation model and satisfy IoU target")
    event["payload"]["constraints"] = ["RALPH_REQUIRED_METRIC: IOU >= 98%"]
    run_id = await session.submit_task_event(event)
    assert run_id

    await session.process_run(run_id)
    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.WAITING_PLAN_REVIEW
    assert "without an explicit shell run_training step" in (run.error_message or "")

    steps = await db.list_run_steps(run_id)
    assert steps == []
    await db.close()


@pytest.mark.asyncio
async def test_ralph_bootstrap_only_plan_bypasses_plan_review_for_quality_target(tmp_path: Path):
    session, db, _, _, settings = await create_session(tmp_path)
    workspace = settings.workspace_root / "demo"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".ralph_workspace_tree.md").write_text("demo/\n", encoding="utf-8")
    (workspace / "prd.json").write_text('{"userStories":[{"id":"us1","passes":false}]}', encoding="utf-8")

    event = task_event(workspace_id="demo", goal="обучи сегментатор")
    event["payload"]["execution_mode"] = "ralph_story"
    event["payload"]["constraints"] = ["RALPH_REQUIRED_METRIC: iou >= 95%"]
    run_id = await session.submit_task_event(event)
    assert run_id

    await session.process_run(run_id)
    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.CONTEXT_READY
    assert run.error_message == "RALPH PRD bootstrap completed"

    steps = await db.list_run_steps(run_id)
    assert any(step["step_id"] == "ralph-prd-bootstrap" for step in steps)
    await db.close()


@pytest.mark.asyncio
async def test_ralph_preparatory_plan_bypasses_plan_review_for_quality_target(tmp_path: Path):
    session, db, _, _, _ = await create_session(tmp_path)
    use_case = session._process_run_uc
    plan = PlannerPlan(
        version="1.0",
        summary="Stub Ralph preparatory plan (training is deferred)",
        steps=[
            {
                "id": "s1",
                "title": "prepare planning-only report",
                "action": "shell",
                "step_intent": "general",
                "command": "printf 'ok\\n'",
                "risk_level": "low",
            }
        ],
    )

    issue = use_case.execution_guard_service.plan_quality_execution_issue(
        task={
            "goal": "RALPH preparatory story",
            "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
            "payload_json": json.dumps({"payload": {"execution_mode": "ralph_story"}}),
        },
        workspace_path=tmp_path / "workspace" / "demo",
        plan=plan,
    )

    assert issue is None
    await db.close()


@pytest.mark.asyncio
async def test_quality_replan_block_reason_for_stub_plan_without_training(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="Stub plan",
        steps=[
            {
                "id": "s1",
                "title": "verify metrics via codex",
                "action": "codex",
                "step_intent": "verify_metrics",
                "instruction": "check metrics",
                "risk_level": "low",
            }
        ],
    )
    session, db, _, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan))
    event = task_event(goal="Train segmentation model and satisfy IoU target")
    event["payload"]["constraints"] = ["RALPH_REQUIRED_METRIC: IOU >= 98%"]
    run_id = await session.submit_task_event(event)
    assert run_id

    await db.set_plan(run_id, plan.model_dump(mode="json"))
    run = await db.get_run(run_id)
    assert run is not None

    reason = session._process_run_uc.execution_guard_service.quality_replan_block_reason(
        run=run,
        task={
            "goal": event["payload"]["goal"],
            "constraints_json": json.dumps(event["payload"]["constraints"]),
            "payload_json": json.dumps({"payload": {"execution_mode": "plan_execute"}}),
        },
        workspace_path=tmp_path / "workspace" / "demo",
    )

    assert reason is not None
    assert "without an explicit shell run_training step" in reason
    await db.close()


@pytest.mark.asyncio
async def test_learning_notes_saved_and_reused_in_next_codex_prompt(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_cli_cmd="cat",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)

    runner = CodexRunner(settings)
    run_path = settings.runs_root / "run-notes"
    run_path.mkdir(parents=True, exist_ok=True)

    first = PlannerStep(
        id="c1",
        title="first codex step",
        action="codex",
        instruction="Generate a short checklist for project status.",
        risk_level="low",
    )
    first_result = await runner.execute_step(
        run_id="run-notes",
        step=first,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert first_result.status == "completed"

    notes_path = workspace / "knowledge" / "codex_learning_notes.md"
    assert notes_path.exists()
    assert "step=c1" in notes_path.read_text(encoding="utf-8")

    second = PlannerStep(
        id="c2",
        title="second codex step",
        action="codex",
        instruction="Revisit previous checklist and adjust for ML training.",
        risk_level="low",
    )
    second_result = await runner.execute_step(
        run_id="run-notes",
        step=second,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert second_result.status == "completed"

    second_prompt = (run_path / "c2.prompt.txt").read_text(encoding="utf-8")
    assert "Persistent execution notes (learned from previous codex runs):" in second_prompt
    assert "step=c1" in second_prompt
    assert "action=codex" in second_prompt


@pytest.mark.asyncio
async def test_workspace_snapshot_is_injected_into_codex_prompt(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_cli_cmd="cat",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    run_path = settings.runs_root / "run-snapshot"
    run_path.mkdir(parents=True, exist_ok=True)
    (workspace / ".agent").mkdir(parents=True, exist_ok=True)
    (workspace / ".agent" / "workspace_snapshot.md").write_text(
        "Workspace root: workspace\nRecently changed files:\n- scripts/train.py\n",
        encoding="utf-8",
    )

    runner = CodexRunner(settings)
    step = PlannerStep(
        id="snapshot-step",
        title="use snapshot",
        action="codex",
        instruction="Update training script using current workspace paths.",
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-snapshot",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "completed"
    prompt_text = (run_path / "snapshot-step.prompt.txt").read_text(encoding="utf-8")
    assert "Current workspace snapshot (authoritative paths):" in prompt_text
    assert "scripts/train.py" in prompt_text


@pytest.mark.asyncio
async def test_selected_skill_context_is_injected_into_codex_prompt(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        runs_root=tmp_path / "workspace" / "runs",
        codex_cli_cmd="cat",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    run_path = settings.runs_root / "run-skill-context"
    run_path.mkdir(parents=True, exist_ok=True)
    (workspace / "knowledge" / "skills" / "pytorch-lightning").mkdir(parents=True, exist_ok=True)
    (workspace / "knowledge" / "skills" / "pytorch-lightning" / "SKILL.md").write_text(
        "# Lightning\nPrefer deterministic trainer loops.\n",
        encoding="utf-8",
    )

    runner = CodexRunner(settings)
    step = PlannerStep(
        id="skill-step",
        title="use skill context",
        action="codex",
        instruction="Refactor the trainer with stable experiment loops.",
        skill_paths=["knowledge/skills/pytorch-lightning/SKILL.md"],
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-skill-context",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "completed"
    prompt_text = (run_path / "skill-step.prompt.txt").read_text(encoding="utf-8")
    assert "Selected skill context (apply these instructions before coding):" in prompt_text
    assert "Lightning" in prompt_text
    assert "skill-context:" in (result.command or "")


@pytest.mark.asyncio
async def test_database_records_and_lists_experiment_attempts(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()
    await db.record_experiment_attempt(
        workspace_id="demo",
        goal_signature="sig-1",
        run_id="run-1",
        task_id="task-1",
        run_attempt=1,
        verification_status="passed",
        quality_status="failed",
        quality_reason="accuracy below target",
        metrics={"accuracy": 0.87},
        hyperparameters={"epochs": 5, "learning_rate": 0.001},
        strategy={"chosen_intervention_id": "targeted_finetune"},
        skill_paths=["skills/pytorch-lightning/SKILL.md"],
    )

    attempts = await db.list_experiment_attempts(
        workspace_id="demo",
        goal_signature="sig-1",
        limit=5,
    )
    assert len(attempts) == 1
    assert attempts[0]["metrics"]["accuracy"] == pytest.approx(0.87)
    assert attempts[0]["hyperparameters"]["epochs"] == 5
    assert attempts[0]["skill_paths"] == ["skills/pytorch-lightning/SKILL.md"]


@pytest.mark.asyncio
async def test_database_run_retention_deletes_old_history_rows_and_logs_stats(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()

    old_ts = "2026-01-01T00:00:00+00:00"
    new_ts = "2026-03-14T00:00:00+00:00"

    await db.conn.execute(
        """
        INSERT INTO run_events(event_id, stream, event_type, run_id, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("evt-old", "tasks", "task.submitted", "run-old", "{}", old_ts),
    )
    await db.conn.execute(
        """
        INSERT INTO run_events(event_id, stream, event_type, run_id, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("evt-new", "tasks", "task.submitted", "run-new", "{}", new_ts),
    )
    await db.conn.execute(
        """
        INSERT INTO run_steps(run_id, step_id, step_title, step_index, action, command, status, stdout_text, stderr_text, duration_ms, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("run-old", "s1", "old step", 0, "shell", "echo old", "completed", "", "", 1, old_ts),
    )
    await db.conn.execute(
        """
        INSERT INTO policy_decisions(run_id, layer, subject, decision, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("run-old", "policy", "subject", "ALLOW", "ok", old_ts),
    )
    await db.conn.commit()

    stats = await db.run_retention(days=30)
    assert stats["deleted_run_events"] >= 1
    assert stats["deleted_run_steps"] >= 1
    assert stats["deleted_policy_decisions"] >= 1

    remaining_events = await db._fetchall("SELECT event_id FROM run_events ORDER BY event_id")
    assert [row["event_id"] for row in remaining_events] == ["evt-new"]

    retention_rows = await db._fetchall(
        """
        SELECT deleted_run_events, deleted_run_steps, deleted_policy_decisions
        FROM retention_stats
        ORDER BY id DESC
        LIMIT 1
        """
    )
    assert retention_rows
    assert int(retention_rows[0]["deleted_run_events"]) >= 1


@pytest.mark.asyncio
async def test_database_pdf_repository_roundtrip(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()

    doc_id, created = await db.upsert_pdf_document(
        path="knowledge/pdfs/doc1.pdf",
        content_hash="hash-1",
        mtime=123.0,
        page_count=2,
    )
    assert created is True
    assert doc_id > 0

    chunk_ids = await db.replace_pdf_chunks(
        doc_id,
        "knowledge/pdfs/doc1.pdf",
        [
            (0, 1, "alpha beta gamma"),
            (1, 2, "delta epsilon"),
        ],
    )
    assert len(chunk_ids) == 2

    await db.set_chunk_embeddings(
        [
            (chunk_ids[0], [0.1, 0.2]),
            (chunk_ids[1], [0.3, 0.4]),
        ]
    )

    known_paths = await db.list_known_pdf_paths()
    assert known_paths == ["knowledge/pdfs/doc1.pdf"]

    hashes = await db.get_pdf_path_hashes()
    assert hashes == {"knowledge/pdfs/doc1.pdf": "hash-1"}

    fts_rows = await db.fts_search("alpha", top_k=5)
    assert len(fts_rows) == 1
    assert fts_rows[0].document_path == "knowledge/pdfs/doc1.pdf"
    assert "alpha beta gamma" in fts_rows[0].text

    vector_rows = await db.vector_candidates()
    assert len(vector_rows) == 2
    assert all(row.embedding is not None for row in vector_rows)

    await db.remove_pdf_document("knowledge/pdfs/doc1.pdf")
    assert await db.list_known_pdf_paths() == []


@pytest.mark.asyncio
async def test_shell_output_path_marker_is_added_to_files_changed(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    run_path = settings.runs_root / "run-output-paths"
    run_path.mkdir(parents=True, exist_ok=True)
    runner = CodexRunner(settings)

    step = PlannerStep(
        id="shell-path-step",
        title="emit metrics path",
        action="shell",
        command="mkdir -p reports && printf 'ok\\n' > reports/metrics.md && printf 'metrics_path=reports/metrics.md\\n'",
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-output-paths",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "completed"
    assert "reports/metrics.md" in result.files_changed


@pytest.mark.asyncio
async def test_codex_auto_materializes_single_expected_file_from_stdout(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
        codex_cli_cmd="cat",
    )
    workspace = settings.workspace_root
    workspace.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    run_path = settings.runs_root / "run-auto-repair"
    run_path.mkdir(parents=True, exist_ok=True)
    runner = CodexRunner(settings)

    step = PlannerStep(
        id="auto-repair",
        title="create file via codex",
        action="codex",
        instruction=(
            "Create scripts/generated_check.py with the following content:\n"
            "```python\nprint('ok')\n```\n"
        ),
        expected_artifacts=[
            {"path": "scripts/generated_check.py", "kind": "file", "must_exist": True},
        ],
        risk_level="low",
    )
    result = await runner.execute_step(
        run_id="run-auto-repair",
        step=step,
        workspace_path=workspace,
        run_path=run_path,
    )
    assert result.status == "completed"
    assert result.auto_repaired is True
    assert "AUTO_REPAIRED" in result.summary
    assert (workspace / "scripts" / "generated_check.py").exists()


def task_event(workspace_id: str = "ws-1", goal: str = "Implement change", priority: str = "normal") -> dict:
    return {
        "event_id": str(uuid4()),
        "event_type": "task.submitted",
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": str(uuid4()),
        "workspace_id": workspace_id,
        "priority": priority,
        "payload": {
            "goal": goal,
            "constraints": ["keep tests green"],
            "pdf_scope": [],
            "execution_mode": "plan_execute",
        },
    }

@pytest.mark.asyncio
async def test_task_event_deduplication(tmp_path: Path):
    session, db, _, _, _ = await create_session(tmp_path)
    event = task_event()
    run_id_1 = await session.submit_task_event(event)
    run_id_2 = await session.submit_task_event(event)
    assert run_id_1 is not None
    assert run_id_2 is None
    runnable = await db.list_runnable_runs()
    assert len(runnable) == 1
    await db.close()


@pytest.mark.asyncio
async def test_workspace_lock_allows_single_active_run(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="sleep",
        steps=[
            {
                "id": "s1",
                "title": "sleep",
                "action": "shell",
                "instruction": "sleep",
                "command": "sleep 1",
                "risk_level": "low",
            }
        ],
    )
    session, db, _, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan), timeout_sec=3)
    event1 = task_event(workspace_id="shared")
    event2 = task_event(workspace_id="shared")
    run1 = await session.submit_task_event(event1)
    run2 = await session.submit_task_event(event2)
    assert run1 and run2

    first = asyncio.create_task(session.process_run(run1))
    await asyncio.sleep(0.2)
    await session.process_run(run2)
    run2_state = await db.get_run(run2)
    assert run2_state is not None
    assert run2_state.status == RunStatus.RECEIVED
    await first

    await session.process_run(run2)
    run2_done = await db.get_run(run2)
    assert run2_done is not None
    assert run2_done.status == RunStatus.COMPLETED
    await db.close()
@pytest.mark.asyncio
async def test_ralph_story_requires_quality_target_before_planning(tmp_path: Path):
    session, db, _, _, _ = await create_session(tmp_path)
    event = task_event(goal="Train segmentation model", workspace_id="ws-ralph")
    event["payload"]["execution_mode"] = "ralph_story"
    run_id = await session.submit_task_event(event)
    assert run_id
    await session.process_run(run_id)

    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.WAITING_PLAN_REVIEW
    assert "requires an explicit quality target" in (run.error_message or "")
    await db.close()


@pytest.mark.asyncio
async def test_risky_step_requires_approval_then_executes(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="approval flow",
        steps=[
            {
                "id": "r1",
                "title": "risky read",
                "action": "read",
                "instruction": "summarize sensitive data",
                "command": None,
                "risk_level": "high",
            }
        ],
    )
    session, db, _, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan))
    run_id = await session.submit_task_event(task_event())
    assert run_id
    await session.process_run(run_id)
    state = await db.get_run(run_id)
    assert state is not None
    assert state.status == RunStatus.WAITING_APPROVAL

    approved = await session.handle_control_event(
        {
            "event_id": str(uuid4()),
            "event_type": "run.approve",
            "schema_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"run_id": run_id},
        }
    )
    assert approved
    await session.process_run(run_id)
    done = await db.get_run(run_id)
    assert done is not None
    assert done.status == RunStatus.COMPLETED
    await db.close()


@pytest.mark.asyncio
async def test_timeout_retries_then_fails(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="timeouts",
        steps=[
            {
                "id": "t1",
                "title": "slow command",
                "action": "shell",
                "instruction": "sleep",
                "command": "sleep 1.2",
                "risk_level": "low",
            }
        ],
    )
    session, db, _, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan), timeout_sec=1)
    run_id = await session.submit_task_event(task_event())
    assert run_id
    await session.process_run(run_id)
    r1 = await db.get_run(run_id)
    assert r1 is not None and r1.status == RunStatus.EXECUTING
    await session.process_run(run_id)
    r2 = await db.get_run(run_id)
    assert r2 is not None and r2.status == RunStatus.EXECUTING
    await session.process_run(run_id)
    r3 = await db.get_run(run_id)
    assert r3 is not None and r3.status == RunStatus.FAILED
    assert "infra failure after retries" in (r3.error_message or "")
    await db.close()


@pytest.mark.asyncio
async def test_artifact_bundle_and_result_event_created(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="success path",
        steps=[
            {
                "id": "a1",
                "title": "noop",
                "action": "read",
                "instruction": "just proceed",
                "command": None,
                "risk_level": "low",
            }
        ],
    )
    session, db, bus, _, settings = await create_session(tmp_path, planner=StaticPlanner(plan))
    run_id = await session.submit_task_event(task_event())
    assert run_id
    await session.process_run(run_id)
    run = await db.get_run(run_id)
    assert run is not None and run.status == RunStatus.COMPLETED
    run_dir = settings.runs_root / run_id
    assert (run_dir / "patches.tar.gz").exists()
    assert (run_dir / "result.json").exists()
    assert bus.results
    assert bus.results[0]["status"] == "completed"
    await db.close()


@pytest.mark.asyncio
async def test_cancel_stops_run_and_marks_cancelled(tmp_path: Path):
    plan = PlannerPlan(
        version="1.0",
        summary="cancel flow",
        steps=[
            {
                "id": "c1",
                "title": "sleep long",
                "action": "shell",
                "instruction": "sleep",
                "command": "sleep 5",
                "risk_level": "low",
            }
        ],
    )
    session, db, bus, _, _ = await create_session(tmp_path, planner=StaticPlanner(plan), timeout_sec=10)
    run_id = await session.submit_task_event(task_event())
    assert run_id
    worker = asyncio.create_task(session.process_run(run_id))
    await asyncio.sleep(0.3)
    cancelled = await session.handle_control_event(
        {
            "event_id": str(uuid4()),
            "event_type": "run.cancel",
            "schema_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"run_id": run_id, "reason": "user request"},
        }
    )
    assert cancelled
    await worker
    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.CANCELLED
    assert any(item["status"] == "cancelled" for item in bus.results)
    await db.close()


@pytest.mark.asyncio
async def test_planner_failure_marks_run_failed(tmp_path: Path):
    session, db, _, _, _ = await create_session(tmp_path, planner=FailingPlanner())
    run_id = await session.submit_task_event(task_event())
    assert run_id
    await session.process_run(run_id)
    run = await db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.FAILED
    assert "planner error" in (run.error_message or "")
    await db.close()

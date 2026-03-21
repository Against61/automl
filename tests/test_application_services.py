from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from orchestrator.application.services.plan_contract_service import PlanContractService
from orchestrator.application.services.baseline_research_service import BaselineResearchService
from orchestrator.application.services.improvement_strategy_service import ImprovementStrategyService
from orchestrator.application.services.micro_training_policy_service import MicroTrainingPolicyService
from orchestrator.application.services.metric_interpretation_service import MetricInterpretation
from orchestrator.application.services.prompt_content_service import PromptContentService
from orchestrator.application.services.quality_gate_service import QualityGateService
from orchestrator.application.services.recovery_service import MissingFileRecoveryService
from orchestrator.application.services.task_intent_service import TaskIntentService
from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.application.use_cases.run_tick.hyperparameters import HyperparameterService
from orchestrator.application.use_cases.run_tick.planning_context import PlanningContextService
from orchestrator.application.use_cases.process_run_tick import ProcessRunTickUseCase
from orchestrator.persistence.db import Database, normalize_verification_payload
from orchestrator.execution.verifier import VerificationResult
from orchestrator.persistence.schemas import (
    ArtifactKind,
    PlannerStep,
    Priority,
    StepOperation,
    TaskPayload,
    TaskSubmittedEvent,
)
from orchestrator.planning.ralph import RalphBacklogService


def test_plan_contract_service_passes_when_artifact_exists(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True)
    step = PlannerStep(
        id="s1",
        title="write file",
        action="codex",
        instruction="create file",
        expected_artifacts=["result.md"],
        stop_condition="step completed",
        risk_level="low",
    )
    (tmp_path / "result.md").write_text("ok", encoding="utf-8")

    class Result:
        files_changed = ["result.md"]
        stdout_text = "ok"

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is True
    assert "passed" in reason


def test_plan_contract_service_fails_missing_artifact_for_codex_without_git(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True)
    step = PlannerStep(
        id="s-missing",
        title="train model",
        action="codex",
        instruction="train and save metrics",
        expected_artifacts=["metrics.json"],
        stop_condition="training completes and metrics file is saved",
        risk_level="medium",
    )

    class Result:
        files_changed: list[str] = []
        stdout_text = "summary: command completed"

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is False
    assert (
        "missing expected artifacts" in reason
        or "verify_metrics intent missing metric keys" in reason
    )


def test_plan_contract_service_supports_hidden_artifact_path(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True)
    step = PlannerStep(
        id="s-hidden",
        title="bootstrap",
        action="codex",
        instruction="create ralph snapshot",
        expected_artifacts=[".ralph_workspace_tree.md exists"],
        stop_condition="snapshot file exists",
        risk_level="low",
    )
    (tmp_path / ".ralph_workspace_tree.md").write_text("demo/", encoding="utf-8")

    class Result:
        files_changed: list[str] = []
        stdout_text = "summary: command completed"

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is True
    assert "passed" in reason


def test_plan_contract_service_free_mode_bypasses_contract_failures(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="free")
    step = PlannerStep(
        id="s-free",
        title="strict check",
        action="codex",
        instruction="create metrics",
        expected_artifacts=["metrics.md"],
        stop_condition="metrics file exists",
        risk_level="low",
    )

    class Result:
        files_changed: list[str] = []
        stdout_text = "summary: command completed"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is True
    assert "free" in reason


def test_plan_contract_service_strict_verify_metrics_requires_metric_keys(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="strict")
    step = PlannerStep(
        id="s-metrics",
        title="verify metrics",
        action="codex",
        step_intent="verify_metrics",
        instruction="verify target metrics",
        expected_artifacts=[
            {
                "path": "metrics.md",
                "kind": "metrics",
                "must_exist": False,
                "metric_keys": ["test_accuracy"],
            }
        ],
        stop_condition="metric key test_accuracy is present",
        risk_level="low",
    )

    class Result:
        files_changed: list[str] = []
        stdout_text = "summary: command completed\nloss: 0.9"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is False
    assert "missing metric keys" in reason


def test_plan_contract_service_verify_metrics_accepts_val_acc_alias(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="strict")
    step = PlannerStep(
        id="s-metrics-alias",
        title="verify validation metrics",
        action="verify",
        step_intent="verify_metrics",
        instruction="verify val acc",
        expected_artifacts=[
            {
                "path": "metrics.md",
                "kind": "metrics",
                "must_exist": True,
                "metric_keys": ["val_acc"],
            }
        ],
        stop_condition="validation accuracy is present in metrics.md",
        risk_level="low",
    )
    (tmp_path / "metrics.md").write_text(
        "# Metrics\n\n- Best validation accuracy: 92.46%\n",
        encoding="utf-8",
    )

    class Result:
        files_changed: list[str] = []
        stdout_text = "summary: verify completed"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is True
    assert "passed" in reason


def test_plan_contract_service_balanced_relaxes_missing_metrics_path_when_metrics_exist(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-metrics-path-drift",
        title="train and save metrics",
        action="shell",
        step_intent="run_training",
        instruction="run training",
        expected_artifacts=[
            {
                "path": "outputs/metrics_smoke_test.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics file exists",
        risk_level="low",
    )
    (tmp_path / "metrics.md").write_text("accuracy: 0.92\n", encoding="utf-8")

    class Result:
        files_changed: list[str] = []
        stdout_text = "Epoch 1/1 loss=0.4 accuracy=0.92"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is True
    assert "passed" in reason


def test_plan_contract_service_verify_metrics_accepts_fresh_named_metrics_fallback(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-metrics-fallback",
        title="verify smoke metrics",
        action="verify",
        step_intent="verify_metrics",
        instruction="verify accuracy metric",
        expected_artifacts=[
            {
                "path": "metrics.json",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics file contains accuracy",
        risk_level="low",
    )
    fallback_path = tmp_path / "fashionmnist_smoke_test_metrics.md"
    fallback_path.write_text("Accuracy: 0.92\n", encoding="utf-8")

    class Result:
        files_changed = ["fashionmnist_smoke_test_metrics.md"]
        stdout_text = "summary: verify completed"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is True
    assert "passed" in reason


def test_plan_contract_service_balanced_does_not_relax_on_stale_metrics_file(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-metrics-path-stale",
        title="train and save metrics",
        action="shell",
        step_intent="run_training",
        instruction="run training",
        expected_artifacts=[
            {
                "path": "outputs/metrics_smoke_test.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics file exists",
        risk_level="low",
    )
    metrics_path = tmp_path / "metrics.md"
    metrics_path.write_text("accuracy: 0.92\n", encoding="utf-8")
    stale_ts = metrics_path.stat().st_mtime - 600
    os.utime(metrics_path, (stale_ts, stale_ts))

    class Result:
        files_changed: list[str] = []
        stdout_text = "Epoch 1/1 loss=0.4 accuracy=0.92"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is False
    assert "missing expected artifacts" in reason


def test_plan_contract_service_strict_keeps_missing_metrics_path_failure(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="strict")
    step = PlannerStep(
        id="s-metrics-path-strict",
        title="verify metrics strict",
        action="verify",
        step_intent="verify_metrics",
        instruction="verify metrics",
        expected_artifacts=[
            {
                "path": "metrics/metrics.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="accuracy is present",
        risk_level="low",
    )
    (tmp_path / "metrics.md").write_text("accuracy: 0.92\n", encoding="utf-8")

    class Result:
        files_changed: list[str] = []
        stdout_text = "verify done"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is False
    assert (
        "missing expected artifacts" in reason
        or "verify_metrics intent missing metric keys" in reason
    )


def test_planner_step_artifact_parser_ignores_structural_meta_tokens():
    step = PlannerStep(
        id="s-meta-artifact",
        title="bootstrap prd",
        action="codex",
        instruction="prepare prd",
        expected_artifacts=["meta.workspace_snapshot in prd.json"],
        risk_level="low",
    )
    assert len(step.expected_artifacts) == 1
    spec = step.expected_artifacts[0]
    assert spec.path == "prd.json"
    assert spec.kind.value == "file"
    assert spec.must_exist is True


def test_planner_step_artifact_parser_keeps_real_file_paths():
    step = PlannerStep(
        id="s-file-artifact",
        title="write metrics",
        action="codex",
        instruction="save metrics",
        expected_artifacts=["metrics.md"],
        risk_level="low",
    )
    assert len(step.expected_artifacts) == 1
    spec = step.expected_artifacts[0]
    assert spec.path == "metrics.md"
    assert spec.must_exist is True


def test_planner_step_defaults_run_training_metrics_artifact_to_metrics_json():
    step = PlannerStep(
        id="s-train-default-metrics",
        title="run smoke training",
        action="shell",
        step_intent="run_training",
        command="python train.py --smoke-test",
        expected_artifacts=[],
        stop_condition="training metrics are written",
        risk_level="low",
    )
    metric_specs = [artifact for artifact in step.expected_artifacts if artifact.kind == ArtifactKind.metrics]
    assert len(metric_specs) == 1
    assert metric_specs[0].path == "metrics.json"
    assert metric_specs[0].must_exist is True
    assert metric_specs[0].must_be_nonempty is True


def test_planner_step_defaults_verify_metrics_artifact_to_metrics_json():
    step = PlannerStep(
        id="s-verify-default-metrics",
        title="verify evaluation metrics",
        action="verify",
        step_intent="verify_metrics",
        expected_artifacts=[],
        stop_condition="evaluation metrics are present",
        risk_level="low",
    )
    metric_specs = [artifact for artifact in step.expected_artifacts if artifact.kind == ArtifactKind.metrics]
    assert len(metric_specs) == 1
    assert metric_specs[0].path == "metrics.json"
    assert metric_specs[0].must_exist is True
    assert metric_specs[0].must_be_nonempty is True


def test_recovery_service_detects_and_rewrites_python_target(tmp_path: Path):
    service = MissingFileRecoveryService()
    stderr = "python: can't open file '/app/workspace/demo/scripts/run.py': [Errno 2] No such file or directory"
    decision = service.detect_missing_python_file(stderr)
    assert decision.detected is True
    assert decision.missing_path is not None

    script_path = tmp_path / "scripts" / "run.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('ok')", encoding="utf-8")
    candidates = service.find_python_file_candidates(decision.missing_path, tmp_path)
    assert any(path.name == "run.py" for path in candidates)

    step = PlannerStep(
        id="s2",
        title="run script",
        action="shell",
        instruction="run",
        command="python scripts/run_missing.py",
        commands=["python scripts/run_missing.py"],
        risk_level="low",
    )
    repaired = service.replace_missing_file_in_step(
        step=step,
        expected_missing="scripts/run_missing.py",
        replacement="scripts/run.py",
    )
    assert repaired is not None
    assert "run.py" in (repaired.command or "")


def test_extract_hyperparameters_from_command():
    command = (
        "python train.py --epochs 8 --lr=0.001 --batch-size 64 "
        "--optimizer adam --weight-decay 0.0005 --seed 42"
    )
    parsed = HyperparameterService.extract_from_command(command)
    assert parsed["epochs"] == 8
    assert parsed["learning_rate"] == 0.001
    assert parsed["batch_size"] == 64
    assert parsed["optimizer"] == "adam"
    assert parsed["weight_decay"] == 0.0005
    assert parsed["seed"] == 42


def test_extract_hyperparameters_supports_env_style_assignments():
    command = "EPOCHS=12 LR=3e-4 BATCH_SIZE=128 python train.py --workers 4"
    parsed = HyperparameterService.extract_from_command(command)
    assert parsed["epochs"] == 12
    assert parsed["learning_rate"] == 3e-4
    assert parsed["batch_size"] == 128
    assert parsed["workers"] == 4


def test_extract_hyperparameters_ignores_codex_cli_command():
    command = "codex exec --model gpt-5.3-codex --skip-git-repo-check"
    parsed = HyperparameterService.extract_from_command(command)
    assert parsed == {}


def test_planner_step_derives_stepio_contract_defaults():
    step = PlannerStep(
        id="train-step",
        title="Run training epoch",
        action="shell",
        step_type="change",
        command="python train.py --epochs 5 --lr 0.001",
        expected_artifacts=[
            {"path": "metrics.md", "kind": "metrics", "must_exist": True, "metric_keys": ["accuracy"]}
        ],
        stop_condition="metrics are produced",
        risk_level="medium",
    )
    assert step.operation == StepOperation.run_training
    assert "python train.py --epochs 5 --lr 0.001" in step.inputs.commands
    assert step.expected_outputs.artifacts
    assert step.expected_outputs.metrics_required == ["accuracy"]
    assert step.expected_outputs.stop_condition == "metrics are produced"
    assert step.policy.risk == "medium"


async def _quality_eval(goal: str, metrics: dict[str, float]) -> tuple[bool, str]:
    backlog = RalphBacklogService()
    service = QualityGateService(ralph_backlog=backlog)
    task = {
        "goal": goal,
        "constraints_json": json.dumps([]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics=metrics,
    )
    return await service.evaluate(task=task, workspace_path=Path("."), verification=verification)


def test_quality_gate_service_uses_goal_requirement():
    import asyncio

    ok, reason_ok = asyncio.run(_quality_eval("accuracy >= 90%", {"test_accuracy": 0.91}))
    assert ok is True
    assert "passed" in reason_ok

    fail, reason_fail = asyncio.run(_quality_eval("accuracy >= 95%", {"test_accuracy": 0.91}))
    assert fail is False
    assert "failed" in reason_fail


def test_quality_gate_service_percent_requirement_normalizes_ratio_and_percent_metric_values():
    import asyncio

    ratio_ok, _ = asyncio.run(_quality_eval("dice >= 90%", {"dice": 0.91}))
    percent_ok, _ = asyncio.run(_quality_eval("dice >= 90%", {"dice": 91.0}))
    percent_fail, reason_fail = asyncio.run(_quality_eval("dice >= 97%", {"dice": 77.18860167952153}))

    assert ratio_ok is True
    assert percent_ok is True
    assert percent_fail is False
    assert "failed" in reason_fail


def test_quality_gate_service_prefers_validation_or_test_metric_over_train():
    import asyncio

    ok, reason = asyncio.run(
        _quality_eval(
            "accuracy >= 90%",
            {
                "train_accuracy": 0.99,
                "test_accuracy": 0.91,
                "val_accuracy": 0.88,
            },
        )
    )
    assert ok is False
    assert "failed" in reason


def test_quality_gate_service_fails_when_split_leakage_detected():
    import asyncio

    ok, reason = asyncio.run(
        _quality_eval(
            "accuracy >= 90%",
            {
                "accuracy": 0.99,
                "split_leakage_detected": True,
            },
        )
    )
    assert ok is False
    assert "reuses training examples" in reason


def test_quality_gate_service_blocks_reference_fixture_report():
    import asyncio

    ok, reason = asyncio.run(
        _quality_eval(
            "accuracy >= 90%",
            {
                "accuracy": 1.0,
                "reference_evaluation_fixture_detected": True,
                "non_production_report_detected": True,
            },
        )
    )
    assert ok is False
    assert "reference_evaluation_fixture" in reason


def test_quality_gate_service_blocks_planning_only_report():
    import asyncio

    ok, reason = asyncio.run(
        _quality_eval(
            "accuracy >= 90%",
            {
                "accuracy": 1.0,
                "planning_only_report_detected": True,
                "non_production_report_detected": True,
            },
        )
    )
    assert ok is False
    assert "planning-only report" in reason


def test_plan_contract_service_fails_on_overlap_marker_in_metrics_artifact(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-overlap",
        title="run smoke training",
        action="shell",
        step_intent="run_training",
        instruction="run smoke training and save metrics",
        expected_artifacts=[
            {
                "path": "smoke_test_metrics.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics are saved",
        risk_level="medium",
    )
    (tmp_path / "smoke_test_metrics.md").write_text(
        "\n".join(
            [
                "# Metrics",
                "- Evaluation split: `same balanced smoke subset used for the short training run`",
                "| key | value |",
                "| --- | --- |",
                "| accuracy | 1.000000 |",
            ]
        ),
        encoding="utf-8",
    )

    class Result:
        files_changed = ["smoke_test_metrics.md"]
        stdout_text = "accuracy=1.0"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is False
    assert "evaluation/train overlap" in reason


def test_plan_contract_service_fails_on_train_subset_overfit_marker(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-overlap-train-subset",
        title="run smoke training",
        action="shell",
        step_intent="run_training",
        instruction="run smoke training and save metrics",
        expected_artifacts=[
            {
                "path": "smoke_test_metrics.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics are saved",
        risk_level="medium",
    )
    (tmp_path / "smoke_test_metrics.md").write_text(
        "\n".join(
            [
                "# Metrics",
                "| Metric | Value |",
                "| --- | --- |",
                "| Evaluation split | Train subset (smoke test overfit check) |",
                "| Accuracy | 1.000000 |",
                "The accuracy gate applies to the training subset used for the smoke test.",
            ]
        ),
        encoding="utf-8",
    )

    class Result:
        files_changed = ["smoke_test_metrics.md"]
        stdout_text = "accuracy=1.0"
        stderr_text = ""

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=Result())
    assert passed is False
    assert "evaluation/train overlap" in reason


def test_plan_contract_service_fails_on_stale_metrics_artifact_for_current_expected_path(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-stale-current-metrics",
        title="run training and overwrite metrics",
        action="shell",
        step_intent="run_training",
        instruction="run training and update metrics.md",
        expected_artifacts=[
            {
                "path": "metrics.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics are updated",
        risk_level="medium",
    )
    metrics_path = tmp_path / "metrics.md"
    metrics_path.write_text("accuracy: 0.91\n", encoding="utf-8")

    stdout_path = tmp_path / "train.stdout.log"
    stdout_path.write_text("Epoch 1/10 started\n", encoding="utf-8")
    stderr_path = tmp_path / "train.stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    older_ts = stdout_path.stat().st_mtime - 30
    os.utime(metrics_path, (older_ts, older_ts))

    result = type(
        "Result",
        (),
        {
            "files_changed": [],
            "stdout_text": "training completed",
            "stderr_text": "",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )()

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=result)
    assert passed is False
    assert "stale metrics artifact reused without refresh: metrics.md" in reason


def test_plan_contract_service_accepts_refreshed_metrics_artifact_for_current_expected_path(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="s-fresh-current-metrics",
        title="run training and overwrite metrics",
        action="shell",
        step_intent="run_training",
        instruction="run training and update metrics.md",
        expected_artifacts=[
            {
                "path": "metrics.md",
                "kind": "metrics",
                "must_exist": True,
                "must_be_nonempty": True,
                "metric_keys": ["accuracy"],
            }
        ],
        stop_condition="metrics are updated",
        risk_level="medium",
    )
    stdout_path = tmp_path / "train.stdout.log"
    stdout_path.write_text("Epoch 1/10 started\n", encoding="utf-8")
    stderr_path = tmp_path / "train.stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    metrics_path = tmp_path / "metrics.md"
    metrics_path.write_text("accuracy: 0.94\n", encoding="utf-8")

    result = type(
        "Result",
        (),
        {
            "files_changed": ["metrics.md"],
            "stdout_text": "Epoch 10/10 accuracy=0.94",
            "stderr_text": "",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )()

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=result)
    assert passed is True
    assert "passed" in reason


def test_plan_contract_service_does_not_treat_prd_json_as_stale_metrics_artifact(tmp_path: Path):
    service = PlanContractService(plan_review_enabled=True, strictness="balanced")
    step = PlannerStep(
        id="ralph-prd-bootstrap",
        title="Create/update RALPH PRD",
        action="codex",
        step_intent="modify_file",
        instruction="prepare prd only",
        expected_artifacts=[
            {
                "path": "prd.json",
                "kind": "file",
                "must_exist": True,
                "must_be_nonempty": True,
            }
        ],
        stop_condition="PRD exists in workspace and contains at least one pending story",
        risk_level="medium",
    )
    prd_path = tmp_path / "prd.json"
    prd_path.write_text("{\"meta\": {\"project\": \"demo\"}}\n", encoding="utf-8")

    stdout_path = tmp_path / "ralph-prd-bootstrap.stdout.log"
    stdout_path.write_text("updated prd\n", encoding="utf-8")
    stderr_path = tmp_path / "ralph-prd-bootstrap.stderr.log"
    stderr_path.write_text("", encoding="utf-8")

    older_ts = stdout_path.stat().st_mtime - 30
    os.utime(prd_path, (older_ts, older_ts))

    result = type(
        "Result",
        (),
        {
            "files_changed": [],
            "stdout_text": "updated prd",
            "stderr_text": "",
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        },
    )()

    passed, reason = service.evaluate(step=step, workspace_path=tmp_path, result=result)
    assert passed is True
    assert "passed" in reason


@pytest.mark.asyncio
async def test_verification_set_keeps_history(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()
    task_id = str(uuid4())
    await db.upsert_task(
        TaskSubmittedEvent(
            event_id=uuid4(),
            event_type="task.submitted",
            schema_version="1.0",
            task_id=task_id,
            workspace_id="ws-1",
            priority=Priority.normal,
            payload=TaskPayload(
                goal="verify history test",
                constraints=[],
                pdf_scope=[],
            ),
        )
    )
    run_id = await db.create_or_get_run(task_id=task_id, workspace_id="ws-1", priority=Priority.normal)
    await db.set_verification(
        run_id,
        {"status": "passed", "commands": [], "metrics": {"test_accuracy": 0.87}, "passed": True},
    )
    first = await db.get_run(run_id)
    assert first is not None
    assert first.verification_json is not None
    assert first.verification_json.get("attempt") == 1
    assert first.verification_json.get("latest_attempt") == 1
    assert first.verification_json.get("history") == []

    await db.set_verification(
        run_id,
        {"status": "failed", "commands": [], "metrics": {"test_accuracy": 0.84}, "passed": False},
    )
    second = await db.get_run(run_id)
    assert second is not None
    assert second.verification_json is not None
    assert second.verification_json.get("attempt") == 2
    history = second.verification_json.get("history")
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0].get("status") == "passed"
    assert second.verification_json.get("metrics", {}).get("test_accuracy") == 0.84


@pytest.mark.asyncio
async def test_set_plan_resets_next_step_index(tmp_path: Path):
    db = Database(tmp_path / "orchestrator.db")
    await db.connect()
    task_id = str(uuid4())
    await db.upsert_task(
        TaskSubmittedEvent(
            event_id=uuid4(),
            event_type="task.submitted",
            schema_version="1.0",
            task_id=task_id,
            workspace_id="ws-plan-reset",
            priority=Priority.normal,
            payload=TaskPayload(
                goal="plan reset index test",
                constraints=[],
                pdf_scope=[],
            ),
        )
    )
    run_id = await db.create_or_get_run(
        task_id=task_id,
        workspace_id="ws-plan-reset",
        priority=Priority.normal,
    )
    await db.set_next_step_index(run_id, 2)
    await db.set_plan(
        run_id,
        {
            "version": "1.0",
            "summary": "new plan",
            "steps": [
                {
                    "id": "s1",
                    "title": "first step",
                    "action": "read",
                    "instruction": "start from first step",
                    "risk_level": "low",
                }
            ],
        },
    )
    run = await db.get_run(run_id)
    assert run is not None
    assert run.next_step_index == 0


def test_improvement_strategy_service_builds_strategy_and_artifact(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = ImprovementStrategyService(quality_gate_service=quality)

    task = {
        "goal": "Improve classification quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: val_accuracy >= 95%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={"train_accuracy": 0.99, "val_accuracy": 0.86},
    )
    previous_verification = {
        "attempt": 2,
        "latest_hyperparameters": {"epochs": 5, "learning_rate": 0.001},
        "hyperparameter_attempts": [
            {
                "step_id": "train-1",
                "hyperparameters": {"epochs": 5, "learning_rate": 0.001},
            }
        ],
    }

    strategy = service.build_for_quality_failure(
        run_id="run-123",
        task=task,
        workspace_path=tmp_path,
        verification=verification,
        previous_verification=previous_verification,
        quality_reason="quality gate failed",
    )
    assert strategy["kind"] == "quality_improvement_strategy"
    assert strategy["diagnosis"]["pattern"] == "overfitting"
    assert strategy["chosen_intervention_id"] is not None
    artifact_rel = strategy["artifact_path"]
    artifact = tmp_path / artifact_rel
    assert artifact.exists()


def test_micro_training_policy_starts_with_one_epoch_baseline(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = MicroTrainingPolicyService(quality_gate_service=quality)

    task = {
        "goal": "Improve segmentation quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }

    policy = service.build_from_previous_verification(
        task=task,
        workspace_path=tmp_path,
        previous_verification=None,
    )

    assert policy["next_epochs"] == 1
    assert policy["phase"] == "baseline"
    assert policy["force_strategy_reset"] is False


def test_micro_training_policy_promotes_from_one_epoch_to_two(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = MicroTrainingPolicyService(quality_gate_service=quality)

    task = {
        "goal": "Improve segmentation quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }
    previous_verification = {
        "attempt": 1,
        "metrics": {"eval_mean_iou": 0.61},
        "latest_hyperparameters": {"epochs": 1},
        "quality_gate": {"status": "failed", "reason": "quality gate failed"},
    }

    policy = service.build_from_previous_verification(
        task=task,
        workspace_path=tmp_path,
        previous_verification=previous_verification,
    )

    assert policy["next_epochs"] == 2
    assert policy["phase"] == "growth_check"
    assert policy["force_strategy_reset"] is False


def test_micro_training_policy_promotes_to_five_epochs_when_metric_grows(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = MicroTrainingPolicyService(quality_gate_service=quality)

    task = {
        "goal": "Improve segmentation quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }
    previous_verification = {
        "attempt": 2,
        "metrics": {"eval_mean_iou": 0.68},
        "latest_hyperparameters": {"epochs": 2},
        "quality_gate": {"status": "failed", "reason": "quality gate failed"},
        "history": [
            {
                "attempt": 1,
                "metrics": {"eval_mean_iou": 0.61},
                "latest_hyperparameters": {"epochs": 1},
                "quality_gate": {"status": "failed", "reason": "quality gate failed"},
            }
        ],
    }

    policy = service.build_from_previous_verification(
        task=task,
        workspace_path=tmp_path,
        previous_verification=previous_verification,
    )

    assert policy["next_epochs"] == 5
    assert policy["phase"] == "expand"
    assert policy["metric_growth"] == pytest.approx(0.07, rel=1e-6)
    assert policy["force_strategy_reset"] is False


def test_micro_training_policy_resets_strategy_when_two_epoch_run_does_not_improve(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = MicroTrainingPolicyService(quality_gate_service=quality)

    task = {
        "goal": "Improve segmentation quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }
    previous_verification = {
        "attempt": 2,
        "metrics": {"eval_mean_iou": 0.60},
        "latest_hyperparameters": {"epochs": 2},
        "quality_gate": {"status": "failed", "reason": "quality gate failed"},
        "history": [
            {
                "attempt": 1,
                "metrics": {"eval_mean_iou": 0.61},
                "latest_hyperparameters": {"epochs": 1},
                "quality_gate": {"status": "failed", "reason": "quality gate failed"},
            }
        ],
    }

    policy = service.build_from_previous_verification(
        task=task,
        workspace_path=tmp_path,
        previous_verification=previous_verification,
    )

    assert policy["next_epochs"] == 1
    assert policy["phase"] == "reset_strategy"
    assert policy["force_strategy_reset"] is True


def test_micro_training_policy_resets_strategy_after_failed_five_epoch_run(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = MicroTrainingPolicyService(quality_gate_service=quality)

    task = {
        "goal": "Improve segmentation quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }
    previous_verification = {
        "attempt": 3,
        "metrics": {"eval_mean_iou": 0.72},
        "latest_hyperparameters": {"epochs": 5},
        "quality_gate": {"status": "failed", "reason": "quality gate failed"},
        "history": [],
    }

    policy = service.build_from_previous_verification(
        task=task,
        workspace_path=tmp_path,
        previous_verification=previous_verification,
    )

    assert policy["next_epochs"] == 1
    assert policy["phase"] == "reset_strategy"
    assert policy["force_strategy_reset"] is True


def test_quality_gate_service_selects_real_accuracy_not_sample_count_metric() -> None:
    quality = QualityGateService(ralph_backlog=RalphBacklogService())
    metrics = {
        "test_samples_available_but_not_used_for_reported_accuracy": 10000,
        "training_accuracy": 0.2773,
        "train_accuracy": 0.8854,
        "out_evaluation_accuracy": 0.8755,
        "eval_accuracy": 0.875,
    }

    value = quality.select_metric_value(metrics, "accuracy")
    assert value == pytest.approx(0.875, rel=1e-6)


@pytest.mark.asyncio
async def test_quality_gate_service_resolves_iou_from_eval_mean_iou(tmp_path: Path) -> None:
    quality = QualityGateService(ralph_backlog=RalphBacklogService())
    task = {
        "goal": "Train segmentation model and report IoU",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: IOU >= 98%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={
            "train_mean_iou": 0.99,
            "eval_mean_iou": 0.63,
            "loss": 0.42,
        },
    )

    passed, reason = await quality.evaluate(
        task=task,
        workspace_path=tmp_path,
        verification=verification,
    )

    assert passed is False
    assert "quality gate failed" in reason
    assert "0.63" in reason
    assert verification.details["metric_resolution"]["resolved_metric_key"] == "eval_mean_iou"
    assert verification.details["metric_resolution"]["mode"] == "preferred_alias"


class _DummyMetricInterpreter:
    async def resolve_metric(self, *, required_metric_key: str, metrics: dict[str, object], workspace_path: Path):
        assert required_metric_key == "iou"
        assert "loss" in metrics
        return MetricInterpretation(
            resolved_metric_key="eval_mean_iou",
            resolved_value=0.91,
            confidence="high",
            reason="mean IoU reported on eval split",
        )


@pytest.mark.asyncio
async def test_quality_gate_service_uses_metric_interpreter_when_metric_missing(tmp_path: Path) -> None:
    quality = QualityGateService(
        ralph_backlog=RalphBacklogService(),
        metric_interpreter=_DummyMetricInterpreter(),
    )
    task = {
        "goal": "Train segmentation model and report IoU",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: IOU >= 90%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={"loss": 0.12},
    )

    passed, reason = await quality.evaluate(
        task=task,
        workspace_path=tmp_path,
        verification=verification,
    )

    assert passed is True
    assert reason == "quality gate passed"
    assert verification.metrics["iou"] == pytest.approx(0.91, rel=1e-6)
    assert verification.metrics["iou_resolved_from"] == "eval_mean_iou"
    assert verification.details["metric_resolution"]["mode"] == "semantic_fallback"
    assert verification.details["metric_resolution"]["resolved_metric_key"] == "eval_mean_iou"


@pytest.mark.asyncio
async def test_quality_gate_service_prefers_eval_acc_alias_over_train_accuracy(tmp_path: Path) -> None:
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    task = {
        "goal": "Train a simple fashionMNIST baseline and report metrics",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 96%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={
            "train_accuracy": 0.980333,
            "eval_acc": 0.931,
            "accuracy_target": 0.96,
        },
    )

    passed, reason = await quality.evaluate(
        task=task,
        workspace_path=tmp_path,
        verification=verification,
    )

    assert passed is False
    assert "quality gate failed" in reason
    assert "0.931" in reason


@pytest.mark.asyncio
async def test_quality_gate_service_rejects_accuracy_as_primary_metric_for_segmentation_task(tmp_path: Path) -> None:
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    task = {
        "goal": "Train segmentation model on coco dataset",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 96%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={
            "eval_accuracy": 1.0,
            "test_accuracy": 1.0,
            "split_integrity_passed": True,
        },
    )

    passed, reason = await quality.evaluate(
        task=task,
        workspace_path=tmp_path,
        verification=verification,
    )

    assert passed is False
    assert reason == "metric 'iou' not found in verification output"


@pytest.mark.asyncio
async def test_quality_gate_service_rejects_metric_intent_drift_before_threshold_check(tmp_path: Path) -> None:
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    task = {
        "goal": "Train segmentation model on coco dataset",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 96%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={
            "metric_intent_drift_detected": True,
            "primary_metric_key": "accuracy",
            "eval_accuracy": 1.0,
        },
    )

    passed, reason = await quality.evaluate(
        task=task,
        workspace_path=tmp_path,
        verification=verification,
    )

    assert passed is False
    assert reason == "quality gate failed: metrics artifact intent does not match inferred task intent"


def test_task_intent_service_infers_detection_family_from_metric_and_text(tmp_path: Path) -> None:
    service = TaskIntentService()
    intent = service.infer(
        goal="Train detector for object detection on warehouse images",
        constraints=["RALPH_REQUIRED_METRIC: mAP50 >= 80%"],
        workspace_path=tmp_path,
    )

    assert intent.task_family == "detection"
    assert intent.primary_metric_key == "map50"
    assert intent.requires_real_dataset_smoke is True
    assert "map50" in intent.preferred_metrics


def test_improvement_strategy_service_uses_same_metric_resolution_as_quality_gate(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = ImprovementStrategyService(quality_gate_service=quality)

    task = {
        "goal": "Improve classification quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 97%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={
            "test_samples_available_but_not_used_for_reported_accuracy": 10000,
            "training_accuracy": 0.2773,
            "train_accuracy": 0.8854,
            "out_evaluation_accuracy": 0.8755,
            "eval_accuracy": 0.875,
        },
    )

    strategy = service.build_for_quality_failure(
        run_id="run-accuracy-resolution",
        task=task,
        workspace_path=tmp_path,
        verification=verification,
        previous_verification=None,
        quality_reason="quality gate failed: accuracy=0.875 >= 0.97 required (unit=%)",
    )

    assert strategy["objective"]["current_value"] == pytest.approx(0.875, rel=1e-6)
    assert strategy["objective"]["gap"] == pytest.approx(0.095, rel=1e-6)


def test_improvement_strategy_service_promotes_relevant_skills_and_experiment_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = ImprovementStrategyService(quality_gate_service=quality)

    skills_root = tmp_path / "knowledge" / "skills"
    (skills_root / "pytorch-lightning").mkdir(parents=True, exist_ok=True)
    (skills_root / "pytorch-lightning" / "SKILL.md").write_text("# Lightning\nUse Lightning loops.\n", encoding="utf-8")
    (skills_root / "seaborn").mkdir(parents=True, exist_ok=True)
    (skills_root / "seaborn" / "SKILL.md").write_text("# Seaborn\nUse Seaborn diagnostics.\n", encoding="utf-8")
    (skills_root / "sklearn").mkdir(parents=True, exist_ok=True)
    (skills_root / "sklearn" / "SKILL.md").write_text("# Sklearn\nUse sklearn reports.\n", encoding="utf-8")
    monkeypatch.delenv("CODEX_HOME", raising=False)

    task = {
        "goal": "Train FashionMNIST baseline with PyTorch",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 97%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={"train_accuracy": 0.93, "val_accuracy": 0.88},
    )
    experiment_history = [
        {
            "run_id": "old-run-1",
            "attempt": 1,
            "quality_status": "failed",
            "quality_reason": "accuracy below target",
            "metrics": {"accuracy": 0.84},
            "hyperparameters": {"epochs": 5, "learning_rate": 0.001},
        }
    ]

    strategy = service.build_for_quality_failure(
        run_id="run-skill-history",
        task=task,
        workspace_path=tmp_path,
        verification=verification,
        previous_verification=None,
        quality_reason="quality gate failed: accuracy=0.88 >= 0.97 required (unit=%)",
        experiment_history=experiment_history,
    )

    chosen = strategy["chosen_intervention"]
    assert chosen["skill_paths"]
    assert all("lightning" not in path.lower() for path in chosen["skill_paths"])
    assert any("seaborn" in path.lower() or "sklearn" in path.lower() for path in chosen["skill_paths"])
    assert strategy["history"]["experiment_attempts"][0]["run_id"] == "old-run-1"
    assert any("Apply these skills as execution context" in item for item in strategy["planner_directives"])


def test_workspace_snapshot_service_writes_snapshot_and_detects_output_paths(tmp_path: Path):
    service = WorkspaceSnapshotService()
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts" / "metrics.md").write_text("accuracy: 0.99\n", encoding="utf-8")

    step = PlannerStep(
        id="train-1",
        title="run smoke test",
        action="shell",
        instruction="run smoke test",
        expected_artifacts=[
            {"path": "outputs/metrics.md", "kind": "metrics", "must_exist": True},
        ],
        risk_level="low",
    )

    class Result:
        files_changed: list[str] = []
        stdout_text = "accuracy=99%\nmetrics_path=scripts/metrics.md\n"
        stderr_text = ""

    payload = service.refresh(tmp_path, step=step, result=Result())
    assert (tmp_path / ".agent" / "workspace_snapshot.json").exists()
    assert (tmp_path / ".agent" / "workspace_snapshot.md").exists()
    assert (tmp_path / ".ralph_workspace_tree.md").exists()
    assert "scripts/metrics.md" in payload["detected_output_paths"]

    summary = service.render_prompt_summary(payload)
    assert "scripts/metrics.md" in summary
    assert "outputs/metrics.md: missing" in summary


def test_workspace_snapshot_service_compacts_dataset_inventory(tmp_path: Path):
    service = WorkspaceSnapshotService()
    images_dir = tmp_path / "data" / "images"
    labels_dir = tmp_path / "data" / "labels"
    meta_dir = tmp_path / "data" / "meta"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(12):
        (images_dir / f"sample_{idx}.jpg").write_text("x", encoding="utf-8")
    for idx in range(3):
        (labels_dir / f"sample_{idx}.txt").write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
    (meta_dir / "dataset.json").write_text('{"classes":["item"]}', encoding="utf-8")

    payload = service.refresh(tmp_path)
    inventory = payload["inventory"]
    summary = service.render_prompt_summary(payload)
    ralph_snapshot = (tmp_path / ".ralph_workspace_tree.md").read_text(encoding="utf-8")

    assert inventory["total_files"] == 16
    formats = {item["suffix"]: item for item in inventory["formats"]}
    assert formats[".jpg"]["count"] == 12
    assert len(formats[".jpg"]["samples"]) == 10
    assert formats[".txt"]["count"] == 3
    assert formats[".json"]["count"] == 1
    assert "Total files: 16" in ralph_snapshot
    assert ".jpg: 12 file(s)" in ralph_snapshot
    assert "data/meta/dataset.json" in ralph_snapshot
    assert "Workspace file inventory: 16 files across 3 formats" in summary


def test_prompt_content_service_summarizes_large_metrics_json_without_dumping_lists(tmp_path: Path):
    service = PromptContentService()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "summary": {"accuracy": 0.91, "eval_mean_iou": 0.63, "loss": 0.22},
                "train_samples": list(range(500)),
                "eval_samples": [f"img_{index:04d}.jpg" for index in range(300)],
            }
        ),
        encoding="utf-8",
    )

    rendered = service.render_file_for_prompt(
        metrics_path,
        purpose="metrics_artifact",
        focus_terms=["accuracy", "iou", "loss"],
    )

    assert "[structured artifact summary]" in rendered
    assert "$.summary.accuracy = 0.91" in rendered
    assert "$.summary.eval_mean_iou = 0.63" in rendered
    assert "$.train_samples = list(500)" in rendered
    assert "img_0000.jpg" not in rendered
    assert "499" not in rendered


def test_prompt_content_service_compacts_large_json_payload_for_prompt():
    service = PromptContentService()
    payload = {
        "metrics": {
            "accuracy": 0.91,
            "train_examples": list(range(50)),
            "notes": "x" * 400,
        },
        "history": [{"attempt": index, "value": index / 100.0} for index in range(20)],
    }

    compact = service.compact_json_for_prompt(payload, focus_terms=["accuracy"])

    assert compact["metrics"]["accuracy"] == 0.91
    assert isinstance(compact["metrics"]["train_examples"], list)
    assert "... (+" in compact["metrics"]["train_examples"][-1]
    assert compact["metrics"]["notes"].endswith("chars)")
    assert isinstance(compact["history"], list)


def test_normalize_verification_payload_preserves_attempt_history_contract():
    previous = {
        "status": "failed",
        "metrics": {"val_accuracy": 0.82},
        "latest_attempt": 2,
        "history": [{"status": "passed", "metrics": {"val_accuracy": 0.8}}],
    }
    current = {"status": "failed", "metrics": {"val_accuracy": 0.86}}

    normalized = normalize_verification_payload(current, previous)

    assert normalized["attempt"] == 3
    assert normalized["latest_attempt"] == 3
    assert isinstance(normalized["history"], list)
    assert len(normalized["history"]) == 2
    assert normalized["history"][-1]["status"] == "failed"


def test_normalize_verification_payload_compacts_nested_strategy_history():
    previous = {
        "status": "failed",
        "metrics": {"accuracy": 0.82},
        "latest_attempt": 2,
        "improvement_strategy": {
            "kind": "quality_improvement_strategy",
            "quality_reason": "accuracy below target",
            "diagnosis": {"pattern": "near_target_plateau", "confidence": "medium"},
            "objective": {"metric_key": "accuracy", "target": 0.97, "current_value": 0.82},
            "chosen_intervention_id": "targeted_finetune",
            "chosen_intervention": {
                "id": "targeted_finetune",
                "actions": ["lr search", "batch tune"],
                "skill_paths": ["skills/pytorch-lightning/SKILL.md"],
            },
            "candidate_interventions": [{"id": "x"}],
            "history": {"experiment_attempts": [{"strategy": {"too": "big"}}]},
        },
    }
    current = {"status": "failed", "metrics": {"accuracy": 0.86}}

    normalized = normalize_verification_payload(current, previous)

    compact = normalized["history"][-1]["improvement_strategy"]
    assert compact["chosen_intervention_id"] == "targeted_finetune"
    assert "candidate_interventions" not in compact
    assert "history" not in compact


def test_improvement_strategy_service_compacts_experiment_history_strategies(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = ImprovementStrategyService(quality_gate_service=quality)

    task = {
        "goal": "Improve classification quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 97%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={"train_accuracy": 0.95, "eval_accuracy": 0.88},
    )
    experiment_history = [
        {
            "run_id": "old-run-1",
            "attempt": 1,
            "quality_status": "failed",
            "quality_reason": "accuracy below target",
            "metrics": {"accuracy": 0.84},
            "hyperparameters": {"epochs": 5, "learning_rate": 0.001},
            "strategy": {
                "chosen_intervention_id": "capacity_and_schedule_upgrade",
                "chosen_intervention": {"id": "capacity_and_schedule_upgrade", "skill_paths": ["skills/x/SKILL.md"]},
                "history": {"experiment_attempts": [{"nested": "too_big"}]},
                "candidate_interventions": [{"id": "too_big"}],
            },
        }
    ]

    strategy = service.build_for_quality_failure(
        run_id="run-compact-history",
        task=task,
        workspace_path=tmp_path,
        verification=verification,
        previous_verification=None,
        quality_reason="quality gate failed",
        experiment_history=experiment_history,
    )

    history_item = strategy["history"]["experiment_attempts"][0]
    assert history_item["run_id"] == "old-run-1"
    assert history_item["strategy"]["chosen_intervention_id"] == "capacity_and_schedule_upgrade"
    assert "history" not in history_item["strategy"]
    assert "candidate_interventions" not in history_item["strategy"]


def test_improvement_strategy_service_rotates_intervention_when_micro_training_branch_is_exhausted(tmp_path: Path):
    backlog = RalphBacklogService()
    quality = QualityGateService(ralph_backlog=backlog)
    service = ImprovementStrategyService(quality_gate_service=quality)

    task = {
        "goal": "Improve classification quality",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 97%"]),
    }
    verification = VerificationResult(
        status="passed",
        passed=True,
        commands=[],
        metrics={"train_accuracy": 0.95, "eval_accuracy": 0.88},
    )
    previous_verification = {
        "attempt": 2,
        "improvement_strategy": {
            "chosen_intervention_id": "capacity_and_schedule_upgrade",
            "chosen_intervention": {"id": "capacity_and_schedule_upgrade"},
        },
    }

    strategy = service.build_for_quality_failure(
        run_id="run-rotate-micro-policy",
        task=task,
        workspace_path=tmp_path,
        verification=verification,
        previous_verification=previous_verification,
        quality_reason="quality gate failed",
        micro_training_policy={
            "force_strategy_reset": True,
            "reason": "metric did not improve from 1 to 2 epochs",
            "next_epochs": 1,
        },
    )

    assert strategy["chosen_intervention_id"] != "capacity_and_schedule_upgrade"
    assert "rotated away from capacity_and_schedule_upgrade" in strategy["selection_reason"]
    assert any("1 epoch" in item for item in strategy["planner_directives"])


def test_planning_context_service_builds_structured_experiment_memory_summary():
    summary = PlanningContextService.build_experiment_memory_summary(
        [
            {
                "attempt": 1,
                "quality_status": "failed",
                "metrics": {"eval_accuracy": 0.82},
                "hyperparameters": {"epochs": 1, "learning_rate": 0.001},
                "strategy": {"chosen_intervention_id": "capacity_and_schedule_upgrade"},
            },
            {
                "attempt": 2,
                "quality_status": "failed",
                "metrics": {"eval_accuracy": 0.85},
                "hyperparameters": {"epochs": 2, "learning_rate": 0.001},
                "strategy": {"chosen_intervention_id": "capacity_and_schedule_upgrade"},
            },
            {
                "attempt": 3,
                "quality_status": "failed",
                "metrics": {"eval_accuracy": 0.851},
                "hyperparameters": {"epochs": 2, "learning_rate": 0.0005},
                "strategy": {"chosen_intervention_id": "targeted_finetune"},
            },
        ]
    )

    assert "attempts_analyzed: 3" in summary
    assert "latest_attempt:" in summary
    assert "best_attempt:" in summary
    assert "recent_metric_deltas:" in summary
    assert "repeated_interventions:" in summary


def test_planning_context_service_filters_lightning_skill_paths():
    verification = {
        "improvement_strategy": {
            "chosen_intervention": {
                "skill_paths": [
                    "skills/pytorch-lightning/SKILL.md",
                    "skills/seaborn/SKILL.md",
                ]
            }
        }
    }

    selected = PlanningContextService.selected_skill_paths_from_verification(verification)

    assert selected == ["skills/seaborn/SKILL.md"]


def test_baseline_research_service_builds_dataset_brief(tmp_path: Path):
    service = BaselineResearchService()
    task = {
        "goal": "Train a FashionMNIST classifier and report accuracy",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: accuracy >= 92%"]),
    }

    summary = service.build_summary(
        task=task,
        workspace_path=tmp_path,
        experiment_history=[
            {"attempt": 1, "metrics": {"eval_accuracy": 0.881}},
            {"attempt": 2, "metrics": {"eval_accuracy": 0.884}},
            {"attempt": 3, "metrics": {"eval_accuracy": 0.885}},
        ],
        previous_verification={"metrics": {"eval_accuracy": 0.885}},
    )

    assert "dataset_hint: fashionmnist" in summary.lower()
    assert "baseline_expectation:" in summary
    assert "next_research_focus:" in summary

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.config import Settings
from orchestrator.execution.verifier import Verifier


def _make_verifier(tmp_path: Path) -> Verifier:
    settings = Settings(
        _env_file=None,
        llm_provider="stub",
        sqlite_path=tmp_path / "orchestrator.db",
        workspace_root=tmp_path / "workspace",
        pdf_root=tmp_path / "workspace" / "knowledge" / "pdfs",
        runs_root=tmp_path / "workspace" / "runs",
    )
    return Verifier(settings)


def test_extract_metrics_from_markdown_bold_percent(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    metrics = verifier._extract_metrics_from_text("Best test accuracy: **87.57%** at epoch **5**.")
    assert "best_test_accuracy" in metrics
    assert metrics["best_test_accuracy"] == pytest.approx(0.8757, rel=1e-6)


def test_extract_metrics_from_text_ignores_operational_noise_and_keeps_multiple_metrics(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    text = "\n".join(
        [
            "session id: 019c83c2-ee10-7130-b32b-b64d7b3add05",
            "provider: openai",
            "Epoch 1/1 | loss=1.5534 | test_acc=70.12%",
        ]
    )
    metrics = verifier._extract_metrics_from_text(text)
    assert "session_id" not in metrics
    assert "provider" not in metrics
    assert metrics["loss"] == pytest.approx(1.5534, rel=1e-6)
    assert metrics["test_accuracy"] == pytest.approx(0.7012, rel=1e-6)


def test_read_workspace_metrics_parses_markdown_table(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "training_metrics_report.md").write_text(
        "\n".join(
            [
                "# Metrics",
                "",
                "| Epoch | Train Loss | Test Accuracy |",
                "| --- | ---: | ---: |",
                "| 1 | 0.5893 | 83.14% |",
                "| 2 | 0.4023 | 85.29% |",
                "| 3 | 0.3610 | 85.66% |",
                "| 4 | 0.3398 | 86.32% |",
                "| 5 | 0.3242 | 87.67% |",
            ]
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("epoch") == 5
    assert metrics.get("train_loss") == 0.3242
    assert metrics.get("test_accuracy") == 0.8767


def test_read_workspace_metrics_parses_metric_value_table(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.md").write_text(
        "\n".join(
            [
                "# Metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                "| Final Test Accuracy | 92.24% |",
                "| Final Test Loss | 0.381 |",
            ]
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("final_test_accuracy") == pytest.approx(0.9224, rel=1e-6)
    assert metrics.get("final_test_loss") == pytest.approx(0.381, rel=1e-6)


def test_read_workspace_metrics_prefers_structured_metrics_json_over_markdown(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "eval_accuracy": 0.931,
                    "train_accuracy": 0.980333,
                    "loss": 0.244,
                },
                "split": {"disjoint_from_training": True},
            }
        ),
        encoding="utf-8",
    )
    (workspace / "metrics.md").write_text(
        "\n".join(
            [
                "# Metrics",
                "",
                "- Eval Accuracy: 75.00%",
                "- Train Accuracy: 99.00%",
            ]
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("eval_accuracy") == pytest.approx(0.931, rel=1e-6)
    assert metrics.get("train_accuracy") == pytest.approx(0.980333, rel=1e-6)
    assert metrics.get("loss") == pytest.approx(0.244, rel=1e-6)
    assert metrics.get("split_integrity_passed") is True
    assert metrics.get("split_leakage_detected") is False


def test_read_workspace_metrics_discovers_nested_metrics_json(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    metrics_dir = workspace / "smoke_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accuracy": 0.4375,
                "eval_accuracy": 0.4375,
                "loss": 1.988,
            }
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("accuracy") == pytest.approx(0.4375, rel=1e-6)
    assert metrics.get("eval_accuracy") == pytest.approx(0.4375, rel=1e-6)
    assert metrics.get("loss") == pytest.approx(1.988, rel=1e-6)


def test_read_workspace_metrics_with_run_id_prefers_only_current_run_scoped_metrics(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    current_run_metrics = workspace / ".openin" / "runs" / "run-current" / "metrics.json"
    current_run_metrics.parent.mkdir(parents=True, exist_ok=True)
    current_run_metrics.write_text(
        json.dumps(
            {
                "dice": 45.0,
                "eval_accuracy": 0.48,
                "split_integrity_passed": True,
            }
        ),
        encoding="utf-8",
    )
    (workspace / ".openin" / "check_preflight_metrics.json").write_text(
        json.dumps(
            {
                "dice": 99.3,
                "eval_accuracy": 0.99,
                "metrics_artifact_kind": "preflight",
            }
        ),
        encoding="utf-8",
    )
    foreign_metrics = workspace / ".openin" / "runs" / "run-foreign" / "metrics.json"
    foreign_metrics.parent.mkdir(parents=True, exist_ok=True)
    foreign_metrics.write_text(
        json.dumps({"dice": 88.0, "eval_accuracy": 0.88}),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace, run_id="run-current")

    assert metrics.get("dice") == pytest.approx(45.0, rel=1e-6)
    assert metrics.get("eval_accuracy") == pytest.approx(0.48, rel=1e-6)
    assert metrics.get("metrics_artifact_kind") is None


def test_read_workspace_metrics_with_run_id_does_not_fall_back_to_preflight(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    run_dir = workspace / ".openin" / "runs" / "run-current"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "preflight_metrics.json").write_text(
        json.dumps({"dice": 99.3, "eval_accuracy": 0.99}),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace, run_id="run-current")

    assert metrics == {}


def test_read_workspace_metrics_recursively_extracts_nested_iou_metrics(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.json").write_text(
        json.dumps(
            {
                "split_integrity_passed": True,
                "smoke_results": {
                    "train_mean_iou": 0.71,
                    "eval_mean_iou": 0.63,
                    "train_per_class_iou": {"shoe": 0.81, "objects": 0.61},
                    "eval_per_class_iou": {"shoe": 0.72, "objects": 0.54},
                    "smoke_split_integrity": {"train_eval_disjoint": True},
                },
            }
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("train_mean_iou") == pytest.approx(0.71, rel=1e-6)
    assert metrics.get("eval_mean_iou") == pytest.approx(0.63, rel=1e-6)
    assert metrics.get("train_per_class_iou_shoe") == pytest.approx(0.81, rel=1e-6)
    assert metrics.get("eval_per_class_iou_objects") == pytest.approx(0.54, rel=1e-6)
    assert metrics.get("split_integrity_passed") is True
    assert metrics.get("split_leakage_detected") is False


def test_read_workspace_metrics_preserves_planning_only_flags_from_json(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.json").write_text(
        json.dumps(
            {
                "planning_only_report_detected": True,
                "training_deferred": True,
                "dataset_type": "coco",
                "split_integrity_passed": True,
                "class_count": 2,
            }
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("planning_only_report_detected") is True
    assert metrics.get("training_deferred") is True
    assert metrics.get("dataset_type") == "coco"
    assert metrics.get("split_integrity_passed") is True


@pytest.mark.asyncio
async def test_verifier_adds_intent_validation_details_when_metrics_primary_metric_matches(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.json").write_text(
        json.dumps(
            {
                "task_family": "segmentation",
                "primary_metric_key": "iou",
                "eval_mean_iou": 0.62,
            }
        ),
        encoding="utf-8",
    )
    task = {
        "goal": "Train segmentation model on coco dataset",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }

    result = await verifier.run(workspace, task=task, story_id="US-002")

    assert result.details["task_intent"]["task_family"] == "segmentation"
    assert result.details["intent_validation"]["status"] == "passed"
    assert result.metrics["metric_intent_drift_detected"] is False


@pytest.mark.asyncio
async def test_verifier_run_ignores_stale_workspace_metrics_when_run_id_is_provided(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    current_run_metrics = workspace / ".openin" / "runs" / "run-current" / "metrics.json"
    current_run_metrics.parent.mkdir(parents=True, exist_ok=True)
    current_run_metrics.write_text(
        json.dumps(
            {
                "task_family": "segmentation",
                "primary_metric_key": "dice",
                "dice": 45.0,
                "eval_accuracy": 0.48,
            }
        ),
        encoding="utf-8",
    )
    (workspace / ".openin" / "check_preflight_metrics.json").write_text(
        json.dumps(
            {
                "task_family": "segmentation",
                "primary_metric_key": "dice",
                "dice": 99.3,
                "eval_accuracy": 0.99,
            }
        ),
        encoding="utf-8",
    )
    task = {
        "goal": "Train segmentation model on coco dataset",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: dice >= 95%"]),
    }

    result = await verifier.run(workspace, run_id="run-current", task=task)

    assert result.metrics["dice"] == pytest.approx(45.0, rel=1e-6)


@pytest.mark.asyncio
async def test_verifier_marks_metric_intent_drift_when_reported_primary_metric_conflicts(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics.json").write_text(
        json.dumps(
            {
                "task_family": "segmentation",
                "primary_metric_key": "accuracy",
                "eval_accuracy": 0.99,
            }
        ),
        encoding="utf-8",
    )
    task = {
        "goal": "Train segmentation model on coco dataset",
        "constraints_json": json.dumps(["RALPH_REQUIRED_METRIC: iou >= 95%"]),
    }

    result = await verifier.run(workspace, task=task)

    assert result.details["intent_validation"]["status"] == "failed"
    assert "does not match" in result.details["intent_validation"]["reason"]
    assert result.metrics["metric_intent_drift_detected"] is True


def test_read_workspace_metrics_detects_split_leakage_from_report_text(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "smoke_test_metrics.md").write_text(
        "\n".join(
            [
                "# Smoke metrics",
                "",
                "- Evaluation split: `same balanced smoke subset used for the short training run`",
                "| key | value |",
                "| --- | --- |",
                "| accuracy | 1.000000 |",
            ]
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("split_leakage_detected") is True
    assert metrics.get("split_integrity_passed") is False


def test_read_workspace_metrics_detects_train_subset_overfit_check_phrase(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "smoke_test_metrics.md").write_text(
        "\n".join(
            [
                "# Smoke metrics",
                "",
                "| Metric | Value |",
                "| --- | --- |",
                "| Evaluation split | Train subset (smoke test overfit check) |",
                "| Accuracy | 100.00% |",
                "",
                "The accuracy gate applies to the training subset used for the smoke test.",
            ]
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("split_leakage_detected") is True
    assert metrics.get("split_integrity_passed") is False


def test_read_workspace_metrics_parses_backtick_accuracy_bullet_and_non_production_flags(tmp_path: Path):
    verifier = _make_verifier(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "metrics_report.md").write_text(
        "\n".join(
            [
                "# PennFudanPed Detection Metrics Report",
                "",
                "## Evaluation Summary",
                "",
                "- Run type: `reference_evaluation_fixture`",
                "- Accuracy: `1.0000`",
                "- Notes: `Reference markdown fixture generated from oracle predictions to validate reporting.`",
            ]
        ),
        encoding="utf-8",
    )

    metrics = verifier._read_workspace_metrics(workspace)
    assert metrics.get("accuracy") == pytest.approx(1.0, rel=1e-6)
    assert metrics.get("reference_evaluation_fixture_detected") is True
    assert metrics.get("oracle_predictions_detected") is True
    assert metrics.get("non_production_report_detected") is True

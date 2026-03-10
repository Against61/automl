from __future__ import annotations

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

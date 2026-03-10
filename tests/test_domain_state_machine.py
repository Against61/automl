from __future__ import annotations

import pytest

from orchestrator.domain.errors import InvalidTransitionError
from orchestrator.domain.quality_gate import QualityRequirement, evaluate_quality, parse_quality_requirement_from_text
from orchestrator.domain.state_machine import RunStateMachine
from orchestrator.persistence.schemas import RunStatus


def test_run_state_machine_valid_sequence():
    status = RunStatus.RECEIVED
    status = RunStateMachine.transition(status, "context_ready")
    assert status == RunStatus.CONTEXT_READY
    status = RunStateMachine.transition(status, "plan_ready")
    assert status == RunStatus.PLAN_READY
    status = RunStateMachine.transition(status, "start_execution")
    assert status == RunStatus.EXECUTING
    status = RunStateMachine.transition(status, "verify")
    assert status == RunStatus.VERIFYING
    status = RunStateMachine.transition(status, "package")
    assert status == RunStatus.PACKAGING
    status = RunStateMachine.transition(status, "complete")
    assert status == RunStatus.COMPLETED


def test_run_state_machine_invalid_transition():
    with pytest.raises(InvalidTransitionError):
        RunStateMachine.transition(RunStatus.RECEIVED, "verify")


def test_quality_requirement_parse_and_evaluate():
    requirement = parse_quality_requirement_from_text("Need accuracy >= 95%")
    assert requirement is not None
    assert requirement.metric_key == "accuracy"
    assert requirement.metric_min == 0.95

    result_fail = evaluate_quality({"test_accuracy": 0.93}, requirement)
    assert result_fail.passed is False

    result_pass = evaluate_quality({"test_accuracy": 0.951}, requirement)
    assert result_pass.passed is True


def test_quality_evaluate_with_explicit_requirement():
    requirement = QualityRequirement(metric_key="accuracy", metric_min=0.8)
    result = evaluate_quality({"accuracy": "82%"}, requirement)
    assert result.passed is True


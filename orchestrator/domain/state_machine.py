from __future__ import annotations

from dataclasses import dataclass

from orchestrator.domain.errors import InvalidTransitionError
from orchestrator.persistence.schemas import RunStatus


@dataclass(frozen=True, slots=True)
class TransitionRule:
    source: RunStatus
    action: str
    target: RunStatus


class RunStateMachine:
    """Single source of truth for run status transitions."""

    _rules: tuple[TransitionRule, ...] = (
        TransitionRule(RunStatus.RECEIVED, "context_ready", RunStatus.CONTEXT_READY),
        TransitionRule(RunStatus.CONTEXT_READY, "plan_ready", RunStatus.PLAN_READY),
        TransitionRule(RunStatus.CONTEXT_READY, "start_execution", RunStatus.EXECUTING),
        TransitionRule(RunStatus.PLAN_READY, "start_execution", RunStatus.EXECUTING),
        TransitionRule(RunStatus.PLAN_READY, "request_approval", RunStatus.WAITING_APPROVAL),
        TransitionRule(RunStatus.PLAN_READY, "request_plan_review", RunStatus.WAITING_PLAN_REVIEW),
        TransitionRule(RunStatus.EXECUTING, "request_approval", RunStatus.WAITING_APPROVAL),
        TransitionRule(RunStatus.EXECUTING, "request_plan_review", RunStatus.WAITING_PLAN_REVIEW),
        TransitionRule(RunStatus.EXECUTING, "verify", RunStatus.VERIFYING),
        TransitionRule(RunStatus.WAITING_APPROVAL, "resume_execution", RunStatus.EXECUTING),
        TransitionRule(RunStatus.WAITING_PLAN_REVIEW, "resume_execution", RunStatus.EXECUTING),
        TransitionRule(RunStatus.WAITING_PLAN_REVIEW, "context_ready", RunStatus.CONTEXT_READY),
        TransitionRule(RunStatus.VERIFYING, "package", RunStatus.PACKAGING),
        TransitionRule(RunStatus.VERIFYING, "context_ready", RunStatus.CONTEXT_READY),
        TransitionRule(RunStatus.PACKAGING, "complete", RunStatus.COMPLETED),
        TransitionRule(RunStatus.PACKAGING, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.RECEIVED, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.CONTEXT_READY, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.PLAN_READY, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.WAITING_APPROVAL, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.WAITING_PLAN_REVIEW, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.EXECUTING, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.VERIFYING, "fail", RunStatus.FAILED),
        TransitionRule(RunStatus.RECEIVED, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.CONTEXT_READY, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.PLAN_READY, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.WAITING_APPROVAL, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.WAITING_PLAN_REVIEW, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.EXECUTING, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.VERIFYING, "cancel", RunStatus.CANCELLED),
        TransitionRule(RunStatus.PACKAGING, "cancel", RunStatus.CANCELLED),
    )

    @classmethod
    def is_terminal(cls, status: RunStatus) -> bool:
        return status in RunStatus.terminal()

    @classmethod
    def can_transition(cls, source: RunStatus, action: str) -> bool:
        if cls.is_terminal(source):
            return False
        return any(rule.source == source and rule.action == action for rule in cls._rules)

    @classmethod
    def transition(cls, source: RunStatus, action: str) -> RunStatus:
        if cls.is_terminal(source):
            raise InvalidTransitionError(f"cannot transition from terminal state: {source.value}")
        for rule in cls._rules:
            if rule.source == source and rule.action == action:
                return rule.target
        raise InvalidTransitionError(f"invalid transition: {source.value} --{action}--> ?")

    @classmethod
    def infer_action(cls, source: RunStatus, target: RunStatus) -> str:
        if source == target:
            return "noop"
        for rule in cls._rules:
            if rule.source == source and rule.target == target:
                return rule.action
        return "direct_set"


from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from orchestrator.config import Settings
from orchestrator.persistence.schemas import PlannerStep


@dataclass(slots=True)
class PolicyDecision:
    layer: str
    subject: str
    decision: str
    reason: str


class PolicyEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._deny_patterns = [re.compile(p, flags=re.IGNORECASE) for p in settings.deny_regexes]
        self._approval_patterns = [re.compile(p, flags=re.IGNORECASE) for p in settings.approval_regexes]
        self._network_block_patterns = [
            re.compile(p, flags=re.IGNORECASE)
            for p in (r"\bcurl\b", r"\bwget\b", r"\bnc\b", r"\bncat\b")
        ]

    def evaluate_step(self, step: PlannerStep, workspace_root: Path) -> list[PolicyDecision]:
        decisions: list[PolicyDecision] = []
        subject = step.command or (" && ".join(step.commands) if step.commands else step.instruction)

        if step.risk_level == "high":
            decisions.append(
                PolicyDecision(
                    layer="run",
                    subject=subject,
                    decision="REQUIRE_APPROVAL",
                    reason="step risk_level=high",
                )
            )

        for pattern in self._deny_patterns:
            if pattern.search(subject):
                decisions.append(
                    PolicyDecision(
                        layer="global",
                        subject=subject,
                        decision="DENY",
                        reason=f"matched deny pattern: {pattern.pattern}",
                    )
                )

        for pattern in self._approval_patterns:
            if pattern.search(subject):
                decisions.append(
                    PolicyDecision(
                        layer="orchestrator",
                        subject=subject,
                        decision="REQUIRE_APPROVAL",
                        reason=f"matched approval pattern: {pattern.pattern}",
                    )
                )

        command_for_path_check = step.command or (" && ".join(step.commands) if step.commands else None)
        if command_for_path_check and not self._command_paths_allowed(command_for_path_check, workspace_root):
            decisions.append(
                PolicyDecision(
                    layer="sandbox",
                    subject=command_for_path_check,
                    decision="DENY",
                    reason="command references path outside allowed roots",
                )
            )

        if self.settings.block_network_commands and command_for_path_check:
            for pattern in self._network_block_patterns:
                if pattern.search(command_for_path_check):
                    decisions.append(
                        PolicyDecision(
                            layer="sandbox",
                            subject=command_for_path_check,
                            decision="DENY",
                            reason=f"network command blocked: {pattern.pattern}",
                        )
                    )
                    break

        budget_reason = self._training_budget_violation(step=step, workspace_root=workspace_root)
        if budget_reason:
            decisions.append(
                PolicyDecision(
                    layer="orchestrator",
                    subject=subject,
                    decision="DENY",
                    reason=budget_reason,
                )
            )

        if not decisions:
            decisions.append(
                PolicyDecision(
                    layer="global",
                    subject=subject,
                    decision="ALLOW",
                    reason="no rule matched",
                )
            )
        return decisions

    def plan_requires_approval(self, steps: list[PlannerStep], workspace_root: Path) -> tuple[bool, list[PolicyDecision]]:
        merged: list[PolicyDecision] = []
        requires = False
        for step in steps:
            decisions = self.evaluate_step(step, workspace_root)
            merged.extend(decisions)
            for decision in decisions:
                if decision.decision == "DENY":
                    return False, merged
                if decision.decision == "REQUIRE_APPROVAL":
                    requires = True
        return requires, merged

    def _training_budget_violation(self, *, step: PlannerStep, workspace_root: Path) -> str | None:
        if str(step.step_intent) != "StepIntent.run_training" and getattr(step.step_intent, "value", step.step_intent) != "run_training":
            if str(step.operation) != "StepOperation.run_training" and getattr(step.operation, "value", step.operation) != "run_training":
                return None

        command = step.command or (step.commands[0] if step.commands else "")
        if not command:
            return None
        estimate = self._estimate_training_budget(command=command, workspace_root=workspace_root)
        if estimate is None:
            return None
        epochs, trial_count, budget_units = estimate
        if budget_units <= self.settings.cpu_training_budget_epoch_trials_limit:
            return None
        return (
            "training budget exceeds cpu guardrail: "
            f"epochs({epochs}) * trial_count({trial_count}) = {budget_units} > "
            f"{self.settings.cpu_training_budget_epoch_trials_limit}"
        )

    def _estimate_training_budget(self, *, command: str, workspace_root: Path) -> tuple[int, int, int] | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None

        def _read_flag(flag_names: tuple[str, ...]) -> str | None:
            for idx, token in enumerate(tokens):
                for flag in flag_names:
                    if token == flag and idx + 1 < len(tokens):
                        return tokens[idx + 1]
                    if token.startswith(f"{flag}="):
                        return token.split("=", 1)[1]
            return None

        def _to_int(value: str | None) -> int | None:
            if value is None:
                return None
            try:
                return int(float(value))
            except ValueError:
                return None

        epochs = _to_int(_read_flag(("--epochs", "--epoch")))
        trial_count = _to_int(_read_flag(("--trial-count", "--trial_count")))
        script_path = self._extract_python_script_path(tokens=tokens, workspace_root=workspace_root)
        if script_path is not None:
            default_epochs, default_trial_count = self._read_training_defaults(script_path)
            if epochs is None:
                epochs = default_epochs
            if trial_count is None:
                trial_count = default_trial_count

        if epochs is None:
            return None
        if trial_count is None:
            trial_count = 1
        return epochs, trial_count, epochs * trial_count

    def _extract_python_script_path(self, *, tokens: list[str], workspace_root: Path) -> Path | None:
        if not tokens:
            return None
        binary = Path(tokens[0]).name.lower()
        if binary not in {"python", "python3"}:
            return None
        for token in tokens[1:]:
            if token.startswith("-"):
                continue
            candidate = Path(token)
            if not candidate.suffix == ".py":
                return None
            resolved = candidate if candidate.is_absolute() else (workspace_root / candidate)
            try:
                resolved = resolved.resolve()
            except OSError:
                return None
            return resolved if resolved.exists() else None
        return None

    def _read_training_defaults(self, script_path: Path) -> tuple[int | None, int | None]:
        try:
            content = script_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None, None
        epoch_match = re.search(r"--epochs[\"']?,\s*type=\w+,\s*default=(\d+)", content)
        trial_match = re.search(r"--trial[-_]count[\"']?,\s*type=\w+,\s*default=(\d+)", content)
        epochs = int(epoch_match.group(1)) if epoch_match else None
        trial_count = int(trial_match.group(1)) if trial_match else None
        return epochs, trial_count

    def _command_paths_allowed(self, command: str, workspace_root: Path) -> bool:
        allowed_roots = self.settings.allowed_paths + [workspace_root.resolve()]
        try:
            tokens = shlex.split(command)
        except ValueError:
            return False
        for token in tokens:
            if token.startswith("-"):
                continue
            if token.startswith("http://") or token.startswith("https://"):
                continue
            path = Path(token)
            if not path.is_absolute():
                continue
            resolved = path.resolve()
            if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
                return False
        return True

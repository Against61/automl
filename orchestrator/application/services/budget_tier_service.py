from __future__ import annotations

import shlex
from typing import Any

from orchestrator.application.services.evaluation_contract_service import EvaluationContractService
from orchestrator.persistence.schemas import PlannerPlan, PlannerStep, RunRecord


class BudgetTierService:
    def __init__(self, evaluation_contract_service: EvaluationContractService | None = None) -> None:
        self.evaluation_contract_service = evaluation_contract_service or EvaluationContractService()

    def apply_to_step(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
        plan: PlannerPlan,
        step: PlannerStep,
        step_index: int,
    ) -> tuple[PlannerStep, dict[str, Any] | None]:
        contract = self.evaluation_contract_service.load_from_run(run) or self.evaluation_contract_service.load_from_task(task)
        if contract is None:
            return step, None
        tier = self._resolve_tier(run=run, contract=contract, step=step)
        if tier is None:
            if step.action == "codex" and self._has_upcoming_training_step(plan=plan, step_index=step_index):
                return self._with_codex_budget_hint(step=step, contract=contract, training_tier=self._training_tier_name(run, contract)), None
            return step, None
        if step.action == "shell":
            return self._with_shell_budget_env(step=step, contract=contract, tier=tier), {
                "name": tier.name,
                "max_effective_train_seconds": tier.max_effective_train_seconds,
                "max_epochs": tier.max_epochs,
                "max_steps": tier.max_steps,
            }
        return self._with_codex_budget_hint(step=step, contract=contract, training_tier=tier.name), {
            "name": tier.name,
            "max_effective_train_seconds": tier.max_effective_train_seconds,
            "max_epochs": tier.max_epochs,
            "max_steps": tier.max_steps,
        }

    def _resolve_tier(self, *, run: RunRecord, contract, step: PlannerStep):
        if step.action != "shell":
            return None
        if self._is_preflight(step):
            return self._tier(contract, "smoke")
        if step.step_intent.value != "run_training":
            return None
        return self._tier(contract, self._training_tier_name(run, contract))

    @staticmethod
    def _has_upcoming_training_step(*, plan: PlannerPlan, step_index: int) -> bool:
        for next_step in plan.steps[step_index + 1 :]:
            if next_step.action == "shell" and next_step.step_intent.value == "run_training":
                return True
        return False

    @staticmethod
    def _is_preflight(step: PlannerStep) -> bool:
        blob = " ".join(
            [
                step.id,
                step.title,
                step.command or "",
                *(step.commands or []),
            ]
        ).lower()
        return "preflight" in blob or "--preflight" in blob

    @staticmethod
    def _training_tier_name(run: RunRecord, contract) -> str:
        explicit = str(getattr(run, "budget_tier", "") or "").strip().lower()
        if explicit in {"micro", "short", "full"}:
            return explicit
        if int(run.execution_cycle) <= 0:
            return "micro"
        if int(run.execution_cycle) == 1:
            return "short"
        return "full"

    def current_training_tier(self, *, run: RunRecord, task: dict[str, Any]) -> dict[str, Any] | None:
        contract = self.evaluation_contract_service.load_from_run(run) or self.evaluation_contract_service.load_from_task(task)
        if contract is None:
            return None
        tier = self._tier(contract, self._training_tier_name(run, contract))
        if tier is None:
            return None
        return self._tier_payload(tier)

    def next_training_tier(self, *, run: RunRecord, task: dict[str, Any]) -> dict[str, Any] | None:
        contract = self.evaluation_contract_service.load_from_run(run) or self.evaluation_contract_service.load_from_task(task)
        if contract is None:
            return None
        current = self._training_tier_name(run, contract)
        order = ["micro", "short", "full"]
        try:
            current_index = order.index(current)
        except ValueError:
            current_index = 0
        if current_index >= len(order) - 1:
            return None
        next_tier = self._tier(contract, order[current_index + 1])
        if next_tier is None:
            return None
        return self._tier_payload(next_tier)

    @staticmethod
    def _tier(contract, name: str):
        for tier in contract.budget_tiers:
            if tier.name == name:
                return tier
        return None

    def _with_shell_budget_env(self, *, step: PlannerStep, contract, tier) -> PlannerStep:
        env_prefix = self._budget_env_prefix(contract=contract, tier=tier)
        commands = [f"{env_prefix} {command}".strip() for command in (step.commands or [])]
        command = f"{env_prefix} {step.command}".strip() if step.command else None
        return step.model_copy(update={"commands": commands, "command": command})

    def _with_codex_budget_hint(self, *, step: PlannerStep, contract, training_tier: str) -> PlannerStep:
        budget_note = (
            "\n\nRuntime budget contract:\n"
            f"- Generated training code must respect `OPENIN_BUDGET_TIER={training_tier}` at execution time.\n"
            "- Runtime will also provide: `OPENIN_MAX_EFFECTIVE_TRAIN_SECONDS`, `OPENIN_MAX_EPOCHS`, "
            "`OPENIN_MAX_STEPS`, `OPENIN_MICRO_SPLIT_ID`, `OPENIN_MACRO_SPLIT_ID`.\n"
            "- Tier mapping is strict: smoke for preflight, then micro/short/full for training cycles.\n"
            "- Prefer code paths that honor those environment variables without requiring prompt-time edits.\n"
        )
        codex_prompt = (step.codex_prompt or step.instruction or "") + budget_note
        instruction = (step.instruction or step.codex_prompt or step.title) + budget_note
        return step.model_copy(update={"codex_prompt": codex_prompt, "instruction": instruction})

    @staticmethod
    def _budget_env_prefix(*, contract, tier) -> str:
        parts = [
            f"OPENIN_BUDGET_TIER={shlex.quote(tier.name)}",
            f"OPENIN_MAX_EFFECTIVE_TRAIN_SECONDS={shlex.quote(str(tier.max_effective_train_seconds))}",
            f"OPENIN_MICRO_SPLIT_ID={shlex.quote(str(contract.micro_split_id))}",
            f"OPENIN_MACRO_SPLIT_ID={shlex.quote(str(contract.macro_split_id))}",
            f"OPENIN_PROXY_METRIC_KIND={shlex.quote(str(contract.proxy_metric_kind))}",
        ]
        if tier.max_epochs is not None:
            parts.append(f"OPENIN_MAX_EPOCHS={shlex.quote(str(tier.max_epochs))}")
        if tier.max_steps is not None:
            parts.append(f"OPENIN_MAX_STEPS={shlex.quote(str(tier.max_steps))}")
        return " ".join(parts)

    @staticmethod
    def _tier_payload(tier) -> dict[str, Any]:
        return {
            "name": tier.name,
            "max_effective_train_seconds": tier.max_effective_train_seconds,
            "max_epochs": tier.max_epochs,
            "max_steps": tier.max_steps,
        }

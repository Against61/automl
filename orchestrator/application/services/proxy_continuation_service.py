from __future__ import annotations

from typing import Any

from orchestrator.application.services.budget_tier_service import BudgetTierService
from orchestrator.application.services.evaluation_contract_service import EvaluationContractService


class ProxyContinuationService:
    def __init__(
        self,
        *,
        evaluation_contract_service: EvaluationContractService | None = None,
        budget_tier_service: BudgetTierService | None = None,
    ) -> None:
        self.evaluation_contract_service = evaluation_contract_service or EvaluationContractService()
        self.budget_tier_service = budget_tier_service or BudgetTierService(
            evaluation_contract_service=self.evaluation_contract_service
        )

    def decide(
        self,
        *,
        run: Any,
        task: dict[str, Any],
        proxy_metric: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        contract = self.evaluation_contract_service.load_from_run(run) or self.evaluation_contract_service.load_from_task(task)
        if contract is None:
            return None
        current_tier = self.budget_tier_service.current_training_tier(run=run, task=task)
        if not isinstance(current_tier, dict):
            return None
        current_name = str(current_tier.get("name") or "").strip().lower()
        if current_name not in {"micro", "short"}:
            return None
        next_tier = self.budget_tier_service.next_training_tier(run=run, task=task)
        threshold = float(contract.min_effect_size)
        if not isinstance(proxy_metric, dict):
            return {
                "decision": "discard",
                "from_tier": current_name,
                "to_tier": "micro",
                "threshold": threshold,
                "reason": "proxy metric unavailable at non-final tier",
            }

        history_count = int(proxy_metric.get("history_count") or 0)
        proxy_gain = self._to_float(proxy_metric.get("proxy_gain"))
        if history_count <= 0 and isinstance(next_tier, dict):
            return {
                "decision": "promote",
                "from_tier": current_name,
                "to_tier": str(next_tier.get("name") or current_name),
                "threshold": threshold,
                "proxy_gain": proxy_gain,
                "reason": "no comparable proxy baseline exists yet; promote once to establish signal",
            }
        if proxy_gain is not None and proxy_gain >= threshold and isinstance(next_tier, dict):
            return {
                "decision": "promote",
                "from_tier": current_name,
                "to_tier": str(next_tier.get("name") or current_name),
                "threshold": threshold,
                "proxy_gain": proxy_gain,
                "reason": f"proxy gain {proxy_gain:.6g} reached threshold {threshold:.6g}",
            }
        return {
            "decision": "discard",
            "from_tier": current_name,
            "to_tier": "micro",
            "threshold": threshold,
            "proxy_gain": proxy_gain,
            "reason": (
                f"proxy gain {proxy_gain:.6g} below threshold {threshold:.6g}"
                if proxy_gain is not None
                else "proxy gain unavailable at non-final tier"
            ),
        }

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value in {None, ""}:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BudgetTierName = Literal["smoke", "micro", "short", "full"]
MetricDirection = Literal["max", "min"]
MetricScale = Literal["ratio", "percent"]
ProxyMetricKind = Literal["micro_loss", "micro_primary"]


@dataclass(frozen=True, slots=True)
class BudgetTier:
    name: BudgetTierName
    max_effective_train_seconds: int
    max_epochs: int | None
    max_steps: int | None
    requires_real_dataset: bool


@dataclass(frozen=True, slots=True)
class EvaluationContract:
    task_family: str
    primary_metric_key: str
    primary_direction: MetricDirection
    primary_scale: MetricScale
    proxy_metric_kind: ProxyMetricKind
    proxy_direction: MetricDirection
    target_value: float | None
    micro_split_id: str
    macro_split_id: str
    min_effect_size: float
    budget_tiers: tuple[BudgetTier, ...]

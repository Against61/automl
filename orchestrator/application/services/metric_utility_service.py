from __future__ import annotations

from typing import Any


class MetricUtilityService:
    _PERCENT_UNITS = {"%", "percent", "pct"}

    def scale_for_unit(self, unit: str | None) -> str:
        normalized = str(unit or "").strip().lower()
        if normalized in self._PERCENT_UNITS:
            return "percent"
        return "ratio"

    def to_utility(self, raw_value: float, scale: str) -> float:
        if scale == "percent":
            absolute = abs(raw_value)
            if 1.0 < absolute <= 100.0:
                return raw_value / 100.0
        return raw_value

    def target_to_utility(self, target_value: float, unit: str | None) -> float:
        return self.to_utility(target_value, self.scale_for_unit(unit))

    def progress(
        self,
        *,
        utility: float,
        prior_utilities: list[float],
        target_utility: float | None,
    ) -> dict[str, float | int | None]:
        baseline_utility = prior_utilities[0] if prior_utilities else utility
        best_utility_so_far = max(prior_utilities) if prior_utilities else utility
        delta_best = utility - best_utility_so_far
        delta_base = utility - baseline_utility
        if target_utility is not None:
            denominator = max(1e-8, target_utility - baseline_utility)
            gap_closed = delta_base / denominator
        else:
            gap_closed = delta_base
        return {
            "utility": utility,
            "baseline_utility": baseline_utility,
            "best_utility_so_far": best_utility_so_far,
            "delta_best": delta_best,
            "delta_base": delta_base,
            "target_utility": target_utility,
            "gap_closed": gap_closed,
            "history_count": len(prior_utilities),
        }

    def gain_per_budget(self, gain: float | None, effective_train_seconds: float | None) -> float | None:
        if gain is None or effective_train_seconds is None:
            return None
        if effective_train_seconds <= 0:
            return None
        return gain / effective_train_seconds

    def coerce_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", ".")
            if "%" in cleaned:
                cleaned = cleaned.replace("%", "")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

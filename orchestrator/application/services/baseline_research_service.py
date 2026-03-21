from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from orchestrator.application.services.task_intent_service import TaskIntentService
from orchestrator.execution.metric_parsing import normalize_metric_key


class BaselineResearchService:
    _DATASET_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("fashionmnist", ("fashionmnist", "fashion-mnist", "fashion mnist")),
        ("mnist", ("mnist",)),
        ("coco-segmentation", ("coco-segmentation", "coco segmentation", "coco")),
        ("cifar10", ("cifar10", "cifar-10", "cifar 10")),
        ("cifar100", ("cifar100", "cifar-100", "cifar 100")),
    )

    _PROFILE_BRIEFS: dict[tuple[str, str], dict[str, tuple[str, ...] | str]] = {
        (
            "fashionmnist",
            "classification",
        ): {
            "expectation": (
                "A competent CNN baseline should usually move into the high-80s or low-90s quickly; "
                "if it stalls below that, inspect preprocessing, optimizer settings, and label wiring before "
                "trying bigger architecture changes."
            ),
            "focus": (
                "Verify normalization and label mapping before broad hyperparameter search.",
                "Prefer recipe changes with measurable validation lift over longer training alone.",
            ),
        },
        (
            "mnist",
            "classification",
        ): {
            "expectation": (
                "Simple CNN baselines commonly clear very high accuracy early. If early accuracy is weak, "
                "suspect data pipeline or evaluation-split issues before architecture complexity."
            ),
            "focus": (
                "Sanity-check transforms, label mapping, and split integrity.",
                "Use early validation signal to decide whether more epochs are worth paying for.",
            ),
        },
        (
            "coco-segmentation",
            "segmentation",
        ): {
            "expectation": (
                "Early smoke metrics are noisy. First prove annotation loading, mask fidelity, and disjoint "
                "splits before treating IoU or Dice changes as meaningful model progress."
            ),
            "focus": (
                "Prioritize real-mask integrity and dataset adapter correctness before model tuning.",
                "Treat tiny-budget overlap metrics as directional only until the data path is stable.",
            ),
        },
        (
            "generic",
            "classification",
        ): {
            "expectation": (
                "For ordinary classification tasks, early runs should show clear movement if the recipe is healthy. "
                "Flat metrics often point to bad splits, missing normalization, or optimizer misconfiguration."
            ),
            "focus": (
                "Check data pipeline and optimizer assumptions before rotating architecture aggressively.",
                "Prefer interventions that explain metric changes, not just more epochs.",
            ),
        },
        (
            "generic",
            "segmentation",
        ): {
            "expectation": (
                "For segmentation tasks, early effort should validate mask provenance, split integrity, and loss wiring "
                "before interpreting overlap metrics as stable signal."
            ),
            "focus": (
                "Keep acceptance tied to explicit ground-truth masks on a held-out split.",
                "Treat data-path repairs as higher priority than architecture search until masks are trustworthy.",
            ),
        },
        (
            "generic",
            "generic",
        ): {
            "expectation": (
                "Use the first few runs to establish a believable baseline and separate data-path bugs from true "
                "underfitting or overfitting."
            ),
            "focus": (
                "Prefer interventions that produce interpretable metric movement.",
                "Avoid repeating recipe changes that already failed without measurable lift.",
            ),
        },
    }

    _METRIC_PREFERENCE: tuple[tuple[str, bool], ...] = (
        ("eval_accuracy", True),
        ("test_accuracy", True),
        ("accuracy", True),
        ("val_accuracy", True),
        ("out_evaluation_accuracy", True),
        ("iou", True),
        ("mean_iou", True),
        ("dice", True),
        ("map50_95", True),
        ("map50", True),
        ("f1", True),
        ("roc_auc", True),
        ("eval_loss", False),
        ("val_loss", False),
        ("loss", False),
    )

    def __init__(self) -> None:
        self.task_intent_service = TaskIntentService()

    def build_summary(
        self,
        *,
        task: dict[str, Any],
        workspace_path: Path,
        experiment_history: list[dict[str, Any]] | None = None,
        previous_verification: dict[str, Any] | None = None,
    ) -> str:
        intent = self.task_intent_service.infer_from_task(task=task, workspace_path=workspace_path)
        dataset_hint = self._detect_dataset_hint(task=task, workspace_path=workspace_path)
        profile = self._profile_for(dataset_hint=dataset_hint, task_family=intent.task_family)
        requirement_metric = self._extract_required_metric(task)
        current_key, current_value = self._extract_current_metric(
            previous_verification=previous_verification,
            preferred_metric=requirement_metric or intent.primary_metric_key,
        )
        plateau_state = self._plateau_state(experiment_history or [], preferred_metric=current_key or requirement_metric)

        lines = [
            f"- dataset_hint: {dataset_hint}",
            f"- task_family: {intent.task_family}",
            f"- primary_metric: {requirement_metric or intent.primary_metric_key or 'not set'}",
            f"- baseline_expectation: {profile['expectation']}",
        ]
        if current_key and current_value is not None:
            lines.append(f"- current_position: last seen {current_key}={current_value:.4f}")
        if plateau_state:
            lines.append(f"- recent_pattern: {plateau_state}")
        focus_items = list(profile["focus"])
        if focus_items:
            lines.append(f"- next_research_focus: {focus_items[0]}")
        if len(focus_items) > 1:
            lines.append(f"- secondary_focus: {focus_items[1]}")
        return "\n".join(lines)

    def _detect_dataset_hint(self, *, task: dict[str, Any], workspace_path: Path) -> str:
        try:
            constraints = [str(item).strip() for item in json.loads(task.get("constraints_json") or "[]")]
        except Exception:
            constraints = []
        parts = [str(task.get("goal") or ""), *constraints]
        try:
            root_entries = [item.name for item in workspace_path.iterdir()]
        except OSError:
            root_entries = []
        blob = " ".join(parts + root_entries).lower()
        for dataset_name, markers in self._DATASET_PATTERNS:
            if any(marker in blob for marker in markers):
                return dataset_name
        return "generic"

    def _profile_for(self, *, dataset_hint: str, task_family: str) -> dict[str, tuple[str, ...] | str]:
        return self._PROFILE_BRIEFS.get(
            (dataset_hint, task_family),
            self._PROFILE_BRIEFS.get((dataset_hint, "generic"), self._PROFILE_BRIEFS[("generic", task_family)]),
        )

    def _extract_required_metric(self, task: dict[str, Any]) -> str | None:
        try:
            constraints = [str(item).strip() for item in json.loads(task.get("constraints_json") or "[]")]
        except Exception:
            constraints = []
        for item in constraints:
            lowered = item.lower()
            if "required_metric" not in lowered:
                continue
            parts = item.split(":", 1)
            if len(parts) != 2:
                continue
            tail = parts[1].strip()
            match = re.match(r"([A-Za-z0-9_./ -]+?)\s*(?:>=|<=|==|!=|>|<|=)", tail)
            if not match:
                continue
            normalized = normalize_metric_key(match.group(1))
            if normalized:
                return normalized
        return None

    def _extract_current_metric(
        self,
        *,
        previous_verification: dict[str, Any] | None,
        preferred_metric: str | None,
    ) -> tuple[str | None, float | None]:
        if not isinstance(previous_verification, dict):
            return None, None
        metrics = previous_verification.get("metrics")
        if not isinstance(metrics, dict):
            return None, None
        if preferred_metric:
            preferred_value = self._coerce_float(metrics.get(preferred_metric))
            if preferred_value is not None:
                return preferred_metric, preferred_value
        for key, _higher_is_better in self._METRIC_PREFERENCE:
            value = self._coerce_float(metrics.get(key))
            if value is not None:
                return key, value
        for key, value in metrics.items():
            parsed = self._coerce_float(value)
            if parsed is not None:
                return str(key), parsed
        return None, None

    def _plateau_state(self, attempts: list[dict[str, Any]], preferred_metric: str | None) -> str | None:
        comparable: list[float] = []
        for item in attempts[-4:]:
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics")
            if not isinstance(metrics, dict):
                continue
            value = None
            if preferred_metric:
                value = self._coerce_float(metrics.get(preferred_metric))
            if value is None:
                for key, _higher_is_better in self._METRIC_PREFERENCE:
                    value = self._coerce_float(metrics.get(key))
                    if value is not None:
                        break
            if value is not None:
                comparable.append(value)
        if len(comparable) < 3:
            return None
        if max(comparable) - min(comparable) <= 0.01:
            return "recent attempts are within a narrow metric band; prefer strategy or recipe changes over more identical epochs"
        if comparable[-1] > comparable[0]:
            return "recent attempts are still improving; expanding budget is defensible if recipe changes stay small"
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip().replace(",", ".")
            cleaned = re.sub(r"[^0-9eE+\-.]", "", raw)
            if not cleaned:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.application.services.task_intent_service import TaskIntentService
from orchestrator.execution.metric_parsing import normalize_metric_key

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database
    from orchestrator.persistence.common import PdfChunkRow


class BaselineResearchService:
    _DATASET_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("fashionmnist", ("fashionmnist", "fashion-mnist", "fashion mnist")),
        ("mnist", ("mnist",)),
        ("coco-segmentation", ("coco-segmentation", "coco segmentation", "coco")),
        ("cifar10", ("cifar10", "cifar-10", "cifar 10")),
        ("cifar100", ("cifar100", "cifar-100", "cifar 100")),
    )

    _FALLBACK_BRIEFS: dict[tuple[str, str], dict[str, tuple[str, ...] | str]] = {
        ("fashionmnist", "classification"): {
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
        ("mnist", "classification"): {
            "expectation": (
                "Simple CNN baselines commonly clear very high accuracy early. If early accuracy is weak, "
                "suspect data pipeline or evaluation-split issues before architecture complexity."
            ),
            "focus": (
                "Sanity-check transforms, label mapping, and split integrity.",
                "Use early validation signal to decide whether more epochs are worth paying for.",
            ),
        },
        ("coco-segmentation", "segmentation"): {
            "expectation": (
                "Early smoke metrics are noisy. First prove annotation loading, mask fidelity, and disjoint "
                "splits before treating IoU or Dice changes as meaningful model progress."
            ),
            "focus": (
                "Prioritize real-mask integrity and dataset adapter correctness before model tuning.",
                "Treat tiny-budget overlap metrics as directional only until the data path is stable.",
            ),
        },
        ("generic", "classification"): {
            "expectation": (
                "For ordinary classification tasks, early runs should show clear movement if the recipe is healthy. "
                "Flat metrics often point to bad splits, missing normalization, or optimizer misconfiguration."
            ),
            "focus": (
                "Check data pipeline and optimizer assumptions before rotating architecture aggressively.",
                "Prefer interventions that explain metric changes, not just more epochs.",
            ),
        },
        ("generic", "segmentation"): {
            "expectation": (
                "For segmentation tasks, early effort should validate mask provenance, split integrity, and loss wiring "
                "before interpreting overlap metrics as stable signal."
            ),
            "focus": (
                "Keep acceptance tied to explicit ground-truth masks on a held-out split.",
                "Treat data-path repairs as higher priority than architecture search until masks are trustworthy.",
            ),
        },
        ("generic", "generic"): {
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

    _METRIC_PREFERENCE: tuple[str, ...] = (
        "eval_accuracy",
        "test_accuracy",
        "accuracy",
        "val_accuracy",
        "out_evaluation_accuracy",
        "iou",
        "mean_iou",
        "dice",
        "map50_95",
        "map50",
        "f1",
        "roc_auc",
    )

    def __init__(self, db: Database) -> None:
        self.db = db
        self.task_intent_service = TaskIntentService()

    async def build_summary(
        self,
        *,
        task: dict[str, Any],
        workspace_path: Path,
        experiment_history: list[dict[str, Any]] | None = None,
        previous_verification: dict[str, Any] | None = None,
    ) -> str:
        intent = self.task_intent_service.infer_from_task(task=task, workspace_path=workspace_path)
        dataset_hint = self._detect_dataset_hint(task=task, workspace_path=workspace_path)
        requirement_metric = self._extract_required_metric(task) or intent.primary_metric_key or "not set"
        current_key, current_value = self._extract_current_metric(
            previous_verification=previous_verification,
            preferred_metric=requirement_metric if requirement_metric != "not set" else None,
        )
        pdf_scope = self._extract_pdf_scope(task)
        queries = self._build_lookup_queries(
            dataset_hint=dataset_hint,
            task_family=intent.task_family,
            metric_key=requirement_metric,
            experiment_history=experiment_history or [],
        )
        hits = await self._lookup_research_hits(queries=queries, pdf_scope=pdf_scope)
        plateau_state = self._plateau_state(experiment_history or [], preferred_metric=current_key or requirement_metric)

        lines = [
            f"- dataset_hint: {dataset_hint}",
            f"- task_family: {intent.task_family}",
            f"- primary_metric: {requirement_metric}",
        ]
        if current_key and current_value is not None:
            lines.append(f"- current_position: last seen {current_key}={current_value:.4f}")
        if plateau_state:
            lines.append(f"- recent_pattern: {plateau_state}")

        if hits:
            lines.append("- lookup_mode: pdf_fts")
            lines.append(f"- research_queries: {' | '.join(queries[:2])}")
            for index, hit in enumerate(hits[:3], start=1):
                lines.append(
                    f"- research_hit_{index}: {hit.document_path}:p{hit.page_number} -> {self._snippet(hit.text)}"
                )
            lines.extend(self._research_focus_from_hits(hits))
            return "\n".join(lines)

        fallback = self._fallback_profile(dataset_hint=dataset_hint, task_family=intent.task_family)
        lines.append("- lookup_mode: heuristic_fallback")
        lines.append(f"- baseline_expectation: {fallback['expectation']}")
        focus_items = list(fallback["focus"])
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

    def _fallback_profile(self, *, dataset_hint: str, task_family: str) -> dict[str, tuple[str, ...] | str]:
        return self._FALLBACK_BRIEFS.get(
            (dataset_hint, task_family),
            self._FALLBACK_BRIEFS.get((dataset_hint, "generic"), self._FALLBACK_BRIEFS[("generic", task_family)]),
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

    def _extract_pdf_scope(self, task: dict[str, Any]) -> list[str] | None:
        try:
            values = json.loads(task.get("pdf_scope_json") or "[]")
        except Exception:
            return None
        if not isinstance(values, list):
            return None
        cleaned = [str(item).strip() for item in values if str(item).strip()]
        return cleaned or None

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
        for key in self._METRIC_PREFERENCE:
            value = self._coerce_float(metrics.get(key))
            if value is not None:
                return key, value
        for key, value in metrics.items():
            parsed = self._coerce_float(value)
            if parsed is not None:
                return str(key), parsed
        return None, None

    def _build_lookup_queries(
        self,
        *,
        dataset_hint: str,
        task_family: str,
        metric_key: str,
        experiment_history: list[dict[str, Any]],
    ) -> list[str]:
        queries: list[str] = []
        if dataset_hint != "generic":
            queries.append(f"{dataset_hint} {task_family} {metric_key} baseline")
            queries.append(f"{dataset_hint} {metric_key} validation split")
        queries.append(f"{task_family} {metric_key} baseline")
        queries.append(f"{task_family} {metric_key} optimizer schedule")
        if self._recent_attempts_are_flat(experiment_history):
            queries.append(f"{task_family} {metric_key} plateau learning rate schedule")
        deduped: list[str] = []
        for query in queries:
            normalized = " ".join(re.sub(r"[^a-zA-Z0-9_]+", " ", query).split())
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped[:4]

    async def _lookup_research_hits(
        self,
        *,
        queries: list[str],
        pdf_scope: list[str] | None,
    ) -> list[PdfChunkRow]:
        hits: list[PdfChunkRow] = []
        seen: set[tuple[str, int]] = set()
        for query in queries:
            try:
                rows = await self.db.fts_search(query, top_k=4, pdf_scope=pdf_scope)
            except Exception:
                continue
            for row in rows:
                key = (row.document_path, row.page_number)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(row)
                if len(hits) >= 4:
                    return hits
        return hits

    def _research_focus_from_hits(self, hits: list[PdfChunkRow]) -> list[str]:
        lines: list[str] = []
        combined = " ".join(row.text.lower() for row in hits[:3])
        if "validation" in combined or "held-out" in combined or "split" in combined:
            lines.append("- next_research_focus: keep validation and held-out split handling explicit in the next recipe change")
        if "scheduler" in combined or "warmup" in combined or "cosine" in combined:
            lines.append("- secondary_focus: compare the current optimizer schedule against the retrieved scheduler guidance before changing architecture")
        elif "augmentation" in combined or "regularization" in combined:
            lines.append("- secondary_focus: check whether the next intervention should prioritize augmentation or regularization instead of more epochs")
        if not lines:
            lines.append("- next_research_focus: map retrieved baseline guidance to the current workspace recipe before broad search")
        return lines[:2]

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
                for key in self._METRIC_PREFERENCE:
                    value = self._coerce_float(metrics.get(key))
                    if value is not None:
                        break
            if value is not None:
                comparable.append(value)
        if len(comparable) < 3:
            return None
        if max(comparable) - min(comparable) <= 0.01:
            return "recent attempts are within a narrow metric band; prefer recipe changes over more identical epochs"
        if comparable[-1] > comparable[0]:
            return "recent attempts are still improving; expanding budget is defensible if recipe changes stay small"
        return None

    def _recent_attempts_are_flat(self, attempts: list[dict[str, Any]]) -> bool:
        values: list[float] = []
        for item in attempts[-4:]:
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for key in self._METRIC_PREFERENCE:
                value = self._coerce_float(metrics.get(key))
                if value is not None:
                    values.append(value)
                    break
        if len(values) < 3:
            return False
        return max(values) - min(values) <= 0.01

    @staticmethod
    def _snippet(text: str, limit: int = 220) -> str:
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit].rstrip()}..."

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

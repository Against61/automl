from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from orchestrator.execution.metric_parsing import normalize_metric_key


@dataclass(frozen=True, slots=True)
class TaskIntent:
    task_family: str
    metric_family: str
    primary_metric_key: str | None
    preferred_metrics: tuple[str, ...]
    requires_real_dataset_smoke: bool
    evidence: tuple[str, ...]

    def as_constraints(self) -> list[str]:
        constraints = [f"TASK_FAMILY: {self.task_family}"]
        if self.metric_family:
            constraints.append(f"PRIMARY_METRIC_FAMILY: {self.metric_family}")
        if self.primary_metric_key:
            constraints.append(f"PRIMARY_METRIC_KEY: {self.primary_metric_key}")
        if self.preferred_metrics:
            constraints.append(f"PREFERRED_METRICS: {', '.join(self.preferred_metrics)}")
        constraints.append(
            f"REAL_DATASET_SMOKE_REQUIRED: {'true' if self.requires_real_dataset_smoke else 'false'}"
        )
        return constraints

    def supports_metric(self, metric_key: str | None) -> bool:
        normalized = normalize_metric_key(str(metric_key or ""))
        if not normalized:
            return False
        allowed = {normalize_metric_key(item) for item in self.preferred_metrics}
        if normalized in allowed:
            return True

        family_aliases = {
            "classification": {"accuracy", "f1", "f1_macro", "roc_auc", "auc", "precision", "recall"},
            "segmentation": {"iou", "mean_iou", "miou", "dice", "dice_score", "jaccard"},
            "detection": {"map", "map50", "map50_95", "precision", "recall", "f1"},
            "regression": {"rmse", "mae", "mse", "r2", "mape"},
        }
        return normalized in family_aliases.get(self.task_family, set())


class TaskIntentService:
    _QUALITY_RE = re.compile(
        r"(?i)(?:required[_\s-]*metric|ralph_required_metric)\s*[:=]\s*(?P<metric>[a-zA-Z][a-zA-Z0-9_\s\-\./%]*?)\s*(?:>=|<=|!=|==|>|<|=)"
    )

    _TEXT_MARKERS: dict[str, tuple[str, ...]] = {
        "segmentation": (
            "segmentation",
            "segmentator",
            "segment",
            "сегмент",
            "mask",
            "masks",
            "polygon",
            "polygons",
            "iou",
            "dice",
            "semantic segmentation",
            "instance segmentation",
        ),
        "detection": (
            "detection",
            "detector",
            "detect",
            "bbox",
            "bounding box",
            "bounding boxes",
            "object detection",
            "yolo",
            "map50",
            "map50_95",
        ),
        "classification": (
            "classification",
            "classifier",
            "classify",
            "accuracy",
            "top1",
            "top-1",
            "label",
            "class names",
            "category classification",
        ),
        "regression": (
            "regression",
            "regressor",
            "rmse",
            "mae",
            "mse",
            "r2",
            "mape",
            "forecast",
            "predict value",
            "numeric target",
        ),
    }

    _WORKSPACE_HINTS: dict[str, tuple[str, ...]] = {
        "segmentation": ("segmentation", "mask", "masks", "polygons", "instances"),
        "detection": ("bbox", "boxes", "bounding", "yolo"),
        "classification": ("class", "classes", "labels"),
        "regression": ("targets.csv", "target.csv", "regression"),
    }

    _FAMILY_DEFAULTS: dict[str, tuple[str, tuple[str, ...], bool, str]] = {
        "segmentation": ("iou", ("iou", "mean_iou", "dice"), True, "overlap"),
        "detection": ("map50_95", ("map50_95", "map50", "precision", "recall"), True, "detection"),
        "classification": ("accuracy", ("accuracy", "f1", "roc_auc"), False, "classification"),
        "regression": ("rmse", ("rmse", "mae", "r2"), False, "regression"),
    }

    def infer_from_task(
        self,
        *,
        task: dict,
        workspace_path: Path | None = None,
        extra_texts: Iterable[str] | None = None,
    ) -> TaskIntent:
        goal = str(task.get("goal", ""))
        try:
            constraints = [str(item).strip() for item in json.loads(task.get("constraints_json") or "[]")]
        except Exception:
            constraints = []
        return self.infer(
            goal=goal,
            constraints=constraints,
            workspace_path=workspace_path,
            extra_texts=extra_texts,
        )

    def infer(
        self,
        *,
        goal: str,
        constraints: list[str],
        workspace_path: Path | None = None,
        extra_texts: Iterable[str] | None = None,
    ) -> TaskIntent:
        evidence: list[str] = []
        explicit_metric = self._extract_explicit_metric([goal, *constraints, *(extra_texts or [])])
        metric_family = self._family_from_metric(explicit_metric)
        if metric_family:
            evidence.append(f"explicit_metric:{explicit_metric}")

        normalized_blob = self._normalized_blob(goal, constraints, extra_texts)
        family = metric_family
        scored_family, text_hits = self._family_from_text(normalized_blob)
        if scored_family:
            if family is None or family == scored_family or len(text_hits) >= 2:
                family = scored_family
            evidence.extend(f"text:{item}" for item in text_hits[:4])

        if family is None and workspace_path is not None:
            hinted_family, workspace_hits = self._family_from_workspace(workspace_path)
            if hinted_family:
                family = hinted_family
                evidence.extend(f"workspace:{item}" for item in workspace_hits[:4])

        if family is None:
            family = "generic"

        if family in self._FAMILY_DEFAULTS:
            default_metric, preferred_metrics, requires_real_dataset_smoke, metric_family = self._FAMILY_DEFAULTS[family]
            primary_metric = explicit_metric if explicit_metric and self._family_from_metric(explicit_metric) == family else default_metric
            if explicit_metric and explicit_metric not in preferred_metrics and self._family_from_metric(explicit_metric) == family:
                preferred_metrics = (explicit_metric, *preferred_metrics)
            return TaskIntent(
                task_family=family,
                metric_family=metric_family,
                primary_metric_key=primary_metric,
                preferred_metrics=tuple(dict.fromkeys(preferred_metrics)),
                requires_real_dataset_smoke=requires_real_dataset_smoke,
                evidence=tuple(dict.fromkeys(evidence)),
            )

        preferred_metrics = (explicit_metric,) if explicit_metric else ()
        return TaskIntent(
            task_family="generic",
            metric_family="generic",
            primary_metric_key=explicit_metric,
            preferred_metrics=preferred_metrics,
            requires_real_dataset_smoke=False,
            evidence=tuple(dict.fromkeys(evidence)),
        )

    def _normalized_blob(
        self,
        goal: str,
        constraints: list[str],
        extra_texts: Iterable[str] | None,
    ) -> str:
        text_parts = [goal, *constraints]
        if extra_texts:
            text_parts.extend(str(item) for item in extra_texts if str(item).strip())
        return " ".join(" ".join(text_parts).lower().split())

    def _extract_explicit_metric(self, texts: Iterable[str]) -> str | None:
        for text in texts:
            raw = str(text or "").strip()
            if not raw:
                continue
            match = self._QUALITY_RE.search(raw)
            if not match:
                continue
            metric = normalize_metric_key(match.group("metric"))
            if metric:
                return metric
        return None

    def _family_from_metric(self, metric_key: str | None) -> str | None:
        normalized = normalize_metric_key(str(metric_key or ""))
        if not normalized:
            return None
        metric_to_family = {
            "iou": "segmentation",
            "mean_iou": "segmentation",
            "miou": "segmentation",
            "dice": "segmentation",
            "dice_score": "segmentation",
            "jaccard": "segmentation",
            "map": "detection",
            "map50": "detection",
            "map50_95": "detection",
            "accuracy": "classification",
            "f1": "classification",
            "f1_macro": "classification",
            "roc_auc": "classification",
            "auc": "classification",
            "rmse": "regression",
            "mae": "regression",
            "mse": "regression",
            "r2": "regression",
            "mape": "regression",
        }
        return metric_to_family.get(normalized)

    def _family_from_text(self, normalized_blob: str) -> tuple[str | None, list[str]]:
        best_family: str | None = None
        best_hits: list[str] = []
        best_score = 0
        for family, markers in self._TEXT_MARKERS.items():
            hits = [marker for marker in markers if marker in normalized_blob]
            if len(hits) > best_score:
                best_family = family
                best_hits = hits
                best_score = len(hits)
        return best_family, best_hits

    def _family_from_workspace(self, workspace_path: Path) -> tuple[str | None, list[str]]:
        if not workspace_path.exists():
            return None, []
        hits: dict[str, list[str]] = {family: [] for family in self._WORKSPACE_HINTS}
        try:
            candidates = workspace_path.rglob("*")
        except OSError:
            return None, []
        for index, path in enumerate(candidates):
            if index >= 300:
                break
            name = path.name.lower()
            if not name:
                continue
            for family, markers in self._WORKSPACE_HINTS.items():
                for marker in markers:
                    if marker in name and marker not in hits[family]:
                        hits[family].append(marker)
        best_family = None
        best_hits: list[str] = []
        best_score = 0
        for family, family_hits in hits.items():
            if len(family_hits) > best_score:
                best_family = family
                best_hits = family_hits
                best_score = len(family_hits)
        return best_family, best_hits

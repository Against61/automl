from __future__ import annotations

import json
import re
from pathlib import Path

from orchestrator.execution.metric_parsing import (
    extract_metrics_from_text,
    extract_numeric_metrics,
    extract_report_context_flags,
    extract_split_integrity_flags,
    looks_like_metric_key,
    normalize_metric_key,
)


class WorkspaceMetricsReader:
    def read_workspace_metrics(self, workspace_path: Path) -> dict[str, float | int | str | bool]:
        structured_metrics: dict[str, float | int | str | bool] = {}
        fallback_metrics: dict[str, float | int | str | bool] = {}
        candidates: list[Path] = [
            workspace_path / "metrics.json",
            workspace_path / "results.json",
            workspace_path / "metrics.md",
        ]
        for pattern in ("*metrics*.json", "*metrics*.md", "*metrics*.markdown"):
            for path in sorted(workspace_path.rglob(pattern)):
                if path not in candidates:
                    candidates.append(path)

        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() == ".json":
                for key, value in self.read_json_metrics(path).items():
                    structured_metrics.setdefault(key, value)
            else:
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                fallback_metrics.update(extract_metrics_from_text(text))
                fallback_metrics.update(self.extract_metrics_from_markdown_table(text))
                fallback_metrics.update(extract_split_integrity_flags(text))
                fallback_metrics.update(extract_report_context_flags(text))

        merged = dict(structured_metrics)
        for key, value in fallback_metrics.items():
            merged.setdefault(key, value)
        return merged

    def extract_metrics_from_markdown_table(self, text: str) -> dict[str, float | int | str | bool]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        metrics: dict[str, float | int | str | bool] = {}
        idx = 0
        while idx + 2 < len(lines):
            header = lines[idx]
            sep = lines[idx + 1]
            if not (header.startswith("|") and sep.startswith("|")):
                idx += 1
                continue
            if "---" not in sep:
                idx += 1
                continue

            header_cells = [cell.strip().lower().replace(" ", "_") for cell in header.strip("|").split("|")]
            data_rows: list[list[str]] = []
            row_idx = idx + 2
            while row_idx < len(lines) and lines[row_idx].startswith("|"):
                cells = [cell.strip() for cell in lines[row_idx].strip("|").split("|")]
                if len(cells) == len(header_cells):
                    data_rows.append(cells)
                row_idx += 1

            if data_rows:
                if self.looks_like_key_value_metric_table(header_cells):
                    name_idx = 0
                    value_idx = 1
                    for row in data_rows:
                        metric_name = normalize_metric_key(row[name_idx])
                        if not metric_name:
                            continue
                        parsed = self.parse_numeric_cell(row[value_idx])
                        if parsed is None:
                            continue
                        metrics[metric_name] = parsed
                else:
                    last_row = data_rows[-1]
                    for key, raw in zip(header_cells, last_row):
                        parsed = self.parse_numeric_cell(raw)
                        if parsed is None:
                            continue
                        metrics[key] = parsed

            idx = row_idx
        return metrics

    def looks_like_key_value_metric_table(self, header_cells: list[str]) -> bool:
        if len(header_cells) < 2:
            return False
        first = header_cells[0]
        second = header_cells[1]
        return first in {"metric", "name", "measure", "key"} and second in {"value", "score", "metric_value"}

    def parse_numeric_cell(self, raw: str) -> float | int | None:
        normalized = raw.replace("**", "").replace("__", "").strip()
        is_percent = "%" in normalized
        normalized = normalized.replace("%", "").strip()
        if not normalized:
            return None
        if not re.match(r"^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$", normalized):
            return None
        value = float(normalized)
        if is_percent and value > 1.0:
            value = value / 100.0
        if value.is_integer():
            return int(value)
        return value

    def read_json_metrics(self, path: Path) -> dict[str, float | int | str | bool]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return self.extract_json_metrics_payload(raw)

    def extract_json_metrics_payload(self, raw: object) -> dict[str, float | int | str | bool]:
        if not isinstance(raw, dict):
            return {}

        metrics: dict[str, float | int | str | bool] = {}
        self.collect_json_metrics_recursive(raw, metrics)
        return metrics

    def collect_json_metrics_recursive(
        self,
        raw: object,
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        if isinstance(raw, dict):
            metrics.update(extract_numeric_metrics(raw))
            self.extract_split_flags_from_mapping(raw, metrics)
            self.extract_report_flags_from_mapping(raw, metrics)
            for raw_key, value in raw.items():
                normalized_key = normalize_metric_key(str(raw_key))
                if isinstance(value, dict):
                    self.flatten_metric_mapping(normalized_key, value, metrics)
                    self.collect_json_metrics_recursive(value, metrics)
                elif isinstance(value, list):
                    for item in value:
                        self.collect_json_metrics_recursive(item, metrics)
            return
        if isinstance(raw, list):
            for item in raw:
                self.collect_json_metrics_recursive(item, metrics)

    def flatten_metric_mapping(
        self,
        parent_key: str,
        mapping: dict[str, object],
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        if not parent_key or not looks_like_metric_key(parent_key):
            return
        for child_key, child_value in mapping.items():
            normalized_child = normalize_metric_key(str(child_key))
            if not normalized_child:
                continue
            parsed = extract_numeric_metrics({f"{parent_key}_{normalized_child}": child_value})
            metrics.update(parsed)

    def extract_split_flags_from_mapping(
        self,
        mapping: dict[str, object],
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        direct_pass_keys = (
            "split_integrity_passed",
            "split_leakage_detected",
        )
        for key in direct_pass_keys:
            value = mapping.get(key)
            if isinstance(value, bool):
                metrics[key] = value
        disjoint_keys = (
            "disjoint_from_training",
            "train_eval_disjoint",
            "eval_disjoint_from_training",
            "test_disjoint_from_training",
        )
        for key in disjoint_keys:
            value = mapping.get(key)
            if isinstance(value, bool):
                metrics["split_integrity_passed"] = value
                metrics["split_leakage_detected"] = not value
                return

    def extract_report_flags_from_mapping(
        self,
        mapping: dict[str, object],
        metrics: dict[str, float | int | str | bool],
    ) -> None:
        bool_keys = (
            "planning_only_report_detected",
            "non_production_report_detected",
            "reference_evaluation_fixture_detected",
            "oracle_predictions_detected",
            "training_deferred",
            "dataset_parse_ok",
            "class_consistency_across_splits",
            "split_nonempty",
            "budget_respected",
        )
        for key in bool_keys:
            value = mapping.get(key)
            if isinstance(value, bool):
                metrics[key] = value

        string_keys = (
            "dataset_type",
            "evaluation_split",
            "task_family",
            "primary_metric_key",
            "budget_tier",
        )
        for key in string_keys:
            value = mapping.get(key)
            if isinstance(value, str) and value.strip():
                metrics[key] = value.strip()

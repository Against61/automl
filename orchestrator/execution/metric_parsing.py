from __future__ import annotations

import re
from typing import Any


_METRIC_RE = re.compile(
    r"(?P<key>[a-zA-Z][a-zA-Z0-9_ ]{1,60})\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*%?)"
)
_KEY_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_METRIC_TOKENS = {
    "acc",
    "accuracy",
    "loss",
    "precision",
    "recall",
    "f1",
    "auc",
    "iou",
    "miou",
    "dice",
    "map",
    "mae",
    "mse",
    "rmse",
    "r2",
    "bleu",
    "rouge",
    "perplexity",
    "epoch",
    "epochs",
}
_METRIC_PREFIXES = ("train_", "val_", "validation_", "test_", "eval_", "best_", "final_")
_DENY_EXACT = {
    "id",
    "session_id",
    "request_id",
    "task_id",
    "run_id",
    "step_id",
    "model",
    "provider",
    "workdir",
    "approval",
    "sandbox",
    "status",
    "port",
    "pid",
    "cf_ray",
    "duration_ms",
    "tokens_used",
}
_DENY_PREFIXES = (
    "session",
    "request",
    "task",
    "run",
    "step",
    "model",
    "provider",
    "approval",
    "sandbox",
    "reasoning",
    "workdir",
    "token",
    "duration",
    "created",
    "updated",
    "timestamp",
    "path",
    "file",
    "line",
)

_SPLIT_LEAKAGE_MARKERS = (
    "same balanced smoke subset used for the short training run",
    "same smoke subset used for the short training run",
    "same subset used for the short training run",
    "same subset used for training and evaluation",
    "evaluation split: `same ",
    "train subset (smoke test overfit check)",
    "accuracy gate applies to the training subset used for the smoke test",
    "accuracy gate applies to the training subset",
)

_SPLIT_INTEGRITY_PASS_MARKERS = (
    "data leakage check: `pass`",
    "data leakage check: pass",
    "split integrity: `pass`",
    "split integrity: pass",
    "disjoint split: `pass`",
    "disjoint split: pass",
)

_NON_PRODUCTION_REPORT_MARKERS: dict[str, tuple[str, ...]] = {
    "reference_evaluation_fixture_detected": (
        "reference_evaluation_fixture",
        "reference evaluation fixture",
    ),
    "oracle_predictions_detected": (
        "oracle predictions",
        "oracle prediction",
    ),
    "planning_only_report_detected": (
        "planning-only",
        "planning only",
    ),
}


def normalize_metric_key(raw: str) -> str:
    normalized = raw.replace("**", "").replace("__", "").strip().lower()
    normalized = _KEY_CLEAN_RE.sub("_", normalized).strip("_")
    aliases = {
        "acc1": "accuracy",
        "top1_acc": "accuracy",
        "top_1_acc": "accuracy",
        "val_acc": "val_accuracy",
        "test_acc": "test_accuracy",
        "train_acc": "train_accuracy",
        "eval_acc": "eval_accuracy",
        "evaluation_acc": "eval_accuracy",
        "held_out_accuracy": "eval_accuracy",
        "held_out_acc": "eval_accuracy",
        "held_out_eval_accuracy": "eval_accuracy",
        "held_out_test_accuracy": "test_accuracy",
        "out_evaluation_accuracy": "eval_accuracy",
        "intersection_over_union": "iou",
        "mean_intersection_over_union": "mean_iou",
        "mean_intersection_union": "mean_iou",
        "m_iou": "miou",
        "mean_iou_score": "mean_iou",
        "jaccard": "iou",
        "jaccard_index": "iou",
        "jaccard_score": "iou",
    }
    return aliases.get(normalized, normalized)


def looks_like_metric_key(raw: str) -> bool:
    key = normalize_metric_key(raw)
    if not key:
        return False
    if key in _DENY_EXACT:
        return False
    if key.startswith(_DENY_PREFIXES):
        return False
    tokens = [token for token in key.split("_") if token]
    if any(token in _METRIC_TOKENS for token in tokens):
        return True
    if any(key.startswith(prefix) for prefix in _METRIC_PREFIXES):
        suffix = key
        for prefix in _METRIC_PREFIXES:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                break
        suffix_tokens = [token for token in suffix.split("_") if token]
        if any(token in _METRIC_TOKENS for token in suffix_tokens):
            return True
    return False


def coerce_metric_value(raw: str, *, is_percent: bool) -> float | int | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    if is_percent and value > 1.0:
        value = value / 100.0
    if value.is_integer():
        return int(value)
    return value


def extract_metrics_from_text(text: str) -> dict[str, float | int | str | bool]:
    found: dict[str, float | int | str | bool] = {}
    for line in (text or "").splitlines():
        normalized_line = line.replace("**", "").replace("__", "").replace("`", "")
        for match in _METRIC_RE.finditer(normalized_line):
            key = normalize_metric_key(match.group("key"))
            if not looks_like_metric_key(key):
                continue
            value = coerce_metric_value(match.group("value"), is_percent=match.group(0).strip().endswith("%"))
            if value is None:
                continue
            found[key] = value
    return found


def extract_numeric_metrics(mapping: dict[str, Any]) -> dict[str, float | int | str | bool]:
    parsed: dict[str, float | int | str | bool] = {}
    for raw_key, raw_value in mapping.items():
        key = normalize_metric_key(str(raw_key))
        if not looks_like_metric_key(key):
            continue
        if isinstance(raw_value, bool):
            parsed[key] = raw_value
            continue
        if isinstance(raw_value, int | float):
            parsed[key] = raw_value
            continue
        if isinstance(raw_value, str):
            normalized = raw_value.strip()
            if not normalized:
                continue
            is_percent = "%" in normalized
            normalized = normalized.replace("%", "").strip()
            value = coerce_metric_value(normalized, is_percent=is_percent)
            if value is None:
                continue
            parsed[key] = value
    return parsed


def extract_split_integrity_flags(text: str) -> dict[str, float | int | str | bool]:
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return {}

    if any(marker in normalized for marker in _SPLIT_LEAKAGE_MARKERS):
        return {
            "split_leakage_detected": True,
            "split_integrity_passed": False,
        }

    if (
        "evaluation split" in normalized
        and "train subset" in normalized
        and ("smoke test" in normalized or "training subset" in normalized)
    ):
        return {
            "split_leakage_detected": True,
            "split_integrity_passed": False,
        }

    if any(marker in normalized for marker in _SPLIT_INTEGRITY_PASS_MARKERS):
        return {
            "split_leakage_detected": False,
            "split_integrity_passed": True,
        }

    return {}


def extract_report_context_flags(text: str) -> dict[str, float | int | str | bool]:
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return {}

    flags: dict[str, float | int | str | bool] = {}
    for key, markers in _NON_PRODUCTION_REPORT_MARKERS.items():
        if any(marker in normalized for marker in markers):
            flags[key] = True

    if flags:
        flags["non_production_report_detected"] = True
    return flags


def text_indicates_split_leakage(text: str) -> bool:
    flags = extract_split_integrity_flags(text)
    return bool(flags.get("split_leakage_detected") is True or flags.get("split_integrity_passed") is False)

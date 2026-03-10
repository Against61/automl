from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class PlanContractResult:
    passed: bool
    reason: str


def looks_like_smoke_requirement(stop_condition: str) -> bool:
    lowered = stop_condition.lower()
    markers = ("smoke", "validate", "verification", "test", "checks")
    return any(marker in lowered for marker in markers)


def extract_expected_paths(expected_artifacts: Iterable[str]) -> list[str]:
    allowed_exts = {
        ".py",
        ".json",
        ".md",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".png",
        ".jpg",
        ".jpeg",
        ".pt",
        ".pth",
        ".onnx",
        ".pkl",
    }
    quoted_re = re.compile(r"[`'\\\"]([^`'\"\\]+)[`'\\\"]")
    paths: list[str] = []
    seen: set[str] = set()
    for artifact in expected_artifacts:
        parts = [part.strip() for part in re.split(r"[,;\n]", artifact) if part.strip()]
        for part in parts:
            candidates = [part]
            candidates.extend(match.group(1).strip() for match in quoted_re.finditer(part))
            for candidate in candidates:
                ext = Path(candidate).suffix.lower()
                if ext not in allowed_exts:
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                paths.append(candidate)
    return paths


def missing_paths(expected_paths: list[str], workspace_path: Path) -> list[str]:
    missing: list[str] = []
    for expected in expected_paths:
        path = Path(expected)
        resolved = path if path.is_absolute() else workspace_path / path
        if not resolved.exists():
            missing.append(expected)
    return missing


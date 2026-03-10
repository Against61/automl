#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "orchestrator"


DOMAIN_FORBIDDEN_PREFIXES = (
    "orchestrator.runtime",
    "orchestrator.planning",
    "orchestrator.execution",
    "orchestrator.knowledge",
    "orchestrator.app",
)

APP_FORBIDDEN_PREFIXES = (
    "orchestrator.runtime.service",
    "orchestrator.runtime.session",
    "orchestrator.app",
)

# Transitional allowlist during incremental migration.
ALLOWLIST = {
    "orchestrator/application/use_cases/process_run_tick.py",
}


def iter_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


def has_forbidden_import(modules: list[str], forbidden_prefixes: tuple[str, ...]) -> list[str]:
    bad: list[str] = []
    for mod in modules:
        for prefix in forbidden_prefixes:
            if mod == prefix or mod.startswith(f"{prefix}."):
                bad.append(mod)
                break
    return bad


def main() -> int:
    errors: list[str] = []

    for path in sorted((ORCH / "domain").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        imports = iter_imports(path)
        bad = has_forbidden_import(imports, DOMAIN_FORBIDDEN_PREFIXES)
        for mod in bad:
            errors.append(f"{rel}: forbidden domain import -> {mod}")

    for path in sorted((ORCH / "application").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOWLIST:
            continue
        imports = iter_imports(path)
        bad = has_forbidden_import(imports, APP_FORBIDDEN_PREFIXES)
        for mod in bad:
            errors.append(f"{rel}: forbidden application import -> {mod}")

    if errors:
        print("architecture import boundary check failed:")
        for item in errors:
            print(f"- {item}")
        return 1
    print("architecture import boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

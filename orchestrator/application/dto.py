from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.persistence.schemas import RunStatus


@dataclass(slots=True)
class SubmitTaskCommand:
    payload: dict[str, Any]


@dataclass(slots=True)
class HandleControlCommand:
    payload: dict[str, Any]


@dataclass(slots=True)
class ProcessRunTickCommand:
    run_id: str


@dataclass(slots=True)
class FinalizeRunCommand:
    run_id: str
    final_status: RunStatus
    workspace_path: Path


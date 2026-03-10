from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime, timezone
from typing import Any

from orchestrator.persistence.schemas import RunStatus


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class DomainEvent:
    event_type: str
    run_id: str
    timestamp: datetime = field(default_factory=utc_now)
    payload: dict[str, Any] | None = None


@dataclass(slots=True)
class RunTransitioned(DomainEvent):
    from_status: RunStatus | None = None
    to_status: RunStatus | None = None

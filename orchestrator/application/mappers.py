from __future__ import annotations

from orchestrator.domain.run_aggregate import RunAggregate
from orchestrator.persistence.schemas import RunRecord


def to_run_aggregate(record: RunRecord) -> RunAggregate:
    return RunAggregate(run_id=record.run_id, status=record.status)


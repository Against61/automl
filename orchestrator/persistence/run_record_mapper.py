from __future__ import annotations

from datetime import datetime

from orchestrator.persistence.common import json_loads
from orchestrator.persistence.schemas import Priority, RunRecord, RunStatus


def row_to_run_record(row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        task_id=row["task_id"],
        workspace_id=row["workspace_id"],
        priority=Priority(row["priority"]),
        status=RunStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        execution_cycle=int(row["execution_cycle"] or 0),
        cycle_started_at=datetime.fromisoformat(row["cycle_started_at"]) if row["cycle_started_at"] else None,
        attempts_by_stage=json_loads(row["attempts_by_stage_json"], {}),
        next_step_index=row["next_step_index"],
        plan_json=json_loads(row["plan_json"], None),
        context_json=json_loads(row["context_json"], None),
        verification_json=json_loads(row["verification_json"], None),
        error_message=row["error_message"],
        approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
        goal_signature=row["goal_signature"],
    )

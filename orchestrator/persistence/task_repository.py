from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.persistence.common import json_dumps, utc_now_iso
from orchestrator.persistence.schemas import TaskSubmittedEvent

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert_task(
        self,
        event: TaskSubmittedEvent,
        evaluation_contract_json: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now_iso()
        async with self.db._lock:
            await self.db.conn.execute(
                """
                INSERT INTO tasks(
                    task_id,
                    workspace_id,
                    priority,
                    goal,
                    constraints_json,
                    pdf_scope_json,
                    evaluation_contract_json,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    workspace_id=excluded.workspace_id,
                    priority=excluded.priority,
                    goal=excluded.goal,
                    constraints_json=excluded.constraints_json,
                    pdf_scope_json=excluded.pdf_scope_json,
                    evaluation_contract_json=COALESCE(excluded.evaluation_contract_json, tasks.evaluation_contract_json),
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(event.task_id),
                    event.workspace_id,
                    event.priority.value,
                    event.payload.goal,
                    json_dumps(event.payload.constraints),
                    json_dumps(event.payload.pdf_scope),
                    json_dumps(evaluation_contract_json) if evaluation_contract_json else None,
                    json_dumps(event.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
            await self.db.conn.commit()

    async def set_evaluation_contract(self, task_id: str, evaluation_contract_json: dict[str, Any]) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE tasks SET evaluation_contract_json = ?, updated_at = ? WHERE task_id = ?",
                (json_dumps(evaluation_contract_json), utc_now_iso(), task_id),
            )
            await self.db.conn.commit()

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = await self.db._fetchone("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if not row:
            return None
        return dict(row)

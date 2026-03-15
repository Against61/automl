from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.persistence.common import json_dumps, utc_now_iso
from orchestrator.persistence.schemas import TaskSubmittedEvent

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def upsert_task(self, event: TaskSubmittedEvent) -> None:
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
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    workspace_id=excluded.workspace_id,
                    priority=excluded.priority,
                    goal=excluded.goal,
                    constraints_json=excluded.constraints_json,
                    pdf_scope_json=excluded.pdf_scope_json,
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
                    json_dumps(event.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
            await self.db.conn.commit()

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = await self.db._fetchone("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if not row:
            return None
        return dict(row)

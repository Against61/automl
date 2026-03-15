from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.persistence.common import utc_now_iso

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class RunHistoryRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def insert_run_step(
        self,
        run_id: str,
        step_id: str,
        step_title: str | None,
        step_index: int,
        action: str,
        command: str | None,
        status: str,
        stdout_text: str,
        stderr_text: str,
        duration_ms: int,
        created_at: str | None = None,
    ) -> None:
        ts = created_at or utc_now_iso()
        async with self.db._lock:
            await self.db.conn.execute(
                """
                INSERT INTO run_steps(
                    run_id, step_id, step_title, step_index, action, command, status, stdout_text, stderr_text, duration_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_id,
                    step_title,
                    step_index,
                    action,
                    command,
                    status,
                    stdout_text,
                    stderr_text,
                    duration_ms,
                    ts,
                ),
            )
            await self.db.conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (ts, run_id),
            )
            await self.db.conn.commit()

    async def count_executed_steps(self, run_id: str) -> int:
        row = await self.db._fetchone(
            "SELECT COUNT(*) AS c FROM run_steps WHERE run_id = ? AND status = ?",
            (run_id, "completed"),
        )
        return int(row["c"]) if row else 0

    async def count_attempted_steps(self, run_id: str) -> int:
        row = await self.db._fetchone(
            "SELECT COUNT(*) AS c FROM run_steps WHERE run_id = ?",
            (run_id,),
        )
        return int(row["c"]) if row else 0

    async def list_run_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self.db._fetchall(
            """
            SELECT step_id, step_title, step_index, action, command, status, stdout_text, stderr_text, duration_ms, created_at
            FROM run_steps
            WHERE run_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (run_id,),
        )
        return [dict(row) for row in rows]

    async def get_latest_failed_step(self, run_id: str) -> dict[str, Any] | None:
        row = await self.db._fetchone(
            """
            SELECT step_id, step_title, step_index, action, command, status, stdout_text, stderr_text, duration_ms, created_at
            FROM run_steps
            WHERE run_id = ? AND status = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (run_id, "failed"),
        )
        return dict(row) if row else None

    async def has_completed_step(self, run_id: str, step_id: str) -> bool:
        row = await self.db._fetchone(
            """
            SELECT 1 AS exists_flag
            FROM run_steps
            WHERE run_id = ? AND step_id = ? AND status = ?
            LIMIT 1
            """,
            (run_id, step_id, "completed"),
        )
        return row is not None

    async def add_artifact(self, run_id: str, kind: str, path: str) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                """
                INSERT INTO artifacts(run_id, kind, path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, kind, path, utc_now_iso()),
            )
            await self.db.conn.commit()

    async def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self.db._fetchall(
            "SELECT kind, path, created_at FROM artifacts WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        return [dict(r) for r in rows]

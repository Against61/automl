from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from orchestrator.persistence.common import utc_now_iso
from orchestrator.persistence.schemas import RunStatus

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class MaintenanceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def log_retention_stats(
        self,
        deleted_run_events: int,
        deleted_run_steps: int,
        deleted_policy_decisions: int,
    ) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                """
                INSERT INTO retention_stats(executed_at, deleted_run_events, deleted_run_steps, deleted_policy_decisions)
                VALUES (?, ?, ?, ?)
                """,
                (utc_now_iso(), deleted_run_events, deleted_run_steps, deleted_policy_decisions),
            )
            await self.db.conn.commit()

    async def run_retention(self, days: int) -> dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        async with self.db._lock:
            cur1 = await self.db.conn.execute("DELETE FROM run_events WHERE created_at < ?", (cutoff,))
            cur2 = await self.db.conn.execute("DELETE FROM run_steps WHERE created_at < ?", (cutoff,))
            cur3 = await self.db.conn.execute("DELETE FROM policy_decisions WHERE created_at < ?", (cutoff,))
            deleted_run_events = cur1.rowcount
            deleted_run_steps = cur2.rowcount
            deleted_policy_decisions = cur3.rowcount
            await self.db.conn.commit()
        await self.log_retention_stats(
            deleted_run_events=deleted_run_events,
            deleted_run_steps=deleted_run_steps,
            deleted_policy_decisions=deleted_policy_decisions,
        )
        return {
            "deleted_run_events": deleted_run_events,
            "deleted_run_steps": deleted_run_steps,
            "deleted_policy_decisions": deleted_policy_decisions,
        }

    async def list_run_directories_for_cleanup(self, days: int) -> list[str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = await self.db._fetchall(
            """
            SELECT run_id FROM runs
            WHERE updated_at < ? AND status IN (?, ?, ?)
            """,
            (cutoff, RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value),
        )
        return [str(r["run_id"]) for r in rows]

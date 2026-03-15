from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.persistence.common import utc_now_iso

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class PolicyDecisionsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def add_policy_decision(
        self,
        run_id: str,
        layer: str,
        subject: str,
        decision: str,
        reason: str,
    ) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                """
                INSERT INTO policy_decisions(run_id, layer, subject, decision, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, layer, subject, decision, reason, utc_now_iso()),
            )
            await self.db.conn.commit()

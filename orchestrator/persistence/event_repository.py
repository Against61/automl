from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite

from orchestrator.persistence.common import utc_now_iso

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class EventRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def record_stream_event(
        self,
        event_id: str,
        stream: str,
        event_type: str,
        payload_json: str,
        run_id: str | None = None,
    ) -> bool:
        async with self.db._lock:
            try:
                await self.db.conn.execute(
                    """
                    INSERT INTO run_events(event_id, stream, event_type, run_id, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, stream, event_type, run_id, payload_json, utc_now_iso()),
                )
                await self.db.conn.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

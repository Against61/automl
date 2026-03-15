from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database

logger = logging.getLogger(__name__)


class SqliteRecoveryService:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def is_database_corruption(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "malformed" in message
            or "integrity check failed" in message
            or "file is not a database" in message
        )

    async def is_database_healthy(self) -> bool:
        if not self.db.sqlite_path.exists():
            return True
        if self.db.sqlite_path.stat().st_size == 0:
            return True
        if self.db._conn is None:
            return False
        try:
            cursor = await self.db.conn.execute("PRAGMA integrity_check")
            row = await cursor.fetchone()
            await cursor.close()
            return row is not None and str(row[0]).lower() == "ok"
        except sqlite3.DatabaseError:
            return False
        except Exception:
            return False

    async def repair_database_file(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidates = [
            self.db.sqlite_path,
            self.db.sqlite_path.with_name(f"{self.db.sqlite_path.name}-wal"),
            self.db.sqlite_path.with_name(f"{self.db.sqlite_path.name}-shm"),
            self.db.sqlite_path.with_name(f"{self.db.sqlite_path.name}-journal"),
        ]
        archived: list[str] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            backup = candidate.with_name(f"{candidate.name}.{ts}.corrupt")
            try:
                shutil.move(candidate.as_posix(), backup.as_posix())
                archived.append(backup.name)
            except (OSError, PermissionError):
                continue
        if archived:
            logger.warning(
                "archived corrupted sqlite artifacts for %s: %s",
                self.db.sqlite_path,
                ", ".join(archived),
            )

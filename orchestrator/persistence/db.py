from __future__ import annotations

import asyncio
import shutil
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import aiosqlite

from orchestrator.persistence.schemas import Priority, RunRecord, RunStatus, TaskSubmittedEvent


logger = logging.getLogger(__name__)


class _RecoveredConnection:
    """Proxy over aiosqlite connection with DB-repair retry on execute paths."""

    def __init__(self, database: "Database") -> None:
        self._database = database

    async def execute(self, sql: str, parameters: Sequence[Any] = ()) -> Any:
        return await self._database._execute_with_repair(sql, parameters, method="execute")

    async def executemany(self, sql: str, parameters: Iterable[Sequence[Any]]) -> Any:
        return await self._database._executemany_with_repair(sql, parameters, method="executemany")

    async def executescript(self, script: str) -> Any:
        return await self._database._executescript_with_repair(script)

    async def commit(self) -> None:
        return await self._database._commit_with_repair()

    async def rollback(self) -> None:
        return await self._database._rollback_with_repair()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._database._conn, name)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


def normalize_verification_payload(
    current_payload: dict[str, Any],
    previous_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(current_payload or {})
    history: list[dict[str, Any]] = []
    if previous_payload:
        previous_history = previous_payload.get("history")
        if isinstance(previous_history, list):
            history = [entry for entry in previous_history if isinstance(entry, dict)]
        legacy_entry = {
            key: value
            for key, value in previous_payload.items()
            if key not in {"history", "attempt", "latest_attempt"}
        }
        if legacy_entry:
            history.append(legacy_entry)

    previous_attempt = 0
    if previous_payload:
        raw_attempt = previous_payload.get("attempt")
        if raw_attempt is not None:
            try:
                previous_attempt = int(raw_attempt)
            except (TypeError, ValueError):
                previous_attempt = 0
        elif previous_payload.get("latest_attempt") is not None:
            try:
                previous_attempt = int(previous_payload.get("latest_attempt"))
            except (TypeError, ValueError):
                previous_attempt = 0
    attempt = max(len(history) + 1, previous_attempt + 1)
    normalized["attempt"] = attempt
    normalized["latest_attempt"] = attempt
    normalized["history"] = history
    return normalized


@dataclass(slots=True)
class PdfChunkRow:
    chunk_id: int
    document_path: str
    page_number: int
    chunk_index: int
    text: str
    embedding: list[float] | None


class Database:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        await self._connect(allow_repair=True)

    async def _connect(self, allow_repair: bool) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = await aiosqlite.connect(self.sqlite_path.as_posix())
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._conn.execute("PRAGMA synchronous=NORMAL")

            healthy = await self._is_database_healthy()
            if not healthy:
                raise sqlite3.DatabaseError("database integrity check failed")

            await self.initialize_schema()
        except sqlite3.DatabaseError as exc:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None

            if allow_repair and self._is_database_corruption(exc):
                await self._repair_database_file()
                await self._connect(allow_repair=False)
                return
            raise
        except Exception:
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
            raise

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @staticmethod
    def _is_database_corruption(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "malformed" in message
            or "integrity check failed" in message
            or "file is not a database" in message
        )

    async def _with_repair(self, operation: str, fn):
        try:
            return await fn()
        except sqlite3.DatabaseError as exc:
            if not self._is_database_corruption(exc):
                raise
            logger.warning(
                "database corruption detected during %s on %s; repairing SQLite file",
                operation,
                self.sqlite_path,
            )
            if self._conn is not None:
                await self._conn.close()
                self._conn = None
            await self._repair_database_file()
            await self._connect(allow_repair=False)
            return await fn()

    async def _execute_with_repair(
        self,
        sql: str,
        parameters: Sequence[Any] = (),
        *,
        method: str = "execute",
    ):
        return await self._with_repair(method, lambda: self._conn.execute(sql, parameters))

    async def _executemany_with_repair(
        self,
        sql: str,
        parameters: Iterable[Sequence[Any]],
        *,
        method: str = "executemany",
    ):
        return await self._with_repair(method, lambda: self._conn.executemany(sql, parameters))

    async def _executescript_with_repair(self, script: str):
        return await self._with_repair("executescript", lambda: self._conn.executescript(script))

    async def _commit_with_repair(self) -> None:
        return await self._with_repair("commit", lambda: self._conn.commit())

    async def _rollback_with_repair(self) -> None:
        return await self._with_repair("rollback", lambda: self._conn.rollback())

    async def _is_database_healthy(self) -> bool:
        if not self.sqlite_path.exists():
            return True
        if self.sqlite_path.stat().st_size == 0:
            return True
        if self._conn is None:
            return False
        try:
            cursor = await self.conn.execute("PRAGMA integrity_check")
            row = await cursor.fetchone()
            await cursor.close()
            return row is not None and str(row[0]).lower() == "ok"
        except sqlite3.DatabaseError:
            return False
        except Exception:
            return False

    async def _repair_database_file(self) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidates = [
            self.sqlite_path,
            self.sqlite_path.with_name(f"{self.sqlite_path.name}-wal"),
            self.sqlite_path.with_name(f"{self.sqlite_path.name}-shm"),
            self.sqlite_path.with_name(f"{self.sqlite_path.name}-journal"),
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
                self.sqlite_path,
                ", ".join(archived),
            )

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("database is not connected")
        return _RecoveredConnection(self)  # type: ignore[return-value]

    async def initialize_schema(self) -> None:
        async with self._lock:
            await self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    constraints_json TEXT NOT NULL,
                    pdf_scope_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    goal_signature TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    attempts_by_stage_json TEXT NOT NULL DEFAULT '{}',
                    next_step_index INTEGER NOT NULL DEFAULT 0,
                    plan_json TEXT,
                    context_json TEXT,
                    verification_json TEXT,
                    error_message TEXT,
                    approved_at TEXT,
                    cancelled_at TEXT,
                    cancelled_reason TEXT,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_status_priority_created
                    ON runs(status, priority, created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_workspace_status
                    ON runs(workspace_id, status);

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    stream TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    run_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);
                CREATE INDEX IF NOT EXISTS idx_run_events_created_at ON run_events(created_at);

                CREATE TABLE IF NOT EXISTS run_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    step_title TEXT,
                    step_index INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    command TEXT,
                    status TEXT NOT NULL,
                    stdout_text TEXT,
                    stderr_text TEXT,
                    duration_ms INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_run_steps_run_id ON run_steps(run_id);
                CREATE INDEX IF NOT EXISTS idx_run_steps_created_at ON run_steps(created_at);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);

                CREATE TABLE IF NOT EXISTS workspace_locks (
                    workspace_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    locked_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS policy_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_policy_decisions_run_id ON policy_decisions(run_id);
                CREATE INDEX IF NOT EXISTS idx_policy_decisions_created_at ON policy_decisions(created_at);

                CREATE TABLE IF NOT EXISTS pdf_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    content_hash TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    page_count INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pdf_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    page_number INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES pdf_documents(id) ON DELETE CASCADE
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_pdf_chunks_doc_idx
                    ON pdf_chunks(document_id, chunk_index);

                CREATE TABLE IF NOT EXISTS pdf_embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    vector_json TEXT NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES pdf_chunks(id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS pdf_chunks_fts
                    USING fts5(chunk_id UNINDEXED, path UNINDEXED, text);

                CREATE TABLE IF NOT EXISTS retention_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    executed_at TEXT NOT NULL,
                    deleted_run_events INTEGER NOT NULL,
                    deleted_run_steps INTEGER NOT NULL,
                    deleted_policy_decisions INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id TEXT NOT NULL,
                    goal_signature TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    run_attempt INTEGER NOT NULL,
                    verification_status TEXT,
                    quality_status TEXT,
                    quality_reason TEXT,
                    metrics_json TEXT NOT NULL,
                    hyperparameters_json TEXT NOT NULL,
                    strategy_json TEXT,
                    skill_paths_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, run_attempt)
                );

                CREATE INDEX IF NOT EXISTS idx_experiment_attempts_lookup
                    ON experiment_attempts(workspace_id, goal_signature, created_at);
                CREATE INDEX IF NOT EXISTS idx_experiment_attempts_run
                    ON experiment_attempts(run_id, created_at);
                """
            )
            await self._ensure_runs_goal_signature_column()
            await self._ensure_run_steps_step_title_column()
            await self.conn.commit()

    async def rehydrate_terminal_runs_from_artifacts(self, runs_root: Path) -> dict[str, int]:
        stats = {
            "restored_tasks": 0,
            "restored_runs": 0,
            "restored_steps": 0,
            "restored_artifacts": 0,
            "skipped_existing_runs": 0,
            "skipped_invalid_files": 0,
        }
        if not runs_root.exists():
            return stats

        result_files = sorted(runs_root.glob("*/result.json"))
        if not result_files:
            return stats

        async with self._lock:
            for result_path in result_files:
                payload = self._load_result_artifact_payload(result_path)
                if payload is None:
                    stats["skipped_invalid_files"] += 1
                    continue

                run_id = str(payload.get("run_id") or "").strip()
                task_id = str(payload.get("task_id") or "").strip()
                status_value = str(payload.get("status") or "").strip().upper()
                if not run_id or not task_id or status_value not in RunStatus.__members__:
                    stats["skipped_invalid_files"] += 1
                    continue
                status = RunStatus(status_value)
                if status not in RunStatus.terminal():
                    stats["skipped_invalid_files"] += 1
                    continue

                existing = await self._fetchone("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
                if existing:
                    stats["skipped_existing_runs"] += 1
                    continue

                recovery = payload.get("recovery") if isinstance(payload.get("recovery"), dict) else {}
                task_meta = recovery.get("task") if isinstance(recovery.get("task"), dict) else {}
                workspace_id = str(
                    recovery.get("workspace_id")
                    or task_meta.get("workspace_id")
                    or "__recovered__"
                ).strip() or "__recovered__"
                priority_raw = str(
                    recovery.get("priority")
                    or task_meta.get("priority")
                    or Priority.normal.value
                ).strip().lower()
                priority = Priority(priority_raw) if priority_raw in Priority._value2member_map_ else Priority.normal
                created_at = self._coerce_iso_timestamp(
                    recovery.get("created_at"),
                    fallback_path=result_path,
                )
                updated_at = self._coerce_iso_timestamp(
                    recovery.get("updated_at"),
                    fallback_path=result_path,
                )

                task_goal = str(task_meta.get("goal") or "Recovered task from run artifact").strip()
                constraints = task_meta.get("constraints") if isinstance(task_meta.get("constraints"), list) else []
                pdf_scope = task_meta.get("pdf_scope") if isinstance(task_meta.get("pdf_scope"), list) else []
                payload_json = task_meta.get("payload_json") if isinstance(task_meta.get("payload_json"), dict) else {
                    "task_id": task_id,
                    "workspace_id": workspace_id,
                    "payload": {
                        "goal": task_goal,
                        "constraints": constraints,
                        "pdf_scope": pdf_scope,
                        "execution_mode": "plan_execute",
                    },
                }

                await self.conn.execute(
                    """
                    INSERT OR IGNORE INTO tasks(
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
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        workspace_id,
                        priority.value,
                        task_goal,
                        _json_dumps(constraints),
                        _json_dumps(pdf_scope),
                        _json_dumps(payload_json),
                        self._coerce_iso_timestamp(task_meta.get("created_at"), fallback_path=result_path),
                        self._coerce_iso_timestamp(task_meta.get("updated_at"), fallback_path=result_path),
                    ),
                )
                stats["restored_tasks"] += 1

                await self.conn.execute(
                    """
                    INSERT INTO runs(
                        run_id,
                        task_id,
                        workspace_id,
                        priority,
                        status,
                        goal_signature,
                        created_at,
                        updated_at,
                        attempts_by_stage_json,
                        next_step_index,
                        plan_json,
                        context_json,
                        verification_json,
                        error_message,
                        approved_at,
                        cancelled_at,
                        cancelled_reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        task_id,
                        workspace_id,
                        priority.value,
                        status.value,
                        None,
                        created_at,
                        updated_at,
                        "{}",
                        0,
                        _json_dumps(recovery.get("plan_json")) if recovery.get("plan_json") is not None else None,
                        _json_dumps(recovery.get("context_json")) if recovery.get("context_json") is not None else None,
                        _json_dumps(recovery.get("verification_json")) if recovery.get("verification_json") is not None else None,
                        recovery.get("error_message"),
                        None,
                        updated_at if status == RunStatus.CANCELLED else None,
                        recovery.get("error_message") if status == RunStatus.CANCELLED else None,
                    ),
                )
                stats["restored_runs"] += 1

                for step in payload.get("steps") or []:
                    if not isinstance(step, dict):
                        continue
                    await self.conn.execute(
                        """
                        INSERT INTO run_steps(
                            run_id, step_id, step_title, step_index, action, command, status, stdout_text, stderr_text, duration_ms, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            str(step.get("step_id") or "recovered-step"),
                            str(step.get("step_title") or "") or None,
                            int(step.get("step_index") or 0),
                            str(step.get("action") or "read"),
                            step.get("command"),
                            str(step.get("status") or "completed"),
                            str(step.get("stdout_text") or ""),
                            str(step.get("stderr_text") or ""),
                            int(step.get("duration_ms") or 0),
                            self._coerce_iso_timestamp(step.get("created_at"), fallback_path=result_path),
                        ),
                    )
                    stats["restored_steps"] += 1

                artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
                for artifact in artifacts:
                    if not isinstance(artifact, dict):
                        continue
                    path = str(artifact.get("path") or "").strip()
                    kind = str(artifact.get("kind") or "").strip() or "generic"
                    if not path:
                        continue
                    await self.conn.execute(
                        """
                        INSERT INTO artifacts(run_id, kind, path, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            kind,
                            path,
                            self._coerce_iso_timestamp(artifact.get("created_at"), fallback_path=result_path),
                        ),
                    )
                    stats["restored_artifacts"] += 1

                packaged = recovery.get("packaged_artifacts") if isinstance(recovery.get("packaged_artifacts"), dict) else {}
                for kind, key in (("patch_bundle", "patch_bundle_path"), ("report_json", "report_json_path")):
                    path = str(packaged.get(key) or "").strip()
                    if not path:
                        continue
                    await self.conn.execute(
                        """
                        INSERT INTO artifacts(run_id, kind, path, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run_id, kind, path, updated_at),
                    )
                    stats["restored_artifacts"] += 1

            await self.conn.commit()
        return stats

    def _load_result_artifact_payload(self, result_path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def _coerce_iso_timestamp(self, raw: Any, *, fallback_path: Path) -> str:
        if isinstance(raw, str):
            try:
                datetime.fromisoformat(raw)
                return raw
            except ValueError:
                pass
        try:
            return datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            return _utc_now_iso()

    async def _fetchone(self, query: str, params: Sequence[Any] = ()) -> aiosqlite.Row | None:
        cursor = await self.conn.execute(query, params)
        try:
            return await cursor.fetchone()
        finally:
            await cursor.close()

    async def _fetchall(self, query: str, params: Sequence[Any] = ()) -> list[aiosqlite.Row]:
        cursor = await self.conn.execute(query, params)
        try:
            return await cursor.fetchall()
        finally:
            await cursor.close()

    async def _ensure_runs_goal_signature_column(self) -> None:
        cursor = await self.conn.execute("PRAGMA table_info(runs)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "goal_signature" in columns:
            return
        await self.conn.execute("ALTER TABLE runs ADD COLUMN goal_signature TEXT")

    async def _ensure_run_steps_step_title_column(self) -> None:
        cursor = await self.conn.execute("PRAGMA table_info(run_steps)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "step_title" in columns:
            return
        await self.conn.execute("ALTER TABLE run_steps ADD COLUMN step_title TEXT")

    async def record_stream_event(
        self,
        event_id: str,
        stream: str,
        event_type: str,
        payload_json: str,
        run_id: str | None = None,
    ) -> bool:
        async with self._lock:
            try:
                await self.conn.execute(
                    """
                    INSERT INTO run_events(event_id, stream, event_type, run_id, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (event_id, stream, event_type, run_id, payload_json, _utc_now_iso()),
                )
                await self.conn.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def upsert_task(self, event: TaskSubmittedEvent) -> None:
        now = _utc_now_iso()
        async with self._lock:
            await self.conn.execute(
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
                    _json_dumps(event.payload.constraints),
                    _json_dumps(event.payload.pdf_scope),
                    _json_dumps(event.model_dump(mode="json")),
                    now,
                    now,
                ),
            )
            await self.conn.commit()

    async def create_or_get_run(
        self,
        task_id: str,
        workspace_id: str,
        priority: Priority,
        goal_signature: str | None = None,
    ) -> str:
        async with self._lock:
            row = await self._fetchone(
                """
                SELECT run_id, status, goal_signature
                FROM runs
                WHERE task_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (task_id,),
            )
            if row:
                row_signature = row["goal_signature"]
                status = RunStatus(row["status"])
                if row_signature == goal_signature and status not in RunStatus.terminal():
                    return str(row["run_id"])
            run_id = str(uuid4())
            now = _utc_now_iso()
            await self.conn.execute(
                """
                INSERT INTO runs(
                    run_id,
                    task_id,
                    workspace_id,
                    priority,
                    status,
                    goal_signature,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    workspace_id,
                    priority.value,
                    RunStatus.RECEIVED.value,
                    goal_signature,
                    now,
                    now,
                ),
            )
            await self.conn.commit()
            return run_id

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        row = await self._fetchone("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        if not row:
            return None
        task = dict(row)
        return task

    async def get_run(self, run_id: str) -> RunRecord | None:
        row = await self._fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not row:
            return None
        return RunRecord(
            run_id=row["run_id"],
            task_id=row["task_id"],
            workspace_id=row["workspace_id"],
            priority=Priority(row["priority"]),
            status=RunStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            attempts_by_stage=_json_loads(row["attempts_by_stage_json"], {}),
            next_step_index=row["next_step_index"],
            plan_json=_json_loads(row["plan_json"], None),
            context_json=_json_loads(row["context_json"], None),
            verification_json=_json_loads(row["verification_json"], None),
            error_message=row["error_message"],
            approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
            goal_signature=row["goal_signature"],
        )

    async def list_runnable_runs(self, limit: int = 20) -> list[str]:
        rows = await self._fetchall(
            """
            SELECT run_id FROM runs
            WHERE status IN (?, ?, ?, ?, ?, ?)
            ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END ASC, created_at ASC
            LIMIT ?
            """,
            (
                RunStatus.RECEIVED.value,
                RunStatus.CONTEXT_READY.value,
                RunStatus.PLAN_READY.value,
                RunStatus.EXECUTING.value,
                RunStatus.VERIFYING.value,
                RunStatus.PACKAGING.value,
                limit,
            ),
        )
        return [str(row["run_id"]) for row in rows]

    async def list_nonterminal_runs(self) -> list[str]:
        rows = await self._fetchall(
            """
            SELECT run_id FROM runs
            WHERE status NOT IN (?, ?, ?)
            """,
            (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value),
        )
        return [str(row["run_id"]) for row in rows]

    async def acquire_workspace_lock(self, workspace_id: str, run_id: str) -> bool:
        async with self._lock:
            row = await self._fetchone(
                "SELECT run_id FROM workspace_locks WHERE workspace_id = ?",
                (workspace_id,),
            )
            if row and row["run_id"] != run_id:
                locked_run = await self.get_run(str(row["run_id"]))
                if not locked_run or locked_run.status in RunStatus.terminal():
                    await self.conn.execute(
                        "DELETE FROM workspace_locks WHERE workspace_id = ?",
                        (workspace_id,),
                    )
                    await self.conn.execute(
                        """
                        INSERT INTO workspace_locks(workspace_id, run_id, locked_at)
                        VALUES (?, ?, ?)
                        """,
                        (workspace_id, run_id, _utc_now_iso()),
                    )
                    await self.conn.commit()
                    return True
                return False
            if row and row["run_id"] == run_id:
                return True
            await self.conn.execute(
                """
                INSERT INTO workspace_locks(workspace_id, run_id, locked_at)
                VALUES (?, ?, ?)
                """,
                (workspace_id, run_id, _utc_now_iso()),
            )
            await self.conn.commit()
            return True

    async def release_workspace_lock(self, workspace_id: str, run_id: str) -> None:
        async with self._lock:
            await self.conn.execute(
                "DELETE FROM workspace_locks WHERE workspace_id = ? AND run_id = ?",
                (workspace_id, run_id),
            )
            await self.conn.commit()

    async def update_run_status(self, run_id: str, status: RunStatus, error_message: str | None = None) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE runs
                SET status = ?, error_message = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (status.value, error_message, _utc_now_iso(), run_id),
            )
            await self.conn.commit()

    async def set_context(self, run_id: str, context: list[dict[str, Any]]) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE runs SET context_json = ?, updated_at = ? WHERE run_id = ?",
                (_json_dumps(context), _utc_now_iso(), run_id),
            )
            await self.conn.commit()

    async def set_plan(self, run_id: str, plan: dict[str, Any]) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE runs
                SET plan_json = ?,
                    -- A newly accepted plan must always start from its first step.
                    -- Keeping a stale index can skip initial steps after replans.
                    next_step_index = 0,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (_json_dumps(plan), _utc_now_iso(), run_id),
            )
            await self.conn.commit()

    async def set_goal_signature(self, run_id: str, goal_signature: str) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE runs SET goal_signature = ?, updated_at = ? WHERE run_id = ?",
                (goal_signature, _utc_now_iso(), run_id),
            )
            await self.conn.commit()

    async def increment_stage_attempt(self, run_id: str, stage: str) -> int:
        run = await self.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        attempts = dict(run.attempts_by_stage)
        current = int(attempts.get(stage, 0)) + 1
        attempts[stage] = current
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE runs
                SET attempts_by_stage_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (_json_dumps(attempts), _utc_now_iso(), run_id),
            )
            await self.conn.commit()
        return current

    async def set_next_step_index(self, run_id: str, next_index: int) -> None:
        async with self._lock:
            await self.conn.execute(
                "UPDATE runs SET next_step_index = ?, updated_at = ? WHERE run_id = ?",
                (next_index, _utc_now_iso(), run_id),
            )
            await self.conn.commit()

    async def set_verification(self, run_id: str, payload: dict[str, Any]) -> None:
        row = await self._fetchone("SELECT verification_json FROM runs WHERE run_id = ?", (run_id,))
        previous_raw = row["verification_json"] if row else None
        previous_payload = _json_loads(previous_raw, None)
        if not isinstance(previous_payload, dict):
            previous_payload = None

        async with self._lock:
            await self.conn.execute(
                "UPDATE runs SET verification_json = ?, updated_at = ? WHERE run_id = ?",
                (
                    _json_dumps(normalize_verification_payload(payload, previous_payload)),
                    _utc_now_iso(),
                    run_id,
                ),
            )
            await self.conn.commit()

    async def set_approved(self, run_id: str) -> bool:
        run = await self.get_run(run_id)
        if not run:
            return False
        if run.status not in {RunStatus.WAITING_APPROVAL, RunStatus.WAITING_PLAN_REVIEW}:
            return False
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE runs
                SET approved_at = ?, status = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (_utc_now_iso(), RunStatus.PLAN_READY.value, _utc_now_iso(), run_id),
            )
            await self.conn.commit()
        return True

    async def cancel_run(self, run_id: str, reason: str | None = None) -> bool:
        run = await self.get_run(run_id)
        if not run:
            return False
        if run.status in RunStatus.terminal():
            return False
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE runs
                SET status = ?, cancelled_reason = ?, cancelled_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (RunStatus.CANCELLED.value, reason, _utc_now_iso(), _utc_now_iso(), run_id),
            )
            await self.conn.commit()
        return True

    async def reset_run_for_retry(self, run_id: str) -> bool:
        run = await self.get_run(run_id)
        if not run or run.status != RunStatus.FAILED:
            return False
        async with self._lock:
            await self.conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    error_message = NULL,
                    next_step_index = 0,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (RunStatus.EXECUTING.value, _utc_now_iso(), run_id),
            )
            await self.conn.commit()
        return True

    async def reset_run_for_goal_change(self, run_id: str, goal_signature: str | None = None) -> bool:
        run = await self.get_run(run_id)
        if not run:
            return False
        if run.status in RunStatus.terminal():
            return False
        async with self._lock:
            await self.conn.execute(
                "DELETE FROM run_steps WHERE run_id = ?",
                (run_id,),
            )
            await self.conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    error_message = NULL,
                    goal_signature = COALESCE(?, goal_signature),
                    context_json = NULL,
                    plan_json = NULL,
                    verification_json = NULL,
                    next_step_index = 0,
                    attempts_by_stage_json = '{}',
                    approved_at = NULL,
                    cancelled_reason = NULL,
                    cancelled_at = NULL,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (RunStatus.RECEIVED.value, goal_signature, _utc_now_iso(), run_id),
            )
            await self.conn.commit()
        return True

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
        ts = created_at or _utc_now_iso()
        async with self._lock:
            await self.conn.execute(
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
            await self.conn.execute(
                "UPDATE runs SET updated_at = ? WHERE run_id = ?",
                (ts, run_id),
            )
            await self.conn.commit()

    async def count_executed_steps(self, run_id: str) -> int:
        row = await self._fetchone(
            "SELECT COUNT(*) AS c FROM run_steps WHERE run_id = ? AND status = ?",
            (run_id, "completed"),
        )
        return int(row["c"]) if row else 0

    async def count_attempted_steps(self, run_id: str) -> int:
        row = await self._fetchone(
            "SELECT COUNT(*) AS c FROM run_steps WHERE run_id = ?",
            (run_id,),
        )
        return int(row["c"]) if row else 0

    async def list_run_steps(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            """
            SELECT step_id, step_title, step_index, action, command, status, stdout_text, stderr_text, duration_ms, created_at
            FROM run_steps
            WHERE run_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (run_id,),
        )
        return [dict(row) for row in rows]

    async def record_experiment_attempt(
        self,
        *,
        workspace_id: str,
        goal_signature: str,
        run_id: str,
        task_id: str,
        run_attempt: int,
        verification_status: str | None,
        quality_status: str | None,
        quality_reason: str | None,
        metrics: dict[str, Any],
        hyperparameters: dict[str, Any],
        strategy: dict[str, Any] | None,
        skill_paths: list[str],
        created_at: str | None = None,
    ) -> None:
        ts = created_at or _utc_now_iso()
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO experiment_attempts(
                    workspace_id,
                    goal_signature,
                    run_id,
                    task_id,
                    run_attempt,
                    verification_status,
                    quality_status,
                    quality_reason,
                    metrics_json,
                    hyperparameters_json,
                    strategy_json,
                    skill_paths_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, run_attempt) DO UPDATE SET
                    verification_status=excluded.verification_status,
                    quality_status=excluded.quality_status,
                    quality_reason=excluded.quality_reason,
                    metrics_json=excluded.metrics_json,
                    hyperparameters_json=excluded.hyperparameters_json,
                    strategy_json=excluded.strategy_json,
                    skill_paths_json=excluded.skill_paths_json,
                    created_at=excluded.created_at
                """,
                (
                    workspace_id,
                    goal_signature,
                    run_id,
                    task_id,
                    int(run_attempt),
                    verification_status,
                    quality_status,
                    quality_reason,
                    _json_dumps(metrics or {}),
                    _json_dumps(hyperparameters or {}),
                    _json_dumps(strategy) if strategy is not None else None,
                    _json_dumps(skill_paths or []),
                    ts,
                ),
            )
            await self.conn.commit()

    async def list_experiment_attempts(
        self,
        *,
        workspace_id: str,
        goal_signature: str,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            """
            SELECT
                workspace_id,
                goal_signature,
                run_id,
                task_id,
                run_attempt,
                verification_status,
                quality_status,
                quality_reason,
                metrics_json,
                hyperparameters_json,
                strategy_json,
                skill_paths_json,
                created_at
            FROM experiment_attempts
            WHERE workspace_id = ? AND goal_signature = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (workspace_id, goal_signature, int(limit)),
        )
        attempts: list[dict[str, Any]] = []
        for row in reversed(rows):
            attempts.append(
                {
                    "workspace_id": row["workspace_id"],
                    "goal_signature": row["goal_signature"],
                    "run_id": row["run_id"],
                    "task_id": row["task_id"],
                    "attempt": int(row["run_attempt"]),
                    "verification_status": row["verification_status"],
                    "quality_status": row["quality_status"],
                    "quality_reason": row["quality_reason"],
                    "metrics": _json_loads(row["metrics_json"], {}),
                    "hyperparameters": _json_loads(row["hyperparameters_json"], {}),
                    "strategy": _json_loads(row["strategy_json"], None),
                    "skill_paths": _json_loads(row["skill_paths_json"], []),
                    "created_at": row["created_at"],
                }
            )
        return attempts

    async def get_latest_failed_step(self, run_id: str) -> dict[str, Any] | None:
        row = await self._fetchone(
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
        row = await self._fetchone(
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
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO artifacts(run_id, kind, path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, kind, path, _utc_now_iso()),
            )
            await self.conn.commit()

    async def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            "SELECT kind, path, created_at FROM artifacts WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        )
        return [dict(r) for r in rows]

    async def add_policy_decision(
        self,
        run_id: str,
        layer: str,
        subject: str,
        decision: str,
        reason: str,
    ) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO policy_decisions(run_id, layer, subject, decision, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, layer, subject, decision, reason, _utc_now_iso()),
            )
            await self.conn.commit()

    async def upsert_pdf_document(
        self,
        path: str,
        content_hash: str,
        mtime: float,
        page_count: int,
    ) -> tuple[int, bool]:
        async with self._lock:
            row = await self._fetchone(
                "SELECT id, content_hash FROM pdf_documents WHERE path = ?",
                (path,),
            )
            now = _utc_now_iso()
            if row is None:
                cursor = await self.conn.execute(
                    """
                    INSERT INTO pdf_documents(path, content_hash, mtime, page_count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (path, content_hash, mtime, page_count, now),
                )
                await self.conn.commit()
                return int(cursor.lastrowid), True
            changed = row["content_hash"] != content_hash
            await self.conn.execute(
                """
                UPDATE pdf_documents
                SET content_hash = ?, mtime = ?, page_count = ?, updated_at = ?
                WHERE path = ?
                """,
                (content_hash, mtime, page_count, now, path),
            )
            await self.conn.commit()
            return int(row["id"]), changed

    async def remove_pdf_document(self, path: str) -> None:
        async with self._lock:
            row = await self._fetchone("SELECT id FROM pdf_documents WHERE path = ?", (path,))
            if not row:
                return
            doc_id = int(row["id"])
            await self.conn.execute("DELETE FROM pdf_chunks_fts WHERE path = ?", (path,))
            await self.conn.execute("DELETE FROM pdf_chunks WHERE document_id = ?", (doc_id,))
            await self.conn.execute("DELETE FROM pdf_documents WHERE id = ?", (doc_id,))
            await self.conn.commit()

    async def replace_pdf_chunks(
        self,
        document_id: int,
        path: str,
        chunks: Iterable[tuple[int, int, str]],
    ) -> list[int]:
        async with self._lock:
            existing = await self._fetchall(
                "SELECT id FROM pdf_chunks WHERE document_id = ?",
                (document_id,),
            )
            existing_ids = [int(r["id"]) for r in existing]
            if existing_ids:
                await self.conn.executemany("DELETE FROM pdf_embeddings WHERE chunk_id = ?", [(cid,) for cid in existing_ids])
            await self.conn.execute("DELETE FROM pdf_chunks_fts WHERE path = ?", (path,))
            await self.conn.execute("DELETE FROM pdf_chunks WHERE document_id = ?", (document_id,))
            created_ids: list[int] = []
            for chunk_index, page_number, text in chunks:
                cursor = await self.conn.execute(
                    """
                    INSERT INTO pdf_chunks(document_id, chunk_index, page_number, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, chunk_index, page_number, text),
                )
                chunk_id = int(cursor.lastrowid)
                created_ids.append(chunk_id)
                await self.conn.execute(
                    "INSERT INTO pdf_chunks_fts(chunk_id, path, text) VALUES (?, ?, ?)",
                    (chunk_id, path, text),
                )
            await self.conn.commit()
            return created_ids

    async def set_chunk_embeddings(self, vectors: Iterable[tuple[int, list[float]]]) -> None:
        async with self._lock:
            for chunk_id, vector in vectors:
                await self.conn.execute(
                    """
                    INSERT INTO pdf_embeddings(chunk_id, vector_json)
                    VALUES (?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET vector_json = excluded.vector_json
                    """,
                    (chunk_id, _json_dumps(vector)),
                )
            await self.conn.commit()

    async def list_known_pdf_paths(self) -> list[str]:
        rows = await self._fetchall("SELECT path FROM pdf_documents")
        return [str(r["path"]) for r in rows]

    async def fts_search(self, query: str, top_k: int, pdf_scope: list[str] | None = None) -> list[PdfChunkRow]:
        if not query.strip():
            return []
        params: list[Any] = [query]
        sql = """
            SELECT c.id AS chunk_id, d.path AS document_path, c.page_number, c.chunk_index, c.text, e.vector_json
            FROM pdf_chunks_fts f
            JOIN pdf_chunks c ON c.id = CAST(f.chunk_id AS INTEGER)
            JOIN pdf_documents d ON d.id = c.document_id
            LEFT JOIN pdf_embeddings e ON e.chunk_id = c.id
            WHERE pdf_chunks_fts MATCH ?
        """
        if pdf_scope:
            placeholders = ",".join("?" for _ in pdf_scope)
            sql += f" AND d.path IN ({placeholders})"
            params.extend(pdf_scope)
        sql += " ORDER BY bm25(pdf_chunks_fts) LIMIT ?"
        params.append(top_k)
        rows = await self._fetchall(sql, params)
        return [
            PdfChunkRow(
                chunk_id=int(row["chunk_id"]),
                document_path=str(row["document_path"]),
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                embedding=_json_loads(row["vector_json"], None),
            )
            for row in rows
        ]

    async def vector_candidates(self, pdf_scope: list[str] | None = None) -> list[PdfChunkRow]:
        params: list[Any] = []
        sql = """
            SELECT c.id AS chunk_id, d.path AS document_path, c.page_number, c.chunk_index, c.text, e.vector_json
            FROM pdf_chunks c
            JOIN pdf_documents d ON d.id = c.document_id
            JOIN pdf_embeddings e ON e.chunk_id = c.id
        """
        if pdf_scope:
            placeholders = ",".join("?" for _ in pdf_scope)
            sql += f" WHERE d.path IN ({placeholders})"
            params.extend(pdf_scope)
        rows = await self._fetchall(sql, params)
        return [
            PdfChunkRow(
                chunk_id=int(row["chunk_id"]),
                document_path=str(row["document_path"]),
                page_number=int(row["page_number"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
                embedding=_json_loads(row["vector_json"], None),
            )
            for row in rows
        ]

    async def get_pdf_path_hashes(self) -> dict[str, str]:
        rows = await self._fetchall("SELECT path, content_hash FROM pdf_documents")
        return {str(r["path"]): str(r["content_hash"]) for r in rows}

    async def log_retention_stats(self, deleted_run_events: int, deleted_run_steps: int, deleted_policy_decisions: int) -> None:
        async with self._lock:
            await self.conn.execute(
                """
                INSERT INTO retention_stats(executed_at, deleted_run_events, deleted_run_steps, deleted_policy_decisions)
                VALUES (?, ?, ?, ?)
                """,
                (_utc_now_iso(), deleted_run_events, deleted_run_steps, deleted_policy_decisions),
            )
            await self.conn.commit()

    async def run_retention(self, days: int) -> dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        async with self._lock:
            cur1 = await self.conn.execute("DELETE FROM run_events WHERE created_at < ?", (cutoff,))
            cur2 = await self.conn.execute("DELETE FROM run_steps WHERE created_at < ?", (cutoff,))
            cur3 = await self.conn.execute("DELETE FROM policy_decisions WHERE created_at < ?", (cutoff,))
            deleted_run_events = cur1.rowcount
            deleted_run_steps = cur2.rowcount
            deleted_policy_decisions = cur3.rowcount
            await self.conn.commit()
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
        rows = await self._fetchall(
            """
            SELECT run_id FROM runs
            WHERE updated_at < ? AND status IN (?, ?, ?)
            """,
            (cutoff, RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value),
        )
        return [str(r["run_id"]) for r in rows]

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

import aiosqlite

from orchestrator.persistence.artifact_recovery import ArtifactRecoveryService
from orchestrator.persistence.common import PdfChunkRow
from orchestrator.persistence.event_repository import EventRepository
from orchestrator.persistence.experiment_attempts_repository import ExperimentAttemptsRepository
from orchestrator.persistence.maintenance_repository import MaintenanceRepository
from orchestrator.persistence.pdf_repository import PdfRepository
from orchestrator.persistence.policy_decisions_repository import PolicyDecisionsRepository
from orchestrator.persistence.run_history_repository import RunHistoryRepository
from orchestrator.persistence.run_repository import RunRepository
from orchestrator.persistence.schema_manager import SchemaManager
from orchestrator.persistence.sqlite_recovery import SqliteRecoveryService
from orchestrator.persistence.task_repository import TaskRepository
from orchestrator.persistence.schemas import Priority, RunRecord, RunStatus, TaskSubmittedEvent
from orchestrator.persistence.verification_payloads import normalize_verification_payload


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


class Database:
    def __init__(self, sqlite_path: Path):
        self.sqlite_path = sqlite_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self.events = EventRepository(self)
        self.tasks = TaskRepository(self)
        self.runs = RunRepository(self)
        self.run_history = RunHistoryRepository(self)
        self.experiment_attempts = ExperimentAttemptsRepository(self)
        self.policy_decisions = PolicyDecisionsRepository(self)
        self.maintenance = MaintenanceRepository(self)
        self.pdf = PdfRepository(self)
        self.schema_manager = SchemaManager(self)
        self.sqlite_recovery = SqliteRecoveryService(self)
        self.artifact_recovery = ArtifactRecoveryService(self)

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
        return SqliteRecoveryService.is_database_corruption(exc)

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
        return await self.sqlite_recovery.is_database_healthy()

    async def _repair_database_file(self) -> None:
        await self.sqlite_recovery.repair_database_file()

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("database is not connected")
        return _RecoveredConnection(self)  # type: ignore[return-value]

    async def initialize_schema(self) -> None:
        await self.schema_manager.initialize_schema()

    async def rehydrate_terminal_runs_from_artifacts(self, runs_root: Path) -> dict[str, int]:
        return await self.artifact_recovery.rehydrate_terminal_runs_from_artifacts(runs_root)

    def _load_result_artifact_payload(self, result_path: Path) -> dict[str, Any] | None:
        return self.artifact_recovery.load_result_artifact_payload(result_path)

    def _coerce_iso_timestamp(self, raw: Any, *, fallback_path: Path) -> str:
        return self.artifact_recovery.coerce_iso_timestamp(raw, fallback_path=fallback_path)

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
        await self.schema_manager.ensure_runs_goal_signature_column()

    async def _ensure_runs_execution_cycle_columns(self) -> None:
        await self.schema_manager.ensure_runs_execution_cycle_columns()

    async def _ensure_run_steps_step_title_column(self) -> None:
        await self.schema_manager.ensure_run_steps_step_title_column()

    async def record_stream_event(
        self,
        event_id: str,
        stream: str,
        event_type: str,
        payload_json: str,
        run_id: str | None = None,
    ) -> bool:
        return await self.events.record_stream_event(event_id, stream, event_type, payload_json, run_id)

    async def upsert_task(self, event: TaskSubmittedEvent) -> None:
        await self.tasks.upsert_task(event)

    async def create_or_get_run(
        self,
        task_id: str,
        workspace_id: str,
        priority: Priority,
        goal_signature: str | None = None,
    ) -> str:
        return await self.runs.create_or_get_run(task_id, workspace_id, priority, goal_signature)

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        return await self.tasks.get_task(task_id)

    async def get_run(self, run_id: str) -> RunRecord | None:
        return await self.runs.get_run(run_id)

    async def list_runs(
        self,
        *,
        limit: int = 50,
        statuses: Sequence[RunStatus] | None = None,
    ) -> list[RunRecord]:
        return await self.runs.list_runs(limit=limit, statuses=statuses)

    async def list_runnable_runs(self, limit: int = 20) -> list[str]:
        return await self.runs.list_runnable_runs(limit)

    async def list_nonterminal_runs(self) -> list[str]:
        return await self.runs.list_nonterminal_runs()

    async def acquire_workspace_lock(self, workspace_id: str, run_id: str) -> bool:
        return await self.runs.acquire_workspace_lock(workspace_id, run_id)

    async def release_workspace_lock(self, workspace_id: str, run_id: str) -> None:
        await self.runs.release_workspace_lock(workspace_id, run_id)

    async def update_run_status(self, run_id: str, status: RunStatus, error_message: str | None = None) -> None:
        await self.runs.update_run_status(run_id, status, error_message)

    async def set_context(self, run_id: str, context: list[dict[str, Any]]) -> None:
        await self.runs.set_context(run_id, context)

    async def set_plan(self, run_id: str, plan: dict[str, Any]) -> None:
        await self.runs.set_plan(run_id, plan)

    async def set_goal_signature(self, run_id: str, goal_signature: str) -> None:
        await self.runs.set_goal_signature(run_id, goal_signature)

    async def increment_stage_attempt(self, run_id: str, stage: str) -> int:
        return await self.runs.increment_stage_attempt(run_id, stage)

    async def set_next_step_index(self, run_id: str, next_index: int) -> None:
        await self.runs.set_next_step_index(run_id, next_index)

    async def advance_execution_cycle(self, run_id: str) -> int:
        return await self.runs.advance_execution_cycle(run_id)

    async def set_verification(self, run_id: str, payload: dict[str, Any]) -> None:
        await self.runs.set_verification(run_id, payload)

    async def set_approved(self, run_id: str) -> bool:
        return await self.runs.set_approved(run_id)

    async def cancel_run(self, run_id: str, reason: str | None = None) -> bool:
        return await self.runs.cancel_run(run_id, reason)

    async def reset_run_for_retry(self, run_id: str) -> bool:
        return await self.runs.reset_run_for_retry(run_id)

    async def reset_run_for_goal_change(self, run_id: str, goal_signature: str | None = None) -> bool:
        return await self.runs.reset_run_for_goal_change(run_id, goal_signature)

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
        await self.run_history.insert_run_step(
            run_id=run_id,
            step_id=step_id,
            step_title=step_title,
            step_index=step_index,
            action=action,
            command=command,
            status=status,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            duration_ms=duration_ms,
            created_at=created_at,
        )

    async def count_executed_steps(self, run_id: str) -> int:
        return await self.run_history.count_executed_steps(run_id)

    async def count_attempted_steps(self, run_id: str) -> int:
        return await self.run_history.count_attempted_steps(run_id)

    async def list_run_steps(self, run_id: str) -> list[dict[str, Any]]:
        return await self.run_history.list_run_steps(run_id)

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
        recipe_snapshot: dict[str, Any] | None,
        recipe_diff: dict[str, Any] | None,
        strategy: dict[str, Any] | None,
        skill_paths: list[str],
        created_at: str | None = None,
    ) -> None:
        await self.experiment_attempts.record_experiment_attempt(
            workspace_id=workspace_id,
            goal_signature=goal_signature,
            run_id=run_id,
            task_id=task_id,
            run_attempt=run_attempt,
            verification_status=verification_status,
            quality_status=quality_status,
            quality_reason=quality_reason,
            metrics=metrics,
            hyperparameters=hyperparameters,
            recipe_snapshot=recipe_snapshot,
            recipe_diff=recipe_diff,
            strategy=strategy,
            skill_paths=skill_paths,
            created_at=created_at,
        )

    async def list_experiment_attempts(
        self,
        *,
        workspace_id: str,
        goal_signature: str,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        return await self.experiment_attempts.list_experiment_attempts(
            workspace_id=workspace_id,
            goal_signature=goal_signature,
            limit=limit,
        )

    async def get_latest_failed_step(self, run_id: str) -> dict[str, Any] | None:
        return await self.run_history.get_latest_failed_step(run_id)

    async def has_completed_step(self, run_id: str, step_id: str) -> bool:
        return await self.run_history.has_completed_step(run_id, step_id)

    async def add_artifact(self, run_id: str, kind: str, path: str) -> None:
        await self.run_history.add_artifact(run_id, kind, path)

    async def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        return await self.run_history.get_artifacts(run_id)

    async def add_policy_decision(
        self,
        run_id: str,
        layer: str,
        subject: str,
        decision: str,
        reason: str,
    ) -> None:
        await self.policy_decisions.add_policy_decision(run_id, layer, subject, decision, reason)

    async def upsert_pdf_document(
        self,
        path: str,
        content_hash: str,
        mtime: float,
        page_count: int,
    ) -> tuple[int, bool]:
        return await self.pdf.upsert_pdf_document(path, content_hash, mtime, page_count)

    async def remove_pdf_document(self, path: str) -> None:
        await self.pdf.remove_pdf_document(path)

    async def replace_pdf_chunks(
        self,
        document_id: int,
        path: str,
        chunks: Iterable[tuple[int, int, str]],
    ) -> list[int]:
        return await self.pdf.replace_pdf_chunks(document_id, path, chunks)

    async def set_chunk_embeddings(self, vectors: Iterable[tuple[int, list[float]]]) -> None:
        await self.pdf.set_chunk_embeddings(vectors)

    async def list_known_pdf_paths(self) -> list[str]:
        return await self.pdf.list_known_pdf_paths()

    async def fts_search(self, query: str, top_k: int, pdf_scope: list[str] | None = None) -> list[PdfChunkRow]:
        return await self.pdf.fts_search(query, top_k, pdf_scope)

    async def vector_candidates(self, pdf_scope: list[str] | None = None) -> list[PdfChunkRow]:
        return await self.pdf.vector_candidates(pdf_scope)

    async def get_pdf_path_hashes(self) -> dict[str, str]:
        return await self.pdf.get_pdf_path_hashes()

    async def log_retention_stats(self, deleted_run_events: int, deleted_run_steps: int, deleted_policy_decisions: int) -> None:
        await self.maintenance.log_retention_stats(
            deleted_run_events=deleted_run_events,
            deleted_run_steps=deleted_run_steps,
            deleted_policy_decisions=deleted_policy_decisions,
        )

    async def run_retention(self, days: int) -> dict[str, int]:
        return await self.maintenance.run_retention(days)

    async def list_run_directories_for_cleanup(self, days: int) -> list[str]:
        return await self.maintenance.list_run_directories_for_cleanup(days)

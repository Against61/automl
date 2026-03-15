from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
from uuid import uuid4

from orchestrator.persistence.common import json_dumps, json_loads, utc_now_iso
from orchestrator.persistence.run_record_mapper import row_to_run_record
from orchestrator.persistence.schemas import Priority, RunRecord, RunStatus
from orchestrator.persistence.verification_payloads import normalize_verification_payload

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class RunRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def create_or_get_run(
        self,
        task_id: str,
        workspace_id: str,
        priority: Priority,
        goal_signature: str | None = None,
    ) -> str:
        async with self.db._lock:
            row = await self.db._fetchone(
                """
                SELECT run_id, status, goal_signature, evaluation_contract_json
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
                    if not row["evaluation_contract_json"]:
                        task_row = await self.db._fetchone(
                            "SELECT evaluation_contract_json FROM tasks WHERE task_id = ?",
                            (task_id,),
                        )
                        if task_row and task_row["evaluation_contract_json"]:
                            await self.db.conn.execute(
                                "UPDATE runs SET evaluation_contract_json = ?, updated_at = ? WHERE run_id = ?",
                                (task_row["evaluation_contract_json"], utc_now_iso(), str(row["run_id"])),
                            )
                            await self.db.conn.commit()
                    return str(row["run_id"])
            task_row = await self.db._fetchone(
                "SELECT evaluation_contract_json FROM tasks WHERE task_id = ?",
                (task_id,),
            )
            evaluation_contract_json = task_row["evaluation_contract_json"] if task_row else None
            run_id = str(uuid4())
            now = utc_now_iso()
            await self.db.conn.execute(
                """
                INSERT INTO runs(
                    run_id,
                    task_id,
                    workspace_id,
                    priority,
                    status,
                    goal_signature,
                    evaluation_contract_json,
                    budget_tier,
                    execution_cycle,
                    cycle_started_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    workspace_id,
                    priority.value,
                    RunStatus.RECEIVED.value,
                    goal_signature,
                    evaluation_contract_json,
                    "micro",
                    0,
                    now,
                    now,
                    now,
                ),
            )
            await self.db.conn.commit()
            return run_id

    async def get_run(self, run_id: str) -> RunRecord | None:
        row = await self.db._fetchone("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not row:
            return None
        return row_to_run_record(row)

    async def list_runs(
        self,
        *,
        limit: int = 50,
        statuses: Sequence[RunStatus] | None = None,
    ) -> list[RunRecord]:
        params: list[Any] = []
        where_clause = ""
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where_clause = f"WHERE status IN ({placeholders})"
            params.extend(status.value for status in statuses)
        params.append(max(1, int(limit)))
        rows = await self.db._fetchall(
            f"""
            SELECT * FROM runs
            {where_clause}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            tuple(params),
        )
        return [row_to_run_record(row) for row in rows]

    async def list_runnable_runs(self, limit: int = 20) -> list[str]:
        rows = await self.db._fetchall(
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
        rows = await self.db._fetchall(
            """
            SELECT run_id FROM runs
            WHERE status NOT IN (?, ?, ?)
            """,
            (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value),
        )
        return [str(row["run_id"]) for row in rows]

    async def acquire_workspace_lock(self, workspace_id: str, run_id: str) -> bool:
        async with self.db._lock:
            row = await self.db._fetchone(
                "SELECT run_id FROM workspace_locks WHERE workspace_id = ?",
                (workspace_id,),
            )
            if row and row["run_id"] != run_id:
                locked_run = await self.get_run(str(row["run_id"]))
                if not locked_run or locked_run.status in RunStatus.terminal():
                    await self.db.conn.execute(
                        "DELETE FROM workspace_locks WHERE workspace_id = ?",
                        (workspace_id,),
                    )
                    await self.db.conn.execute(
                        """
                        INSERT INTO workspace_locks(workspace_id, run_id, locked_at)
                        VALUES (?, ?, ?)
                        """,
                        (workspace_id, run_id, utc_now_iso()),
                    )
                    await self.db.conn.commit()
                    return True
                return False
            if row and row["run_id"] == run_id:
                return True
            await self.db.conn.execute(
                """
                INSERT INTO workspace_locks(workspace_id, run_id, locked_at)
                VALUES (?, ?, ?)
                """,
                (workspace_id, run_id, utc_now_iso()),
            )
            await self.db.conn.commit()
            return True

    async def release_workspace_lock(self, workspace_id: str, run_id: str) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "DELETE FROM workspace_locks WHERE workspace_id = ? AND run_id = ?",
                (workspace_id, run_id),
            )
            await self.db.conn.commit()

    async def update_run_status(self, run_id: str, status: RunStatus, error_message: str | None = None) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                """
                UPDATE runs
                SET status = ?, error_message = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (status.value, error_message, utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def set_context(self, run_id: str, context: list[dict[str, Any]]) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE runs SET context_json = ?, updated_at = ? WHERE run_id = ?",
                (json_dumps(context), utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def set_plan(self, run_id: str, plan: dict[str, Any]) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                """
                UPDATE runs
                SET plan_json = ?,
                    next_step_index = 0,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (json_dumps(plan), utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def set_goal_signature(self, run_id: str, goal_signature: str) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE runs SET goal_signature = ?, updated_at = ? WHERE run_id = ?",
                (goal_signature, utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def set_evaluation_contract(self, run_id: str, evaluation_contract_json: dict[str, Any]) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE runs SET evaluation_contract_json = ?, updated_at = ? WHERE run_id = ?",
                (json_dumps(evaluation_contract_json), utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def set_budget_tier(self, run_id: str, budget_tier: str) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE runs SET budget_tier = ?, updated_at = ? WHERE run_id = ?",
                (str(budget_tier or "micro"), utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def increment_stage_attempt(self, run_id: str, stage: str) -> int:
        run = await self.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        attempts = dict(run.attempts_by_stage)
        current = int(attempts.get(stage, 0)) + 1
        attempts[stage] = current
        async with self.db._lock:
            await self.db.conn.execute(
                """
                UPDATE runs
                SET attempts_by_stage_json = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (json_dumps(attempts), utc_now_iso(), run_id),
            )
            await self.db.conn.commit()
        return current

    async def set_next_step_index(self, run_id: str, next_index: int) -> None:
        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE runs SET next_step_index = ?, updated_at = ? WHERE run_id = ?",
                (next_index, utc_now_iso(), run_id),
            )
            await self.db.conn.commit()

    async def advance_execution_cycle(self, run_id: str) -> int:
        run = await self.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")
        next_cycle = int(run.execution_cycle) + 1
        now = utc_now_iso()
        async with self.db._lock:
            await self.db.conn.execute(
                """
                UPDATE runs
                SET execution_cycle = ?,
                    cycle_started_at = ?,
                    next_step_index = 0,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (next_cycle, now, now, run_id),
            )
            await self.db.conn.commit()
        return next_cycle

    async def set_verification(self, run_id: str, payload: dict[str, Any]) -> None:
        row = await self.db._fetchone("SELECT verification_json FROM runs WHERE run_id = ?", (run_id,))
        previous_raw = row["verification_json"] if row else None
        previous_payload = json_loads(previous_raw, None)
        if not isinstance(previous_payload, dict):
            previous_payload = None

        async with self.db._lock:
            await self.db.conn.execute(
                "UPDATE runs SET verification_json = ?, updated_at = ? WHERE run_id = ?",
                (
                    json_dumps(normalize_verification_payload(payload, previous_payload)),
                    utc_now_iso(),
                    run_id,
                ),
            )
            await self.db.conn.commit()

    async def set_approved(self, run_id: str) -> bool:
        run = await self.get_run(run_id)
        if not run:
            return False
        if run.status not in {RunStatus.WAITING_APPROVAL, RunStatus.WAITING_PLAN_REVIEW}:
            return False
        async with self.db._lock:
            now = utc_now_iso()
            await self.db.conn.execute(
                """
                UPDATE runs
                SET approved_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (now, now, run_id),
            )
            await self.db.conn.commit()
        return True

    async def cancel_run(self, run_id: str, reason: str | None = None) -> bool:
        run = await self.get_run(run_id)
        if not run:
            return False
        if run.status in RunStatus.terminal():
            return False
        async with self.db._lock:
            now = utc_now_iso()
            await self.db.conn.execute(
                """
                UPDATE runs
                SET status = ?, cancelled_reason = ?, cancelled_at = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (RunStatus.CANCELLED.value, reason, now, now, run_id),
            )
            await self.db.conn.commit()
        return True

    async def reset_run_for_retry(self, run_id: str) -> bool:
        run = await self.get_run(run_id)
        if not run or run.status != RunStatus.FAILED:
            return False
        next_cycle = int(run.execution_cycle) + 1
        now = utc_now_iso()
        async with self.db._lock:
            await self.db.conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    error_message = NULL,
                    execution_cycle = ?,
                    cycle_started_at = ?,
                    next_step_index = 0,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (RunStatus.EXECUTING.value, next_cycle, now, now, run_id),
            )
            await self.db.conn.commit()
        return True

    async def reset_run_for_goal_change(self, run_id: str, goal_signature: str | None = None) -> bool:
        run = await self.get_run(run_id)
        if not run:
            return False
        if run.status in RunStatus.terminal():
            return False
        next_cycle = int(run.execution_cycle) + 1
        now = utc_now_iso()
        async with self.db._lock:
            await self.db.conn.execute("DELETE FROM run_steps WHERE run_id = ?", (run_id,))
            await self.db.conn.execute(
                """
                UPDATE runs
                SET status = ?,
                    error_message = NULL,
                    goal_signature = COALESCE(?, goal_signature),
                    budget_tier = 'micro',
                    execution_cycle = ?,
                    cycle_started_at = ?,
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
                (RunStatus.RECEIVED.value, goal_signature, next_cycle, now, now, run_id),
            )
            await self.db.conn.commit()
        return True

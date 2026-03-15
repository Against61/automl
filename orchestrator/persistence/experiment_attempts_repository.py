from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.persistence.common import json_dumps, json_loads, utc_now_iso
from orchestrator.persistence.verification_payloads import compact_strategy_payload

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class ExperimentAttemptsRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

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
        ts = created_at or utc_now_iso()
        async with self.db._lock:
            await self.db.conn.execute(
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
                    json_dumps(metrics or {}),
                    json_dumps(hyperparameters or {}),
                    json_dumps(compact_strategy_payload(strategy)) if strategy is not None else None,
                    json_dumps(skill_paths or []),
                    ts,
                ),
            )
            await self.db.conn.commit()

    async def list_experiment_attempts(
        self,
        *,
        workspace_id: str,
        goal_signature: str,
        limit: int = 12,
    ) -> list[dict[str, Any]]:
        rows = await self.db._fetchall(
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
                    "metrics": json_loads(row["metrics_json"], {}),
                    "hyperparameters": json_loads(row["hyperparameters_json"], {}),
                    "strategy": json_loads(row["strategy_json"], None),
                    "skill_paths": json_loads(row["skill_paths_json"], []),
                    "created_at": row["created_at"],
                }
            )
        return attempts

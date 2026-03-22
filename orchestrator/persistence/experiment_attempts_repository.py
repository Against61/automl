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
        recipe_snapshot: dict[str, Any] | None,
        recipe_diff: dict[str, Any] | None,
        strategy: dict[str, Any] | None,
        skill_paths: list[str],
        created_at: str | None = None,
    ) -> None:
        ts = created_at or utc_now_iso()
        persisted_recipe_diff = recipe_diff
        if persisted_recipe_diff is None and recipe_snapshot is not None:
            previous_snapshot = await self._latest_recipe_snapshot(
                workspace_id=workspace_id,
                goal_signature=goal_signature,
                run_id=run_id,
                run_attempt=run_attempt,
            )
            persisted_recipe_diff = self._recipe_diff(previous_snapshot, recipe_snapshot)
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
                    recipe_snapshot_json,
                    recipe_diff_json,
                    strategy_json,
                    skill_paths_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, run_attempt) DO UPDATE SET
                    verification_status=excluded.verification_status,
                    quality_status=excluded.quality_status,
                    quality_reason=excluded.quality_reason,
                    metrics_json=excluded.metrics_json,
                    hyperparameters_json=excluded.hyperparameters_json,
                    recipe_snapshot_json=excluded.recipe_snapshot_json,
                    recipe_diff_json=excluded.recipe_diff_json,
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
                    json_dumps(recipe_snapshot) if recipe_snapshot is not None else None,
                    json_dumps(persisted_recipe_diff) if persisted_recipe_diff is not None else None,
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
                recipe_snapshot_json,
                recipe_diff_json,
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
                    "recipe_snapshot": json_loads(row["recipe_snapshot_json"], None),
                    "recipe_diff": json_loads(row["recipe_diff_json"], None),
                    "strategy": json_loads(row["strategy_json"], None),
                    "skill_paths": json_loads(row["skill_paths_json"], []),
                    "created_at": row["created_at"],
                }
            )
        return attempts

    async def _latest_recipe_snapshot(
        self,
        *,
        workspace_id: str,
        goal_signature: str,
        run_id: str,
        run_attempt: int,
    ) -> dict[str, Any] | None:
        row = await self.db._fetchone(
            """
            SELECT recipe_snapshot_json
            FROM experiment_attempts
            WHERE workspace_id = ?
              AND goal_signature = ?
              AND NOT (run_id = ? AND run_attempt = ?)
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (workspace_id, goal_signature, run_id, int(run_attempt)),
        )
        if not row:
            return None
        return json_loads(row["recipe_snapshot_json"], None)

    @staticmethod
    def _recipe_diff(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(current, dict):
            return None
        if not isinstance(previous, dict):
            return {"changed_keys": sorted(current.keys()), "changes": {key: {"before": None, "after": value} for key, value in current.items()}}
        changes: dict[str, dict[str, Any]] = {}
        for key in sorted(set(previous.keys()) | set(current.keys())):
            before = previous.get(key)
            after = current.get(key)
            if before == after:
                continue
            changes[key] = {"before": before, "after": after}
        if not changes:
            return None
        return {
            "changed_keys": list(changes.keys()),
            "changes": changes,
        }

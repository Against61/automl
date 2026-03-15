from __future__ import annotations

from typing import TYPE_CHECKING

from orchestrator.persistence.schema_sql import BASE_SCHEMA_SQL

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class SchemaManager:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def initialize_schema(self) -> None:
        async with self.db._lock:
            await self.db.conn.executescript(BASE_SCHEMA_SQL)
            await self.ensure_tasks_evaluation_contract_column()
            await self.ensure_runs_goal_signature_column()
            await self.ensure_runs_evaluation_contract_column()
            await self.ensure_runs_budget_tier_column()
            await self.ensure_runs_execution_cycle_columns()
            await self.ensure_run_steps_step_title_column()
            await self.ensure_experiment_attempt_metric_columns()
            await self.db.conn.commit()

    async def ensure_tasks_evaluation_contract_column(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(tasks)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "evaluation_contract_json" in columns:
            return
        await self.db.conn.execute("ALTER TABLE tasks ADD COLUMN evaluation_contract_json TEXT")

    async def ensure_runs_goal_signature_column(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(runs)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "goal_signature" in columns:
            return
        await self.db.conn.execute("ALTER TABLE runs ADD COLUMN goal_signature TEXT")

    async def ensure_runs_evaluation_contract_column(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(runs)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "evaluation_contract_json" in columns:
            return
        await self.db.conn.execute("ALTER TABLE runs ADD COLUMN evaluation_contract_json TEXT")

    async def ensure_runs_budget_tier_column(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(runs)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "budget_tier" not in columns:
            await self.db.conn.execute("ALTER TABLE runs ADD COLUMN budget_tier TEXT NOT NULL DEFAULT 'micro'")
        await self.db.conn.execute("UPDATE runs SET budget_tier = COALESCE(NULLIF(budget_tier, ''), 'micro')")

    async def ensure_runs_execution_cycle_columns(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(runs)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "execution_cycle" not in columns:
            await self.db.conn.execute("ALTER TABLE runs ADD COLUMN execution_cycle INTEGER NOT NULL DEFAULT 0")
        if "cycle_started_at" not in columns:
            await self.db.conn.execute("ALTER TABLE runs ADD COLUMN cycle_started_at TEXT")
            await self.db.conn.execute(
                "UPDATE runs SET cycle_started_at = COALESCE(cycle_started_at, created_at)"
            )

    async def ensure_run_steps_step_title_column(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(run_steps)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "step_title" in columns:
            return
        await self.db.conn.execute("ALTER TABLE run_steps ADD COLUMN step_title TEXT")

    async def ensure_experiment_attempt_metric_columns(self) -> None:
        cursor = await self.db.conn.execute("PRAGMA table_info(experiment_attempts)")
        try:
            columns = [str(row["name"]) for row in (await cursor.fetchall())]
        finally:
            await cursor.close()
        if "final_metric_json" not in columns:
            await self.db.conn.execute("ALTER TABLE experiment_attempts ADD COLUMN final_metric_json TEXT")
        if "budget_tier_json" not in columns:
            await self.db.conn.execute("ALTER TABLE experiment_attempts ADD COLUMN budget_tier_json TEXT")
        if "proxy_metric_json" not in columns:
            await self.db.conn.execute("ALTER TABLE experiment_attempts ADD COLUMN proxy_metric_json TEXT")
        if "search_metric_json" not in columns:
            await self.db.conn.execute("ALTER TABLE experiment_attempts ADD COLUMN search_metric_json TEXT")

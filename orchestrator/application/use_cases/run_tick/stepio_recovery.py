from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import PlannerPlan, RunRecord, RunStatus, StepIOResult

logger = logging.getLogger(__name__)


class StepioRecoveryService:
    def __init__(
        self,
        *,
        db: Database,
        bus: Any,
        set_status: Callable[[str, RunStatus, str | None, RunRecord | None], Awaitable[None]],
    ) -> None:
        self.db = db
        self.bus = bus
        self.set_status = set_status

    @staticmethod
    def stepio_result_sort_key(path: Path) -> tuple[int, str]:
        name = path.name
        match = re.search(r"\.step_result(?:\.(\d+))?\.json$", name)
        if not match:
            return (0, name)
        suffix = match.group(1)
        index = 1 if suffix is None else int(suffix)
        return (index, name)

    @staticmethod
    def load_stepio_result(path: Path) -> StepIOResult | None:
        try:
            return StepIOResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("unable to parse stepio artifact: %s", path)
            return None

    @staticmethod
    def read_text_if_exists(path_value: str | None) -> str:
        if not path_value:
            return ""
        try:
            return Path(path_value).read_text(encoding="utf-8")
        except OSError:
            return ""

    @staticmethod
    def infer_reconciled_action(payload: StepIOResult) -> str:
        if payload.operation == "verify_metrics":
            return "verify"
        command = str(payload.command or "").strip().lower()
        if command.startswith("codex "):
            return "codex"
        if command:
            return "shell"
        return "read"

    async def reconcile_stepio_artifacts(
        self,
        *,
        run: RunRecord,
        run_path: Path,
    ) -> dict[str, StepIOResult]:
        if not run_path.exists():
            return {}

        stepio_files = sorted(run_path.glob("*.step_result*.json"), key=self.stepio_result_sort_key)
        if not stepio_files:
            return {}

        plan_index_by_id: dict[str, int] = {}
        plan_title_by_id: dict[str, str] = {}
        if run.plan_json:
            try:
                current_plan = PlannerPlan.model_validate(run.plan_json)
                plan_index_by_id = {step.id: index for index, step in enumerate(current_plan.steps)}
                plan_title_by_id = {step.id: step.title for step in current_plan.steps}
            except Exception:
                plan_index_by_id = {}
                plan_title_by_id = {}

        existing_rows = await self.db.list_run_steps(run.run_id)
        existing_counts: dict[str, int] = {}
        cycle_started_at = run.cycle_started_at or run.created_at
        for row in existing_rows:
            step_id = str(row.get("step_id") or "").strip()
            if not step_id:
                continue
            created_at_raw = str(row.get("created_at") or "").strip()
            if created_at_raw:
                try:
                    row_created_at = datetime.fromisoformat(created_at_raw)
                except ValueError:
                    row_created_at = None
                if row_created_at is not None and row_created_at < cycle_started_at:
                    continue
            existing_counts[step_id] = existing_counts.get(step_id, 0) + 1

        grouped: dict[str, list[tuple[Path, StepIOResult]]] = {}
        latest_by_step_id: dict[str, StepIOResult] = {}
        inserted = 0

        for path in stepio_files:
            try:
                artifact_created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=cycle_started_at.tzinfo or None)
            except OSError:
                continue
            if artifact_created_at < cycle_started_at:
                continue
            payload = self.load_stepio_result(path)
            if payload is None:
                continue
            grouped.setdefault(payload.step_id, []).append((path, payload))
            latest_by_step_id[payload.step_id] = payload

        for step_id, entries in grouped.items():
            entries.sort(key=lambda item: self.stepio_result_sort_key(item[0]))
            already_recorded = existing_counts.get(step_id, 0)
            for path, payload in entries[already_recorded:]:
                stdout_text = self.read_text_if_exists(payload.stdout_path)
                stderr_text = self.read_text_if_exists(payload.stderr_path)
                prefixed_stdout = f"summary: {payload.summary}"
                if stdout_text:
                    prefixed_stdout = f"{prefixed_stdout}\n{stdout_text}".strip()
                created_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
                await self.db.insert_run_step(
                    run_id=run.run_id,
                    step_id=payload.step_id,
                    step_title=plan_title_by_id.get(payload.step_id) or payload.step_id,
                    step_index=plan_index_by_id.get(payload.step_id, 0),
                    action=self.infer_reconciled_action(payload),
                    command=payload.command,
                    status="completed" if payload.status in {"completed", "auto_repaired"} else "failed",
                    stdout_text=prefixed_stdout,
                    stderr_text=stderr_text,
                    duration_ms=payload.duration_ms,
                    created_at=created_at,
                )
                inserted += 1

        if inserted:
            await self.bus.publish_internal(
                "run.stepio_reconciled",
                {
                    "run_id": run.run_id,
                    "inserted_steps": inserted,
                },
            )
        return latest_by_step_id

    async def sync_run_progress_from_stepio(
        self,
        *,
        run: RunRecord,
        latest_by_step_id: dict[str, StepIOResult],
    ) -> bool:
        if not run.plan_json:
            return False
        try:
            plan = PlannerPlan.model_validate(run.plan_json)
        except Exception:
            return False
        if not plan.steps:
            return False

        completed_prefix = 0
        for step in plan.steps:
            payload = latest_by_step_id.get(step.id)
            if payload is None or payload.status not in {"completed", "auto_repaired"}:
                break
            completed_prefix += 1

        changed = False
        if completed_prefix > run.next_step_index:
            await self.db.set_next_step_index(run.run_id, completed_prefix)
            changed = True

        if run.status == RunStatus.EXECUTING and completed_prefix >= len(plan.steps):
            await self.set_status(run.run_id, RunStatus.VERIFYING, None, None)
            await self.bus.publish_internal(
                "run.progress_recovered",
                {
                    "run_id": run.run_id,
                    "next_step_index": completed_prefix,
                    "transition": "EXECUTING->VERIFYING",
                },
            )
            return True

        if changed:
            await self.bus.publish_internal(
                "run.progress_recovered",
                {
                    "run_id": run.run_id,
                    "next_step_index": completed_prefix,
                },
            )
        return changed

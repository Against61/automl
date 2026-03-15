from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.persistence.common import json_dumps, utc_now_iso
from orchestrator.persistence.schemas import Priority, RunStatus

if TYPE_CHECKING:
    from orchestrator.persistence.db import Database


class ArtifactRecoveryService:
    def __init__(self, db: Database) -> None:
        self.db = db

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

        async with self.db._lock:
            for result_path in result_files:
                payload = self.load_result_artifact_payload(result_path)
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

                existing = await self.db._fetchone("SELECT 1 FROM runs WHERE run_id = ?", (run_id,))
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
                created_at = self.coerce_iso_timestamp(recovery.get("created_at"), fallback_path=result_path)
                updated_at = self.coerce_iso_timestamp(recovery.get("updated_at"), fallback_path=result_path)

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

                await self.db.conn.execute(
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
                        json_dumps(constraints),
                        json_dumps(pdf_scope),
                        json_dumps(payload_json),
                        self.coerce_iso_timestamp(task_meta.get("created_at"), fallback_path=result_path),
                        self.coerce_iso_timestamp(task_meta.get("updated_at"), fallback_path=result_path),
                    ),
                )
                stats["restored_tasks"] += 1

                await self.db.conn.execute(
                    """
                    INSERT INTO runs(
                        run_id,
                        task_id,
                        workspace_id,
                        priority,
                        status,
                        goal_signature,
                        execution_cycle,
                        cycle_started_at,
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
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        task_id,
                        workspace_id,
                        priority.value,
                        status.value,
                        None,
                        0,
                        created_at,
                        created_at,
                        updated_at,
                        "{}",
                        0,
                        json_dumps(recovery.get("plan_json")) if recovery.get("plan_json") is not None else None,
                        json_dumps(recovery.get("context_json")) if recovery.get("context_json") is not None else None,
                        json_dumps(recovery.get("verification_json")) if recovery.get("verification_json") is not None else None,
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
                    await self.db.conn.execute(
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
                            self.coerce_iso_timestamp(step.get("created_at"), fallback_path=result_path),
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
                    await self.db.conn.execute(
                        """
                        INSERT INTO artifacts(run_id, kind, path, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            kind,
                            path,
                            self.coerce_iso_timestamp(artifact.get("created_at"), fallback_path=result_path),
                        ),
                    )
                    stats["restored_artifacts"] += 1

                packaged = recovery.get("packaged_artifacts") if isinstance(recovery.get("packaged_artifacts"), dict) else {}
                for kind, key in (("patch_bundle", "patch_bundle_path"), ("report_json", "report_json_path")):
                    path = str(packaged.get(key) or "").strip()
                    if not path:
                        continue
                    await self.db.conn.execute(
                        """
                        INSERT INTO artifacts(run_id, kind, path, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (run_id, kind, path, updated_at),
                    )
                    stats["restored_artifacts"] += 1

            await self.db.conn.commit()
        return stats

    @staticmethod
    def load_result_artifact_payload(result_path: Path) -> dict[str, Any] | None:
        try:
            raw = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def coerce_iso_timestamp(raw: Any, *, fallback_path: Path) -> str:
        if isinstance(raw, str):
            try:
                datetime.fromisoformat(raw)
                return raw
            except ValueError:
                pass
        try:
            return datetime.fromtimestamp(fallback_path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            return utc_now_iso()

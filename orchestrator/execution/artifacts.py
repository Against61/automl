from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from orchestrator.persistence.db import Database


class ArtifactPublisher:
    def __init__(self, db: Database, runs_root: Path):
        self.db = db
        self.runs_root = runs_root
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _git_patch(self, workspace_path: Path) -> str:
        try:
            output = subprocess.check_output(
                ["git", "-C", workspace_path.as_posix(), "diff", "--binary"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            return output
        except Exception:
            return ""

    async def package(
        self,
        run_id: str,
        task_id: str,
        workspace_path: Path,
        status: str,
        summary: dict[str, Any],
    ) -> dict[str, str]:
        run_dir = self.run_dir(run_id)
        steps = await self.db.list_run_steps(run_id)
        artifacts = await self.db.get_artifacts(run_id)
        run = await self.db.get_run(run_id)
        task = await self.db.get_task(task_id)
        patch_diff = self._git_patch(workspace_path)
        patch_path = run_dir / "patch.diff"
        patch_path.write_text(patch_diff, encoding="utf-8")

        result_payload = {
            "run_id": run_id,
            "task_id": task_id,
            "status": status,
            "summary": summary,
            "steps": steps,
            "artifacts": artifacts,
            "recovery": {
                "workspace_id": run.workspace_id if run else workspace_path.name,
                "priority": run.priority.value if run else "normal",
                "created_at": run.created_at.isoformat() if run else None,
                "updated_at": run.updated_at.isoformat() if run else None,
                "error_message": run.error_message if run else None,
                "plan_json": run.plan_json if run else None,
                "context_json": run.context_json if run else None,
                "verification_json": run.verification_json if run else None,
                "task": {
                    "goal": task.get("goal") if task else "",
                    "constraints": json.loads(task.get("constraints_json", "[]")) if task else [],
                    "pdf_scope": json.loads(task.get("pdf_scope_json", "[]")) if task else [],
                    "payload_json": json.loads(task.get("payload_json", "{}")) if task else {},
                    "workspace_id": task.get("workspace_id") if task else run.workspace_id if run else workspace_path.name,
                    "priority": task.get("priority") if task else run.priority.value if run else "normal",
                    "created_at": task.get("created_at") if task else None,
                    "updated_at": task.get("updated_at") if task else None,
                },
                "packaged_artifacts": {
                    "patch_bundle_path": (run_dir / "patches.tar.gz").as_posix(),
                    "report_json_path": (run_dir / "result.json").as_posix(),
                },
            },
        }
        result_json_path = run_dir / "result.json"
        result_json_path.write_text(json.dumps(result_payload, ensure_ascii=True, indent=2), encoding="utf-8")

        bundle_path = run_dir / "patches.tar.gz"
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(patch_path, arcname="patch.diff")
            tar.add(result_json_path, arcname="result.json")

        await self.db.add_artifact(run_id, "patch_bundle", bundle_path.as_posix())
        await self.db.add_artifact(run_id, "report_json", result_json_path.as_posix())
        return {
            "patch_bundle_path": bundle_path.as_posix(),
            "report_json_path": result_json_path.as_posix(),
        }

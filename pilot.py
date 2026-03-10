from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from orchestrator.config import Settings
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.runtime.bus import InMemoryEventBus
from orchestrator.execution.codex_runner import CodexRunner
from orchestrator.persistence.db import Database
from orchestrator.planning.planner import make_planner
from orchestrator.execution.policy import PolicyEngine
from orchestrator.planning.ralph import RalphBacklogService
from orchestrator.persistence.schemas import RunStatus
from orchestrator.runtime.session import SessionManager
from orchestrator.execution.verifier import Verifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual pilot loop (plan -> execute -> observe)")
    parser.add_argument("--goal", required=True, help="Task goal for planner")
    parser.add_argument("--workspace-id", default="default", help="Workspace identifier")
    parser.add_argument("--priority", default="normal", choices=["high", "normal", "low"])
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--auto-approve", action="store_true", help="Auto approve WAITING_APPROVAL steps")
    return parser.parse_args()


def append_session_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


async def run_loop(args: argparse.Namespace) -> int:
    settings = Settings()
    settings.session_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)

    db = Database(settings.sqlite_path)
    await db.connect()
    bus = InMemoryEventBus()
    await bus.initialize()

    planner = make_planner(settings)
    policy_engine = PolicyEngine(settings)
    ralph_backlog = RalphBacklogService(
        prd_file_name=settings.ralph_prd_file_name,
        progress_file_name=settings.ralph_progress_file_name,
    )
    codex_runner = CodexRunner(settings)
    verifier = Verifier(settings)
    artifacts = ArtifactPublisher(db, settings.runs_root)
    session = SessionManager(
        settings=settings,
        db=db,
        bus=bus,
        planner=planner,
        policy_engine=policy_engine,
        ralph_backlog=ralph_backlog,
        codex_runner=codex_runner,
        verifier=verifier,
        artifact_publisher=artifacts,
    )

    task_event = {
        "event_id": str(uuid4()),
        "event_type": "task.submitted",
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": str(uuid4()),
        "workspace_id": args.workspace_id,
        "priority": args.priority,
        "payload": {
            "goal": args.goal,
            "constraints": [],
            "pdf_scope": [],
            "execution_mode": "plan_execute",
        },
    }
    run_id = await session.submit_task_event(task_event)
    if not run_id:
        print("failed to create run (duplicate event)")
        await db.close()
        await bus.close()
        return 1

    for iteration in range(1, max(1, args.max_iterations) + 1):
        await session.process_run(run_id)
        run = await db.get_run(run_id)
        if run is None:
            break
        executed_steps = await db.count_executed_steps(run_id)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "iteration": iteration,
            "run_id": run_id,
            "status": run.status.value,
            "executed_steps": executed_steps,
            "error_message": run.error_message,
        }
        append_session_log(settings.session_jsonl_path, entry)
        print(f"[{iteration}] status={run.status.value} executed_steps={executed_steps}")

        if run.status in RunStatus.terminal():
            artifacts_list = await db.get_artifacts(run_id)
            print(f"terminal={run.status.value} artifacts={len(artifacts_list)}")
            await db.close()
            await bus.close()
            return 0 if run.status == RunStatus.COMPLETED else 2

        should_auto_approve = args.auto_approve or settings.auto_approve_in_pilot
        if run.status == RunStatus.WAITING_APPROVAL and should_auto_approve:
            await session.handle_control_event(
                {
                    "event_id": str(uuid4()),
                    "event_type": "run.approve",
                    "schema_version": "1.0",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "payload": {"run_id": run_id},
                }
            )

        await asyncio.sleep(0.1)

    print("max iterations reached before terminal state")
    await db.close()
    await bus.close()
    return 3


def main() -> None:
    args = parse_args()
    raise SystemExit(asyncio.run(run_loop(args)))


if __name__ == "__main__":
    main()

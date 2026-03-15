from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from .bus import RedisEventBus
from orchestrator.execution.artifacts import ArtifactPublisher
from orchestrator.execution.codex_runner import CodexRunner
from orchestrator.config import Settings
from orchestrator.persistence.db import Database
from orchestrator.planning.planner import make_planner
from orchestrator.execution.policy import PolicyEngine
from orchestrator.planning.ralph import RalphBacklogService
from orchestrator.persistence.schemas import RunStatus
from orchestrator.runtime.session import SessionManager
from orchestrator.execution.verifier import Verifier

logger = logging.getLogger(__name__)


class OrchestratorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.sqlite_path)
        self.bus = RedisEventBus(
            redis_url=settings.redis_url,
            stream_tasks=settings.stream_tasks,
            stream_control=settings.stream_control,
            stream_internal=settings.stream_internal,
            stream_results=settings.stream_results,
            consumer_group=settings.redis_consumer_group,
            consumer_name=settings.redis_consumer_name,
        )
        planner = make_planner()
        policy_engine = PolicyEngine(settings)
        ralph_backlog = RalphBacklogService(
            prd_file_name=settings.ralph_prd_file_name,
            progress_file_name=settings.ralph_progress_file_name,
        )
        codex_runner = CodexRunner(settings)
        verifier = Verifier(settings)
        artifact_publisher = ArtifactPublisher(self.db, settings.runs_root)
        self.session = SessionManager(
            settings=settings,
            db=self.db,
            bus=self.bus,
            planner=planner,
            policy_engine=policy_engine,
            ralph_backlog=ralph_backlog,
            codex_runner=codex_runner,
            verifier=verifier,
            artifact_publisher=artifact_publisher,
        )
        self._tasks: list[asyncio.Task[Any]] = []
        self._run_queue: asyncio.Queue[str] = asyncio.Queue()
        self._queued: set[str] = set()
        self._ready = False
        self._stop = asyncio.Event()
        self._dispatch_lock = asyncio.Lock()

    async def start(self) -> None:
        await self.db.connect()
        recovery_stats = await self.db.rehydrate_terminal_runs_from_artifacts(self.settings.runs_root)
        if recovery_stats["restored_runs"] or recovery_stats["restored_tasks"]:
            logger.info("filesystem recovery stats: %s", recovery_stats)
        await self.bus.initialize()
        await self._recover_nonterminal_runs()
        self._tasks.append(asyncio.create_task(self._task_gateway_loop(), name="gateway-tasks"))
        self._tasks.append(asyncio.create_task(self._control_gateway_loop(), name="gateway-control"))
        self._tasks.append(asyncio.create_task(self._dispatcher_loop(), name="dispatcher"))
        self._tasks.append(asyncio.create_task(self._retention_loop(), name="retention"))
        for idx in range(self.settings.worker_concurrency):
            self._tasks.append(asyncio.create_task(self._worker_loop(idx), name=f"worker-{idx+1}"))
        self._ready = True

    async def stop(self) -> None:
        self._ready = False
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("background task failed on shutdown")
        self._tasks.clear()
        await self.bus.close()
        await self.db.close()

    def readiness(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "worker_concurrency": self.settings.worker_concurrency,
            "queue_size": self._run_queue.qsize(),
        }

    async def _recover_nonterminal_runs(self) -> None:
        run_ids = await self.db.list_nonterminal_runs()
        for run_id in run_ids:
            await self.enqueue_run(run_id)

    async def enqueue_run(self, run_id: str) -> None:
        async with self._dispatch_lock:
            if run_id in self._queued:
                return
            self._queued.add(run_id)
            await self._run_queue.put(run_id)

    async def _task_gateway_loop(self) -> None:
        async for message_id, payload in self.bus.iter_task_events():
            if self._stop.is_set():
                return
            if not message_id:
                continue
            try:
                run_id = await self.session.submit_task_event(payload)
                if run_id:
                    await self.enqueue_run(run_id)
            except ValidationError as exc:
                logger.error("invalid task event: %s", exc)
            except Exception:
                logger.exception("failed to handle task event")
            finally:
                await self.bus.ack_task(message_id)

    async def _control_gateway_loop(self) -> None:
        async for message_id, payload in self.bus.iter_control_events():
            if self._stop.is_set():
                return
            if not message_id:
                continue
            try:
                changed = await self.session.handle_control_event(payload)
                run_id = payload.get("payload", {}).get("run_id")
                if changed and run_id:
                    await self.enqueue_run(str(run_id))
            except ValidationError as exc:
                logger.error("invalid control event: %s", exc)
            except Exception:
                logger.exception("failed to handle control event")
            finally:
                await self.bus.ack_control(message_id)

    async def _dispatcher_loop(self) -> None:
        while not self._stop.is_set():
            try:
                run_ids = await self.db.list_runnable_runs(limit=self.settings.worker_concurrency * 4)
                for run_id in run_ids:
                    await self.enqueue_run(run_id)
            except Exception:
                logger.exception("dispatcher failure")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.poll_interval_sec)
            except asyncio.TimeoutError:
                pass

    async def _worker_loop(self, worker_idx: int) -> None:
        while not self._stop.is_set():
            run_id = await self._run_queue.get()
            async with self._dispatch_lock:
                self._queued.discard(run_id)
            try:
                await self.session.process_run(run_id)
                run = await self.db.get_run(run_id)
                if (
                    run
                    and run.status not in RunStatus.terminal()
                    and run.status not in {RunStatus.WAITING_APPROVAL, RunStatus.WAITING_PLAN_REVIEW}
                ):
                    await self.enqueue_run(run_id)
            except Exception:
                logger.exception("worker %d failed for run %s", worker_idx, run_id)
            finally:
                self._run_queue.task_done()

    async def _retention_loop(self) -> None:
        interval = max(3600, int(self.settings.poll_interval_sec * 20))
        while not self._stop.is_set():
            try:
                stats = await self.db.run_retention(self.settings.retention_days)
                logger.info("retention stats: %s", stats)
            except Exception:
                logger.exception("retention failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def submit_control_event(self, event_type: str, run_id: str, reason: str | None = None) -> str:
        payload = {
            "event_id": str(uuid4()),
            "event_type": event_type,
            "schema_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"run_id": run_id, "reason": reason},
        }
        return await self.bus.publish_control(payload)

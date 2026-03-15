from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request

from orchestrator.persistence.schemas import (
    ApproveRequest,
    CancelRequest,
    EventCreateRequest,
    HealthResponse,
    ReadyResponse,
    RunStatus,
)

router = APIRouter()


def get_service(request: Request):
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise RuntimeError("service is not initialized")
    return service

@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    service = get_service(request)
    return HealthResponse(app=service.settings.app_name)


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(request: Request) -> ReadyResponse:
    service = get_service(request)
    details = service.readiness()
    return ReadyResponse(status="ready" if details.get("ready") else "not_ready", details=details)


@router.get("/runs")
async def list_runs(
    request: Request,
    status: list[str] | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> dict:
    service = get_service(request)
    statuses: list[RunStatus] | None = None
    if status:
        statuses = []
        for item in status:
            normalized = str(item or "").strip()
            try:
                statuses.append(RunStatus(normalized))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"invalid status: {normalized}") from exc
    runs = await service.db.list_runs(limit=limit, statuses=statuses)
    return {
        "runs": [
            {
                "run_id": run.run_id,
                "task_id": run.task_id,
                "workspace_id": run.workspace_id,
                "priority": run.priority.value,
                "status": run.status.value,
                "created_at": run.created_at.isoformat(),
                "updated_at": run.updated_at.isoformat(),
                "error_message": run.error_message,
            }
            for run in runs
        ]
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict:
    service = get_service(request)
    run = await service.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    artifacts = await service.db.get_artifacts(run_id)
    steps = await service.db.list_run_steps(run_id)
    payload = run.model_dump(mode="json")
    payload["artifacts"] = artifacts
    payload["steps"] = steps
    return payload


@router.get("/status/{run_id}")
async def status(run_id: str, request: Request) -> dict:
    service = get_service(request)
    run = await service.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run_id,
        "status": run.status.value,
        "updated_at": run.updated_at.isoformat(),
        "error_message": run.error_message,
    }


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request) -> dict:
    service = get_service(request)
    task = await service.db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.get("/artifacts/{run_id}")
async def get_artifacts(run_id: str, request: Request) -> dict:
    service = get_service(request)
    run = await service.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {"run_id": run_id, "artifacts": await service.db.get_artifacts(run_id)}


@router.post("/event")
async def create_event(payload: EventCreateRequest, request: Request) -> dict:
    service = get_service(request)
    constraints = [str(item).strip() for item in payload.constraints if str(item).strip()]

    metric_key = (payload.required_metric_key or "").strip()
    metric_min = payload.required_metric_min
    if metric_key and metric_min is not None:
        existing_metric = any(
            item.lower().startswith("ralph_required_metric:")
            or item.lower().startswith("required_metric:")
            for item in constraints
        )
        if not existing_metric:
            use_percent = metric_min > 1.0
            metric_value = f"{metric_min:g}%" if use_percent else f"{metric_min:g}"
            constraints.append(f"RALPH_REQUIRED_METRIC: {metric_key} >= {metric_value}")

    if payload.max_quality_retries is not None:
        existing_retry = any(item.lower().startswith("max_quality_retries:") for item in constraints)
        if not existing_retry:
            constraints.append(f"MAX_QUALITY_RETRIES: {int(payload.max_quality_retries)}")

    event = {
        "event_id": str(uuid4()),
        "event_type": "task.submitted",
        "schema_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": str(uuid4()),
        "workspace_id": payload.workspace_id,
        "priority": payload.priority.value,
        "payload": {
            "goal": payload.goal,
            "constraints": constraints,
            "pdf_scope": payload.pdf_scope,
            "execution_mode": payload.execution_mode,
        },
    }
    run_id = await service.session.submit_task_event(event)
    if not run_id:
        raise HTTPException(status_code=409, detail="duplicate event")
    await service.enqueue_run(run_id)
    return {"accepted": True, "run_id": run_id}


@router.post("/control/approve")
async def approve_run(payload: ApproveRequest, request: Request) -> dict:
    service = get_service(request)
    message_id = await service.submit_control_event("run.approve", str(payload.run_id))
    return {"accepted": True, "stream_id": message_id}


@router.post("/control/cancel")
async def cancel_run(payload: CancelRequest, request: Request) -> dict:
    service = get_service(request)
    message_id = await service.submit_control_event("run.cancel", str(payload.run_id), reason=payload.reason)
    return {"accepted": True, "stream_id": message_id}

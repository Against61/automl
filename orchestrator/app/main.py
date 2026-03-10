from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from orchestrator.app.api import router
from orchestrator.config import get_settings
from orchestrator.runtime.service import OrchestratorService


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _configure_logging(settings.log_level)
    service = OrchestratorService(settings)
    app.state.service = service
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(title="Codex Orchestrator", version="0.1.0", lifespan=lifespan)
app.include_router(router)

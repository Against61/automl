from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlannerStepRequest(BaseModel):
    goal: str
    step_id: str
    step_type: Literal["change", "check"] = "change"
    commands: list[str] = Field(default_factory=list)
    codex_prompt: str | None = None
    expected_artifacts: list[str] = Field(default_factory=list)
    stop_condition: str = ""
    risk_level: Literal["low", "medium", "high"] = "low"


class PlannerStepResult(BaseModel):
    step_id: str
    exit_code: int
    summary: str
    diff_stats: str = ""
    test_report: str = ""
    files_changed: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_path: str | None = None


class ObservationSummary(BaseModel):
    git_status_porcelain: str = ""
    git_diff_stat: str = ""
    important_files: list[str] = Field(default_factory=list)
    notes: str = ""

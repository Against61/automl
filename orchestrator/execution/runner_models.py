from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StepExecutionResult:
    status: str
    exit_code: int
    summary: str
    stdout_text: str
    stderr_text: str
    duration_ms: int
    command: str | None
    diff_stats: str = ""
    test_report: str = ""
    files_changed: list[str] = None  # type: ignore[assignment]
    errors: list[str] = None  # type: ignore[assignment]
    stdout_path: str | None = None
    stderr_path: str | None = None
    log_path: str | None = None
    is_infra_error: bool = False
    auto_repaired: bool = False
    missing_artifact: str | None = None

    def __post_init__(self) -> None:
        if self.files_changed is None:
            self.files_changed = []
        if self.errors is None:
            self.errors = []


@dataclass(slots=True)
class TimeoutProfile:
    hard_timeout_sec: int | None
    idle_timeout_sec: int | None
    max_wall_clock_sec: int | None
    label: str = "default"


class IdleProcessTimeout(RuntimeError):
    pass

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    app_name: str = "codex-orchestrator"
    environment: str = "dev"
    log_level: str = "INFO"

    sqlite_path: Path = Path("./data/orchestrator.db")
    workspace_root: Path = Path("./workspace")
    pdf_root: Path = Path("./workspace/knowledge/pdfs")
    runs_root: Path = Path("./workspace/runs")

    redis_url: str = "redis://localhost:6379/0"
    stream_tasks: str = "agent.tasks"
    stream_internal: str = "agent.internal"
    stream_results: str = "agent.results"
    stream_control: str = "agent.control"
    redis_consumer_group: str = "orchestrator"
    redis_consumer_name: str = "worker-1"

    worker_concurrency: int = Field(default=2, ge=1, le=32)
    poll_interval_sec: float = Field(default=2.0, ge=0.5, le=30.0)
    retention_days: int = Field(default=30, ge=1, le=365)
    max_run_steps: int = Field(default=20, ge=1, le=200)
    auto_approve_in_pilot: bool = False

    session_jsonl_path: Path = Path("./workspace/session.jsonl")
    important_files: str = "README.md,pyproject.toml"

    llm_provider: str = "openai"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    codex_model: str = "gpt-5.3-codex-spark"
    research_use_arxiv: bool = True
    research_use_hf_papers: bool = True
    research_backend_url: str = ""
    research_backend_api_key: str | None = None
    research_http_timeout_sec: float = Field(default=8.0, ge=1.0, le=60.0)
    research_max_hits: int = Field(default=3, ge=1, le=10)

    codex_cli_cmd: str = "codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check"
    codex_use_openai_api_key: bool = True
    auto_install_missing_python_modules: bool = True
    execution_failure_replan_enabled: bool = True
    max_execution_replans: int = Field(default=1, ge=0, le=5)
    max_quality_replans: int | None = None
    plan_review_enabled: bool = True
    plan_review_manual_fallback: bool = True
    contract_strictness: str = "balanced"
    experiment_history_context_limit: int = Field(default=12, ge=1, le=100)
    ralph_prd_file_name: str = "prd.json"
    ralph_progress_file_name: str = "progress.txt"
    ralph_auto_queue_next_story: bool = False
    codex_step_timeout_sec: int = Field(default=900, ge=1, le=7200)
    training_idle_timeout_sec: int = Field(default=1200, ge=0, le=86400)
    training_max_wall_clock_sec: int = Field(default=0, ge=0, le=604800)
    max_stdio_bytes: int = Field(default=262_144, ge=4096)
    cpu_training_budget_epoch_trials_limit: int = Field(default=20, ge=1, le=500)

    verify_commands: str = ""
    allow_paths: str = "./workspace"
    require_approval_patterns: str = r"git push|git commit|rm -rf|docker|kubectl|terraform apply"
    deny_patterns: str = r"(^|\s)rm -rf\s+/|(^|\s)shutdown\s|(^|\s)reboot\s|(^|\s)mkfs\s"
    block_network_commands: bool = True

    @property
    def verify_command_list(self) -> list[str]:
        raw = self.verify_commands.strip()
        return [part.strip() for part in raw.split(";;") if part.strip()]

    @property
    def quality_replan_limit(self) -> int:
        return self.max_execution_replans if self.max_quality_replans is None else self.max_quality_replans

    @property
    def allowed_paths(self) -> list[Path]:
        values = [value.strip() for value in self.allow_paths.split(",") if value.strip()]
        if not values:
            values = [str(self.workspace_root)]
        return [Path(v).resolve() for v in values]

    @property
    def important_file_list(self) -> list[str]:
        values = [value.strip() for value in self.important_files.split(",") if value.strip()]
        return values

    @property
    def approval_regexes(self) -> list[str]:
        raw = self.require_approval_patterns.strip()
        if not raw:
            return []
        if ";;" in raw:
            return [v.strip() for v in raw.split(";;") if v.strip()]
        return [raw]

    @property
    def deny_regexes(self) -> list[str]:
        raw = self.deny_patterns.strip()
        if not raw:
            return []
        if ";;" in raw:
            return [v.strip() for v in raw.split(";;") if v.strip()]
        return [raw]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.workspace_root.mkdir(parents=True, exist_ok=True)
    settings.runs_root.mkdir(parents=True, exist_ok=True)
    settings.session_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return settings

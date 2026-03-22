BASE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    priority TEXT NOT NULL,
    goal TEXT NOT NULL,
    constraints_json TEXT NOT NULL,
    pdf_scope_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    priority TEXT NOT NULL,
    status TEXT NOT NULL,
    goal_signature TEXT,
    execution_cycle INTEGER NOT NULL DEFAULT 0,
    cycle_started_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    attempts_by_stage_json TEXT NOT NULL DEFAULT '{}',
    next_step_index INTEGER NOT NULL DEFAULT 0,
    plan_json TEXT,
    context_json TEXT,
    verification_json TEXT,
    error_message TEXT,
    approved_at TEXT,
    cancelled_at TEXT,
    cancelled_reason TEXT,
    FOREIGN KEY(task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_status_priority_created
    ON runs(status, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_workspace_status
    ON runs(workspace_id, status);

CREATE TABLE IF NOT EXISTS run_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    stream TEXT NOT NULL,
    event_type TEXT NOT NULL,
    run_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_events_run_id ON run_events(run_id);
CREATE INDEX IF NOT EXISTS idx_run_events_created_at ON run_events(created_at);

CREATE TABLE IF NOT EXISTS run_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    step_title TEXT,
    step_index INTEGER NOT NULL,
    action TEXT NOT NULL,
    command TEXT,
    status TEXT NOT NULL,
    stdout_text TEXT,
    stderr_text TEXT,
    duration_ms INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_steps_run_id ON run_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_run_steps_created_at ON run_steps(created_at);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_id ON artifacts(run_id);

CREATE TABLE IF NOT EXISTS workspace_locks (
    workspace_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    locked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policy_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    layer TEXT NOT NULL,
    subject TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_policy_decisions_run_id ON policy_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_created_at ON policy_decisions(created_at);

CREATE TABLE IF NOT EXISTS pdf_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    mtime REAL NOT NULL,
    page_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pdf_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES pdf_documents(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pdf_chunks_doc_idx
    ON pdf_chunks(document_id, chunk_index);

CREATE TABLE IF NOT EXISTS pdf_embeddings (
    chunk_id INTEGER PRIMARY KEY,
    vector_json TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES pdf_chunks(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS pdf_chunks_fts
    USING fts5(chunk_id UNINDEXED, path UNINDEXED, text);

CREATE TABLE IF NOT EXISTS retention_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    executed_at TEXT NOT NULL,
    deleted_run_events INTEGER NOT NULL,
    deleted_run_steps INTEGER NOT NULL,
    deleted_policy_decisions INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    goal_signature TEXT NOT NULL,
    run_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    run_attempt INTEGER NOT NULL,
    verification_status TEXT,
    quality_status TEXT,
    quality_reason TEXT,
    metrics_json TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    recipe_snapshot_json TEXT,
    recipe_diff_json TEXT,
    strategy_json TEXT,
    skill_paths_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(run_id, run_attempt)
);

CREATE INDEX IF NOT EXISTS idx_experiment_attempts_lookup
    ON experiment_attempts(workspace_id, goal_signature, created_at);
CREATE INDEX IF NOT EXISTS idx_experiment_attempts_run
    ON experiment_attempts(run_id, created_at);
"""

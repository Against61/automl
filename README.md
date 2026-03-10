# Codex Event-Driven Orchestrator

Long-running daemon service that receives tasks from Redis Streams, plans/executes work with an LLM, delegates execution to Codex CLI, and publishes structured run artifacts.

## Features
- Event-driven runtime (`agent.tasks`, `agent.control`, `agent.results`, `agent.internal`)
- SQLite state store + workspace filesystem artifacts
- FSM lifecycle: `RECEIVED -> CONTEXT_READY -> PLAN_READY -> WAITING_APPROVAL/WAITING_PLAN_REVIEW? -> EXECUTING -> VERIFYING -> PACKAGING -> terminal`
- Skill-first workflow: Codex skills + workspace skills as the primary reusable knowledge source
- Policy chain with `ALLOW | REQUIRE_APPROVAL | DENY`
- Manual approval gate for risky steps
- Patch bundle + `result.json` output
- Manual pilot loop script (`pilot.py`) for quick local orchestration

## Architecture
- Layered model: `domain` + `application` + infrastructure adapters + runtime + API.
- `runtime/session.py` is a compatibility facade; orchestration logic lives in `application/use_cases/process_run_tick.py`.
- Domain transition rules are centralized in `orchestrator/domain/state_machine.py`.
- PDF retrieval/indexing has been removed from the runtime path; knowledge reuse is expected through skills and workspace notes.
- Improvement strategy discovery now scans both `workspace/<workspace_id>/knowledge/skills` and `CODEX_HOME/skills`.
- Architecture guide: `ARCHITECTURE.md`.
- Import-boundary check:
```bash
python scripts/check_arch_imports.py
```

## Quickstart
1. Install dependencies:
```bash
pip install -e .[dev]
```

2. Set environment in `.env` (create it if needed).
Set `OPENAI_API_KEY` for OpenAI mode, or switch to `LLM_PROVIDER=stub` for local/offline development.
Planner behavior is controlled by `PLANNER_MODE`:
- `llm_plan` (default): planner builds structured multi-step plan via LLM/stub planner.
- `codex_only`: no planner reasoning; orchestrator emits one synthetic Codex execution step.
Task execution mode is controlled per event payload:
- `execution_mode=plan_execute` (default flow)
- `execution_mode=ralph_story` (RALPH backlog flow from `prd.json`)
For `ralph_story`, planning is always forced through a structured planner path (multi-step), even if global
`PLANNER_MODE=codex_only`.
For non-interactive orchestration use `CODEX_CLI_CMD="codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check"` (default in compose).
If you want Codex CLI to use a fixed model, set `CODEX_MODEL` (example: `gpt-5.3-codex-spark`).
If you want Planner to use API key but Codex to use only `codex login` session, set:
`CODEX_USE_OPENAI_API_KEY=false`.
If shell Python steps fail with missing imports (for example `torch`), runner can auto-install once:
`AUTO_INSTALL_MISSING_PYTHON_MODULES=true` (enabled by default).
DDD use-case layer toggle (currently on by default):
`DDD_USE_CASES_ENABLED=true`.
If a Python script fails with `argparse` unknown flags (for example `--smoke_test`), runner auto-tries:
flag normalization (`--smoke-test`) and then drops unknown flags once.
On non-infrastructure execution failures, orchestrator can auto-replan the run instead of immediate stop:
`EXECUTION_FAILURE_REPLAN_ENABLED=true`, bounded by `MAX_EXECUTION_REPLANS` (default: 1).
After each successful step, the runtime also performs a lightweight plan-contract review (expected artifacts / stop condition hints). If the review fails:
- `PLAN_REVIEW_ENABLED=true` keeps the check active,
- `PLAN_REVIEW_MANUAL_FALLBACK=true` puts run into `WAITING_PLAN_REVIEW` so you can approve with `/control/approve`.
Contract strictness is configurable:
- `CONTRACT_STRICTNESS=free` (minimal checks),
- `CONTRACT_STRICTNESS=balanced` (default),
- `CONTRACT_STRICTNESS=strict` (strong artifact/intent checks).
Planner steps support typed `expected_artifacts` (`path/kind/must_exist/must_be_nonempty/metric_keys`) and `step_intent`.
For codex steps the runner appends a hard execution tail to force file writes; if exactly one expected file is missing but stdout contains a code block, it is auto-materialized and marked `AUTO_REPAIRED`.
For RALPH mode:
- PRD file path: `workspace/<workspace_id>/prd.json` (override with `RALPH_PRD_FILE_NAME`)
- Progress log path: `workspace/<workspace_id>/progress.txt` (override with `RALPH_PROGRESS_FILE_NAME`)
- RALPH run always starts with `ralph-prd-bootstrap` step, where Codex builds/updates `prd.json` from the task goal and quality requirements before selecting the next story.
- RALPH requires explicit quality target (for example in goal/constraints: `RALPH_REQUIRED_METRIC: test_accuracy >= 0.95`); without it run moves to `WAITING_PLAN_REVIEW`.
- After bootstrap, story planning is planner-driven (LLM/stub planner), not hardcoded single-step Codex.
- If bootstrap creates no pending stories, run fails with `no pending Ralph stories...`.
- Bootstrap failures are handled by normal execution replan flow (`MAX_EXECUTION_REPLANS`).
- If a story passes verification, orchestrator marks it `passes=true` and can enqueue next story when
  `RALPH_AUTO_QUEUE_NEXT_STORY=true` (off by default).
- Ready-to-use data scientist template is available at `workspace/demo/prd.json`.
Minimal `prd.json` shape:
```json
{
  "userStories": [
    {
      "id": "US-1",
      "name": "Train baseline model",
      "description": "Implement MNIST baseline training with metrics.",
      "priority": 1,
      "passes": false,
      "acceptanceCriteria": [
        "training script exists",
        "metrics file is produced"
      ]
    }
  ]
}
```

3. Run service:
```bash
uvicorn orchestrator.app.main:app --host 0.0.0.0 --port 8080
```

4. Submit task (thin HTTP path):
```bash
curl -X POST http://localhost:8080/event \
  -H 'content-type: application/json' \
  -d '{"workspace_id":"demo","goal":"make a small change and validate","execution_mode":"plan_execute"}'
```
RALPH mode example:
```bash
curl -X POST http://localhost:8080/event \
  -H 'content-type: application/json' \
  -d '{"workspace_id":"demo","goal":"Execute next RALPH story","execution_mode":"ralph_story"}'
```

## Run with Docker Compose
```bash
docker compose up --build
```

Streamlit UI will be available at `http://localhost:8501`.

## Codex In Container (login + 2FA)
1. Rebuild the image so `codex` is installed:
```bash
docker compose down
docker compose build --no-cache --progress=plain orchestrator
docker compose up -d
```

2. Verify CLI exists:
```bash
docker compose exec -it orchestrator sh -lc "which codex && codex --version"
```

3. Login interactively (password + 2FA supported):
```bash
docker compose exec -it orchestrator sh -lc "codex login"
```

Auth is persisted in `./codex-home` (mounted to `/root/.codex` and used as `CODEX_HOME`/`CODEX_CONFIG_HOME`), so you should not need to login again after container restart.

If `OPENAI_API_KEY` is set, orchestrator also performs a one-time background bootstrap with:
`codex login --with-api-key` before the first `codex exec`.
Set `CODEX_USE_OPENAI_API_KEY=false` to disable this behavior.

If you see `401 Unauthorized: Missing bearer or basic authentication`, verify inside container:
```bash
docker compose exec -it orchestrator sh -lc 'echo "OPENAI_API_KEY=${OPENAI_API_KEY:+set}" && echo "CODEX_HOME=$CODEX_HOME" && ls -la $CODEX_HOME'
```

If Codex logs show `Sandbox(LandlockRestrict)` and says it could not write files, do not use `--full-auto` in `CODEX_CLI_CMD`.
Use:
`codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check`.

## API
- `GET /healthz`
- `GET /readyz`
- `POST /event`
- `GET /status/{run_id}`
- `GET /artifacts/{run_id}`
- `GET /runs/{run_id}`
- `GET /tasks/{task_id}`
- `POST /control/approve`
- `POST /control/cancel`

## Streamlit UI
Run locally:
```bash
streamlit run streamlit_app.py
```

Run in compose:
```bash
docker compose up --build
```
Then open `http://localhost:8501`.

UI supports:
- task submission (`/event`)
- live run monitoring with auto-refresh and step feed
- approve/cancel controls
- artifacts and logs preview

## Manual Pilot
```bash
python pilot.py --goal "fix failing tests in workspace" --workspace-id demo --auto-approve
```
Pilot writes incremental state into `SESSION_JSONL_PATH` (default: `./workspace/session.jsonl`).

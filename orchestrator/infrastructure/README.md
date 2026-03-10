# Infrastructure Adapters

This project uses a pragmatic hexagonal layout.

Current adapter implementations:

- `orchestrator/persistence/*` — SQLite repository adapter
- `orchestrator/runtime/bus.py` — Redis/InMemory event bus adapter
- `orchestrator/planning/*` — planner adapters (OpenAI/stub/codex-only)
- `orchestrator/execution/*` — executor/policy/verifier/artifact adapters
- `workspace/knowledge/*` — runtime knowledge artifacts (skills, notes, improvement strategies)

Application and domain layers must depend on ports, not concrete adapter internals.

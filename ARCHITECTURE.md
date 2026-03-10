# Orchestrator Architecture (DDD + Hexagonal, Pragmatic Core)

## Layers

- `orchestrator/domain`
  - Pure business rules and invariants.
  - No imports from runtime/planning/execution/persistence/knowledge.
- `orchestrator/application`
  - Use-cases and ports.
  - Orchestrates domain rules through abstract ports.
  - Key services:
    - `application/services/plan_contract_service.py`
    - `application/services/recovery_service.py`
    - `application/services/quality_gate_service.py`
- `orchestrator/infrastructure`
  - Adapter documentation and wrappers.
  - Concrete adapters currently live in existing packages.
- `orchestrator/runtime`
  - Composition root and compatibility facades.
- `orchestrator/app`
  - API layer (FastAPI endpoints).

## Where to change what

- Plan generation behavior: `orchestrator/planning/planner.py`
- Codex execution behavior: `orchestrator/execution/codex_runner.py`
- Policy rules: `orchestrator/execution/policy.py`
- Domain status transitions: `orchestrator/domain/state_machine.py`
- Run orchestration use-case: `orchestrator/application/use_cases/process_run_tick.py`
- API contracts/endpoints: `orchestrator/app/api.py`

## Compatibility

- HTTP endpoints and stream payloads are unchanged.
- Existing runtime entrypoint is unchanged: `orchestrator.app.main:app`.
- `orchestrator/runtime/session.py` remains as compatibility facade.

# Orchestrator Architecture

## Overview

The runtime is now stage-based.

`ProcessRunTickUseCase` is no longer the place where business logic lives inline. It is a thin coordinator that:

1. loads the current `run` and `task`
2. performs goal-signature drift checks
3. delegates stepio recovery
4. dispatches control to stage handlers
5. publishes terminal results

The actual run logic is split into small stage and helper services under:

- `orchestrator/application/use_cases/run_tick`

## Layers

### `orchestrator/domain`
- Pure business rules and invariants.
- No knowledge of runtime, Codex, SQLite, Redis, or planner prompts.
- Main file:
  - `orchestrator/domain/state_machine.py`

### `orchestrator/application`
- Use cases and orchestration services.
- Owns run lifecycle decisions, verification persistence flow, planning context assembly, and stage dispatch.

Main entrypoint:
- `orchestrator/application/use_cases/process_run_tick.py`

Run tick support modules:
- `orchestrator/application/use_cases/run_tick/coordinator_support.py`
- `orchestrator/application/use_cases/run_tick/context_stage.py`
- `orchestrator/application/use_cases/run_tick/planning_stage.py`
- `orchestrator/application/use_cases/run_tick/execution_stage.py`
- `orchestrator/application/use_cases/run_tick/verification_stage.py`
- `orchestrator/application/use_cases/run_tick/packaging_stage.py`
- `orchestrator/application/use_cases/run_tick/stepio_recovery.py`
- `orchestrator/application/use_cases/run_tick/planning_context.py`
- `orchestrator/application/use_cases/run_tick/verification_flow.py`
- `orchestrator/application/use_cases/run_tick/execution_guards.py`
- `orchestrator/application/use_cases/run_tick/hyperparameters.py`

Application services used by stages:
- `orchestrator/application/services/plan_contract_service.py`
- `orchestrator/application/services/recovery_service.py`
- `orchestrator/application/services/quality_gate_service.py`
- `orchestrator/application/services/improvement_strategy_service.py`
- `orchestrator/application/services/ralph_service.py`
- `orchestrator/application/services/workspace_snapshot_service.py`

### `orchestrator/planning`
- Plan generation and Ralph backlog logic.
- Main files:
  - `orchestrator/planning/planner.py`
  - `orchestrator/planning/planner_sanitizer.py`
  - `orchestrator/planning/stub_plan_support.py`
  - `orchestrator/planning/ralph.py`

### `orchestrator/execution`
- Concrete execution adapters and result extraction.
- Main files:
  - `orchestrator/execution/codex_runner.py`
  - `orchestrator/execution/codex_cli.py`
  - `orchestrator/execution/codex_prompting.py`
  - `orchestrator/execution/shell_command_normalizer.py`
  - `orchestrator/execution/runner_models.py`
  - `orchestrator/execution/subprocess_support.py`
  - `orchestrator/execution/stepio_support.py`
  - `orchestrator/execution/command_recovery.py`
  - `orchestrator/execution/command_sequence.py`
  - `orchestrator/execution/workspace_metrics_reader.py`
  - `orchestrator/execution/intent_validation.py`
  - `orchestrator/execution/policy.py`
  - `orchestrator/execution/verifier.py`
  - `orchestrator/execution/artifacts.py`

### `orchestrator/persistence`
- SQLite state store and schemas.
- Main files:
  - `orchestrator/persistence/db.py`
  - `orchestrator/persistence/common.py`
  - `orchestrator/persistence/event_repository.py`
  - `orchestrator/persistence/task_repository.py`
  - `orchestrator/persistence/run_repository.py`
  - `orchestrator/persistence/run_record_mapper.py`
  - `orchestrator/persistence/run_history_repository.py`
  - `orchestrator/persistence/experiment_attempts_repository.py`
  - `orchestrator/persistence/policy_decisions_repository.py`
  - `orchestrator/persistence/maintenance_repository.py`
  - `orchestrator/persistence/pdf_repository.py`
  - `orchestrator/persistence/verification_payloads.py`
  - `orchestrator/persistence/schema_sql.py`
  - `orchestrator/persistence/schema_manager.py`
  - `orchestrator/persistence/sqlite_recovery.py`
  - `orchestrator/persistence/artifact_recovery.py`
  - `orchestrator/persistence/schemas.py`

### `orchestrator/runtime`
- Composition root and runtime loops.
- Main files:
  - `orchestrator/runtime/service.py`
  - `orchestrator/runtime/session.py`

### `orchestrator/app`
- FastAPI API layer.
- Main file:
  - `orchestrator/app/api.py`

## Runtime Composition

The runtime is wired in:

- `orchestrator/runtime/service.py`

`OrchestratorService` constructs:

1. `Database`
2. `RedisEventBus`
3. `Planner`
4. `PolicyEngine`
5. `RalphBacklogService`
6. `CodexRunner`
7. `Verifier`
8. `ArtifactPublisher`
9. `SessionManager`

`SessionManager` remains a compatibility facade over:

- `ProcessRunTickUseCase`

The composition root is intentionally thin:

- `OrchestratorService` keeps live runtime handles only for:
  - `Database`
  - `RedisEventBus`
  - `SessionManager`
- `SessionManager` keeps use-case handles only:
  - `ProcessRunTickUseCase`
  - `SubmitTaskUseCase`
  - `HandleControlEventUseCase`
  - `FinalizeRunUseCase`

Planner, policy, verifier, Ralph backlog, and artifact publisher are now wiring-time dependencies, not long-lived runtime state exposed on the facade.

The runtime loops are still:

1. task gateway
2. control gateway
3. dispatcher
4. worker pool
5. retention loop

## Run Tick Structure

### Coordinator

`orchestrator/application/use_cases/process_run_tick.py`

Responsibilities:

1. lock the run with a per-run mutex
2. load `run` and `task`
3. perform goal signature drift reset
4. invoke stepio reconciliation
5. dispatch to stage handlers in order
6. handle top-level exceptions

### Stage Handlers

#### `context_stage.py`
Handles:
- `WAITING_APPROVAL`
- `WAITING_PLAN_REVIEW`
- `RECEIVED`
- `PLAN_READY`

Typical transitions:
- `WAITING_APPROVAL -> CONTEXT_READY`
- `WAITING_PLAN_REVIEW -> CONTEXT_READY`
- `RECEIVED -> CONTEXT_READY`
- `PLAN_READY -> EXECUTING`

#### `planning_stage.py`
Handles:
- `CONTEXT_READY`

Responsibilities:
- build plan context
- ask Ralph for story-specific plans
- fallback to planner build/replan
- attach selected skills
- apply quality-plan guard
- apply policy decisions for the whole plan
- transition to:
  - `WAITING_PLAN_REVIEW`
  - `WAITING_APPROVAL`
  - `PLAN_READY`
  - `FAILED`

#### `execution_stage.py`
Handles:
- `EXECUTING`

Responsibilities:
- per-step policy checks
- per-step approval gate
- synthetic smoke guard
- step execution via `CodexRunner`
- missing-file recovery
- soft threshold failure normalization
- plan-contract checks
- execution replans
- story-cycle advance
- transition to:
  - `WAITING_APPROVAL`
  - `WAITING_PLAN_REVIEW`
  - `CONTEXT_READY`
  - `VERIFYING`
  - `FAILED`

#### `verification_stage.py`
Handles:
- `VERIFYING`

Responsibilities:
- run `Verifier`
- attach hyperparameter context
- evaluate quality gate
- build improvement strategy
- persist verification artifacts
- record experiment attempts
- schedule quality replans
- transition to:
  - `CONTEXT_READY`
  - `PACKAGING`
  - `FAILED`

#### `packaging_stage.py`
Handles:
- `PACKAGING`

Responsibilities:
- compute final summary
- package artifacts
- publish result event
- run Ralph completion hook
- release workspace lock
- transition to:
  - `COMPLETED`
  - `FAILED`

### `coordinator_support.py`
Purpose:
- state transition publishing
- workspace path resolution
- task submission event persistence
- control event handling
- cancellation finalization
- result event publishing
- execution-cycle replan scheduling

## Run Tick Helper Services

### `planning_context.py`
Purpose:
- build `PlanInput`
- compact previous verification
- summarize experiment history
- attach previously selected skill paths
- compute task goal signatures

### `verification_flow.py`
Purpose:
- persist `verification.latest.json`
- persist `verification.attempt_N.json`
- persist `improvement_strategy.latest.json`
- write experiment history artifacts
- write experiment attempts to SQLite
- parse `max_quality_retries`

### `execution_guards.py`
Purpose:
- manual approval checks
- plan contract evaluation
- missing-file recovery
- synthetic smoke guard
- planning-only quality skip
- quality-plan guard
- execution failure formatting

### `stepio_recovery.py`
Purpose:
- reconcile `*.step_result*.json` into `run_steps`
- ignore stale artifacts from previous execution cycles
- recover `next_step_index`
- recover `EXECUTING -> VERIFYING` when stepio already proves completion

### `hyperparameters.py`
Purpose:
- extract hyperparameters from shell commands
- build verification hyperparameter history context

## Current Control Flow

For one run, the effective flow is:

1. `ProcessRunTickUseCase.process_run(run_id)`
2. load run/task
3. goal signature drift check
4. `StepioRecoveryService.reconcile_stepio_artifacts(...)`
5. `StepioRecoveryService.sync_run_progress_from_stepio(...)`
6. `RunContextStage.handle_waiting_status(...)`
7. `RunContextStage.handle_received(...)`
8. `RunPlanningStage.handle_context_ready(...)`
9. `RunContextStage.handle_plan_ready(...)`
10. `RunExecutionStage.handle_executing(...)`
11. `RunVerificationStage.handle_verifying(...)`
12. `RunPackagingStage.handle_packaging(...)`

The coordinator loop repeats until the run becomes terminal or waits for user input.

## Where To Change What

- Plan generation behavior:
  - `orchestrator/planning/planner.py`
  - `orchestrator/planning/planner_sanitizer.py`
  - `orchestrator/planning/stub_plan_support.py`

- Ralph story / PRD behavior:
  - `orchestrator/application/services/ralph_service.py`
  - `orchestrator/planning/ralph.py`

- Run planning orchestration:
  - `orchestrator/application/use_cases/run_tick/planning_stage.py`
  - `orchestrator/application/use_cases/run_tick/planning_context.py`

- Step execution orchestration:
  - `orchestrator/application/use_cases/run_tick/execution_stage.py`

- Codex execution behavior:
  - `orchestrator/execution/codex_runner.py`
  - `orchestrator/execution/codex_cli.py`
  - `orchestrator/execution/codex_prompting.py`
  - `orchestrator/execution/shell_command_normalizer.py`
  - `orchestrator/execution/runner_models.py`
  - `orchestrator/execution/subprocess_support.py`
  - `orchestrator/execution/stepio_support.py`
  - `orchestrator/execution/command_recovery.py`
  - `orchestrator/execution/command_sequence.py`
  - `orchestrator/execution/workspace_metrics_reader.py`
  - `orchestrator/execution/intent_validation.py`

- Policy rules:
  - `orchestrator/execution/policy.py`
  - `orchestrator/application/use_cases/run_tick/execution_guards.py`

- Verification and quality gate behavior:
- `orchestrator/application/use_cases/run_tick/verification_stage.py`
- `orchestrator/application/use_cases/run_tick/verification_flow.py`
- `orchestrator/application/services/quality_gate_service.py`
- `orchestrator/execution/verifier.py`

- Stepio recovery behavior:
  - `orchestrator/application/use_cases/run_tick/stepio_recovery.py`

- Domain status transitions:
  - `orchestrator/domain/state_machine.py`

- API contracts/endpoints:
  - `orchestrator/app/api.py`

- Runtime wiring:
- `orchestrator/runtime/service.py`

- SQLite helper/data shaping:
  - `orchestrator/persistence/common.py`
  - `orchestrator/persistence/verification_payloads.py`
  - `orchestrator/persistence/schema_sql.py`
  - `orchestrator/persistence/schema_manager.py`
  - `orchestrator/persistence/sqlite_recovery.py`
  - `orchestrator/persistence/artifact_recovery.py`
  - `orchestrator/persistence/run_record_mapper.py`
  - `orchestrator/persistence/event_repository.py`
  - `orchestrator/persistence/task_repository.py`
  - `orchestrator/persistence/run_repository.py`
  - `orchestrator/persistence/run_history_repository.py`
  - `orchestrator/persistence/experiment_attempts_repository.py`
  - `orchestrator/persistence/policy_decisions_repository.py`
  - `orchestrator/persistence/maintenance_repository.py`
  - `orchestrator/persistence/pdf_repository.py`

## Compatibility

- HTTP endpoints and event payloads are unchanged.
- Runtime entrypoint is unchanged.
- `orchestrator/runtime/session.py` still exists as compatibility facade.
- The main refactor is internal: stage logic moved out of `process_run_tick.py` into `run_tick/` modules.

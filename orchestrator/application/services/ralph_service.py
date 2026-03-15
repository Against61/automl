from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from orchestrator.application.services.quality_gate_service import QualityGateService
from orchestrator.application.services.workspace_snapshot_service import WorkspaceSnapshotService
from orchestrator.config import Settings
from orchestrator.execution.verifier import VerificationResult
from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import PlannerPlan, Priority, RetrievedContext, RunRecord, RunResultSummary
from orchestrator.planning.planner import CodexOnlyPlanner, PlanInput, Planner, StubPlanner
from orchestrator.planning.ralph import RalphBacklogError, RalphBacklogService, RalphStory
from orchestrator.runtime.bus import InMemoryEventBus, RedisEventBus

logger = logging.getLogger(__name__)

EventBus = RedisEventBus | InMemoryEventBus

_RALPH_WORKSPACE_SNAPSHOT_FILE = ".ralph_workspace_tree.md"


class RalphScenarioService:
    def __init__(
        self,
        *,
        settings: Settings,
        backlog: RalphBacklogService,
        planner: Planner,
        db: Database,
        bus: EventBus,
        quality_gate_service: QualityGateService,
    ):
        self.settings = settings
        self.backlog = backlog
        self.planner = planner
        self.db = db
        self.bus = bus
        self.quality_gate_service = quality_gate_service
        self.workspace_snapshot_service = WorkspaceSnapshotService()

    def resolve_next_story(self, workspace_path: Path, explicit_story_id: str | None = None) -> RalphStory | None:
        prd = self.backlog.load_prd(workspace_path)
        story = None
        if explicit_story_id:
            story = self.backlog.pick_by_id(prd, explicit_story_id)
        if story is None:
            story = self.backlog.pick_next_story(prd)
        return story

    async def has_completed_prd_bootstrap_step(self, run_id: str) -> bool:
        return await self.db.has_completed_step(run_id, self.ralph_prd_bootstrap_step_id())

    def ralph_prd_bootstrap_step_id(self) -> str:
        return "ralph-prd-bootstrap"

    def is_prd_bootstrap_step(self, step_id: str) -> bool:
        return step_id == self.ralph_prd_bootstrap_step_id()

    def planner_for_ralph(self) -> Planner:
        if isinstance(self.planner, CodexOnlyPlanner):
            return StubPlanner()
        return self.planner

    def execution_mode(self, task: dict[str, Any]) -> str:
        try:
            payload = json.loads(task.get("payload_json") or "{}")
        except json.JSONDecodeError:
            return "plan_execute"
        if not isinstance(payload, dict):
            return "plan_execute"
        inner = payload.get("payload")
        if not isinstance(inner, dict):
            return "plan_execute"
        mode = str(inner.get("execution_mode", "plan_execute")).strip().lower()
        if mode not in {"plan_execute", "ralph_story"}:
            return "plan_execute"
        return mode

    def is_ralph_task(self, task: dict[str, Any]) -> bool:
        return self.execution_mode(task) == "ralph_story"

    async def maybe_build_plan(
        self,
        task: dict[str, Any],
        run: RunRecord,
        contexts: list[RetrievedContext],
        workspace_path: Path,
        last_failed_step: dict[str, Any] | None,
        previous_verification: dict[str, Any] | None = None,
        experiment_history_summary: str | None = None,
    ) -> PlannerPlan | None:
        if not self.is_ralph_task(task):
            return None
        return await self.build_next_plan(
            task=task,
            run=run,
            contexts=contexts,
            workspace_path=workspace_path,
            last_failed_step=last_failed_step,
            previous_verification=previous_verification,
            experiment_history_summary=experiment_history_summary,
        )

    def ralph_quality_requirement_missing_reason(self, task: dict[str, Any], workspace_path: Path) -> str | None:
        if not self.is_ralph_task(task):
            return None
        requirement = self.quality_gate_service.extract_requirement(
            task=task,
            workspace_path=workspace_path,
        )
        if requirement:
            return None
        return (
            "ralph_story requires an explicit quality target "
            "(for example: RALPH_REQUIRED_METRIC: test_accuracy >= 95%)."
        )

    async def maybe_handle_successful_step(
        self,
        task: dict[str, Any],
        step_id: str,
        workspace_path: Path,
    ) -> tuple[bool, str]:
        if not self.is_ralph_task(task):
            return False, ""
        return await self.handle_successful_step(
            step_id=step_id,
            workspace_path=workspace_path,
        )

    async def maybe_evaluate_quality_gate(
        self,
        task: dict[str, Any],
        workspace_path: Path,
        verification: VerificationResult,
        previous_verification: dict[str, Any] | None = None,
    ) -> tuple[bool, str] | None:
        # quality gate is intentionally common for plan_execute/ralph_story paths.
        # `verification` + goal/constraints can still produce no explicit requirement.
        return await self.evaluate_quality_gate(
            task=task,
            workspace_path=workspace_path,
            verification=verification,
            previous_verification=previous_verification,
        )

    async def maybe_handle_completion(
        self,
        run: RunRecord,
        task: dict[str, Any],
        summary: RunResultSummary,
        workspace_path: Path,
    ) -> None:
        if not self.is_ralph_task(task):
            return
        await self.handle_story_completion(
            run=run,
            task=task,
            summary=summary,
            workspace_path=workspace_path,
        )

    async def build_next_plan(
        self,
        task: dict[str, Any],
        run: RunRecord,
        contexts: list[RetrievedContext],
        workspace_path: Path,
        last_failed_step: dict[str, Any] | None,
        previous_verification: dict[str, Any] | None = None,
        experiment_history_summary: str | None = None,
    ) -> PlannerPlan:
        should_bootstrap_prd = not await self.has_completed_prd_bootstrap_step(run.run_id)
        if should_bootstrap_prd:
            return await self.build_prd_bootstrap_plan(
                task=task,
                run=run,
                contexts=contexts,
                workspace_path=workspace_path,
                last_failed_step=last_failed_step,
                previous_verification=previous_verification,
                experiment_history_summary=experiment_history_summary,
            )
        return await self.build_story_plan(
            task=task,
            run=run,
            contexts=contexts,
            workspace_path=workspace_path,
            last_failed_step=last_failed_step,
            previous_verification=previous_verification,
            experiment_history_summary=experiment_history_summary,
        )

    async def handle_successful_step(self, step_id: str, workspace_path: Path) -> tuple[bool, str]:
        if not self.is_prd_bootstrap_step(step_id):
            return False, ""
        await self.annotate_prd_with_workspace_snapshot(workspace_path=workspace_path)
        return True, "RALPH PRD bootstrap completed"

    def extract_primary_user_goal(self, task: dict[str, Any], constraints: list[str] | None = None) -> str | None:
        prefix = "PRIMARY_USER_GOAL:"
        values = constraints if constraints is not None else json.loads(task["constraints_json"])
        for raw in values:
            value = str(raw).strip()
            if value.startswith(prefix):
                extracted = value[len(prefix) :].strip()
                return extracted or None

        raw_goal = str(task.get("goal", "")).strip()
        if raw_goal and not raw_goal.startswith("[RALPH "):
            return raw_goal
        return None

    def extract_story_id(self, task: dict[str, Any], constraints: list[str] | None = None) -> str | None:
        prefix = "RALPH_STORY_ID:"
        values = constraints if constraints is not None else json.loads(task["constraints_json"])
        for raw in values:
            value = str(raw).strip()
            if value.startswith(prefix):
                extracted = value[len(prefix) :].strip()
                return extracted or None

        raw_goal = str(task.get("goal", "")).strip()
        if raw_goal.startswith("[RALPH "):
            marker = raw_goal[len("[RALPH ") :]
            story_id = marker.split("]", 1)[0].strip()
            return story_id or None
        return None

    def has_primary_goal_constraint(self, constraints: list[str]) -> bool:
        return any(str(item).strip().startswith("PRIMARY_USER_GOAL:") for item in constraints)

    def _ralph_workspace_snapshot_path(self, workspace_path: Path) -> Path:
        return self.workspace_snapshot_service.ralph_snapshot_path(workspace_path)

    def _build_workspace_tree_snapshot(self, workspace_path: Path, max_depth: int = 4) -> str:
        payload = self.workspace_snapshot_service.refresh(workspace_path)
        return str(payload.get("tree") or "")

    async def annotate_prd_with_workspace_snapshot(self, workspace_path: Path) -> None:
        prd_path = workspace_path / self.settings.ralph_prd_file_name
        if not prd_path.is_file():
            return
        try:
            payload = json.loads(prd_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(payload, dict):
            return

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
            payload["meta"] = meta

        meta["workspace_snapshot"] = {
            "path": self._ralph_workspace_snapshot_path(workspace_path).name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace_root": workspace_path.name,
        }

        try:
            prd_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("failed to annotate PRD with workspace snapshot: %s", prd_path)

    async def build_prd_bootstrap_plan(
        self,
        task: dict[str, Any],
        run: RunRecord,
        contexts: list[RetrievedContext],
        workspace_path: Path,
        last_failed_step: dict[str, Any] | None,
        previous_verification: dict[str, Any] | None = None,
        experiment_history_summary: str | None = None,
    ) -> PlannerPlan:
        constraints = [str(item).strip() for item in json.loads(task["constraints_json"]) if str(item).strip()]
        primary_goal = self.extract_primary_user_goal(task, constraints=constraints)
        primary_goal_block = primary_goal or "not provided"
        quality_requirement = self.quality_gate_service.extract_requirement(task=task, workspace_path=workspace_path)
        task_intent = self.quality_gate_service.infer_task_intent(task=task, workspace_path=workspace_path)
        quality_line = "not provided"
        if quality_requirement:
            metric = quality_requirement.get("metric_key", "metric")
            operator = quality_requirement.get("operator", ">=")
            target = quality_requirement.get("target", "value")
            unit = quality_requirement.get("unit", "ratio")
            quality_line = f"{metric} {operator} {target} ({unit})"
        intent_block = (
            f"task_family={task_intent.task_family}; "
            f"primary_metric={task_intent.primary_metric_key or 'not set'}; "
            f"preferred_metrics={', '.join(task_intent.preferred_metrics) or 'none'}; "
            f"real_dataset_smoke_required={'true' if task_intent.requires_real_dataset_smoke else 'false'}"
        )

        previous_error = run.error_message or "none"
        experiment_history_block = experiment_history_summary or "none"
        failed_step_block = (
            json.dumps(last_failed_step, ensure_ascii=True, indent=2) if last_failed_step else "none"
        )
        verification_block = previous_verification or {}
        if isinstance(verification_block, dict) and verification_block:
            verification_block = json.dumps(verification_block, ensure_ascii=True, indent=2)
        else:
            verification_block = "none"
        constraints_block = "\n".join(f"- {c}" for c in constraints if c.strip()) or "- none"
        context_lines: list[str] = []
        for idx, item in enumerate(contexts[:6], start=1):
            snippet = " ".join(item.snippet.split())
            if len(snippet) > 600:
                snippet = f"{snippet[:600]}..."
            context_lines.append(
                f"{idx}. {item.document_path}:p{item.page_number} (score={item.confidence:.3f}) -> {snippet}"
            )
        context_block = "\n".join(context_lines) if context_lines else "- no retrieved context"
        workspace_tree_snapshot = self._build_workspace_tree_snapshot(workspace_path)

        bootstrap_prompt = (
            "Create or update a Ralph PRD file in workspace.\n\n"
            f"Workspace PRD file: `{self.settings.ralph_prd_file_name}`\n"
            f"Workspace tree snapshot:\n{workspace_tree_snapshot}\n\n"
            "Use this goal and constraints to generate a concrete backlog with user stories.\n\n"
            f"Goal:\n{task['goal']}\n\n"
            f"Primary user goal:\n{primary_goal_block}\n\n"
            f"Quality requirement:\n{quality_line}\n\n"
            f"Inferred task intent:\n{intent_block}\n\n"
            f"Constraints:\n{constraints_block}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Experiment history:\n{experiment_history_block}\n\n"
            f"Previous execution error:\n{previous_error}\n\n"
            f"Last failed step snapshot:\n{failed_step_block}\n\n"
            f"Previous verification context:\n{verification_block}\n\n"
            "Output contract:\n"
            "- If PRD exists, preserve existing metadata/version fields.\n"
            "- Add `meta.workspace_snapshot` with file name and generation timestamp (use path .ralph_workspace_tree.md).\n"
            "- Use only existing paths from the provided workspace tree unless a step explicitly asks to create a new artifact.\n"
            "- Treat the current working directory as the workspace root; do not generate shell paths prefixed with the workspace_id directory name.\n"
            "- Ensure at least one user story is present with `passes: false`.\n"
            "- Set acceptance criteria that directly supports the goal and quality requirement.\n"
            "- Keep successMetrics and acceptanceCriteria aligned with the inferred task intent and preferred metric family.\n"
            "- For any smoke-test story, require a disjoint evaluation split and explicitly forbid reusing the exact training subset as the reported acceptance metric.\n"
            "- For any training story, require reported quality metrics to come from a held-out or otherwise disjoint evaluation split.\n"
            "- Do not start training or execute model code yet; only prepare PRD.\n"
            "- Save the file as valid JSON at the path above.\n"
            "- Print a short summary of created/updated sections in stdout.\n"
            "- Include explicit fields: `meta`, `globalConstraints`, `successMetrics`, `userStories`.\n"
            "- For each story in `userStories`, use keys:\n"
            "  - id, name, description, priority, passes, acceptanceCriteria.\n"
            "- Keep the result deterministic and no placeholders.\n"
        )
        return PlannerPlan(
            version="1.0",
            summary="Bootstrap PRD for RALPH story execution",
            steps=[
                {
                    "id": self.ralph_prd_bootstrap_step_id(),
                    "title": "Create/update RALPH PRD",
                    "step_type": "change",
                    "operation": "edit_code",
                    "step_intent": "modify_file",
                    "commands": [],
                    "codex_prompt": bootstrap_prompt,
                    "expected_artifacts": [
                        {
                            "path": self.settings.ralph_prd_file_name,
                            "kind": "file",
                            "must_exist": True,
                            "must_be_nonempty": True,
                        },
                        {
                            "path": _RALPH_WORKSPACE_SNAPSHOT_FILE,
                            "kind": "file",
                            "must_exist": True,
                        },
                        {"path": None, "kind": "generic", "must_exist": False},
                    ],
                    "stop_condition": "PRD exists in workspace and contains at least one pending story",
                    "action": "codex",
                    "instruction": bootstrap_prompt,
                    "command": None,
                    "risk_level": "medium",
                }
            ],
        )

    async def build_story_plan(
        self,
        task: dict[str, Any],
        run: RunRecord,
        contexts: list[RetrievedContext],
        workspace_path: Path,
        last_failed_step: dict[str, Any] | None,
        previous_verification: dict[str, Any] | None = None,
        experiment_history_summary: str | None = None,
    ) -> PlannerPlan:
        prd = self.backlog.load_prd(workspace_path)
        story = self.backlog.pick_next_story(prd)
        if story is None:
            raise RalphBacklogError(
                f"no pending Ralph stories in prd={self.settings.ralph_prd_file_name}; "
                "create a new user story or submit a plan_execute task"
            )

        constraints = [str(item).strip() for item in json.loads(task["constraints_json"]) if str(item).strip()]
        constraints = [item for item in constraints if not item.startswith("RALPH_STORY_ID:")]
        primary_goal = self.extract_primary_user_goal(task, constraints=constraints)
        if primary_goal and not self.has_primary_goal_constraint(constraints):
            constraints.append(f"PRIMARY_USER_GOAL: {primary_goal}")
        constraints.append(f"RALPH_STORY_ID: {story.story_id}")
        task_intent = self.quality_gate_service.infer_task_intent(
            task=task,
            workspace_path=workspace_path,
            story_id=story.story_id,
        )
        for inferred_constraint in task_intent.as_constraints():
            prefix = inferred_constraint.split(":", 1)[0].strip()
            if not any(str(item).strip().startswith(f"{prefix}:") for item in constraints):
                constraints.append(inferred_constraint)
        story_requirement = self.quality_gate_service.extract_requirement(
            task=task,
            workspace_path=workspace_path,
            story_id=story.story_id,
        )
        if story_requirement:
            req_unit = story_requirement.get("unit", "ratio")
            req_target = story_requirement.get("target")
            req_metric = story_requirement.get("metric_key")
            req_operator = story_requirement.get("operator")
            constraints.append(f"RALPH_REQUIRED_METRIC: {req_metric} {req_operator} {req_target} ({req_unit})")
        ralph_guard = "RALPH mode: do not create commits or push in MVP."
        if ralph_guard not in constraints:
            constraints.append(ralph_guard)

        context_lines: list[str] = []
        for idx, item in enumerate(contexts[:6], start=1):
            snippet = " ".join(item.snippet.split())
            if len(snippet) > 600:
                snippet = f"{snippet[:600]}..."
            context_lines.append(
                f"{idx}. {item.document_path}:p{item.page_number} (score={item.confidence:.3f}) -> {snippet}"
            )
        context_block = "\n".join(context_lines) if context_lines else "- no retrieved context"
        constraints_block = "\n".join(f"- {c}" for c in constraints if c.strip()) or "- none"
        acceptance_block = (
            "\n".join(f"- {item}" for item in story.acceptance_criteria)
            if story.acceptance_criteria
            else "- not specified"
        )
        workspace_tree_snapshot = self._build_workspace_tree_snapshot(workspace_path)
        previous_error = run.error_message or "none"
        experiment_history_block = experiment_history_summary or "none"
        primary_goal_block = primary_goal or "not provided"
        verification_block = previous_verification or {}
        if isinstance(verification_block, dict) and verification_block:
            verification_block = json.dumps(verification_block, ensure_ascii=True, indent=2)
        else:
            verification_block = "none"
        intent_block = (
            f"- task_family: {task_intent.task_family}\n"
            f"- primary_metric_key: {task_intent.primary_metric_key or 'not set'}\n"
            f"- preferred_metrics: {', '.join(task_intent.preferred_metrics) or 'none'}\n"
            f"- real_dataset_smoke_required: {'true' if task_intent.requires_real_dataset_smoke else 'false'}\n"
            f"- evidence: {', '.join(task_intent.evidence) or 'none'}"
        )
        augmented_goal = (
            f"Implement RALPH story {story.story_id}: {story.title}\n\n"
            f"Primary user goal (highest priority):\n{primary_goal_block}\n\n"
            f"Story description:\n{story.description}\n\n"
            f"Acceptance criteria:\n{acceptance_block}\n\n"
            f"Inferred task intent:\n{intent_block}\n\n"
            f"Task constraints:\n{constraints_block}\n\n"
            f"Retrieved context:\n{context_block}\n\n"
            f"Workspace tree snapshot:\n{workspace_tree_snapshot}\n\n"
            f"Experiment history:\n{experiment_history_block}\n\n"
            f"Previous execution error:\n{previous_error}\n\n"
            f"Previous verification context:\n{verification_block}\n\n"
            "Rules:\n"
            "- Work only on this story in this run.\n"
            "- Primary user goal has priority over repository defaults/examples.\n"
            "- Do not switch to another dataset or task unless user goal explicitly requires it.\n"
            "- Treat the current working directory as the workspace root; all shell paths must be workspace-relative and must not start with the workspace_id directory name.\n"
            "- Any smoke-test or training metric used for acceptance must come from a disjoint evaluation split; reuse of the training subset is invalid.\n"
            "- A train-subset overfit check is diagnostic only and cannot satisfy the story or quality gate.\n"
            "- Metrics artifacts must explicitly record the evaluation split or split-integrity note.\n"
            "- Metric family, smoke policy, and artifact content must stay aligned with the inferred task intent.\n"
            "- If a step executes model training for real, represent it as an explicit shell command step; use Codex steps to prepare code/config only.\n"
            "- If quality target was not met before, change hyperparameters using previous verification history.\n"
            "- If `improvement_strategy` is present in previous verification context, follow its chosen intervention first.\n"
            "- Use codex actions for implementation/editing steps.\n"
            "- Return concise summary of changed files and validations.\n"
        )
        planner_input = PlanInput(
            goal=augmented_goal,
            constraints=constraints,
            contexts=contexts,
            workspace_id=run.workspace_id,
            workspace_snapshot_summary=workspace_tree_snapshot,
            experiment_history_summary=experiment_history_summary,
            previous_error=run.error_message,
            last_failed_step=last_failed_step,
            previous_verification=previous_verification,
        )
        return await self.planner_for_ralph().build_plan(planner_input)

    async def evaluate_quality_gate(
        self,
        task: dict[str, Any],
        workspace_path: Path,
        verification: VerificationResult,
        previous_verification: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        story_id = self.extract_story_id(task)
        return await self.quality_gate_service.evaluate(
            task=task,
            workspace_path=workspace_path,
            verification=verification,
            story_id=story_id,
        )

    async def handle_story_completion(
        self,
        run: RunRecord,
        task: dict[str, Any],
        summary: RunResultSummary,
        workspace_path: Path,
    ) -> None:
        prd = self.backlog.load_prd(workspace_path)
        constraints = [str(item).strip() for item in json.loads(task["constraints_json"]) if str(item).strip()]
        story_id = self.extract_story_id(task, constraints=constraints)
        if not story_id:
            story = self.backlog.pick_next_story(prd)
            if story is None:
                self.backlog.append_progress(
                    workspace_path,
                    f"run={run.run_id} completed with no pending stories left",
                )
                return
            story_id = story.story_id

        updated_prd = self.backlog.mark_story_passed(workspace_path, story_id)
        self.backlog.append_progress(
            workspace_path,
            (
                f"run={run.run_id} story={story_id} status=passed "
                f"planned_steps={summary.planned_steps} executed_steps={summary.executed_steps}"
            ),
        )

        next_story = self.backlog.pick_next_story(updated_prd)
        if next_story is None:
            self.backlog.append_progress(
                workspace_path,
                f"run={run.run_id} completed with no pending stories left",
            )
            return

        if not self.settings.ralph_auto_queue_next_story:
            return

        event = self._build_followup_ralph_event(
            task=task,
            workspace_id=run.workspace_id,
            next_story_priority=self.backlog.map_story_priority(next_story),
            next_story_id=next_story.story_id,
            next_story_title=next_story.title,
            next_story_description=next_story.description,
            next_story_acceptance=next_story.acceptance_criteria,
        )
        stream_id = await self.bus.publish_task(event)
        await self.bus.publish_internal(
            "run.ralph_next_story_queued",
            {
                "origin_run_id": run.run_id,
                "next_story_id": next_story.story_id,
                "stream_id": stream_id,
            },
        )

    def _build_followup_ralph_event(
        self,
        task: dict[str, Any],
        workspace_id: str,
        next_story_priority: Priority,
        next_story_id: str,
        next_story_title: str,
        next_story_description: str,
        next_story_acceptance: list[str],
    ) -> dict[str, Any]:
        base_constraints = json.loads(task["constraints_json"])
        primary_goal = self.extract_primary_user_goal(task, constraints=base_constraints)
        acceptance_lines = "\n".join(f"- {item}" for item in next_story_acceptance) if next_story_acceptance else ""
        goal = f"[RALPH {next_story_id}] {next_story_title}\n{next_story_description}".strip()
        if acceptance_lines:
            goal = f"{goal}\n\nAcceptance criteria:\n{acceptance_lines}"
        if primary_goal:
            goal = f"{goal}\n\nPrimary user goal:\n{primary_goal}"
        constraints = [str(item).strip() for item in base_constraints if str(item).strip()]
        constraints = [item for item in constraints if not item.startswith("RALPH_STORY_ID:")]
        ralph_guard = "RALPH mode: do not create commits or push in MVP."
        if ralph_guard not in constraints:
            constraints.append(ralph_guard)
        if primary_goal and not self.has_primary_goal_constraint(constraints):
            constraints.append(f"PRIMARY_USER_GOAL: {primary_goal}")
        constraints.append(f"RALPH_STORY_ID: {next_story_id}")
        return {
            "event_id": str(uuid4()),
            "event_type": "task.submitted",
            "schema_version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": str(uuid4()),
            "workspace_id": workspace_id,
            "priority": next_story_priority.value,
            "payload": {
                "goal": goal,
                "constraints": constraints,
                "pdf_scope": json.loads(task["pdf_scope_json"]),
                "execution_mode": "ralph_story",
            },
        }

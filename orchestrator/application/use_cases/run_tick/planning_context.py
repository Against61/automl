from __future__ import annotations

import hashlib
import json
from typing import Any

from orchestrator.persistence.db import Database
from orchestrator.persistence.schemas import PlannerPlan, RetrievedContext, RunRecord
from orchestrator.planning.planner import PlanInput


class PlanningContextService:
    _EXCLUDED_SKILL_TOKENS = ("lightning",)
    _METRIC_PREFERENCE: tuple[tuple[str, bool], ...] = (
        ("eval_accuracy", True),
        ("test_accuracy", True),
        ("accuracy", True),
        ("val_accuracy", True),
        ("out_evaluation_accuracy", True),
        ("iou", True),
        ("mean_iou", True),
        ("dice", True),
        ("map50_95", True),
        ("map50", True),
        ("f1", True),
        ("roc_auc", True),
        ("eval_loss", False),
        ("val_loss", False),
        ("loss", False),
    )

    def __init__(self, db: Database, experiment_history_context_limit: int) -> None:
        self.db = db
        self.experiment_history_context_limit = experiment_history_context_limit

    def build_plan_input(
        self,
        *,
        task: dict[str, Any],
        run: RunRecord,
        workspace_id: str,
        contexts: list[RetrievedContext],
        workspace_snapshot_summary: str | None,
        experiment_history_summary: str | None,
        experiment_memory_summary: str | None,
        baseline_research_summary: str | None,
        last_failed_step: dict[str, Any] | None,
        previous_verification: dict[str, Any] | None = None,
        micro_training_policy: dict[str, Any] | None = None,
    ) -> PlanInput:
        return PlanInput(
            goal=task["goal"],
            constraints=json.loads(task["constraints_json"]),
            contexts=contexts,
            workspace_id=workspace_id,
            workspace_snapshot_summary=workspace_snapshot_summary,
            experiment_history_summary=experiment_history_summary,
            experiment_memory_summary=experiment_memory_summary,
            baseline_research_summary=baseline_research_summary,
            previous_error=run.error_message,
            last_failed_step=last_failed_step,
            previous_verification=previous_verification,
            micro_training_policy=micro_training_policy,
        )

    def latest_verification_snapshot(self, verification_json: dict[str, Any] | None) -> dict[str, Any] | None:
        if not verification_json:
            return None
        snapshot = {
            key: value
            for key, value in verification_json.items()
            if key
            in {
                "status",
                "commands",
                "metrics",
                "metric_resolution",
                "latest_hyperparameters",
                "hyperparameter_attempts",
                "quality_gate",
                "attempt",
                "latest_attempt",
                "micro_training_policy",
            }
        }
        history_raw = verification_json.get("history")
        compact_history: list[dict[str, Any]] = []
        if isinstance(history_raw, list):
            for entry in history_raw[-8:]:
                if not isinstance(entry, dict):
                    continue
                compact_history.append(
                    {
                        "attempt": entry.get("attempt"),
                        "status": entry.get("status"),
                        "metrics": entry.get("metrics", {}),
                        "latest_hyperparameters": entry.get("latest_hyperparameters", {}),
                        "hyperparameter_attempts": (entry.get("hyperparameter_attempts") or [])[-4:],
                    }
                )
        strategy = self.compact_strategy_summary(verification_json.get("improvement_strategy"))
        if strategy:
            snapshot["improvement_strategy"] = strategy
        if compact_history:
            snapshot["history"] = compact_history
            snapshot["attempt_history"] = compact_history
        return snapshot

    async def list_experiment_history(
        self,
        *,
        run: RunRecord,
        task: dict[str, Any],
    ) -> list[dict[str, Any]]:
        goal_signature = run.goal_signature or self.build_task_signature_from_record(task)
        if not goal_signature:
            return []
        return await self.db.list_experiment_attempts(
            workspace_id=run.workspace_id,
            goal_signature=goal_signature,
            limit=self.experiment_history_context_limit,
        )

    @staticmethod
    def format_experiment_history_summary(attempts: list[dict[str, Any]]) -> str:
        if not attempts:
            return "none"
        lines: list[str] = []
        for item in attempts[-8:]:
            if not isinstance(item, dict):
                continue
            run_id = str(item.get("run_id") or "n/a")
            attempt = item.get("attempt")
            quality_status = str(item.get("quality_status") or "n/a")
            quality_reason = str(item.get("quality_reason") or "").strip()
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            hyperparameters = item.get("hyperparameters") if isinstance(item.get("hyperparameters"), dict) else {}
            skill_paths = item.get("skill_paths") if isinstance(item.get("skill_paths"), list) else []
            chosen = None
            strategy = item.get("strategy") if isinstance(item.get("strategy"), dict) else {}
            if strategy:
                chosen = (
                    strategy.get("chosen_intervention", {}).get("id")
                    if isinstance(strategy.get("chosen_intervention"), dict)
                    else strategy.get("chosen_intervention_id")
                )
            metric_preview = ", ".join(
                f"{key}={metrics[key]}"
                for key in list(metrics.keys())[:3]
            ) or "no metrics"
            hp_preview = ", ".join(
                f"{key}={value}" for key, value in list(hyperparameters.items())[:4]
            ) or "no hyperparameters"
            skill_preview = ", ".join(str(item) for item in skill_paths[:3]) or "no skills"
            line = (
                f"- run={run_id} attempt={attempt} quality={quality_status} "
                f"chosen={chosen or 'n/a'} metrics=[{metric_preview}] "
                f"hyperparameters=[{hp_preview}] skills=[{skill_preview}]"
            )
            if quality_reason:
                line += f" reason={quality_reason}"
            lines.append(line)
        return "\n".join(lines) if lines else "none"

    @staticmethod
    def build_experiment_memory_summary(attempts: list[dict[str, Any]]) -> str:
        valid_attempts = [item for item in attempts if isinstance(item, dict)]
        if not valid_attempts:
            return "none"

        recent = valid_attempts[-8:]
        latest = recent[-1]
        best = PlanningContextService._best_attempt(recent)
        delta_lines = PlanningContextService._recent_metric_deltas(recent)
        intervention_summary = PlanningContextService._repeated_interventions(recent)
        hp_summary = PlanningContextService._recent_hyperparameter_moves(recent)

        lines = [f"- attempts_analyzed: {len(recent)}"]
        latest_line = PlanningContextService._attempt_memory_line("latest_attempt", latest)
        if latest_line:
            lines.append(latest_line)
        best_line = PlanningContextService._attempt_memory_line("best_attempt", best)
        if best_line:
            lines.append(best_line)
        if delta_lines:
            lines.append(f"- recent_metric_deltas: {'; '.join(delta_lines)}")
        if intervention_summary:
            lines.append(f"- repeated_interventions: {intervention_summary}")
        if hp_summary:
            lines.append(f"- recent_hyperparameter_moves: {hp_summary}")
        heuristic = PlanningContextService._heuristic_readout(recent)
        if heuristic:
            lines.append(f"- heuristic_readout: {heuristic}")
        return "\n".join(lines)

    @staticmethod
    def selected_skill_paths_from_verification(verification: dict[str, Any] | None) -> list[str]:
        if not isinstance(verification, dict):
            return []
        strategy = verification.get("improvement_strategy")
        if not isinstance(strategy, dict):
            return []
        chosen = strategy.get("chosen_intervention")
        if not isinstance(chosen, dict):
            return []
        values = chosen.get("skill_paths")
        if not isinstance(values, list):
            return []
        skill_paths: list[str] = []
        for item in values:
            value = str(item).strip()
            if value and not PlanningContextService._is_excluded_skill_path(value) and value not in skill_paths:
                skill_paths.append(value)
        return skill_paths[:3]

    def attach_selected_skills_to_plan(
        self,
        *,
        plan: PlannerPlan,
        previous_verification: dict[str, Any] | None,
        experiment_history: list[dict[str, Any]] | None = None,
    ) -> PlannerPlan:
        selected_skill_paths = self.selected_skill_paths_from_verification(previous_verification)
        if not selected_skill_paths:
            selected_skill_paths = self.selected_skill_paths_from_experiment_history(experiment_history or [])
        if not selected_skill_paths:
            return plan
        attached = False
        for step in plan.steps:
            if step.action != "codex":
                continue
            if step.skill_paths:
                attached = True
                continue
            step.skill_paths = list(selected_skill_paths)
            attached = True
        return plan if attached else plan

    @staticmethod
    def selected_skill_paths_from_experiment_history(attempts: list[dict[str, Any]]) -> list[str]:
        for item in reversed(attempts):
            if not isinstance(item, dict):
                continue
            strategy = item.get("strategy")
            if isinstance(strategy, dict):
                chosen = strategy.get("chosen_intervention")
                if isinstance(chosen, dict):
                    values = chosen.get("skill_paths")
                    if isinstance(values, list):
                        paths = [
                            str(value).strip()
                            for value in values
                            if str(value).strip() and not PlanningContextService._is_excluded_skill_path(str(value))
                        ]
                        if paths:
                            return paths[:3]
            values = item.get("skill_paths")
            if isinstance(values, list):
                paths = [
                    str(value).strip()
                    for value in values
                    if str(value).strip() and not PlanningContextService._is_excluded_skill_path(str(value))
                ]
                if paths:
                    return paths[:3]
        return []

    @staticmethod
    def compact_strategy_summary(strategy: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(strategy, dict):
            return None
        diagnosis = strategy.get("diagnosis") if isinstance(strategy.get("diagnosis"), dict) else {}
        objective = strategy.get("objective") if isinstance(strategy.get("objective"), dict) else {}
        chosen = strategy.get("chosen_intervention") if isinstance(strategy.get("chosen_intervention"), dict) else {}
        return {
            "kind": strategy.get("kind"),
            "quality_reason": strategy.get("quality_reason"),
            "objective": {
                "metric_key": objective.get("metric_key"),
                "target": objective.get("target"),
                "current_value": objective.get("current_value"),
                "gap": objective.get("gap"),
                "unit": objective.get("unit"),
            },
            "diagnosis": {
                "pattern": diagnosis.get("pattern"),
                "confidence": diagnosis.get("confidence"),
            },
            "chosen_intervention_id": strategy.get("chosen_intervention_id") or chosen.get("id"),
            "chosen_intervention": {
                "id": chosen.get("id"),
                "type": chosen.get("type"),
                "description": chosen.get("description"),
                "actions": list(chosen.get("actions") or [])[:5] if isinstance(chosen.get("actions"), list) else [],
                "skill_paths": (
                    [
                        str(item).strip()
                        for item in list(chosen.get("skill_paths") or [])[:4]
                        if str(item).strip() and not PlanningContextService._is_excluded_skill_path(str(item))
                    ]
                    if isinstance(chosen.get("skill_paths"), list)
                    else []
                ),
            },
            "planner_directives": (
                list(strategy.get("planner_directives") or [])[:4]
                if isinstance(strategy.get("planner_directives"), list)
                else []
            ),
        }

    @staticmethod
    def coerce_run_contexts(context_json: list[dict[str, Any]] | None) -> list[RetrievedContext]:
        contexts_raw = context_json or []
        validated: list[RetrievedContext] = []
        for item in contexts_raw:
            if not isinstance(item, dict):
                continue
            validated.append(
                RetrievedContext.model_validate(
                    {
                        "snippet": item.get("snippet", ""),
                        "document_path": item.get("document_path", ""),
                        "page_number": int(item.get("page_number", 1)),
                        "confidence": float(item.get("confidence", 0.0)),
                    }
                )
            )
        return validated

    @classmethod
    def _is_excluded_skill_path(cls, value: str) -> bool:
        lowered = value.strip().lower()
        return any(token in lowered for token in cls._EXCLUDED_SKILL_TOKENS)

    @classmethod
    def _best_attempt(cls, attempts: list[dict[str, Any]]) -> dict[str, Any] | None:
        best_item: dict[str, Any] | None = None
        best_score: float | None = None
        for item in attempts:
            metric = cls._extract_metric_signal(item)
            if metric is None:
                continue
            _key, value, higher_is_better = metric
            score = value if higher_is_better else -value
            if best_score is None or score > best_score:
                best_score = score
                best_item = item
        return best_item

    @classmethod
    def _attempt_memory_line(cls, label: str, item: dict[str, Any] | None) -> str | None:
        if not isinstance(item, dict):
            return None
        metric = cls._extract_metric_signal(item)
        metric_block = "metric=n/a"
        if metric is not None:
            metric_key, value, _higher_is_better = metric
            metric_block = f"{metric_key}={value:.4f}"
        chosen = cls._extract_intervention_id(item)
        return (
            f"- {label}: attempt={item.get('attempt', 'n/a')} "
            f"quality={item.get('quality_status') or 'n/a'} "
            f"{metric_block} intervention={chosen or 'n/a'}"
        )

    @classmethod
    def _recent_metric_deltas(cls, attempts: list[dict[str, Any]]) -> list[str]:
        deltas: list[str] = []
        previous_metric: tuple[str, float, bool] | None = None
        previous_attempt = None
        for item in attempts[-4:]:
            metric = cls._extract_metric_signal(item)
            if metric is None:
                continue
            metric_key, value, _higher_is_better = metric
            attempt_number = item.get("attempt")
            if previous_metric is not None and previous_metric[0] == metric_key:
                delta = value - previous_metric[1]
                deltas.append(f"{previous_attempt}->{attempt_number} {metric_key} {delta:+.4f}")
            previous_metric = metric
            previous_attempt = attempt_number
        return deltas[:3]

    @classmethod
    def _repeated_interventions(cls, attempts: list[dict[str, Any]]) -> str | None:
        counts: dict[str, int] = {}
        for item in attempts:
            intervention = cls._extract_intervention_id(item)
            if not intervention:
                continue
            counts[intervention] = counts.get(intervention, 0) + 1
        repeated = [
            f"{key} x{value}"
            for key, value in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
            if value > 1
        ]
        return ", ".join(repeated[:3]) or None

    @classmethod
    def _recent_hyperparameter_moves(cls, attempts: list[dict[str, Any]]) -> str | None:
        latest_values: dict[str, Any] = {}
        for item in attempts[-3:]:
            hyperparameters = item.get("hyperparameters")
            if not isinstance(hyperparameters, dict):
                continue
            for key, value in list(hyperparameters.items())[:4]:
                latest_values[str(key)] = value
        if not latest_values:
            return None
        pairs = [f"{key}={value}" for key, value in list(latest_values.items())[:5]]
        return ", ".join(pairs)

    @classmethod
    def _heuristic_readout(cls, attempts: list[dict[str, Any]]) -> str | None:
        comparable: list[tuple[str, float]] = []
        for item in attempts[-4:]:
            metric = cls._extract_metric_signal(item)
            if metric is None:
                continue
            metric_key, value, _higher_is_better = metric
            comparable.append((metric_key, value))
        if len(comparable) < 3:
            return None
        latest_key = comparable[-1][0]
        same_metric_values = [value for key, value in comparable if key == latest_key]
        if len(same_metric_values) >= 3 and max(same_metric_values) - min(same_metric_values) <= 0.01:
            return "recent comparable attempts are near-flat; prefer a strategy change over repeating the same recipe"
        if comparable[-1][1] > comparable[0][1]:
            return "current recipe is still moving in the right direction; expanding budget is justified before broad search"
        return None

    @classmethod
    def _extract_metric_signal(cls, item: dict[str, Any]) -> tuple[str, float, bool] | None:
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            return None
        for key, higher_is_better in cls._METRIC_PREFERENCE:
            value = cls._to_float(metrics.get(key))
            if value is not None:
                return key, value, higher_is_better
        for key, value in metrics.items():
            parsed = cls._to_float(value)
            if parsed is not None:
                return str(key), parsed, True
        return None

    @staticmethod
    def _extract_intervention_id(item: dict[str, Any]) -> str | None:
        strategy = item.get("strategy")
        if not isinstance(strategy, dict):
            return None
        chosen = strategy.get("chosen_intervention")
        if isinstance(chosen, dict):
            value = str(chosen.get("id") or "").strip()
            if value:
                return value
        value = str(strategy.get("chosen_intervention_id") or "").strip()
        return value or None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", ".")
            try:
                return float(cleaned)
            except ValueError:
                return None
        return None

    @staticmethod
    def build_task_goal_signature(payload_dict: dict[str, Any]) -> str:
        normalized = {
            "goal": str(payload_dict.get("goal", "")).strip(),
            "constraints": sorted(str(item) for item in payload_dict.get("constraints", [])),
            "execution_mode": str(payload_dict.get("execution_mode", "plan_execute")),
            "pdf_scope": sorted(str(item) for item in payload_dict.get("pdf_scope", [])),
        }
        serialized = json.dumps(normalized, ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def build_task_signature_from_record(self, task: dict[str, Any]) -> str | None:
        payload_data = self.extract_task_payload(task)
        try:
            return self.build_task_goal_signature(payload_data)
        except Exception:
            return None

    @staticmethod
    def extract_task_payload(task: dict[str, Any]) -> dict[str, Any]:
        def _safe_list(value: str) -> list[Any]:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []

        constraints: list[str] = []
        for raw in _safe_list(task.get("constraints_json", "[]")):
            value = str(raw).strip()
            if value:
                constraints.append(value)
        pdf_scope: list[str] = []
        for raw in _safe_list(task.get("pdf_scope_json", "[]")):
            value = str(raw).strip()
            if value:
                pdf_scope.append(value)
        execution_mode = "plan_execute"
        try:
            payload = json.loads(task.get("payload_json", "{}") or "{}")
            if isinstance(payload, dict):
                inner = payload.get("payload")
                if isinstance(inner, dict):
                    payload = inner
            mode = None
            if isinstance(payload, dict):
                mode = payload.get("execution_mode")
            if isinstance(mode, str) and mode.strip():
                execution_mode = mode.strip().lower()
        except json.JSONDecodeError:
            pass
        return {
            "goal": str(task.get("goal", "")).strip(),
            "constraints": constraints,
            "execution_mode": execution_mode,
            "pdf_scope": pdf_scope,
        }

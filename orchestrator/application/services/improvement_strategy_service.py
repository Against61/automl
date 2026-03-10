from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.application.services.quality_gate_service import QualityGateService
from orchestrator.execution.verifier import VerificationResult


class ImprovementStrategyService:
    def __init__(self, quality_gate_service: QualityGateService):
        self.quality_gate_service = quality_gate_service

    def build_for_quality_failure(
        self,
        *,
        run_id: str,
        task: dict[str, Any],
        workspace_path: Path,
        verification: VerificationResult,
        previous_verification: dict[str, Any] | None,
        quality_reason: str,
        experiment_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        requirement = self.quality_gate_service.extract_requirement(
            task=task,
            workspace_path=workspace_path,
        ) or {}
        metric_key = str(requirement.get("metric_key", "")).strip()
        operator = str(requirement.get("operator", ">=")).strip() or ">="
        unit = str(requirement.get("unit", "ratio")).strip() or "ratio"
        target = self._to_float(requirement.get("target"))
        if target is not None and unit in {"%", "percent", "pct"}:
            target = target / 100.0

        metrics = dict(verification.metrics or {})
        metric_value = self.quality_gate_service.select_metric_value(metrics, metric_key) if metric_key else None
        metric_gap = None
        if metric_value is not None and target is not None:
            metric_gap = target - metric_value

        train_metric = self.quality_gate_service.select_metric_value(metrics, "train_accuracy")
        val_metric = self.quality_gate_service.select_metric_value(metrics, "val_accuracy")
        if val_metric is None:
            val_metric = self.quality_gate_service.select_metric_value(metrics, metric_key) if metric_key else None
        overfit_gap = None
        if train_metric is not None and val_metric is not None:
            overfit_gap = train_metric - val_metric

        hyperparameter_attempts = self._collect_hyperparameter_attempts(previous_verification)
        latest_hyperparameters = hyperparameter_attempts[-1].get("hyperparameters", {}) if hyperparameter_attempts else {}
        available_skills = self._discover_workspace_skills(workspace_path)
        experiment_attempts = [item for item in (experiment_history or []) if isinstance(item, dict)]
        diagnosis = self._diagnose(
            metric_gap=metric_gap,
            overfit_gap=overfit_gap,
            metric_value=metric_value,
            target_value=target,
        )
        relevant_skills = self._select_relevant_skills(
            available_skills=available_skills,
            diagnosis=diagnosis,
            goal_text=str(task.get("goal") or ""),
            latest_hyperparameters=latest_hyperparameters,
            experiment_history=experiment_attempts,
        )
        candidates = self._build_candidate_interventions(
            diagnosis=diagnosis,
            available_skills=available_skills,
            relevant_skills=relevant_skills,
            latest_hyperparameters=latest_hyperparameters,
            experiment_history=experiment_attempts,
        )
        chosen = candidates[0] if candidates else {}
        attempt_number = int(previous_verification.get("attempt", 0) if isinstance(previous_verification, dict) else 0) + 1
        strategy_path = self._strategy_path(workspace_path, run_id=run_id, attempt_number=attempt_number)

        strategy = {
            "version": "1.0",
            "kind": "quality_improvement_strategy",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "quality_reason": quality_reason,
            "objective": {
                "metric_key": metric_key,
                "operator": operator,
                "target": target,
                "unit": unit,
                "current_value": metric_value,
                "gap": metric_gap,
            },
            "diagnosis": diagnosis,
            "history": {
                "attempt_number": attempt_number,
                "latest_hyperparameters": latest_hyperparameters,
                "hyperparameter_attempts": hyperparameter_attempts[-8:],
                "experiment_attempts": experiment_attempts[-8:],
                "last_metrics": metrics,
            },
            "candidate_interventions": candidates,
            "chosen_intervention_id": chosen.get("id"),
            "chosen_intervention": chosen,
            "research_tasks": self._research_tasks(diagnosis=diagnosis, available_skills=relevant_skills or available_skills),
            "planner_directives": self._planner_directives(chosen=chosen),
            "recommended_skills": relevant_skills,
            "available_skills": available_skills,
            "artifact_path": str(strategy_path.relative_to(workspace_path)),
        }

        strategy_path.parent.mkdir(parents=True, exist_ok=True)
        strategy_path.write_text(json.dumps(strategy, ensure_ascii=True, indent=2), encoding="utf-8")
        self._append_markdown_summary(workspace_path=workspace_path, strategy=strategy)
        return strategy

    def _discover_workspace_skills(self, workspace_path: Path) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        roots: list[tuple[Path, str]] = []

        workspace_root = workspace_path / "knowledge" / "skills"
        roots.append((workspace_root, "workspace"))

        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            roots.append((Path(codex_home) / "skills", "codex_home"))
        else:
            roots.append((Path.home() / ".codex" / "skills", "codex_home"))

        for root, source in roots:
            if not root.exists():
                continue
            for skill_md in sorted(root.rglob("SKILL.md")):
                name = skill_md.parent.name
                try:
                    rel = skill_md.relative_to(root.parent if source == "codex_home" else workspace_path)
                except ValueError:
                    rel = skill_md
                entry = (source, name)
                if entry in seen:
                    continue
                seen.add(entry)
                found.append({"name": name, "path": str(rel), "source": source})
        return found[:20]

    def _diagnose(
        self,
        *,
        metric_gap: float | None,
        overfit_gap: float | None,
        metric_value: float | None,
        target_value: float | None,
    ) -> dict[str, Any]:
        if overfit_gap is not None and overfit_gap >= 0.08:
            return {
                "pattern": "overfitting",
                "confidence": "high",
                "evidence": [
                    "train metric is significantly better than validation metric",
                    f"overfit_gap={overfit_gap:.4f}",
                ],
            }
        if metric_gap is not None and metric_gap > 0.12:
            return {
                "pattern": "underfitting_or_data_pipeline_limit",
                "confidence": "medium",
                "evidence": [
                    f"metric gap to target is large: {metric_gap:.4f}",
                    "consider model capacity, data quality, and stronger training recipe",
                ],
            }
        if metric_gap is not None and metric_gap > 0:
            return {
                "pattern": "near_target_plateau",
                "confidence": "medium",
                "evidence": [
                    f"model is near target but still below threshold: gap={metric_gap:.4f}",
                ],
            }
        return {
            "pattern": "insufficient_signal",
            "confidence": "low",
            "evidence": [
                f"metric_value={metric_value}",
                f"target_value={target_value}",
            ],
        }

    def _build_candidate_interventions(
        self,
        *,
        diagnosis: dict[str, Any],
        available_skills: list[dict[str, str]],
        relevant_skills: list[dict[str, str]],
        latest_hyperparameters: dict[str, Any],
        experiment_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pattern = str(diagnosis.get("pattern", "insufficient_signal"))
        skill_paths = [item["path"] for item in relevant_skills]
        skill_names = [item["name"] for item in relevant_skills]
        has_lightning = any("lightning" in item["name"].lower() or "lightning" in item["path"].lower() for item in available_skills)

        interventions: list[dict[str, Any]] = []
        if pattern == "overfitting":
            interventions.append(
                {
                    "id": "data_aug_regularization",
                    "type": "data_and_regularization",
                    "description": "Increase augmentation strength, add regularization, and reduce overfitting.",
                    "actions": [
                        "add/strengthen stochastic augmentations",
                        "increase dropout or weight decay",
                        "enable early stopping on validation metric",
                    ],
                    "expected_gain": "better validation generalization",
                    "cost_level": "medium",
                    "skill_paths": skill_paths,
                    "skill_names": skill_names,
                }
            )
        elif pattern == "underfitting_or_data_pipeline_limit":
            interventions.append(
                {
                    "id": "capacity_and_schedule_upgrade",
                    "type": "model_and_training_regime",
                    "description": "Increase model capacity and improve training schedule before more random search.",
                    "actions": [
                        "upgrade backbone/head capacity",
                        "use LR scheduler with warmup",
                        "increase effective epochs with early stopping guard",
                    ],
                    "expected_gain": "higher ceiling metric",
                    "cost_level": "high",
                    "skill_paths": skill_paths,
                    "skill_names": skill_names,
                }
            )
        elif pattern == "near_target_plateau":
            interventions.append(
                {
                    "id": "targeted_finetune",
                    "type": "fine_tuning",
                    "description": "Perform targeted finetuning around known good baseline.",
                    "actions": [
                        "small LR search around current best",
                        "batch size / optimizer tuning",
                        "class-weighted loss if class imbalance is visible",
                    ],
                    "expected_gain": "small but meaningful metric lift",
                    "cost_level": "low",
                    "skill_paths": skill_paths,
                    "skill_names": skill_names,
                }
            )
        else:
            interventions.append(
                {
                    "id": "observability_and_baseline_reset",
                    "type": "diagnostics",
                    "description": "Harden diagnostics before next heavy training iteration.",
                    "actions": [
                        "add per-class metrics and confusion matrix",
                        "verify split/data leakage assumptions",
                        "create stable baseline config file",
                    ],
                    "expected_gain": "clearer optimization signal",
                    "cost_level": "low",
                    "skill_paths": skill_paths,
                    "skill_names": skill_names,
                }
            )

        interventions.append(
            {
                "id": "skill_and_research_assisted_iteration",
                "type": "knowledge_driven",
                "description": "Use external playbooks/skills and short research pass to choose the next architecture recipe.",
                "actions": [
                    "summarize what already failed from hyperparameter history",
                    "consult relevant skills before coding",
                    "encode chosen recipe in explicit training config",
                ],
                "expected_gain": "avoid repeated failed recipes",
                "cost_level": "medium",
                "skill_paths": skill_paths,
                "skill_names": skill_names,
                "history_reference_count": len(experiment_history),
            }
        )
        if has_lightning:
            interventions.append(
                {
                    "id": "lightning_refactor_for_repeatability",
                    "type": "framework",
                    "description": "Use PyTorch Lightning style loops/checkpointing for deterministic experiment cycles.",
                    "actions": [
                        "move training loop to Lightning module/datamodule",
                        "log metrics consistently each epoch",
                        "persist best checkpoint by validation metric",
                    ],
                    "expected_gain": "more stable and reproducible iteration loop",
                    "cost_level": "medium",
                    "skill_paths": [item["path"] for item in relevant_skills if "lightning" in item["name"].lower() or "lightning" in item["path"].lower()],
                    "skill_names": [item["name"] for item in relevant_skills if "lightning" in item["name"].lower() or "lightning" in item["path"].lower()],
                }
            )

        if latest_hyperparameters:
            interventions[0]["latest_hyperparameters"] = latest_hyperparameters
        return interventions

    def _research_tasks(self, *, diagnosis: dict[str, Any], available_skills: list[dict[str, str]]) -> list[str]:
        pattern = str(diagnosis.get("pattern", "insufficient_signal"))
        tasks = [
            f"Summarize 3 concrete tactics for pattern `{pattern}` from trusted docs/papers.",
            "Map tactics to current codebase files before editing.",
            "Reject tactics already attempted with no gain.",
        ]
        if available_skills:
            names = ", ".join(item["name"] for item in available_skills[:3])
            tasks.append(f"Apply workspace skill instructions before coding: {names}.")
        return tasks

    def _planner_directives(self, *, chosen: dict[str, Any]) -> list[str]:
        directives = [
            "Start next plan with chosen_intervention actions before generic hyperparameter search.",
            "Do not repeat same failed hyperparameter set unless rationale is explicit.",
            "After training, save metrics and checkpoint artifacts and compare against previous best.",
            f"Primary intervention for next iteration: {chosen.get('id', 'n/a')}",
        ]
        skill_paths = [str(item).strip() for item in (chosen.get("skill_paths") or []) if str(item).strip()]
        if skill_paths:
            directives.append(
                "Apply these skills as execution context before coding: "
                + ", ".join(skill_paths[:4])
            )
        return directives

    def _select_relevant_skills(
        self,
        *,
        available_skills: list[dict[str, str]],
        diagnosis: dict[str, Any],
        goal_text: str,
        latest_hyperparameters: dict[str, Any],
        experiment_history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        pattern = str(diagnosis.get("pattern", "insufficient_signal"))
        goal_blob = goal_text.lower()
        has_history = bool(experiment_history)
        ranked: list[tuple[int, dict[str, str]]] = []
        for item in available_skills:
            name = str(item.get("name") or "").lower()
            path = str(item.get("path") or "").lower()
            score = 0
            if "lightning" in name or "lightning" in path:
                score += 5 if pattern in {"underfitting_or_data_pipeline_limit", "near_target_plateau"} else 3
            if "seaborn" in name or "seaborn" in path:
                score += 4 if pattern in {"overfitting", "insufficient_signal"} else 2
            if "scikit" in name or "sklearn" in name or "scikit" in path:
                score += 3 if pattern in {"insufficient_signal", "overfitting"} else 2
            if "kaggle" in name or "kaggle" in path:
                score += 3 if has_history else 2
            if "jupyter" in name or "notebook" in name or "jupyter" in path:
                score += 2 if pattern == "insufficient_signal" else 1
            if any(token in goal_blob for token in ("fashionmnist", "mnist", "cnn", "pytorch", "train")):
                if "lightning" in name or "kaggle" in name:
                    score += 1
            if latest_hyperparameters and ("lightning" in name or "kaggle" in name):
                score += 1
            if score > 0:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        selected = [item for _, item in ranked[:3]]
        if selected:
            return selected
        return available_skills[:2]

    def _collect_hyperparameter_attempts(self, previous_verification: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(previous_verification, dict):
            return []
        attempts = previous_verification.get("hyperparameter_attempts")
        if isinstance(attempts, list):
            return [item for item in attempts if isinstance(item, dict)]
        return []

    def _strategy_path(self, workspace_path: Path, *, run_id: str, attempt_number: int) -> Path:
        return (
            workspace_path
            / "knowledge"
            / "improvement_strategies"
            / f"strategy_{run_id}_attempt_{attempt_number}.json"
        )

    def _append_markdown_summary(self, *, workspace_path: Path, strategy: dict[str, Any]) -> None:
        path = workspace_path / "knowledge" / "improvement_strategies" / "strategies.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        objective = strategy.get("objective", {})
        diagnosis = strategy.get("diagnosis", {})
        chosen = strategy.get("chosen_intervention", {})
        line = (
            f"- [{strategy.get('generated_at')}] run={strategy.get('run_id')} "
            f"metric={objective.get('metric_key')} current={objective.get('current_value')} "
            f"target={objective.get('target')} diagnosis={diagnosis.get('pattern')} "
            f"chosen={chosen.get('id')}"
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def _to_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip().replace(",", ".")
            is_percent = "%" in raw
            cleaned = re.sub(r"[^0-9eE+\-\.]", "", raw)
            if not cleaned:
                return None
            try:
                parsed = float(cleaned)
            except ValueError:
                return None
            if is_percent and parsed > 1.0:
                parsed = parsed / 100.0
            return parsed
        return None

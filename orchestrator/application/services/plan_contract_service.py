from __future__ import annotations

import re
from itertools import product
from pathlib import Path
from typing import Any

from orchestrator.execution.metric_parsing import text_indicates_split_leakage
from orchestrator.persistence.schemas import ArtifactKind, ArtifactSpec, StepIntent


_METRIC_LINE_RE = re.compile(
    r"(?P<key>[a-zA-Z][a-zA-Z0-9_ ]{1,80})\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:\s*%?)"
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_MULTISPACE_RE = re.compile(r"\s+")


class PlanContractService:
    def __init__(self, *, plan_review_enabled: bool, strictness: str = "balanced"):
        self.plan_review_enabled = plan_review_enabled
        mode = (strictness or "").strip().lower()
        self.strictness = mode if mode in {"free", "balanced", "strict"} else "balanced"

    def evaluate(self, step: Any, workspace_path: Path, result: Any) -> tuple[bool, str]:
        if not self.plan_review_enabled:
            return True, "plan review disabled"
        if self.strictness == "free":
            return True, "contract strictness is free"

        specs = self._coerce_specs(getattr(step, "expected_artifacts", []) or [])
        if not specs:
            return True, "no expected artifacts declared"
        intent = self._coerce_intent(getattr(step, "step_intent", StepIntent.general))
        specs = self._apply_default_metrics_specs(specs, intent)

        reason_parts: list[str] = []
        failure = False

        expected_paths = [spec.path for spec in specs if spec.path and spec.must_exist]
        missing_paths = self._check_artifact_paths(expected_paths, workspace_path)
        missing_paths = self._relax_metrics_path_misses(
            missing_paths=missing_paths,
            specs=specs,
            workspace_path=workspace_path,
            intent=intent,
            result=result,
        )
        has_git_workspace = (workspace_path / ".git").is_dir()

        if missing_paths:
            failure = True
            reason_parts.append(f"missing expected artifacts: {', '.join(sorted(missing_paths))}")

        if (
            self._expected_changed_files_requested(specs)
            and not has_git_workspace
            and not getattr(result, "files_changed", None)
            and getattr(step, "action", "") == "codex"
        ):
            result.files_changed = ["__workspace_without_git__"]
        elif (
            self._expected_changed_files_requested(specs)
            and self.strictness == "strict"
            and not getattr(result, "files_changed", None)
        ):
            failure = True
            reason_parts.append("expected modified files not reported")

        if self._looks_like_smoke_requirement(getattr(step, "stop_condition", "")) and not result.stdout_text.strip():
            failure = True
            reason_parts.append("smoke/check output missing")

        nonempty_failures = self._check_nonempty_specs(specs, workspace_path)
        if nonempty_failures:
            failure = True
            reason_parts.append(f"empty artifacts: {', '.join(nonempty_failures)}")

        freshness_reason = self._check_stale_metrics_artifacts(specs, workspace_path, intent, result)
        if freshness_reason:
            failure = True
            reason_parts.append(freshness_reason)

        leakage_reason = self._check_split_leakage(specs, workspace_path, intent, result)
        if leakage_reason:
            failure = True
            reason_parts.append(leakage_reason)

        intent_ok, intent_reason = self._check_step_intent(intent, specs, workspace_path, result)
        if not intent_ok:
            failure = True
            reason_parts.append(intent_reason)

        if not failure:
            return True, "contract checks passed"
        return False, "; ".join(reason_parts)

    def _apply_default_metrics_specs(self, specs: list[ArtifactSpec], intent: StepIntent) -> list[ArtifactSpec]:
        if intent not in {StepIntent.run_training, StepIntent.verify_metrics}:
            return specs

        copied = [spec.model_copy(deep=True) for spec in specs]
        metric_spec: ArtifactSpec | None = None
        for spec in copied:
            if spec.kind == ArtifactKind.metrics:
                metric_spec = spec
                break

        if metric_spec is None:
            copied.append(
                ArtifactSpec(
                    path="metrics.json",
                    kind=ArtifactKind.metrics,
                    must_exist=True,
                    must_be_nonempty=True,
                )
            )
            return copied

        if not metric_spec.path:
            metric_spec.path = "metrics.json"
        metric_spec.must_exist = True
        metric_spec.must_be_nonempty = True
        return copied

    def _metrics_candidate_paths(
        self,
        *,
        specs: list[ArtifactSpec],
        workspace_path: Path,
        result: Any,
        include_recent: bool = True,
    ) -> list[Path]:
        candidates: list[Path] = []
        seen: set[Path] = set()

        def _add(target: Path | None) -> None:
            if target is None:
                return
            try:
                resolved = target.resolve()
            except OSError:
                return
            if resolved in seen or not resolved.is_file():
                return
            try:
                if not resolved.is_relative_to(workspace_path.resolve()):
                    return
            except ValueError:
                return
            seen.add(resolved)
            candidates.append(resolved)

        for spec in specs:
            if not self._is_metrics_like_spec(spec) or not spec.path:
                continue
            _add(self._resolve_target(spec.path, workspace_path))

        for rel_path in self._normalized_changed_paths(getattr(result, "files_changed", None), workspace_path):
            target = workspace_path / rel_path
            if self._looks_like_metrics_artifact(target):
                _add(target)

        if include_recent:
            for rel_path in self._recent_metrics_artifacts(workspace_path):
                _add(workspace_path / rel_path)

        return candidates

    def _relax_metrics_path_misses(
        self,
        *,
        missing_paths: list[str],
        specs: list[ArtifactSpec],
        workspace_path: Path,
        intent: StepIntent,
        result: Any,
    ) -> list[str]:
        if not missing_paths:
            return missing_paths
        if self.strictness == "strict":
            return missing_paths
        if intent not in {StepIntent.run_training, StepIntent.verify_metrics}:
            return missing_paths
        metrics_candidates = self._metrics_candidate_paths(
            specs=specs,
            workspace_path=workspace_path,
            result=result,
        )
        if not metrics_candidates:
            return missing_paths

        spec_by_path = {
            spec.path: spec
            for spec in specs
            if spec.path and spec.must_exist
        }
        unresolved: list[str] = []
        for path in missing_paths:
            spec = spec_by_path.get(path)
            if spec is None:
                unresolved.append(path)
                continue
            if not self._is_metrics_like_spec(spec):
                unresolved.append(path)
                continue
        return unresolved

    def _is_metrics_like_spec(self, spec: ArtifactSpec) -> bool:
        if spec.kind in {ArtifactKind.metrics, ArtifactKind.report}:
            return True
        filename = Path(spec.path or "").name.lower()
        if not filename:
            return False
        if "metric" in filename or "report" in filename:
            return True
        return filename in {
            "results.json",
            "result.json",
            "verification.json",
            "metrics.json",
            "metrics.md",
            "metrics.markdown",
            "report.json",
            "report.md",
            "report.markdown",
        }

    def _recent_metrics_artifacts(self, workspace_path: Path, *, max_age_sec: int = 120) -> list[str]:
        candidates: list[Path] = [
            workspace_path / "metrics.md",
            workspace_path / "metrics.json",
            workspace_path / "results.json",
        ]
        for pattern in ("*metrics*.md", "*metrics*.markdown", "*metrics*.json", "*report*.md"):
            candidates.extend(sorted(workspace_path.rglob(pattern)))
        seen: set[Path] = set()
        recent: list[str] = []
        now_ts = __import__("time").time()
        workspace_root = workspace_path.resolve()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                if not candidate.is_file() or candidate.stat().st_size <= 0:
                    continue
                if now_ts - candidate.stat().st_mtime > max_age_sec:
                    continue
                resolved = candidate.resolve()
                if not resolved.is_relative_to(workspace_root):
                    continue
                recent.append(str(resolved.relative_to(workspace_root)))
            except (OSError, ValueError):
                continue
        return recent

    def _normalized_changed_paths(self, files_changed: list[str] | None, workspace_path: Path) -> set[str]:
        normalized: set[str] = set()
        for path in files_changed or []:
            target = self._resolve_target(path, workspace_path)
            if target is None:
                continue
            try:
                normalized.add(str(target.resolve().relative_to(workspace_path.resolve())))
            except ValueError:
                continue
            except OSError:
                continue
        return normalized

    def _check_split_leakage(self, specs: list[ArtifactSpec], workspace_path: Path, intent: StepIntent, result: Any) -> str | None:
        if intent not in {StepIntent.run_training, StepIntent.verify_metrics}:
            return None

        candidate_paths: list[Path] = []
        seen: set[Path] = set()

        for candidate in self._metrics_candidate_paths(
            specs=specs,
            workspace_path=workspace_path,
            result=result,
        ):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidate_paths.append(resolved)

        for path in candidate_paths:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if self._artifact_indicates_split_leakage(content):
                rel_path = str(path.relative_to(workspace_path.resolve()))
                return f"detected evaluation/train overlap in metrics artifact: {rel_path}"
        return None

    def _check_stale_metrics_artifacts(
        self,
        specs: list[ArtifactSpec],
        workspace_path: Path,
        intent: StepIntent,
        result: Any,
    ) -> str | None:
        if intent not in {StepIntent.run_training, StepIntent.verify_metrics}:
            return None
        step_start_ts = self._step_start_timestamp(result)
        if step_start_ts is None:
            return None

        changed_paths = self._normalized_changed_paths(getattr(result, "files_changed", None), workspace_path)
        fallback_candidates = {
            str(path.relative_to(workspace_path.resolve()))
            for path in self._metrics_candidate_paths(
                specs=specs,
                workspace_path=workspace_path,
                result=result,
            )
        }
        for spec in specs:
            if not spec.path or not self._is_metrics_like_spec(spec):
                continue
            target = self._resolve_target(spec.path, workspace_path)
            if target is None or not target.exists():
                continue
            try:
                rel_path = str(target.resolve().relative_to(workspace_path.resolve()))
            except (OSError, ValueError):
                continue
            if rel_path in changed_paths:
                continue
            if rel_path not in fallback_candidates and fallback_candidates:
                continue
            try:
                artifact_mtime = target.stat().st_mtime
            except OSError:
                continue
            if artifact_mtime + 1.0 < step_start_ts:
                return f"stale metrics artifact reused without refresh: {rel_path}"
        return None

    def _step_start_timestamp(self, result: Any) -> float | None:
        candidates: list[float] = []
        for attr in ("stdout_path", "stderr_path"):
            raw_path = getattr(result, attr, None)
            if not raw_path:
                continue
            try:
                stat = Path(str(raw_path)).stat()
            except OSError:
                continue
            candidates.append(stat.st_mtime)
        if not candidates:
            return None
        return min(candidates)

    def _artifact_indicates_split_leakage(self, text: str) -> bool:
        return text_indicates_split_leakage(text)

    def _looks_like_metrics_artifact(self, path: Path) -> bool:
        lowered = path.name.lower()
        if "metric" in lowered or "report" in lowered:
            return True
        return lowered in {
            "results.json",
            "result.json",
            "verification.json",
            "metrics.json",
            "metrics.md",
            "metrics.markdown",
            "report.json",
            "report.md",
            "report.markdown",
        }

    def _coerce_specs(self, artifacts: list[Any]) -> list[ArtifactSpec]:
        normalized: list[ArtifactSpec] = []
        for artifact in artifacts:
            if isinstance(artifact, ArtifactSpec):
                normalized.append(artifact)
                continue
            if isinstance(artifact, dict):
                try:
                    normalized.append(ArtifactSpec.model_validate(artifact))
                except Exception:
                    continue
                continue
            if isinstance(artifact, str):
                inferred = self._coerce_string_artifact(artifact)
                if inferred is not None:
                    normalized.append(inferred)
        return normalized

    def _coerce_string_artifact(self, raw: str) -> ArtifactSpec | None:
        value = raw.strip()
        if not value:
            return None
        path = self._extract_path_from_text(value)
        if not path:
            return ArtifactSpec(path=None, kind=ArtifactKind.generic, must_exist=False)
        lowered = value.lower()
        kind = ArtifactKind.file
        if "metric" in lowered:
            kind = ArtifactKind.metrics
        elif "report" in lowered:
            kind = ArtifactKind.report
        elif any(path.endswith(ext) for ext in (".ckpt", ".pt", ".pth")):
            kind = ArtifactKind.checkpoint
        return ArtifactSpec(path=path, kind=kind, must_exist=True)

    def _extract_path_from_text(self, text: str) -> str | None:
        cleaned = text.strip().strip("`\"'")
        if cleaned and ("/" in cleaned or "\\" in cleaned) and not any(ch.isspace() for ch in cleaned):
            return cleaned.rstrip(".,;:")
        tokens = [token.strip("`\"'[],(){}") for token in cleaned.split()]
        for token in tokens:
            token = token.rstrip(".,;:")
            if not token:
                continue
            if "/" in token or "\\" in token:
                return token
            if re.match(r"^[A-Za-z0-9._-]+\.[A-Za-z0-9]+$", token):
                return token
            if token.startswith(".") and "." in token:
                return token
        return None

    def _looks_like_path(self, value: str, allowed_exts: set[str]) -> bool:
        if any(ch in value for ch in ("/", "\\")):
            return True
        for ext in allowed_exts:
            if value.endswith(ext):
                return True
        if value.startswith(".") and "." in value:
            return True
        if re.match(r"^[A-Za-z0-9._-]+\\.[A-Za-z0-9]+$", value):
            return True
        return False

    def _check_artifact_paths(self, artifact_paths: list[str], workspace_path: Path) -> list[str]:
        missing: list[str] = []
        workspace_root = workspace_path.resolve()
        for artifact_path in artifact_paths:
            normalized = self._normalize_expected_artifact_path(artifact_path, workspace_path)
            path = Path(normalized)
            target = path if path.is_absolute() else workspace_path / path
            target = target.expanduser().resolve()
            if path.is_absolute():
                try:
                    if not target.is_relative_to(workspace_root):
                        continue
                except ValueError:
                    continue
            if target.exists():
                continue

            if self.strictness != "strict" and self._resolve_fuzzy_match(target, workspace_root):
                continue

            if self.strictness != "strict" and not path.is_absolute() and "/" not in normalized and "\\" not in normalized:
                matches = [
                    match
                    for match in workspace_root.rglob(normalized)
                    if match.is_relative_to(workspace_root)
                ]
                if len(matches) == 1:
                    continue

            missing.append(artifact_path)
        return missing

    def _check_nonempty_specs(self, specs: list[ArtifactSpec], workspace_path: Path) -> list[str]:
        failures: list[str] = []
        for spec in specs:
            if not spec.path or not spec.must_be_nonempty:
                continue
            target = self._resolve_target(spec.path, workspace_path)
            if target is None or not target.exists():
                continue
            try:
                if target.stat().st_size == 0:
                    failures.append(spec.path)
            except OSError:
                failures.append(spec.path)
        return failures

    def _resolve_target(self, path: str, workspace_path: Path) -> Path | None:
        normalized = self._normalize_expected_artifact_path(path, workspace_path)
        if not normalized:
            return None
        target = Path(normalized)
        if not target.is_absolute():
            target = workspace_path / target
        try:
            return target.expanduser().resolve()
        except OSError:
            return None

    def _resolve_fuzzy_match(self, expected: Path, workspace_root: Path) -> bool:
        if not expected.name:
            return False

        basename = expected.name
        matches = [match for match in workspace_root.rglob(basename) if match.is_relative_to(workspace_root)]
        if not matches:
            return False

        if len(matches) == 1:
            return True

        expected_parent = expected.parent.as_posix().rstrip("/")
        if expected_parent not in {"", "."}:
            direct_parents = [
                match
                for match in matches
                if match.parent.as_posix().endswith(expected_parent.rstrip("/"))
            ]
            if len(direct_parents) == 1:
                return True

        if expected.suffix and len(expected.parts) >= 2:
            tail_len = min(2, len(expected.parts))
            suffix = "/".join(expected.parts[-tail_len:])
            suffix_matches = [
                match
                for match in matches
                if match.as_posix().endswith(suffix)
            ]
            if len(suffix_matches) == 1:
                return True

        return False

    def _normalize_expected_artifact_path(self, artifact_path: str, workspace_path: Path) -> str:
        cleaned = artifact_path.strip().strip("`\"'[](){}")
        if not cleaned:
            return cleaned

        cleaned = cleaned.replace("\\", "/")
        cleaned = cleaned.strip()
        cleaned = cleaned.strip("/")
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if not parts:
            return ""

        if parts and parts[0] == "workspace":
            parts = parts[1:]

        workspace_name = workspace_path.name
        if parts and parts[0] == workspace_name:
            parts = parts[1:]

        joined = "/".join(parts)
        if not joined:
            return ""
        return joined

    def _expected_changed_files_requested(self, artifacts: list[ArtifactSpec]) -> bool:
        markers = (
            "changed file",
            "files changed",
            "diff",
            "patch",
            "modification",
        )
        lowered = " | ".join(
            f"{artifact.path or ''}::{artifact.kind.value}".lower()
            for artifact in artifacts
        )
        return any(marker in lowered for marker in markers)

    def _looks_like_smoke_requirement(self, stop_condition: str) -> bool:
        lowered = stop_condition.lower()
        markers = ("smoke", "validate", "verification", "test", "checks")
        return any(marker in lowered for marker in markers)

    def _coerce_intent(self, value: Any) -> StepIntent:
        if isinstance(value, StepIntent):
            return value
        try:
            return StepIntent(str(value))
        except ValueError:
            return StepIntent.general

    def _check_step_intent(
        self,
        intent: StepIntent,
        specs: list[ArtifactSpec],
        workspace_path: Path,
        result: Any,
    ) -> tuple[bool, str]:
        output = f"{getattr(result, 'stdout_text', '')}\n{getattr(result, 'stderr_text', '')}".lower()
        metrics_specs = [spec for spec in specs if spec.kind == ArtifactKind.metrics and spec.path]
        if intent == StepIntent.run_training:
            has_training_signal = any(marker in output for marker in ("epoch", "train", "loss", "accuracy"))
            has_metrics_file = any((self._resolve_target(spec.path, workspace_path) or Path()).exists() for spec in metrics_specs if spec.path)
            if self.strictness == "strict" and not (has_training_signal or has_metrics_file):
                return False, "run_training intent has no training signal or metrics artifact"
        if intent == StepIntent.verify_metrics:
            expected_keys = {
                key.strip().lower()
                for spec in metrics_specs
                for key in spec.metric_keys
                if key.strip()
            }
            if expected_keys:
                metric_text_parts = [output]
                for target in self._metrics_candidate_paths(
                    specs=metrics_specs,
                    workspace_path=workspace_path,
                    result=result,
                ):
                    try:
                        metric_text_parts.append(target.read_text(encoding="utf-8", errors="ignore"))
                    except OSError:
                        continue
                metric_text = "\n".join(metric_text_parts)
                extracted_pairs = self._extract_metric_pairs(metric_text)
                missing = [
                    key
                    for key in expected_keys
                    if not self._metric_key_present(
                        expected_key=key,
                        text=metric_text,
                        extracted_pairs=extracted_pairs,
                    )
                ]
                if missing:
                    return False, f"verify_metrics intent missing metric keys: {', '.join(missing)}"
            elif self.strictness == "strict":
                if not self._extract_metric_pairs(output):
                    return False, "verify_metrics intent did not produce metric output"
        return True, "intent checks passed"

    def _metric_key_present(self, expected_key: str, text: str, extracted_pairs: dict[str, float]) -> bool:
        normalized_text = self._normalize_metric_text(text)
        haystack = f" {normalized_text} " if normalized_text else ""
        extracted_normalized = {self._normalize_metric_key(key) for key in extracted_pairs}

        for alias in self._metric_key_aliases(expected_key):
            alias_text = self._normalize_metric_text(alias)
            if alias_text and f" {alias_text} " in haystack:
                return True
            alias_key = self._normalize_metric_key(alias)
            if alias_key and alias_key in extracted_normalized:
                return True
        return False

    def _metric_key_aliases(self, key: str) -> set[str]:
        raw_tokens = [token for token in re.split(r"[^a-zA-Z0-9]+", key.lower()) if token]
        if not raw_tokens:
            return set()

        token_aliases: dict[str, list[str]] = {
            "acc": ["acc", "accuracy"],
            "val": ["val", "valid", "validation"],
            "train": ["train", "training"],
            "test": ["test", "testing"],
            "prec": ["prec", "precision"],
            "rec": ["rec", "recall"],
            "f1": ["f1", "f1 score", "f1score"],
            "miou": ["miou", "mean iou", "mean_iou"],
            "iou": ["iou", "jaccard"],
        }

        options = [token_aliases.get(token, [token]) for token in raw_tokens]
        aliases: set[str] = set()
        for combo in product(*options):
            phrase = " ".join(item for item in combo if item).strip()
            if phrase:
                aliases.add(phrase)
        aliases.add(" ".join(raw_tokens))
        aliases.add("_".join(raw_tokens))
        return aliases

    def _normalize_metric_text(self, value: str) -> str:
        normalized = _NON_ALNUM_RE.sub(" ", value.lower())
        return _MULTISPACE_RE.sub(" ", normalized).strip()

    def _normalize_metric_key(self, value: str) -> str:
        normalized = _NON_ALNUM_RE.sub("_", value.lower()).strip("_")
        return re.sub(r"_+", "_", normalized)

    def _extract_metric_pairs(self, text: str) -> dict[str, float]:
        found: dict[str, float] = {}
        for line in text.splitlines():
            normalized = line.replace("**", "").replace("__", "")
            match = _METRIC_LINE_RE.search(normalized)
            if not match:
                continue
            key = match.group("key").strip().lower().replace(" ", "_")
            value_raw = match.group("value")
            try:
                value = float(value_raw)
            except ValueError:
                continue
            if "%" in normalized and value > 1.0:
                value = value / 100.0
            found[key] = value
        return found

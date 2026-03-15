from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.application.services.prompt_content_service import PromptContentService
from orchestrator.persistence.schemas import ArtifactSpec


class CodexPromptSupport:
    def __init__(
        self,
        *,
        prompt_content_service: PromptContentService,
        soft_failure_markers: tuple[str, ...],
    ) -> None:
        self.prompt_content_service = prompt_content_service
        self.soft_failure_markers = soft_failure_markers

    def expected_artifact_paths(self, artifacts: list[ArtifactSpec], workspace_path: Path) -> list[str]:
        paths: list[str] = []
        for spec in artifacts:
            if not spec.path or not spec.must_exist:
                continue
            normalized = self.normalize_expected_path(spec.path, workspace_path)
            if normalized and normalized not in paths:
                paths.append(normalized)
        return paths

    def normalize_expected_path(self, raw_path: str, workspace_path: Path) -> str | None:
        cleaned = raw_path.strip().strip("`\"'")
        if not cleaned:
            return None
        cleaned = cleaned.replace("\\", "/").strip("/")
        parts = [part for part in cleaned.split("/") if part and part != "."]
        if not parts:
            return None
        if parts[0] == "workspace":
            parts = parts[1:]
        workspace_name = workspace_path.name
        if parts and parts[0] == workspace_name:
            parts = parts[1:]
        if not parts:
            return None
        return "/".join(parts)

    def extract_first_code_block(self, text: str) -> str | None:
        if not text:
            return None
        matches = re.findall(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", text, flags=re.DOTALL)
        for block in matches:
            candidate = block.strip("\n")
            if len(candidate) < 4:
                continue
            return candidate + "\n"
        return None

    def learning_notes_path(self, workspace_path: Path) -> Path:
        return workspace_path / "knowledge" / "codex_learning_notes.md"

    def workspace_snapshot_markdown_path(self, workspace_path: Path) -> Path:
        return workspace_path / ".agent" / "workspace_snapshot.md"

    def load_learning_notes(self, workspace_path: Path) -> str:
        path = self.learning_notes_path(workspace_path)
        if not path.exists():
            return ""
        return self.prompt_content_service.render_file_for_prompt(path, purpose="learning_notes")

    def load_workspace_snapshot_summary(self, workspace_path: Path) -> str:
        path = self.workspace_snapshot_markdown_path(workspace_path)
        if not path.exists():
            return ""
        return self.prompt_content_service.render_file_for_prompt(path, purpose="workspace_snapshot")

    def inject_workspace_snapshot(self, base_prompt: str, workspace_path: Path) -> str:
        snapshot = self.load_workspace_snapshot_summary(workspace_path)
        if not snapshot:
            return base_prompt
        header = "Current workspace snapshot (authoritative paths):"
        return f"{header}\n\n{snapshot}\n\nTask prompt:\n{base_prompt}"

    def inject_learning_notes(self, base_prompt: str, workspace_path: Path) -> str:
        notes = self.load_learning_notes(workspace_path)
        if not notes:
            return base_prompt
        header = "Persistent execution notes (learned from previous codex runs):"
        return f"{header}\n\n{notes}\n\nTask prompt:\n{base_prompt}"

    def inject_skill_context(
        self,
        *,
        base_prompt: str,
        workspace_path: Path,
        skill_paths: list[str],
    ) -> tuple[str, list[str]]:
        normalized_skill_paths = [str(item).strip() for item in skill_paths if str(item).strip()]
        if not normalized_skill_paths:
            return base_prompt, []
        loaded_chunks: list[str] = []
        used_paths: list[str] = []
        for raw_path in normalized_skill_paths[:3]:
            resolved = self.resolve_skill_path(raw_path, workspace_path)
            if resolved is None or not resolved.exists():
                continue
            content = self.prompt_content_service.render_file_for_prompt(
                resolved,
                purpose="skill",
            ).strip()
            if not content:
                continue
            loaded_chunks.append(f"### {resolved.parent.name} ({raw_path})\n{content}")
            used_paths.append(raw_path)
        if not loaded_chunks:
            return base_prompt, []
        header = "Selected skill context (apply these instructions before coding):"
        return f"{header}\n\n" + "\n\n".join(loaded_chunks) + f"\n\nTask prompt:\n{base_prompt}", used_paths

    def resolve_skill_path(self, raw_path: str, workspace_path: Path) -> Path | None:
        cleaned = raw_path.strip().strip("`\"'")
        if not cleaned:
            return None
        candidate = Path(cleaned)
        if candidate.is_absolute():
            return candidate
        workspace_candidate = workspace_path / cleaned
        if workspace_candidate.exists():
            return workspace_candidate
        if cleaned.startswith("knowledge/"):
            direct_workspace = workspace_path / cleaned
            if direct_workspace.exists():
                return direct_workspace
        codex_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
        codex_candidate = codex_home / cleaned
        if codex_candidate.exists():
            return codex_candidate
        if not cleaned.endswith("SKILL.md"):
            workspace_matches = sorted((workspace_path / "knowledge" / "skills").glob(f"**/{cleaned}/SKILL.md"))
            if workspace_matches:
                return workspace_matches[0]
            codex_matches = sorted((codex_home / "skills").glob(f"**/{cleaned}/SKILL.md"))
            if codex_matches:
                return codex_matches[0]
        return None

    def append_learning_note(
        self,
        *,
        workspace_path: Path,
        step_id: str,
        step_action: str,
        status: str,
        summary: str,
        stdout_text: str,
        stderr_text: str,
    ) -> None:
        notes_path = self.learning_notes_path(workspace_path)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        normalized_summary = summary.replace("\n", " ").strip()
        if len(normalized_summary) > 240:
            normalized_summary = f"{normalized_summary[:240]}..."
        marker = " soft_failure=true" if self.result_has_codex_soft_failure(stdout_text, stderr_text) else ""
        line = (
            f"- ts={timestamp} step={step_id} action={step_action} status={status}"
            f"{marker} summary={normalized_summary}"
        )
        with notes_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def result_has_codex_soft_failure(self, stdout_text: str, stderr_text: str) -> bool:
        text = f"{stdout_text}\n{stderr_text}".lower()
        return any(marker in text for marker in self.soft_failure_markers)

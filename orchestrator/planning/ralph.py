from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.persistence.schemas import Priority


@dataclass(slots=True)
class RalphStory:
    story_id: str
    title: str
    description: str
    priority: int
    acceptance_criteria: list[str]
    raw: dict[str, Any]


class RalphBacklogError(RuntimeError):
    pass


class RalphBacklogService:
    def __init__(self, prd_file_name: str = "prd.json", progress_file_name: str = "progress.txt") -> None:
        self.prd_file_name = prd_file_name
        self.progress_file_name = progress_file_name

    def load_prd(self, workspace_path: Path) -> dict[str, Any]:
        prd_path = workspace_path / self.prd_file_name
        if not prd_path.is_file():
            raise RalphBacklogError(f"RALPH PRD file not found: {prd_path.as_posix()}")
        try:
            payload = json.loads(prd_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RalphBacklogError(f"invalid RALPH PRD JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RalphBacklogError("RALPH PRD root must be an object")
        stories = payload.get("userStories")
        if not isinstance(stories, list):
            raise RalphBacklogError("RALPH PRD must include 'userStories' array")
        return payload

    def pick_next_story(self, prd: dict[str, Any]) -> RalphStory | None:
        stories = prd.get("userStories") or []
        candidates: list[RalphStory] = []
        for item in stories:
            if not isinstance(item, dict):
                continue
            if bool(item.get("passes")):
                continue
            story_id = str(item.get("id", "")).strip()
            if not story_id:
                continue
            title = str(item.get("name") or item.get("title") or story_id).strip()
            description = str(
                item.get("description")
                or item.get("goal")
                or item.get("asA")
                or item.get("as_a")
                or title
            ).strip()
            acceptance_raw = item.get("acceptanceCriteria") or item.get("acceptance_criteria") or []
            acceptance_criteria: list[str] = []
            if isinstance(acceptance_raw, list):
                acceptance_criteria = [str(v).strip() for v in acceptance_raw if str(v).strip()]
            priority_raw = item.get("priority", 2)
            try:
                priority = int(priority_raw)
            except (TypeError, ValueError):
                priority = 2
            candidates.append(
                RalphStory(
                    story_id=story_id,
                    title=title,
                    description=description,
                    priority=priority,
                    acceptance_criteria=acceptance_criteria,
                    raw=item,
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda s: (s.priority, s.story_id))
        return candidates[0]

    def pick_by_id(self, prd: dict[str, Any], story_id: str) -> RalphStory | None:
        stories = prd.get("userStories") or []
        target = str(story_id).strip()
        if not target:
            return None

        for item in stories:
            if not isinstance(item, dict):
                continue
            if str(item.get("id", "")).strip() != target:
                continue
            title = str(item.get("name") or item.get("title") or target).strip()
            description = str(
                item.get("description")
                or item.get("goal")
                or item.get("asA")
                or item.get("as_a")
                or title
            ).strip()
            acceptance_raw = item.get("acceptanceCriteria") or item.get("acceptance_criteria") or []
            acceptance_criteria: list[str] = []
            if isinstance(acceptance_raw, list):
                acceptance_criteria = [str(v).strip() for v in acceptance_raw if str(v).strip()]
            priority_raw = item.get("priority", 2)
            try:
                priority = int(priority_raw)
            except (TypeError, ValueError):
                priority = 2
            return RalphStory(
                story_id=target,
                title=title,
                description=description,
                priority=priority,
                acceptance_criteria=acceptance_criteria,
                raw=item,
            )
        return None

    def mark_story_passed(self, workspace_path: Path, story_id: str) -> dict[str, Any]:
        prd = self.load_prd(workspace_path)
        updated = False
        stories = prd.get("userStories") or []
        for item in stories:
            if not isinstance(item, dict):
                continue
            if str(item.get("id", "")).strip() != story_id:
                continue
            item["passes"] = True
            updated = True
            break
        if not updated:
            raise RalphBacklogError(f"story not found in PRD: {story_id}")
        prd_path = workspace_path / self.prd_file_name
        prd_path.write_text(json.dumps(prd, ensure_ascii=True, indent=2), encoding="utf-8")
        return prd

    def append_progress(self, workspace_path: Path, line: str) -> None:
        progress_path = workspace_path / self.progress_file_name
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{ts}] {line.strip()}\n")

    def map_story_priority(self, story: RalphStory) -> Priority:
        if story.priority <= 1:
            return Priority.high
        if story.priority >= 3:
            return Priority.low
        return Priority.normal

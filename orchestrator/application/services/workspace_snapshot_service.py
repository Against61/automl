from __future__ import annotations

import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceSnapshotService:
    _ignore_dirs = {
        ".git",
        ".venv",
        ".agent",
        "artifacts",
        "runs",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
        ".DS_Store",
        "venv",
        ".cache",
    }
    _path_marker_re = re.compile(
        r"(?im)^\s*([A-Za-z_][A-Za-z0-9_]*(?:_path|_file|_dir))\s*=\s*([^\n]+?)\s*$"
    )
    _interesting_suffixes = {
        ".md",
        ".json",
        ".yaml",
        ".yml",
        ".py",
        ".txt",
        ".csv",
        ".pth",
        ".pt",
        ".ckpt",
        ".onnx",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }

    def snapshot_json_path(self, workspace_path: Path) -> Path:
        return workspace_path / ".agent" / "workspace_snapshot.json"

    def snapshot_markdown_path(self, workspace_path: Path) -> Path:
        return workspace_path / ".agent" / "workspace_snapshot.md"

    def ralph_snapshot_path(self, workspace_path: Path) -> Path:
        return workspace_path / ".ralph_workspace_tree.md"

    def refresh(
        self,
        workspace_path: Path,
        *,
        step: Any | None = None,
        result: Any | None = None,
    ) -> dict[str, Any]:
        workspace_path.mkdir(parents=True, exist_ok=True)
        tree_text = self._build_tree_snapshot(workspace_path)
        recent_files = self._recent_files(workspace_path)
        touched_files = self._normalize_paths(getattr(result, "files_changed", None), workspace_path)
        detected_output_paths = self._extract_output_paths(
            stdout_text=getattr(result, "stdout_text", ""),
            stderr_text=getattr(result, "stderr_text", ""),
            workspace_path=workspace_path,
        )
        for item in detected_output_paths:
            if item not in touched_files:
                touched_files.append(item)

        expected_artifacts = self._collect_expected_artifacts(step=step, workspace_path=workspace_path)
        basename_candidates = self._collect_basename_candidates(
            workspace_path=workspace_path,
            focus_paths=[item["path"] for item in expected_artifacts if item.get("path")],
        )

        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "workspace_root": workspace_path.name,
            "recent_files": recent_files,
            "touched_files": touched_files,
            "detected_output_paths": detected_output_paths,
            "expected_artifacts": expected_artifacts,
            "basename_candidates": basename_candidates,
            "tree": tree_text,
        }
        self._write_payload(workspace_path, payload)
        return payload

    def load_or_refresh(self, workspace_path: Path) -> dict[str, Any]:
        snapshot_path = self.snapshot_json_path(workspace_path)
        if snapshot_path.exists():
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return payload
            except (OSError, json.JSONDecodeError):
                pass
        return self.refresh(workspace_path)

    def render_prompt_summary(self, payload: dict[str, Any]) -> str:
        recent_files = payload.get("recent_files") if isinstance(payload.get("recent_files"), list) else []
        touched_files = payload.get("touched_files") if isinstance(payload.get("touched_files"), list) else []
        detected_output_paths = (
            payload.get("detected_output_paths") if isinstance(payload.get("detected_output_paths"), list) else []
        )
        expected_artifacts = (
            payload.get("expected_artifacts") if isinstance(payload.get("expected_artifacts"), list) else []
        )
        basename_candidates = (
            payload.get("basename_candidates") if isinstance(payload.get("basename_candidates"), dict) else {}
        )
        tree = str(payload.get("tree") or "").strip()
        tree_lines = tree.splitlines()[:40]

        lines = [
            f"Workspace snapshot refreshed at: {payload.get('generated_at', 'unknown')}",
            f"Workspace root: {payload.get('workspace_root', 'workspace')}",
        ]
        if touched_files:
            lines.append("Recently changed files:")
            lines.extend(f"- {item}" for item in touched_files[:20])
        if detected_output_paths:
            lines.append("Detected output paths from command output:")
            lines.extend(f"- {item}" for item in detected_output_paths[:20])
        if expected_artifacts:
            lines.append("Expected artifact status:")
            for item in expected_artifacts[:20]:
                exists_label = "exists" if item.get("exists") else "missing"
                lines.append(f"- {item.get('path')}: {exists_label}")
        if basename_candidates:
            lines.append("Path candidates by basename:")
            for key, values in list(basename_candidates.items())[:10]:
                lines.append(f"- {key}: {', '.join(values[:4])}")
        if recent_files:
            lines.append("Recent workspace files:")
            lines.extend(f"- {item}" for item in recent_files[:20])
        if tree_lines:
            lines.append("Workspace tree excerpt:")
            lines.extend(tree_lines)
        return "\n".join(lines)

    def get_prompt_summary(self, workspace_path: Path) -> str:
        payload = self.load_or_refresh(workspace_path)
        return self.render_prompt_summary(payload)

    def _write_payload(self, workspace_path: Path, payload: dict[str, Any]) -> None:
        json_path = self.snapshot_json_path(workspace_path)
        md_path = self.snapshot_markdown_path(workspace_path)
        ralph_path = self.ralph_snapshot_path(workspace_path)
        try:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
            md_path.write_text(self.render_prompt_summary(payload), encoding="utf-8")
            ralph_path.write_text(str(payload.get("tree") or ""), encoding="utf-8")
        except OSError:
            return

    def _collect_expected_artifacts(self, *, step: Any | None, workspace_path: Path) -> list[dict[str, Any]]:
        if step is None:
            return []
        artifacts = getattr(step, "expected_artifacts", None) or []
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for spec in artifacts:
            raw_path = getattr(spec, "path", None)
            normalized = self._normalize_path(raw_path, workspace_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            abs_path = workspace_path / normalized
            exists = abs_path.exists()
            size = None
            if exists and abs_path.is_file():
                try:
                    size = abs_path.stat().st_size
                except OSError:
                    size = None
            items.append(
                {
                    "path": normalized,
                    "exists": exists,
                    "size": size,
                    "kind": str(getattr(spec, "kind", "file")),
                }
            )
        return items

    def _collect_basename_candidates(self, *, workspace_path: Path, focus_paths: list[str]) -> dict[str, list[str]]:
        basenames = {Path(item).name for item in focus_paths if item}
        if not basenames:
            return {}
        candidates: dict[str, list[str]] = {}
        for path in workspace_path.rglob("*"):
            if not path.is_file():
                continue
            if self._should_ignore(path, workspace_path):
                continue
            if path.name not in basenames:
                continue
            rel = path.relative_to(workspace_path).as_posix()
            candidates.setdefault(path.name, [])
            if rel not in candidates[path.name]:
                candidates[path.name].append(rel)
        return candidates

    def _recent_files(self, workspace_path: Path, limit: int = 50) -> list[str]:
        files: list[tuple[float, str]] = []
        for path in workspace_path.rglob("*"):
            if not path.is_file():
                continue
            if self._should_ignore(path, workspace_path):
                continue
            try:
                files.append((path.stat().st_mtime, path.relative_to(workspace_path).as_posix()))
            except OSError:
                continue
        files.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in files[:limit]]

    def _extract_output_paths(self, *, stdout_text: str, stderr_text: str, workspace_path: Path) -> list[str]:
        text = "\n".join([stdout_text or "", stderr_text or ""])
        results: list[str] = []
        seen: set[str] = set()
        for _, raw_path in self._path_marker_re.findall(text):
            normalized = self._normalize_output_marker_path(raw_path, workspace_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results

    def _normalize_output_marker_path(self, raw_path: str, workspace_path: Path) -> str | None:
        value = str(raw_path or "").strip().strip("`\"'")
        if not value:
            return None
        value = value.split()[0].strip()
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(workspace_path.resolve()).as_posix()
            except ValueError:
                return None
        normalized = self._normalize_path(value, workspace_path)
        if not normalized:
            return None
        suffix = Path(normalized).suffix.lower()
        if suffix in self._interesting_suffixes or "/" in normalized:
            return normalized
        return None

    def _normalize_paths(self, values: list[str] | None, workspace_path: Path) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            normalized = self._normalize_path(item, workspace_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results

    def _normalize_path(self, raw_path: str | None, workspace_path: Path) -> str | None:
        if not raw_path:
            return None
        cleaned = str(raw_path).strip().strip("`\"'")
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

    def _build_tree_snapshot(self, workspace_path: Path, max_depth: int = 4) -> str:
        root = workspace_path.resolve()
        if not root.is_dir():
            return f"{workspace_path.name}/"

        lines = [f"{workspace_path.name}/"]
        queue: deque[tuple[Path, str, int]] = deque([(root, "", 0)])
        while queue:
            current_path, prefix, depth = queue.popleft()
            if depth >= max_depth:
                continue
            try:
                entries = sorted(
                    [entry for entry in current_path.iterdir() if entry.name not in self._ignore_dirs],
                    key=lambda path: (path.is_file(), path.name.lower()),
                )
            except OSError:
                continue
            total = len(entries)
            for idx, entry in enumerate(entries):
                is_last = idx == total - 1
                connector = "└── " if is_last else "├── "
                line = f"{prefix}{connector}{entry.name}"
                if entry.is_dir():
                    line += "/"
                    lines.append(line)
                    child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                    queue.append((entry, child_prefix, depth + 1))
                else:
                    lines.append(line)
        return "\n".join(lines)

    def _should_ignore(self, path: Path, workspace_path: Path) -> bool:
        try:
            rel = path.relative_to(workspace_path)
        except ValueError:
            return True
        return any(part in self._ignore_dirs for part in rel.parts[:-1])

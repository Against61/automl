from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


class PromptContentService:
    _full_code_file_bytes = 64 * 1024
    _full_text_file_bytes = 4 * 1024
    _max_chars = 6_000
    _max_focus_lines_per_term = 3
    _max_preview_lines = 20
    _max_json_parse_bytes = 512 * 1024
    _max_compact_items = 8
    _max_compact_depth = 4
    _max_compact_string_chars = 240
    _code_suffixes = {
        ".py",
        ".sh",
        ".bash",
        ".zsh",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cc",
        ".cpp",
        ".h",
        ".hpp",
    }
    _structured_suffixes = {
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
    }
    _binary_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".pdf",
        ".onnx",
        ".pt",
        ".pth",
        ".ckpt",
        ".bin",
        ".npy",
        ".npz",
        ".zip",
        ".tar",
        ".gz",
    }
    _default_focus_terms = (
        "accuracy",
        "iou",
        "miou",
        "mean_iou",
        "jaccard",
        "loss",
        "precision",
        "recall",
        "f1",
        "threshold",
        "split",
        "train",
        "eval",
        "test",
        "val",
    )

    def render_file_for_prompt(
        self,
        path: Path,
        *,
        purpose: str = "generic",
        focus_terms: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        if not path.exists() or not path.is_file():
            return ""
        try:
            size = path.stat().st_size
        except OSError:
            return ""
        suffix = path.suffix.lower()
        normalized_focus = self._normalize_focus_terms(focus_terms)
        relative_name = path.as_posix()

        if purpose == "learning_notes":
            return self._render_line_window(
                path,
                label=relative_name,
                size=size,
                max_lines=60,
                window="tail",
            )
        if purpose in {"workspace_snapshot", "skill"}:
            if size <= self._full_text_file_bytes:
                return self._truncate(self._safe_read_text(path))
            return self._render_line_window(
                path,
                label=relative_name,
                size=size,
                max_lines=80,
                window="head",
                focus_terms=normalized_focus,
            )
        if suffix in self._binary_suffixes:
            return f"[binary artifact omitted] file={relative_name} size_bytes={size}"
        if suffix in self._code_suffixes and size <= self._full_code_file_bytes:
            return self._truncate(self._safe_read_text(path))
        if suffix in self._structured_suffixes:
            return self._render_structured_file(
                path,
                label=relative_name,
                size=size,
                focus_terms=normalized_focus,
            )
        if size <= self._full_text_file_bytes:
            return self._truncate(self._safe_read_text(path))
        return self._render_line_window(
            path,
            label=relative_name,
            size=size,
            max_lines=self._max_preview_lines,
            window="head_tail",
            focus_terms=normalized_focus,
        )

    def compact_json_for_prompt(
        self,
        value: Any,
        *,
        focus_terms: list[str] | tuple[str, ...] | None = None,
    ) -> Any:
        return self._compact_value(
            value,
            focus_terms=self._normalize_focus_terms(focus_terms),
            depth=0,
        )

    def compact_text_for_prompt(
        self,
        text: str | None,
        *,
        label: str = "text",
        focus_terms: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if len(raw) <= self._max_chars:
            return raw
        normalized_focus = self._normalize_focus_terms(focus_terms)
        lines = raw.splitlines()
        focus_hits: list[str] = []
        counts = {term: 0 for term in normalized_focus}
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            for term in normalized_focus:
                if term in lowered and counts[term] < self._max_focus_lines_per_term:
                    focus_hits.append(stripped[:240])
                    counts[term] += 1
                    break
        head = [line.strip()[:240] for line in lines[:10] if line.strip()]
        tail = [line.strip()[:240] for line in lines[-10:] if line.strip()]
        rendered: list[str] = [f"[compact {label}] original_chars={len(raw)}"]
        if focus_hits:
            rendered.append("Focused lines:")
            rendered.extend(f"- {line}" for line in focus_hits[:12])
        rendered.append("Head:")
        rendered.extend(f"- {line}" for line in head[:10])
        if tail:
            rendered.append("Tail:")
            rendered.extend(f"- {line}" for line in tail[:10])
        return self._truncate("\n".join(rendered))

    def _render_structured_file(
        self,
        path: Path,
        *,
        label: str,
        size: int,
        focus_terms: tuple[str, ...],
    ) -> str:
        if size <= self._full_text_file_bytes:
            return self._truncate(self._safe_read_text(path))
        if size <= self._max_json_parse_bytes:
            try:
                payload = json.loads(self._safe_read_text(path))
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                summary = self._summarize_json_payload(payload, focus_terms=focus_terms)
                rendered = "\n".join(
                    [
                        f"[structured artifact summary] file={label} size_bytes={size}",
                        *summary,
                    ]
                )
                return self._truncate(rendered)
        return self._render_line_window(
            path,
            label=label,
            size=size,
            max_lines=self._max_preview_lines,
            window="head_tail",
            focus_terms=focus_terms,
        )

    def _summarize_json_payload(
        self,
        payload: Any,
        *,
        focus_terms: tuple[str, ...],
    ) -> list[str]:
        lines: list[str] = []
        if isinstance(payload, dict):
            lines.append(f"top_level_type=dict keys={len(payload)}")
            if payload:
                lines.append("top_level_keys=" + ", ".join(str(key) for key in list(payload.keys())[:20]))
        elif isinstance(payload, list):
            lines.append(f"top_level_type=list items={len(payload)}")
        else:
            lines.append(f"top_level_type={type(payload).__name__}")

        focused, generic = self._collect_scalar_paths(payload, focus_terms=focus_terms)
        if focused:
            lines.append("focused_scalars:")
            lines.extend(f"- {path} = {value}" for path, value in focused[:12])
        if generic:
            lines.append("scalar_preview:")
            lines.extend(f"- {path} = {value}" for path, value in generic[:12])
        if not focused and not generic:
            lines.append("scalar_preview: none")
        return lines

    def _collect_scalar_paths(
        self,
        payload: Any,
        *,
        focus_terms: tuple[str, ...],
    ) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
        focused: list[tuple[str, Any]] = []
        generic: list[tuple[str, Any]] = []
        queue: deque[tuple[str, Any, int]] = deque([("$", payload, 0)])
        seen = 0
        while queue and seen < 64:
            path, value, depth = queue.popleft()
            seen += 1
            if isinstance(value, dict):
                for key, child in list(value.items())[: self._max_compact_items]:
                    queue.append((f"{path}.{key}", child, depth + 1))
                if len(value) > self._max_compact_items:
                    generic.append((f"{path}.__truncated_keys__", len(value) - self._max_compact_items))
                continue
            if isinstance(value, list):
                generic.append((path, f"list({len(value)})"))
                continue
            if isinstance(value, str):
                scalar_value = self._truncate_scalar(value)
            elif isinstance(value, bool | int | float) or value is None:
                scalar_value = value
            else:
                scalar_value = type(value).__name__
            target = focused if self._path_matches_focus(path, focus_terms) else generic
            target.append((path, scalar_value))
        return focused, generic

    def _render_line_window(
        self,
        path: Path,
        *,
        label: str,
        size: int,
        max_lines: int,
        window: str,
        focus_terms: tuple[str, ...] = (),
    ) -> str:
        head: list[tuple[int, str]] = []
        tail: deque[tuple[int, str]] = deque(maxlen=max_lines)
        focus_hits: list[tuple[int, str]] = []
        focus_counts = {term: 0 for term in focus_terms}
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line_no, raw in enumerate(handle, start=1):
                    line = raw.rstrip()
                    if not line:
                        continue
                    clipped = line[:240]
                    if len(head) < max_lines:
                        head.append((line_no, clipped))
                    tail.append((line_no, clipped))
                    lowered = clipped.lower()
                    for term in focus_terms:
                        if term in lowered and focus_counts[term] < self._max_focus_lines_per_term:
                            focus_hits.append((line_no, clipped))
                            focus_counts[term] += 1
                            break
        except OSError:
            return ""

        lines: list[str] = [f"[compact file summary] file={label} size_bytes={size}"]
        if focus_hits:
            lines.append("Focused matches:")
            for line_no, text in focus_hits[:12]:
                lines.append(f"- L{line_no}: {text}")
        if window in {"head", "head_tail"} and head:
            lines.append("Head:")
            for line_no, text in head[:max_lines]:
                lines.append(f"- L{line_no}: {text}")
        if window in {"tail", "head_tail"} and tail:
            lines.append("Tail:")
            for line_no, text in list(tail)[-max_lines:]:
                lines.append(f"- L{line_no}: {text}")
        return self._truncate("\n".join(lines))

    def _compact_value(
        self,
        value: Any,
        *,
        focus_terms: tuple[str, ...],
        depth: int,
    ) -> Any:
        if depth >= self._max_compact_depth:
            return f"<truncated depth={depth}>"
        if isinstance(value, dict):
            items = list(value.items())
            if focus_terms:
                items.sort(key=lambda item: (0 if self._path_matches_focus(str(item[0]), focus_terms) else 1))
            compact: dict[str, Any] = {}
            for key, child in items[: self._max_compact_items]:
                compact[str(key)] = self._compact_value(child, focus_terms=focus_terms, depth=depth + 1)
            if len(items) > self._max_compact_items:
                compact["__truncated_keys__"] = len(items) - self._max_compact_items
            return compact
        if isinstance(value, list):
            compact_items = [
                self._compact_value(item, focus_terms=focus_terms, depth=depth + 1)
                for item in value[: self._max_compact_items]
            ]
            if len(value) > self._max_compact_items:
                compact_items.append(f"... (+{len(value) - self._max_compact_items} more items)")
            return compact_items
        if isinstance(value, str):
            return self._truncate_scalar(value)
        return value

    def _truncate_scalar(self, value: str) -> str:
        if len(value) <= self._max_compact_string_chars:
            return value
        return value[: self._max_compact_string_chars] + f"... (+{len(value) - self._max_compact_string_chars} chars)"

    def _safe_read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return ""

    def _normalize_focus_terms(
        self,
        focus_terms: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        raw_terms = list(focus_terms or self._default_focus_terms)
        normalized = [str(term).strip().lower() for term in raw_terms if str(term).strip()]
        return tuple(dict.fromkeys(normalized))

    def _path_matches_focus(self, path: str, focus_terms: tuple[str, ...]) -> bool:
        lowered = path.lower()
        return any(term in lowered for term in focus_terms)

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars] + f"\n... [truncated {len(text) - self._max_chars} chars]"

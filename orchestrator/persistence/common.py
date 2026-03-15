from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def json_loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)


@dataclass(slots=True)
class PdfChunkRow:
    chunk_id: int
    document_path: str
    page_number: int
    chunk_index: int
    text: str
    embedding: list[float] | None

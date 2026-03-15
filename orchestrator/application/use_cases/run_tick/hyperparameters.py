from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from orchestrator.persistence.db import Database


class HyperparameterService:
    def __init__(self, db: Database) -> None:
        self.db = db

    async def build_context(self, run_id: str) -> dict[str, Any]:
        try:
            steps = await self.db.list_run_steps(run_id)
        except Exception:
            return {}

        attempts: list[dict[str, Any]] = []
        last_signature: str | None = None
        for row in steps:
            command = str(row.get("command") or "").strip()
            if not command:
                continue
            if self.is_llm_or_agent_command(command):
                continue
            hyperparameters = self.extract_from_command(command)
            if not hyperparameters:
                continue
            signature = json.dumps(hyperparameters, ensure_ascii=True, sort_keys=True)
            if signature == last_signature:
                continue
            last_signature = signature
            attempts.append(
                {
                    "step_id": str(row.get("step_id") or ""),
                    "step_index": int(row.get("step_index") or 0),
                    "status": str(row.get("status") or ""),
                    "command": command,
                    "hyperparameters": hyperparameters,
                }
            )

        if not attempts:
            return {}
        return {
            "latest_hyperparameters": attempts[-1]["hyperparameters"],
            "hyperparameter_attempts": attempts[-10:],
        }

    @classmethod
    def extract_from_command(cls, command: str) -> dict[str, Any]:
        if not command:
            return {}
        if cls.is_llm_or_agent_command(command):
            return {}
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()

        aliases = {
            "epochs": "epochs",
            "epoch": "epochs",
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "learning-rate": "learning_rate",
            "batch_size": "batch_size",
            "batch-size": "batch_size",
            "bs": "batch_size",
            "test_batch_size": "test_batch_size",
            "test-batch-size": "test_batch_size",
            "optimizer": "optimizer",
            "optim": "optimizer",
            "weight_decay": "weight_decay",
            "weight-decay": "weight_decay",
            "wd": "weight_decay",
            "momentum": "momentum",
            "dropout": "dropout",
            "model": "model",
            "model_name": "model",
            "seed": "seed",
            "workers": "workers",
            "num_workers": "workers",
            "num-workers": "workers",
            "img_size": "img_size",
            "image_size": "img_size",
            "image-size": "img_size",
        }
        params: dict[str, Any] = {}

        def normalize_key(raw: str) -> str | None:
            key = raw.strip().lstrip("-").replace(".", "_").replace(" ", "_").lower()
            return aliases.get(key)

        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token.startswith("--"):
                flag = token[2:]
                value: str | None = None
                if "=" in flag:
                    key_raw, value = flag.split("=", 1)
                else:
                    key_raw = flag
                    if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                        value = tokens[idx + 1]
                        idx += 1
                    else:
                        value = "true"
                normalized = normalize_key(key_raw)
                if normalized:
                    parsed = cls.coerce_value(value)
                    if normalized == "model" and cls.looks_like_llm_model_name(parsed):
                        idx += 1
                        continue
                    params[normalized] = parsed
            elif "=" in token and not token.startswith("-"):
                key_raw, value = token.split("=", 1)
                normalized = normalize_key(key_raw)
                if normalized:
                    parsed = cls.coerce_value(value)
                    if normalized == "model" and cls.looks_like_llm_model_name(parsed):
                        idx += 1
                        continue
                    params[normalized] = parsed
            idx += 1
        return params

    @staticmethod
    def is_llm_or_agent_command(command: str) -> bool:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if not tokens:
            return False
        first = Path(tokens[0]).name.lower()
        if first in {"codex", "claude", "chatgpt"}:
            return True
        if first.startswith("codex-"):
            return True
        joined = " ".join(tokens[:4]).lower()
        if "codex exec" in joined:
            return True
        return False

    @staticmethod
    def looks_like_llm_model_name(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        raw = value.strip().lower()
        if not raw:
            return False
        if raw.startswith(("gpt-", "o1", "o3", "o4", "codex", "claude", "gemini")):
            return True
        if raw.endswith("-codex") or raw.endswith("-spark"):
            return True
        return False

    @staticmethod
    def coerce_value(value: str | None) -> Any:
        if value is None:
            return None
        raw = str(value).strip()
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if re.fullmatch(r"[-+]?\d+", raw):
            try:
                return int(raw)
            except ValueError:
                return raw
        if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", raw) or re.fullmatch(
            r"[-+]?\d+(?:[eE][-+]?\d+)", raw
        ):
            try:
                return float(raw)
            except ValueError:
                return raw
        return raw

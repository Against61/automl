from __future__ import annotations

import re
import shlex
from pathlib import Path


class ShellCommandNormalizer:
    _OUTPUT_PATH_FLAGS = {
        "--metrics-path",
        "--output",
        "--output-path",
        "--results-path",
        "--report-path",
        "--artifact-path",
        "--preflight-metrics-path",
    }
    _OUTPUT_FILENAMES = {
        "metrics.json",
        "preflight_metrics.json",
        "results.json",
        "metrics.md",
        "metrics.markdown",
    }
    _RUN_SCOPED_METRICS_RE = re.compile(
        r"^(?:\./)?\.openin/runs/[0-9a-fA-F-]{36}/(?P<name>(?:preflight_)?metrics\.json)$"
    )

    def shell_primary_binary(self, command: str) -> str:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        for token in tokens:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token):
                continue
            return Path(token).name.lower()
        return ""

    def normalize_python_runtime_command(self, command: str, workspace_path: Path) -> str:
        if not command:
            return command
        if any(op in command for op in ("&&", "||", ";", "|")):
            return command
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command
        if not tokens:
            return command

        py_idx = self.python_token_index(tokens)
        if py_idx is None:
            return command

        self.normalize_smoke_flag(tokens, py_idx)
        script_idx = self.python_script_index(tokens, py_idx)
        if script_idx >= len(tokens):
            return " ".join(shlex.quote(token) for token in tokens)
        script_token = tokens[script_idx]
        if script_token.startswith("-") or script_token == "-m" or not script_token.endswith(".py"):
            return " ".join(shlex.quote(token) for token in tokens)

        resolved = self.resolve_python_script_token(script_token, workspace_path)
        if resolved:
            tokens[script_idx] = resolved
        self.maybe_switch_script_to_module(
            tokens=tokens,
            script_idx=script_idx,
            workspace_path=workspace_path,
        )
        self.ensure_workspace_pythonpath(tokens, py_idx)
        return " ".join(shlex.quote(token) for token in tokens)

    def ensure_workspace_pythonpath(self, tokens: list[str], python_idx: int) -> None:
        for idx, token in enumerate(tokens):
            if idx >= python_idx:
                break
            if token.startswith("PYTHONPATH="):
                return
        tokens.insert(python_idx, "PYTHONPATH=.")

    def python_script_index(self, tokens: list[str], python_idx: int) -> int:
        idx = python_idx + 1
        while idx < len(tokens):
            token = tokens[idx]
            if token in {"-u", "-B", "-E", "-s", "-S", "-O", "-OO", "-X", "-W", "-q", "-I", "-P"}:
                idx += 1
                if token in {"-X", "-W"} and idx < len(tokens):
                    idx += 1
                continue
            if token.startswith("-"):
                idx += 1
                continue
            break
        return idx

    def maybe_switch_script_to_module(
        self,
        *,
        tokens: list[str],
        script_idx: int,
        workspace_path: Path,
    ) -> None:
        if script_idx >= len(tokens):
            return
        script_token = tokens[script_idx]
        if script_token.startswith("-"):
            return
        script_path = Path(script_token)
        if script_path.is_absolute():
            try:
                rel_path = script_path.resolve().relative_to(workspace_path.resolve())
            except ValueError:
                return
        else:
            rel_path = script_path

        if rel_path.suffix != ".py" or len(rel_path.parts) < 2:
            return
        module_root = rel_path.parts[0]
        module_name = ".".join(rel_path.with_suffix("").parts)
        source_path = workspace_path / rel_path
        if not source_path.exists():
            return
        if not self.script_uses_module_root(source_path, module_root):
            return
        tokens[script_idx : script_idx + 1] = ["-m", module_name]

    def script_uses_module_root(self, script_path: Path, module_root: str) -> bool:
        pattern = re.compile(rf"(^|\n)\s*(from|import)\s+{re.escape(module_root)}(\.|\\b)")
        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return False
        return bool(pattern.search(text))

    def python_token_index(self, tokens: list[str]) -> int | None:
        for idx, token in enumerate(tokens):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token):
                continue
            binary = Path(token).name.lower()
            if binary in {"python", "python3"}:
                return idx
            return None
        return None

    def normalize_smoke_flag(self, tokens: list[str], python_idx: int) -> None:
        if "--mode" in tokens:
            return
        normalized: list[str] = []
        replaced = False
        for idx, token in enumerate(tokens):
            if idx <= python_idx:
                normalized.append(token)
                continue
            if token in {"--smoke_test", "--smoke-test"}:
                if not replaced:
                    normalized.extend(["--mode", "smoke"])
                    replaced = True
                continue
            normalized.append(token)
        if replaced:
            tokens[:] = normalized

    def resolve_python_script_token(self, script_token: str, workspace_path: Path) -> str | None:
        script_path = Path(script_token)
        if script_path.is_absolute():
            if script_path.exists():
                return script_token
        else:
            candidate = (workspace_path / script_path).resolve()
            if candidate.exists():
                return script_token

        fallback = self.find_python_script_candidate(script_path.name, workspace_path)
        if fallback is None:
            return None
        try:
            return fallback.relative_to(workspace_path.resolve()).as_posix()
        except ValueError:
            return fallback.as_posix()

    def find_python_script_candidate(self, filename: str, workspace_path: Path) -> Path | None:
        root = workspace_path.resolve()
        exact_matches = [
            path for path in root.rglob(filename)
            if path.is_file() and path.suffix == ".py"
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            exact_matches.sort(key=lambda p: len(p.as_posix()))
            return exact_matches[0]

        target_stem = re.sub(r"[^a-z0-9]", "", Path(filename).stem.lower())
        if not target_stem:
            return None
        fuzzy: list[Path] = []
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            stem = re.sub(r"[^a-z0-9]", "", path.stem.lower())
            if stem == target_stem:
                fuzzy.append(path)
        if len(fuzzy) == 1:
            return fuzzy[0]
        if len(fuzzy) > 1:
            fuzzy.sort(key=lambda p: len(p.as_posix()))
            return fuzzy[0]
        return None

    def sanitize_shell_command(self, command: str, workspace_path: Path) -> str:
        if not command:
            return command

        workspace_name = workspace_path.name
        workspace_abs = workspace_path.resolve().as_posix().rstrip("/")
        if not workspace_name:
            return command

        sanitized = command
        sanitized = sanitized.replace(f"{workspace_abs}/", "./")
        sanitized = sanitized.replace(f"{workspace_abs}", ".")

        workspace_root_token = re.escape(workspace_name)
        sanitized = re.sub(
            rf"(^|[\s\"'\(\[])(?:\./)?workspace/{workspace_root_token}(?=\s|/|$|[;&|)])",
            lambda m: f"{m.group(1)}.",
            sanitized,
        )
        sanitized = re.sub(
            rf"(^|[\s\"'\(\[])(?:\./)?{workspace_root_token}(?=(/|$|[\s;&|)\]]))",
            lambda m: f"{m.group(1)}.",
            sanitized,
        )
        sanitized = self.rewrite_missing_relative_paths(sanitized, workspace_path)
        return sanitized

    def rewrite_run_scoped_metrics_paths(self, command: str, *, run_id: str) -> str:
        if not command or not str(run_id).strip():
            return command
        if any(op in command for op in ("&&", "||", ";", "|")):
            return command
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command
        if not tokens:
            return command

        rewritten = False
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token in self._OUTPUT_PATH_FLAGS and idx + 1 < len(tokens):
                replacement = self._rewrite_metrics_output_token(tokens[idx + 1], run_id=run_id)
                if replacement != tokens[idx + 1]:
                    tokens[idx + 1] = replacement
                    rewritten = True
                idx += 2
                continue
            for flag in self._OUTPUT_PATH_FLAGS:
                prefix = f"{flag}="
                if token.startswith(prefix):
                    original_value = token[len(prefix) :]
                    replacement = self._rewrite_metrics_output_token(original_value, run_id=run_id)
                    if replacement != original_value:
                        tokens[idx] = f"{prefix}{replacement}"
                        rewritten = True
                    break
            idx += 1
        if not rewritten:
            return command
        return " ".join(shlex.quote(token) for token in tokens)

    def _rewrite_metrics_output_token(self, token: str, *, run_id: str) -> str:
        cleaned = str(token or "").strip()
        if not cleaned:
            return cleaned
        normalized = cleaned.replace("\\", "/").strip()
        basename = Path(normalized).name.lower()
        if basename not in {"metrics.json", "preflight_metrics.json"}:
            return cleaned
        if basename == "preflight_metrics.json":
            return f".openin/runs/{run_id}/preflight_metrics.json"
        if normalized in {"metrics.json", "./metrics.json"}:
            return f".openin/runs/{run_id}/metrics.json"
        if self._RUN_SCOPED_METRICS_RE.match(normalized):
            return f".openin/runs/{run_id}/metrics.json"
        return cleaned

    def rewrite_missing_relative_paths(self, command: str, workspace_path: Path) -> str:
        if any(op in command for op in ("&&", "||", ";", "|")):
            return command
        try:
            tokens = shlex.split(command)
        except ValueError:
            return command
        if not tokens:
            return command

        rewritten = False
        result_tokens: list[str] = []
        for idx, token in enumerate(tokens):
            if idx == 0 or token.startswith("-") or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token):
                result_tokens.append(token)
                continue
            if self.should_preserve_output_path_token(tokens=tokens, idx=idx):
                result_tokens.append(token)
                continue
            replacement = self.resolve_relative_file_argument(token, workspace_path)
            if replacement and replacement != token:
                result_tokens.append(replacement)
                rewritten = True
            else:
                result_tokens.append(token)
        if not rewritten:
            return command
        return " ".join(shlex.quote(token) for token in result_tokens)

    def should_preserve_output_path_token(self, *, tokens: list[str], idx: int) -> bool:
        token = tokens[idx]
        basename = Path(token).name.lower()
        if basename not in self._OUTPUT_FILENAMES:
            return False
        if idx > 0 and tokens[idx - 1] in self._OUTPUT_PATH_FLAGS:
            return True
        if idx > 0 and tokens[idx - 1] in {">", ">>", "1>", "2>"}:
            return True
        if basename in {"metrics.json", "preflight_metrics.json"}:
            return True
        return False

    def resolve_relative_file_argument(self, token: str, workspace_path: Path) -> str | None:
        if not token or token.startswith(("/", "~")):
            return None
        if any(ch in token for ch in "*?[]{}|"):
            return None
        path_token = Path(token)
        if path_token.exists():
            return None
        candidate = workspace_path / path_token
        if candidate.exists():
            return None
        if not ("/" in token or path_token.suffix):
            return None

        fallback = self.find_workspace_file_candidate(path_token.name, workspace_path)
        if fallback is None:
            return None
        try:
            return fallback.relative_to(workspace_path.resolve()).as_posix()
        except ValueError:
            return fallback.as_posix()

    def find_workspace_file_candidate(self, filename: str, workspace_path: Path) -> Path | None:
        root = workspace_path.resolve()
        matches = [path for path in root.rglob(filename) if path.is_file()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            matches.sort(key=lambda p: len(p.as_posix()))
            return matches[0]
        return None

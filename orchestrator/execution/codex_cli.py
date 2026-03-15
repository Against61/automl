from __future__ import annotations

import os


def normalize_codex_command(base_cmd: list[str], *, model: str = "") -> list[str]:
    if not base_cmd:
        return ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"]

    binary = os.path.basename(base_cmd[0])
    if binary != "codex":
        return base_cmd

    known_subcommands = {"exec", "login", "logout", "completion", "mcp", "sandbox", "help"}
    tail = base_cmd[1:]
    if not tail:
        normalized = [base_cmd[0], "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"]
        if model.strip():
            normalized.extend(["--model", model.strip()])
        return normalized

    subcommand = None
    for token in tail:
        if token.startswith("-"):
            continue
        subcommand = token
        break

    if subcommand in known_subcommands:
        if subcommand != "exec":
            return base_cmd
        normalized = list(base_cmd)
        has_model_flag = any(
            token in {"-m", "--model"} or token.startswith("--model=")
            for token in normalized
        )
        has_danger_flag = "--dangerously-bypass-approvals-and-sandbox" in normalized
        has_full_auto = "--full-auto" in normalized
        has_sandbox_flag = (
            "--sandbox" in normalized
            or "-s" in normalized
            or any(token.startswith("--sandbox=") for token in normalized)
        )
        if model.strip() and not has_model_flag:
            normalized.extend(["--model", model.strip()])
        if has_full_auto and not has_danger_flag and not has_sandbox_flag:
            normalized = [token for token in normalized if token != "--full-auto"]
            normalized.append("--dangerously-bypass-approvals-and-sandbox")
        elif not has_full_auto and not has_danger_flag and not has_sandbox_flag:
            normalized.append("--dangerously-bypass-approvals-and-sandbox")
        if "--skip-git-repo-check" not in normalized:
            normalized.append("--skip-git-repo-check")
        return normalized

    normalized = [base_cmd[0], "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", *tail]
    has_model_flag = any(
        token in {"-m", "--model"} or token.startswith("--model=")
        for token in normalized
    )
    if model.strip() and not has_model_flag:
        normalized.extend(["--model", model.strip()])
    return normalized

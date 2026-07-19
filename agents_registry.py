"""Live Claude Code session registry, via `claude agents --json`.

Turns a bare session_id into something a human recognizes: the session's name
("voice hub"), its project, and whether it is an interactive
or a background session. Cached and non-blocking by contract: the hub must stay
useful when this returns nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

CLAUDE_EXE = os.environ.get("CLAUDE_CODE_EXECPATH") or "claude"
REFRESH_S = 10.0
TIMEOUT_S = 8.0

_cache: dict[str, dict] = {}
_cache_ts = 0.0


def _query() -> list[dict]:
    proc = subprocess.run(
        [CLAUDE_EXE, "agents", "--json"],
        capture_output=True, text=True, timeout=TIMEOUT_S,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    data = json.loads(proc.stdout)
    return data if isinstance(data, list) else []


def refresh(force: bool = False) -> dict[str, dict]:
    """Refresh the sessionId -> record map. Never raises."""
    global _cache, _cache_ts
    if not force and (time.time() - _cache_ts) < REFRESH_S:
        return _cache
    try:
        _cache = {r["sessionId"]: r for r in _query() if r.get("sessionId")}
    except Exception:
        pass  # keep the previous cache; enrichment is best-effort
    _cache_ts = time.time()
    return _cache


def lookup(session_id: str) -> dict | None:
    if not session_id:
        return None
    return refresh().get(session_id)


def describe(session_id: str, fallback_cwd: str = "") -> tuple[str, str]:
    """Return (session_name, project) for display. Empty strings when unknown."""
    rec = lookup(session_id) or {}
    name = rec.get("name", "")
    cwd = rec.get("cwd", "") or fallback_cwd
    project = os.path.basename(cwd.rstrip("\\/")) if cwd else ""
    return name, project


def resume_command(session_id: str) -> str:
    return f"claude --resume {session_id}"

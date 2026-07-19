"""Event spool + session identity for the Voice Hub.

say.py writes one JSON file per event (queued/speaking/done); hub.py
consumes them. Identity comes from the environment Claude Code exports to every
subprocess (CLAUDE_CODE_SESSION_ID, CLAUDE_PID), so a speaking session is always
identified exactly, with no cooperation from the agent authoring the utterance.

Navigation targets are best-effort: a session running in its own console window
gets an HWND; sessions inside the desktop app or background jobs get none, and
the hub falls back to copying a `claude --resume <id>` command.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path

import platform_native

HUB_DIR = Path(__file__).resolve().parent / "hub"
EVENTS_DIR = HUB_DIR / "events"
HUB_LOCK = HUB_DIR / "hub.lock"
EVENT_TTL_S = 3600.0

# Processes that host a session in a window we can foreground.
_WINDOWED_HOSTS = {"windowsterminal.exe": "wt", "cmd.exe": "console",
                   "conhost.exe": "console", "powershell.exe": "console",
                   "pwsh.exe": "console", "openconsole.exe": "wt"}


# ----------------------------- spool ---------------------------------------

def write_event(state: str, tag: str, preset: str = "neutral",
                text_head: str = "", nav: dict | None = None,
                identity: dict | None = None) -> Path:
    """Append one event to the spool. Returns the file written."""
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    evt = {"ts": time.time(), "pid": os.getpid(), "state": state, "tag": tag,
           "preset": preset, "text_head": text_head[:80], "nav": nav}
    evt.update(identity or capture_identity())
    path = EVENTS_DIR / f"{evt['ts']:017.3f}-{evt['pid']}-{state}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(evt), encoding="utf-8")
    tmp.replace(path)  # rename is atomic: the hub never sees half-written JSON
    return path


def read_events(consume: bool = True) -> list[dict]:
    """Return spooled events oldest-first, deleting them unless consume=False.
    Corrupt or stale files are dropped silently (spool hygiene)."""
    if not EVENTS_DIR.exists():
        return []
    out: list[dict] = []
    now = time.time()
    for f in sorted(EVENTS_DIR.glob("*.json")):
        try:
            evt = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            f.unlink(missing_ok=True)
            continue
        if now - float(evt.get("ts", 0)) > EVENT_TTL_S:
            f.unlink(missing_ok=True)
            continue
        out.append(evt)
        if consume:
            f.unlink(missing_ok=True)
    return out


# ----------------------------- identity ------------------------------------

def capture_identity() -> dict:
    """Session identity from the env Claude Code exports to subprocesses."""
    claude_pid = os.environ.get("CLAUDE_PID", "")
    return {
        "session_id": os.environ.get("CLAUDE_CODE_SESSION_ID", ""),
        "claude_pid": int(claude_pid) if claude_pid.isdigit() else 0,
        "cwd": os.environ.get("CLAUDE_PROJECT_DIR", "") or os.getcwd(),
    }


# ----------------------------- window lookup -------------------------------

def _main_hwnd_for_pid(pid: int) -> int:
    """First visible top-level window owned by pid, or 0."""
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        owner = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            if user32.GetWindowTextLengthW(hwnd) > 0:  # skip helper windows
                found.append(int(ctypes.cast(hwnd, ctypes.c_void_p).value or 0))
                return False
        return True

    user32.EnumWindows(cb, None)
    return found[0] if found else 0


def _console_title() -> str:
    """Console title, or "" where the platform has no equivalent (macOS).

    This is only a fallback for session identity; CLAUDE_CODE_SESSION_ID is the
    primary source, so an empty title costs nothing on platforms without one.
    """
    return platform_native.console_title()


def capture_nav() -> dict | None:
    """Return {hwnd, window_kind, tab_title} if this process's session lives in
    a window we can foreground, else None (desktop app / background job)."""
    try:
        import psutil
        for p in psutil.Process().parents():
            try:
                kind = _WINDOWED_HOSTS.get(p.name().lower())
            except psutil.Error:
                continue
            if not kind:
                continue
            hwnd = _main_hwnd_for_pid(p.pid)
            if hwnd:
                return {"hwnd": hwnd, "window_kind": kind,
                        "tab_title": _console_title()}
    except Exception:
        pass
    return None

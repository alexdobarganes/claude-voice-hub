"""Spool carrying what Alex said *into* the sessions.

`hub_events.py` is the outbound half: say.py writes, the hub reads. This is the
inbound half, and it is deliberately its own module with its own directory
rather than a `state` on the existing events. The two flow in opposite
directions between different processes, and merging them would mean every
reader filtering out messages meant for someone else.

Same shape as the event spool, for the same reasons: one JSON file per message,
written to a temp name and renamed so a reader never sees half a message,
oldest-first by filename, deleted on read.

Addressing is by session id, with a broadcast fallback. A message whose target
is unknown goes to `*` and is delivered to whoever asks first: that is the right
default because the common case is Alex answering the session that just spoke to
him, and the wrong default (dropping it) loses what he said.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

INBOX_DIR = Path(__file__).resolve().parent / "hub" / "inbox"
BROADCAST = "*"
MESSAGE_TTL_S = 300.0   # what Alex said two minutes ago is no longer an answer


_BROADCAST_FILE = "all"   # BROADCAST is "*", which no Windows path may contain


def _safe(target: str) -> str:
    """Sanitized session id, or BROADCAST. Never trust an id into a path."""
    if not target or target == BROADCAST:
        return BROADCAST
    return "".join(c for c in target if c.isalnum() or c in "-_") or BROADCAST


def _path_token(target: str) -> str:
    """The same value, spelled so it can appear in a filename."""
    safe = _safe(target)
    return _BROADCAST_FILE if safe == BROADCAST else safe


def put(text: str, target: str = BROADCAST, source: str = "assistant",
        meta: dict | None = None) -> Path:
    """Deliver `text` to `target` (a session id, or BROADCAST)."""
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    msg = {"ts": time.time(), "text": text, "target": _safe(target),
           "source": source, "pid": os.getpid()}
    msg.update(meta or {})
    path = INBOX_DIR / f"{msg['ts']:017.3f}-{_path_token(target)}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(msg), encoding="utf-8")
    tmp.replace(path)
    return path


def take(session_id: str = "", accept_broadcast: bool = True) -> dict | None:
    """Claim the oldest message for this session, or None.

    Claiming is the unlink: whichever process removes the file owns the
    message. Two sessions polling at once cannot both act on the same answer.
    """
    if not INBOX_DIR.exists():
        return None
    want = _safe(session_id) if session_id else None
    now = time.time()
    for f in sorted(INBOX_DIR.glob("*.json")):
        try:
            msg = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            f.unlink(missing_ok=True)
            continue
        if now - float(msg.get("ts", 0)) > MESSAGE_TTL_S:
            f.unlink(missing_ok=True)
            continue
        target = msg.get("target", BROADCAST)
        if target == want or (accept_broadcast and target == BROADCAST):
            try:
                f.unlink()
            except OSError:
                continue  # someone else claimed it first; keep looking
            return msg
    return None


def peek() -> list[dict]:
    """Undelivered messages, oldest first. For the hub and for debugging."""
    if not INBOX_DIR.exists():
        return []
    out: list[dict] = []
    for f in sorted(INBOX_DIR.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


# --------------------------- open questions ---------------------------------
# A session waiting on an answer publishes here. The assistant reads it to know
# that speech right now is probably a reply, and to whom. This is what lets the
# wake word be required only for *unprompted* speech: when a session has just
# asked you something, answering it directly is how conversation works, and
# demanding a name first would make the channel feel like a kiosk.

ASKING_DIR = Path(__file__).resolve().parent / "hub" / "asking"
ASK_TTL_S = 300.0


def open_question(session_id: str, tag: str, text: str) -> Path:
    """Record that this session is waiting for a spoken answer."""
    ASKING_DIR.mkdir(parents=True, exist_ok=True)
    path = ASKING_DIR / f"{_path_token(session_id)}-{os.getpid()}.json"
    path.write_text(json.dumps({"ts": time.time(), "session_id": session_id,
                                "tag": tag, "text": text, "pid": os.getpid()}),
                    encoding="utf-8")
    return path


def close_question(session_id: str) -> None:
    (ASKING_DIR / f"{_path_token(session_id)}-{os.getpid()}.json").unlink(
        missing_ok=True)


def open_questions() -> list[dict]:
    """Sessions currently waiting on an answer, most recently asked first."""
    if not ASKING_DIR.exists():
        return []
    out: list[dict] = []
    now = time.time()
    for f in ASKING_DIR.glob("*.json"):
        try:
            q = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            f.unlink(missing_ok=True)
            continue
        if now - float(q.get("ts", 0)) > ASK_TTL_S:
            f.unlink(missing_ok=True)
            continue
        out.append(q)
    return sorted(out, key=lambda q: q.get("ts", 0), reverse=True)


def clear() -> int:
    """Drop everything undelivered. Returns how many were dropped."""
    if not INBOX_DIR.exists():
        return 0
    n = 0
    for f in INBOX_DIR.glob("*.json"):
        f.unlink(missing_ok=True)
        n += 1
    return n

"""Tests for the hub event spool + session identity."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hub_events


def test_write_then_read_consumes(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_events, "EVENTS_DIR", tmp_path)
    hub_events.write_event("queued", "Billing api, retry", text_head="hola")
    hub_events.write_event("speaking", "Billing api, retry")
    evts = hub_events.read_events()
    assert [e["state"] for e in evts] == ["queued", "speaking"]
    assert evts[0]["tag"] == "Billing api, retry"
    assert hub_events.read_events() == []  # consumed


def test_read_keeps_when_not_consuming(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_events, "EVENTS_DIR", tmp_path)
    hub_events.write_event("done", "X")
    assert len(hub_events.read_events(consume=False)) == 1
    assert len(hub_events.read_events(consume=False)) == 1


def test_corrupt_event_is_deleted_and_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_events, "EVENTS_DIR", tmp_path)
    bad = tmp_path / "0000000001.000-1-queued.json"
    bad.write_text("{not json", encoding="utf-8")
    assert hub_events.read_events() == []
    assert not bad.exists()


def test_stale_event_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_events, "EVENTS_DIR", tmp_path)
    p = hub_events.write_event("done", "old")
    evt = json.loads(p.read_text(encoding="utf-8"))
    evt["ts"] = time.time() - hub_events.EVENT_TTL_S - 10
    p.write_text(json.dumps(evt), encoding="utf-8")
    assert hub_events.read_events() == []


def test_text_head_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_events, "EVENTS_DIR", tmp_path)
    hub_events.write_event("speaking", "T", text_head="x" * 500)
    assert len(hub_events.read_events()[0]["text_head"]) == 80


def test_identity_from_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "abc-123")
    monkeypatch.setenv("CLAUDE_PID", "4242")
    ident = hub_events.capture_identity()
    assert ident["session_id"] == "abc-123"
    assert ident["claude_pid"] == 4242


def test_identity_tolerates_missing_env(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CLAUDE_PID", raising=False)
    ident = hub_events.capture_identity()
    assert ident["session_id"] == ""
    assert ident["claude_pid"] == 0


def test_event_carries_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_events, "EVENTS_DIR", tmp_path)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "sess-9")
    hub_events.write_event("speaking", "Tag")
    assert hub_events.read_events()[0]["session_id"] == "sess-9"


def test_capture_nav_none_when_no_windowed_host(monkeypatch):
    class FakeProc:
        def name(self):
            return "svchost.exe"

    monkeypatch.setattr(hub_events, "_main_hwnd_for_pid", lambda pid: 999)
    import psutil
    monkeypatch.setattr(psutil.Process, "parents", lambda self: [FakeProc()])
    assert hub_events.capture_nav() is None

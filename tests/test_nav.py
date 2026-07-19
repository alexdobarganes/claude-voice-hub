"""Tests for adaptive navigation and its primitives (no focus stealing)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agents_registry
import nav


def test_build_unicode_inputs_pairs_down_up():
    inputs = nav._build_unicode_inputs("hi")
    assert len(inputs) == 4  # 2 chars, down + up each
    assert inputs[0].ki.dwFlags == nav.KEYEVENTF_UNICODE
    assert inputs[1].ki.dwFlags == nav.KEYEVENTF_UNICODE | nav.KEYEVENTF_KEYUP
    assert inputs[0].ki.wScan == ord("h")


def test_build_unicode_inputs_handles_accents():
    assert nav._build_unicode_inputs("ñ")[0].ki.wScan == ord("ñ")


def test_clipboard_roundtrip():
    assert nav.set_clipboard("voice hub ñ test") is True


def test_go_to_session_prefers_window(monkeypatch):
    calls = []
    monkeypatch.setattr(nav, "_force_foreground", lambda h: calls.append(h))
    monkeypatch.setattr(nav, "_select_wt_tab", lambda h, t: False)
    status = nav.go_to_session({"nav": {"hwnd": 777, "window_kind": "console"},
                                "session_id": "s1"})
    assert calls == [777]
    assert status == "ventana enfocada"


def test_go_to_session_selects_wt_tab(monkeypatch):
    monkeypatch.setattr(nav, "_force_foreground", lambda h: None)
    monkeypatch.setattr(nav, "_select_wt_tab", lambda h, t: True)
    status = nav.go_to_session({"nav": {"hwnd": 1, "window_kind": "wt",
                                        "tab_title": "example-app"}})
    assert status == "ventana + tab"


def test_go_to_session_falls_back_to_clipboard(monkeypatch):
    copied = []
    monkeypatch.setattr(nav, "set_clipboard", lambda t: copied.append(t) or True)
    monkeypatch.setattr(nav, "find_claude_app_hwnd", lambda: 0)
    status = nav.go_to_session({"nav": None, "session_id": "abc-123"})
    assert copied == ["claude --resume abc-123"]
    assert "resume copiado" in status


def test_go_to_session_without_identity_does_nothing():
    assert nav.go_to_session({"nav": None, "session_id": ""}) == ""


def test_wt_tab_select_skips_empty_title():
    assert nav._select_wt_tab(123, "") is False


def test_describe_uses_registry(monkeypatch):
    monkeypatch.setattr(agents_registry, "lookup",
                        lambda sid: {"name": "voice hub", "cwd": "D:\\proj\\example-app"})
    assert agents_registry.describe("x") == ("voice hub", "example-app")


def test_describe_falls_back_to_cwd(monkeypatch):
    monkeypatch.setattr(agents_registry, "lookup", lambda sid: None)
    assert agents_registry.describe("x", "D:\\proj\\other") == ("", "other")


def test_registry_survives_broken_cli(monkeypatch):
    monkeypatch.setattr(agents_registry, "_query",
                        lambda: (_ for _ in ()).throw(OSError("claude missing")))
    agents_registry._cache_ts = 0.0
    assert agents_registry.refresh(force=True) == agents_registry._cache

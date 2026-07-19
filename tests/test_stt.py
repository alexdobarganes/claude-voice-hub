"""Tests for transcription plumbing and dictation delivery (no mic, no network)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import nav
import stt


@pytest.fixture()
def wav(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF0000WAVE")
    return p


def test_transcribe_uses_scribe_response(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request", lambda p: {"text": " hola mundo "})
    assert stt.transcribe(wav) == "hola mundo"


def test_transcribe_falls_back_to_whisper(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request",
                        lambda p: (_ for _ in ()).throw(OSError("net down")))
    monkeypatch.setattr(stt, "_whisper_available", lambda: True)
    monkeypatch.setattr(stt, "_whisper_transcribe", lambda p: "desde whisper")
    assert stt.transcribe(wav) == "desde whisper"


def test_transcribe_raises_when_all_backends_fail(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request",
                        lambda p: (_ for _ in ()).throw(OSError("net down")))
    monkeypatch.setattr(stt, "_whisper_available", lambda: False)
    with pytest.raises(stt.SttError):
        stt.transcribe(wav)


def test_empty_scribe_text_falls_through(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request", lambda p: {"text": "   "})
    monkeypatch.setattr(stt, "_whisper_available", lambda: False)
    with pytest.raises(stt.SttError):
        stt.transcribe(wav)


def test_stop_without_start_raises():
    with pytest.raises(stt.SttError):
        stt.Recorder().stop()


def test_delivery_types_into_console_window(monkeypatch):
    typed, focused = [], []
    monkeypatch.setattr(nav, "_force_foreground", lambda h: focused.append(h))
    monkeypatch.setattr(nav, "type_text", lambda t, press_enter=True: typed.append(t))
    monkeypatch.setattr(nav.time, "sleep", lambda s: None)
    status = nav.deliver_text({"nav": {"hwnd": 55, "window_kind": "console"}}, "hola")
    assert typed == ["hola"] and focused == [55]
    assert status == "enviado a la ventana"


def test_delivery_never_types_into_desktop_app(monkeypatch):
    """The core safety property: with no window of its own, text is copied,
    never injected, since it could land in the wrong session."""
    typed, copied = [], []
    monkeypatch.setattr(nav, "type_text",
                        lambda t, press_enter=True: typed.append(t))
    monkeypatch.setattr(nav, "set_clipboard", lambda t: copied.append(t) or True)
    monkeypatch.setattr(nav, "find_claude_app_hwnd", lambda: 0)
    status = nav.deliver_text({"nav": None, "session_id": "s1"}, "hola")
    assert typed == []
    assert copied == ["hola"]
    assert "copiado" in status


def test_delivery_reports_clipboard_failure(monkeypatch):
    monkeypatch.setattr(nav, "set_clipboard", lambda t: False)
    assert nav.deliver_text({"nav": None}, "hola") == "no se pudo copiar"

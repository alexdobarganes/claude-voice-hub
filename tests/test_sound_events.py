"""Scribe answers non-speech with confident bracketed captions. Delivering one
as dictation is worse than failing, so they must be rejected."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt


@pytest.fixture()
def wav(tmp_path):
    p = tmp_path / "x.wav"
    p.write_bytes(b"RIFF0000WAVE")
    return p


@pytest.mark.parametrize("caption", ["[outro jingle]", "[keyboard clacking]",
                                     "[BLANK_AUDIO]", " [music] "])
def test_pure_sound_event_is_rejected(monkeypatch, wav, caption):
    monkeypatch.setattr(stt, "_scribe_request", lambda p: {"text": caption})
    with pytest.raises(stt.SttError) as exc:
        stt.transcribe(wav)
    assert "habla" in str(exc.value)


def test_speech_with_a_sound_event_keeps_the_speech(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request",
                        lambda p: {"text": "[clears throat] mergea el PR"})
    assert stt.transcribe(wav) == "mergea el PR"


def test_plain_speech_untouched(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request", lambda p: {"text": "hola mundo"})
    assert stt.transcribe(wav) == "hola mundo"


def test_scribe_failure_still_falls_back_to_whisper(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request",
                        lambda p: (_ for _ in ()).throw(OSError("net down")))
    monkeypatch.setattr(stt, "_whisper_available", lambda: True)
    monkeypatch.setattr(stt, "_whisper_transcribe", lambda p: "desde whisper")
    assert stt.transcribe(wav) == "desde whisper"


def test_whisper_sound_event_also_rejected(monkeypatch, wav):
    monkeypatch.setattr(stt, "_scribe_request",
                        lambda p: (_ for _ in ()).throw(OSError("net down")))
    monkeypatch.setattr(stt, "_whisper_available", lambda: True)
    monkeypatch.setattr(stt, "_whisper_transcribe", lambda p: "[silence]")
    with pytest.raises(stt.SttError):
        stt.transcribe(wav)

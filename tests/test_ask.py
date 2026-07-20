"""The --ask contract: stdout carries the answer, exit code carries the truth.

The agent that ran say.py through Bash acts on these two things alone, so the
one outcome that must be impossible is exit 0 with an answer nobody gave.
"""
from __future__ import annotations

import sys

import pytest

import say
import stt


class _Hub:
    def __init__(self):
        self.states = []

    def emit(self, state):
        self.states.append(state)


class _Overlay:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class _Args:
    ask_timeout = 5.0
    ask_tail = 1.0


@pytest.fixture(autouse=True)
def no_assistant(monkeypatch):
    """These cases are about the --ask contract, not about who owns the mic.

    Without this they pass or fail depending on whether a real assistant
    happens to be running on the developer's machine, which is not a property
    a test should have.
    """
    import assistant
    monkeypatch.setattr(assistant, "is_running", lambda: False)


def test_a_captured_answer_goes_to_stdout_with_exit_zero(monkeypatch, capsys):
    monkeypatch.setattr(stt, "listen_once", lambda **kw: "abre PR")
    assert say.listen_for_answer(_Hub(), _Overlay(), _Args()) == 0
    assert capsys.readouterr().out.strip() == "abre PR"


def test_accents_survive_the_trip(monkeypatch, capsys):
    """Windows hands stdout a cp1252 codec by default, which turned a real
    answer into 'mant?n ... olv?date'."""
    monkeypatch.setattr(stt, "listen_once", lambda **kw: "sí, mantén ñ y é")
    assert say.listen_for_answer(_Hub(), _Overlay(), _Args()) == 0
    assert "sí, mantén ñ y é" in capsys.readouterr().out


def test_silence_is_reported_as_unanswered_not_as_an_answer(monkeypatch, capsys):
    def refuse(**kw):
        raise stt.SttError("nadie contesto")
    monkeypatch.setattr(stt, "listen_once", refuse)
    assert say.listen_for_answer(_Hub(), _Overlay(), _Args()) == say.ASK_NO_ANSWER
    assert capsys.readouterr().out.strip() == ""


def test_an_empty_transcript_is_not_an_answer(monkeypatch, capsys):
    monkeypatch.setattr(stt, "listen_once", lambda **kw: "   ")
    assert say.listen_for_answer(_Hub(), _Overlay(), _Args()) == say.ASK_NO_ANSWER
    assert capsys.readouterr().out.strip() == ""


def test_a_transcription_crash_is_still_just_unanswered(monkeypatch):
    """Anything unexpected must degrade to 'nobody answered' rather than kill
    the session that asked."""
    def boom(**kw):
        raise RuntimeError("network died mid-request")
    monkeypatch.setattr(stt, "listen_once", boom)
    assert say.listen_for_answer(_Hub(), _Overlay(), _Args()) == say.ASK_NO_ANSWER


def test_the_hub_is_told_we_are_listening(monkeypatch):
    """The hub row is the only place you can see WHICH session is waiting."""
    monkeypatch.setattr(stt, "listen_once", lambda **kw: "sí")
    hub = _Hub()
    say.listen_for_answer(hub, _Overlay(), _Args())
    assert "listening" in hub.states


def test_the_speaking_visual_is_dismissed_before_the_mic_opens(monkeypatch):
    monkeypatch.setattr(stt, "listen_once", lambda **kw: "sí")
    overlay = _Overlay()
    say.listen_for_answer(_Hub(), overlay, _Args())
    assert overlay.stopped


def test_a_reply_in_another_language_is_not_a_reply():
    """Measured twice with a live mic: a question asked in Spanish came back
    with fluent, high-confidence English from something else playing in the
    room. Real audio, real transcription, wrong speaker."""
    with pytest.raises(stt.SttError):
        stt.check_answer({"language_code": "eng", "language_probability": 0.97},
                         expect_lang="es")


def test_the_expected_language_passes():
    stt.check_answer({"language_code": "spa", "language_probability": 0.87},
                     expect_lang="es")


def test_an_unsure_language_call_is_not_used_to_reject():
    """Short answers ('sí', 'dale') give weak language detection. Rejecting on
    a coin flip would throw away real answers to save nothing."""
    stt.check_answer({"language_code": "eng", "language_probability": 0.31},
                     expect_lang="es")


def test_no_expectation_means_no_check():
    stt.check_answer({"language_code": "eng", "language_probability": 0.99},
                     expect_lang=None)


def test_say_passes_the_question_language_through(monkeypatch, capsys):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return "dale"
    monkeypatch.setattr(stt, "listen_once", fake)
    say.listen_for_answer(_Hub(), _Overlay(), _Args(), "es")
    assert seen["expect_lang"] == "es"


# --------------------------- who owns the mic -------------------------------

def test_without_the_assistant_we_open_the_mic_ourselves(monkeypatch):
    """Listening degrades the way speech already does when the hub is missing:
    worse, but never absent."""
    import assistant
    monkeypatch.setattr(assistant, "is_running", lambda: False)
    monkeypatch.setattr(stt, "listen_once", lambda **kw: "por el micro")
    assert say._await_answer(_Args(), "es") == "por el micro"


def test_with_the_assistant_up_we_wait_on_the_inbox(monkeypatch):
    """Two processes cannot record from one device sanely."""
    import assistant, inbox
    monkeypatch.setattr(assistant, "is_running", lambda: True)
    monkeypatch.setattr(stt, "listen_once",
                        lambda **kw: pytest.fail("must not open the mic"))
    monkeypatch.setattr(inbox, "take", lambda sid, **kw: {"text": "por el inbox"})
    monkeypatch.setattr(inbox, "open_question", lambda *a, **k: None)
    monkeypatch.setattr(inbox, "close_question", lambda *a, **k: None)
    assert say._await_answer(_Args(), "es") == "por el inbox"


def test_the_question_is_closed_even_when_nobody_answers(monkeypatch):
    """A session that gave up must stop claiming answers meant for others."""
    import assistant, inbox
    closed = []
    monkeypatch.setattr(assistant, "is_running", lambda: True)
    monkeypatch.setattr(inbox, "take", lambda sid, **kw: None)
    monkeypatch.setattr(inbox, "open_question", lambda *a, **k: None)
    monkeypatch.setattr(inbox, "close_question", lambda sid: closed.append(sid))
    monkeypatch.setattr(say, "_INBOX_POLL_S", 0.01)

    class Quick(_Args):
        ask_timeout = 0.05
    with pytest.raises(stt.SttError):
        say._await_answer(Quick(), "es")
    assert closed

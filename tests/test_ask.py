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

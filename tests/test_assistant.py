"""The policy governing an always-open microphone.

handle_utterance is pure so this can be tested without a sound card, which
matters more here than usual: the failure being guarded against is the assistant
acting on something nobody said to it.
"""
from __future__ import annotations

import pytest

import assistant
import inbox

SESSIONS = [
    {"session_id": "s1", "name": "billing api"},
    {"session_id": "s2", "name": "voice hub"},
]
ASKED = [{"session_id": "s2", "tag": "voice hub", "ts": 100.0}]


# --------------------------- unprompted speech ------------------------------

def test_the_room_is_ignored_when_nobody_asked_anything():
    """Correctly transcribed speech that was simply not for us. This is the
    case a live mic produced, not a hypothetical."""
    assert assistant.handle_utterance(
        "Mira lo que me hice de eso de la palomita", SESSIONS, []) is None


def test_being_named_is_what_makes_it_ours():
    d = assistant.handle_utterance("Claude, abre el PR", SESSIONS, [])
    assert d is not None and d["addressed"]
    assert d["text"] == "abre el pr"


def test_naming_a_session_after_the_wake_word_routes_past_the_assistant():
    """'Claude, billing api, abre el PR' belongs to billing api, not to us."""
    d = assistant.handle_utterance(
        "Claude, billing api, abre el PR", SESSIONS, [])
    assert d["target"] == "s1"


def test_being_named_with_no_session_goes_to_broadcast():
    d = assistant.handle_utterance("Claude, abre el PR", SESSIONS, [])
    assert d["target"] == inbox.BROADCAST


def test_silence_is_never_a_message():
    assert assistant.handle_utterance("", SESSIONS, []) is None
    assert assistant.handle_utterance("   ", SESSIONS, ASKED) is None


# --------------------------- answering --------------------------------------

def test_an_open_question_lets_you_answer_without_the_wake_word():
    """Answering a question you were just asked is how conversation works;
    demanding a name first would make this feel like a kiosk."""
    d = assistant.handle_utterance("dale, abre PR", SESSIONS, ASKED)
    assert d is not None and not d["addressed"]
    assert d["target"] == "s2"


def test_an_answer_goes_to_the_session_that_asked_most_recently():
    asked = [{"session_id": "s2", "ts": 200.0}, {"session_id": "s1", "ts": 100.0}]
    assert assistant.handle_utterance("sí", SESSIONS, asked)["target"] == "s2"


def test_naming_a_session_beats_who_asked_last():
    """If Alex names one, he means that one, whoever happens to be waiting."""
    d = assistant.handle_utterance("billing api, para eso", SESSIONS, ASKED)
    assert d["target"] == "s1"


def test_the_room_still_reaches_a_waiting_session():
    """The deliberate cost of skipping the wake word while a question is open:
    whatever is said in the room during that window is treated as the answer.
    Pinned so the trade is visible rather than discovered later."""
    d = assistant.handle_utterance("y luego fuimos al cine", SESSIONS, ASKED)
    assert d is not None and d["target"] == "s2"


# --------------------------- the lock ---------------------------------------

def test_a_fresh_lock_from_another_process_means_running(tmp_path, monkeypatch):
    lock = tmp_path / "assistant.lock"
    lock.write_text("999999")
    monkeypatch.setattr(assistant, "ASSISTANT_LOCK", lock)
    monkeypatch.setitem(__import__("sys").modules, "psutil",
                        type("m", (), {"pid_exists": staticmethod(lambda p: True)}))
    assert assistant.is_running()


def test_a_stale_lock_does_not_count_as_running(tmp_path, monkeypatch):
    """The heartbeat stopped: that is a crash, and the mic must be reclaimable
    or listening stays broken until someone notices and deletes a file."""
    import os
    lock = tmp_path / "assistant.lock"
    lock.write_text("999999")
    old = 0
    os.utime(lock, (old, old))
    monkeypatch.setattr(assistant, "ASSISTANT_LOCK", lock)
    assert not assistant.is_running()


def test_no_lock_means_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(assistant, "ASSISTANT_LOCK", tmp_path / "nope.lock")
    assert not assistant.is_running()


def test_our_own_lock_does_not_block_us(tmp_path, monkeypatch):
    import os
    lock = tmp_path / "assistant.lock"
    lock.write_text(str(os.getpid()))
    monkeypatch.setattr(assistant, "ASSISTANT_LOCK", lock)
    assert not assistant.is_running()


def test_a_hallucinated_transcript_is_never_a_message():
    """The local model answers noise with letters from an alphabet nobody in
    the room was speaking. Even with a question open, that is not an answer."""
    junk = "ू ॉ, ॑ ॗ ख़ OH ६astics © başall comprehensive"
    assert assistant.handle_utterance(junk, SESSIONS, []) is None
    assert assistant.handle_utterance(junk, SESSIONS, ASKED) is None

"""The inbound spool: what Alex said, on its way into a session."""
from __future__ import annotations

import time

import pytest

import inbox


@pytest.fixture(autouse=True)
def spool(tmp_path, monkeypatch):
    monkeypatch.setattr(inbox, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(inbox, "ASKING_DIR", tmp_path / "asking")


def test_a_targeted_message_reaches_its_session():
    inbox.put("abre PR", target="s1")
    assert inbox.take("s1")["text"] == "abre PR"


def test_a_message_is_claimed_once_and_only_once():
    """Claiming is the unlink, so two sessions polling together cannot both act
    on the same answer."""
    inbox.put("abre PR", target="s1")
    assert inbox.take("s1") is not None
    assert inbox.take("s1") is None


def test_a_session_does_not_receive_another_sessions_message():
    inbox.put("abre PR", target="s1")
    assert inbox.take("s2", accept_broadcast=False) is None
    assert inbox.take("s1") is not None


def test_broadcast_reaches_whoever_asks_first():
    inbox.put("sí", target=inbox.BROADCAST)
    assert inbox.take("s9")["text"] == "sí"


def test_broadcast_can_be_refused():
    inbox.put("sí", target=inbox.BROADCAST)
    assert inbox.take("s1", accept_broadcast=False) is None


def test_messages_arrive_oldest_first():
    inbox.put("primero", target="s1")
    time.sleep(0.01)
    inbox.put("segundo", target="s1")
    assert inbox.take("s1")["text"] == "primero"
    assert inbox.take("s1")["text"] == "segundo"


def test_a_stale_message_is_dropped_rather_than_delivered():
    """What Alex said five minutes ago is not an answer to what you just
    asked."""
    p = inbox.put("hace mucho", target="s1")
    import json
    msg = json.loads(p.read_text(encoding="utf-8"))
    msg["ts"] = time.time() - inbox.MESSAGE_TTL_S - 10
    p.write_text(json.dumps(msg), encoding="utf-8")
    assert inbox.take("s1") is None


def test_a_corrupt_message_is_dropped_not_raised():
    inbox.INBOX_DIR.mkdir(parents=True, exist_ok=True)
    (inbox.INBOX_DIR / "00000000000000.000-x.json").write_text("{not json")
    assert inbox.take("s1") is None


def test_an_empty_inbox_is_just_empty():
    assert inbox.take("s1") is None
    assert inbox.peek() == []


def test_a_session_id_cannot_escape_the_spool_directory():
    inbox.put("x", target="../../etc/passwd")
    assert all(".." not in p.name for p in inbox.INBOX_DIR.glob("*"))


# --------------------------- open questions ---------------------------------

def test_an_open_question_is_visible_then_closed():
    inbox.open_question("s1", "billing api", "¿mergeo?")
    assert [q["session_id"] for q in inbox.open_questions()] == ["s1"]
    inbox.close_question("s1")
    assert inbox.open_questions() == []


def test_open_questions_come_back_most_recent_first():
    inbox.open_question("s1", "a", "?")
    time.sleep(0.01)
    inbox.open_question("s2", "b", "?")
    assert [q["session_id"] for q in inbox.open_questions()] == ["s2", "s1"]


def test_an_abandoned_question_expires():
    """A session that died mid-question must not keep claiming answers."""
    import json
    inbox.open_question("s1", "a", "?")
    f = next(inbox.ASKING_DIR.glob("*.json"))
    q = json.loads(f.read_text(encoding="utf-8"))
    q["ts"] = time.time() - inbox.ASK_TTL_S - 10
    f.write_text(json.dumps(q), encoding="utf-8")
    assert inbox.open_questions() == []

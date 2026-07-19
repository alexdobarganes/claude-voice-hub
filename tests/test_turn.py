"""Turn ordering: what the old lock could not do, and the bug it had."""
from __future__ import annotations

import json
import os
import time

import pytest

import turn


@pytest.fixture(autouse=True)
def queue_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(turn, "QUEUE_DIR", tmp_path / "queue")
    return turn.QUEUE_DIR


def _ticket(priority: str, ts: float, pid: int):
    """Write a ticket as if another process had queued at `ts`."""
    turn.QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    p = turn.QUEUE_DIR / turn._ticket_name(priority, ts, pid)
    p.write_text(json.dumps({"pid": pid, "ts": ts, "priority": priority}),
                 encoding="utf-8")
    return p


def _holder(pid: int, age: float = 0.0):
    """Plant a holder file as if `pid` were speaking right now."""
    turn.QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    p = turn._holder_path()
    p.write_text(json.dumps({"pid": pid, "ts": time.time()}), encoding="utf-8")
    if age:
        old = time.time() - age
        os.utime(p, (old, old))
    return p


def test_acquires_immediately_when_nothing_is_queued():
    t = turn.acquire("outcome", timeout=1.0)
    assert t.granted
    assert turn._holder_path().exists()
    t.release()
    assert not turn._holder_path().exists()


def test_the_ticket_is_dropped_once_the_turn_is_held():
    """Otherwise a speaker shows up in its own queue, behind itself."""
    t = turn.acquire("outcome", timeout=1.0)
    assert turn.pending() == []
    t.release()


def test_release_is_idempotent():
    t = turn.acquire(timeout=1.0)
    t.release()
    t.release()  # callers put this in a finally


def test_a_speaker_in_progress_is_never_preempted():
    """The bug the first implementation had: ranking waiters against the
    current holder let an outcome cut into a ping that was already playing.
    You cannot un-play audio that is already coming out of the speaker."""
    _holder(pid=4242)
    t = turn.acquire("blocker", timeout=0.3)
    assert not t.granted        # waited, then spoke anyway rather than cut in
    t.release()
    assert turn._holder_path().exists()   # and did not steal the holder


def test_blocker_overtakes_an_outcome_that_queued_earlier():
    """The whole reason for replacing the lock: rank beats arrival among waiters."""
    _ticket("outcome", time.time() - 5, 999)
    blocker = _ticket("blocker", time.time(), 1000)
    assert turn._live_tickets(turn.STALE_S)[0] == blocker


def test_same_band_is_strict_arrival_order():
    """Within a band it must be FIFO. The old spin loop was arbitrary here."""
    first = _ticket("outcome", 1000.0, 1)
    second = _ticket("outcome", 1001.0, 2)
    third = _ticket("outcome", 1002.0, 3)
    assert turn._live_tickets(1e9) == [first, second, third]


def test_waits_while_a_higher_rank_is_queued_ahead():
    _ticket("blocker", time.time(), 4242)
    t = turn.acquire("ping", timeout=0.3)
    assert not t.granted        # never silent: speaks anyway, but knows it
    t.release()


def test_a_crashed_speaker_is_reclaimed():
    _holder(pid=4242, age=turn.STALE_S + 10)
    t = turn.acquire("outcome", timeout=3.0)
    assert t.granted
    assert json.loads(turn._holder_path().read_text())["pid"] == os.getpid()
    t.release()


def test_a_crashed_waiter_stops_blocking_the_queue():
    dead = _ticket("blocker", time.time(), 4242)
    os.utime(dead, (0, 0))      # last heartbeat long ago
    t = turn.acquire("outcome", timeout=2.0)
    assert t.granted
    assert not dead.exists()
    t.release()


def test_release_cannot_remove_another_processs_turn():
    """The old release() unlinked a fixed path, so an overrunning holder would
    delete a lock that by then belonged to someone else."""
    _holder(pid=4242)
    stolen = turn.Turn(turn._holder_path(), granted=True)
    stolen.release()
    assert turn._holder_path().exists()
    assert json.loads(turn._holder_path().read_text())["pid"] == 4242


def test_an_ungranted_turn_never_touches_the_holder():
    _holder(pid=4242)
    t = turn.acquire("blocker", timeout=0.2)
    assert not t.granted
    t.release()
    assert json.loads(turn._holder_path().read_text())["pid"] == 4242


def test_unknown_priority_falls_back_to_the_default_band():
    assert turn.priority_rank("nonsense") == turn.PRIORITIES[turn.DEFAULT_PRIORITY]


def test_pending_reports_waiters_in_service_order():
    _ticket("ping", 1000.0, 1)
    _ticket("blocker", 2000.0, 2)
    order = [p["priority"] for p in turn.pending(stale=1e9)]
    assert order == ["blocker", "ping"]


def test_pending_ignores_the_holder_file():
    _holder(pid=4242)
    assert turn.pending() == []


def test_pending_is_empty_when_nothing_has_queued():
    assert turn.pending() == []


def test_is_speaking_reports_a_live_holder():
    """The assistant keeps its hands off the mic while the machine talks."""
    _holder(pid=4242)
    assert turn.is_speaking()


def test_is_speaking_is_false_with_nobody_holding():
    assert not turn.is_speaking()


def test_a_crashed_holder_does_not_look_like_speech_forever():
    _holder(pid=4242, age=turn.STALE_S + 10)
    assert not turn.is_speaking()


def test_an_utterance_entirely_inside_a_capture_is_still_detected():
    """The failure this exists for: a 30 s capture contained a whole spoken
    line, so checking 'speaking now' at both ends saw silence both times, and
    the listener transcribed the machine's own voice."""
    capture_start = time.time()
    t = turn.acquire("outcome", timeout=1.0)
    t.release()                       # spoke and finished, all within the window
    assert turn.spoke_since(capture_start)


def test_speech_before_the_capture_does_not_taint_it():
    t = turn.acquire("outcome", timeout=1.0)
    t.release()
    time.sleep(0.01)
    assert not turn.spoke_since(time.time())


def test_spoke_since_is_false_when_nothing_ever_spoke():
    assert not turn.spoke_since(time.time() - 10)

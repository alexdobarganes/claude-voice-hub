"""Priority-ordered turn taking for concurrent say.py processes.

Replaces the single `say-voice.lock` file. That lock serialized playback, which
was the important part, but it could not rank utterances and it was not even
first-come-first-served: a spin loop over `O_EXCL` hands the file to whichever
waiter happens to call `os.open` first, so among several sessions an arbitrary
one won. The hub's visible queue is built from arrival order, so the UI could
disagree with what actually played.

Two separate jobs, deliberately kept apart:

- **Who may speak at all** is mutual exclusion, and it stays an atomic
  `O_EXCL` create on a single holder file. Anything softer (comparing sorted
  directory listings and then acting) has a window where two processes both
  decide they are next, and the cost of losing that race is two voices at once.
- **Who goes next** is ordering, and that is what the ticket files are for.
  Each waiter writes a ticket whose *filename* carries its rank and arrival
  time, and only the process whose ticket sorts first is allowed to try for the
  holder file.

Ranking applies among *waiters* only. A turn already in progress is never
preempted: you cannot un-play audio that is already coming out of the speaker,
so a blocker arriving mid-utterance goes to the front of the queue, not to the
front of the sound card.

Ownership is checked on release. The old `release()` unlinked a fixed path
unconditionally, so a holder that overran `stale` would delete a lock that by
then belonged to someone else. Here a holder file records its pid and is only
removed by the pid that wrote it.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

QUEUE_DIR = Path(__file__).resolve().parent / "hub" / "queue"
HOLDER_NAME = "holder.json"

# Lower sorts first. These names are the vocabulary the voice skill authors in,
# so they are what an agent picks when it decides how much an utterance matters.
PRIORITIES = {"blocker": 0, "question": 1, "outcome": 2, "ping": 3}
DEFAULT_PRIORITY = "outcome"

# How long an utterance is still worth saying after it was queued. None means
# forever: a blocker does not stop mattering because it waited, and a question
# still needs answering.
#
# Progress pings are different in kind. "Still refactoring" is a statement about
# *now*, and read out forty seconds late behind three other sessions it is worse
# than nothing -- it describes a state that has passed and pushes back everything
# behind it. So this is the one place where going silent is the correct
# behaviour rather than a failure.
SHELF_LIFE_S = {"ping": 40.0, "outcome": 180.0, "question": None, "blocker": None}

HEARTBEAT_S = 5.0
STALE_S = 30.0          # ~6 missed heartbeats: a real crash, not a slow machine
POLL_S = 0.15


def priority_rank(priority: str) -> int:
    """Rank for a priority name; unknown names fall back to the default band."""
    return PRIORITIES.get(priority, PRIORITIES[DEFAULT_PRIORITY])


def _holder_path() -> Path:
    return QUEUE_DIR / HOLDER_NAME


class Turn:
    """A held place in the speech queue.

    `release()` is idempotent so callers can put it in a `finally` without
    tracking whether it already ran.
    """

    def __init__(self, holder: Path, granted: bool, expired: bool = False) -> None:
        self.holder = holder
        self.granted = granted   # False when we gave up waiting and spoke anyway
        # True when the utterance sat in the queue past its shelf life. The
        # caller is expected to stay silent: see SHELF_LIFE_S.
        self.expired = expired
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    def _start_heartbeat(self) -> None:
        """Keep the holder file fresh while we speak.

        This is what lets `stale` be short. Without a heartbeat the timeout has
        to cover the longest legitimate utterance, which is why the old lock
        waited three minutes before reclaiming after a crash.
        """
        def beat() -> None:
            while not self._stop.wait(HEARTBEAT_S):
                try:
                    os.utime(self.holder, None)
                except OSError:
                    return  # reclaimed or released; nothing left to keep alive
        self._beat = threading.Thread(target=beat, daemon=True)
        self._beat.start()

    def release(self) -> None:
        self._stop.set()
        _mark_spoke()
        if not self.granted:
            return
        try:
            data = json.loads(self.holder.read_text(encoding="utf-8"))
        except Exception:
            return  # gone, or unreadable: not ours to remove
        if data.get("pid") == os.getpid():
            try:
                self.holder.unlink()
            except OSError:
                pass

    def __enter__(self) -> "Turn":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def _ticket_name(priority: str, ts: float, pid: int) -> str:
    # Fixed-width timestamp so a plain lexicographic sort is chronological,
    # matching how hub_events.py names its own files.
    return f"{priority_rank(priority)}-{ts:017.3f}-{pid}.json"


def _live_tickets(stale: float) -> list[Path]:
    """Waiting tickets in service order, dropping any whose process went away."""
    live: list[Path] = []
    now = time.time()
    for f in sorted(QUEUE_DIR.glob("*.json")):
        if f.name == HOLDER_NAME:
            continue
        try:
            if now - f.stat().st_mtime > stale:
                f.unlink(missing_ok=True)
                continue
        except OSError:
            continue  # vanished between glob and stat: simply not live
        live.append(f)
    return live


def _holder_is_stale(stale: float) -> bool:
    try:
        return time.time() - _holder_path().stat().st_mtime > stale
    except OSError:
        return False


def _try_take_holder() -> bool:
    """Atomically claim the right to speak. False if someone already has it."""
    try:
        fd = os.open(str(_holder_path()), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        json.dump({"pid": os.getpid(), "ts": time.time()}, fh)
    return True


def acquire(priority: str = DEFAULT_PRIORITY, timeout: float = 120.0,
            stale: float = STALE_S) -> Turn:
    """Wait for this process's turn to speak, and return it.

    Blocks until nobody is speaking and this ticket sorts first among the
    waiters. Never blocks forever and never goes silent: on `timeout` it returns
    an ungranted Turn and the caller speaks anyway, because a late voice is
    recoverable and a silently dropped one is not.
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.time()
    payload = json.dumps({"pid": os.getpid(), "ts": ts, "priority": priority})
    ticket = QUEUE_DIR / _ticket_name(priority, ts, os.getpid())
    ticket.write_text(payload, encoding="utf-8")

    shelf = SHELF_LIFE_S.get(priority, SHELF_LIFE_S[DEFAULT_PRIORITY])
    deadline = time.monotonic() + timeout
    try:
        while True:
            if shelf is not None and time.time() - ts > shelf:
                # Waited past the point where saying this still helps.
                return Turn(_holder_path(), granted=False, expired=True)
            waiting = _live_tickets(stale)
            if ticket not in waiting:
                # Our ticket was reclaimed as stale (a machine so loaded the
                # heartbeat could not run, or the clock moved). Rewrite it
                # rather than wait in a queue we are no longer part of.
                ticket.write_text(payload, encoding="utf-8")
                continue
            if waiting[0] == ticket:
                if _try_take_holder():
                    turn = Turn(_holder_path(), granted=True)
                    turn._start_heartbeat()
                    return turn
                if _holder_is_stale(stale):
                    # The speaker crashed mid-utterance. Reclaim and retry; the
                    # next loop re-checks ordering, so this does not jump anyone.
                    _holder_path().unlink(missing_ok=True)
                    continue
            if time.monotonic() >= deadline:
                print(f"[turn] timeout waiting for the queue "
                      f"({max(0, len(waiting) - 1)} ahead); speaking anyway",
                      file=sys.stderr)
                return Turn(_holder_path(), granted=False)
            time.sleep(POLL_S)
    finally:
        # Once we hold the turn (or gave up) the ticket has done its job: the
        # holder file is what marks us as speaking. Leaving it would make us
        # look like we were still queued behind ourselves.
        ticket.unlink(missing_ok=True)


LAST_SPEECH_NAME = "last_speech.json"


def _mark_spoke() -> None:
    """Record when audio last stopped going out. Best-effort, never raises."""
    try:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        (QUEUE_DIR / LAST_SPEECH_NAME).write_text(
            json.dumps({"end": time.time()}), encoding="utf-8")
    except Exception:
        pass


def last_speech_end() -> float:
    """When the machine last finished speaking, as an epoch time. 0 if never."""
    try:
        return float(json.loads(
            (QUEUE_DIR / LAST_SPEECH_NAME).read_text(encoding="utf-8"))["end"])
    except Exception:
        return 0.0


def spoke_since(t: float) -> bool:
    """True if any audio went out after `t`.

    Checking whether we are speaking *right now* is not enough for a listener
    that records in long windows: an utterance can start and finish entirely
    inside one capture, and both the before and after checks then see silence.
    Observed exactly that way -- a thirty-second capture came back as a verbatim
    transcript of the machine's own voice.
    """
    return last_speech_end() > t or is_speaking()


def is_speaking(stale: float = STALE_S) -> bool:
    """True while some process holds the turn, i.e. audio is going out now.

    The assistant uses this to keep its hands off the microphone while the
    machine is talking. Without it, one capture contained the tail of a spoken
    instruction followed by the reply to it, transcribed as a single sentence:
    the listener heard itself and could not tell its own voice from Alex's.
    """
    holder = _holder_path()
    try:
        return time.time() - holder.stat().st_mtime <= stale
    except OSError:
        return False


def pending(stale: float = STALE_S) -> list[dict]:
    """Waiting tickets in service order, as dicts. For the hub and debugging."""
    if not QUEUE_DIR.exists():
        return []
    out: list[dict] = []
    for f in _live_tickets(stale):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def clear_legacy_lock() -> None:
    """Remove the pre-turn.py lockfile if one is lying around.

    A say.py from before this change may have left one; nothing reads it now, so
    it would sit in the temp dir forever looking like a bug.
    """
    try:
        (Path(tempfile.gettempdir()) / "say-voice.lock").unlink(missing_ok=True)
    except OSError:
        pass

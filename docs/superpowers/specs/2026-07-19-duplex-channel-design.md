# A duplex, prioritised voice channel

Status: approved scope. Supersedes nothing; extends the architecture in
[`docs/design.md`](../../design.md).

## The problem this fixes

The channel built so far is a loudspeaker. A session speaks, and there it ends.
Two consequences, both measured against real use rather than argued:

1. **Questions cost a context switch.** When a session needs an answer it speaks
   the question and stops. Answering means finding that session and typing.
   Dictation exists (`stt.py`), but it delivers through the clipboard and a
   human paste: the person is still the transport.
2. **Nothing has rank.** `acquire_speech_lock` (`say.py:722`) treats "halfway
   through the refactor" and "production is down" identically. Worse, it is not
   even first-come-first-served: it is a spin loop over `O_EXCL`, so among
   several waiters an arbitrary one wins. The hub's visible queue is ordered by
   arrival of the `queued` event, so **the UI can disagree with what actually
   plays**.

Two latent bugs sit next to this, and the first turns real the moment a process
holds its turn while waiting for an answer:

- `release()` (`say.py:742`) unlinks the lockfile without checking it still owns
  it. Hold the lock past `stale=180s`, another process deletes your lockfile and
  enters; when you finish, your `release()` deletes *theirs*.
- `hub.stt_target` (`hub.py:157`) is never cleared after delivery.

## What is being built

### 1. `say.py --ask`: the question returns the answer

```bash
python say.py --ask "Voice hub: ¿mergeo a main o abro PR?"
# speaks the question, listens, prints the transcript to stdout:
#   abre PR
```

The agent gets the spoken answer **as the result of its Bash call** and
continues. That is the whole point: the keyboard leaves the loop.

Exit codes are the contract, because the failure case must not be papered over:

| Code | Meaning |
| --- | --- |
| 0 | A transcript was captured; it is on stdout |
| 3 | Nobody answered (timeout, silence, or transcription failed) |

On 3 the agent knows it was not answered and decides what to do. It never
receives an invented answer — the same reasoning that already makes `stt.py`
reject silence rather than transcribe it (`stt.py:145`), because a hosted model
asked to transcribe silence will confidently make something up.

**Who owns the microphone.** `say.py` records directly, importing `stt.py`. The
alternative was to delegate to the hub over a reverse spool, reusing its device
warm-up. Rejected: it makes the hub a hard dependency of a critical path, and
the architecture's most valuable property today is that *the hub can die,
restart or be absent without affecting speech*. The hub only observes, via a new
`listening` event state.

**Hands-free, but bounded.** The mic is not always open. It opens only after an
explicit `--ask`, and closes on whichever comes first: trailing silence after
speech (endpointing), or a hard timeout. `stt.py` already rejects silence and
bracketed sound events, which is the defence against capturing audio that was
not addressed to it. The orb shows a distinct listening state, so it is never
ambiguous whether the mic is live.

### 2. Real priority: `turn.py` replaces the lock

A ticket directory ordered by (priority, arrival):

```
hub/queue/{prio}-{ts:017.3f}-{pid}.json
```

Each process writes its ticket, then waits until its own is first in sorted
order. Bands:

```
blocker (0) > question (1) > outcome (2) > ping (3)
```

A real blocker overtakes three progress pings. Within a band, arrival order is
exact rather than arbitrary, so the hub's queue stops lying.

Two properties fall out of the design rather than being bolted on:

- **The ownership bug cannot recur.** A ticket's filename contains its own pid,
  so a process can only ever unlink the file it created.
- **Crash recovery gets 6× faster.** A held turn is heartbeated by a daemon
  thread, so "stale" no longer has to be generous enough to cover a slow
  utterance. It drops from 180s to 30s: a crashed `say.py` blocks the others for
  half a minute instead of three.

Never blocks forever and never goes silent: on timeout the caller proceeds
without a turn, exactly as the lock did.

### 3. The two bugs

`release()` becomes ownership-safe by construction (above). `stt_target` is
reset to `None` after delivery.

## Explicitly out of scope

Deferred for the same reason `design.md` deferred per-session voices — these are
better decided after living with the above for a few days:

- **Coalescence.** Merging five simultaneous outcomes into one spoken summary.
  Needs a voice that speaks *for* several sessions, which is a new actor.
- **Global status.** Asking the hub "how is everyone doing?" rather than being
  told by one session.

## Testing

The existing suite covers the event spool, nav, sound events and STT; `say.py`
and `overlay.py` remain untested, and this work does not fix that in general.
What it does test:

- `turn.py` in full, since it is new, small and pure enough to test properly:
  ordering across bands, FIFO within a band, stale reclamation, ownership
  (a release never removes another's ticket), and timeout fallthrough.
- `stt.listen_once` endpointing and its failure paths, with the stream stubbed
  the way `_scribe_request` is already stubbed today.
- The `--ask` exit-code contract.

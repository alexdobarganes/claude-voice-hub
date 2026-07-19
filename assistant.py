"""The voice assistant: one process that owns the microphone.

Until now, listening only happened inside `say.py --ask`: a session spoke a
question, opened the mic itself, and closed it when it had an answer. That works
for answering, and not at all for starting. Speak when no session happens to be
asking and nobody is listening -- which is exactly what happened to Alex, twice,
mid-conversation.

This process closes that half. It holds the microphone continuously and turns
what it hears into messages on the inbox spool, which sessions claim. Two rules
shape everything else:

**One owner.** Two processes cannot record from the same device sanely, so while
this is alive `say.py --ask` does not open the mic; it publishes its question
and waits on the inbox instead (see `inbox.open_question`). If this process is
absent, `--ask` falls back to opening the mic itself, so listening degrades the
way speech already does when the hub is missing.

**Addressed or answering.** An open mic in a real room hears other people, a
television, and this machine's own voice through the speakers. Measured here, a
capture taken while nobody was talking to it returned correctly transcribed
Spanish that simply was not for us. So unprompted speech must name the assistant
(see `wake.py`), and only speech that arrives while a session has an open
question skips that requirement -- because answering a question you were just
asked is how conversation works, and demanding a name first would make this feel
like a kiosk.

Transcription is two-tier for cost, not for quality: every utterance in the room
gets a local pass to decide whether it was even for us, and only what survives
goes to the hosted model. Sending every voice in the room to a paid API to find
out it was the television is not a thing to do quietly.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import inbox
import turn
import wake

THIS_DIR = Path(__file__).resolve().parent
ASSISTANT_LOCK = THIS_DIR / "hub" / "assistant.lock"
LOG = THIS_DIR / "hub" / "assistant.log"

LOCAL_MODEL = os.environ.get("TTS_WAKE_MODEL", "small")
IDLE_EXIT_S = float(os.environ.get("TTS_ASSISTANT_IDLE", "0") or 0)  # 0 = forever
LOCK_STALE_S = 30.0
HEARTBEAT_S = 5.0
# Let the room go quiet after our own audio before trusting the mic again:
# speakers ring, and a capture opened on the last syllable starts with it.
SPEAK_SETTLE_S = 0.6
# How long a message waits for a session to claim it before we fall back to
# the clipboard. Long enough for a --ask poll, short enough not to feel lost.
UNCLAIMED_GRACE_S = 8.0


def log(msg: str) -> None:
    """Append to the assistant log.

    This runs headless like the hub does, so a file is the only place a failure
    can surface. Debug it here or not at all.
    """
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


# ----------------------------- singleton ------------------------------------

def is_running() -> bool:
    """True if another assistant holds the lock and is still alive."""
    try:
        if not ASSISTANT_LOCK.exists():
            return False
        if time.time() - ASSISTANT_LOCK.stat().st_mtime > LOCK_STALE_S:
            return False   # heartbeat stopped: crashed, not running
        pid = int(ASSISTANT_LOCK.read_text().strip())
    except Exception:
        return False
    if pid == os.getpid():
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True   # cannot tell; assume alive rather than double-open the mic


def _take_lock() -> bool:
    ASSISTANT_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if is_running():
        return False
    ASSISTANT_LOCK.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_lock() -> None:
    try:
        if ASSISTANT_LOCK.exists() and \
                ASSISTANT_LOCK.read_text().strip() == str(os.getpid()):
            ASSISTANT_LOCK.unlink()
    except Exception:
        pass


# ----------------------------- local pass -----------------------------------

class LocalTranscriber:
    """Cheap, offline first pass, used only to decide if speech was for us.

    Loaded once and kept: the model costs seconds to bring up and milliseconds
    to run, so paying that per utterance would put the cost in exactly the wrong
    place.
    """

    def __init__(self, model_name: str = LOCAL_MODEL) -> None:
        self.model_name = model_name
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        try:
            self._model = WhisperModel(self.model_name, device="cuda",
                                       compute_type="float16")
            log(f"local model {self.model_name} on cuda")
        except Exception as e:
            log(f"cuda unavailable ({e}); falling back to cpu")
            self._model = WhisperModel(self.model_name, device="cpu",
                                       compute_type="int8")

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe, with the model's own VAD doing the first rejection.

        `vad_filter=True` is not an optimisation here. Handed a stretch of room
        hum with no speech in it, the model does not return an empty string --
        it returns confident nonsense in whatever script it drifts into
        ('ჲლლლბგეი...', observed). Letting its VAD drop the non-speech regions
        first means it is usually asked to transcribe nothing at all, which is
        the one input it handles by saying nothing.
        """
        self.load()
        segments, _info = self._model.transcribe(
            str(wav_path), vad_filter=True, language=None,
            vad_parameters={"min_silence_duration_ms": 300})
        return " ".join(s.text.strip() for s in segments).strip()


# ----------------------------- the loop -------------------------------------

def handle_utterance(text: str, sessions: list[dict],
                     questions: list[dict]) -> dict | None:
    """Decide what an utterance was, and to whom. None means: not for us.

    Pure, so the policy that governs an always-open microphone can be tested
    without one. Returns the inbox payload to deliver.
    """
    if not (text or "").strip() or not wake.looks_like_speech(text):
        return None

    named = wake.route(text, sessions)
    stripped = wake.strip_address(text)

    if stripped is not None:
        # Addressed by name. Route on what is left, so "Claude, billing api,
        # abre el PR" reaches billing api rather than the assistant itself.
        body = stripped or text
        target = wake.route(body, sessions) or named or inbox.BROADCAST
        return {"text": body or text, "target": target, "addressed": True}

    if questions:
        # Someone is waiting on an answer, so speech right now is most likely
        # it. Prefer a session named out loud; otherwise the most recent asker.
        target = named or questions[0].get("session_id") or inbox.BROADCAST
        return {"text": text, "target": target, "addressed": False}

    return None   # the room, not us


def escalate_unclaimed(grace_s: float = UNCLAIMED_GRACE_S) -> int:
    """Deliver messages no session came to collect. Returns how many.

    Without this the feature has a dead end in the middle of it: a session
    sitting in `--ask` claims its answer within a poll, but speaking when
    nothing is waiting drops the message into a spool with no reader, and it
    stays there. Alex talks, and nothing happens -- which is the exact
    complaint the assistant was built to fix, moved one layer down.

    The fallback is the same one nav.py already settled on for dictation: put
    it on the clipboard and focus the app. Typing blind into the desktop app
    would land in whichever session happens to be open, which may not be the
    one meant. So this stops one paste short of hands-free, on purpose.
    """
    import nav
    delivered = 0
    now = time.time()
    for msg in inbox.peek():
        if now - float(msg.get("ts", 0)) < grace_s:
            continue
        text = (msg.get("text") or "").strip()
        target = msg.get("target", inbox.BROADCAST)
        # Claim it first. If a session takes it between peek and here, the
        # unlink fails and it is theirs, not ours to deliver twice.
        claimed = inbox.take(target if target != inbox.BROADCAST else "",
                             accept_broadcast=True)
        if not claimed or not text:
            continue
        status = nav.deliver_text({"session_id": target, "nav": None}, text)
        log(f"escalated to clipboard ({status}): {text[:50]!r}")
        delivered += 1
    return delivered


def _sessions() -> list[dict]:
    """Live sessions as {session_id, name} for routing. Never raises."""
    try:
        import agents_registry
        return [{"session_id": sid, "name": rec.get("name") or ""}
                for sid, rec in agents_registry.refresh().items()]
    except Exception:
        return []   # routing degrades to the default target, never to a crash


def listen_forever(local: "LocalTranscriber | None" = None,
                   once: bool = False) -> None:
    """Own the mic and turn what is said to us into inbox messages."""
    import stt

    local = local or LocalTranscriber()
    # Load before opening the mic, not on the first transcription. The model
    # takes ~14 s to come up; lazily, that cost lands *after* the first capture,
    # so the very first thing said to a freshly started assistant is heard by
    # nobody. Which is precisely the failure this process exists to remove.
    try:
        local.load()
    except Exception as e:
        log(f"local model failed to load: {e}")
        return

    device = stt.pick_input_device()
    log(f"listening on device {device}")
    last_beat = 0.0
    started = time.time()

    while True:
        now = time.time()
        if now - last_beat > HEARTBEAT_S:
            try:
                ASSISTANT_LOCK.write_text(str(os.getpid()), encoding="utf-8")
            except Exception:
                pass
            last_beat = now
        if IDLE_EXIT_S and now - started > IDLE_EXIT_S:
            log("idle exit")
            return
        escalate_unclaimed()

        # Do not record while the machine is talking. One capture that spanned
        # a spoken instruction and the reply to it came back as a single
        # sentence beginning with the assistant's own words -- it had no way to
        # tell its voice from Alex's, because acoustically there is none.
        if turn.is_speaking():
            time.sleep(SPEAK_SETTLE_S)
            continue

        capture_start = time.time()
        try:
            wav = stt.capture_until_silence(timeout=30.0, silence_tail=1.0,
                                            device=device)
            log(f"captured {wav.stat().st_size / 32000:.1f}s")
        except stt.SttError:
            # Log it. Silence producing no line at all makes a listener that is
            # working indistinguishable from one that has wedged.
            log("nothing said in that window")
            continue
        except Exception as e:
            log(f"capture failed: {e}")
            time.sleep(1.0)
            continue

        if turn.spoke_since(capture_start):
            # Some of this recording is our own voice. Asking whether we are
            # speaking *now* was not enough: an utterance can begin and end
            # inside one capture window, and it did -- a thirty-second capture
            # came back as a verbatim transcript of the machine's own speech.
            # Discarding costs one repeat; keeping it means answering ourselves.
            log("discarded: our own audio overlapped this capture")
            wav.unlink(missing_ok=True)
            time.sleep(SPEAK_SETTLE_S)
            continue

        try:
            heard = local.transcribe(wav)
        except Exception as e:
            log(f"local transcribe failed: {e}")
            heard = ""
        finally:
            wav.unlink(missing_ok=True)

        if not heard:
            log("heard nothing (vad dropped it)")
            continue
        questions = inbox.open_questions()
        decision = handle_utterance(heard, _sessions(), questions)
        if decision is None:
            # Say which rule rejected it. "ignored" alone makes an assistant
            # that cannot hear you indistinguishable from one ignoring you.
            why = ("not speech" if not wake.looks_like_speech(heard)
                   else "not addressed, nobody waiting")
            log(f"ignored ({why}): {heard[:70]!r}")
            if once:
                return
            continue

        inbox.put(decision["text"], target=decision["target"],
                  meta={"addressed": decision["addressed"], "heard": heard})
        log(f"-> {decision['target']}: {decision['text'][:60]!r}")
        if once:
            return


def _synthesize(phrase: str, out: Path) -> bool:
    import subprocess
    subprocess.run(
        [sys.executable, str(THIS_DIR / "say.py"), "--no-play", "--no-hub",
         "--raw", "--out", str(out), phrase],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return out.exists()


def self_test(cases: "list[tuple[str, bool]] | None" = None,
              verbose: bool = True) -> int:
    """Run synthesized speech through the whole decision path. No mic, no Alex.

    Everything except the microphone is exercised: real synthesis, the local
    recognizer, wake matching, routing. That covers the part that was hardest to
    get right and least amenable to guessing -- what a recognizer actually
    returns for the name -- without needing someone in the room to perform on
    request. Verifying with a person present is still the real test; this is
    what makes the other ninety percent of the iterations cheap, and it caught a
    real bug on its first run.

    The two error kinds are reported separately because they are not equally
    bad. A MISS costs Alex saying the name again. A FALSE WAKE delivers into a
    session something he never said to it.
    """
    import tempfile

    if cases is None:
        sys.path.insert(0, str(THIS_DIR / "tests"))
        import wake_corpus
        cases = wake_corpus.ALL

    local = LocalTranscriber()
    local.load()
    misses: list[tuple[str, str]] = []
    false_wakes: list[tuple[str, str]] = []
    broken: list[str] = []

    for phrase, expected in cases:
        wav = Path(tempfile.gettempdir()) / f"selftest-{abs(hash(phrase))}.wav"
        if not _synthesize(phrase, wav):
            broken.append(phrase)
            continue
        heard = local.transcribe(wav)
        wav.unlink(missing_ok=True)
        got = handle_utterance(heard, _sessions(), []) is not None
        if got == expected:
            if verbose:
                print(f"  ok   {phrase!r}")
            continue
        (misses if expected else false_wakes).append((phrase, heard))
        print(f"  {'MISS' if expected else 'FALSE WAKE'}  said={phrase!r}\n"
              f"        heard={heard!r}")

    total = len(cases)
    bad = len(misses) + len(false_wakes) + len(broken)
    print(f"\n{total - bad}/{total} correct   "
          f"misses={len(misses)}  false wakes={len(false_wakes)}  "
          f"synth failures={len(broken)}")
    if false_wakes:
        print("False wakes are the serious ones: they put words Alex never "
              "said into a session.")
    return 1 if bad else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return self_test()
    if not _take_lock():
        print("assistant already running", file=sys.stderr)
        return 0
    log(f"assistant up (pid {os.getpid()})")
    try:
        listen_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _release_lock()
        log("assistant down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

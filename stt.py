"""Push-to-talk speech-to-text for the Voice Hub.

Records the default mic, transcribes via ElevenLabs Scribe, falls back to
faster-whisper when it is installed. Delivery is the caller's job: see
nav.deliver_text, which never types blind into an ambiguous target.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 16000

# Below this RMS the capture is silence. Measured on this machine: a dead line
# input reads ~10, a live mic hearing a quiet room reads ~106. Silence must be
# rejected rather than transcribed: Scribe answers silence with a confident
# hallucination ('[outro jingle]'), which would be delivered as if it were real.
SILENCE_RMS = 35.0

# Inputs that are loopbacks, virtual cables or line ins rather than microphones.
# They can carry loud audio (music, system output) and would otherwise win a
# loudness contest against the actual microphone.
_NOT_A_MIC = ("voicemeeter", "cable", "virtual", "stereo mix", "line in",
              "sound mapper", "primary sound", "wave", "aux", "analogue")

# Host APIs to use only as a last resort. Windows exposes the same physical mic
# through several of them, and WDM-KS is the raw kernel-streaming path: measured
# here, one webcam mic reads a noise floor of ~834 through WDM-KS and ~22
# through MME. Since the device is chosen by which one hears most, WDM-KS won
# every time and took the noise with it -- the calibrated threshold landed at
# 2501, above ordinary speech, and questions went unanswered while they were
# being answered. It also rejects blocking reads outright.
_LAST_RESORT_HOSTS = ("windows wdm-ks",)


def _load_dotenv() -> None:
    envf = THIS_DIR / ".env"
    if not envf.exists():
        return
    for ln in envf.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


class SttError(RuntimeError):
    pass


def _query_devices() -> list:
    import sounddevice as sd
    return list(sd.query_devices())


def _probe_level(index: int, seconds: float = 0.35) -> float:
    """RMS of a short capture from one device. 0.0 when it cannot be opened."""
    import numpy as np
    import sounddevice as sd
    try:
        rec = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                     channels=1, dtype="int16", device=index)
        sd.wait()
        return float(np.sqrt(np.mean(rec.astype("float64") ** 2)))
    except Exception:
        return 0.0


_picked_device: int | None = None
_picked = False


def pick_input_device(force: bool = False) -> int | None:
    """Index of the microphone to record from.

    The OS default is not trustworthy here: on this machine it is a Focusrite
    line input that reads as silence while you talk into the webcam mic. So
    probe the real microphones and take the one that actually hears the room.
    Set TTS_STT_DEVICE to pin a device and skip probing.

    "Hears most" is only a sound rule among comparable devices. Windows exposes
    one physical mic through several host APIs, and the noisiest of those wins a
    loudness contest purely by being noisier, not by hearing better. So
    last-resort host APIs are considered only when nothing else responds.

    The result is cached: probing costs ~0.35s per candidate, and paying that
    on every dictation would swallow the first words spoken after the hotkey.
    """
    global _picked_device, _picked
    override = os.environ.get("TTS_STT_DEVICE", "").strip()
    if override.isdigit():
        return int(override)
    if _picked and not force:
        return _picked_device
    best = _probe_best(preferred=True)
    if best is None:
        best = _probe_best(preferred=False)
    _picked_device, _picked = best, True
    return best


def _host_api_name(dev: dict) -> str:
    import sounddevice as sd
    try:
        return (sd.query_hostapis()[dev.get("hostapi", -1)]["name"] or "").lower()
    except Exception:
        return ""


def _probe_best(preferred: bool) -> int | None:
    """Loudest responding mic, restricted to (or excluding) preferred hosts."""
    best, best_level = None, 0.0
    for idx, dev in enumerate(_query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = (dev.get("name") or "").lower()
        if any(bad in name for bad in _NOT_A_MIC):
            continue
        last_resort = any(h in _host_api_name(dev) for h in _LAST_RESORT_HOSTS)
        if last_resort == preferred:
            continue
        level = _probe_level(idx)
        if level > best_level:
            best, best_level = idx, level
    return best


def _close_stream(stream) -> None:
    stream.stop()
    stream.close()


class Recorder:
    """Start/stop mic capture; stop() writes a 16 kHz mono PCM16 WAV."""

    def __init__(self, device: int | None = None) -> None:
        self._frames: list = []
        self._stream = None
        self.device = device
        self.level = 0.0

    def start(self) -> None:
        import sounddevice as sd
        self._frames = []
        if self.device is None:
            self.device = pick_input_device()

        def cb(indata, _frames, _time, _status):
            self._frames.append(indata.copy())

        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                      dtype="int16", callback=cb,
                                      device=self.device)
        self._stream.start()

    def stop(self) -> Path:
        import numpy as np
        import soundfile as sf
        if self._stream is None:
            raise SttError("recorder was never started")
        _close_stream(self._stream)
        self._stream = None
        if not self._frames:
            raise SttError("no audio captured")
        audio = np.concatenate(self._frames)
        self.level = float(np.sqrt(np.mean(audio.astype("float64") ** 2)))
        if self.level < SILENCE_RMS:
            raise SttError(
                f"no se escucho nada (nivel {self.level:.0f}, dispositivo "
                f"{self.device}); revisa que hablas al microfono correcto")
        out = Path(tempfile.gettempdir()) / f"stt-{uuid.uuid4().hex}.wav"
        sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
        return out


class Endpointer:
    """Decides when a hands-free capture is over, from a stream of block levels.

    Push-to-talk has an obvious end: the key comes up. Hands-free has to infer
    one, and the two ways to get it wrong are not symmetric. Cutting someone off
    mid-sentence loses the answer; hanging on a little too long costs a pause.
    So the rule is trailing silence after speech, never a fixed window.

    The speech threshold is calibrated from the room rather than hard-coded.
    A fixed number cannot work across machines: on this one a dead line input
    reads ~10 while a live mic in a quiet room reads ~106, so any constant is
    either deaf in one room or permanently triggered in the other.

    Pure and stepped by `feed()` so it can be tested without a sound card.
    """

    # Multiple of the measured noise floor that counts as someone talking.
    # Measured against a webcam mic in this room: the floor is not stationary,
    # its median wanders between ~300 and ~820 with excursions past 1700. A
    # tight multiple over a mean would sit inside that wander and call the room
    # a speaker, so the floor is taken as a high percentile and the margin is
    # wide.
    SPEECH_OVER_FLOOR = 3.0
    FLOOR_PERCENTILE = 0.9
    FLOOR_FALLBACK = SILENCE_RMS

    # Speech may dip below threshold between syllables; noise does not have to.
    # Tolerating a short dip keeps a run intact without letting scattered
    # spikes join up into one.
    GAP_TOLERANCE_BLOCKS = 2

    def __init__(self, timeout: float = 20.0, silence_tail: float = 1.4,
                 calibrate_s: float = 0.8, max_total: float = 60.0,
                 min_speech_s: float = 0.25, block_s: float = 0.05,
                 settle_s: float = 0.5) -> None:
        self.timeout = timeout
        self.silence_tail = silence_tail
        # Throw the opening away before measuring anything. Two things
        # contaminate it, and both inflate the floor: the tail of the question
        # we just spoke still decaying in the room, and the mic's automatic
        # gain ramping up from cold. Measured here, that window reads a median
        # of ~718 against a true floor of ~22 -- calibrating on it set a
        # threshold of 2068, which is above normal speech, so the session sat
        # there deaf while it was being answered.
        self.settle_s = settle_s
        self.calibrate_s = settle_s + calibrate_s
        self.max_total = max_total
        # The run has to be *consecutive*. Counting blocks cumulatively was the
        # bug that let an empty room answer a question: over twenty seconds a
        # fan and a door clear any cumulative total, and Scribe answers noise
        # with a fluent sentence rather than admitting it heard nothing. A
        # sustained run is what actually distinguishes speech from a room.
        #
        # 0.25 s is deliberately short. The measured noise excursions here last
        # about 0.1 s, so a quarter second already separates them, while a
        # longer requirement would start rejecting real one-word answers. The
        # cost of rejecting a genuine "sí" is asking again; the cost of
        # accepting a fan is acting on an answer nobody gave.
        self.min_blocks = max(1, int(round(min_speech_s / block_s)))
        self.floor: float | None = None
        self.threshold = float("inf")
        self._floor_samples: list[float] = []
        self._run = 0
        self._gap = 0
        self.heard_speech = False
        self._last_speech = 0.0

    def _calibrate(self) -> None:
        if self._floor_samples:
            ordered = sorted(self._floor_samples)
            idx = min(len(ordered) - 1,
                      int(len(ordered) * self.FLOOR_PERCENTILE))
            self.floor = ordered[idx]
        else:
            self.floor = self.FLOOR_FALLBACK
        # A floor of ~0 (muted or dead input) would make any breath count as
        # speech, so never calibrate below the known-silence level.
        self.threshold = max(self.floor * self.SPEECH_OVER_FLOOR,
                             self.FLOOR_FALLBACK)

    def feed(self, rms: float, elapsed: float) -> str:
        """Return 'calibrating' | 'listening' | 'speech' | 'done' | 'timeout'."""
        if elapsed < self.settle_s:
            return "calibrating"   # measured but discarded: see settle_s
        if elapsed < self.calibrate_s:
            self._floor_samples.append(rms)
            return "calibrating"
        if self.floor is None:
            self._calibrate()

        if rms >= self.threshold:
            self._run += 1
            self._gap = 0
            self._last_speech = elapsed
            if self._run >= self.min_blocks:
                self.heard_speech = True
        else:
            self._gap += 1
            if self._gap > self.GAP_TOLERANCE_BLOCKS:
                self._run = 0

        if self.heard_speech:
            if elapsed - self._last_speech >= self.silence_tail:
                return "done"
            if elapsed >= self.max_total:
                return "done"
            return "speech"
        if elapsed >= self.timeout:
            return "timeout"
        return "listening"


BLOCK_S = 0.05


def _open_input_stream(device: int | None, callback):
    """Open the mic. Isolated so tests can stub it, like _scribe_request.

    Callback-driven rather than blocking reads: PortAudio's WDM-KS host API
    rejects blocking reads outright ('Blocking API not supported yet'), which
    is the API a webcam mic lands on here. Recorder already takes this path;
    hands-free capture has no reason to take a different one.
    """
    import sounddevice as sd
    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                          blocksize=int(SAMPLE_RATE * BLOCK_S),
                          callback=callback, device=device)


def capture_until_silence(timeout: float = 20.0, silence_tail: float = 1.4,
                          device: int | None = None,
                          on_state=None) -> Path:
    """Record hands-free until the speaker stops. Returns a 16 kHz mono WAV.

    Raises SttError if nobody spoke, which the caller must surface rather than
    smooth over: a session that asked a question needs to learn it was not
    answered instead of receiving an invention.
    """
    import numpy as np
    import soundfile as sf

    if device is None:
        device = pick_input_device()
    ep = Endpointer(timeout=timeout, silence_tail=silence_tail, block_s=BLOCK_S)
    blocks: list = []
    pending: list = []

    def cb(indata, _frames, _time, _status):
        pending.append(indata.copy())

    stream = _open_input_stream(device, cb)
    stream.start()
    started = _now()
    last_state = ""
    state = "listening"
    try:
        while True:
            if not pending:
                time.sleep(BLOCK_S / 2)
                # Still step the clock, or a mic that delivers nothing at all
                # would spin here forever instead of timing out.
                state = ep.feed(0.0, _now() - started)
            else:
                while pending:
                    block = pending.pop(0).reshape(-1)
                    blocks.append(block)
                    rms = (float(np.sqrt(np.mean(block.astype("float64") ** 2)))
                           if block.size else 0.0)
                    state = ep.feed(rms, _now() - started)
            if on_state and state != last_state:
                on_state(state)
                last_state = state
            if state in ("done", "timeout"):
                break
    finally:
        _close_stream(stream)

    if not ep.heard_speech:
        raise SttError(
            f"nadie contesto (nivel de sala {ep.floor or 0:.0f}, umbral "
            f"{ep.threshold:.0f}, dispositivo {device}); nada que transcribir")
    audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype="int16")
    out = Path(tempfile.gettempdir()) / f"stt-{uuid.uuid4().hex}.wav"
    sf.write(str(out), audio, SAMPLE_RATE, subtype="PCM_16")
    return out


def listen_once(timeout: float = 20.0, silence_tail: float = 1.4,
                device: int | None = None, on_state=None) -> str:
    """Hands-free capture plus transcription. Raises SttError if unanswered."""
    wav = capture_until_silence(timeout=timeout, silence_tail=silence_tail,
                                device=device, on_state=on_state)
    try:
        return transcribe(wav)
    finally:
        wav.unlink(missing_ok=True)


def _now() -> float:
    return time.monotonic()


def _scribe_request(wav_path: Path) -> dict:
    """POST the wav to ElevenLabs Scribe. Isolated so tests can stub it."""
    import urllib.request
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise SttError("no ELEVENLABS_API_KEY")
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model_id\"\r\n\r\n"
        f"scribe_v1\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
    ).encode() + wav_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        "https://api.elevenlabs.io/v1/speech-to-text", data=body, method="POST",
        headers={"xi-api-key": key,
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _whisper_transcribe(wav_path: Path) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="auto")
    segments, _info = model.transcribe(str(wav_path))
    return " ".join(s.text.strip() for s in segments).strip()


def strip_sound_events(text: str) -> str:
    """Drop Scribe's bracketed sound-event captions and keep actual speech.

    Given non-speech, Scribe confidently returns things like '[outro jingle]' or
    '[keyboard clacking]'. Delivering those as dictation is worse than failing,
    so anything left bracketed counts as nothing said.
    """
    return re.sub(r"\[[^\]]*\]", " ", text).strip(" .,;:-\t\n")


def transcribe(wav_path: Path) -> str:
    """Speech in the recording. Raises SttError when nothing was actually said."""
    import sys
    raw = ""
    try:
        raw = str(_scribe_request(wav_path).get("text", "")).strip()
    except Exception as e:
        print(f"[stt] scribe failed: {e}", file=sys.stderr)

    text = strip_sound_events(raw)
    if text:
        return text
    if raw:
        # Scribe heard the recording but found no words in it. Falling through
        # to another backend would only produce a second guess at noise, so
        # report it instead of delivering a caption as if it were dictation.
        raise SttError(f"no se detecto habla, solo sonidos ({raw[:40]})")

    if _whisper_available():
        try:
            text = strip_sound_events(_whisper_transcribe(wav_path))
            if text:
                return text
        except Exception as e:
            print(f"[stt] whisper failed: {e}", file=sys.stderr)
    raise SttError("all STT backends failed")

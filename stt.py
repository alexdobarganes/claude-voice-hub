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

    The result is cached: probing costs ~0.35s per candidate, and paying that
    on every dictation would swallow the first words spoken after the hotkey.
    """
    global _picked_device, _picked
    override = os.environ.get("TTS_STT_DEVICE", "").strip()
    if override.isdigit():
        return int(override)
    if _picked and not force:
        return _picked_device
    best, best_level = None, 0.0
    for idx, dev in enumerate(_query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        name = (dev.get("name") or "").lower()
        if any(bad in name for bad in _NOT_A_MIC):
            continue
        level = _probe_level(idx)
        if level > best_level:
            best, best_level = idx, level
    _picked_device, _picked = best, True
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

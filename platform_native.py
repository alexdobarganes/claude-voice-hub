"""Platform-specific primitives, behind one interface.

Everything the rest of the code needs from the OS lives here: the built-in
text-to-speech voice, a last-resort way to play a file, and the console title.
Each function picks an implementation from `sys.platform` and **degrades to a
no-op rather than raising** when the platform has nothing to offer, which is the
same contract `SapiBackend.available()` already followed on Windows.

Why this exists: the native voice is the end of the fallback chain. When every
network and model backend fails, this is what still speaks, so it has to work on
every platform we claim to support -- otherwise the chain silently ends one link
early.

Status: the Windows path is in daily use. The macOS path is written against the
documented behaviour of `say(1)` and `afplay(1)` but has not been run on a Mac.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


# ----------------------------- native voice --------------------------------

def native_tts_available() -> bool:
    """True if this platform ships a usable offline voice."""
    if IS_WINDOWS:
        return True  # SAPI is part of the OS
    if IS_MACOS:
        return shutil.which("say") is not None
    return False


def _mac_voice_for(lang: str) -> str:
    """Pick an installed macOS voice for `lang`, or "" to use the default.

    `say -v '?'` lists voices as "Name  lang_REGION  # sample sentence", so the
    locale prefix is what we match on. Returning "" is fine: `say` then uses the
    system voice, which is the correct degradation.
    """
    if lang != "es":
        return ""
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, check=True).stdout
    except Exception:
        return ""
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("es"):
            return parts[0]
    return ""


def native_tts_synth(text: str, lang: str, out: Path, *, rate_int: int = 3) -> Path:
    """Synthesize `text` to a WAV at `out` with the OS voice.

    `rate_int` is the SAPI-style -10..+10 delta; each platform maps it onto its
    own scale so callers do not have to care which one is running.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    if IS_WINDOWS:
        rate = max(-10, min(10, int(rate_int or 0) + 3))  # keep the old baseline
        ps = f"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voices = $s.GetInstalledVoices() | ForEach-Object {{ $_.VoiceInfo.Name }}
$es = $voices | Where-Object {{ $_ -match 'Sabina|Helena|Dalia|Raul|Pablo|Spanish|Espanol' }} | Select-Object -First 1
if ('{lang}' -eq 'es' -and $es) {{ $s.SelectVoice($es) }}
elseif ('{lang}' -eq 'es') {{ try {{ $s.SelectVoice('Microsoft Zira Desktop') }} catch {{}} }}
else {{ try {{ $s.SelectVoice('Microsoft Zira Desktop') }} catch {{}} }}
$s.Rate = {rate}
$s.Volume = 100
$s.SetOutputToWaveFile('{out}')
$s.Speak(@'
{text}
'@)
$s.SetOutputToNull()
$s.Dispose()
"""
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True,
                       capture_output=True, text=True)
        return out

    if IS_MACOS:
        # `say` speaks in words per minute around a ~175 default. Map the -10..+10
        # delta onto +-100 wpm so the extremes stay intelligible.
        wpm = max(90, min(320, 175 + int(rate_int or 0) * 10))
        cmd = ["say", "-r", str(wpm), "-o", str(out),
               "--data-format=LEI16@22050"]  # LEI16 + .wav gives a plain RIFF file
        voice = _mac_voice_for(lang)
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return out

    raise RuntimeError(f"no native voice on {sys.platform}")


# ----------------------------- playback fallback ---------------------------

def play_file(path: Path, *, blocking: bool = True) -> bool:
    """Play a file with an OS tool. Last resort for when sounddevice is absent.

    Returns False when the platform gives us nothing, so the caller can report
    silence instead of assuming the audio was heard.
    """
    if IS_WINDOWS:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
                       check=False)
        return True
    if IS_MACOS:
        if shutil.which("afplay") is None:
            return False
        run = subprocess.run if blocking else subprocess.Popen
        run(["afplay", str(path)])
        return True
    return False


# ----------------------------- console identity ----------------------------

def console_title() -> str:
    """Title of the console this process is attached to, or "".

    Windows exposes it directly. macOS terminals do not offer an equivalent, and
    that is not a problem: this is only a *fallback* for session identity, and
    the primary source is the CLAUDE_CODE_SESSION_ID environment variable.
    """
    if not IS_WINDOWS:
        return ""
    buf = ctypes.create_unicode_buffer(512)
    n = ctypes.windll.kernel32.GetConsoleTitleW(buf, 512)
    return buf.value.strip() if n else ""

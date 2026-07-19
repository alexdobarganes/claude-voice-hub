"""Platform-specific primitives, behind one interface.

Everything the rest of the code needs from the OS lives here: the built-in
text-to-speech voice, a last-resort way to play a file, and the console title.
Each function picks an implementation from `sys.platform` and **degrades to a
no-op rather than raising** when the platform has nothing to offer, which is the
same contract the Windows-only SAPI backend already followed.

Why this exists: the native voice is the end of the fallback chain. When every
network and model backend fails, this is what still speaks, so it has to work on
every platform we claim to support -- otherwise the chain silently ends one link
early.

Status: the Windows path is in daily use. The macOS path is written against the
documented behaviour of `say(1)` and `afplay(1)` but has not been run on a Mac.
"""

from __future__ import annotations

import ctypes
import re
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

    `say -v '?'` prints "Name  locale  # sample sentence". Splitting on
    whitespace is wrong: modern voice names contain spaces and parentheses
    ("Eddy (Spanish (Spain))    es_ES    # ..."), so the name is everything
    before the locale token, not the first word.

    Returning "" is a correct degradation: `say` then uses the system voice.
    """
    if not lang:
        return ""
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, check=True).stdout
    except Exception:
        return ""
    # Name, then a locale like es_ES / en_GB, then the sample sentence.
    pattern = re.compile(r"^(.*?)\s+([a-z]{2})[_-]([A-Z]{2})\b")
    for line in listing.splitlines():
        m = pattern.match(line)
        if m and m.group(2) == lang:
            return m.group(1).strip()
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
        # Text goes over stdin, not argv: `say` reads stdin when given no text
        # argument, and this way an utterance starting with "-" is spoken rather
        # than parsed as a flag.
        subprocess.run(cmd, input=text, check=True, capture_output=True, text=True)
        return out

    raise RuntimeError(f"no native voice on {sys.platform}")


# ----------------------------- playback fallback ---------------------------

def play_file(path: Path) -> bool:
    """Play a file with an OS tool, blocking until it finishes.

    Last resort for when sounddevice is absent. Returns False when the platform
    gives us nothing, so the caller can report silence instead of assuming the
    audio was heard.
    """
    if IS_WINDOWS:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(New-Object Media.SoundPlayer '{path}').PlaySync()"],
                       check=False)
        return True
    if IS_MACOS:
        if shutil.which("afplay") is None:
            return False
        subprocess.run(["afplay", str(path)], check=False)
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

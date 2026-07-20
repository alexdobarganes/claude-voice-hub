"""Unified TTS tool. Auto-detects best available backend at runtime.

Priority chain:
  1. ElevenLabs (hosted, expressive)                  -- if ELEVENLABS_API_KEY is set
  2. Kokoro (local neural, em_alex es / af_heart en)  -- the fallback: no network, no key, ~350MB
  3. Supertonic (local, lightest)                     -- if installed
  4. Edge TTS (Microsoft neural cloud, free)          -- needs network
  5. F5-TTS (GPU, neural, voice-clones a ref)         -- if TTS_F5_ENABLED=1
  6. SAPI via WAV + sounddevice                       -- always available, robotic

Single entry point handles language detection, voice selection, rate, output
device. Replaces having to remember which backend is alive today.

Quality stack on top of the chosen backend:
  - Text preprocessing: tickers, currency, percentages, large numbers, K/M
    suffixes, common abbreviations, and `<break Nms>` markers get rewritten so
    Edge speaks them naturally (it doesn't accept SSML on the wire).
  - Voice + prosody presets (--preset neutral|excited|calm|dramatic|news|brief)
    encapsulate (voice, rate, pitch) so swapping a vibe is one flag.
  - Post-processing pipeline (pydub): DC offset removal, peak/RMS loudness
    normalization to a consistent target, gentle warm EQ, optional pseudo-stereo
    Haas effect for mono input. Disable with --no-postfx.

Usage:
    python say.py "Hola mundo"                      # auto-detect lang + backend
    python say.py --lang en --rate 4 "Hi"          # English, faster
    python say.py --backend native "..."           # force backend
    python say.py --preset excited "Subio NVDA"    # vibe preset
    python say.py --list                           # show backends + voices + devices
    python say.py --list-presets                   # show all presets
    python say.py --device 8 "..."                 # specific audio device index
    python say.py --no-play --out clip.wav "..."   # generate WAV only
    python say.py --no-postfx "..."                # skip post-processing
    python say.py --seed-ref                       # (re)generate F5 seed ref WAV

Env vars:
    TTS_BACKEND     force backend (elevenlabs | supertonic | kokoro | edge | f5 | native)
    TTS_RATE        SAPI/edge rate delta, -10..+10 (default 0)
    TTS_DEVICE      output device index for sounddevice
    TTS_VOICE_REF   path to reference WAV for F5-TTS voice cloning
    TTS_F5_ENABLED  set to "1" to allow F5-TTS (otherwise skipped to avoid ~1.3GB download)
    TTS_PRESET      default preset name
    TTS_NO_POSTFX   set to "1" to disable post-processing
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import platform_native
import turn

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent


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

_EL_KEY = os.environ.get("ELEVENLABS_API_KEY", "").strip()
_EL_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or "21m00Tcm4TlvDq8ikWAM"
_EL_MODEL = os.environ.get("ELEVENLABS_MODEL", "eleven_v3")

# Named voice aliases → ElevenLabs voice IDs. Use with --voice <alias>.
_EL_VOICE_ALIASES: dict[str, str] = {
    "liam":   "TX3LPaxmHKxFdv7VOQHJ",  # Energetic, Social Media Creator
    "charlie": "IKne3meq5aSn9XLyUdCD",  # Deep, Confident, Energetic
    "harry":  "SOYHLrjzK2X1ezoPC6cr",  # Fierce Warrior
    "george": "JBFqnCBsd6RMkjVDRZzb",  # Warm, Captivating Storyteller
}

_EL_SETTINGS: dict[str, dict] = {
    "neutral":  {"stability": 0.50, "similarity_boost": 0.80, "style": 0.00, "speed": 1.00},
    "excited":  {"stability": 0.30, "similarity_boost": 0.75, "style": 0.55, "speed": 1.08},
    "calm":     {"stability": 0.75, "similarity_boost": 0.80, "style": 0.10, "speed": 0.94},
    "dramatic": {"stability": 0.28, "similarity_boost": 0.75, "style": 0.80, "speed": 0.90},
    "news":     {"stability": 0.62, "similarity_boost": 0.85, "style": 0.18, "speed": 1.02},
    "brief":    {"stability": 0.50, "similarity_boost": 0.80, "style": 0.00, "speed": 1.15},
}

_AUDIO_TAG_RE = re.compile(r"\[(?:excited|happy|sad|whispers?|shouts?|sighs?|laughs?|nervous|sarcastic|curious|deadpan|reassuring)\]", re.IGNORECASE)


def _strip_audio_tags(text: str) -> str:
    """Remove ElevenLabs audio tags from text for backends that don't support them."""
    return _AUDIO_TAG_RE.sub("", text).strip()
REFS_DIR = THIS_DIR / "refs"
DEFAULT_REF_WAV = REFS_DIR / "jorge_seed.wav"

# ----------------------------- presets -------------------------------------
# A preset is a (voice_es, voice_en, rate, pitch) tuple per vibe. Adding a new
# vibe is a one-line entry below; nothing else needs to change.

@dataclass(frozen=True)
class Preset:
    name: str
    voice_es: str
    voice_en: str
    rate: str   # edge-tts rate string, e.g. "+25%"
    pitch: str  # edge-tts pitch string, e.g. "+8Hz"
    # Style hint for post-processing intensity 0.0..1.0 (1.0 = max warmth/loud).
    warmth: float = 0.5

# Voices: locale-specific Jorge/Dalia for es, multilingual Andrew/Ava for en.
# Multilingual voices can speak Spanish but the locale voices tend to nail
# accent and prosody better. Andrew/Ava are reserved for English unless a
# preset explicitly opts into them.
V_JORGE = "es-MX-JorgeNeural"
V_DALIA = "es-MX-DaliaNeural"
V_ANDREW_ML = "en-US-AndrewMultilingualNeural"
V_AVA_ML = "en-US-AvaMultilingualNeural"
V_BRIAN_ML = "en-US-BrianMultilingualNeural"
V_GUY = "en-US-GuyNeural"

PRESETS: dict[str, Preset] = {
    # "neutral" is the everyday baseline.
    "neutral":  Preset("neutral",  V_JORGE, V_GUY,       "+0%",  "+0Hz",  warmth=0.5),
    "excited":  Preset("excited",  V_JORGE, V_AVA_ML,    "+35%", "+15Hz", warmth=0.7),
    "calm":     Preset("calm",     V_JORGE, V_ANDREW_ML, "+5%",  "+0Hz",  warmth=0.6),
    "dramatic": Preset("dramatic", V_JORGE, V_ANDREW_ML, "-15%", "-6Hz",  warmth=0.8),
    "news":     Preset("news",     V_DALIA, V_AVA_ML,    "+10%", "+0Hz",  warmth=0.4),
    "brief":    Preset("brief",    V_JORGE, V_GUY,       "+40%", "+10Hz", warmth=0.5),
}

DEFAULT_PRESET = os.environ.get("TTS_PRESET", "neutral")


# ----------------------------- text preprocessing --------------------------
# Edge TTS doesn't accept SSML on the WSS wire (the `Communicate` API takes
# plain text + rate/pitch/volume strings). To get SSML-like behavior we
# preprocess the text so the neural model pronounces things correctly.

_TICKER_MAP = {
    # Tickers that Edge mangles by spelling letter-by-letter or mispronouncing.
    # Replacements are tuned for es-MX-Jorge; English wording works too.
    "ES":   "E-S",      # ES futures -> "E-S" not "ess"
    "NQ":   "N-Q",
    "RTY":  "R-T-Y",
    "YM":   "Y-M",
    "NVDA": "en vidia",
    "AAPL": "Apple",
    "GOOG": "Google",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "META": "Meta",
    "AMZN": "Amazon",
    "AMD":  "A-M-D",
    "SPY":  "S-P-Y",
    "QQQ":  "Q-Q-Q",
    "IWM":  "I-W-M",
    "VIX":  "vix",
    "BTC":  "Bitcoin",
    "ETH":  "Ethereum",
    "SOL":  "Solana",
    "FED":  "Fed",
    "FOMC": "F-O-M-C",
    "CPI":  "C-P-I",
    "PMI":  "P-M-I",
}

_MONEY_SUFFIX = {
    "k": " mil",
    "K": " mil",
    "m": " millones",
    "M": " millones",
    "b": " mil millones",
    "B": " mil millones",
}


def _expand_currency(m: re.Match) -> str:
    sym, num, suffix = m.group("sym"), m.group("num"), (m.group("suf") or "")
    unit = {"$": "dolares", "EUR": "euros", "€": "euros", "USD": "dolares",
            "GBP": "libras", "£": "libras", "MXN": "pesos"}.get(sym, "dolares")
    num_clean = num.replace(",", "").replace(".", "")  # crude but works for "1.500" and "1,500"
    # Keep decimal if it actually looks like decimals (e.g., 1.5 vs 1.500 thousands)
    if "." in num and re.match(r"^\d{1,3}\.\d{1,2}$", num):
        num_clean = num  # treat as decimal
    if suffix:
        return f"{num_clean}{_MONEY_SUFFIX[suffix]} {unit}"
    return f"{num_clean} {unit}"


def _expand_bare_kmb(m: re.Match) -> str:
    return f"{m.group('num')}{_MONEY_SUFFIX[m.group('suf')]}"


def _expand_percent(m: re.Match) -> str:
    return f"{m.group(1)} por ciento"


def _expand_breaks(text: str) -> str:
    """Convert `<break 500ms>` markers into actual SSML-like commas/periods.
    Edge TTS doesn't accept real SSML so we use a punctuation proxy: long
    breaks become " ... ", medium become ", ", short become ", ". """
    def repl(m: re.Match) -> str:
        try:
            ms = int(m.group(1))
        except (TypeError, ValueError):
            ms = 250
        if ms >= 700:
            return " ... "
        if ms >= 300:
            return ", "
        return ", "
    return re.sub(r"<break\s+(\d+)\s*ms\s*/?>", repl, text)


def _expand_tickers(text: str) -> str:
    def repl(m: re.Match) -> str:
        word = m.group(0)
        # Match against uppercase form so case-insensitive input still hits.
        return _TICKER_MAP.get(word.upper(), word)
    # Only match standalone uppercase tokens of length 2-5 that aren't normal words.
    return re.sub(r"\b[A-Z]{2,5}\b", repl, text)


def preprocess_text(text: str, lang: str) -> str:
    """Apply SSML-like rewrites so Edge TTS speaks numbers, money, tickers and
    pause markers naturally. Safe to call on already-clean text."""
    t = text
    t = _expand_breaks(t)
    # Currency: "$5K", "EUR 1.500", "USD 2.5M"
    t = re.sub(
        r"(?P<sym>\$|€|£|EUR|USD|GBP|MXN)\s?(?P<num>\d{1,3}(?:[.,]\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d+)?)(?P<suf>[kKmMbB])?",
        _expand_currency, t,
    )
    # Bare K/M/B suffixes on numbers: "5K", "2.5M" (no currency)
    t = re.sub(
        r"(?<![A-Za-z])(?P<num>\d+(?:\.\d+)?)(?P<suf>[kKmMbB])(?![A-Za-z])",
        _expand_bare_kmb, t,
    )
    # Percent: "12%" or "12.5 %"
    t = re.sub(r"(\d+(?:[.,]\d+)?)\s?%", _expand_percent, t)
    # Tickers (must run after currency so $NVDA already split)
    t = _expand_tickers(t)
    # Collapse runs of spaces.
    t = re.sub(r"[ \t]+", " ", t).strip()
    return t


# ----------------------------- post-processing -----------------------------

@dataclass
class PostFxConfig:
    """Post-processing pipeline configuration. All steps optional + tunable."""
    enabled: bool = True
    remove_dc: bool = True
    # Target loudness expressed as RMS dBFS. -16 LUFS ~ approx -18 dBFS RMS for
    # speech, but pydub's dBFS is full-scale peak, so we target RMS via gain.
    target_dbfs: float = -16.0
    # Peak ceiling (true-peak proxy via simple peak normalize after RMS gain).
    peak_ceiling_db: float = -1.5
    # Gentle warm EQ: low-shelf boost (Hz, dB), high-shelf cut (Hz, dB).
    low_shelf_hz: int = 180
    low_shelf_db: float = 2.5
    high_shelf_hz: int = 5500
    high_shelf_db: float = -1.5
    # Stereo widening: Haas-style delay in ms for the right channel when input
    # is mono. 0 to disable.
    stereo_haas_ms: float = 12.0


def _apply_postfx(in_path: Path, out_path: Path, cfg: PostFxConfig, warmth: float = 0.5) -> bool:
    """Run the post-processing pipeline. Returns True on success.
    Reads in_path (mp3 or wav), writes WAV (PCM 16) to out_path.

    Warmth in [0..1] scales the EQ + stereo width intensity, letting presets
    feel different without redefining the whole pipeline.
    """
    if not cfg.enabled:
        return False
    try:
        from pydub import AudioSegment
        from pydub.effects import low_pass_filter, high_pass_filter
        import numpy as np
    except Exception as e:
        print(f"[postfx] pydub/numpy not available: {e}", file=sys.stderr)
        return False

    try:
        if str(in_path).lower().endswith(".mp3"):
            seg = AudioSegment.from_mp3(str(in_path))
        else:
            seg = AudioSegment.from_file(str(in_path))
    except Exception as e:
        print(f"[postfx] decode failed: {e}", file=sys.stderr)
        return False

    # 1. DC offset removal (high-pass at 30 Hz also kills sub-rumble).
    if cfg.remove_dc:
        seg = high_pass_filter(seg, cutoff=30)

    # 2. Loudness normalization: bring RMS to target, then cap peak.
    if seg.rms > 0:
        # pydub.dBFS is peak-relative; we want RMS target. Compute RMS-dBFS:
        rms_dbfs = 20.0 * (np.log10(seg.rms / seg.max_possible_amplitude) if seg.rms > 0 else -60.0)
        gain = cfg.target_dbfs - rms_dbfs
        # Don't blow eardrums: clip gain.
        gain = float(max(-12.0, min(12.0, gain)))
        seg = seg.apply_gain(gain)
        # Then ensure peak doesn't exceed ceiling.
        peak_dbfs = seg.max_dBFS
        if peak_dbfs > cfg.peak_ceiling_db:
            seg = seg.apply_gain(cfg.peak_ceiling_db - peak_dbfs)

    # 3. Warm EQ. pydub doesn't ship true shelving filters, so we synthesize a
    # low-shelf boost by overlaying a bass-band copy of the signal on itself.
    # Intensity scaled by `warmth`. (We skip a high-shelf cut because pydub's
    # overlay sums without phase inversion, so a "negative-gain" treble overlay
    # would *boost* highs, not cut them. The low-shelf alone gives the warm
    # tilt we want.)
    w = float(max(0.0, min(1.0, warmth)))
    if w > 0:
        # Pre-attenuate so the bass overlay headroom still fits under ceiling.
        seg = seg.apply_gain(-2.0)
        bass = low_pass_filter(seg, cutoff=cfg.low_shelf_hz)
        seg = seg.overlay(bass.apply_gain(cfg.low_shelf_db * w), gain_during_overlay=0)
        # Re-cap peak after EQ.
        peak_dbfs = seg.max_dBFS
        if peak_dbfs > cfg.peak_ceiling_db:
            seg = seg.apply_gain(cfg.peak_ceiling_db - peak_dbfs)

    # 4. Pseudo-stereo Haas widening for mono input.
    if cfg.stereo_haas_ms > 0 and seg.channels == 1:
        delay_ms = int(cfg.stereo_haas_ms * (0.6 + 0.8 * w))
        silence = AudioSegment.silent(duration=delay_ms, frame_rate=seg.frame_rate)
        right = silence + seg
        # Pad left with equal trailing silence so lengths match.
        left = seg + AudioSegment.silent(duration=delay_ms, frame_rate=seg.frame_rate)
        seg = AudioSegment.from_mono_audiosegments(left, right)

    # 5. Ensure 16-bit PCM WAV out.
    seg = seg.set_sample_width(2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg.export(str(out_path), format="wav")
    return True


# ----------------------------- speaking overlay -----------------------------
# While audio plays, an animated neon particle orb floats at the top-center of
# the screen with the session tag underneath, so you know WHICH session is
# talking even when you are in the browser. Implemented in overlay.py as a Win32
# layered window (real per-pixel alpha via UpdateLayeredWindow, click-through,
# never steals focus) — see that file for why tkinter's color-key isn't enough.
# Runs as a detached pythonw process that exits when we close its stdin pipe.
# Disable with --no-overlay or TTS_NO_OVERLAY=1.
#
# It is a separate process, not an import, which is what keeps it optional: on a
# platform without the Win32 layered-window path the overlay simply fails to
# start and speech carries on unaffected.

_OVERLAY_PY = THIS_DIR / "overlay.py"


def derive_overlay_label(text: str) -> str:
    """Best-effort session tag from the utterance itself. The voice skill
    mandates opening every line with a 2-4 word session tag, e.g.
    'Billing api, retry logic: ...' — the part before the first colon is the
    tag. Fall back to the first few words."""
    # Strip ANY bracketed performance tag (the skill uses more than the
    # official list, e.g. [calm], [serious]) plus break markers.
    t = re.sub(r"\[[a-zA-Z ]{2,24}\]", " ", text)
    t = re.sub(r"<break[^>]*>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    head = t.split(":", 1)[0].strip(" .,!¡¿?")
    if 0 < len(head) <= 40 and head != t:
        return head
    return " ".join(t.split()[:4])[:40] or "Claude"


# Frequency split for the orb. Speech lives mostly in the low/mid bands; the
# high band is what carries consonants and sibilance, which is why driving a
# separate visual off it makes the motion read as "locked to the voice" rather
# than "reacting to volume".
BANDS_HZ = ((60.0, 350.0), (350.0, 2200.0), (2200.0, 9000.0))
ANALYSIS_HOP = 1.0 / 120.0     # twice the frame rate, so every frame interpolates

# Envelope shaping, in milliseconds. Tunable without editing code so the feel
# can be tasted rather than argued about.
#   SMOOTH  — symmetric de-jitter before shaping. Higher = calmer.
#   ATTACK  — how gradually the orb swells INTO a syllable.
#   RELEASE — how gradually it settles after one. Longer than attack is what
#             makes the motion feel natural rather than mechanical.
ORB_SMOOTH_MS = float(os.environ.get("TTS_ORB_SMOOTH", 30))
ORB_ATTACK_MS = float(os.environ.get("TTS_ORB_ATTACK", 45))
ORB_RELEASE_MS = float(os.environ.get("TTS_ORB_RELEASE", 140))


def _smooth_zero_phase(row, width_ms: float, hop: float):
    """Symmetric moving average, applied forward then backward so it removes
    jitter without shifting anything in time."""
    import numpy as np
    taps = max(1, int(round(width_ms / 1000.0 / hop)))
    if taps < 2:
        return row
    k = np.hanning(taps + 2)[1:-1]
    k /= k.sum()
    pad = np.pad(row, taps, mode="edge")
    fwd = np.convolve(pad, k, mode="same")
    back = np.convolve(fwd[::-1], k, mode="same")[::-1]
    return back[taps:-taps]


def _shape_attack_release(row, attack_ms: float, release_ms: float, hop: float):
    """Give the envelope attack/release character without introducing lag.

    Smoothing with an asymmetric kernel — a short rise and a long tail — is what
    produces the "quick in, gentle out" motion that reads as natural. The catch
    is that an asymmetric kernel drags the signal along with it by its own
    centroid, so the result is shifted late; that shift is computed and undone
    here (with sub-frame interpolation), leaving the shape but not the delay.

    The obvious alternative, a classic attack/release follower run forward and
    backward, was tried and rejected: whether the two passes are combined with
    max or mean, they fill in the gaps between syllables and flatten the whole
    envelope (measured: dynamic range collapsed to 0.09-0.61 and even a clear
    pause in the speech stopped being visible).
    """
    import numpy as np
    n_att = max(1, int(round(attack_ms / 1000.0 / hop)))
    n_rel = max(1, int(round(release_ms / 1000.0 / hop)))
    rise = np.hanning(2 * n_att + 1)[:n_att + 1]
    tail = np.exp(-np.arange(1, n_rel + 1) / (n_rel / 2.5))
    k = np.concatenate([rise, tail])
    k /= k.sum()

    delay = float((np.arange(len(k)) * k).sum())     # kernel centroid
    pad = np.pad(row, len(k), mode="edge")
    y = np.convolve(pad, k, mode="full")[len(k):len(k) + len(row) + len(k)]

    i0 = int(np.floor(delay))
    frac = delay - i0
    idx = np.arange(len(row)) + i0
    a = np.take(y, np.clip(idx, 0, len(y) - 1))
    b = np.take(y, np.clip(idx + 1, 0, len(y) - 1))
    return a * (1.0 - frac) + b * frac


def analyze_pcm(data, sr: int, hop: float = ANALYSIS_HOP, ref: float | None = None):
    """Per-band envelopes for a block of mono float samples.

    Split out so the streaming path can run the *same* analysis as the
    file path instead of a cruder per-chunk approximation: three bands, dB
    scale, shared normalisation, then zero-phase smoothing and attack/release
    shaping. Returns (bands, ref_used) — the caller passes `ref` back in on the
    next block so the normalisation stays consistent across a stream.
    """
    import numpy as np
    n = max(1, int(sr * hop))
    win = max(n * 2, 256)
    n_frames = len(data) // n
    if n_frames < 2:
        return None, ref

    idx = np.arange(n_frames) * n
    starts = np.clip(idx - win // 2, 0, max(0, len(data) - win))
    frames = np.stack([data[s:s + win] for s in starts])
    if frames.shape[1] < win:
        frames = np.pad(frames, ((0, 0), (0, win - frames.shape[1])))
    spec = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1.0 / sr)

    raw = []
    for lo, hi in BANDS_HZ:
        sel = (freqs >= lo) & (freqs < hi)
        raw.append(np.sqrt(spec[:, sel].sum(axis=1)) if sel.any() else np.zeros(n_frames))
    raw = np.array(raw)

    # Pitch, as a fourth channel. Estimated by autocorrelation rather than by
    # picking the loudest FFT bin: at this hop the spectrum's resolution is
    # ~60 Hz, which cannot tell a low male voice from a high one, and the
    # loudest bin is usually a formant anyway rather than the fundamental.
    # Autocorrelation looks for the period the waveform actually repeats at.
    pit_win = 1024
    pit_starts = np.clip(idx - pit_win // 2, 0, max(0, len(data) - pit_win))
    pf = np.stack([data[s:s + pit_win] for s in pit_starts])
    if pf.shape[1] < pit_win:
        pf = np.pad(pf, ((0, 0), (0, pit_win - pf.shape[1])))
    pf = pf - pf.mean(axis=1, keepdims=True)
    ac = np.fft.irfft(np.abs(np.fft.rfft(pf * np.hanning(pit_win), axis=1)) ** 2, axis=1)
    lo_lag, hi_lag = int(sr / 400), int(sr / 70)       # 70..400 Hz of voice
    hi_lag = min(hi_lag, pit_win // 2)
    seg = ac[:, lo_lag:hi_lag]
    best = seg.argmax(axis=1) + lo_lag
    strength = seg.max(axis=1) / np.maximum(ac[:, 0], 1e-9)
    f0 = sr / np.maximum(best, 1)
    # Where the frame is unvoiced the peak means nothing, so hold mid-range.
    pitch = np.where(strength > 0.30, np.clip((f0 - 80.0) / 220.0, 0, 1), 0.42)

    block_ref = float(np.percentile(raw.max(axis=0), 97))
    # Track the loudest thing heard so far rather than renormalising every
    # block: per-block normalisation would make a quiet passage bloom to full
    # scale just because nothing louder happened to be in that block.
    ref_used = block_ref if ref is None else max(ref, block_ref)
    if ref_used <= 1e-9:
        return None, ref_used

    db = 20.0 * np.log10(np.maximum(raw / ref_used, 1e-4))
    lvl = np.clip((db + 42.0) / 42.0, 0.0, 1.0)
    out = []
    for row in lvl:
        row = _smooth_zero_phase(row, ORB_SMOOTH_MS, hop)
        row = _shape_attack_release(row, ORB_ATTACK_MS, ORB_RELEASE_MS, hop)
        out.append(np.clip(row, 0.0, 1.0))
    # Pitch is smoothed but NOT attack/release shaped: it is a position, not an
    # energy, so an envelope follower would drag it toward the last loud note.
    out.append(np.clip(_smooth_zero_phase(pitch, 90.0, hop), 0.0, 1.0))
    return np.array(out), ref_used


def analyze_audio(path: Path, hop: float = ANALYSIS_HOP):
    """Per-band envelopes plus pitch for a rendered clip, for the overlay.

    A thin wrapper over `analyze_pcm` on purpose: this used to carry its own
    copy of the analysis, and the copies drifted the moment one of them gained
    a channel — the streaming path started emitting pitch while this one kept
    returning three rows. One implementation, two callers.

    Returns (channels, hop) where channels is [low, mid, high, pitch], each a
    0..1 sequence, or None if the clip can't be analyzed. Analysis never blocks
    playback: on failure the visual simply idles.
    """
    try:
        import numpy as np
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception as e:
        print(f"[overlay] analysis unavailable: {e}", file=sys.stderr)
        return None
    try:
        bands, _ = analyze_pcm(data, sr, hop)
        if bands is None:
            return None
        return [[round(float(v), 3) for v in row] for row in bands], hop
    except Exception as e:
        print(f"[overlay] analysis failed: {e}", file=sys.stderr)
        return None


class OverlayHandle:
    """Controls the spawned orb overlay. Every method is best-effort: the
    overlay is decoration, so a failure here must never affect playback."""

    def __init__(self, proc) -> None:
        self._proc = proc

    def send(self, line: str) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.write((line + "\n").encode())
                self._proc.stdin.flush()
        except Exception:
            pass

    def send_words(self, words, delay: float = 0.0) -> None:
        """Word timings for the live caption, on the same clock as the audio."""
        if not words:
            return
        payload = "|".join(f"{s:.3f},{e:.3f},{w}" for s, e, w in words
                           if w and "|" not in w)
        self.send(f"WORDS {delay:.4f} {payload}")

    def send_bands(self, bands, hop: float, delay: float = 0.0) -> None:
        """Hand the overlay the per-band envelopes plus how long until the first
        sample is actually audible, so it can line the animation up exactly."""
        if not bands:
            return
        payload = "|".join(",".join(str(v) for v in row) for row in bands)
        self.send(f"ENV3 {hop:.6f} {delay:.4f} {payload}")

    def stop(self) -> None:
        # Closing stdin signals the overlay to fade out and exit, even through
        # the venv launcher indirection; terminate() alone only kills the
        # launcher process, not the python that actually owns the window.
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=1.0)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass


class _NullOverlay:
    def send(self, line: str) -> None: ...
    def send_words(self, words, delay: float = 0.0) -> None: ...
    def send_words(self, words, delay: float = 0.0) -> None:
        """Word timings for the live caption, on the same clock as the audio."""
        if not words:
            return
        payload = "|".join(f"{s:.3f},{e:.3f},{w}" for s, e, w in words
                           if w and "|" not in w)
        self.send(f"WORDS {delay:.4f} {payload}")

    def send_bands(self, bands, hop: float, delay: float = 0.0) -> None: ...
    def stop(self) -> None: ...


def start_overlay(label: str, visual: str | None = None):
    """Show the speaking visual; returns a handle with send/stop. Never fatal."""
    if os.environ.get("TTS_NO_OVERLAY") == "1" or not _OVERLAY_PY.exists():
        return _NullOverlay()
    import subprocess
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    visual = visual or os.environ.get("TTS_VOICE_VISUAL", "orb")
    try:
        proc = subprocess.Popen(
            [str(pyw if pyw.exists() else exe), str(_OVERLAY_PY), label, visual],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"[overlay] failed to start: {e}", file=sys.stderr)
        return _NullOverlay()
    return OverlayHandle(proc)


# ----------------------------- voice hub -----------------------------------
# The hub is a separate persistent panel (bottom-right) listing the speaking
# session, the queue behind it and recent speakers, each row clickable to jump
# to that session. It complements the overlay above: the overlay says who is
# talking right now, the hub says who is waiting and lets you go there.

def _launch_hub() -> bool:
    """Start hub.py detached. True if the process started."""
    import subprocess
    hub_script = THIS_DIR / "hub.py"
    if not hub_script.exists():
        return False
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    try:
        subprocess.Popen(
            [str(pyw if pyw.exists() else exe), str(hub_script)],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return True
    except Exception as e:
        print(f"[hub] launch failed: {e}", file=sys.stderr)
        return False


def ensure_hub_running() -> bool:
    """True if the hub is (now) running. A dead lock is cleaned and relaunched."""
    try:
        import hub_events
        lock = hub_events.HUB_LOCK
        if lock.exists():
            try:
                import psutil
                if psutil.pid_exists(int(lock.read_text().strip())):
                    return True
                lock.unlink(missing_ok=True)
            except Exception:
                pass
        return _launch_hub()
    except Exception:
        return False


class _HubReporter:
    """Emits queued/speaking/done for one utterance. No-op when disabled, so
    call sites stay free of conditionals."""

    def __init__(self, enabled: bool, label: str, preset: str, text: str) -> None:
        self.enabled = enabled
        self.label, self.preset, self.text = label, preset, text
        self.nav = self.identity = None
        if enabled:
            try:
                import hub_events
                self.nav = hub_events.capture_nav()
                self.identity = hub_events.capture_identity()
            except Exception:
                self.enabled = False

    def emit(self, state: str) -> None:
        if not self.enabled:
            return
        try:
            import hub_events
            hub_events.write_event(state, self.label, preset=self.preset,
                                   text_head=self.text, nav=self.nav,
                                   identity=self.identity)
        except Exception:
            pass  # the hub is never allowed to break speaking


# ----------------------------- helpers -------------------------------------

# ----------------------------- asking --------------------------------------
# Playback serialization now lives in turn.py, which ranks utterances instead of
# handing the queue to whichever process happened to call os.open first.

ASK_NO_ANSWER = 3
_INBOX_POLL_S = 0.2


def _await_answer(args, expect_lang: str | None) -> str:
    """Get Alex's spoken answer, from the assistant if it is up, else the mic.

    Two processes cannot record from one device sanely, so whoever owns the
    microphone decides how this works. When assistant.py is running it owns it:
    we publish the question and wait for the answer to arrive on the inbox.
    When it is not, we open the mic ourselves.

    The fallback is the point. Listening degrades the way speech already does
    when the hub is missing: worse, but never absent.
    """
    import stt
    try:
        import assistant
        import inbox
        assistant_up = assistant.is_running()
    except Exception:
        assistant_up = False

    if not assistant_up:
        return stt.listen_once(timeout=args.ask_timeout,
                               silence_tail=args.ask_tail,
                               expect_lang=expect_lang)

    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "")
    inbox.open_question(session_id, getattr(args, "label", "") or "", "")
    try:
        deadline = time.monotonic() + args.ask_timeout
        while time.monotonic() < deadline:
            msg = inbox.take(session_id)
            if msg and msg.get("text", "").strip():
                return msg["text"]
            time.sleep(_INBOX_POLL_S)
        raise stt.SttError("nadie contesto (via asistente)")
    finally:
        inbox.close_question(session_id)


def listen_for_answer(hub: "_HubReporter", overlay, args,
                      expect_lang: str | None = None) -> int:
    """Capture a spoken answer and print the transcript to stdout.

    That stdout is the return value of the Bash call the agent made, which is
    the entire point of --ask: the answer comes back as tool output and the
    session continues without anyone walking to a keyboard.

    The caller still holds its turn while this runs, so no other session can
    start talking over the answer. Returns an exit code: 0 with a transcript on
    stdout, ASK_NO_ANSWER if nobody answered. Silence is never smoothed over
    into a plausible answer — a session that asked needs to learn it was not
    answered, and Scribe will happily invent a caption for an empty room.
    """
    overlay.stop()  # speaking is over; what is live now is the microphone
    hub.emit("listening")
    try:
        answer = _await_answer(args, expect_lang).strip()
    except Exception as e:
        print(f"[ask] no answer: {e}", file=sys.stderr)
        return ASK_NO_ANSWER
    if not answer:
        print("[ask] no answer: empty transcript", file=sys.stderr)
        return ASK_NO_ANSWER
    # Force UTF-8 on the way out. Windows hands stdout a cp1252 codec by
    # default, which turned a real answer into 'mant?n ... olv?date': the
    # agent would act on a mangled version of what was actually said, and
    # Spanish answers carry accents in almost every sentence.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(answer)
    return 0


def parse_headers(text: str) -> tuple[str, str | None, str | None]:
    """Strip leading `#preset=...` / `#lang=...` header lines (never spoken).

    The voice skill authors utterances with an optional header as the first
    line(s), e.g. `#preset=calm` then the utterance. say.py takes those as flags,
    so without this the literal header would be spoken aloud. Returns
    (text_without_headers, preset_or_None, lang_or_None).
    """
    preset: str | None = None
    lang: str | None = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln.startswith("#"):
            break
        tokens = ln[1:].replace(",", " ").split()
        kvs: list[tuple[str, str]] = []
        ok = bool(tokens)
        for tok in tokens:
            tok = tok.lstrip("#")
            if "=" not in tok:
                ok = False
                break
            k, v = tok.split("=", 1)
            if k.strip().lower() not in ("preset", "lang"):
                ok = False
                break
            kvs.append((k.strip().lower(), v.strip()))
        if not ok:
            break
        for k, v in kvs:
            if k == "preset":
                preset = v
            else:
                lang = v
        i += 1
    return "\n".join(lines[i:]).strip(), preset, lang


def detect_lang(text: str) -> str:
    if re.search(r"[ñáéíóúü¿¡]|\b(hola|de|que|para|por|los|las|con|esto|ahora|sí|no|también|esto)\b",
                 text, flags=re.IGNORECASE):
        return "es"
    return "en"


def get_focusrite_index() -> int | None:
    try:
        import sounddevice as sd
    except Exception:
        return None
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_output_channels", 0) > 0 and "Focusrite" in d.get("name", ""):
            return i
    return None


def play_audio(path: Path, device: int | None = None,
               on_start: Callable[[float], None] | None = None) -> None:
    """Play any audio file via sounddevice on the chosen device.
    Decodes mp3 via pydub if needed; falls back to PowerShell WPF MediaPlayer
    if sounddevice can't decode. Never silently exits.

    `on_start(latency)` fires the moment the output stream starts, with the
    device's remaining buffer latency in seconds — i.e. sound becomes audible
    `latency` seconds later. That's what lets the orb line up with the voice.
    """
    if not path.exists():
        # Maybe the path is .wav but only .mp3 exists.
        alt = path.with_suffix(".mp3")
        if alt.exists():
            path = alt
        else:
            print(f"[play] file missing: {path}", file=sys.stderr)
            return

    suffix = path.suffix.lower()
    try:
        import sounddevice as sd
        import numpy as np
        if suffix == ".mp3":
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(str(path))
            arr = np.array(seg.get_array_of_samples()).astype(np.float32) / (2 ** (8 * seg.sample_width - 1))
            if seg.channels == 2:
                arr = arr.reshape(-1, 2)
            sr = seg.frame_rate
        else:
            import soundfile as sf
            arr, sr = sf.read(str(path), dtype="float32")
        # Pad ~0.5s of trailing silence so the device/driver doesn't clip the
        # last word(s): sounddevice's blocking play can stop the stream before
        # the final output buffer drains. The clip then lands in the silence.
        _tail = int(sr * 0.5)
        if arr.ndim == 1:
            arr = np.concatenate([arr, np.zeros(_tail, dtype=arr.dtype)])
        else:
            arr = np.concatenate([arr, np.zeros((_tail, arr.shape[1]), dtype=arr.dtype)], axis=0)
        dev = device if device is not None else get_focusrite_index()
        channels = 1 if arr.ndim == 1 else arr.shape[1]

        # Drive the stream explicitly instead of sd.play() so we know the exact
        # instant sound reaches the speaker. sd.play() hides the stream, and the
        # gap between "call play" and "first sample audible" is large and very
        # device-dependent: on the Focusrite it measured ~360 ms (176 ms to open
        # the stream + 182 ms of buffered output latency). Anything synced to
        # the call instead of the stream therefore runs a third of a second
        # early — which is exactly what made the orb look out of step.
        def _stream_play(target_dev) -> None:
            st = sd.OutputStream(samplerate=sr, channels=channels, device=target_dev)
            st.start()
            try:
                if on_start is not None:
                    # Everything already written is still queued in the device
                    # buffer, so audio becomes audible st.latency from now.
                    on_start(float(st.latency))
                st.write(arr)
            finally:
                st.stop()
                st.close()

        try:
            _stream_play(dev)
        except Exception as e:
            print(f"[play] device {dev} failed ({e}); using default", file=sys.stderr)
            try:
                _stream_play(None)
            except Exception as e2:
                print(f"[play] default device failed ({e2}); using sd.play", file=sys.stderr)
                if on_start is not None:
                    on_start(0.0)
                sd.play(arr, samplerate=sr, blocking=True)
        return
    except Exception as e:
        print(f"[play] sounddevice path failed: {e}; using the OS player", file=sys.stderr)

    if suffix == ".mp3" and platform_native.IS_WINDOWS:
        # SoundPlayer only handles WAV, so MP3 needs MediaPlayer and an explicit
        # wait for the duration to become known.
        import subprocess
        ps = f"""
Add-Type -AssemblyName PresentationCore
$mp = New-Object System.Windows.Media.MediaPlayer
$mp.Open([uri]'{path.as_posix()}')
$mp.Play()
Start-Sleep -Milliseconds 200
while ($mp.NaturalDuration.HasTimeSpan -eq $false) {{ Start-Sleep -Milliseconds 100 }}
$dur = $mp.NaturalDuration.TimeSpan.TotalMilliseconds
Start-Sleep -Milliseconds ($dur + 200)
$mp.Close()
"""
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
        return

    if not platform_native.play_file(path):
        print(f"[play] no OS player available on {sys.platform}; audio not heard",
              file=sys.stderr)


# ----------------------------- backends ------------------------------------

class Backend:
    name: str
    def available(self) -> bool: ...
    def synth(self, text: str, lang: str, out: Path, **kw) -> Path:
        """Synthesize `text` and write audio to `out` (or sibling path).
        Returns the actual file written (may be .mp3 if WAV conversion failed).
        """
        raise NotImplementedError


SPANISH_VARIANT = os.environ.get("TTS_ES_VARIANT", "es-419")


def _spanish_g2p():
    """Kokoro's Spanish phonemizer, in Latin American Spanish.

    Kokoro maps its 'e' language code to plain espeak `es`, which is peninsular:
    it cecea, so "voz" comes out /boθ/ and "gracias" /gɾaθjas/. Alex is Latin
    American and the requested voice is es-MX, so `es-419` (seseo) is the accent
    that reads as native rather than as a foreigner doing Spain.

    Returns None if no variant loads, leaving Kokoro's default in place: a
    slightly wrong accent is worth far more than silence.

    `es-419` is espeak-ng's current tag for Latin American Spanish; older builds
    only shipped `es-la`. Try the configured one first, then that legacy tag,
    before giving up.
    """
    from misaki import espeak
    variants = [SPANISH_VARIANT]
    if "es-la" not in variants:
        variants.append("es-la")   # legacy tag on older espeak-ng builds
    last_err = None
    for variant in variants:
        try:
            return espeak.EspeakG2P(language=variant)
        except Exception as e:
            last_err = e
    print(f"[kokoro] no Latin American Spanish variant loaded "
          f"(tried {variants}: {last_err}); using espeak's default", file=sys.stderr)
    return None


class KokoroBackend(Backend):
    name = "kokoro"
    _pipelines: dict = {}  # keyed by lang_code

    def available(self) -> bool:
        try:
            import kokoro  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, text: str, lang: str, out: Path, *, voice: str | None = None, **kw) -> Path:
        import os
        if platform_native.IS_WINDOWS:
            # Point phonemizer at the standard installer location. Elsewhere,
            # leave the environment alone: espeak-ng lands on the library path
            # and phonemizer finds it on its own.
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY",
                                  r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
            os.environ.setdefault("PHONEMIZER_ESPEAK_PATH",
                                  r"C:\Program Files\eSpeak NG\espeak-ng.exe")
        import numpy as np
        import soundfile as sf
        from kokoro import KPipeline

        lang_code = "e" if lang == "es" else "a"  # 'e'=Spanish, 'a'=American English
        if lang_code not in self._pipelines:
            pipeline = KPipeline(lang_code=lang_code)
            if lang_code == "e":
                pipeline.g2p = _spanish_g2p() or pipeline.g2p
            self.__class__._pipelines[lang_code] = pipeline
        pipeline = self._pipelines[lang_code]

        # Ignore Edge-style voice names (contain "Neural") — use Kokoro defaults.
        if voice is None or "Neural" in str(voice):
            voice = "em_alex" if lang == "es" else "af_heart"

        chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
        if not chunks:
            raise RuntimeError("Kokoro returned no audio chunks")
        audio = np.concatenate(chunks)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), audio, 24000)
        return out


class SupertonicBackend(Backend):
    """Supertonic 3 — on-device ONNX, 31 languages, expression tags (<laugh>, <sigh>, <breath>).
    99M params, ~200MB first-run download, no API key, no GPU required.
    """
    name = "supertonic"
    _tts = None
    _style = None

    def available(self) -> bool:
        try:
            import supertonic  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, text: str, lang: str, out: Path, *, voice: str | None = None, **kw) -> Path:
        import soundfile as sf
        from supertonic import TTS

        if self.__class__._tts is None:
            self.__class__._tts = TTS(auto_download=True)
            # Voice style name is opaque; "M1" is a sensible default male voice.
            # Future: expose --voice supertonic:M2 etc once we know the catalog.
            self.__class__._style = self.__class__._tts.get_voice_style(voice_name="M1")

        # supertonic v3 supports 31 ISO language codes; fall back to en if unknown.
        st_lang = lang if lang in ("es", "en", "ja", "ar", "bg", "cs", "da", "de", "el",
                                    "et", "fi", "hr", "hu", "id", "it", "lt", "lv", "nl",
                                    "pl", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
                                    "fr", "ko", "pt", "zh") else "en"

        wav, _dur = self.__class__._tts.synthesize(
            text=text,
            lang=st_lang,
            voice_style=self.__class__._style,
        )
        # supertonic returns shape (1, N) channels-first at the model's native
        # sample rate (44100 in v3); squeeze to 1D mono and use the model's
        # actual sample rate so the WAV header matches the data — writing the
        # wrong rate caused 0.54x playback ("monster voice") at 24000.
        import numpy as np
        wav_mono = np.asarray(wav).squeeze()
        sample_rate = int(getattr(self.__class__._tts, "sample_rate", 44100))
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), wav_mono, sample_rate, subtype="PCM_16")
        return out


class EdgeBackend(Backend):
    name = "edge"

    def available(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, text: str, lang: str, out: Path, *,
              voice: str | None = None, rate: str = "+0%", pitch: str = "+0Hz",
              **kw) -> Path:
        import asyncio
        import edge_tts

        if voice is None:
            voice = V_JORGE if lang == "es" else V_GUY

        mp3_path = out.with_suffix(".mp3")
        mp3_path.parent.mkdir(parents=True, exist_ok=True)

        async def _run() -> None:
            comm = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
            await comm.save(str(mp3_path))

        asyncio.run(_run())
        return mp3_path  # post-processing handles MP3 -> WAV.


class F5Backend(Backend):
    name = "f5"
    _model = None

    def available(self) -> bool:
        # Gate behind explicit opt-in so we don't trigger a ~1.3 GB first-run
        # download just by importing. The user can flip TTS_F5_ENABLED=1 once
        # they've decided to pay the download cost.
        if os.environ.get("TTS_F5_ENABLED") != "1":
            return False
        try:
            import f5_tts  # noqa: F401
            return True
        except Exception:
            return False

    def synth(self, text: str, lang: str, out: Path, *, ref_wav: str | None = None,
              **kw) -> Path:
        from f5_tts.api import F5TTS
        if self._model is None:
            self._model = F5TTS()
        ref = ref_wav or os.environ.get("TTS_VOICE_REF") or (
            str(DEFAULT_REF_WAV) if DEFAULT_REF_WAV.exists() else None
        )
        if not ref:
            raise RuntimeError(
                "F5-TTS requires a reference WAV. Run `say.py --seed-ref` to create one, "
                "or set TTS_VOICE_REF."
            )
        wav, sr, _ = self._model.infer(
            gen_text=text,
            ref_file=ref,
            ref_text="",
            show_info=lambda *a, **k: None,
            progress=None,
        )
        import soundfile as sf
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), wav, sr)
        return out


class NativeBackend(Backend):
    """The OS voice: SAPI on Windows, `say` on macOS.

    This is the end of the fallback chain. It needs no network, no model and no
    API key, so when everything above it fails this is what still speaks.
    """

    name = "native"

    def available(self) -> bool:
        return platform_native.native_tts_available()

    def synth(self, text: str, lang: str, out: Path, *, rate_int: int = 3, **kw) -> Path:
        # Callers may pass Edge-style rate strings in `rate`; we ignore those and
        # use `rate_int`, which each platform maps onto its own scale.
        return platform_native.native_tts_synth(text, lang, out, rate_int=rate_int)


# Kept so existing configs and scripts that name "sapi" keep working.
SapiBackend = NativeBackend


def words_from_alignment(text: str, alignment) -> list | None:
    """Turn per-character timings into (start, end, word) triples.

    The text we send carries things the listener never hears — performance tags
    like `[excited]` and `<break time="0.4s" />` — and the alignment covers
    those characters too, because they were part of the input. Emitting them
    would put literal markup in the caption, so any character inside brackets or
    angle brackets is skipped while its time is still consumed. That keeps the
    remaining words on the model's own clock instead of shifting them earlier.
    """
    if not alignment:
        return None
    try:
        chars = alignment["characters"]
        starts = alignment["character_start_times_seconds"]
        ends = alignment["character_end_times_seconds"]
    except (KeyError, TypeError):
        return None
    if not chars or len(chars) != len(starts) or len(chars) != len(ends):
        return None

    words, buf, w_start, w_end = [], [], None, None
    depth_sq = depth_ang = 0
    for ch, t0, t1 in zip(chars, starts, ends):
        if ch == "[":
            depth_sq += 1
            continue
        if ch == "]":
            depth_sq = max(0, depth_sq - 1)
            continue
        if ch == "<":
            depth_ang += 1
            continue
        if ch == ">":
            depth_ang = max(0, depth_ang - 1)
            continue
        if depth_sq or depth_ang:
            continue
        if ch.isspace():
            if buf:
                words.append((w_start, w_end, "".join(buf)))
                buf, w_start, w_end = [], None, None
            continue
        buf.append(ch)
        w_start = t0 if w_start is None else min(w_start, t0)
        w_end = t1 if w_end is None else max(w_end, t1)
    if buf:
        words.append((w_start, w_end, "".join(buf)))
    return words or None


class ElevenLabsBackend(Backend):
    name = "elevenlabs"

    def available(self) -> bool:
        return bool(_EL_KEY)

    def synth(self, text: str, lang: str, out: Path, *, preset: str = "neutral", voice: str | None = None, **kw) -> Path:
        import json
        import urllib.request
        import numpy as np
        import wave

        # Resolve ElevenLabs voice: check alias table first, then env default.
        # Never use Edge-TTS voice names (e.g. "es-MX-JorgeNeural") as an EL voice_id.
        voice_id = _EL_VOICE_ALIASES.get((voice or "").lower()) or _EL_VOICE or "21m00Tcm4TlvDq8ikWAM"
        # The `with-timestamps` variant returns the audio together with the
        # start and end time of every CHARACTER. That is what makes a live
        # caption possible without guessing: word timings come from the model
        # that produced the speech, not from estimating where words probably
        # fall inside the envelope.
        base = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        url = f"{base}/with-timestamps?output_format=pcm_24000"
        raw = {**_EL_SETTINGS.get(preset, _EL_SETTINGS["neutral"]), "use_speaker_boost": True}
        speed = raw.pop("speed", None)  # speed is top-level, not inside voice_settings
        payload: dict = {"text": text, "model_id": _EL_MODEL, "voice_settings": raw}
        if speed is not None and speed != 1.0:
            payload["speed"] = speed
        body = json.dumps(payload).encode()

        def _post(u: str, accept: str):
            req = urllib.request.Request(u, data=body, method="POST", headers={
                "xi-api-key": _EL_KEY,
                "Content-Type": "application/json",
                "Accept": accept,
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()

        self.words = None
        try:
            import base64
            data = json.loads(_post(url, "application/json").decode())
            pcm = base64.b64decode(data["audio_base64"])
            self.words = words_from_alignment(text, data.get("alignment")
                                              or data.get("normalized_alignment"))
        except Exception as e:
            # Captions are a bonus; speech is not. Fall back to the plain
            # endpoint so a timestamp failure never costs us the utterance.
            print(f"[caption] timestamps unavailable ({e}); speaking without", file=sys.stderr)
            pcm = _post(f"{base}?output_format=pcm_24000", "audio/pcm")
        if not pcm:
            raise RuntimeError("ElevenLabs returned empty audio")
        audio = np.frombuffer(pcm, dtype=np.int16)
        wav_path = out.with_suffix(".wav")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(audio.tobytes())
        return wav_path


def stream_elevenlabs(text: str, *, preset: str = "neutral", voice: str | None = None,
                      device: int | None = None, overlay=None) -> bool:
    """Low-latency path: stream PCM from ElevenLabs and play it as it arrives.

    Bypasses the file + post-processing pipeline so the first audio reaches the
    speaker as soon as the model emits it (time-to-first-sound, not
    time-to-full-clip). No post-FX (it needs the whole clip), so output is the
    raw model audio. Returns True on success, False to let the caller fall back
    to the normal synth-then-play path.
    """
    if not _EL_KEY:
        return False
    import json
    import urllib.request
    import numpy as np
    import sounddevice as sd

    voice_id = _EL_VOICE_ALIASES.get((voice or "").lower()) or _EL_VOICE or "21m00Tcm4TlvDq8ikWAM"
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
           f"?output_format=pcm_24000")
    raw = {**_EL_SETTINGS.get(preset, _EL_SETTINGS["neutral"]), "use_speaker_boost": True}
    speed = raw.pop("speed", None)
    payload: dict = {"text": text, "model_id": _EL_MODEL, "voice_settings": raw}
    if speed is not None and speed != 1.0:
        payload["speed"] = speed
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"xi-api-key": _EL_KEY,
                                          "Content-Type": "application/json",
                                          "Accept": "audio/pcm"})
    dev = device if device is not None else get_focusrite_index()
    try:
        resp = urllib.request.urlopen(req, timeout=25)
    except Exception as e:
        print(f"[stream] request failed: {e}", file=sys.stderr)
        return False

    SR = 24000
    SEG_S = 0.22            # analysis block; also how far analysis trails writing
    OVERLAP_S = 0.25        # context carried back so shaping has no edge seam
    # Order matters here. Writing first and analysing after cannot work: once
    # the device buffer is full `stream.write` blocks, so the writer runs in
    # lockstep with the speaker and a block's envelope is only ready when that
    # block is already being heard (measured: 134 ms LATE, median). Pre-buffering
    # does not rescue it either, because the head start collapses to whatever
    # the device buffer holds. So each block is analysed BEFORE it is handed to
    # the device: the envelope then leaves ahead of everything still queued, and
    # the only cost is one block of delay before the first sound.

    leftover = b""          # carry an odd trailing byte between chunks (int16 = 2 bytes/frame)
    started = False
    pending = np.zeros(0, np.float32)     # audio written but not yet analyzed
    history = np.zeros(0, np.float32)     # tail kept for overlap context
    frames_written = 0                    # position of `pending` in the stream
    ref_level = None
    try:
        stream = sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16", device=dev)
    except Exception as e:
        print(f"[stream] device {dev} unavailable ({e}); using default", file=sys.stderr)
        stream = sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16")
    latency = float(getattr(stream, "latency", 0.0) or 0.0)
    t_start = None                        # set when playback actually begins
    lead_debug: list = []

    def flush_analysis(force: bool = False) -> None:
        """Analyze the next block, send its envelope, then hand it to the device.

        Analysing before writing is what gives the streaming path the same
        quality as a complete clip: the block is still ours when its three-band
        envelope goes out, so the overlay receives it with the whole device
        queue as lead time, and the zero-phase shaping has real look-ahead
        instead of having to guess from the past.
        """
        nonlocal pending, history, frames_written, ref_level, t_start, started
        need = int(SR * SEG_S)
        if pending.size < need and not (force and pending.size > int(SR * 0.02)):
            return
        if t_start is None:
            stream.start()
            t_start = time.monotonic()

        if overlay is not None:
            ctx = int(SR * OVERLAP_S)
            block = np.concatenate([history[-ctx:], pending]) if history.size else pending
            lead = min(ctx, history.size)
            bands, ref_level = analyze_pcm(block, SR, ref=ref_level)
            if bands is not None:
                skip = int(round(lead / SR / ANALYSIS_HOP))
                emit = bands[:, skip:]
                if emit.shape[1] > 0:
                    # Nothing of this block has been queued yet, so it becomes
                    # audible only once everything already queued has drained.
                    audible_at = t_start + latency + frames_written / SR
                    raw = audible_at - time.monotonic()
                    lead_debug.append(raw)
                    overlay.send_bands(
                        [[round(float(v), 3) for v in row] for row in emit],
                        ANALYSIS_HOP, delay=max(0.0, raw))
            history = np.concatenate([history, pending])[-int(SR * 1.0):]

        stream.write((pending * 32768.0).astype(np.int16))
        started = True
        frames_written += pending.size
        pending = np.zeros(0, np.float32)

    try:
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf = leftover + chunk
            n = len(buf) - (len(buf) % 2)   # whole frames only
            leftover = buf[n:]
            if n:
                samples = np.frombuffer(buf[:n], dtype=np.int16)
                pending = np.concatenate([pending, samples / 32768.0])
                flush_analysis()
    finally:
        try:
            flush_analysis(force=True)       # the tail, which is short of a block
        except Exception as e:
            print(f"[stream] final block skipped: {e}", file=sys.stderr)
        if lead_debug and os.environ.get("TTS_STREAM_STATS") == "1":
            import statistics
            print(f"[stream] envelope lead: median {statistics.median(lead_debug)*1000:.0f} ms, "
                  f"min {min(lead_debug)*1000:.0f} ms over {len(lead_debug)} blocks",
                  file=sys.stderr)
        stream.stop()
        stream.close()
        resp.close()
    return started


# Order matters: first-available wins unless --backend forces.
# ElevenLabs first (inspiring voice, dynamics). Kokoro is the designated fallback
# when the key is absent or the network is not there: it is local, needs no key,
# and its em_alex/af_heart voices are the closest in character to the hosted one.
# Supertonic is lighter but flatter, so it sits behind Kokoro rather than ahead.
BACKENDS: list[Backend] = [ElevenLabsBackend(), KokoroBackend(), SupertonicBackend(), EdgeBackend(), F5Backend(), NativeBackend()]

# The OS voice used to be Windows-only and was called "sapi". Keep the old name
# working so existing scripts, env vars and muscle memory do not break.
BACKEND_ALIASES = {"sapi": "native", "say": "native", "os": "native"}


def fallback_chain(after: Backend) -> list[Backend]:
    """The backends still worth trying when `after` fails mid-synthesis.

    `available()` answers "is this installed", which is not the same question as
    "will this speak right now". A key expires, a network drops, a model refuses
    to load -- all of it surfaces at synth time, on the backend already chosen.
    When that happens the next voice should be the next one in the documented
    order. Jumping straight to the robotic OS voice while Kokoro sits installed
    one link down is a far bigger drop in quality than an expired key warrants,
    and Alex hears that drop without being told why.
    """
    try:
        start = BACKENDS.index(after) + 1
    except ValueError:
        start = 0
    return [b for b in BACKENDS[start:] if b.available()]


def synth_with_fallback(backend: Backend, text: str, lang: str, out: Path,
                        *, forced: bool, **synth_kw) -> tuple[Backend, Path]:
    """Synthesize with `backend`; on a runtime failure, walk the rest of the
    chain. Returns (backend_that_spoke, output_path).

    `available()` answers "is this installed", not "will this speak right now":
    an expired key or a dropped network only surfaces here, at synth time. When
    it does, the next voice heard should be the next one in the documented order
    rather than a jump straight to the robotic OS voice.

    `forced` is the one exception. When the caller asked for a specific
    `--backend`, substituting a different voice is worse than failing: they
    cannot tell it happened. So a forced backend re-raises instead of degrading,
    the same contract pick_backend() honours at selection time.
    """
    # ElevenLabs takes tags natively; everyone else gets them stripped.
    first_text = text if backend.name == "elevenlabs" else _strip_audio_tags(text)
    try:
        return backend, backend.synth(first_text, lang, out, **synth_kw)
    except Exception as e:
        if forced:
            raise
        last_err = e
        for alt in fallback_chain(backend):
            print(f"[say] {backend.name} failed: {last_err} -- falling back to {alt.name}",
                  file=sys.stderr)
            try:
                return alt, alt.synth(_strip_audio_tags(text), lang, out, **synth_kw)
            except Exception as alt_err:
                last_err = alt_err
        raise last_err


def pick_backend(force: str | None = None) -> Backend:
    force = BACKEND_ALIASES.get(force, force)
    if force:
        for b in BACKENDS:
            if b.name == force:
                # When forced, ignore opt-in gates so explicit choice always works.
                if force == "f5":
                    try:
                        import f5_tts  # noqa: F401
                        return b
                    except Exception as e:
                        raise RuntimeError(f"f5_tts import failed: {e}")
                if force == "elevenlabs":
                    if not bool(_EL_KEY):
                        raise RuntimeError("elevenlabs not available: no ELEVENLABS_API_KEY in env")
                    return b
                if not b.available():
                    raise RuntimeError(f"backend {force!r} not available")
                return b
        raise RuntimeError(f"unknown backend {force!r}")
    for b in BACKENDS:
        if b.available():
            return b
    raise RuntimeError("no TTS backend available")


# ----------------------------- preset application --------------------------

def resolve_preset(name: str | None, strict: bool = False) -> Preset:
    """The named preset, or the default.

    `strict` decides what an unknown name costs, and the two callers differ.
    A typo in `--preset` is a human at a prompt who can retype it, so failing
    loudly is right. A typo in a `#preset=` header is inside content that was
    meant to be spoken, written by an agent that will not see the error -- and
    exiting there means Alex simply never hears the line. Everything else in
    this project degrades rather than going silent; a misspelled adverb should
    not be the one exception.
    """
    name = (name or DEFAULT_PRESET).lower()
    if name in PRESETS:
        return PRESETS[name]
    if strict:
        raise SystemExit(f"unknown preset {name!r}; try one of: {', '.join(PRESETS)}")
    print(f"[say] unknown preset {name!r}; using {DEFAULT_PRESET}", file=sys.stderr)
    return PRESETS[DEFAULT_PRESET]


def apply_rate_delta(base_rate_str: str, delta: int) -> str:
    """Layer a -10..+10 integer delta on top of an edge-tts rate string.
    Each step is ~5%. Clamped to +/-50%."""
    base = int(base_rate_str.rstrip("%"))
    pct = max(-50, min(50, base + delta * 5))
    return f"+{pct}%" if pct >= 0 else f"{pct}%"


# ----------------------------- seed reference (F5) -------------------------

def seed_jorge_ref() -> Path:
    """Generate a ~12s Jorge reference WAV from Edge TTS. Run once to enable
    F5-TTS voice cloning of the same voice we already like."""
    import asyncio
    import edge_tts
    from pydub import AudioSegment

    REFS_DIR.mkdir(parents=True, exist_ok=True)
    seed_text = (
        "Esta es una grabación de referencia para el clon de voz. "
        "Voy a hablar en tono natural, con un ritmo cómodo, "
        "para que el modelo aprenda mi timbre, mi cadencia y mi forma de pronunciar. "
        "Cuando termines de escuchar esto, podrás clonar mi voz a tu gusto."
    )
    mp3 = REFS_DIR / "jorge_seed.mp3"

    async def _run() -> None:
        c = edge_tts.Communicate(seed_text, voice=V_JORGE, rate="+15%", pitch="+5Hz")
        await c.save(str(mp3))

    asyncio.run(_run())
    seg = AudioSegment.from_mp3(str(mp3))
    # F5-TTS expects 24 kHz mono WAV; that's what Edge ships, but convert
    # explicitly so the file format is unambiguous.
    seg = seg.set_frame_rate(24000).set_channels(1).set_sample_width(2)
    seg.export(str(DEFAULT_REF_WAV), format="wav")
    mp3.unlink(missing_ok=True)
    return DEFAULT_REF_WAV


# ----------------------------- main ----------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="*")
    p.add_argument("--lang", default=None)
    p.add_argument("--rate", type=int, default=int(os.environ.get("TTS_RATE", 0)),
                   help="-10..+10 delta on top of preset rate (5%% per step)")
    p.add_argument("--backend", default=os.environ.get("TTS_BACKEND"))
    _dev_env = os.environ.get("TTS_DEVICE")
    p.add_argument("--device", type=lambda x: int(x), default=int(_dev_env) if _dev_env else None)
    p.add_argument("--voice", default=None, help="override voice (e.g. es-MX-DaliaNeural)")
    p.add_argument("--pitch", default=None, help="pitch override e.g. +12Hz, -8Hz")
    p.add_argument("--edge-rate", default=None, help="raw Edge rate string e.g. +30%%")
    p.add_argument("--preset", default=None,
                   help=f"vibe preset: {', '.join(PRESETS)}")
    p.add_argument("--ref", default=None, help="reference WAV for F5-TTS voice cloning")
    p.add_argument("--out", default=None)
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--no-postfx", action="store_true",
                   help="skip post-processing (faster, less polished)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--seed-ref", action="store_true",
                   help="(re)generate F5-TTS reference WAV from Jorge and exit")
    p.add_argument("--raw", action="store_true",
                   help="skip text preprocessing (tickers, currency, breaks)")
    # ON by default: the non-streaming path builds the whole clip before playing,
    # so a long utterance blows the 25/30s request timeouts and silently degrades
    # to SAPI (measured 2026-07-19: a ~1400-char line failed with-timestamps AND
    # the plain endpoint, and came out in the Windows voice). Streaming starts
    # playing as the model emits, so length stops mattering. Cost: no post-FX and
    # no per-word captions; the orb still animates (it analyses blocks live).
    # Opt out per call with --no-stream, or globally with TTS_STREAM=0.
    p.add_argument("--stream", action=argparse.BooleanOptionalAction,
                   default=os.environ.get("TTS_STREAM", "1") != "0",
                   help="ElevenLabs low-latency: play PCM as it streams (no post-FX). "
                        "Default ON; use --no-stream for the post-processed path")
    p.add_argument("--label", default=None,
                   help="session tag shown in the on-screen overlay (default: derived from the utterance's opening tag)")
    p.add_argument("--no-hub", action="store_true",
                   help="don't report this utterance to the voice hub panel")
    p.add_argument("--visual", default=None, choices=("orb", "neuro"),
                   help="on-screen visual while speaking: orb (default), or neuro "
                        "(a neuron that fires on each syllable, from experiments/). "
                        "Also settable with TTS_VOICE_VISUAL")
    p.add_argument("--no-overlay", action="store_true",
                   help="don't show the on-screen speaking indicator")
    # --ask turns the loudspeaker into a conversation: speak a question, listen
    # hands-free, and print the transcript to stdout. The agent that ran this
    # through Bash receives the spoken answer as its tool output and carries on,
    # so answering no longer costs a trip to the keyboard.
    p.add_argument("--ask", action="store_true",
                   help="after speaking, listen for a spoken answer and print the "
                        "transcript to stdout (exit 3 if nobody answered)")
    p.add_argument("--ask-timeout", type=float, default=20.0,
                   help="seconds to wait for an answer to begin (default 20)")
    p.add_argument("--ask-tail", type=float, default=1.4,
                   help="seconds of silence that end the answer (default 1.4)")
    p.add_argument("--priority", default=None,
                   choices=tuple(turn.PRIORITIES),
                   help="how far this utterance jumps the queue: blocker > "
                        "question > outcome > ping. --ask implies question")
    args = p.parse_args()

    # Without playback we never reach the listening step, so --ask would exit 0
    # having asked nobody anything. Failing loudly beats an agent believing it
    # got an answer it never waited for.
    if args.ask and args.no_play:
        print("--ask needs playback: it speaks a question and then listens",
              file=sys.stderr)
        return 1

    if args.seed_ref:
        out = seed_jorge_ref()
        print(f"[seed] wrote {out}", file=sys.stderr)
        return 0

    if args.list:
        print("Available backends:")
        for b in BACKENDS:
            print(f"  {b.name:6} available={b.available()}")
        print()
        print("Presets:")
        for name, ps in PRESETS.items():
            print(f"  {name:9} es={ps.voice_es:24} en={ps.voice_en:30} rate={ps.rate} pitch={ps.pitch}")
        print()
        try:
            import sounddevice as sd
            print("Output devices:")
            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) > 0:
                    star = "  <-- Focusrite" if "Focusrite" in d["name"] else ""
                    print(f"  [{i:>2}] {d['name']}{star}")
        except Exception as e:
            print(f"sounddevice not available: {e}")
        return 0

    if args.list_presets:
        for name, ps in PRESETS.items():
            print(f"{name:9} es={ps.voice_es:24} en={ps.voice_en:30} rate={ps.rate} pitch={ps.pitch} warmth={ps.warmth}")
        return 0

    raw_text = " ".join(args.text).strip()
    raw_text, hdr_preset, hdr_lang = parse_headers(raw_text)
    if not raw_text:
        print("no text", file=sys.stderr)
        return 1
    lang = args.lang or hdr_lang or detect_lang(raw_text)
    text = raw_text if args.raw else preprocess_text(raw_text, lang)

    overlay_label = args.label or derive_overlay_label(raw_text)
    hub = _HubReporter(not args.no_hub and ensure_hub_running(),
                       overlay_label, args.preset or hdr_preset or DEFAULT_PRESET,
                       raw_text)

    # A question outranks a plain outcome by default: it is the one kind of
    # utterance that leaves a session blocked until it is answered.
    priority = args.priority or ("question" if args.ask else turn.DEFAULT_PRIORITY)
    turn.clear_legacy_lock()

    # A typo in --preset is a human who can retype it; a typo in a
    # #preset= header is inside something that was meant to be spoken.
    preset = (resolve_preset(args.preset, strict=True) if args.preset
              else resolve_preset(hdr_preset))
    voice = args.voice or (preset.voice_es if lang == "es" else preset.voice_en)
    # rate priority: explicit --edge-rate > preset + delta from --rate
    if args.edge_rate is not None:
        rate_str = args.edge_rate
    else:
        rate_str = apply_rate_delta(preset.rate, args.rate)
    pitch_str = args.pitch if args.pitch is not None else preset.pitch

    # Low-latency streaming path (ElevenLabs only). Plays as audio arrives,
    # skipping the file + post-FX pipeline. Falls through to the normal path on
    # failure so we never go silent.
    if args.stream and not args.no_play and _EL_KEY and (args.backend in (None, "elevenlabs")):
        print(f"[say] streaming via elevenlabs preset={preset.name} voice={voice} lang={lang}",
              file=sys.stderr)
        hub.emit("queued")
        speech_turn = turn.acquire(priority)
        if speech_turn.expired:
            # Waited past the point where saying this still helps. The one case
            # where silence is right rather than a failure: see turn.SHELF_LIFE_S.
            print(f"[say] dropped: a {priority} that waited too long to matter",
                  file=sys.stderr)
            hub.emit("done")
            return 0
        hub.emit("speaking")
        overlay = _NullOverlay() if args.no_overlay else start_overlay(overlay_label, args.visual)
        try:
            if stream_elevenlabs(text, preset=preset.name, voice=args.voice,
                                 device=args.device, overlay=overlay):
                # The turn is still held here, so the mic opens before anyone
                # else can start talking over the answer.
                return listen_for_answer(hub, overlay, args, lang) if args.ask else 0
            print("[say] stream produced no audio; falling back to synth", file=sys.stderr)
        except Exception as e:
            print(f"[say] stream failed: {e}; falling back to synth", file=sys.stderr)
        finally:
            overlay.stop()
            speech_turn.release()
            hub.emit("done")

    # Start the visual BEFORE synthesizing. It stays invisible until the first
    # audio message reaches it, so this costs nothing visually — but it gives a
    # heavy renderer the synthesis window to warm up in. The GPU face needs
    # ~2.5 s to bring up torch and CUDA; started at playback it would miss the
    # opening of every short utterance.
    overlay = (_NullOverlay() if (args.no_play or args.no_overlay)
               else start_overlay(overlay_label, args.visual))

    # Per-PID temp file: concurrent sessions must not clobber each other's WAV.
    out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / f"say-{os.getpid()}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    backend = pick_backend(args.backend)
    print(f"[say] backend={backend.name} preset={preset.name} voice={voice} "
          f"rate={rate_str} pitch={pitch_str} lang={lang} postfx={not args.no_postfx}",
          file=sys.stderr)

    # Synth -> raw_out (may be .mp3 from Edge, .wav from others). Post-FX keys
    # off whoever actually spoke, so take back the (possibly substituted) backend.
    backend, raw_out = synth_with_fallback(
        backend, text, lang, out, forced=bool(args.backend),
        voice=voice, rate=rate_str, pitch=pitch_str,
        ref_wav=args.ref, rate_int=args.rate, preset=preset.name,
    )

    # Post-process. If postfx is disabled or fails, play raw_out directly.
    # Supertonic ships clean already; post-FX (tuned for Kokoro/SAPI) distorts it.
    use_postfx = (
        not args.no_postfx
        and os.environ.get("TTS_NO_POSTFX") != "1"
        and backend.name != "supertonic"
    )
    final_path = raw_out
    if use_postfx:
        cfg = PostFxConfig()
        ok = _apply_postfx(raw_out, out, cfg, warmth=preset.warmth)
        if ok:
            final_path = out
            # Remove the intermediate mp3 if it sits next to the final wav.
            if raw_out != out and raw_out.exists():
                try:
                    raw_out.unlink()
                except Exception:
                    pass
        else:
            # Keep the raw file so playback still works.
            print("[say] post-processing skipped; playing raw output", file=sys.stderr)

    if not args.no_play:
        # Analyze BEFORE taking the lock: with several sessions queued, doing it
        # here means the orb is ready the instant we get to play, instead of
        # stalling the queue while we crunch numbers.
        analysis = None if args.no_overlay else analyze_audio(final_path)
        hub.emit("queued")
        speech_turn = turn.acquire(priority)
        if speech_turn.expired:
            print(f"[say] dropped: a {priority} that waited too long to matter",
                  file=sys.stderr)
            overlay.stop()
            hub.emit("done")
            return 0
        hub.emit("speaking")
        try:
            def on_start(latency: float) -> None:
                # Fires when the stream starts; `latency` is how much longer
                # until that audio is actually audible.
                if analysis:
                    bands, hop = analysis
                    overlay.send_bands(bands, hop, delay=latency)
                # Same clock, so the caption lands with the word being spoken.
                overlay.send_words(getattr(backend, "words", None), delay=latency)

            play_audio(final_path, device=args.device, on_start=on_start)
            if args.ask:
                return listen_for_answer(hub, overlay, args, lang)
        finally:
            overlay.stop()
            speech_turn.release()
            hub.emit("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

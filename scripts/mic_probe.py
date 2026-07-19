"""Acoustic loopback probe — verify speakers actually emit audible sound.

Plays a known test tone through the default output device while recording
the default input device (mic) in full-duplex. Then FFTs the recording and
checks for a sharp peak at the test frequency.

Logic:
  - Ambient noise is broadband -> low energy in any single narrow bin.
  - A pure sine tone -> a sharp, tall peak at exactly that frequency.
  - So: peak energy at the test freq / median energy of neighbor bins = SNR.
    High SNR => the mic genuinely heard the tone => speakers are emitting.

Asymmetric result:
  - POSITIVE  -> strong evidence: speakers + mic both work, sound is in the room.
  - NEGATIVE  -> inconclusive: headphones, mic muted/disabled, or low volume.
"""
from __future__ import annotations
import sys
import numpy as np

try:
    import sounddevice as sd
except Exception as e:  # pragma: no cover
    print(f"PROBE-ERROR: sounddevice import failed: {e}")
    sys.exit(2)

SR        = 44100
FREQ      = 660.0     # test tone (Hz) — mid-range, uncommon in ambient noise
TONE_AMP  = 0.5
TONE_SEC  = 1.6
BASE_SEC  = 0.8


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def main() -> int:
    # --- device info -------------------------------------------------------
    try:
        di = sd.query_devices(kind="input")
        do = sd.query_devices(kind="output")
    except Exception as e:
        print(f"PROBE-ERROR: no audio device: {e}")
        return 2
    print(f"input device : {di['name']}")
    print(f"output device: {do['name']}")

    # --- baseline (ambient, no tone) --------------------------------------
    try:
        base = sd.rec(int(BASE_SEC * SR), samplerate=SR, channels=1, blocking=True)
    except Exception as e:
        print(f"PROBE-ERROR: mic record failed: {e}")
        return 2
    base = base[:, 0]
    base_rms = rms(base)

    # --- play test tone + record simultaneously ---------------------------
    t = np.arange(int(TONE_SEC * SR)) / SR
    tone = (TONE_AMP * np.sin(2 * np.pi * FREQ * t)).astype(np.float32).reshape(-1, 1)
    try:
        rec = sd.playrec(tone, samplerate=SR, channels=1, blocking=True)
    except Exception as e:
        print(f"PROBE-ERROR: full-duplex play+record failed: {e}")
        return 2
    rec = rec[:, 0]
    # drop first 120ms (output latency ramp-up)
    rec = rec[int(0.12 * SR):]
    play_rms = rms(rec)

    # --- FFT: look for a peak at FREQ -------------------------------------
    win  = rec * np.hanning(len(rec))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(rec), 1.0 / SR)

    band     = (freqs > FREQ - 25) & (freqs < FREQ + 25)
    neighbor = (((freqs > FREQ - 320) & (freqs < FREQ - 70)) |
                ((freqs > FREQ + 70) & (freqs < FREQ + 320)))
    peak_e  = float(spec[band].max())
    noise_e = float(np.median(spec[neighbor]))
    snr     = peak_e / (noise_e + 1e-9)
    rms_ratio = play_rms / base_rms

    print(f"baseline rms : {base_rms:.5f}")
    print(f"playing rms  : {play_rms:.5f}   (x{rms_ratio:.1f} vs baseline)")
    print(f"tone SNR     : {snr:.1f}   (peak@{FREQ:.0f}Hz / neighbor noise)")

    # --- verdict ----------------------------------------------------------
    if snr >= 15:
        print("VERDICT: STRONG-YES - mic clearly heard the tone. Speakers are emitting audible sound.")
        return 0
    if snr >= 6:
        print("VERDICT: YES - tone detected above ambient. Speakers are emitting sound (low volume or distant mic).")
        return 0
    print("VERDICT: NO-SIGNAL - tone not detected. Inconclusive: headphones, mic muted/disabled, or volume off.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

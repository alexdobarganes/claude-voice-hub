"""Tests for input-device selection and the silence guard.

Root cause these cover (reproduced 2026-07-18): the hub recorded from the OS
default input (a Focusrite line input at RMS ~10, effectively silent) instead of
the mic you speak into, and Scribe answered near-silence with a hallucinated
caption ('[outro jingle]') rather than an error. The bad text was then delivered
as if valid, so dictation looked broken with no failure anywhere.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import stt


@pytest.fixture(autouse=True)
def _clear_device_cache():
    """pick_input_device caches its probe; tests must each start cold."""
    stt._picked, stt._picked_device = False, None
    yield
    stt._picked, stt._picked_device = False, None


DEVICES = [
    {"name": "Microsoft Sound Mapper - Input", "max_input_channels": 2},
    {"name": "Analogue 1 + 2 (Focusrite USB Audio)", "max_input_channels": 2},
    {"name": "VoiceMeeter Output (VB-Audio)", "max_input_channels": 2},
    {"name": "Microphone (Logi C270 HD WebCam)", "max_input_channels": 1},
    {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 2},
    {"name": "Speakers (Realtek)", "max_input_channels": 0},
]


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("TTS_STT_DEVICE", "3")
    assert stt.pick_input_device() == 3


def test_picks_loudest_device(monkeypatch):
    monkeypatch.delenv("TTS_STT_DEVICE", raising=False)
    monkeypatch.setattr(stt, "_query_devices", lambda: DEVICES)
    # The Focusrite is the OS default but silent; the webcam mic hears the room.
    levels = {1: 11.0, 3: 120.0}
    monkeypatch.setattr(stt, "_probe_level", lambda idx: levels.get(idx, 0.0))
    assert stt.pick_input_device() == 3


def test_skips_virtual_loopback_devices(monkeypatch):
    monkeypatch.delenv("TTS_STT_DEVICE", raising=False)
    monkeypatch.setattr(stt, "_query_devices", lambda: DEVICES)
    # A loopback carrying music must never win over a quieter real mic.
    monkeypatch.setattr(stt, "_probe_level",
                        lambda idx: {2: 9000.0, 4: 9000.0, 3: 90.0}.get(idx, 0.0))
    assert stt.pick_input_device() == 3


def test_skips_output_only_devices(monkeypatch):
    monkeypatch.delenv("TTS_STT_DEVICE", raising=False)
    monkeypatch.setattr(stt, "_query_devices", lambda: DEVICES)
    monkeypatch.setattr(stt, "_probe_level", lambda idx: 500.0)
    assert stt.pick_input_device() != 5


def test_silent_recording_is_rejected(tmp_path, monkeypatch):
    """The guard that stops a hallucinated transcription of silence."""
    rec = stt.Recorder()
    rec._stream = object()  # pretend a stream was open
    rec._frames = [np.zeros((16000, 1), dtype=np.int16)]
    monkeypatch.setattr(stt, "_close_stream", lambda s: None)
    with pytest.raises(stt.SttError) as exc:
        rec.stop()
    assert "escuch" in str(exc.value).lower()


def test_audible_recording_passes(tmp_path, monkeypatch):
    rec = stt.Recorder()
    rec._stream = object()
    loud = (np.random.default_rng(0).normal(0, 2500, (16000, 1))).astype(np.int16)
    rec._frames = [loud]
    monkeypatch.setattr(stt, "_close_stream", lambda s: None)
    out = rec.stop()
    assert out.exists() and out.stat().st_size > 1000


def test_silence_threshold_is_above_measured_dead_input():
    """The dead Focusrite input measured RMS 10.7; a live mic measured 106+."""
    assert 10.7 < stt.SILENCE_RMS < 106.0

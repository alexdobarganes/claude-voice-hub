"""Device choice across host APIs.

Windows exposes one physical microphone through several host APIs. Choosing by
"which one hears most" then means choosing by which driver is noisiest, which
is how --ask ended up deaf: the noisy path calibrated a speech threshold above
ordinary speech, and questions went unanswered while they were being answered.
"""
from __future__ import annotations

import pytest

import stt


MME = {"name": "MME"}
WDMKS = {"name": "Windows WDM-KS"}
WASAPI = {"name": "Windows WASAPI"}


@pytest.fixture(autouse=True)
def uncached(monkeypatch):
    monkeypatch.setattr(stt, "_picked", False)
    monkeypatch.setattr(stt, "_picked_device", None)
    monkeypatch.delenv("TTS_STT_DEVICE", raising=False)


def _install(monkeypatch, devices, hostapis, levels):
    monkeypatch.setattr(stt, "_query_devices", lambda: devices)
    monkeypatch.setattr(stt, "_probe_level", lambda i, **kw: levels.get(i, 0.0))
    import sys, types
    fake = types.SimpleNamespace(query_hostapis=lambda: hostapis)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)


def test_the_quiet_driver_wins_over_the_noisy_one_for_the_same_mic(monkeypatch):
    """The real case: index 3 is the C270 on MME, index 54 the same mic on
    WDM-KS reading four times the noise. WDM-KS used to win on volume alone."""
    devices = [
        {"name": "Microphone (Logi C270)", "max_input_channels": 1, "hostapi": 0},
        {"name": "Microphone (Logi C270)", "max_input_channels": 1, "hostapi": 1},
    ]
    _install(monkeypatch, devices, [MME, WDMKS], {0: 22.0, 1: 834.0})
    assert stt.pick_input_device() == 0


def test_wdm_ks_is_still_used_when_it_is_all_there_is(monkeypatch):
    """Deprioritized, not banned: a machine with only WDM-KS must still hear."""
    devices = [
        {"name": "Microphone (Logi C270)", "max_input_channels": 1, "hostapi": 0},
    ]
    _install(monkeypatch, devices, [WDMKS], {0: 500.0})
    assert stt.pick_input_device() == 0


def test_among_preferred_hosts_the_loudest_still_wins(monkeypatch):
    """The original rule survives where it was right: the OS default here is a
    line input that reads as silence while you talk into the webcam."""
    devices = [
        {"name": "Focusrite Analogue 1", "max_input_channels": 1, "hostapi": 0},
        {"name": "Microphone (Logi C270)", "max_input_channels": 1, "hostapi": 2},
    ]
    _install(monkeypatch, devices, [MME, WDMKS, WASAPI], {0: 8.0, 1: 300.0})
    assert stt.pick_input_device() == 1


def test_loopbacks_are_still_excluded(monkeypatch):
    """A virtual cable can be louder than any mic while hearing only itself."""
    devices = [
        {"name": "CABLE Output (VB-Audio)", "max_input_channels": 2, "hostapi": 0},
        {"name": "Microphone (Logi C270)", "max_input_channels": 1, "hostapi": 0},
    ]
    _install(monkeypatch, devices, [MME], {0: 9000.0, 1: 120.0})
    assert stt.pick_input_device() == 1


def test_an_explicit_pin_skips_probing_entirely(monkeypatch):
    monkeypatch.setenv("TTS_STT_DEVICE", "7")
    monkeypatch.setattr(stt, "_query_devices",
                        lambda: pytest.fail("should not probe when pinned"))
    assert stt.pick_input_device() == 7


def test_no_usable_input_returns_none(monkeypatch):
    _install(monkeypatch, [], [MME], {})
    assert stt.pick_input_device() is None

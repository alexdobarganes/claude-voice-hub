"""Tests for the platform seam.

These run identically on every platform: the OS calls are stubbed, so what is
under test is the dispatch and the parsing, not the host. That matters because
the macOS paths cannot be exercised on the machine where this was written --
these tests are the only thing standing between a macOS typo and a user.
"""
import subprocess
import sys

import platform_native


# `say -v '?'` output. The awkward part is real: modern voice names contain
# spaces and nested parentheses, so the locale is not simply the second token.
SAY_VOICES = """\
Albert              en_US    # Hello! My name is Albert.
Alice               it_IT    # Ciao! Mi chiamo Alice.
Eddy (Spanish (Spain)) es_ES  # Hola, me llamo Eddy.
Grandma (German (Germany)) de_DE # Hallo, ich heisse Grandma.
Rocko (Spanish (Mexico)) es_MX # Hola, me llamo Rocko.
Samantha            en_US    # Hello! My name is Samantha.
"""


def _stub_say_listing(monkeypatch):
    def fake_run(cmd, **kw):
        assert cmd[:3] == ["say", "-v", "?"]
        return subprocess.CompletedProcess(cmd, 0, stdout=SAY_VOICES, stderr="")
    monkeypatch.setattr(platform_native.subprocess, "run", fake_run)


def test_mac_voice_keeps_names_containing_spaces(monkeypatch):
    """A voice name is everything before the locale, not the first word.

    Splitting on whitespace yields "Eddy", and `say -v Eddy` fails: the real
    name is "Eddy (Spanish (Spain))".
    """
    _stub_say_listing(monkeypatch)
    assert platform_native._mac_voice_for("es") == "Eddy (Spanish (Spain))"


def test_mac_voice_matches_the_language_not_the_region(monkeypatch):
    _stub_say_listing(monkeypatch)
    assert platform_native._mac_voice_for("it") == "Alice"
    assert platform_native._mac_voice_for("de") == "Grandma (German (Germany))"


def test_mac_voice_falls_back_to_system_voice_when_language_absent(monkeypatch):
    """Returning "" is the degradation: `say` then picks the system voice."""
    _stub_say_listing(monkeypatch)
    assert platform_native._mac_voice_for("ja") == ""


def test_mac_voice_survives_say_being_missing(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("say")
    monkeypatch.setattr(platform_native.subprocess, "run", boom)
    assert platform_native._mac_voice_for("es") == ""


def test_console_title_is_always_a_string():
    """Callers treat it as identity metadata; it must never raise or be None."""
    assert isinstance(platform_native.console_title(), str)
    if not platform_native.IS_WINDOWS:
        assert platform_native.console_title() == ""


def test_native_tts_available_is_a_bool_everywhere():
    assert platform_native.native_tts_available() in (True, False)


def test_exactly_one_platform_flag_is_set():
    assert not (platform_native.IS_WINDOWS and platform_native.IS_MACOS)
    assert platform_native.IS_WINDOWS == (sys.platform == "win32")
    assert platform_native.IS_MACOS == (sys.platform == "darwin")

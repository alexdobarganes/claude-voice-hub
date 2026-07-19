"""Backend selection: the fallback chain, and what forcing one means.

This is the logic any future split into a `backends/` package has to preserve
exactly, and it is the part most likely to break silently in such a move -- an
import error in one backend must not change which one gets picked, and a forced
choice must never quietly become a different voice.

The backends themselves stay untested here: they need a network, a GPU or a
model on disk. What is testable is the ordering and the gates, and those are
where a refactor goes wrong.
"""
from __future__ import annotations

import pytest

import say


class _Fake:
    """Stands in for a Backend without needing anything installed."""

    def __init__(self, name: str, available: bool = True) -> None:
        self.name = name
        self._available = available

    def available(self) -> bool:
        return self._available

    def synth(self, *a, **kw):
        raise AssertionError("selection tests never synthesize")


def _chain(monkeypatch, *pairs):
    backends = [_Fake(n, ok) for n, ok in pairs]
    monkeypatch.setattr(say, "BACKENDS", backends)
    return backends


# --------------------------- automatic choice -------------------------------

def test_the_first_available_backend_wins(monkeypatch):
    """Documented as first-available-wins, in the README's stated order."""
    _chain(monkeypatch, ("elevenlabs", False), ("kokoro", True), ("edge", True))
    assert say.pick_backend().name == "kokoro"


def test_an_unavailable_backend_is_skipped_not_fatal(monkeypatch):
    _chain(monkeypatch, ("elevenlabs", False), ("kokoro", False), ("native", True))
    assert say.pick_backend().name == "native"


def test_the_os_voice_is_the_last_resort(monkeypatch):
    """Robotic, but it needs no network, key or model, so it is always there."""
    _chain(monkeypatch, ("elevenlabs", False), ("native", True))
    assert say.pick_backend().name == "native"


def test_no_backend_at_all_is_an_error_not_silence(monkeypatch):
    _chain(monkeypatch, ("elevenlabs", False), ("kokoro", False))
    with pytest.raises(RuntimeError):
        say.pick_backend()


def test_the_real_chain_has_a_backend_that_always_works():
    """Native TTS needs nothing, so the shipped order can never come up empty
    on a supported platform."""
    assert any(b.name == "native" for b in say.BACKENDS)
    assert say.BACKENDS[-1].name == "native"


# --------------------------- forcing one ------------------------------------

def test_forcing_a_backend_selects_exactly_it(monkeypatch):
    _chain(monkeypatch, ("elevenlabs", True), ("kokoro", True))
    assert say.pick_backend("kokoro").name == "kokoro"


def test_forcing_an_unavailable_backend_fails_rather_than_substituting(monkeypatch):
    """Silently speaking in a different voice than the one asked for is worse
    than not speaking: the caller cannot tell it happened."""
    _chain(monkeypatch, ("elevenlabs", True), ("kokoro", False))
    with pytest.raises(RuntimeError):
        say.pick_backend("kokoro")


def test_forcing_an_unknown_name_is_an_error(monkeypatch):
    _chain(monkeypatch, ("kokoro", True))
    with pytest.raises(RuntimeError):
        say.pick_backend("nonexistent")


def test_aliases_resolve_to_real_backend_names(monkeypatch):
    _chain(monkeypatch, *[(n, True) for n in {b.name for b in say.BACKENDS}])
    for alias, real in say.BACKEND_ALIASES.items():
        assert say.pick_backend(alias).name == real


def test_forcing_elevenlabs_without_a_key_says_why(monkeypatch):
    """The common misconfiguration, and the message has to name the cause."""
    _chain(monkeypatch, ("elevenlabs", True))
    monkeypatch.setattr(say, "_EL_KEY", "")
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        say.pick_backend("elevenlabs")

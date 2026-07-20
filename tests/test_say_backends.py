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
    """Stands in for a Backend without needing anything installed.

    `synth_ok=False` makes it fail at synth time the way an expired key or a
    dropped network does -- installed (available) but unable to speak right now.
    """

    def __init__(self, name: str, available: bool = True, synth_ok: bool = True) -> None:
        self.name = name
        self._available = available
        self._synth_ok = synth_ok
        self.synth_calls = 0

    def available(self) -> bool:
        return self._available

    def synth(self, text, lang, out, **kw):
        self.synth_calls += 1
        if not self._synth_ok:
            raise RuntimeError(f"{self.name} cannot speak right now")
        return out


def _chain(monkeypatch, *specs):
    """Each spec is (name, available) or (name, available, synth_ok)."""
    backends = [_Fake(*spec) for spec in specs]
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


# ------------------- degrading when synth fails at runtime -------------------
#
# available() answers "is it installed", not "will it speak right now". The
# runtime failures -- expired key, dropped network, model that won't load --
# all land after a backend has already been picked.

def test_a_runtime_failure_degrades_one_link_not_all_the_way_down(monkeypatch):
    """An ElevenLabs 401 should be heard as Kokoro, not as the robotic OS voice."""
    chain = _chain(monkeypatch, ("elevenlabs", True), ("kokoro", True), ("native", True))
    assert [b.name for b in say.fallback_chain(chain[0])] == ["kokoro", "native"]


def test_degrading_skips_backends_that_are_not_installed(monkeypatch):
    chain = _chain(monkeypatch, ("elevenlabs", True), ("kokoro", False), ("native", True))
    assert [b.name for b in say.fallback_chain(chain[0])] == ["native"]


def test_degrading_never_retries_the_backend_that_just_failed(monkeypatch):
    """Re-running the failure would double the wait before Alex hears anything."""
    chain = _chain(monkeypatch, ("elevenlabs", True), ("kokoro", True))
    assert chain[0] not in say.fallback_chain(chain[0])


def test_the_last_backend_has_nothing_left_to_fall_back_to(monkeypatch):
    """Nothing below native, so the error must surface rather than loop."""
    chain = _chain(monkeypatch, ("elevenlabs", True), ("native", True))
    assert say.fallback_chain(chain[1]) == []


# ------------ degrading at synth time: who actually gets to speak ------------

def _out(tmp_path):
    return tmp_path / "say.wav"


def test_a_runtime_failure_hands_off_to_the_next_backend(monkeypatch, tmp_path):
    """ElevenLabs 401 should be spoken by Kokoro, not dropped to the OS voice."""
    el, ko, na = _chain(monkeypatch,
                        ("elevenlabs", True), ("kokoro", True), ("native", True))
    el._synth_ok = False
    spoke, _ = say.synth_with_fallback(el, "hi", "en", _out(tmp_path), forced=False)
    assert spoke is ko
    assert ko.synth_calls == 1 and na.synth_calls == 0


def test_it_keeps_walking_until_one_can_speak(monkeypatch, tmp_path):
    el, ko, na = _chain(monkeypatch,
                        ("elevenlabs", True, False), ("kokoro", True, False), ("native", True))
    spoke, _ = say.synth_with_fallback(el, "hi", "en", _out(tmp_path), forced=False)
    assert spoke is na


def test_a_forced_backend_that_fails_is_never_substituted(monkeypatch, tmp_path):
    """The regression the reviewer caught: forcing --backend X and having X fail
    at synth time must raise, not quietly speak in a different voice. Silently
    speaking the wrong voice is worse than not speaking -- the caller can't tell.
    """
    el, ko = _chain(monkeypatch, ("elevenlabs", True), ("kokoro", True))
    el._synth_ok = False
    with pytest.raises(RuntimeError):
        say.synth_with_fallback(el, "hi", "en", _out(tmp_path), forced=True)
    assert ko.synth_calls == 0   # never reached for another voice


def test_when_nothing_downstream_can_speak_the_error_surfaces(monkeypatch, tmp_path):
    el, na = _chain(monkeypatch, ("elevenlabs", True, False), ("native", True, False))
    with pytest.raises(RuntimeError):
        say.synth_with_fallback(el, "hi", "en", _out(tmp_path), forced=False)


def test_kokoro_is_the_fallback_for_elevenlabs():
    """When the key or the network is missing, the next voice heard should be
    Kokoro -- local, keyless, and closest in character to the hosted one.
    Supertonic is lighter but flatter, so installing it must not silently
    demote Kokoro out of the fallback slot.
    """
    names = [b.name for b in say.BACKENDS]
    assert names[0] == "elevenlabs"
    assert names[1] == "kokoro"
    assert names.index("kokoro") < names.index("supertonic")


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

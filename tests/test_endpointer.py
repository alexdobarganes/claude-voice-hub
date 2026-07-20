"""Endpointing: when a hands-free answer is over.

The two failure modes are not symmetric. Cutting someone off mid-sentence loses
the answer; waiting a beat too long costs a pause. These tests pin the first.
"""
from __future__ import annotations

import stt

QUIET = 100.0     # a live mic in a quiet room, measured on this machine
LOUD = 900.0      # someone talking


def _run(levels, dt=0.05, **kw):
    """Feed a sequence of block levels and return the states produced.

    settle_s defaults to 0 here so these cases exercise endpointing rather than
    the discarded opening window; the settle itself is tested on its own below.
    """
    kw.setdefault("settle_s", 0.0)
    ep = stt.Endpointer(**kw)
    return ep, [ep.feed(rms, i * dt) for i, rms in enumerate(levels)]


def test_calibrates_from_the_room_before_deciding_anything():
    ep, states = _run([QUIET] * 4, calibrate_s=0.2)
    assert states[:4].count("calibrating") == 4
    assert ep.floor is None or ep.floor == QUIET


def test_room_tone_alone_is_never_mistaken_for_speech():
    """A fixed threshold could not do this: quiet-room tone here reads ~100,
    which is well above the level a dead input reads."""
    ep, states = _run([QUIET] * 40, calibrate_s=0.2, timeout=10.0)
    assert not ep.heard_speech
    assert "speech" not in states


def test_speech_then_trailing_silence_ends_the_capture():
    levels = [QUIET] * 8 + [LOUD] * 10 + [QUIET] * 40
    ep, states = _run(levels, calibrate_s=0.2, silence_tail=1.0, timeout=30.0)
    assert ep.heard_speech
    assert states[-1] == "done"


def test_a_pause_mid_sentence_does_not_cut_the_speaker_off():
    """Half a second of thinking is not the end of an answer."""
    levels = ([QUIET] * 8 + [LOUD] * 6 + [QUIET] * 10 + [LOUD] * 6 + [QUIET] * 4)
    ep, states = _run(levels, calibrate_s=0.2, silence_tail=1.0, timeout=30.0)
    assert "done" not in states  # 10 blocks x 50 ms = 0.5 s < 1.0 s tail
    assert ep.heard_speech


def test_nobody_speaking_times_out_rather_than_hanging():
    levels = [QUIET] * 60
    ep, states = _run(levels, calibrate_s=0.2, timeout=1.0)
    assert states[-1] == "timeout"
    assert not ep.heard_speech


def test_a_dead_input_cannot_calibrate_its_way_into_hearing_things():
    """Floor ~0 would make any breath count as speech if the threshold were a
    pure multiple of it."""
    ep, _ = _run([0.0] * 8 + [stt.SILENCE_RMS * 0.5] * 10, calibrate_s=0.2)
    assert ep.threshold >= stt.SILENCE_RMS
    assert not ep.heard_speech


def test_an_isolated_burst_is_not_an_answer():
    """A door, a keystroke or a blip from the speakers clears the threshold for
    a block or two. Accepting that as speech is how an empty room came back
    with a fluent transcript of something nobody said."""
    levels = [QUIET] * 8 + [LOUD] * 2 + [QUIET] * 40
    ep, states = _run(levels, calibrate_s=0.2, timeout=1.0, min_speech_s=0.35)
    assert not ep.heard_speech
    assert states[-1] == "timeout"


def test_scattered_noise_never_accumulates_into_an_answer():
    """The bug a real empty room found: counting speech blocks cumulatively
    means a fan and a door add up to a sentence given enough seconds. Runs have
    to be consecutive."""
    levels = [QUIET] * 8 + ([LOUD] * 2 + [QUIET] * 8) * 12
    ep, states = _run(levels, calibrate_s=0.2, timeout=4.0, min_speech_s=0.35)
    assert not ep.heard_speech
    assert states[-1] == "timeout"


def test_a_wandering_noise_floor_does_not_read_as_speech():
    """Measured in this room: the floor's median wanders 300 -> 820 with peaks
    past 1700, all of it with nobody talking."""
    import itertools
    wander = list(itertools.chain.from_iterable(
        [[300.0] * 6, [820.0] * 6, [1700.0] * 2, [400.0] * 6] * 6))
    ep, _ = _run([400.0] * 16 + wander, calibrate_s=0.8, timeout=30.0)
    assert not ep.heard_speech


def test_sustained_speech_clears_the_minimum():
    levels = [QUIET] * 8 + [LOUD] * 12 + [QUIET] * 40
    ep, _ = _run(levels, calibrate_s=0.2, silence_tail=1.0, min_speech_s=0.35)
    assert ep.heard_speech


def test_the_contaminated_opening_never_reaches_the_floor():
    """The question we just spoke is still decaying in the room and the mic's
    AGC is ramping. Measured here that window reads a median of ~718 against a
    true floor of ~22; calibrating on it produced a threshold of 2068, above
    normal speech, so the session sat deaf while it was being answered."""
    tts_tail = [718.0] * 10          # 0.5 s of our own voice + AGC ramp
    quiet_room = [22.0] * 16         # the actual floor, once things settle
    ep, _ = _run(tts_tail + quiet_room + [900.0] * 10,
                 settle_s=0.5, calibrate_s=0.8, silence_tail=1.0)
    assert ep.floor is not None and ep.floor < 100
    assert ep.threshold < 900        # i.e. it can still hear a person
    assert ep.heard_speech


def test_without_the_settle_the_same_room_goes_deaf():
    """Pins the failure the settle exists to prevent."""
    tts_tail = [718.0] * 10
    quiet_room = [22.0] * 16
    ep, _ = _run(tts_tail + quiet_room + [900.0] * 10,
                 settle_s=0.0, calibrate_s=0.8, silence_tail=1.0)
    assert ep.threshold > 900
    assert not ep.heard_speech


def test_a_very_long_answer_is_capped():
    levels = [QUIET] * 4 + [LOUD] * 200
    ep, states = _run(levels, calibrate_s=0.1, max_total=2.0, timeout=30.0)
    assert states[-1] == "done"

"""The pure text path through say.py: headers, preprocessing, labels, presets.

say.py is the largest and most-used module in the project and has had no tests
at all, which is also why it is the one nobody wants to refactor. These cover
the parts that are pure functions of their input -- the parts a backend split
would have to move without changing -- so that refactor stops being a leap.

Synthesis, playback and post-FX are still uncovered: they need audio hardware or
a network. That boundary is where this file stops on purpose, not where it ran
out of effort.
"""
from __future__ import annotations

import pytest

import say


# --------------------------- headers ----------------------------------------
# The voice skill authors utterances with a leading `#preset=` line. Getting
# this wrong does not fail quietly: the header is read out loud.

def test_a_preset_header_is_taken_as_a_flag_not_spoken():
    text, preset, lang = say.parse_headers("#preset=calm\nYa está listo")
    assert text == "Ya está listo"
    assert preset == "calm"
    assert lang is None


def test_preset_and_lang_can_share_one_header_line():
    text, preset, lang = say.parse_headers("#preset=excited lang=en\nAll green")
    assert (text, preset, lang) == ("All green", "excited", "en")


def test_headers_may_be_comma_separated():
    _, preset, lang = say.parse_headers("#preset=calm, lang=es\nHola")
    assert (preset, lang) == ("calm", "es")


def test_several_header_lines_are_all_consumed():
    text, preset, lang = say.parse_headers("#preset=calm\n#lang=en\nOK then")
    assert (text, preset, lang) == ("OK then", "calm", "en")


def test_text_with_no_header_is_untouched():
    assert say.parse_headers("Ya está listo") == ("Ya está listo", None, None)


def test_a_line_starting_with_hash_that_is_not_a_header_stays_in_the_text():
    """Otherwise a markdown heading would silently vanish from the utterance."""
    text, preset, _ = say.parse_headers("# Resumen\nTodo bien")
    assert text.startswith("# Resumen")
    assert preset is None


def test_an_unknown_key_does_not_swallow_the_line():
    text, preset, _ = say.parse_headers("#volume=9\nHola")
    assert text.startswith("#volume=9")
    assert preset is None


def test_a_header_after_the_text_is_not_a_header():
    text, preset, _ = say.parse_headers("Hola\n#preset=calm")
    assert "#preset=calm" in text
    assert preset is None


def test_empty_input_is_handled():
    assert say.parse_headers("") == ("", None, None)


# --------------------------- language ---------------------------------------

@pytest.mark.parametrize("text", [
    "Ya está listo",
    "¿Terminaste?",
    "los tests pasan",
    "esto no compila",
])
def test_spanish_is_detected(text):
    assert say.detect_lang(text) == "es"


@pytest.mark.parametrize("text", [
    "Build is green",
    "All tests pass",
    "deploy finished",
])
def test_english_is_detected(text):
    assert say.detect_lang(text) == "en"


# --------------------------- preprocessing ----------------------------------
# Rule of thumb pinned by these: numbers and symbols must not be read out
# character by character, and nothing may be mangled into something wrong.

def test_percentages_are_spoken_as_words():
    assert "por ciento" in say.preprocess_text("subió 12%", "es")
    assert "%" not in say.preprocess_text("subió 12%", "es")


def test_currency_becomes_words():
    out = say.preprocess_text("cuesta $5K", "es")
    assert "mil" in out and "dolares" in out
    assert "$" not in out


def test_bare_magnitudes_expand():
    assert "millones" in say.preprocess_text("son 2.5M de filas", "es")


def test_a_magnitude_letter_inside_a_word_is_left_alone():
    """'3Kb' must not become 'tres mil b'."""
    assert say.preprocess_text("un archivo 3Kb", "es") == "un archivo 3Kb"


def test_known_tickers_are_rewritten_for_the_voice():
    assert "en vidia" in say.preprocess_text("NVDA subió", "es")
    assert "E-S" in say.preprocess_text("el ES abrió", "es")


def test_break_markers_become_pauses_not_literal_text():
    out = say.preprocess_text("listo <break 700ms/> ya", "es")
    assert "break" not in out
    assert "..." in out


def test_clean_text_survives_preprocessing_unchanged():
    """It is called on every utterance, so it must be safe on ordinary prose."""
    plain = "El build está verde y los tests pasan"
    assert say.preprocess_text(plain, "es") == plain


def test_preprocessing_never_leaves_double_spaces():
    assert "  " not in say.preprocess_text("subió  12%   hoy", "es")


# --------------------------- overlay label ----------------------------------
# The label is what identifies the speaking session on screen and in the hub,
# so a wrong one means Alex looks at the wrong tab.

def test_the_session_tag_before_the_colon_becomes_the_label():
    assert say.derive_overlay_label("Billing api, retry logic: ya está") \
        == "Billing api, retry logic"


def test_performance_tags_are_not_part_of_the_label():
    assert say.derive_overlay_label("[calm] Voice hub: listo") == "Voice hub"


def test_break_markers_are_not_part_of_the_label():
    label = say.derive_overlay_label('Voice hub: <break time="0.4s" /> listo')
    assert label == "Voice hub"


def test_without_a_colon_the_opening_words_are_used():
    assert say.derive_overlay_label("Ya terminé de correr los tests") \
        == "Ya terminé de correr"


def test_a_very_long_head_is_not_used_as_a_label():
    """A whole sentence before a colon is prose, not a session tag."""
    long_head = "esto es una frase larguísima que no es una etiqueta de sesión"
    assert say.derive_overlay_label(f"{long_head}: hola") != long_head


def test_there_is_always_some_label():
    assert say.derive_overlay_label("") == "Claude"
    assert say.derive_overlay_label("[calm]") == "Claude"


# --------------------------- presets ----------------------------------------

def test_every_documented_preset_resolves():
    for name in say.PRESETS:
        assert say.resolve_preset(name).name == name


def test_an_unknown_preset_in_a_header_falls_back_rather_than_silencing():
    """A typo in a #preset= header is inside content that was meant to be
    spoken, written by an agent that will not see the error. Exiting there
    means Alex simply never hears the line."""
    assert say.resolve_preset("clam").name == say.DEFAULT_PRESET


def test_an_unknown_preset_on_the_command_line_still_fails_loudly():
    """That one is a human at a prompt who can retype it."""
    with pytest.raises(SystemExit):
        say.resolve_preset("clam", strict=True)


def test_no_preset_gives_the_default():
    assert say.resolve_preset(None).name == say.DEFAULT_PRESET


def test_a_rate_delta_shifts_the_preset_rate():
    """Documented as ~5% per step."""
    assert say.apply_rate_delta("+0%", 4) == "+20%"
    assert say.apply_rate_delta("+10%", -1) == "+5%"
    assert say.apply_rate_delta("+0%", -2) == "-10%"


def test_the_rate_is_clamped_so_speech_stays_intelligible():
    assert say.apply_rate_delta("+0%", 99) == "+50%"
    assert say.apply_rate_delta("+0%", -99) == "-50%"


def test_a_zero_delta_leaves_the_rate_alone():
    assert say.apply_rate_delta("+7%", 0) == "+7%"


# --------------------------- audio tags -------------------------------------

def test_audio_tags_are_stripped_for_backends_that_cannot_perform_them():
    """Only ElevenLabs performs these. Anywhere else they would be read out."""
    assert "[excited]" not in say._strip_audio_tags("[excited] ya está")
    assert "ya está" in say._strip_audio_tags("[excited] ya está")

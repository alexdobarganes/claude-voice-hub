"""Being addressed: the guard the acoustic tests could not be."""
from __future__ import annotations

import pytest

import wake


# --------------------------- addressing -------------------------------------

@pytest.mark.parametrize("said", [
    "Claude, abre el PR",
    "claude abre el PR",
    "Oye Claude, abre el PR",
    "hey Claude abre el PR",
    "Clod, abre el PR",          # what a recognizer actually returns
    "Cloud abre el PR",
    "Claudio, abre el PR",
])
def test_being_called_by_name_counts_however_it_was_heard(said):
    assert wake.strip_address(said) == "abre el pr"


def test_the_name_alone_is_still_an_address():
    """'Claude?' means 'are you there', not nothing."""
    assert wake.strip_address("Claude?") == ""


def test_a_room_talking_about_something_else_is_ignored():
    """Measured with a live mic: correctly transcribed Spanish that simply was
    not for us."""
    assert wake.strip_address("Mira lo que me hice de eso de la palomita") is None


def test_english_audio_from_a_nearby_screen_is_ignored():
    assert wake.strip_address("And? I have a friend. That's a surprise") is None


def test_a_passing_mention_is_not_an_address():
    """Talking *about* it is not talking *to* it."""
    assert wake.strip_address("le estaba diciendo a Juan que Claude ya lo hizo") is None


def test_accents_and_punctuation_do_not_matter():
    assert wake.strip_address("¡Oye, Claude! ¿mergeas?") == "mergeas"


def test_silence_is_not_an_address():
    assert wake.strip_address("") is None
    assert wake.strip_address("   ") is None


def test_filler_after_the_name_is_dropped():
    assert wake.strip_address("Claude, eh, abre el PR") == "abre el pr"


# --------------------------- routing ----------------------------------------

SESSIONS = [
    {"session_id": "s1", "name": "billing api"},
    {"session_id": "s2", "name": "voice hub"},
    {"session_id": "s3", "name": "orb"},
]


def test_naming_a_session_routes_to_it():
    assert wake.route("billing api, abre el PR", SESSIONS) == "s1"


def test_half_a_name_does_not_route():
    """Predictable beats clever: 'lo del billing' and 'sube la api de pagos'
    are both one word out of 'billing api', and nothing distinguishes them
    without ranking word distinctiveness, which is guesswork. Both fall back."""
    assert wake.route("lo del billing ya está", SESSIONS) is None
    assert wake.route("sube la api de pagos", SESSIONS) is None


def test_the_more_specific_full_match_wins():
    sessions = SESSIONS + [{"session_id": "s4", "name": "billing api staging"}]
    assert wake.route("billing api staging, para", sessions) == "s4"


def test_a_one_word_name_needs_the_whole_word():
    assert wake.route("el orb se ve raro", SESSIONS) == "s3"
    assert wake.route("se ve raro", SESSIONS) is None


def test_naming_nobody_routes_nowhere():
    assert wake.route("abre el PR", SESSIONS) is None


def test_routing_survives_sessions_with_no_usable_name():
    assert wake.route("abre el PR", [{"session_id": "s1", "name": ""}]) is None


# --------------------------- plausibility -----------------------------------

def test_noise_hallucinated_into_another_alphabet_is_not_speech():
    """Verbatim from an empty room, via the local recognizer."""
    assert not wake.looks_like_speech("ू ॉ, ॑ ॗ ख़ OH ६astics © başall comprehensive")


def test_ordinary_spanish_is_speech():
    assert wake.looks_like_speech("Claude, abre el PR ¿vale?")


def test_ordinary_english_is_speech():
    assert wake.looks_like_speech("open the pull request please")


def test_nothing_is_not_speech():
    assert not wake.looks_like_speech("")
    assert not wake.looks_like_speech("   ")
    assert not wake.looks_like_speech("... ,, !!")


def test_a_stray_symbol_does_not_disqualify_a_real_sentence():
    assert wake.looks_like_speech("sí © dale, abre el PR")


def test_the_variant_a_recognizer_actually_returned():
    """Observed verbatim: 'Claude' spoken in Spanish came back as 'club'."""
    assert wake.strip_address("club, abre el PR") == "abre el pr"


def test_a_misheard_vocative_still_counts_as_an_address():
    """From the self-test: 'Oye Claude, para el deploy' came back as
    'Oje Claude! Parallde ploi!'. 'oje' is not 'oye', and a perfectly good
    address was being thrown away over one letter."""
    assert wake.strip_address("Oje Claude! Parallde ploi!") is not None


def test_an_ordinary_word_before_the_name_is_still_not_an_address():
    """The slack must not be so wide that talking about it becomes talking
    to it."""
    assert wake.strip_address("creo que Claude ya lo hizo") is None


def test_the_second_mishearing_of_the_same_phrase_also_counts():
    """Same sentence, two runs, two spellings: 'Oje Claude!' and 'Ojeh, Claude.'
    Tolerance has to scale with word length to cover both."""
    assert wake.strip_address("Ojeh, Claude. Parallel deploy.") is not None


@pytest.mark.parametrize("said", [
    "creo que Claude ya lo hizo",
    "que Claude lo revise",
    "dile a Claude que pare",
])
def test_talking_about_it_is_still_not_talking_to_it(said):
    """Pins the words the tolerance must never swallow."""
    assert wake.strip_address(said) is None


# --------------------------- near misses of the name ------------------------

def test_a_near_miss_of_a_listed_variant_still_wakes():
    """The corpus found 'clud' after 'club'. Enumerating by hand does not
    converge, so near misses of a listed variant count."""
    assert wake.strip_address("¡Clud! ¿Cómo va eso?") is not None


@pytest.mark.parametrize("said", [
    "lo hago ahora mismo",
    "la clave está en el env",
    "la clase esa no compila",
    "el clon del repo está viejo",
    "con lo que me dijiste basta",
])
def test_common_words_near_the_name_do_not_wake_it(said):
    """The short variants are the dangerous ones: one edit from 'clo' admits
    'lo', which would wake the assistant constantly."""
    assert wake.strip_address(said) is None

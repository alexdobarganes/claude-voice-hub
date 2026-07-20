"""Deciding whether something said in the room was said *to* us.

This is the guard that the acoustic tests could not be. An open microphone in a
real room hears other people, a television, another session's own voice coming
back through the speakers. Measured here: a capture taken while nobody was
addressing the machine came back as fluent, high-confidence Spanish -- correctly
transcribed speech that simply was not for us. No amount of level thresholding
separates one voice from another, because the difference is not acoustic.

What separates them is being addressed. So: unprompted speech has to name the
assistant, and everything else in the room is ignored no matter how clear it is.

The matching is deliberately loose. A local transcriber hearing "Claude" spoken
in Spanish returns "Clod", "Clau", "Cloud", "Clot" and worse, and a wake word
that only works when the recognizer is right is a wake word that does not work.
Being too eager costs a discarded sentence; being too strict costs Alex
repeating himself, which is exactly the friction this is supposed to remove.
"""
from __future__ import annotations

import re
import unicodedata

# Spelling variants a recognizer actually produces for the name, not the ways
# it is properly spelled. Order does not matter; longest match wins.
WAKE_WORDS = (
    "claude", "claud", "clod", "cloud", "clode", "clau", "clo", "claudio",
    "cloude", "klaud", "klod", "club", "clob", "asistente", "assistant",
)
# "club" is in that list because a recognizer returned it, verbatim, for "Claude"
# spoken in Spanish. It is a real word, so it will occasionally wake us for
# nothing; that costs one ignored sentence, while leaving it out costs Alex
# saying the name again and wondering why he is being ignored. This list should
# grow the same way: from what was actually heard, not from what should be.

# Address forms that may precede the name: "oye Claude", "hey Claude".
_VOCATIVES = ("oye", "hey", "ey", "eh", "hola", "ok", "okay", "vale", "escucha")

# How far into the utterance the name may appear before we stop believing it
# was an address rather than a passing mention.
ADDRESS_WINDOW_WORDS = 3


def looks_like_speech(text: str, min_words: int = 1) -> bool:
    """False when a transcript is the recognizer's reaction to noise.

    Given a stretch of room noise, a small local model does not return nothing:
    it returns something, and what it returns wanders out of the language it was
    asked about. Observed here, verbatim, from an empty room:

        'ू ॉ, ॑ ॗ ख़ OH ६astics © başall comprehensive'

    The test is script, not vocabulary. Judging the words would need a
    dictionary, and most of that string is Latin anyway -- 'astics', 'başall',
    'comprehensive' are all Latin letters, so any ratio-based test passes it.
    What gives it away is that a single letter belongs to an alphabet nobody in
    the room was speaking, so a single one is enough to reject the whole thing.

    That is strict on purpose. It costs a rejected utterance if a recognizer
    ever emits a stray non-Latin character inside real Spanish or English, which
    is rare; it saves acting on noise, which is not.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    if any("LATIN" not in unicodedata.name(c, "") for c in letters):
        return False
    return len(normalize(text).split()) >= min_words


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a: str, b: str, cap: int = 2) -> int:
    """Levenshtein distance, stopping once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def _is_wake(token: str) -> bool:
    """Whether a word is the assistant's name, allowing for being misheard.

    Enumerating variants by hand does not converge -- the corpus found "clud"
    after "club" -- so near misses of a listed variant count too. But the slack
    has to be length-scaled here even more carefully than for vocatives,
    because the short variants are the dangerous ones: one edit from "clo"
    admits "lo", which is among the most common words in Spanish and would wake
    the assistant constantly.

    So: exact match below five letters, one edit at five or more. Checked
    against the words that must not pass -- "lo" stays out, and "clave" and
    "clase" are two edits from every variant, not one.
    """
    if token in WAKE_WORDS:
        return True
    return any(_edit_distance(token, w, cap=1) <= 1
               for w in WAKE_WORDS if max(len(token), len(w)) >= 5)


def _is_vocative(token: str) -> bool:
    """Whether a word before the name is an address form, allowing for misheard.

    Exact matching was too brittle here, and the self-test caught it twice:
    "Oye Claude, para el deploy" came back first as "Oje Claude!" and then as
    "Ojeh, Claude.", so a perfectly good address was thrown away over one or two
    letters.

    The slack has to scale with length rather than be a constant. Two edits is
    nothing across "ojeh" and "oye", but across "eh" it is the whole word, and
    a flat tolerance of two would make half the language a way of calling
    someone. So: one edit for short forms, two once a word is long enough for
    two edits to still mean something.

    Checked against the words that must NOT pass -- "creo", "que" -- which stay
    three or more edits from every vocative at any tolerance used here.
    """
    for v in _VOCATIVES:
        allowed = 2 if max(len(token), len(v)) >= 4 else 1
        if _edit_distance(token, v, cap=allowed) <= allowed:
            return True
    return False


def strip_address(text: str) -> str | None:
    """The instruction with its address removed, or None if not addressed.

    Returns "" when the utterance was *only* the name ("Claude?"), which is a
    real thing people say and means "are you there", not nothing.
    """
    words = normalize(text).split()
    if not words:
        return None
    for i, w in enumerate(words[:ADDRESS_WINDOW_WORDS]):
        if _is_wake(w):
            # Everything before the name must be an address form, or this was a
            # mention inside a sentence rather than someone calling us.
            if any(not _is_vocative(p) for p in words[:i]):
                return None
            rest = words[i + 1:]
            # A trailing vocative comma ("Claude, ...") leaves nothing to strip,
            # but a leading filler after the name is noise: "claude eh, haz X".
            while rest and _is_vocative(rest[0]):
                rest = rest[1:]
            return " ".join(rest)
    return None


def is_addressed(text: str) -> bool:
    return strip_address(text) is not None


def route(text: str, sessions: list[dict]) -> str | None:
    """Session id named in the utterance, or None.

    `sessions` are dicts with at least `session_id` and a human name (`name` or
    `tag`). Naming a session is how Alex picks one out of several: "billing api,
    abre el PR".

    The whole name is required, not part of it. Partial matching cannot be made
    to behave: "lo del billing" and "sube la api de pagos" are both one word out
    of "billing api", and no rule distinguishes them without ranking how
    distinctive each word is -- which is guesswork dressed as scoring. Being
    strict is cheap here because failing to route is not failing to deliver: the
    message falls back to whoever asked most recently, which is the common case
    anyway. Misrouting an instruction to the wrong session is not cheap.
    """
    said = set(normalize(text).split())
    if not said:
        return None
    best, best_len = None, 0
    for s in sessions:
        label = normalize(str(s.get("name") or s.get("tag") or ""))
        words = [w for w in label.split() if len(w) > 2]
        if not words or not all(w in said for w in words):
            continue
        if len(words) > best_len:   # a longer full match is the more specific one
            best, best_len = s.get("session_id"), len(words)
    return best

"""Phrases the assistant must get right, and what each one is testing.

Kept separate from the unit tests because these are run through real synthesis
and a real recognizer by `assistant.py --self-test`, not asserted against a
string. The unit tests pin the logic; this pins the behaviour end to end, which
is where every wake-word bug so far has actually lived.

`True` means the assistant must treat it as addressed to it. `False` means it
must ignore it, and those matter more: a missed wake costs Alex repeating
himself, while a false wake delivers something he never said into a session.
"""

# --------------------------- must wake --------------------------------------

ADDRESSED = [
    # Plain address, the common case.
    "Claude, abre el PR",
    "Claude, para el deploy",
    "Claude, corre los tests",
    "Claude, ¿cómo va eso?",
    "Claude, mergea a main",
    # With an address form in front.
    "Oye Claude, para el deploy",
    "Hey Claude, abre el PR",
    "Oye Claude, ¿terminaste?",
    "Vale Claude, sigue",
    "Escucha Claude, para todo",
    # The name alone: "are you there".
    "Claude",
    "Claude?",
    # Naming a session after the name.
    "Claude, voice hub, para",
    "Claude, billing api, abre el PR",
    # English, since Alex switches.
    "Claude, open the PR",
    "Hey Claude, stop the deploy",
    # Known failure, kept rather than removed: a Spanish TTS voice reading
    # "Hey Claude, abre el PR" produces audio the recognizer transcribes as
    # Arabic ('هيقلاغ أبري LPR'), which the non-Latin guard then rejects. That
    # guard is not going to be relaxed to make this line pass -- it is what
    # catches room hum turning into Devanagari, and trading a real defence for
    # a better score is how a metric starts lying. This is a limitation of
    # synthesized English-in-a-Spanish-voice, not of the assistant; a person
    # saying it is a different signal. Left here so it stays visible.
    "Hey Claude, abre el PR",
    # Longer instructions: the tail must survive.
    "Claude, cuando termines los tests haz commit y súbelo",
    "Claude, deja eso y revisa el error de producción",
]

# --------------------------- must NOT wake ----------------------------------

IGNORED = [
    # Room conversation. The first is verbatim from a live capture.
    "Mira lo que me hice de eso de la palomita",
    "Y luego fuimos al cine y no había entradas",
    "¿Me pasas el vaso que está en la mesa?",
    "No sé si mañana voy a poder ir",
    # Talking *about* it rather than *to* it.
    "creo que Claude ya lo hizo",
    "dile a Claude que pare",
    "le estaba diciendo a Juan que Claude lo arregló",
    "el problema es que Claude no entendió",
    # English room audio, the kind that produced a false answer once.
    "And I have a friend who says that's a surprise",
    "so then he told me the whole thing was broken",
]

ALL = [(p, True) for p in ADDRESSED] + [(p, False) for p in IGNORED]

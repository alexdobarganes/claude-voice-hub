---
name: voice-hub
description: >-
  Speak short status updates aloud via voice-hub's say.py so the user always
  knows which parallel Claude Code session needs attention. Use when a
  finished task, a blocker, or a question would land faster spoken than
  scrolled past in text — especially with several sessions running at once.
---

# voice-hub

Speak a short line while continuing work. TTS must never block the main task.

## Setup

Run from this repo's own venv — no activation needed, call the interpreter
directly:

```bash
.venv/bin/python say.py "[tag] message"
```

If `.venv` doesn't exist yet, see [CONTRIBUTING.md](../../../CONTRIBUTING.md)
for install steps. With no API keys configured, `say.py` falls through the
backend chain automatically: hosted (ElevenLabs) → local neural (Supertonic,
Kokoro) → Edge TTS → native OS voice. Nothing needs to be started by hand —
`hub.py` (the always-on-top panel showing who's speaking) launches itself
automatically on first use.

If `ffmpeg` isn't on `PATH`, network backends skip audio post-processing and
play a raw file instead; the native backend is unaffected either way. This is
expected, documented degradation — don't work around it.

## When to speak

Fire a line at:

1. **Done** — one-sentence outcome of a finished task.
2. **Blocker** — need user input, auth, or a decision to continue.
3. **Question** — waiting on the user for something specific.

Skip routine tool narration. Prefer one clear line over frequent chatter.

## The tag convention (required)

Every spoken line opens with a short tag naming the session or task, and the
written message in the same turn leads with the same tag. This is how the
user tells parallel sessions apart by ear.

```
Written: "[Billing API] Tests green, PR opened."
Spoken:  say.py "[Billing API] Tests green, PR opened."
```

Session identity for the hub's row/history view is resolved automatically
from `CLAUDE_CODE_SESSION_ID` — the tag is purely for the human ear, so pick
something short and recognizable (feature name, repo, or task).

## Usage

Run non-blocking so the task keeps moving — do not await playback or poll
for it:

```bash
.venv/bin/python say.py "[tag] message"
```

Use the Bash tool with `run_in_background: true`.

Style: 1–2 spoken sentences, plain speech (no markdown, bullets, code, URLs,
or file paths), present tense, concrete.

Useful flags: `--preset excited|calm|dramatic|news|brief|neutral` to change
delivery in one shot; `--no-play --out <file>` to generate without playing.

## Failure handling

If `say.py` errors (no mic, no network, backend failure), note it once in
text and keep working. Never retry in a loop or block the task on audio.

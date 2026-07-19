# voice-hub

Give your parallel Claude Code sessions a voice, and always know which one is talking.

## The problem

Running several Claude Code sessions at once is productive right up to the moment
they all want your attention. Text scrolls past in tabs you are not looking at, and
if you add naive text-to-speech, two agents talk over each other and you cannot tell
which one asked the question.

voice-hub solves both halves:

- **Speech is serialized.** A lock inside `say.py` means utterances queue instead of
  overlapping. One voice at a time, in order.
- **The speaker is identified.** An always-on-top panel shows who is speaking now,
  what is queued behind it, and who spoke recently. Clicking a row jumps to that
  session.

## Platform support

Be aware of what is actually tested before you install this.

| Platform | Status |
| --- | --- |
| Windows | Verified. Developed here and in daily use. |
| macOS | Partly written, **not verified**. See below. |
| Linux | Not supported. |

On macOS, `say.py` and `hub.py` are written to run: the native voice maps to
`say(1)`, playback falls back to `afplay(1)`, and the console-title lookup
degrades to empty (session identity comes from `CLAUDE_CODE_SESSION_ID`, which
is portable). **The orb overlay is Windows-only** — it depends on
`UpdateLayeredWindow` for real per-pixel alpha, and no equivalent has been
written yet. Because the overlay is a separate process rather than an import,
it simply fails to start on macOS and speech continues unaffected.

Jumping to a session (`nav.py`) is also Windows-only; on macOS the hub falls
back to copying a `claude --resume <id>` command to the clipboard.

None of the macOS paths have been executed on a Mac. Reports welcome.

## Installation

Requires Python 3.10+.

**1. Install ffmpeg.** `pydub` shells out to it for audio decoding and post-processing;
without it on your `PATH`, playback of MP3 output fails.

```bash
# Windows
choco install ffmpeg
# macOS
brew install ffmpeg
```

**2. Install the package** with the extras for your platform.

```bash
git clone <your-fork-url> voice-hub
cd voice-hub
python -m venv .venv

# Windows
.venv\Scripts\pip install -e ".[windows,dev]"
# macOS
.venv/bin/pip install -e ".[macos,dev]"
```

Optional extras: `supertonic` (lightest offline backend), `local` (kokoro ~350MB,
f5-tts ~1.3GB), `stt` (offline transcription fallback), `gpu` (torch, for
`experiments/orb_gpu.py`).

**3. Configure a `.env`** in the project root. Everything is optional: with no keys
at all, `say.py` falls through to a local backend.

```ini
ELEVENLABS_API_KEY=...       # enables the highest-quality backend
ELEVENLABS_VOICE_ID=...      # defaults to a stock voice if unset
TTS_F5_ENABLED=1             # opt in to F5-TTS (avoids the 1.3GB download by default)
TTS_ORB_CPU=1                # force the CPU orb renderer
```

## Usage

### Speaking

```bash
python say.py "Build is green, tests pass."
python say.py --lang en --rate 4 "Faster, in English."
python say.py --preset excited "Deploy finished."
python say.py --backend supertonic "Force a specific backend."
python say.py --list                  # available backends, voices, output devices
python say.py --list-presets
python say.py --no-play --out clip.wav "Generate a file, do not play it."
```

Presets (`neutral`, `excited`, `calm`, `dramatic`, `news`, `brief`) bundle voice, rate
and pitch so changing the delivery is one flag. Text is preprocessed before synthesis:
tickers, currency, percentages, large numbers and common abbreviations are rewritten so
they are spoken naturally, and post-FX (DC offset removal, loudness normalization,
gentle EQ) run over the result unless you pass `--no-postfx`.

### The hub

```bash
python hub.py
```

`say.py` launches it automatically when it is not already running, so you rarely start
it by hand. It is a singleton (lockfile) and exits on its own after a stretch of
silence; the next utterance revives it. Rows show the speaking session, the queue and
recent history; clicking one performs the best available jump to that session.

### Dictation

The hub supports push-to-talk dictation (`stt.py`): it records the default microphone,
transcribes via ElevenLabs Scribe, and falls back to `faster-whisper` when installed
(`pip install -e ".[stt]"`). Silence is rejected rather than transcribed, because a hosted model asked
to transcribe silence will confidently invent something.

## Integrating with Claude Code

There is no daemon to install. A Claude Code skill instructs the agent to call `say.py`
directly via Bash when a spoken line would land faster than text: a finished task, a
blocker, a question waiting on you.

The convention that makes parallel sessions usable: **every spoken line opens with a
short tag naming the session or task**, and the written message leads with the same tag.
So you hear "Billing api, tests green" and know exactly which tab to look at, and the hub
shows you the same identity in its row.

Session identity is not guesswork. `CLAUDE_CODE_SESSION_ID` is present in the
environment and inherited by subprocesses, so `say.py` knows exactly which session is
speaking without the agent cooperating, and `agents_registry.py` resolves that id to a
human-readable name and project by wrapping `claude agents --json`.

## Architecture

`say.py` synthesizes and plays audio, holding a lock so concurrent invocations queue.
As it works it writes JSON events (queued / speaking / done) to a spool directory that
`hub.py` tails: the two processes never talk directly, which is why the hub can die,
restart, or be absent without breaking speech. `hub.py` is a tkinter always-on-top
panel that renders the spool and uses `nav.py` to jump to a session. `overlay.py` plus
`orb_render.py` draw an audio-reactive particle orb in a transparent borderless window
while a line is playing.

The backend chain is first-available-wins, in this order, overridable with `--backend`:

1. **ElevenLabs** — hosted, best quality and dynamics. Used when `ELEVENLABS_API_KEY` is set.
2. **Supertonic** — local. Not installed by default: `pip install -e ".[supertonic]"`.
   Without it the chain falls through to Edge TTS, which is free but needs network.
3. **Kokoro** — local neural, no network, ~350MB.
4. **Edge TTS** — Microsoft's free neural cloud voices.
5. **F5-TTS** — GPU voice cloning against a reference clip, behind `TTS_F5_ENABLED=1`.
6. **Native** — the OS voice: SAPI on Windows, `say(1)` on macOS. Robotic, but
   needs no network, no key and no model, so it is always there.

## Status and known limitations

This is personal tooling being opened up, not a polished product. Specifically:

- **macOS is unverified and partial.** The speech and hub paths are written but have
  never been executed on a Mac; the orb overlay and session-jumping are Windows-only.
  Treat macOS as a work in progress, not a supported platform.
- **`say.py` is a ~1,600-line monolith.** Every backend, the text preprocessor, the
  preset system and the post-FX chain live in one file. It should be split into a
  `backends/` package with a common interface. Until then, changes there are riskier
  than they should be.
- **No test coverage on `say.py` or `overlay.py`.** The 43 tests cover the event spool,
  navigation, sound events and STT. The largest and most-used modules are untested.
- **ffmpeg is a hard system dependency**, not installable via pip.
- Linux is not supported and nothing has been done to make it work.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The two areas where help is most valuable are
verifying and fixing macOS, and splitting `say.py` into backend modules.

## License

MIT. See [LICENSE](LICENSE).

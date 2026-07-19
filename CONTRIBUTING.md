# Contributing

## Setup

Python 3.10+ and [ffmpeg](https://ffmpeg.org/) on your `PATH` (`choco install ffmpeg`
on Windows, `brew install ffmpeg` on macOS).

```bash
python -m venv .venv

# Windows
.venv\Scripts\pip install -e ".[windows,dev]"
# macOS
.venv/bin/pip install -e ".[macos,dev]"
```

## Tests

```bash
pytest
```

43 tests, and they need **no credentials, no network and no microphone**. Anything that
would reach the outside world is monkeypatched, and file I/O goes to `tmp_path`. If you
add a test that needs a key or a device, it belongs behind a skip marker, not in the
default run.

## Where help is most valuable

**1. Verify and fix macOS.** The macOS paths are written but have never been executed
on a Mac. Running the test suite, then `say.py` and `hub.py`, and reporting exactly what
breaks is genuinely the highest-value contribution right now. Concrete unknowns: audio
device selection through `sounddevice`, the tkinter always-on-top and transparency
behavior of the hub and overlay, and everything in `nav.py` that jumps to a session.

**2. Split `say.py` into backends.** It is ~1,600 lines holding six TTS backends, the
text preprocessor, the preset system and the post-FX chain. The target shape is a
`backends/` package where each backend is its own module behind the existing `Backend`
interface (`name`, `available()`, synthesis), with `say.py` reduced to CLI, chain
selection and playback. Do this incrementally, one backend per PR, and add tests as
each one comes out. Please do not change behavior and restructure in the same commit.

## Conventions

- Code and comments in English.
- Comments explain *why*, especially where a value was measured on real hardware
  (see the silence threshold in `stt.py`). Do not strip those.
- Keep degradation graceful: a missing key, model or device should fall through to the
  next option, not raise.

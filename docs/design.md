# Voice Hub: navigate to the speaking agent, visible queue, voice reply

> This is the original design document, kept as a record of how the current
> architecture was arrived at.

Status: approved design (phases A and B).

Context: you run many Claude Code sessions in parallel. `say.py` already
serializes audio through a lock and shows an ephemeral pill with the tag of the
session that is speaking. This design turns that pill into a persistent hub
that lets you: (1) click to jump to the speaking agent's session, (2) see the
queue and the history of speakers, (3) reply by voice to the session that asked
a question (push-to-talk STT).

## Post-spike revision: the navigation model changes

The gate spike for Task 1 invalidated the "Windows Terminal tabs" premise and
found a better path. Measured evidence on a real machine:

- **Zero `WindowsTerminal.exe` processes.** There were no WT tabs to select.
- Sessions lived in: the **Claude desktop app** (one window, ~20 session
  processes hanging off it), **one `cmd.exe` window** (`claude agents`), and
  **background jobs with no window at all** (root ancestor `WmiPrvSE.exe`).
- The UI Automation tree of the desktop app exposes only 15 panes and 4
  documents, and **no per-session element**: there is no way to focus one
  specific session through UIA.
- The `claude-cli://` handler exists but (confirmed against the official docs)
  only supports `claude-cli://open?q=&cwd=&repo=`, which **creates new
  sessions**. There is no URI to focus an existing one (open feature request:
  anthropics/claude-code issues #40169 and #44063).
- **`CLAUDE_CODE_SESSION_ID` is in the environment** and is inherited by child
  processes: `say.py` knows exactly which session is speaking, with no
  cooperation from the agent.
- **`claude agents --json`** returns the registry of live sessions with
  `sessionId`, a human-readable `name`, `cwd`, `pid`, `kind`
  (interactive/background) and `status`.

Consequence: "select the WT tab" is dropped as the central mechanism. Exact
session identity (which *is* resolvable) becomes the heart of the hub, and a
click performs **the best available action** for that kind of session.

This is the part of the document worth keeping: the first model was plausible
and wrong, and it took measurement rather than argument to find that out.

## Decisions

- Target environment: mixed and real (desktop app + console window + background
  jobs). Windows Terminal tab support is retained but only applies if you
  actually open WT.
- Navigation interaction: click on the hub's pill/row (no global hotkey for
  navigation).
- Approved scope: navigation + visible queue/history + push-to-talk STT.
  "A different voice per session" is out of scope (revisit later).
- Approach: an incremental "Voice Hub" on the same stack (pythonw + tkinter).
  Rejected: a clickable-pill-only variant (covers neither queue nor STT), and a
  tray app on WebView/WPF (a new stack with no extra functional value).
- Phases: Phase A = hub + navigation + queue/history. Phase B = STT.

## Architecture

Three pieces, all in the repository root:

1. **`say.py`** (existing, modified): stops spawning its own pill and publishes
   JSON events into a spool at `hub/events/`. It launches the hub if it is not
   already running (singleton via the `hub/hub.lock` lockfile holding a PID). If
   the hub fails to start, it falls back to the current pill: speech is never
   left without a visual indicator.
2. **`hub.py`** (new): a persistent UI process (pythonw + tkinter),
   always-on-top, bottom-right corner. It consumes the spool. It exits on its
   own after 10 minutes with no activity; the next `say.py` revives it.
3. **`stt.py`** (new, Phase B): microphone capture + transcription + delivery of
   the text to the target session.

### Event contract (spool)

One JSON file per event, named `<ts>-<pid>-<state>.json`:

```json
{
  "ts": 1752868000.123,
  "pid": 12345,
  "state": "queued | speaking | done",
  "tag": "Billing api, retry logic",
  "preset": "calm",
  "text_head": "first ~80 chars of the utterance",
  "session_id": "19224582-81fd-4bbe-8ee6-7077e41b7a73",
  "claude_pid": 336444,
  "cwd": "/path/to/example-app",
  "nav": {
    "hwnd": 132456,
    "window_kind": "wt | console",
    "tab_title": "example-app voice hub"
  }
}
```

`session_id` comes from `CLAUDE_CODE_SESSION_ID` and `claude_pid` from
`CLAUDE_PID`, both inherited from the environment: exact identity, for free.
`nav` is `null` when the session has no window of its own (desktop app,
background job), which is the majority case today.

- `say.py` emits `queued` when it starts waiting on the speech lock, `speaking`
  when it acquires it, `done` when it releases it. The queue that exists today
  invisibly inside the lock becomes visible.
- `nav` is derived by walking up the parent-process chain looking for a process
  with a visible top-level window (`WindowsTerminal.exe`, or
  `cmd.exe`/`conhost.exe`). For WT, `GetConsoleTitleW()` is also stored as the
  tab title.
- The hub deletes consumed events; any event older than one hour is discarded
  (spool hygiene).

### Session registry (enrichment)

The hub runs `claude agents --json` in the background (cached, refreshed every
~10 s) and joins on `sessionId` to show the session's **human-readable name**
(e.g. `example-app-96`, `voice hub, orb renderer`), its `cwd`, and whether
it is `interactive` or `background`. If the command fails or is slow, the hub
keeps working from the spoken tag: enrichment is never blocking.

## Hub UI

- Main row: 🔊 + the current speaker's tag (equivalent to today's pill).
- Below it: the pending queue (⏳ + tag, in order) and a history of the last 5
  speakers (dimmed, with elapsed minutes).
- Every row that has `nav` is clickable and navigates (see next section). Hand
  cursor and hover state make it obvious what is clickable.
- With no activity (nobody speaking, empty queue) it shrinks to a discreet
  mini-pill showing the last speaker; after 10 minutes idle the process exits.
- Phase B adds a 🎤 button to the hub row.
- Styling: the same palette as the current pill (`#1e1e2e` / `#cdd6f4` /
  `#94e2d5`).

## Click navigation (adaptive action)

A click performs the best available action for that session, in this order:

1. **The session has its own window** (a `cmd.exe`/WT window whose process is an
   ancestor of the session `pid`): `SetForegroundWindow(hwnd)` with the
   foreground-lock workaround (press ALT immediately before the call). If it is
   Windows Terminal, the tab is additionally selected by title via UI Automation
   (`TabItemControl` + the SelectionItem pattern). A complete, frictionless jump.
2. **The session has no window of its own** (desktop app, background job):
   `claude --resume <sessionId>` is copied to the clipboard and the Claude app
   window is focused. The hub row confirms "command copied"; you paste and land
   in that exact session. This is the best jump available given that no API
   exists to focus a specific session.
3. **Nothing resolvable**: the row is not clickable and is shown dimmed.

Step 1 remains the ideal path and applies automatically if sessions are moved
into their own windows or tabs. Step 2 is what covers the common setup today.

## Push-to-talk STT (Phase B)

- Trigger: hold the hub's 🎤 button, or the global hotkey Ctrl+Alt+Space
  (`RegisterHotKey` from the hub; no low-level keyboard hooks if `RegisterHotKey`
  suffices).
- Default target: the last session that spoke with a valid `nav`; clicking a row
  before recording changes the target.
- Recording via `sounddevice` while the button is held.
- Transcription: ElevenLabs Scribe (same API key as the TTS path); local
  faster-whisper on GPU as a fallback if Scribe fails or there is no network.
- Delivery, **adaptive for safety** (post-spike revision):
  - If the target session has its own window (navigation case 1): focus it and
    type with `SendInput` + Enter. The destination is unambiguous.
  - If it does not (desktop app, background job): **copy to the clipboard** and
    focus the app; you paste where it belongs. Typing blind into the app would
    send the text to whichever session happens to be open, which may not be the
    one that asked. That risk is not accepted.
- Guardrail: a 2 s preview of the transcribed text in the hub before delivery;
  Esc cancels.

## Failure modes and degradation

| Failure | Behavior |
| --- | --- |
| Hub fails to start | `say.py` uses the current ephemeral pill (code is kept) |
| Session with no terminal (bg job) | Row visible, dimmed, not clickable |
| UIA cannot find the tab | Foreground the WT window only |
| STT fails / no network | Notice in the hub + beep; nothing is typed |
| Hub crashes mid-run | Stale lockfile (>3 min) is cleaned and the hub relaunched |
| Corrupt spool | Unreadable event is deleted and ignored |

## Validation

- Phase A: the UIA spike is step 1 of the plan. After that, a smoke script that
  launches 3 `say.py` calls in parallel with distinct tags and verifies: queue
  visible and in order, correct history, click navigates to the right tab. Unit
  tests for event parsing and `nav` derivation.
- Phase B: an end-to-end test with a real microphone against a test session;
  verify the preview/cancel guardrail.

## Changes to the voice skill

In the accompanying voice skill, the "On-screen overlay" section is rewritten to
describe the hub (queue + history + click to navigate). How you speak does not
change at all: same CLI, same `#preset=` headers, same mandatory leading tag
(which now also feeds navigation). One new documented flag: `--no-hub` to force
the classic pill.

## Explicitly out of scope

- A distinct ElevenLabs voice per session (the aliases already exist; decide
  after using the hub for a few days).
- Mouse-free hotkey navigation.
- Support for terminals other than Windows Terminal.

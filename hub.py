"""Voice Hub: persistent always-on-top panel showing which Claude Code session
is speaking, what is queued behind it, and who spoke recently.

Each row is clickable and performs the best available jump to that session
(see nav.go_to_session). Singleton via hub/hub.lock; exits after IDLE_EXIT_S
of silence. Launched automatically by say.py; safe to run by hand.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents_registry
import hub_events
import nav

BG, FG, ACCENT = "#11111b", "#cdd6f4", "#a6e3a1"
DIM, SUB, LINE, BADGE = "#585b70", "#7f849c", "#313244", "#1e1e2e"


def _enable_dpi_awareness() -> float:
    """Windows scales a non-DPI-aware process by stretching its bitmap: on a 4K
    screen at 150% every glyph is upscaled and blurry. pythonw.exe is not aware
    by default, so declare it here (before Tk exists) and return the factor so
    pixel paddings keep their physical size."""
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor v2
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return max(1.0, dpi / 96.0)
    except Exception:
        return 1.0


SCALE = _enable_dpi_awareness()


def px(n: float) -> int:
    """A pixel measurement (padding, thickness) in real device pixels."""
    return max(1, int(round(n * SCALE)))


def _clip(text: str, limit: int) -> str:
    """Bound a row's width. A session named after a whole prompt would otherwise
    stretch the panel past the screen and spill outside its own border."""
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pick_mono(root) -> str:
    """The console font. Everything in the hub is machine output — session ids,
    ages, queue positions — and reads better fixed-width."""
    from tkinter import font as tkfont
    families = set(tkfont.families(root))
    for name in ("Cascadia Mono", "Consolas", "Courier New"):
        if name in families:
            return name
    return "TkFixedFont"



POLL_MS = 250
IDLE_EXIT_S = 600
HISTORY_MAX = 5
STATUS_HOLD_S = 3.0
PREVIEW_S = 2000  # ms the transcription is shown before delivery (Esc cancels)

# The hub runs under pythonw with no console, so an unlogged exception is simply
# invisible: dictation failed for hours with no message anywhere. Everything
# interesting goes to this file.
LOG_PATH = hub_events.HUB_DIR / "hub.log"


def log(msg: str) -> None:
    try:
        hub_events.HUB_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{stamp}  {msg}\n")
    except Exception:
        pass


GEOM_PATH = Path(__file__).resolve().parent / "hub_geometry.json"
MIN_W, MIN_H = 240, 120


def _load_geom() -> dict:
    """Where the user last put the panel. Persisted because the hub exits when
    idle and is relaunched by the next utterance — without this, moving it would
    only last until the next silence.

    Coordinates written before we were DPI-aware are in virtualized units, so a
    file with no matching `scale` is discarded rather than placing the panel in
    the wrong spot at the wrong size."""
    try:
        import json
        g = json.loads(GEOM_PATH.read_text(encoding="utf-8"))
        return g if abs(g.get("scale", 0) - SCALE) < 0.01 else {}
    except Exception:
        return {}


def _save_geom(d: dict) -> None:
    try:
        import json
        GEOM_PATH.write_text(json.dumps({**d, "scale": SCALE}), encoding="utf-8")
    except Exception:
        pass


class VoiceHub:
    def __init__(self) -> None:
        self.root = tk.Tk()
        # Point-sized fonts are converted through this factor, so declaring it
        # keeps the panel physically the same size now that we draw at real
        # pixels instead of being stretched.
        self.root.tk.call("tk", "scaling", SCALE * 96.0 / 72.0)
        self.mono = _pick_mono(self.root)
        self.root.overrideredirect(True)
        try:
            self.root.attributes("-topmost", True)
            # Alpha blends the glyphs against the desktop and costs contrast;
            # keep it just shy of opaque.
            self.root.attributes("-alpha", 0.97)
        except Exception:
            pass
        self.frame = tk.Frame(self.root, bg=BG, padx=px(16), pady=px(12),
                              highlightthickness=px(1), highlightbackground=LINE)
        self.frame.pack(fill="both", expand=True)
        self.geom = _load_geom()
        self._drag: tuple | None = None
        # Drag from the panel's own background. Bound here rather than on the
        # toplevel because Tk delivers to the innermost widget: the session rows
        # keep their click-to-jump behaviour and only empty space drags.
        self.frame.bind("<Button-1>", self._move_start)
        self.frame.bind("<B1-Motion>", self._move_drag)
        self.frame.bind("<ButtonRelease-1>", self._drop)
        self.current: dict | None = None
        self.queue: list[dict] = []
        self.history: deque[tuple[dict, float]] = deque(maxlen=HISTORY_MAX)
        self.last_activity = time.time()
        self.status = ""
        self.status_until = 0.0
        self.recorder = None
        self.stt_target: dict | None = None
        self.hotkey_label = ""
        self._render()
        self._tick()
        self._register_hotkey()
        self._warm_microphone()

    # ----------------------------- state -----------------------------------
    def _key(self, evt: dict) -> str:
        """Identify the speaker: session id when known, else the say.py pid."""
        return evt.get("session_id") or f"pid:{evt.get('pid')}"

    def _apply(self, evt: dict) -> None:
        state, key = evt.get("state"), self._key(evt)
        if state == "queued":
            self.queue.append(evt)
        elif state == "speaking":
            self.queue = [q for q in self.queue if self._key(q) != key]
            if self.current and self._key(self.current) != key:
                self.history.appendleft((self.current, time.time()))
            self.current = evt
        elif state == "listening":
            # The session finished asking and is now waiting on a spoken answer.
            # It still holds its turn, so it stays in the hero row: what matters
            # is that you can see WHO is waiting on you, and go there.
            self.current = evt
        elif state == "done":
            if self.current and self._key(self.current) == key:
                self.history.appendleft((self.current, time.time()))
                self.current = None
            self.queue = [q for q in self.queue if self._key(q) != key]
        self.last_activity = time.time()

    def _tick(self) -> None:
        evts = hub_events.read_events()
        if evts:
            for e in evts:
                self._apply(e)
            threading.Thread(target=agents_registry.refresh, daemon=True).start()
            self._render()
        elif self.status and time.time() > self.status_until:
            self.status = ""
            self._render()
        if time.time() - self.last_activity > IDLE_EXIT_S:
            self.root.destroy()
            return
        self.root.after(POLL_MS, self._tick)

    # ----------------------------- render ----------------------------------
    def _identity(self, evt: dict) -> tuple[str, str]:
        """(headline, subtitle). The session's real name leads when we know it,
        since that is what you recognize; the spoken tag drops to the subtitle."""
        tag = evt.get("tag", "?")
        name, project = agents_registry.describe(evt.get("session_id", ""),
                                                 evt.get("cwd", ""))
        if name:
            sub = "  ·  ".join(b for b in (tag, project) if b)
            return _clip(name, 30), _clip(sub, 44)
        return _clip(tag, 30), _clip(project, 44)

    def _clickable(self, widgets, evt: dict | None) -> None:
        if evt is None or not (evt.get("nav") or evt.get("session_id")):
            return
        for w in widgets:
            w.configure(cursor="hand2")
            w.bind("<Button-1>", lambda _e, ev=evt: self._on_row_click(ev))

    def _caption(self, text: str, pady=(0, 3)) -> None:
        tk.Label(self.frame, text=text, font=(self.mono, 7, "bold"),
                 bg=BG, fg=DIM).pack(anchor="w",
                                     pady=(px(pady[0]), px(pady[1])))

    def _hero(self, evt: dict) -> None:
        headline, sub = self._identity(evt)
        # A session waiting on a spoken answer is not the same as one talking at
        # you: it is blocked until you say something. The mic glyph is the
        # difference between "someone is talking" and "someone needs you".
        waiting = evt.get("state") == "listening"
        row = tk.Frame(self.frame, bg=BG)
        row.pack(anchor="w", fill="x")
        icon = tk.Label(row, text="\U0001F3A4" if waiting else "\U0001F50A",
                        font=("Segoe UI Emoji", 13), bg=BG, fg=ACCENT)
        icon.pack(side="left")
        if waiting:
            sub = "esperando tu respuesta"
        col = tk.Frame(row, bg=BG)
        col.pack(side="left", padx=px(8))
        top = tk.Label(col, text=headline, font=(self.mono, 11, "bold"), bg=BG, fg=ACCENT)
        top.pack(anchor="w")
        bottom = tk.Label(col, text=sub, font=(self.mono, 8), bg=BG, fg=SUB)
        bottom.pack(anchor="w")
        self._clickable((row, col, icon, top, bottom), evt)

    def _queue_row(self, evt: dict, position: int) -> None:
        headline, sub = self._identity(evt)
        row = tk.Frame(self.frame, bg=BG)
        row.pack(anchor="w", fill="x", pady=px(1))
        badge = tk.Label(row, text=str(position), font=(self.mono, 9, "bold"),
                         bg=BADGE, fg=SUB, padx=px(5))
        badge.pack(side="left")
        text = f"{headline}  ·  {sub}" if sub else headline
        lbl = tk.Label(row, text=text, font=(self.mono, 9), bg=BG, fg=FG, padx=px(7))
        lbl.pack(side="left")
        self._clickable((row, badge, lbl), evt)

    def _history_row(self, evt: dict, ts: float) -> None:
        headline, sub = self._identity(evt)
        mins = max(0, int((time.time() - ts) / 60))
        row = tk.Frame(self.frame, bg=BG)
        row.pack(anchor="w", fill="x", pady=px(1))
        age = tk.Label(row, text=f"{mins}m".rjust(3), font=(self.mono, 8), bg=BG, fg=DIM)
        age.pack(side="left")
        text = f"{headline}  ·  {sub}" if sub else headline
        lbl = tk.Label(row, text=text, font=(self.mono, 9), bg=BG, fg=SUB, padx=px(8))
        lbl.pack(side="left")
        self._clickable((row, age, lbl), evt)

    def _render(self) -> None:
        for w in self.frame.winfo_children():
            w.destroy()
        if self.current:
            self._caption("HABLANDO")
            self._hero(self.current)
        if self.queue:
            self._caption("EN COLA", pady=(10 if self.current else 0, 3))
            for i, q in enumerate(self.queue, start=1):
                self._queue_row(q, i)
        if self.history:
            self._caption("RECIENTES", pady=(10 if (self.current or self.queue) else 0, 3))
            for evt, ts in self.history:
                self._history_row(evt, ts)
        if not self.current and not self.queue and not self.history:
            tk.Label(self.frame, text="voice hub", font=(self.mono, 9),
                     bg=BG, fg=DIM).pack(anchor="w")
        tk.Frame(self.frame, bg=LINE, height=px(1)).pack(fill="x",
                                                         pady=(px(10), px(7)))
        hint = f"   ·   {self.hotkey_label}" if self.hotkey_label else ""
        if self.recorder is not None:
            target = self._identity(self.stt_target)[0] if self.stt_target else "?"
            label, fg = f"recording -> {target}   ·   click to send{hint}", ACCENT
        else:
            label, fg = f"dictate to the last session{hint}", SUB
        # Always clickable: without it, a taken hotkey leaves no way to stop.
        mic = tk.Label(self.frame, text=f"\U0001F3A4  {label}", font=(self.mono, 8),
                       bg=BG, fg=fg, cursor="hand2")
        mic.pack(anchor="w")
        mic.bind("<Button-1>", lambda _e: self._toggle_recording())
        if self.status:
            tk.Label(self.frame, text=self.status, font=(self.mono, 9),
                     bg=BG, fg=ACCENT, padx=px(6)).pack(anchor="w")
        # Resize grip. The window is overrideredirect, so it has no native
        # border to drag — this is the handle.
        grip_cursor = "size_nw_se" if sys.platform == "win32" else "bottom_right_corner"
        grip = tk.Label(self.frame, text="◢", font=(self.mono, 9),
                        bg=BG, fg=DIM, cursor=grip_cursor)
        grip.pack(anchor="e", pady=(px(4), 0))
        grip.bind("<Button-1>", self._size_start)
        grip.bind("<B1-Motion>", self._size_drag)
        grip.bind("<ButtonRelease-1>", self._drop)

        self.root.update_idletasks()
        self._place()

    # ----------------------------- geometry --------------------------------
    def _place(self) -> None:
        """Honour where the user dragged it; fall back to the bottom-right."""
        w, h = self.root.winfo_width(), self.root.winfo_height()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        g = self.geom
        if "x" in g and "y" in g:
            x = max(0, min(int(g["x"]), sw - px(80)))
            y = max(0, min(int(g["y"]), sh - px(60)))
            if "w" in g and "h" in g:
                self.root.geometry(f"{int(g['w'])}x{int(g['h'])}+{x}+{y}")
            else:
                self.root.geometry(f"+{x}+{y}")
        else:
            self.root.geometry(f"+{sw - w - px(24)}+{sh - h - px(90)}")

    def _move_start(self, e) -> None:
        self._drag = ("move", e.x_root, e.y_root,
                      self.root.winfo_x(), self.root.winfo_y())

    def _move_drag(self, e) -> None:
        if not self._drag or self._drag[0] != "move":
            return
        _, sx, sy, ox, oy = self._drag
        self.geom["x"] = ox + (e.x_root - sx)
        self.geom["y"] = oy + (e.y_root - sy)
        self.root.geometry(f"+{int(self.geom['x'])}+{int(self.geom['y'])}")

    def _size_start(self, e) -> None:
        self._drag = ("size", e.x_root, e.y_root,
                      self.root.winfo_width(), self.root.winfo_height())

    def _size_drag(self, e) -> None:
        if not self._drag or self._drag[0] != "size":
            return
        _, sx, sy, ow, oh = self._drag
        self.geom["w"] = max(px(MIN_W), ow + (e.x_root - sx))
        self.geom["h"] = max(px(MIN_H), oh + (e.y_root - sy))
        self.geom.setdefault("x", self.root.winfo_x())
        self.geom.setdefault("y", self.root.winfo_y())
        self.root.geometry(f"{int(self.geom['w'])}x{int(self.geom['h'])}"
                           f"+{int(self.geom['x'])}+{int(self.geom['y'])}")

    def _drop(self, _e=None) -> None:
        if self._drag:
            self._drag = None
            _save_geom(self.geom)

    # ----------------------------- actions ---------------------------------
    def _on_row_click(self, evt: dict) -> None:
        def work() -> None:
            status = nav.go_to_session(evt)  # UIA/foreground can take ~1s
            self.root.after(0, lambda: self._flash(status or "sin destino"))

        threading.Thread(target=work, daemon=True).start()

    # ----------------------------- push to talk ----------------------------
    def _warm_microphone(self) -> None:
        """Resolve the input device now, in the background.

        Probing costs ~2s across the candidate mics. Paying it on the first
        hotkey press would eat the opening words, so do it at startup.
        """
        def work() -> None:
            try:
                import stt
                log(f"mic: warmed to device {stt.pick_input_device()}")
            except Exception as e:
                log(f"mic: warm failed: {type(e).__name__}: {e}")

        threading.Thread(target=work, daemon=True).start()

    def _register_hotkey(self) -> None:
        """Bind the first dictation hotkey that Windows will give us.

        You may run other dictation software (e.g. Wispr Flow) that already owns
        Ctrl+Alt+Space: RegisterHotKey then fails with 1409 and the shortcut
        silently does nothing. So try candidates in order and show the one that
        actually bound, or say 'click' when every candidate is taken.

        RegisterHotKey/GetMessageW are Win32-only; there is no global-hotkey
        equivalent wired up here yet, so this is a no-op elsewhere (click still
        works, see the mic label handler in _render).
        """
        if sys.platform != "win32":
            self.hotkey_label = ""
            return
        import ctypes.wintypes

        MOD_ALT, MOD_CONTROL, MOD_SHIFT = 0x1, 0x2, 0x4
        VK_SPACE = 0x20
        CANDIDATES = [
            (MOD_CONTROL | MOD_ALT, ord("D"), "Ctrl+Alt+D"),
            (MOD_CONTROL | MOD_SHIFT, VK_SPACE, "Ctrl+Shift+Space"),
            (MOD_CONTROL | MOD_ALT, ord("V"), "Ctrl+Alt+V"),
            (MOD_CONTROL | MOD_ALT, VK_SPACE, "Ctrl+Alt+Space"),
        ]

        def loop() -> None:
            user32 = ctypes.windll.user32
            WM_HOTKEY = 0x0312
            for i, (mods, vk, label) in enumerate(CANDIDATES):
                if user32.RegisterHotKey(None, i + 1, mods, vk):
                    self.hotkey_label = label
                    log(f"hotkey: bound {label}")
                    break
            else:
                self.hotkey_label = ""
                log("hotkey: every candidate taken; click only")
                self.root.after(0, self._render)
                return
            self.root.after(0, self._render)
            msg = ctypes.wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == WM_HOTKEY:
                    self.root.after(0, self._toggle_recording)

        threading.Thread(target=loop, daemon=True).start()

    def _toggle_recording(self) -> None:
        if self.recorder is None:
            self._start_recording()
        else:
            self._finish_recording()

    def _default_target(self) -> dict | None:
        """Reply to whoever spoke last and can still be reached."""
        candidates = ([self.current] if self.current else []) + [h[0] for h in self.history]
        for evt in candidates:
            if evt and (evt.get("nav") or evt.get("session_id")):
                return evt
        return None

    def _start_recording(self) -> None:
        import stt
        target = self.stt_target or self._default_target()
        if target is None:
            self._flash("nadie a quien responder todavia")
            log("record: no target session")
            return
        try:
            rec = stt.Recorder()
            rec.start()
        except Exception as e:
            self._flash(f"mic no disponible: {type(e).__name__}")
            log(f"record: start failed: {type(e).__name__}: {e}")
            return
        self.recorder = rec
        self.stt_target = target
        log(f"record: started on device {rec.device} -> {target.get('tag')}")
        self._flash("")

    def _finish_recording(self) -> None:
        rec, self.recorder = self.recorder, None
        self._flash("transcribiendo...")

        def work() -> None:
            import stt
            try:
                wav = rec.stop()
                log(f"record: stopped, level={rec.level:.0f}, {wav.stat().st_size}b")
            except Exception as e:
                log(f"record: rejected: {e}")
                self.root.after(0, lambda: self._flash(str(e)[:70]))
                return
            try:
                text = stt.transcribe(wav)
            except Exception as e:
                log(f"transcribe: failed: {e}")
                self.root.after(0, lambda: self._flash(str(e)[:70]))
                return
            log(f"transcribe: {text!r}")
            self.root.after(0, lambda: self._preview_and_send(text))

        threading.Thread(target=work, daemon=True).start()

    def _preview_and_send(self, text: str) -> None:
        """Show the transcription for PREVIEW_S; Esc cancels, then deliver."""
        target = self.stt_target or {}
        top = tk.Toplevel(self.root)
        top.overrideredirect(True)
        try:
            top.attributes("-topmost", True)
        except Exception:
            pass
        tk.Label(top, text=f"-> {target.get('tag', '?')}\n{text}\n(Esc cancela)",
                 bg=BG, fg=FG, font=(self.mono, 10), padx=px(14), pady=px(10),
                 justify="left", wraplength=px(520)).pack()
        top.update_idletasks()
        sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
        top.geometry(f"+{sw - top.winfo_width() - px(24)}"
                     f"+{sh - top.winfo_height() - px(300)}")
        cancelled: list[int] = []
        top.bind("<Escape>", lambda _e: (cancelled.append(1), top.destroy()))
        top.focus_force()

        def send() -> None:
            if not top.winfo_exists():
                return
            top.destroy()
            # Reset the target either way. It is chosen per dictation, and a
            # target left stuck here would silently send the next answer to a
            # session that is no longer the one asking.
            self.stt_target = None
            if cancelled:
                self._flash("dictation cancelled")
                log("deliver: cancelled by user")
                return
            status = nav.deliver_text(target, text)
            log(f"deliver: {status}")
            self._flash(status)

        top.after(PREVIEW_S, send)

    def _flash(self, status: str) -> None:
        self.status = status
        self.status_until = time.time() + STATUS_HOLD_S
        self.last_activity = time.time()
        self._render()

    def run(self) -> None:
        self.root.mainloop()


def acquire_singleton() -> bool:
    hub_events.HUB_DIR.mkdir(parents=True, exist_ok=True)
    lock = hub_events.HUB_LOCK
    if lock.exists():
        try:
            import psutil
            if psutil.pid_exists(int(lock.read_text().strip())):
                return False
        except Exception:
            pass
        lock.unlink(missing_ok=True)
    lock.write_text(str(os.getpid()))
    return True


def main() -> None:
    if not acquire_singleton():
        return
    try:
        VoiceHub().run()
    finally:
        try:
            hub_events.HUB_LOCK.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()

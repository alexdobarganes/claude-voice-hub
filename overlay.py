"""Speaking-orb overlay: a true per-pixel-alpha Win32 layered window.

Why not tkinter: tkinter can only fake transparency with `-transparentcolor`,
which color-keys ONE exact color. A glowing orb fades to near-black, so every
dim glow pixel stays opaque and the whole thing reads as a dark rectangle with
fringed edges. Windows' own answer for this is a layered window updated via
UpdateLayeredWindow() with a premultiplied-BGRA bitmap: real 8-bit alpha per
pixel, composited by the DWM. Same mechanism pro overlays use.

Also WS_EX_TRANSPARENT (click-through), WS_EX_NOACTIVATE (never steals focus)
and WS_EX_TOOLWINDOW (no taskbar entry).

Usage (spawned by say.py):
    pythonw overlay.py "Session label"

Audio reactivity — say.py drives us over stdin, one command per line:
    ENV3 <hop> <delay> <lo_csv>|<mid_csv>|<hi_csv>
        Per-band envelopes for the whole clip. `delay` is how many seconds from
        now the first sample becomes audible, which is what keeps the animation
        locked to the voice: the audio device buffers a large and very
        device-dependent amount (~180 ms on the Focusrite here), so treating
        "message received" as "sound started" runs the animation early.
    LVL3 <lo> <mid> <hi>
        Instantaneous per-band level, for the streaming path where there is no
        complete clip to analyze up front.
Anything unparseable is ignored. Closing the pipe fades out and exits; there is
also a hard 3-minute cap so a crashed parent can never leave it on screen.
"""
from __future__ import annotations

import ctypes
import math
import os
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orb_render import Orb  # noqa: E402


def _make_renderer(kind: str):
    """Pick the visual. Every renderer exposes the same (t, bands, angle) ->
    premultiplied BGRA interface, so the loop doesn't care which is running.

    Anything under experiments/ is optional by design: if the import fails the
    CPU orb takes over, so a missing optional dependency degrades the visual
    instead of breaking speech."""
    if kind in ("neuro", "neuron"):
        # An agent drawn as a neuron: it fires on the syllable and the pulse
        # travels out through the arbor. Selectable, not the default — the orb
        # stays the everyday visual.
        try:
            from experiments.neuron_render import Neuron
            return Neuron()
        except Exception as e:
            print(f"[overlay] neuron unavailable ({e}); using orb", file=sys.stderr)
    if kind in ("orb", "orb-gpu") and os.environ.get("TTS_ORB_CPU") != "1":
        # The simulated orb: curl-noise flow, harmonic modes, depth of field,
        # trails. Needs CUDA; the CPU orb is the fallback and is still a good
        # sphere, just a less alive one.
        try:
            from experiments.orb_gpu import OrbGPU
            return OrbGPU()
        except Exception as e:
            print(f"[overlay] gpu orb unavailable ({e}); using cpu orb", file=sys.stderr)
    return Orb()

FPS = 60
# Envelope smoothing as time constants, not per-frame factors, so the motion
# looks the same at any frame rate. Tuned at 30 fps as 0.55/0.16 per frame.
ATTACK_TAU = 0.042     # seconds to ~63% toward a louder target
RELEASE_TAU = 0.191    # slower decay, so tails fall off instead of snapping
FADE_MS = 260          # fade in/out so it never pops on/off
# Slightly translucent so it sits *in* the desktop rather than on top of it.
OPACITY = float(os.environ.get("TTS_OVERLAY_OPACITY", 0.92))
TOP_MARGIN = 12
# Sized for a 4K panel. The overlay declares itself DPI-aware, so these are
# real device pixels and not logical ones: a 19 px caption that looked fine on
# a scaled desktop is a whisper on a 3840-wide screen.
CAPTION_H = int(os.environ.get("TTS_CAPTION_H", 132))
CAP_FONT = int(os.environ.get("TTS_CAPTION_FONT", 40))
CAP_HOLD = 0.55        # seconds a spoken word lingers before it fades away
HARD_CAP_S = 180.0

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32
winmm = ctypes.windll.winmm
user32.SetProcessDPIAware()

LRESULT = ctypes.c_longlong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_long
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)

# Declare signatures explicitly. Without this ctypes assumes c_int, which both
# breaks the WNDPROC callback (LPARAM overflow) and — worse — TRUNCATES 64-bit
# HWND/HDC/HBITMAP handles to 32 bits, producing handles that silently fail.
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                  wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.CreateWindowExW.restype = wintypes.HWND
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                wintypes.UINT]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000
ULW_ALPHA = 0x02
AC_SRC_OVER, AC_SRC_ALPHA = 0x00, 0x01
SW_SHOWNOACTIVATE = 4
HWND_TOPMOST = -1
SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte),
                ("SourceConstantAlpha", ctypes.c_byte), ("AlphaFormat", ctypes.c_byte)]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND, wintypes.HDC, ctypes.POINTER(POINT), ctypes.POINTER(SIZE),
    wintypes.HDC, ctypes.POINTER(POINT), wintypes.COLORREF,
    ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
gdi32.StretchBlt.argtypes = [
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.DWORD]
gdi32.StretchBlt.restype = wintypes.BOOL
gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, ctypes.c_int]


SRCCOPY = 0x00CC0020


def screen_luma(x: int, y: int, w: int, h: int) -> float:
    """Mean brightness (0..1) of the desktop under a rectangle.

    Called while we are still invisible — which is exactly why the visual waits
    for audio before showing: at that moment the grab sees the desktop and not
    our own pixels. Used to decide how much backdrop the visual needs, because
    neon dots on transparency vanish against a white page.
    """
    try:
        sdc = user32.GetDC(None)
        mdc = gdi32.CreateCompatibleDC(sdc)
        sw, sh = max(1, w // 16), max(1, h // 16)     # a thumbnail is plenty
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = sw
        bmi.bmiHeader.biHeight = -sh
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0
        bits = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(mdc, ctypes.byref(bmi), 0,
                                      ctypes.byref(bits), None, 0)
        old = gdi32.SelectObject(mdc, hbmp)
        gdi32.SetStretchBltMode(mdc, 4)               # HALFTONE
        gdi32.StretchBlt(mdc, 0, 0, sw, sh, sdc, x, y, w, h, SRCCOPY)
        buf = (ctypes.c_ubyte * (sw * sh * 4)).from_address(bits.value)
        arr = np.frombuffer(bytes(buf), np.uint8).reshape(sh, sw, 4)[:, :, :3]
        val = float(arr.mean()) / 255.0
        # Mean colour too, so a visual can tint itself with the light of
        # whatever is behind it. Saturation is pushed up first: averaging a
        # screen mostly washes out to grey, and a grey "ambient" tints nothing.
        c = arr.reshape(-1, 3).mean(0)
        c = np.clip(c.mean() + (c - c.mean()) * 2.6, 0, 255)
        screen_luma.last_color = tuple(float(v) for v in c)
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(None, sdc)
        return val
    except Exception as e:
        print(f"[overlay] could not sample the desktop: {e}", file=sys.stderr)
        screen_luma.last_color = (128.0, 128.0, 128.0)
        return 0.0


def capture_region(x: int, y: int, w: int, h: int) -> np.ndarray | None:
    """Grab the desktop behind us at full resolution, as BGR uint8.

    Taken once, in the instant before the visual appears and while the window
    is still fully transparent — so it sees the desktop rather than our own
    pixels. That is also the honest limitation: it is a still. Anything moving
    behind the orb (a video, a scrolling page) refracts as it looked at the
    moment we started speaking. Re-grabbing mid-flight would need the window
    hidden for the grab, which flickers, so a still it is.
    """
    try:
        sdc = user32.GetDC(None)
        mdc = gdi32.CreateCompatibleDC(sdc)
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = 0
        bits = ctypes.c_void_p()
        hbmp = gdi32.CreateDIBSection(mdc, ctypes.byref(bmi), 0,
                                      ctypes.byref(bits), None, 0)
        old = gdi32.SelectObject(mdc, hbmp)
        gdi32.StretchBlt(mdc, 0, 0, w, h, sdc, x, y, w, h, SRCCOPY)
        buf = (ctypes.c_ubyte * (w * h * 4)).from_address(bits.value)
        arr = np.frombuffer(bytes(buf), np.uint8).reshape(h, w, 4)[:, :, :3].copy()
        gdi32.SelectObject(mdc, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mdc)
        user32.ReleaseDC(None, sdc)
        return arr
    except Exception as e:
        print(f"[overlay] could not capture the desktop: {e}", file=sys.stderr)
        return None


def build_scrim(w: int, h: int, strength: float) -> np.ndarray:
    """A soft dark pool for the visual to sit on, premultiplied BGRA.

    Radial and edge-free on purpose: a rectangle of tint would read as a box
    pasted on the screen, which is the thing this overlay works hardest to
    avoid. Strength is driven by how bright the desktop underneath is, so on a
    dark background it stays at zero and nothing changes.
    """
    if strength <= 0.01:
        return np.zeros((h, w, 4), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w * 0.52)) ** 2 + ((yy - h / 2) / (h * 0.52)) ** 2)
    a = np.clip(1.0 - r, 0.0, 1.0) ** 1.6 * strength
    out = np.zeros((h, w, 4), np.uint8)
    out[:, :, 3] = (a * 255).astype(np.uint8)     # black, so RGB stays 0
    return out


class Caption:
    """The word being spoken, drawn under the orb.

    Only the current word is lit. Words that have been said fade out and go;
    words not yet said are not drawn at all. So the caption is never a wall of
    text to read — it is the voice, in step.

    Each word is rasterised ONCE into its own tile and then composited with a
    per-word alpha every frame. Re-rendering the whole strip per frame would
    mean laying out text sixty times a second, and caching whole strips does
    not work either once every word carries its own continuously changing fade.
    """

    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        self.words: list = []
        self.tiles: list = []          # (x, y, RGBA tile) per word
        self.t0 = 0.0
        self._blank = np.zeros((height, width, 4), np.uint8)
        self._buf = np.zeros((height, width, 4), np.uint16)

    def set_words(self, words: list, t0: float) -> None:
        self.words, self.t0, self.tiles = words, t0, []
        f = _mono(CAP_FONT)
        probe = ImageDraw.Draw(Image.new("L", (8, 8)))
        limit = self.w - 40
        # Lay the lines out first: a word has to sit where it would sit in the
        # finished line, or the caption jitters as it reveals.
        lines, cur, cur_w = [], [], 0
        for i, (_s, _e, word) in enumerate(words):
            wpx = probe.textbbox((0, 0), word + " ", font=f)[2]
            if cur and cur_w + wpx > limit:
                lines.append(cur)
                cur, cur_w = [], 0
            cur.append((i, word, wpx))
            cur_w += wpx
        if cur:
            lines.append(cur)

        y = (self.h - CAP_FONT) // 2 - 2
        self.line_of = {}
        for ln, line in enumerate(lines):
            x = (self.w - sum(w for _, _, w in line)) // 2
            for idx, word, wpx in line:
                img = Image.new("RGBA", (wpx + 8, CAP_FONT + 14), (0, 0, 0, 0))
                ImageDraw.Draw(img).text((0, 0), word, font=f,
                                         fill=(255, 255, 255, 255))
                self.tiles.append((x, y, np.asarray(img, np.uint8)))
                self.line_of[idx] = ln
                x += wpx

    def index_at(self, now: float) -> int:
        t = now - self.t0
        idx = -1
        for i, (s, _e, _w) in enumerate(self.words):
            if s <= t:
                idx = i
            else:
                break
        return idx

    def strip(self, now: float) -> np.ndarray:
        if not self.words:
            return self._blank
        t = now - self.t0
        idx = self.index_at(now)
        if idx < 0:
            return self._blank
        buf = self._buf
        buf.fill(0)
        drew = False
        for i, (s, e, _w) in enumerate(self.words):
            if i > idx or self.line_of.get(i) != self.line_of.get(idx):
                continue                      # unsaid, or on another line
            if i == idx:
                # Fade in over the word's own opening, so it arrives with the
                # sound rather than snapping on.
                a = min(1.0, max(0.0, (t - s) / 0.07 + 0.35))
            else:
                gone = t - e
                if gone > CAP_HOLD:
                    continue                  # said and finished: hidden
                a = max(0.0, 1.0 - gone / CAP_HOLD) ** 1.5
            if a <= 0.02:
                continue
            x, y, tile = self.tiles[i]
            th, tw = tile.shape[:2]
            if y + th > self.h or x + tw > self.w or x < 0:
                continue
            alpha = (tile[:, :, 3].astype(np.uint16) * int(a * 255)) // 255
            region = buf[y:y + th, x:x + tw]
            region[:, :, 0] = np.maximum(region[:, :, 0], alpha)   # premultiplied
            region[:, :, 1] = np.maximum(region[:, :, 1], alpha)
            region[:, :, 2] = np.maximum(region[:, :, 2], alpha)
            region[:, :, 3] = np.maximum(region[:, :, 3], alpha)
            drew = True
        return np.minimum(buf, 255).astype(np.uint8) if drew else self._blank


def _font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
    except Exception:
        return ImageFont.load_default()


def _mono(size: int):
    """A console face for the caption. Fixed-width reads as machine output,
    which is what this is, and the heavier stems hold up better against a busy
    desktop than a UI face at the same size."""
    for name in ("CascadiaMono.ttf", "consola.ttf", "CascadiaCode.ttf",
                 "lucon.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
        except Exception:
            continue
    return _font(size)


def build_label_layer(label: str, width: int, height: int, y: int) -> np.ndarray:
    """Render the session chip as its own premultiplied-BGRA layer."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = _font(15, bold=True)
    tw = d.textbbox((0, 0), label, font=f)[2]
    pad_x, pad_y, h = 15, 8, 32
    x0 = (width - (tw + pad_x * 2)) // 2
    box = (x0, y, x0 + tw + pad_x * 2, y + h)
    # Dark glass chip with a soft neon edge — reads on any wallpaper.
    d.rounded_rectangle(box, radius=h // 2, fill=(18, 16, 32, 205),
                        outline=(150, 120, 255, 170), width=1)
    d.text((x0 + pad_x, y + pad_y - 1), label, font=f, fill=(232, 228, 255, 255))

    arr = np.asarray(img, dtype=np.float32)
    a = arr[:, :, 3:4] / 255.0
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = (arr[:, :, 2] * a[:, :, 0]).astype(np.uint8)   # B premultiplied
    out[:, :, 1] = (arr[:, :, 1] * a[:, :, 0]).astype(np.uint8)   # G
    out[:, :, 2] = (arr[:, :, 0] * a[:, :, 0]).astype(np.uint8)   # R
    out[:, :, 3] = arr[:, :, 3].astype(np.uint8)
    return out


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "Claude"
    visual = (sys.argv[2] if len(sys.argv) > 2
              else os.environ.get("TTS_VOICE_VISUAL", "orb")).strip().lower()

    # Watch the pipe from the very first instant, BEFORE building the renderer.
    # The GPU face spends a couple of seconds bringing up torch and CUDA, and
    # say.py now starts us during synthesis; if the parent goes away inside that
    # window nothing was listening, and the process sat on screen forever —
    # even the hard cap never fired, because it is only checked in the render
    # loop, which had not been reached.
    audio: dict = {"bands": None, "hop": 0.0, "t0": 0.0,
                   "live": np.zeros(4, np.float32), "closed": False, "words": None,
                   "t_shown": None}

    def watch_stdin() -> None:
        try:
            for raw in sys.stdin:
                line = raw.strip()
                try:
                    if line.startswith("ENV3 "):
                        _, hop, delay, payload = line.split(" ", 3)
                        seg = np.array(
                            [[float(v) for v in row.split(",") if v]
                             for row in payload.split("|")], np.float32)
                        hop = float(hop)
                        t_seg = time.monotonic() + float(delay)
                        old = audio["bands"]
                        # The streaming path sends the envelope as a run of
                        # consecutive blocks rather than one whole clip, so
                        # stitch a block that continues the existing timeline
                        # onto the end of it. Replacing instead would reset the
                        # clock several times a second and the animation would
                        # keep restarting.
                        if old is not None and audio["hop"] == hop and                                 old.shape[0] == seg.shape[0]:
                            at = int(round((t_seg - audio["t0"]) / hop))
                            if -2 <= at <= old.shape[1] + 30:
                                at = max(at, 0)
                                width = max(old.shape[1], at + seg.shape[1])
                                grown = np.zeros((seg.shape[0], width), np.float32)
                                grown[:, :old.shape[1]] = old
                                grown[:, at:at + seg.shape[1]] = seg
                                audio["bands"] = grown
                                continue
                        audio["bands"], audio["hop"], audio["t0"] = seg, hop, t_seg
                    elif line.startswith("WORDS "):
                        _, delay, payload = line.split(" ", 2)
                        ws = []
                        for item in payload.split("|"):
                            s, e, w = item.split(",", 2)
                            ws.append((float(s), float(e), w))
                        # Same clock as the audio: start counting when the
                        # first sample actually becomes audible.
                        audio["words"] = (ws, time.monotonic() + float(delay))
                    elif line.startswith("LVL3 "):
                        vals = [float(v) for v in line.split()[1:5]]
                        while len(vals) < 4:
                            vals.append(0.4)      # neutral pitch when unstated
                        audio["live"] = np.array(vals, np.float32)
                except Exception:
                    pass          # a malformed line must never kill the overlay
        except Exception:
            pass
        audio["closed"] = True

    threading.Thread(target=watch_stdin, daemon=True).start()

    orb = _make_renderer(visual)
    if audio["closed"] and audio["bands"] is None:
        return 0                  # parent vanished while we were warming up
    orb_w = orb_h = orb.size
    W, H = orb_w, orb_h + 42 + CAPTION_H
    chip = build_label_layer(label, W, H, y=orb_h + CAPTION_H + 2)
    caption = Caption(W, CAPTION_H)
    cap_seen = [None]

    # The chip is static and (by layout) sits entirely below the orb, so the
    # composite is just "copy the orb into the top rows" — the chip rows are
    # written once here and never touched again. If a future layout does make
    # them overlap, fall back to a real per-frame source-over blend.
    frame = np.zeros((H, W, 4), dtype=np.uint8)
    frame[orb_h:] = chip[orb_h:]
    overlaps = bool(chip[:orb_h, :, 3].any())
    chip_f = chip.astype(np.float32) if overlaps else None
    chip_a = (chip[:, :, 3:4].astype(np.float32) / 255.0) if overlaps else None
    canvas = np.zeros((H, W, 4), dtype=np.float32) if overlaps else None

    scrim = {"layer": None}      # filled in once the desktop has been sampled

    def compose(frame_orb: np.ndarray) -> np.ndarray:
        if audio["words"] is not None and cap_seen[0] is not audio["words"]:
            cap_seen[0] = audio["words"]
            caption.set_words(*audio["words"])
        if overlaps:
            canvas.fill(0.0)
            canvas[:orb_h, :, :] = frame_orb
            comp = chip_f + canvas * (1.0 - chip_a)
            out = np.ascontiguousarray(comp.astype(np.uint8))
        else:
            frame[:orb_h] = frame_orb
            out = frame
        # REPLACE the caption band, never blend into it. The frame buffer is
        # reused between frames, so compositing the caption over whatever was
        # already there stacked every word on top of the last one: the line
        # turned into unreadable overlapping glyphs and nothing ever faded.
        # Nothing else draws in this band, so a straight assignment is right.
        out[orb_h:orb_h + CAPTION_H] = caption.strip(time.monotonic())

        sc = scrim["layer"]
        if sc is None:
            return out
        # Source-over of the visual on top of the backdrop; both premultiplied,
        # so it is one lerp by the visual's own alpha.
        inv = (255 - out[:, :, 3:4].astype(np.uint16))
        blended = out.astype(np.uint16) + (sc.astype(np.uint16) * inv) // 255
        return np.ascontiguousarray(np.minimum(blended, 255).astype(np.uint8))

    # --- window -------------------------------------------------------------
    hinst = kernel32.GetModuleHandleW(None)
    cls = ctypes.create_unicode_buffer("ClaudeSpeakingOrb")
    wndproc = WNDPROC(lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l))

    class WNDCLASS(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

    wc = WNDCLASS()
    wc.lpfnWndProc = wndproc
    wc.hInstance = hinst
    wc.lpszClassName = cls.value
    user32.RegisterClassW(ctypes.byref(wc))

    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    # The HD face is a presence, not a status pip: it belongs in the middle of
    # the screen. The small visuals stay tucked under the top edge.
    x = (sw - W) // 2
    y = (sh - H) // 2 if visual in ("hd", "scan") else TOP_MARGIN
    hwnd = user32.CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
        cls.value, cls.value, WS_POPUP & 0xFFFFFFFF, x, y, W, H,
        None, None, hinst, None)
    if not hwnd:
        return 1

    # --- DIB section --------------------------------------------------------
    screen_dc = user32.GetDC(None)
    mem_dc = gdi32.CreateCompatibleDC(screen_dc)
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = W
    bmi.bmiHeader.biHeight = -H          # negative = top-down rows
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0      # BI_RGB
    bits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(mem_dc, ctypes.byref(bmi), 0,
                                  ctypes.byref(bits), None, 0)
    old = gdi32.SelectObject(mem_dc, hbmp)
    buf = (ctypes.c_ubyte * (W * H * 4)).from_address(bits.value)

    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                        SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    pt_dst, pt_src, size = POINT(x, y), POINT(0, 0), SIZE(W, H)

    def push(frame: np.ndarray, alpha: int) -> None:
        ctypes.memmove(buf, frame.ctypes.data, frame.nbytes)
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, alpha, AC_SRC_ALPHA)
        user32.UpdateLayeredWindow(hwnd, screen_dc, ctypes.byref(pt_dst),
                                   ctypes.byref(size), mem_dc, ctypes.byref(pt_src),
                                   0, ctypes.byref(blend), ULW_ALPHA)


    def audio_bands(now: float) -> np.ndarray:
        """Per-band levels at this instant, linearly interpolated between
        analysis frames. The envelope is sampled at 120 Hz against a 60 fps
        loop, so interpolating (rather than snapping to the nearest sample)
        is what removes the last visible stepping from the motion."""
        bands = audio["bands"]
        if bands is None or audio["hop"] <= 0:
            return audio["live"]
        pos = (now - audio["t0"]) / audio["hop"]
        if pos < 0 or pos >= bands.shape[1] - 1:
            # Silence, but hold a neutral pitch so the modes do not snap.
            z = np.zeros(bands.shape[0], np.float32)
            if z.size >= 4:
                z[3] = 0.4
            return z
        i = int(pos)
        f = pos - i
        return bands[:, i] * (1.0 - f) + bands[:, i + 1] * f

    # --- animation loop -----------------------------------------------------
    # Windows' default timer granularity (~15.6 ms) makes time.sleep overshoot
    # badly at a 16.7 ms frame budget; ask for 1 ms while we animate.
    winmm.timeBeginPeriod(1)
    msg = wintypes.MSG()
    start = time.monotonic()
    next_frame = start
    fade_out_at: float | None = None
    level = np.zeros(4, np.float32)
    spin = 0.0
    prev = start
    n_frames = 0
    try:
        while True:
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

            now = time.monotonic()
            elapsed = now - start

            # Stay invisible until audio actually arrives. Two reasons: the
            # visual should appear WITH the voice rather than ahead of it, and
            # the HD renderer needs ~2.5 s to bring up torch and CUDA — so
            # say.py starts us early, during synthesis, and we simply wait. The
            # fade-in is measured from the first audio message, not from launch.
            if audio["t_shown"] is None:
                if audio["bands"] is None and not audio["live"].any():
                    if audio["closed"] or elapsed > HARD_CAP_S:
                        break
                    time.sleep(0.01)
                    next_frame = time.monotonic()
                    continue
                audio["t_shown"] = now
                # Sample the desktop now, in the last instant before we appear:
                # the window is still invisible, so the grab sees what is really
                # behind us rather than our own pixels. Neon dots on
                # transparency read fine on a dark desktop and all but vanish on
                # a white page, so how much backdrop to lay down is decided from
                # what is actually there.
                luma = screen_luma(x, y, W, H)
                strength = float(np.clip((luma - 0.22) / 0.45, 0.0, 1.0)) * 0.82
                if hasattr(orb, "set_backdrop"):
                    orb.set_backdrop(strength)      # GPU visuals do it themselves
                if hasattr(orb, "set_ambient"):
                    orb.set_ambient(getattr(screen_luma, "last_color",
                                            (128.0, 128.0, 128.0)))
                if hasattr(orb, "set_background"):
                    # Full-resolution grab, for visuals that refract what is
                    # behind them rather than just tinting themselves with it.
                    orb.set_background(capture_region(x, y, W, H))
                elif strength > 0.01:
                    scrim["layer"] = build_scrim(W, H, strength)
                if os.environ.get("TTS_OVERLAY_STATS") == "1":
                    print(f"[overlay] desktop luma {luma:.2f} -> backdrop {strength:.2f}",
                          file=sys.stderr)

            visible_for = now - audio["t_shown"]
            if audio["closed"] and fade_out_at is None:
                fade_out_at = now
            if fade_out_at is not None:
                # Let the visual play its own ending if it has one, rather than
                # just being faded out from the outside.
                if hasattr(orb, "begin_collapse"):
                    orb.begin_collapse()
                out_ms = (now - fade_out_at) * 1000.0
                if out_ms >= FADE_MS:
                    break
                alpha = int(255 * (1.0 - out_ms / FADE_MS))
            else:
                alpha = int(255 * min(1.0, visible_for * 1000.0 / FADE_MS))
            alpha = int(alpha * OPACITY)
            if elapsed > HARD_CAP_S:
                break

            dt = min(0.1, now - prev)
            prev = now
            target = audio_bands(now)
            if audio["bands"] is not None:
                # The offline analysis is already smoothed with zero phase
                # shift, so following it directly is what tracks the waveform.
                # Any extra filtering here would only re-introduce lag.
                level = target
            else:
                # Streaming path: chunk levels arrive raw and jumpy, so they do
                # need smoothing. Time-constant based, not per-frame, so it
                # looks the same regardless of frame rate.
                tau = np.where(target > level, ATTACK_TAU, RELEASE_TAU)
                level = level + (target - level) * (1.0 - np.exp(-dt / tau))
            # Idle spin, plus a touch more when he's actually talking.
            spin += (0.55 + 1.4 * float(level[1])) * dt

            push(compose(orb.render(visible_for * 0.35, level, spin)),
                 max(0, min(255, alpha)))
            n_frames += 1
            # Schedule against a running deadline rather than sleeping a fixed
            # delta, so per-frame overshoot doesn't accumulate into drift.
            next_frame += 1.0 / FPS
            slack = next_frame - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -0.25:
                next_frame = time.monotonic()      # fell far behind; resync
    finally:
        winmm.timeEndPeriod(1)
        if os.environ.get("TTS_OVERLAY_STATS") == "1":
            # Measure from when the visual actually appeared, not from launch:
            # the wait for audio (and the renderer's warm-up) renders nothing,
            # and counting it made a healthy 63 fps read as 40.
            t_vis = audio["t_shown"] or start
            dur = time.monotonic() - t_vis
            print(f"[overlay] {n_frames} frames in {dur:.2f}s visible = "
                  f"{n_frames / max(dur, 1e-6):.1f} fps (target {FPS})", file=sys.stderr)
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(None, screen_dc)
        user32.DestroyWindow(hwnd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

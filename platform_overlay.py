"""macOS overlay surface for the speaking visual: a borderless, click-through,
per-pixel-alpha NSWindow.

`overlay.py` owns the Windows path (a layered window driven by
UpdateLayeredWindow). This module is the macOS half of the same idea, behind a
small interface the render loop can drive without knowing which OS it is on:

    win = create_overlay_window(x, y, w, h, "Claude")
    if win is not None:
        win.push(frame, alpha)   # frame: uint8 [h, w, 4], premultiplied BGRA
        win.pump()               # drain pending UI events, never blocks
        win.close()

The contract that matters: **the visual is decorative and must never stop the
voice.** So `create_overlay_window` returns None instead of raising when the
platform cannot give us the surface -- no PyObjC, no window server, a headless
CI box, or simply Windows (where the caller keeps using `overlay.py`). Same
degradation rule `platform_native.py` follows.

Status: **this file has never been run on real Mac hardware.** It is written
against the documented behaviour of AppKit, Quartz and Core Animation. The
comments below flag every place where a wrong assumption would show up as a
visible bug (window in the wrong corner, invisible layer, stolen focus) so
whoever first runs it knows where to look. Treat it as unverified, not as
working code that happens to be untested.
"""

from __future__ import annotations

import sys
from typing import Any

IS_MACOS = sys.platform == "darwin"

# Core Graphics bitmap flags. The renderer hands us premultiplied BGRA -- see
# `orb_render.Orb.render` -- which is byte order B,G,R,A in memory. Read as
# little-endian 32-bit words that is ARGB with alpha first, i.e. exactly
# kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little. This is a
# genuine coincidence with the Windows path (UpdateLayeredWindow wants the same
# bytes), and it is why nothing here shuffles channels: any "conversion" would
# be a bug, not an optimisation.
_ALPHA_PREMULTIPLIED_FIRST = 2
_BYTE_ORDER_32_LITTLE = 1 << 12
_BITMAP_INFO = _ALPHA_PREMULTIPLIED_FIRST | _BYTE_ORDER_32_LITTLE
_RENDERING_INTENT_DEFAULT = 0

# AppKit constants, spelled out because PyObjC does not always export every
# enum member and a missing symbol here would take the whole overlay down.
_STYLE_MASK_BORDERLESS = 0
_BACKING_STORE_BUFFERED = 2
_STATUS_WINDOW_LEVEL = 25          # NSStatusWindowLevel: above normal windows
                                   # and above floating panels (which are 3).
_COLLECTION_CAN_JOIN_ALL_SPACES = 1 << 0
_COLLECTION_STATIONARY = 1 << 4    # do not slide away in Mission Control
_ACTIVATION_POLICY_ACCESSORY = 1   # no Dock tile, no menu bar, never key
_EVENT_MASK_ANY = (1 << 64) - 1    # NSEventMaskAny

# Populated once by `_appkit()`. None means "not attempted yet"; False means
# "attempted and unavailable", so we complain to stderr exactly once instead of
# once per frame.
_MODS: Any = None


def _appkit() -> Any:
    """Import PyObjC lazily and return the pieces we need, or None.

    Deliberately not a module-level import: this file must import cleanly on
    Windows and in a CI container with no PyObjC and no window server, so that
    the pure logic (coordinate conversion, support probing) stays testable
    everywhere. Anything that touches Cocoa goes through here.
    """
    global _MODS
    if _MODS is not None:
        return _MODS or None
    if not IS_MACOS:
        _MODS = False
        return None
    try:
        import AppKit          # type: ignore[import-not-found]
        import Quartz          # type: ignore[import-not-found]
        import Foundation      # type: ignore[import-not-found]
    except Exception as e:     # ImportError, but also a broken PyObjC install
        print(f"[overlay] PyObjC unavailable ({e}); no macOS overlay",
              file=sys.stderr)
        _MODS = False
        return None
    _MODS = {"AppKit": AppKit, "Quartz": Quartz, "Foundation": Foundation}
    return _MODS


def is_supported() -> bool:
    """True only on macOS with a working PyObjC. False everywhere else.

    False is not an error: the caller is expected to fall back to its own
    platform path (Windows) or to run with no visual at all.
    """
    return _appkit() is not None


def backing_scale() -> float:
    """Backing pixels per point on the primary screen (1.0, or 2.0 on Retina).

    Everything this module exposes is measured in DEVICE PIXELS, matching the
    Windows path, which declares itself DPI-aware and works in real pixels. Only
    the Cocoa geometry underneath is in points, and this is the factor between
    the two. Returns 1.0 when it cannot be determined, which degrades to the
    non-Retina behaviour rather than to a crash.
    """
    mods = _appkit()
    if mods is None:
        return 1.0
    try:
        screens = mods["AppKit"].NSScreen.screens()
        if not screens:
            return 1.0
        return float(screens[0].backingScaleFactor()) or 1.0
    except Exception:
        return 1.0


def screen_size() -> tuple[int, int] | None:
    """(width, height) of the primary screen in DEVICE PIXELS, or None.

    "Primary" means the screen carrying the menu bar, which is `screens()[0]`
    and the one Cocoa's global coordinate origin sits on.

    Pixels, not points, deliberately: the caller lays the overlay out in the
    same units the renderer produces, so a 384-pixel orb is 384 real pixels on
    a Retina panel exactly as it is on Windows. `create_overlay_window` divides
    by `backing_scale()` at the last moment, where Cocoa needs points.
    """
    mods = _appkit()
    if mods is None:
        return None
    try:
        screens = mods["AppKit"].NSScreen.screens()
        if not screens:
            return None
        frame = screens[0].frame()
        scale = backing_scale()
        return int(frame.size.width * scale), int(frame.size.height * scale)
    except Exception as e:
        print(f"[overlay] could not read the screen size: {e}", file=sys.stderr)
        return None


def _cocoa_origin_y(top_left_y: int, h: int, screen_h: int) -> int:
    """Convert a top-left-origin y to Cocoa's bottom-left-origin y.

    THE most likely bug in this file. Windows (and the rest of this project)
    measure y downward from the top of the screen and address a window by its
    TOP edge. Cocoa measures y upward from the bottom of the primary screen and
    addresses a window by its BOTTOM edge. So both the direction and the edge
    change:

        cocoa_y = screen_height - (top_left_y + height)

    A window meant to sit 12 px below the top edge of a 1080-point screen and
    500 points tall lands at cocoa y = 1080 - 512 = 568. Get this wrong and the
    overlay appears mirrored vertically (near the bottom instead of the top),
    which is the symptom to look for.

    Kept as a module-level pure function precisely so it can be tested on a
    machine that has no Cocoa at all.
    """
    return screen_h - (top_left_y + h)


class OverlayWindow:
    """A borderless, click-through, always-on-top window with per-pixel alpha.

    Construct it through `create_overlay_window`, never directly: the factory
    is what turns a failure into None instead of an exception.

    Threading: AppKit is main-thread-only. `push`/`pump`/`close` must all be
    called from the thread that created the window, and that thread should be
    the process's main thread. The overlay process drives its own ~60 fps loop
    there, so this holds today; it would break if a future caller moved the
    render loop onto a worker.
    """

    def __init__(self, mods: Any, window: Any, view: Any, w: int, h: int) -> None:
        self._mods = mods
        self._window = window
        self._view = view
        self._w, self._h = w, h
        self._closed = False
        # The CGImage does not copy the pixels; it reads them through the data
        # provider for as long as it is alive, and the layer keeps the image
        # alive for as long as it is on screen. If the bytes object were only a
        # local in `push`, Python would free it the moment push returned and
        # the layer would be sampling freed memory -- which shows up as garbage
        # or a hard crash, not as a clean error. Holding the most recent buffer
        # here is what keeps it alive until the next frame replaces it.
        self._pixels: Any = None
        self._provider: Any = None
        self._image: Any = None

    def push(self, frame: "Any", alpha: int = 255) -> None:
        """Show `frame` (uint8 [h, w, 4], premultiplied BGRA) at `alpha` 0-255.

        `alpha` is a whole-window constant opacity multiplied on top of the
        per-pixel alpha, the analogue of BLENDFUNCTION.SourceConstantAlpha on
        Windows. It is what the fade in/out rides on.

        Never raises: a frame we cannot draw is a dropped frame, not a dead
        voice.
        """
        if self._closed:
            return
        try:
            Quartz = self._mods["Quartz"]
            Foundation = self._mods["Foundation"]

            h, w = frame.shape[0], frame.shape[1]
            # A non-contiguous array would make `tobytes()` silently reorder
            # rows relative to the stride we declare below, so normalise first.
            # `tobytes` also gives us the copy that decouples the image from the
            # renderer's reused frame buffer (`Orb.render` documents that it
            # hands back the same array every call).
            data = frame.tobytes()
            cfdata = Foundation.NSData.dataWithBytes_length_(data, len(data))
            provider = Quartz.CGDataProviderCreateWithCFData(cfdata)
            colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
            image = Quartz.CGImageCreate(
                w, h,
                8,              # bits per component
                32,             # bits per pixel
                w * 4,          # bytes per row: tightly packed, no padding
                colorspace,
                _BITMAP_INFO,
                provider,
                None,           # decode array
                False,          # shouldInterpolate: we render at native size
                _RENDERING_INTENT_DEFAULT,
            )
            if image is None:
                return

            layer = self._view.layer()
            if layer is None:
                return
            # Assigning `contents` is the cheap path: Core Animation uploads the
            # image to the compositor directly. Subclassing NSView.drawRect_
            # would mean a full redraw cycle per frame and an extra blit through
            # the window's backing store, for no benefit -- we have nothing to
            # draw except one already-composited image.
            layer.setContents_(image)
            self._window.setAlphaValue_(max(0, min(255, int(alpha))) / 255.0)

            # Replace the previous frame's references only after the new ones
            # are installed, so there is never an instant where the layer holds
            # an image whose bytes we have already dropped.
            self._pixels, self._provider, self._image = data, provider, image
        except Exception as e:
            print(f"[overlay] frame not shown: {e}", file=sys.stderr)

    def pump(self) -> None:
        """Drain pending UI events without blocking.

        The render loop calls this once per frame. `distantPast` as the
        deadline is what makes it non-blocking: nextEventMatchingMask returns
        immediately with None when the queue is empty instead of waiting for
        input. Events still have to be handed back via `sendEvent_` or the app
        never processes anything (and macOS eventually marks the process as not
        responding, which is the symptom if this loop is ever removed).

        We are click-through and never key, so in practice there is little to
        deliver -- this is hygiene, not interactivity.
        """
        if self._closed:
            return
        try:
            AppKit = self._mods["AppKit"]
            Foundation = self._mods["Foundation"]
            app = AppKit.NSApp()
            if app is None:
                return
            past = Foundation.NSDate.distantPast()
            # Bounded, not `while True`: a pathological event storm must not be
            # able to hold the frame loop hostage past its 16 ms budget.
            for _ in range(64):
                event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                    _EVENT_MASK_ANY, past, AppKit.NSDefaultRunLoopMode, True)
                if event is None:
                    break
                app.sendEvent_(event)
        except Exception as e:
            print(f"[overlay] event pump failed: {e}", file=sys.stderr)

    def close(self) -> None:
        """Take the window off screen and drop the pixel references.

        Idempotent, and safe to call from a `finally` after a failed push.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._window.orderOut_(None)
            self._window.close()
        except Exception as e:
            print(f"[overlay] close failed: {e}", file=sys.stderr)
        finally:
            # Order matters only in that the image must go before the bytes it
            # reads; setting all three to None in one statement is fine because
            # CPython drops them left to right.
            self._image = self._provider = self._pixels = None


def create_overlay_window(x: int, y: int, w: int, h: int,
                          title: str) -> OverlayWindow | None:
    """Create the overlay window at `x, y` (TOP-LEFT origin), or return None.

    `x, y` follow the same convention as the rest of the project and as the
    Windows path: origin at the top-left of the primary screen, y growing
    downward. The conversion to Cocoa's bottom-left origin happens inside; see
    `_cocoa_origin_y`.

    `title` is not displayed -- the window is borderless and has no title bar.
    It is set anyway because it is what shows up in accessibility tooling and
    window listings, which is the only way to identify the thing on screen when
    debugging.

    Returns None on any failure, including "this is not macOS". That is the
    whole point: the caller falls back rather than crashing.
    """
    mods = _appkit()
    if mods is None:
        return None
    try:
        AppKit = mods["AppKit"]

        # Bring the app up before touching windows. Accessory policy is what
        # keeps us out of the Dock and out of the app switcher, and stops the
        # process from ever becoming the active app -- the macOS equivalent of
        # WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE. Setting it after a window exists
        # is documented as unreliable, so it goes first.
        app = AppKit.NSApplication.sharedApplication()
        app.setActivationPolicy_(_ACTIVATION_POLICY_ACCESSORY)

        size = screen_size()
        if size is None:
            return None
        _screen_w, screen_h = size          # device pixels
        scale = backing_scale()

        # Lay out in pixels, then convert to points once, here. Doing the
        # arithmetic in pixels keeps this identical to the Windows path; doing
        # the division at the end is what makes the orb come out the same
        # physical size on a Retina panel as on a 1x one.
        cocoa_y_px = _cocoa_origin_y(y, h, screen_h)
        rect = AppKit.NSMakeRect(x / scale, cocoa_y_px / scale,
                                 w / scale, h / scale)

        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            _STYLE_MASK_BORDERLESS,
            _BACKING_STORE_BUFFERED,
            False,   # defer=False: create the window device now, so the first
                     # frame has somewhere to land instead of being dropped.
        )
        if window is None:
            return None

        window.setTitle_(title)
        # Non-opaque + clear background is the pair that gives real per-pixel
        # alpha. Miss either one and the transparent parts of the frame render
        # as black -- the exact failure tkinter's colour-keying has, and the
        # reason this window exists at all.
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setHasShadow_(False)          # a shadow would outline the
                                             # bounding box of a round orb
        window.setIgnoresMouseEvents_(True)  # click-through
        window.setLevel_(_STATUS_WINDOW_LEVEL)
        window.setCollectionBehavior_(
            _COLLECTION_CAN_JOIN_ALL_SPACES | _COLLECTION_STATIONARY)

        view = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(0, 0, w / scale, h / scale))
        view.setWantsLayer_(True)
        window.setContentView_(view)

        layer = view.layer()
        if layer is not None:
            # contentsScale tells Core Animation how many image pixels make up
            # one point. The frame is w x h DEVICE PIXELS and the layer is
            # w/scale x h/scale points, so the honest value is the screen's own
            # scale. Pinning it to 1 instead would declare a 384-pixel image to
            # be 384 points wide, and on a 2x panel Core Animation would stretch
            # it across 768 physical pixels: an orb at double size and visibly
            # soft. The symptom to look for is a blurry, oversized orb.
            layer.setContentsScale_(scale)
            layer.setContentsGravity_("resize")

        # orderFrontRegardless, never makeKeyAndOrderFront_: the latter makes us
        # the key window, which pulls focus away from whatever the user is
        # typing into. An overlay that steals the keyboard is worse than no
        # overlay.
        window.orderFrontRegardless()
        return OverlayWindow(mods, window, view, w, h)
    except Exception as e:
        print(f"[overlay] macOS overlay unavailable: {e}", file=sys.stderr)
        return None

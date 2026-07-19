"""Tests for the macOS overlay surface.

None of this can be run against a real Cocoa window from CI, let alone from the
Windows machine it was written on. What IS testable is the part most likely to
be wrong: the coordinate conversion, and the promise that every entry point
degrades to None instead of raising on a platform that has no surface to offer.

That promise is the load-bearing one. The overlay is decorative; if it cannot
appear it must get out of the way silently, because say.py treats a crashed
overlay process as nothing more than a missing visual -- speech carries on.
"""
import sys

import platform_overlay


# --- coordinate conversion -------------------------------------------------
# Windows addresses a window by its TOP edge, measuring down from the top of the
# screen. Cocoa addresses it by its BOTTOM edge, measuring up from the bottom.
# Both the direction and the reference edge change, so a naive `screen_h - y`
# is wrong by exactly the window height -- and it is wrong *plausibly*, which is
# why it gets its own tests.

def test_cocoa_origin_flips_and_accounts_for_height():
    # 500-tall window, 12 px below the top of a 1080 screen.
    assert platform_overlay._cocoa_origin_y(12, 500, 1080) == 568


def test_cocoa_origin_at_top_of_screen():
    """Flush with the top edge: the bottom sits at screen_h - height."""
    assert platform_overlay._cocoa_origin_y(0, 500, 1080) == 580


def test_cocoa_origin_at_bottom_of_screen():
    """Flush with the bottom edge is Cocoa's origin, y = 0."""
    assert platform_overlay._cocoa_origin_y(580, 500, 1080) == 0


def test_cocoa_origin_is_its_own_inverse():
    """The conversion is symmetric: applying it twice returns the input.

    A sign error or a dropped height term breaks this, which makes it a cheap
    guard against the whole family of mistakes.
    """
    for y in (0, 12, 300, 580):
        once = platform_overlay._cocoa_origin_y(y, 500, 1080)
        assert platform_overlay._cocoa_origin_y(once, 500, 1080) == y


def test_cocoa_origin_on_a_retina_sized_screen():
    """Pixel arithmetic: a 2x 1080p panel is 3840x2160 device pixels."""
    assert platform_overlay._cocoa_origin_y(24, 1000, 2160) == 1136


# --- degradation off macOS -------------------------------------------------

def test_is_supported_is_false_off_macos():
    if sys.platform != "darwin":
        assert platform_overlay.is_supported() is False


def test_create_returns_none_rather_than_raising_off_macos():
    """The contract the caller relies on: None, never an exception."""
    if sys.platform != "darwin":
        assert platform_overlay.create_overlay_window(0, 0, 100, 100, "t") is None


def test_screen_size_is_none_off_macos():
    if sys.platform != "darwin":
        assert platform_overlay.screen_size() is None


def test_backing_scale_never_raises_and_is_positive():
    """Callers divide by this, so a zero or a raise would be worse than a guess."""
    scale = platform_overlay.backing_scale()
    assert isinstance(scale, float)
    assert scale > 0

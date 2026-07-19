"""Adaptive navigation to a speaking session.

Priority (see docs/2026-07-18-voice-hub-design.md):
  1. Session owns a console/WT window -> foreground it (+ select the WT tab).
  2. Otherwise -> copy `claude --resume <id>` to the clipboard and foreground
     the Claude desktop app. There is no supported API to focus a specific
     session inside the app, so a one-paste jump is the best available action.

Every entry point returns a short status string for the hub to display and
never raises.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time

import agents_registry


# ----------------------------- window primitives ---------------------------

def _force_foreground(hwnd: int) -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(0x12, 0, 0, 0)  # ALT down releases the foreground lock
    try:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
    finally:
        user32.keybd_event(0x12, 0, 2, 0)  # ALT up


def _select_wt_tab(hwnd: int, tab_title: str) -> bool:
    if not tab_title:
        return False
    import uiautomation as uia
    win = uia.ControlFromHandle(hwnd)
    if win is None:
        return False
    needle = tab_title.lower()
    for c, _depth in uia.WalkControl(win, maxDepth=8):
        if c.ControlTypeName == "TabItemControl" and needle in (c.Name or "").lower():
            c.GetSelectionItemPattern().Select()
            return True
    return False


def find_claude_app_hwnd() -> int:
    """HWND of the Claude desktop app window ('Claude'), or 0."""
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if buf.value.strip().lower() == "claude":
            found.append(int(ctypes.cast(hwnd, ctypes.c_void_p).value or 0))
            return False
        return True

    user32.EnumWindows(cb, None)
    return found[0] if found else 0


# ----------------------------- clipboard -----------------------------------

def set_clipboard(text: str) -> bool:
    """Put text on the Windows clipboard (CF_UNICODETEXT)."""
    import time as _time
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    # Handles are pointer-sized: without explicit restypes ctypes truncates
    # them to 32-bit ints and SetClipboardData silently fails on 64-bit.
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    GMEM_MOVEABLE = 0x0002
    CF_UNICODETEXT = 13
    size = (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
    for _ in range(10):  # another process may briefly own the clipboard
        if user32.OpenClipboard(None):
            break
        _time.sleep(0.05)
    else:
        return False
    try:
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, size)
        if not handle:
            return False
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            return False
        try:
            ctypes.memmove(ptr, ctypes.create_unicode_buffer(text), size)
        finally:
            kernel32.GlobalUnlock(handle)
        # SetClipboardData takes ownership of the handle on success.
        return bool(user32.SetClipboardData(CF_UNICODETEXT, handle))
    finally:
        user32.CloseClipboard()


# ----------------------------- keystrokes ----------------------------------

KEYEVENTF_UNICODE, KEYEVENTF_KEYUP = 0x0004, 0x0002
VK_RETURN = 0x0D
INPUT_KEYBOARD = 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def _build_unicode_inputs(text: str) -> list:
    out = []
    for ch in text:
        for flags in (KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP):
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.ki = KEYBDINPUT(0, ord(ch), flags, 0, 0)
            out.append(inp)
    return out


def type_text(text: str, press_enter: bool = True) -> None:
    """Type into the FOCUSED window. Caller must have focused the right target;
    only use this when the target is an unambiguous console window."""
    inputs = _build_unicode_inputs(text)
    if press_enter:
        for flags in (0, KEYEVENTF_KEYUP):
            inp = INPUT()
            inp.type = INPUT_KEYBOARD
            inp.ki = KEYBDINPUT(VK_RETURN, 0, flags, 0, 0)
            inputs.append(inp)
    arr = (INPUT * len(inputs))(*inputs)
    sent = ctypes.windll.user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))
    if sent != len(inputs):
        raise OSError(f"SendInput injected {sent}/{len(inputs)} events")


# ----------------------------- the adaptive action -------------------------

def go_to_session(evt: dict) -> str:
    """Best available jump to the session that produced evt. Returns a short
    status for the hub ('' means nothing could be done)."""
    try:
        nav = evt.get("nav") or {}
        hwnd = nav.get("hwnd") or 0
        if hwnd:
            _force_foreground(hwnd)
            if nav.get("window_kind") == "wt":
                try:
                    if _select_wt_tab(hwnd, nav.get("tab_title", "")):
                        return "ventana + tab"
                except Exception:
                    pass  # UIA failed; the window is focused, good enough
            return "ventana enfocada"

        session_id = evt.get("session_id") or ""
        if not session_id:
            return ""
        copied = set_clipboard(agents_registry.resume_command(session_id))
        app = find_claude_app_hwnd()
        if app:
            _force_foreground(app)
        if copied:
            return "resume copiado, pega en la app"
        return "app enfocada"
    except Exception:
        return ""


def deliver_text(evt: dict, text: str) -> str:
    """Send dictated text to the session that produced evt.

    Types it only when the target is an unambiguous console window. For the
    desktop app the text goes to the clipboard instead: typing blind would land
    in whichever session happens to be open, which may not be the one that
    asked. Returns a short status for the hub.
    """
    try:
        nav_info = evt.get("nav") or {}
        hwnd = nav_info.get("hwnd") or 0
        if hwnd:
            _force_foreground(hwnd)
            if nav_info.get("window_kind") == "wt":
                try:
                    _select_wt_tab(hwnd, nav_info.get("tab_title", ""))
                except Exception:
                    pass
            time.sleep(0.3)  # let the window settle before injecting keys
            type_text(text)
            return "enviado a la ventana"

        if not set_clipboard(text):
            return "no se pudo copiar"
        app = find_claude_app_hwnd()
        if app:
            _force_foreground(app)
        return "texto copiado, pega en la sesion"
    except Exception as e:
        return f"fallo la entrega: {type(e).__name__}"

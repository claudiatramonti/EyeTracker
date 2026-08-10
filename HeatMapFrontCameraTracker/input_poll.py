"""Keyboard input for OpenCV windows (including Windows fallback)."""

import sys
import time

import cv2

# Windows fallback when OpenCV HighGUI does not have focus.
_WIN32_KEY_VKS = (
    (0x51, ord("q")),
    (0x43, ord("c")),
    (0x42, ord("b")),
    (0x52, ord("r")),
    (0x53, ord("s")),
    (0x48, ord("h")),
    (0x4B, ord("k")),
    (0x58, ord("x")),
    (0x4D, ord("m")),
    (0x56, ord("v")),
    (0x20, ord(" ")),
)

_DEBOUNCE_SEC = 0.2
_prev_down = {}
_last_char = None
_last_char_time = 0.0


def _debounced(char):
    global _last_char, _last_char_time
    now = time.perf_counter()
    if char == _last_char and (now - _last_char_time) < _DEBOUNCE_SEC:
        return False
    _last_char = char
    _last_char_time = now
    return True


def _sync_win32_key_state(char, user32):
    """Mark a key as handled so the Win32 fallback cannot fire again for the same press."""
    for vk, mapped in _WIN32_KEY_VKS:
        if mapped != char:
            continue
        user32.GetAsyncKeyState(vk)
        _prev_down[vk] = bool(user32.GetAsyncKeyState(vk) & 0x8000)


def poll_control_key(window_name):
    """Read control keys once per physical press (OpenCV + Win32 fallback)."""
    key = cv2.waitKey(1)
    if key != -1:
        char = key & 0xFF
        if sys.platform == "win32":
            user32 = __import__("ctypes").windll.user32
            _sync_win32_key_state(char, user32)
        if _debounced(char):
            return char
        return 255

    if sys.platform != "win32":
        return 255

    user32 = __import__("ctypes").windll.user32
    for vk, char in _WIN32_KEY_VKS:
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        was_down = _prev_down.get(vk, False)
        _prev_down[vk] = down
        if down and not was_down and _debounced(char):
            return char
    return 255

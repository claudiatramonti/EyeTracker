"""
GazeScreen3D — eye gaze ∩ ArUco screen plane → monitor pixel + heatmap.

Pipeline:
  1. IR eye camera(s) → 3D gaze direction
  2. C at screen center → R_gaze_to_cam (eye space → front camera)
  3. Optional arrow keys at edges → yaw/pitch scales (refine eye→cam only)
  4. Front camera ArUco → screen plane in front-camera coords
  5. Ray from camera origin along rotated gaze ∩ plane → screen pixel

Usage:
  cd GazeScreen3D
  python GazeScreen3D.py
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
for path in (ROOT, os.path.join(REPO, "ArucoScreenPose")):
    if path not in sys.path:
        sys.path.insert(0, path)

from camera_io import CameraReader
from gaze_scale_calib import (
    GazeScaleCalib,
    draw_center_calib_target,
    draw_edge_targets,
    draw_front_preview_not_for_c,
)
from heatmap import GazeHeatmap
from ray_screen import gaze_to_screen, project_gaze_to_front_pixels

import eye_tracker
import screen_pose as sp

WINDOW_NAME = "GazeScreen3D"
DEFAULT_HFOV = sp.DEFAULT_HFOV_DEG

# Synthetic arrow codes (outside ASCII), same idea as HeatMap input_poll.
KEY_UP = 0xE001
KEY_DOWN = 0xE002
KEY_LEFT = 0xE003
KEY_RIGHT = 0xE004

_DEBOUNCE_SEC = 0.2
_last_char = None
_last_char_time = 0.0
_WIN32_KEY_VKS = (
    (0x51, ord("q")),
    (0x43, ord("c")),
    (0x4D, ord("m")),
    (0x4B, ord("k")),
    (0x48, ord("h")),
    (0x56, ord("v")),
    (0x55, ord("u")),
    (0x45, ord("e")),  # reset edge scales
    (0x30, ord("0")),
    (0xBB, ord("=")),
    (0xBD, ord("-")),
    (0x26, KEY_UP),
    (0x28, KEY_DOWN),
    (0x25, KEY_LEFT),
    (0x27, KEY_RIGHT),
)

_OPENCV_ARROW_MAP = {
    2490368: KEY_UP,
    2621440: KEY_DOWN,
    2424832: KEY_LEFT,
    2555904: KEY_RIGHT,
}


def _debounced(char):
    global _last_char, _last_char_time
    now = time.perf_counter()
    if char == _last_char and (now - _last_char_time) < _DEBOUNCE_SEC:
        return False
    _last_char = char
    _last_char_time = now
    return True


def poll_key():
    key = cv2.waitKey(1)
    if key != -1:
        char = _OPENCV_ARROW_MAP.get(key, key & 0xFF)
        if sys.platform == "win32":
            user32 = __import__("ctypes").windll.user32
            for vk, mapped in _WIN32_KEY_VKS:
                if mapped == char:
                    user32.GetAsyncKeyState(vk)
        if _debounced(char):
            return char
        return 255

    if sys.platform != "win32":
        return 255

    user32 = __import__("ctypes").windll.user32
    if not hasattr(poll_key, "_prev"):
        poll_key._prev = {}
    prev = poll_key._prev
    for vk, char in _WIN32_KEY_VKS:
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        was_down = prev.get(vk, False)
        prev[vk] = down
        if down and not was_down and _debounced(char):
            return char
    return 255


def _arrow_to_edge(key):
    return {
        KEY_UP: "top",
        KEY_DOWN: "bottom",
        KEY_LEFT: "left",
        KEY_RIGHT: "right",
    }.get(key)


def selection_gui():
    import tkinter as tk
    from tkinter import ttk

    cameras = sp.detect_cameras()
    root = tk.Tk()
    root.title("GazeScreen3D")
    tk.Label(
        root,
        text="IR eye + front camera → gaze on screen (ArUco plane)",
        font=("Arial", 12, "bold"),
    ).pack(pady=10)

    def labels():
        return [str(c) for c in cameras] if cameras else ["0"]

    left_var = tk.StringVar(value=labels()[0])
    right_var = tk.StringVar(value="None")
    front_var = tk.StringVar(value=labels()[1] if len(cameras) > 1 else labels()[0])
    flip_left = tk.BooleanVar(value=True)
    flip_right = tk.BooleanVar(value=False)
    mirror_left = tk.BooleanVar(value=False)
    mirror_right = tk.BooleanVar(value=True)
    flip_front = tk.BooleanVar(value=False)
    mirror_front = tk.BooleanVar(value=False)

    frame = ttk.Frame(root)
    frame.pack(pady=6)

    def row(r, text, var, values):
        tk.Label(frame, text=text).grid(row=r, column=0, sticky="w", padx=6, pady=3)
        box = ttk.Combobox(frame, textvariable=var, values=values, width=10, state="readonly")
        box.grid(row=r, column=1, padx=6, pady=3)

    row(0, "Left IR:", left_var, labels())
    row(1, "Right IR:", right_var, ["None"] + labels())
    row(2, "Front camera:", front_var, labels())

    checks = ttk.Frame(root)
    checks.pack(pady=4)
    ttk.Checkbutton(checks, text="Flip L (upside-down mount)", variable=flip_left).pack(side="left", padx=4)
    ttk.Checkbutton(checks, text="Mirror L", variable=mirror_left).pack(side="left", padx=4)
    ttk.Checkbutton(checks, text="Flip R", variable=flip_right).pack(side="left", padx=4)
    ttk.Checkbutton(checks, text="Mirror R", variable=mirror_right).pack(side="left", padx=4)
    ttk.Checkbutton(checks, text="Flip front", variable=flip_front).pack(side="left", padx=4)
    ttk.Checkbutton(checks, text="Mirror front", variable=mirror_front).pack(side="left", padx=4)

    tk.Label(
        root,
        text="C = screen center | click IR preview = lock eye center | U unlock | Q quit\n"
        "Flip L = left IR mounted upside-down (corrects image before gaze math).",
        font=("Arial", 9),
    ).pack(pady=6)

    choice = {}

    def parse(v):
        return None if v == "None" else int(v)

    def start():
        choice["left"] = parse(left_var.get())
        choice["right"] = parse(right_var.get())
        choice["front"] = parse(front_var.get())
        choice["flip_left"] = flip_left.get()
        choice["flip_right"] = flip_right.get()
        choice["mirror_left"] = mirror_left.get()
        choice["mirror_right"] = mirror_right.get()
        choice["flip_front"] = flip_front.get()
        choice["mirror_front"] = mirror_front.get()
        root.destroy()

    tk.Button(root, text="Start", command=start).pack(pady=10)
    root.mainloop()
    return choice


def fit_frame(frame, width, height):
    """Letterbox frame into (width, height). Returns canvas, ox, oy, nw, nh."""
    fh, fw = frame.shape[:2]
    scale = min(width / fw, height / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, ox, oy, nw, nh


def maximize_cv_window(window_name):
    """Maximize OpenCV window on Windows (call after first imshow)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
    except (AttributeError, OSError):
        pass


def lock_eye_sphere_center(eye_id, frame_x, frame_y):
    """Click on IR preview: lock 2D eyeball center (does not replace C gaze→cam)."""
    eye_tracker.load_eye_tracking_state(eye_id)
    eye_tracker.sphere_center_locked_2d = True
    eye_tracker.locked_model_center_avg = (int(frame_x), int(frame_y))
    eye_tracker.prev_model_center_avg = eye_tracker.locked_model_center_avg
    if eye_tracker.last_sphere_center is not None:
        eye_tracker.calibrated_sphere_center = eye_tracker.last_sphere_center.copy()
    eye_tracker.save_eye_tracking_state(eye_id)
    print(f"[{eye_id}] Eye center locked at ({frame_x}, {frame_y}) — click IR preview to set")


def unlock_eye_sphere_centers(active_eyes):
    for eye_id in active_eyes:
        eye_tracker.load_eye_tracking_state(eye_id)
        eye_tracker.sphere_center_locked_2d = False
        eye_tracker.calibrated_sphere_center = None
        eye_tracker.save_eye_tracking_state(eye_id)
    print("Eye centers unlocked (auto-track again).")


def make_mouse_handler(layout_state):
    """Map clicks on embedded IR previews to eye-frame coordinates."""

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not layout_state.get("show_previews", True):
            return
        for eye_id, slot in layout_state.get("eyes", {}).items():
            x0, y0, w, h = slot["rect"]
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue
            ox, oy, nw, nh = slot["fit"]
            src_w, src_h = slot["src_size"]
            local_x = x - x0 - ox
            local_y = y - y0 - oy
            if not (0 <= local_x < nw and 0 <= local_y < nh):
                return
            frame_x = int(local_x * src_w / nw)
            frame_y = int(local_y * src_h / nh)
            lock_eye_sphere_center(eye_id, frame_x, frame_y)
            return

    return on_mouse


def draw_hud(canvas, lines, x=16, y=28):
    for text in lines:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24
    return y


def calibrate_gaze(active_eyes, scale_calib=None):
    """Look at screen center, then align combined gaze to front-camera forward."""
    eye_tracker.calibrate_gaze_to_external(active_eyes)
    ok = bool(eye_tracker.calibrated)
    if ok:
        if scale_calib is not None:
            scale_calib.clear_edges()
        print("C: gaze aligned to front camera. Optional: edges with arrow keys.")
    return ok


def _record_edge_sample(scale_calib, edge, gaze, tracker, width_mm, height_mm, screen_w, screen_h):
    if not eye_tracker.calibrated:
        print("Press C at screen center before edge arrows.")
        return
    if not tracker.ready or tracker._rotation is None or tracker._translation is None:
        print("Wait for ArUco pose before edge arrows.")
        return
    ok, message = scale_calib.record_edge(
        edge,
        gaze,
        eye_tracker.R_gaze_to_cam,
        tracker._rotation,
        tracker._translation,
        width_mm,
        height_mm,
        screen_w,
        screen_h,
    )
    print(message if ok else f"Edge calib failed: {message}")


def run(choice):
    left_index = choice["left"]
    right_index = choice.get("right")
    front_index = choice["front"]
    if left_index is None and right_index is None:
        print("Need at least one IR eye camera.")
        return
    if front_index is None:
        print("Need a front camera.")
        return

    if left_index is None:
        left_index, right_index = right_index, None

    active_eyes = tuple(
        eid for eid, idx in (("left", left_index), ("right", right_index)) if idx is not None
    )

    screen_w, screen_h, win_x, win_y = sp.get_window_placement()
    width_mm, height_mm = sp.get_screen_mm(screen_w, screen_h)
    tracker = sp.ScreenPoseTracker(screen_w, screen_h, width_mm, height_mm, hfov_deg=DEFAULT_HFOV)
    heatmap = GazeHeatmap(screen_w, screen_h)
    scale_calib = GazeScaleCalib()

    eye_tracker.set_show_separate_tracking_windows(False)
    eye_tracker.calibrated = False
    eye_tracker.reset_gaze_smoothing()

    readers = {}
    for eye_id, index in (("left", left_index), ("right", right_index)):
        if index is None:
            continue
        reader = CameraReader(index, width=640, height=480)
        reader.start()
        print(f"{eye_id} IR {index}: {reader.backend_name or 'opening…'}")
        readers[eye_id] = reader
        eye_tracker.reset_eye_tracking_state(eye_id)

    front_reader = CameraReader(front_index, width=640, height=480)
    front_reader.start()
    print(f"Front {front_index}: {front_reader.backend_name or 'opening…'}")

    layout_state = {"show_previews": True, "eyes": {}}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, win_x, win_y)
    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)
    # Create the window handle, then maximize (resizeWindow alone stays small on Windows).
    cv2.imshow(WINDOW_NAME, np.zeros((screen_h, screen_w, 3), dtype=np.uint8))
    cv2.waitKey(1)
    maximize_cv_window(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_handler(layout_state))

    show_previews = True
    print("GazeScreen3D ready.")
    print("1) Point front camera at this window until Pose OK")
    print("2) Click IR preview to lock eye center if the yellow circle drifts")
    print("3) Look at the CYAN CROSS at MONITOR CENTER (not the Front preview) and press C")
    print("4) Optional: look at each edge and press arrow keys (refine yaw/pitch scale)")
    print("5) Gaze heatmap follows ray ∩ ArUco plane")
    print("Q quit | C center | arrows edges | E reset edges | U unlock | M V K -/+ FOV")

    try:
        while True:
            layout_state["show_previews"] = show_previews
            layout_state["eyes"] = {}

            # --- IR eyes ---
            for eye_id, reader in readers.items():
                ret, frame = reader.read()
                if not ret:
                    continue
                flip_v = choice["flip_left"] if eye_id == "left" else choice["flip_right"]
                mirror = choice["mirror_left"] if eye_id == "left" else choice["mirror_right"]
                eye_tracker.process_frame(
                    frame,
                    eye_id=eye_id,
                    flip_vertical=flip_v,
                    flip_horizontal=mirror,
                )

            gaze = eye_tracker.refresh_combined_gaze(active_eyes)

            # --- Front + ArUco ---
            front_display = None
            ret_f, front_frame = front_reader.read()
            if ret_f and front_frame is not None:
                if choice["flip_front"]:
                    front_frame = cv2.flip(front_frame, 0)
                if choice["mirror_front"]:
                    front_frame = cv2.flip(front_frame, 1)
                fh, fw = front_frame.shape[:2]
                eye_tracker.configure_external_viewport(fw, fh)
                front_display = tracker.process(front_frame)

            hit = None
            if (
                eye_tracker.calibrated
                and gaze is not None
                and tracker.ready
                and tracker._rotation is not None
                and tracker._translation is not None
            ):
                hit = gaze_to_screen(
                    gaze,
                    eye_tracker.R_gaze_to_cam,
                    tracker._rotation,
                    tracker._translation,
                    width_mm,
                    height_mm,
                    screen_w,
                    screen_h,
                    scale_x=scale_calib.scale_x,
                    scale_y=scale_calib.scale_y,
                )
                if hit is not None:
                    heatmap.add_hit(hit["u"], hit["v"], hit["on_screen"])
                    if front_display is not None:
                        uv = project_gaze_to_front_pixels(
                            gaze,
                            eye_tracker.R_gaze_to_cam,
                            front_display.shape[1],
                            front_display.shape[0],
                            eye_tracker.EXT_FX,
                            eye_tracker.EXT_FY,
                            eye_tracker.EXT_CX,
                            eye_tracker.EXT_CY,
                            scale_x=scale_calib.scale_x,
                            scale_y=scale_calib.scale_y,
                        )
                        if uv is not None:
                            cv2.circle(
                                front_display,
                                (int(uv[0]), int(uv[1])),
                                7,
                                (0, 0, 255),
                                -1,
                            )

            # --- Compose ---
            canvas = heatmap.render_bgr()
            tracker.markers.paste_on(canvas)
            if eye_tracker.calibrated:
                draw_edge_targets(canvas, screen_w, screen_h, scale_calib.edges_done)
            else:
                draw_center_calib_target(canvas, screen_w, screen_h)

            x1, y1, x2, y2 = tracker.markers.preview_rect()
            if show_previews and front_display is not None:
                pw, ph = max(160, x2 - x1), max(120, min(360, (y2 - y1) // 2))
                preview, _, _, _, _ = fit_frame(front_display, pw, ph)
                canvas[y1 : y1 + ph, x1 : x1 + pw] = preview
                cv2.rectangle(canvas, (x1, y1), (x1 + pw, y1 + ph), (180, 180, 180), 1)
                if not eye_tracker.calibrated:
                    draw_front_preview_not_for_c(canvas, x1, y1, pw, ph)
                cv2.putText(
                    canvas,
                    "Front",
                    (x1 + 8, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                eye_y = y1 + ph + 8
                slot_w = (pw - 8) // max(1, len(active_eyes))
                for i, eye_id in enumerate(active_eyes):
                    eye_frame = eye_tracker.get_preview_frame(eye_id)
                    if eye_frame is None:
                        continue
                    ex = x1 + i * (slot_w + 8)
                    eh = min(160, y2 - eye_y)
                    if eh < 40:
                        break
                    eye_prev, ox, oy, nw, nh = fit_frame(eye_frame, slot_w, eh)
                    canvas[eye_y : eye_y + eh, ex : ex + slot_w] = eye_prev

                    locked = eye_tracker.eye_tracking_states[eye_id].get("sphere_center_locked_2d")
                    border = (0, 255, 0) if locked else (180, 180, 180)
                    cv2.rectangle(canvas, (ex, eye_y), (ex + slot_w, eye_y + eh), border, 2)
                    label = f"{eye_id} {'LOCK' if locked else 'click=center'}"
                    cv2.putText(
                        canvas,
                        label,
                        (ex + 6, eye_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    fh, fw = eye_frame.shape[:2]
                    layout_state["eyes"][eye_id] = {
                        "rect": (ex, eye_y, slot_w, eh),
                        "fit": (ox, oy, nw, nh),
                        "src_size": (fw, fh),
                    }

            locked_any = any(
                eye_tracker.eye_tracking_states[eid].get("sphere_center_locked_2d")
                for eid in active_eyes
            )
            lines = [
                f"Front {front_reader.snapshot_status()['fps']:.0f} fps  "
                f"[{front_reader.snapshot_status()['status']}]  HFOV {tracker.hfov_deg:.0f}",
                tracker.status_line() if hasattr(tracker, "status_line") else "",
            ]
            if eye_tracker.calibrated:
                lines.append("Gaze to cam: calibrated (C)")
                lines.append(scale_calib.status_line())
            else:
                lines.append("Gaze to cam: guarda croce CIANO al centro MONITOR, premi C")
            lines.append(
                "Eye center: LOCKED (U unlock)" if locked_any else "Eye center: auto (click IR preview to lock)"
            )

            if hit is not None:
                state = "ON screen" if hit["on_screen"] else "OFF screen"
                lines.append(f"Hit {state}  ({hit['u']:.0f}, {hit['v']:.0f}) px")
            elif not tracker.ready:
                lines.append("Waiting for ArUco screen pose...")
            elif not eye_tracker.calibrated:
                lines.append("Waiting for C calibration...")
            else:
                lines.append("No gaze / no intersection")

            lines.append("Q quit | C center | arrows edges | E reset edges | U M V K | -/+ FOV")
            draw_hud(canvas, [ln for ln in lines if ln], x=x1, y=36)

            cv2.imshow(WINDOW_NAME, canvas)

            key = poll_key()
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrate_gaze(active_eyes, scale_calib)
            elif key == ord("e"):
                scale_calib.clear_edges()
                print("Edge scales reset (sx=sy=1). C kept.")
            elif key == ord("u"):
                unlock_eye_sphere_centers(active_eyes)
            elif key == ord("m"):
                print(f"Markers {'ON' if tracker.markers.toggle() else 'OFF'}")
            elif key == ord("v"):
                show_previews = not show_previews
                print(f"Previews {'ON' if show_previews else 'OFF'}")
            elif key == ord("k"):
                heatmap.clear()
                print("Heatmap cleared")
            elif key in (ord("-"), ord("_")):
                tracker.set_hfov(tracker.hfov_deg - 2.0)
                print(f"HFOV {tracker.hfov_deg:.0f}")
            elif key in (ord("="), ord("+")):
                tracker.set_hfov(tracker.hfov_deg + 2.0)
                print(f"HFOV {tracker.hfov_deg:.0f}")
            elif key == ord("0"):
                tracker.set_hfov(DEFAULT_HFOV)
                print(f"HFOV reset {tracker.hfov_deg:.0f}")
            elif key == ord("h"):
                print(
                    "C = center | arrows = edges (optional scale) | E = reset edges | "
                    "click IR = lock eye center | U unlock"
                )
            else:
                edge = _arrow_to_edge(key)
                if edge is not None:
                    _record_edge_sample(
                        scale_calib,
                        edge,
                        gaze,
                        tracker,
                        width_mm,
                        height_mm,
                        screen_w,
                        screen_h,
                    )
    finally:
        for reader in readers.values():
            reader.stop()
        front_reader.stop()
        cv2.destroyAllWindows()


def main():
    choice = selection_gui()
    if not choice or choice.get("front") is None:
        return
    if choice.get("left") is None and choice.get("right") is None:
        return
    run(choice)


if __name__ == "__main__":
    main()

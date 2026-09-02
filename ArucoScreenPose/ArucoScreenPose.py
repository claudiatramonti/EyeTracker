"""
Front camera only: ArUco screen detection + camera-vs-screen pose.

HUD field meanings: see the module docstring in screen_pose.py.

Usage:
  cd ArucoScreenPose
  python ArucoScreenPose.py
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from camera_io import CameraReader
from screen_pose import (
    DEFAULT_HFOV_DEG,
    ScreenPoseTracker,
    detect_cameras,
    draw_angle_schematic,
    draw_hud,
    fit_frame,
    get_screen_mm,
    get_window_placement,
)

WINDOW_NAME = "ArUco Screen Pose"

_DEBOUNCE_SEC = 0.2
_last_char = None
_last_char_time = 0.0

_WIN32_KEY_VKS = (
    (0x51, ord("q")),
    (0x4D, ord("m")),
    (0x50, ord("p")),
    (0x30, ord("0")),
    (0xBB, ord("=")),
    (0xBD, ord("-")),
)


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
        char = key & 0xFF
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


def selection_gui():
    import tkinter as tk
    from tkinter import ttk

    cameras = detect_cameras()

    root = tk.Tk()
    root.title("ArUco Screen Pose")
    tk.Label(
        root,
        text="Solo camera frontale: ArUco + angolo rispetto allo schermo",
        font=("Arial", 12, "bold"),
    ).pack(pady=10)

    default = str(cameras[0]) if cameras else "0"
    cam_var = tk.StringVar(value=default)
    flip_var = tk.BooleanVar(value=False)
    mirror_var = tk.BooleanVar(value=False)

    controls = ttk.Frame(root)
    controls.pack(pady=6)
    tk.Label(controls, text="Front camera:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
    combo = ttk.Combobox(controls, textvariable=cam_var, width=12)
    combo.grid(row=0, column=1, padx=6, pady=4)
    combo["values"] = [str(c) for c in cameras] if cameras else [default]

    ttk.Checkbutton(root, text="Flip verticale", variable=flip_var).pack(pady=2)
    ttk.Checkbutton(root, text="Mirror orizzontale", variable=mirror_var).pack(pady=2)
    tk.Label(
        root,
        text="I marker ArUco appaiono agli angoli. La preview della camera e' al centro.",
        font=("Arial", 9),
    ).pack(pady=6)

    choice = {"camera": None, "flip": False, "mirror": False}

    def start():
        try:
            choice["camera"] = int(cam_var.get())
        except ValueError:
            choice["camera"] = 0
        choice["flip"] = flip_var.get()
        choice["mirror"] = mirror_var.get()
        root.destroy()

    tk.Button(root, text="Start", command=start).pack(pady=10)
    root.mainloop()
    return choice


def compose_view(screen_w, screen_h, tracker, camera_frame, camera_status):
    """Full window: corner markers, centered camera preview, HUD on top."""
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    tracker.markers.paste_on(canvas)

    x1, y1, x2, y2 = tracker.markers.preview_rect()
    preview_w = max(160, x2 - x1)
    preview_h = max(120, y2 - y1)
    if camera_frame is not None:
        preview = fit_frame(camera_frame, preview_w, preview_h)
        canvas[y1 : y1 + preview_h, x1 : x1 + preview_w] = preview
        cv2.rectangle(canvas, (x1, y1), (x1 + preview_w, y1 + preview_h), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            "Front camera",
            (x1 + 10, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            canvas,
            "Nessun frame dalla camera frontale",
            (x1 + 20, (y1 + y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    draw_hud(canvas, tracker, camera_status, origin=(x1, 36))
    schematic_h = 150
    schematic_y = screen_h - tracker.markers.margin - schematic_h - 8
    draw_angle_schematic(canvas, tracker.angles, x1, schematic_y, size=schematic_h)
    tracker.markers.draw_center(canvas)
    return canvas


def run(camera_index, flip=False, mirror=False, hfov_deg=DEFAULT_HFOV_DEG):
    screen_w, screen_h, win_x, win_y = get_window_placement()
    width_mm, height_mm = get_screen_mm(screen_w, screen_h)
    tracker = ScreenPoseTracker(screen_w, screen_h, width_mm, height_mm, hfov_deg=hfov_deg)

    reader = CameraReader(camera_index, width=640, height=480)
    reader.start()
    if not reader.is_opened():
        print(f"Camera {camera_index} non pronta (riprova in background).")
    else:
        print(f"Front camera {camera_index} ready ({reader.backend_name}).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, win_x, win_y)
    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)

    print("Punta la camera frontale verso questo schermo finche' i 4 marker sono visibili.")
    print("Q esci | M marker | -/+ FOV | 0 reset FOV | P stampa pose")
    print(
        f"Schermo stimato: {width_mm:.0f} x {height_mm:.0f} mm  |  HFOV iniziale {hfov_deg:.0f} deg"
    )

    last_print = 0.0
    try:
        while True:
            ret, frame = reader.read()
            camera_frame = None
            if ret and frame is not None:
                if flip:
                    frame = cv2.flip(frame, 0)
                if mirror:
                    frame = cv2.flip(frame, 1)
                camera_frame = tracker.process(frame)

            display = compose_view(screen_w, screen_h, tracker, camera_frame, reader.snapshot_status())
            cv2.imshow(WINDOW_NAME, display)

            key = poll_key()
            if key == ord("q"):
                break
            if key == ord("m"):
                visible = tracker.markers.toggle()
                print(f"Marker sugli angoli: {'ON' if visible else 'OFF'}")
            elif key in (ord("-"), ord("_")):
                tracker.set_hfov(tracker.hfov_deg - 2.0)
                print(f"HFOV {tracker.hfov_deg:.0f} deg")
            elif key in (ord("="), ord("+")):
                tracker.set_hfov(tracker.hfov_deg + 2.0)
                print(f"HFOV {tracker.hfov_deg:.0f} deg")
            elif key == ord("0"):
                tracker.set_hfov(DEFAULT_HFOV_DEG)
                print(f"HFOV reset {tracker.hfov_deg:.0f} deg")
            elif key == ord("p"):
                now = time.perf_counter()
                if now - last_print > 0.15 and tracker.angles is not None:
                    a = tracker.angles
                    p = a["camera_pos_mm"]
                    print(
                        f"yaw={a['yaw_deg']:+.2f} pitch={a['pitch_deg']:+.2f} "
                        f"roll={a['roll_deg']:+.2f} incidence={a['incidence_deg']:.2f} "
                        f"pos_mm=({p[0]:+.1f},{p[1]:+.1f},{p[2]:+.1f}) "
                        f"reproj={tracker.reprojection_error:.2f}px"
                    )
                    last_print = now
    finally:
        reader.stop()
        cv2.destroyAllWindows()


def main():
    if len(sys.argv) >= 2:
        camera_index = int(sys.argv[1])
        flip = "--flip" in sys.argv
        mirror = "--mirror" in sys.argv
        run(camera_index, flip=flip, mirror=mirror)
        return

    choice = selection_gui()
    if choice["camera"] is None:
        return
    run(choice["camera"], flip=choice["flip"], mirror=choice["mirror"])


if __name__ == "__main__":
    main()

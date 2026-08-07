"""
Stereo IR eye tracker (2 cameras) + optional front camera + fullscreen heatmap.

Self-contained: no dependency on FrontCameraTracker or 3DTracker modules.

Usage:
  cd HeatMapFrontCameraTracker
  python HeatMapFrontCameraTracker.py

Select left/right IR cameras and optional front camera in the GUI.
Gaze on the heatmap uses the combined left+right direction (VR-style fusion).

Calibration on heatmap window: C=center, B=bottom, R=right (optional)
"""

import os
import sys
import threading
import time

import cv2
import tkinter as tk
from tkinter import ttk

import eye_tracker
import gaze_screen

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_CAPTURE_MODES = (
    (
        ("DirectShow", cv2.CAP_DSHOW, None),
        ("MSMF", cv2.CAP_MSMF, None),
    )
    if sys.platform == "win32"
    else (("Auto", cv2.CAP_ANY, None),)
)

CAPTURE_TARGET_FPS = 30
RECONNECT_FAIL_THRESHOLD = 20
RECONNECT_COOLDOWN_SEC = 1.5
CAMERA_OPEN_DELAY_SEC = 0.8

# Windows fallback when OpenCV HighGUI does not have focus (terminal/other window active).
_WIN32_KEY_VKS = (
    (0x51, ord("q")),
    (0x43, ord("c")),
    (0x42, ord("b")),
    (0x52, ord("r")),
    (0x53, ord("s")),
    (0x48, ord("h")),
    (0x4B, ord("k")),
    (0x58, ord("x")),
    (0x20, ord(" ")),
)


def poll_control_key(window_name):
    """Read C/B/R/Q/etc. from OpenCV; on Windows also poll global key state."""
    key = cv2.waitKey(1)
    if key != -1:
        return key & 0xFF

    if sys.platform != "win32":
        return 255

    user32 = __import__("ctypes").windll.user32
    for vk, char in _WIN32_KEY_VKS:
        # Bit 0: key was pressed since the previous GetAsyncKeyState call.
        if user32.GetAsyncKeyState(vk) & 1:
            return char
    return 255


def configure_capture(cap, width=None, height=None, fps_request=30):
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps_request:
        cap.set(cv2.CAP_PROP_FPS, fps_request)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera_capture(index, width=None, height=None, fps_request=CAPTURE_TARGET_FPS, preferred_backend=None):
    modes = CAMERA_CAPTURE_MODES
    if preferred_backend is not None:
        modes = tuple(m for m in modes if m[0] == preferred_backend) + tuple(
            m for m in modes if m[0] != preferred_backend
        )

    for mode_name, backend, fourcc in modes:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        configure_capture(cap, width, height, fps_request)
        ret, frame = cap.read() if cap.isOpened() else (False, None)
        if ret and frame is not None:
            print(f"Camera {index} opened with {mode_name}.")
            return cap, mode_name

        cap.release()
        print(f"Camera {index}: no frame with {mode_name}, trying next backend.")

    return None, None


class CameraReader:
    """Background capture with throttled reads and auto-reconnect."""

    def __init__(self, index, width=640, height=480, fps_request=CAPTURE_TARGET_FPS):
        self.index = index
        self.width = width
        self.height = height
        self.fps_request = fps_request
        self.backend_name = None
        self.cap = None

        self.fps = 0.0
        self.status = "OFFLINE"
        self._lock = threading.Lock()
        self._latest_frame = None
        self._has_frame = False
        self._stop = threading.Event()
        self._thread = None
        self._fail_count = 0

        self._open_capture()

    def _open_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap, self.backend_name = open_camera_capture(
            self.index,
            self.width,
            self.height,
            self.fps_request,
            preferred_backend=self.backend_name,
        )
        if self.cap is not None:
            self.status = "OK"
            self._fail_count = 0
        else:
            self.status = "OFFLINE"

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self.is_opened()
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _reconnect(self):
        self.status = "RECONNECTING"
        print(f"Camera {self.index}: reconnecting ({self.backend_name or 'auto'})...")
        time.sleep(RECONNECT_COOLDOWN_SEC)
        if self._stop.is_set():
            return
        self._open_capture()
        if self.is_opened():
            print(f"Camera {self.index}: reconnected with {self.backend_name}.")
        else:
            print(f"Camera {self.index}: reconnect failed.")

    def _capture_loop(self):
        frame_count = 0
        window_start = time.perf_counter()
        frame_interval = 1.0 / CAPTURE_TARGET_FPS

        while not self._stop.is_set():
            loop_start = time.perf_counter()

            if not self.is_opened():
                self._reconnect()
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._fail_count += 1
                if self._fail_count >= RECONNECT_FAIL_THRESHOLD:
                    self._reconnect()
                time.sleep(0.05)
                continue

            self._fail_count = 0
            if self.status != "OK":
                self.status = "OK"

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - window_start
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                window_start = now

            with self._lock:
                self._latest_frame = frame
                self._has_frame = True

            sleep_time = frame_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self._lock:
            if not self._has_frame or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def snapshot_status(self):
        return {"fps": self.fps, "status": self.status, "backend": self.backend_name or "?"}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status = "OFFLINE"


def stop_all_readers(readers):
    for reader in readers.values():
        reader.stop()


def camera_label(index):
    return str(index)


def other_camera_options(cameras, *reserved):
    reserved_set = {index for index in reserved if index is not None}
    options = ["None"]
    for cam in cameras:
        if cam not in reserved_set:
            options.append(camera_label(cam))
    return options


def default_camera(cameras, *reserved):
    reserved_set = {index for index in reserved if index is not None}
    for cam in cameras:
        if cam not in reserved_set:
            return camera_label(cam)
    return "None"


def parse_camera_choice(value):
    if value is None or value == "None":
        return None
    return int(value)


def selection_gui():
    cameras = eye_tracker.detect_cameras()

    root = tk.Tk()
    root.title("HeatMap Stereo Tracker")
    tk.Label(
        root,
        text="2× IR eye cameras + optional front camera + heatmap",
        font=("Arial", 12, "bold"),
    ).pack(pady=10)

    left_var = tk.StringVar(value=camera_label(cameras[0]) if cameras else "0")
    right_var = tk.StringVar(
        value=camera_label(cameras[1]) if len(cameras) > 1 else "None"
    )
    front_var = tk.StringVar(value="None")

    controls = ttk.Frame(root)
    controls.pack(pady=5)

    def add_row(row, label, variable):
        tk.Label(controls, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        combo = ttk.Combobox(
            controls,
            textvariable=variable,
            state="readonly" if cameras else "disabled",
            width=12,
        )
        combo.grid(row=row, column=1, padx=6, pady=4)
        return combo

    left_dropdown = add_row(0, "Left eye IR:", left_var)
    right_dropdown = add_row(1, "Right eye IR:", right_var)
    front_dropdown = add_row(2, "Front camera:", front_var)

    flip_left_var = tk.BooleanVar(value=True)
    flip_right_var = tk.BooleanVar(value=True)
    mirror_left_var = tk.BooleanVar(value=False)
    mirror_right_var = tk.BooleanVar(value=False)
    flip_front_var = tk.BooleanVar(value=True)
    mirror_front_var = tk.BooleanVar(value=True)
    flip_frame = ttk.Frame(root)
    flip_frame.pack(pady=2)
    ttk.Checkbutton(flip_frame, text="Flip left (V)", variable=flip_left_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Mirror left (L/R)", variable=mirror_left_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Flip right (V)", variable=flip_right_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Mirror right (L/R)", variable=mirror_right_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Flip front (V)", variable=flip_front_var).pack(side="left", padx=6)
    ttk.Checkbutton(flip_frame, text="Mirror front (L/R)", variable=mirror_front_var).pack(side="left", padx=6)

    def current_indices():
        return (
            parse_camera_choice(left_var.get()),
            parse_camera_choice(right_var.get()),
            parse_camera_choice(front_var.get()),
        )

    def refresh(*_):
        if not cameras:
            for dropdown in (left_dropdown, right_dropdown, front_dropdown):
                dropdown.configure(values=["None"])
            left_var.set("0")
            right_var.set("None")
            front_var.set("None")
            return

        left_index, right_index, front_index = current_indices()

        left_dropdown.configure(values=[camera_label(cam) for cam in cameras])
        right_dropdown.configure(values=other_camera_options(cameras, left_index))
        front_dropdown.configure(values=other_camera_options(cameras, left_index, right_index))

        if left_var.get() not in left_dropdown["values"]:
            left_var.set(camera_label(cameras[0]))
        if right_var.get() not in right_dropdown["values"]:
            right_var.set(default_camera(cameras, left_index))
        if front_var.get() not in front_dropdown["values"]:
            front_var.set(default_camera(cameras, left_index, right_index))

    for variable in (left_var, right_var):
        variable.trace_add("write", refresh)
    refresh()

    tk.Label(
        root,
        text="Heatmap: C=center, B=bottom, R=right | Click eye window to lock sphere",
        font=("Arial", 9),
    ).pack(pady=4)

    choice = {
        "left": None,
        "right": None,
        "front": None,
        "flip_left": True,
        "flip_right": True,
        "mirror_left": False,
        "mirror_right": False,
        "flip_front": True,
        "mirror_front": True,
    }

    def start():
        choice["left"] = parse_camera_choice(left_var.get())
        choice["right"] = parse_camera_choice(right_var.get())
        choice["front"] = parse_camera_choice(front_var.get())
        choice["flip_left"] = flip_left_var.get()
        choice["flip_right"] = flip_right_var.get()
        choice["mirror_left"] = mirror_left_var.get()
        choice["mirror_right"] = mirror_right_var.get()
        choice["flip_front"] = flip_front_var.get()
        choice["mirror_front"] = mirror_front_var.get()
        root.destroy()

    tk.Button(root, text="Start", command=start).pack(pady=8)
    root.mainloop()
    return choice


def setup_eye_windows(active_eyes):
    for eye_id in active_eyes:
        eye_tracker.load_eye_tracking_state(eye_id)
        window_name = eye_tracker.eye_window_name("Tracking")
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, eye_tracker.on_mouse_frame_with_rays, eye_id)
        eye_tracker.save_eye_tracking_state(eye_id)


def run(
    left_index,
    right_index,
    front_index,
    flip_left=True,
    flip_right=True,
    mirror_left=False,
    mirror_right=False,
    flip_front=True,
    mirror_front=True,
):
    os.chdir(ROOT_DIR)

    if left_index is None and right_index is None:
        print("Select at least one eye camera.")
        return

    if left_index is None:
        left_index, right_index = right_index, None
    if left_index == right_index:
        right_index = None

    active_eyes = tuple(
        eye_id
        for eye_id, index in (("left", left_index), ("right", right_index))
        if index is not None
    )

    screen_w, screen_h = gaze_screen.get_screen_size()
    heatmap_window = "Gaze Heatmap"
    heatmap_session = gaze_screen.GazeHeatmapSession(screen_w, screen_h)
    gaze_screen.create_fullscreen_window(heatmap_window, screen_w, screen_h)

    heatmap_fps = 0.0
    heatmap_last_frame_time = time.perf_counter()

    print(f"Left eye IR:  {left_index}")
    print(f"Right eye IR: {right_index if right_index is not None else 'disabled'}")
    print(f"Front camera: {front_index if front_index is not None else 'disabled'}")
    print("Fusion: average of available eye gaze directions")
    print("Calibrazione heatmap: C = centro, B = basso, R = destro (opzionale)")
    print("Tasti: C/B/R calibrazione, H reset cal, K reset heatmap, S salva, Q esci")
    if sys.platform == "win32":
        print("Su Windows: se i tasti non rispondono, clicca sulla finestra 'Gaze Heatmap'.")

    readers = {}
    for eye_id, index in (("left", left_index), ("right", right_index)):
        if index is None:
            continue
        reader = CameraReader(index, width=640, height=480)
        reader.start()
        if not reader.is_opened():
            print(f"Warning: {eye_id} eye camera {index} not ready yet (will retry in background).")
        else:
            print(f"{eye_id.title()} eye camera {index} ready ({reader.backend_name}).")
        readers[eye_id] = reader
        eye_tracker.reset_eye_tracking_state(eye_id)
        time.sleep(CAMERA_OPEN_DELAY_SEC)

    front_reader = None
    if front_index is not None and front_index not in {left_index, right_index}:
        front_reader = CameraReader(front_index, width=640, height=480)
        front_reader.start()
        if not front_reader.is_opened():
            print(f"Warning: front camera {front_index} not ready yet (will retry in background).")
        else:
            print(f"Front camera {front_index} ready ({front_reader.backend_name}).")
        time.sleep(CAMERA_OPEN_DELAY_SEC)

    eye_tracker.calibrated = False
    eye_tracker.reset_gaze_smoothing()
    setup_eye_windows(active_eyes)

    try:
        while True:
            for eye_id, reader in readers.items():
                ret, frame = reader.read()
                if not ret:
                    continue

                flip_v = flip_left if eye_id == "left" else flip_right
                mirror = mirror_left if eye_id == "left" else mirror_right
                eye_tracker.process_frame(
                    frame,
                    eye_id=eye_id,
                    flip_vertical=flip_v,
                    flip_horizontal=mirror,
                )

            combined_gaze = eye_tracker.refresh_combined_gaze(active_eyes)

            if front_reader is not None:
                ret_ext, ext_frame = front_reader.read()
                if ret_ext:
                    fh, fw = ext_frame.shape[:2]
                    if fw != eye_tracker.EXT_WIDTH or fh != eye_tracker.EXT_HEIGHT:
                        eye_tracker.configure_external_viewport(fw, fh)

                    display = ext_frame
                    if (fw, fh) != (eye_tracker.EXT_WIDTH, eye_tracker.EXT_HEIGHT):
                        display = cv2.resize(ext_frame, (eye_tracker.EXT_WIDTH, eye_tracker.EXT_HEIGHT))

                    if eye_tracker.calibrated:
                        eye_tracker.update_gaze_circle_from_current_gaze()

                    # Draw gaze in raw camera coords, then flip for display if needed
                    # (keeps the red dot aligned with the scene after display transforms).
                    cv2.circle(display, (eye_tracker.circle_x, eye_tracker.circle_y), 8, (0, 0, 255), -1)
                    if flip_front:
                        display = cv2.flip(display, 0)
                    if mirror_front:
                        display = cv2.flip(display, 1)
                    cv2.imshow("External Camera (Gaze)", display)

            heatmap_session.update(combined_gaze)
            now = time.perf_counter()
            heatmap_fps = 0.9 * heatmap_fps + 0.1 / max(now - heatmap_last_frame_time, 1e-6)
            heatmap_last_frame_time = now

            camera_status = {eye_id: reader.snapshot_status() for eye_id, reader in readers.items()}
            if front_reader is not None:
                camera_status["front"] = front_reader.snapshot_status()

            cv2.imshow(heatmap_window, heatmap_session.render(heatmap_fps, camera_status=camera_status))
            gaze_screen.focus_window(heatmap_window)

            key = poll_control_key(heatmap_window)
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)
            elif key == ord("c"):
                if combined_gaze is None:
                    print("No combined gaze vector yet.")
                else:
                    eye_tracker.calibrate_gaze_to_external(active_eyes)
                    rotation = eye_tracker.R_gaze_to_cam if eye_tracker.calibrated else None
                    _, message = heatmap_session.set_center_calibration(
                        combined_gaze,
                        rotation=rotation,
                    )
                    print(message)
            elif key == ord("b"):
                _, message = heatmap_session.calibrate_bottom(combined_gaze)
                print(message)
            elif key == ord("r"):
                _, message = heatmap_session.calibrate_right(combined_gaze)
                print(message)
            elif key == ord("s"):
                out_path = os.path.join(ROOT_DIR, "gaze_heatmap.png")
                heatmap_session.save_png(out_path)
                print(f"Saved {out_path}")
            else:
                handled, message = heatmap_session.handle_key(key, combined_gaze)
                if handled and message:
                    print(message)
    finally:
        stop_all_readers(readers)
        if front_reader is not None:
            front_reader.stop()
        cv2.destroyAllWindows()


def main():
    choice = selection_gui()
    if choice["left"] is None and choice["right"] is None:
        return
    run(
        choice["left"],
        choice["right"],
        choice["front"],
        flip_left=choice["flip_left"],
        flip_right=choice["flip_right"],
        mirror_left=choice["mirror_left"],
        mirror_right=choice["mirror_right"],
        flip_front=choice["flip_front"],
        mirror_front=choice["mirror_front"],
    )


if __name__ == "__main__":
    main()

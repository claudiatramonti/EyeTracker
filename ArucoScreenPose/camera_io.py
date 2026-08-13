"""USB camera capture thread. HUD fps = frames / elapsed over a 1s window; status OK/OFFLINE/RECONNECTING; backend DirectShow/MSMF."""

import sys
import threading
import time

import cv2

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

            # fps for HUD: count over a 1s window
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
        """fps, status, backend for the HUD capture line."""
        return {"fps": self.fps, "status": self.status, "backend": self.backend_name or "?"}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status = "OFFLINE"

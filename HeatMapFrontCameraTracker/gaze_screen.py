"""Gaze-to-screen mapping and heatmap rendering for HeatMapFrontCameraTracker."""

import sys

import cv2
import numpy as np

MARGIN = 80
DISPLAY_SMOOTH_ALPHA = 0.3


def format_camera_status_lines(camera_status):
    labels = (("left", "L-IR"), ("right", "R-IR"), ("front", "Front"))
    parts = []
    for key, label in labels:
        if key not in camera_status:
            continue
        info = camera_status[key]
        fps = info.get("fps", 0.0)
        status = info.get("status", "?")
        parts.append(f"{label} {fps:.0f} [{status}]")

    if not parts:
        return []
    return ["Cameras: " + "  |  ".join(parts)]


def rotation_from_a_to_b(a, b):
    """Rotation matrix R such that R @ a = b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-6:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        s = np.linalg.norm(v)
    else:
        s = np.linalg.norm(v)
        v = v / s

    vx, vy, vz = v
    k = np.array([
        [0, -vz, vy],
        [vz, 0, -vx],
        [-vy, vx, 0],
    ], dtype=np.float32)

    return np.eye(3, dtype=np.float32) + k * s + (k @ k) * ((1 - c) / (s ** 2))


def normalize_direction(direction):
    if direction is None:
        return None
    direction = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    return direction / norm


def gaze_angles(direction, rotation, swap_axes=False):
    g = rotation @ direction
    if g[2] <= 1e-6:
        return None

    yaw = float(g[0] / g[2])
    pitch = float(g[1] / g[2])
    if swap_axes:
        yaw, pitch = pitch, yaw
    return yaw, pitch


def project_gaze_to_screen(direction, rotation, width, height, scale_u, scale_v, swap_axes=False):
    angles = gaze_angles(direction, rotation, swap_axes=swap_axes)
    if angles is None:
        return None

    yaw, pitch = angles
    cx = width * 0.5
    cy = height * 0.5
    u = int(np.clip(cx + scale_u * yaw, 0, width - 1))
    # Image Y grows downward; negate pitch like front-camera projection in eye_tracker.
    v = int(np.clip(cy - scale_v * pitch, 0, height - 1))
    return u, v


def default_scales(width, height, fx_arg=0, fy_arg=0):
    scale_u = fx_arg or (600.0 * (width / 640.0))
    scale_v = fy_arg or (600.0 * (height / 480.0))
    return scale_u, scale_v


def render_heatmap(accumulator, blur_sigma=25):
    if accumulator.max() <= 0:
        height, width = accumulator.shape
        return np.zeros((height, width, 3), dtype=np.uint8)

    normalized = accumulator / accumulator.max()
    gray = (normalized * 255).astype(np.uint8)

    if blur_sigma > 0:
        k = int(blur_sigma * 4) | 1
        gray = cv2.GaussianBlur(gray, (k, k), blur_sigma)

    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


class GazeHeatmapSession:
    """In-memory heatmap with C/B/R calibration."""

    def __init__(self, width, height, radius=12, fx=0, fy=0, swap_axes=False):
        self.width = width
        self.height = height
        self.radius = radius
        self.fx = fx
        self.fy = fy
        self.swap_axes = swap_axes

        self.cx = width * 0.5
        self.cy = height * 0.5
        self.scale_u, self.scale_v = default_scales(width, height, fx, fy)

        self.rotation = None
        self.center_calibrated = False
        self.vertical_calibrated = False
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self.hits = 0
        self.last_yaw_pitch = None
        self.last_point = None
        self._display_u = None
        self._display_v = None
        self._default_scale_u = self.scale_u

    @property
    def ready(self):
        return self.center_calibrated and self.vertical_calibrated

    @property
    def calibration_step(self):
        if not self.center_calibrated:
            return "center"
        if not self.vertical_calibrated:
            return "bottom"
        if self.scale_u == self._default_scale_u:
            return "right"
        return "done"

    def reset_calibration(self):
        self.rotation = None
        self.center_calibrated = False
        self.vertical_calibrated = False
        self.scale_u, self.scale_v = default_scales(self.width, self.height, self.fx, self.fy)
        self._default_scale_u = self.scale_u
        self._display_u = None
        self._display_v = None

    def reset_heatmap(self):
        self.accumulator.fill(0)
        self.hits = 0
        self._display_u = None
        self._display_v = None

    def set_center_calibration(self, direction, rotation=None):
        direction = normalize_direction(direction)
        if direction is None:
            return False, "No valid gaze direction."

        if rotation is None:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.rotation = rotation_from_a_to_b(direction, forward)
        else:
            self.rotation = np.asarray(rotation, dtype=np.float32).copy()

        self.center_calibrated = True
        self.vertical_calibrated = False
        self.scale_u, self.scale_v = default_scales(self.width, self.height, self.fx, self.fy)
        self._default_scale_u = self.scale_u
        return True, "Center OK. Look at BOTTOM edge and press B."

    def calibrate_bottom(self, direction):
        if not self.center_calibrated or self.rotation is None:
            return False, "Press C first."

        direction = normalize_direction(direction)
        if direction is None:
            return False, "No valid gaze direction."

        angles = gaze_angles(direction, self.rotation, swap_axes=self.swap_axes)
        if angles is None:
            return False, "Bottom calibration failed."

        _, pitch = angles
        if abs(pitch) < 1e-4:
            return False, "Not enough vertical change. Try X, then C and B again."

        bottom_y = self.height - MARGIN
        self.scale_v = (self.cy - bottom_y) / pitch
        self.vertical_calibrated = True
        return True, f"Bottom OK. scale_v={self.scale_v:.1f}"

    def calibrate_right(self, direction):
        if not self.center_calibrated or self.rotation is None:
            return False, "Press C first."

        direction = normalize_direction(direction)
        if direction is None:
            return False, "No valid gaze direction."

        angles = gaze_angles(direction, self.rotation, swap_axes=self.swap_axes)
        if angles is None:
            return False, "Right calibration failed."

        yaw, _ = angles
        if abs(yaw) < 1e-4:
            return False, "Not enough horizontal change. Try X, then C and R again."

        self.scale_u = (self.width - MARGIN - self.cx) / yaw
        return True, f"Right OK. scale_u={self.scale_u:.1f}"

    def toggle_swap_axes(self):
        self.swap_axes = not self.swap_axes
        return f"Swap axes: {'ON' if self.swap_axes else 'OFF'}"

    def update(self, direction):
        direction = normalize_direction(direction)
        self.last_point = None
        self.last_yaw_pitch = None

        if direction is None or self.rotation is None:
            return

        self.last_yaw_pitch = gaze_angles(direction, self.rotation, swap_axes=self.swap_axes)

        if not self.ready:
            return

        point = project_gaze_to_screen(
            direction,
            self.rotation,
            self.width,
            self.height,
            self.scale_u,
            self.scale_v,
            swap_axes=self.swap_axes,
        )
        if point is None:
            return

        u, v = point
        if self._display_u is None:
            self._display_u, self._display_v = float(u), float(v)
        else:
            alpha = DISPLAY_SMOOTH_ALPHA
            self._display_u = (1.0 - alpha) * self._display_u + alpha * u
            self._display_v = (1.0 - alpha) * self._display_v + alpha * v

        smooth_point = (int(round(self._display_u)), int(round(self._display_v)))
        self.last_point = smooth_point
        cv2.circle(self.accumulator, smooth_point, self.radius, 1.0, -1)
        self.hits += 1

    def draw_calibration_guides(self, frame):
        step = self.calibration_step
        if step == "done":
            return

        if step == "center":
            cv2.drawMarker(frame, (int(self.cx), int(self.cy)), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
            cv2.putText(frame, "Look here, press C", (int(self.cx) - 120, int(self.cy) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif step == "bottom":
            bottom_y = self.height - MARGIN
            cv2.circle(frame, (int(self.cx), bottom_y), 18, (0, 255, 255), 2)
            cv2.putText(frame, "Look here, press B", (int(self.cx) - 120, bottom_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif step == "right":
            right_x = self.width - MARGIN
            cv2.circle(frame, (int(right_x), int(self.cy)), 18, (0, 255, 255), 2)
            cv2.putText(frame, "Look here, press R (optional)", (right_x - 220, int(self.cy) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def draw_hud(self, frame, fps, camera_status=None):
        if not self.center_calibrated:
            status = "Step 1: press C at screen center"
        elif not self.vertical_calibrated:
            status = "Step 2: press B at bottom edge"
        else:
            status = "Recording - optional R at right edge"

        lines = [
            status,
            f"Hits: {self.hits}",
            f"Loop FPS: {fps:.1f}",
            "C=center  B=bottom  R=right  H=reset cal  X=swap axes",
            "K=reset heatmap  S=save  Q=quit",
        ]

        status_lines = format_camera_status_lines(camera_status) if camera_status else []
        for index, line in enumerate(status_lines):
            lines.insert(3 + index, line)

        if self.last_yaw_pitch is not None:
            yaw, pitch = self.last_yaw_pitch
            lines.insert(3 + len(status_lines), f"Yaw: {yaw:+.4f}  Pitch: {pitch:+.4f}")

        y = 30
        for text in lines:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y += 28

    def render(self, fps, camera_status=None):
        heatmap = render_heatmap(self.accumulator)
        display = cv2.addWeighted(heatmap, 0.85, np.zeros_like(heatmap), 0.15, 0)
        self.draw_hud(display, fps, camera_status=camera_status)
        self.draw_calibration_guides(display)

        if self.ready and self.last_point is not None:
            cv2.drawMarker(display, self.last_point, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

        return display

    def save_png(self, path):
        cv2.imwrite(path, render_heatmap(self.accumulator, blur_sigma=35))
        return path

    def handle_key(self, key, direction):
        if key == ord("h"):
            self.reset_calibration()
            return True, "Calibration reset."
        if key == ord("k"):
            self.reset_heatmap()
            return True, "Heatmap reset."
        if key == ord("x"):
            return True, self.toggle_swap_axes()
        if key == ord("c"):
            _, message = self.set_center_calibration(direction)
            return True, message
        if key == ord("b"):
            _, message = self.calibrate_bottom(direction)
            return True, message
        if key == ord("r"):
            _, message = self.calibrate_right(direction)
            return True, message
        return False, None


def get_screen_size():
    try:
        import tkinter as tk
    except ImportError:
        return 1920, 1080

    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height


def create_fullscreen_window(name, width, height):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width, height)
    cv2.moveWindow(name, 0, 0)
    # Exclusive fullscreen on Windows often blocks keyboard input for cv2.waitKey().
    if sys.platform != "win32":
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def focus_window(name):
    """Route keyboard input to a specific OpenCV window when supported."""
    select_window = getattr(cv2, "selectWindow", None)
    if select_window is None:
        return
    try:
        select_window(name)
    except cv2.error:
        pass

"""
Front-camera ArUco screen pose (no IR / gaze / heatmap).

PnP from four corner markers → camera position and orientation vs the screen.

Object frame (mm): origin = window center, X right, Y up, Z into the screen.
Camera is on the viewer side (negative Z), looking toward +Z.

Markers (DICT_4X4_50): 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left.

HUD (how each value is computed):
  fps          frames / elapsed_seconds over a 1s window
  HFOV         assumed fallback; or from calib fx when front_camera.npz is present
  schermo mm   GetDeviceCaps mm * (window_px / monitor_px), else px * 25.4/96
  C            camera pos = -R.T @ t   (solvePnP: P_cam = R @ P_obj + t)
  look         R.T @ [0, 0, 1]
  yaw          atan2(look_x, look_z)                    degrees, +right
  pitch        atan2(look_y, hypot(look_x, look_z))     degrees, +up
  roll         atan2(cam_up·right, cam_up·up)           degrees, +clockwise
  dist         abs(C_z) mm  (distance to the screen plane)
  incidenza    acos(clip(look · [0,0,1]))               degrees, 0 = square-on
  riproj       mean ||projectPoints(obj) - detected||   pixels
"""

from __future__ import annotations

import math
import sys
import time

import cv2
import numpy as np

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, cv2.aruco.DetectorParameters())

CORNER_MARKER_IDS = {
    0: "top-left",
    1: "top-right",
    2: "bottom-right",
    3: "bottom-left",
}

DEFAULT_HFOV_DEG = 60.0  # assumed front-camera HFOV (keys - / + / 0)
POSE_SMOOTH_ALPHA = 0.35  # exponential blend of R, t across frames
MAX_REPROJ_ERROR_PX = 12.0  # reject pose if mean reprojection exceeds this
MIN_MARKERS_FOR_POSE = 4  # all corner IDs 0-3 required for stable PnP

# On-screen corner markers for GazeScreen3D / ArucoScreenPose:
# ~14% of shorter screen edge (min 90px, max 176px).
MARKER_SIZE_DIVISOR = 7
MARKER_SIZE_MIN = 90
MARKER_SIZE_MAX = 176


def detect_cameras(max_cams=6):
    available = []
    for index in range(max_cams):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            available.append(index)
            cap.release()
        time.sleep(0.2)
    return available


def _win32_work_area():
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0):
            width = int(rect.right - rect.left)
            height = int(rect.bottom - rect.top)
            if width > 0 and height > 0:
                return width, height, int(rect.left), int(rect.top)
    except (AttributeError, OSError, ValueError):
        pass
    return None


def get_window_placement():
    work_area = _win32_work_area()
    if work_area is not None:
        return work_area
    try:
        import tkinter as tk
    except ImportError:
        return 1920, 1080, 0, 0
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height, 0, 0


def get_screen_mm(window_width, window_height):
    """HUD schermo mm = monitor_mm * (window_px / monitor_px). Fallback: px * 25.4/96."""
    monitor_w_px, monitor_h_px = window_width, window_height
    width_mm = height_mm = None

    if sys.platform == "win32":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            hdc = user32.GetDC(0)
            try:
                width_mm = float(gdi32.GetDeviceCaps(hdc, 4))
                height_mm = float(gdi32.GetDeviceCaps(hdc, 6))
                monitor_w_px = int(gdi32.GetDeviceCaps(hdc, 8) or window_width)
                monitor_h_px = int(gdi32.GetDeviceCaps(hdc, 10) or window_height)
            finally:
                user32.ReleaseDC(0, hdc)
        except (AttributeError, OSError, ValueError):
            width_mm = height_mm = None

    if not width_mm or not height_mm or width_mm < 50 or height_mm < 50:
        width_mm = window_width * 25.4 / 96.0
        height_mm = window_height * 25.4 / 96.0
        return width_mm, height_mm

    scale_x = window_width / max(monitor_w_px, 1)
    scale_y = window_height / max(monitor_h_px, 1)
    return width_mm * scale_x, height_mm * scale_y

def camera_matrix(width, height, hfov_deg):
    """K from assumed HFOV (fallback when no chessboard calib file)."""
    from camera_intrinsics import camera_matrix_from_hfov

    return camera_matrix_from_hfov(width, height, hfov_deg)


def pixel_to_mm(px, py, width_px, height_px, width_mm, height_mm):
    """Pixel (top-left origin) → object mm (Y up): top row → +Y mm."""
    x = (float(px) / width_px) * width_mm - 0.5 * width_mm
    y = (0.5 - float(py) / height_px) * height_mm
    return np.array([x, y, 0.0], dtype=np.float64)


def _orthonormalize(rotation):
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, 2] *= -1
        rotation = u @ vt
    return rotation


def _smooth_pose(prev_rotation, prev_translation, rotation, translation, alpha):
    if prev_rotation is None or prev_translation is None:
        return rotation, translation
    blended = (1.0 - alpha) * prev_rotation + alpha * rotation
    return _orthonormalize(blended), (1.0 - alpha) * prev_translation + alpha * translation


def pose_from_rt(rotation, translation):
    """
    solvePnP gives P_cam = R @ P_obj + t. Then:
      C    = -R.T @ t                         camera position in screen mm
      look = R.T @ [0,0,1]                    optical axis in screen coords
      yaw  = atan2(look_x, look_z)            deg, +right
      pitch= atan2(look_y, hypot(look_x,z))     deg, +up
      roll = atan2(up·right, up·plane_up)     deg, +clockwise
      dist = abs(C_z)                         mm to screen plane
      inc  = acos(look · [0,0,1])             deg vs screen normal
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    camera_pos = -rotation.T @ translation
    look_dir = rotation.T @ np.array([0.0, 0.0, 1.0])
    look_norm = np.linalg.norm(look_dir)
    if look_norm > 1e-9:
        look_dir = look_dir / look_norm

    yaw_deg = math.degrees(math.atan2(look_dir[0], look_dir[2]))
    pitch_deg = math.degrees(math.atan2(look_dir[1], math.hypot(look_dir[0], look_dir[2])))

    cam_up = rotation.T @ np.array([0.0, -1.0, 0.0])
    plane_right = np.array([look_dir[2], 0.0, -look_dir[0]])
    plane_right_n = np.linalg.norm(plane_right)
    if plane_right_n < 1e-6:
        roll_deg = 0.0
    else:
        plane_right = plane_right / plane_right_n
        plane_up = np.cross(plane_right, look_dir)
        roll_deg = math.degrees(
            math.atan2(np.dot(cam_up, plane_right), np.dot(cam_up, plane_up))
        )

    screen_normal_into = np.array([0.0, 0.0, 1.0])
    incidence = math.degrees(
        math.acos(float(np.clip(np.dot(look_dir, screen_normal_into), -1.0, 1.0)))
    )

    return {
        "camera_pos_mm": camera_pos,
        "look_dir": look_dir,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "incidence_deg": incidence,
        "distance_mm": float(abs(camera_pos[2])),
    }


def _reprojection_error(object_points, image_points, rvec, tvec, camera_k, dist_coeffs):
    """HUD riproj = mean pixel distance between detected corners and projectPoints(obj, R, t)."""
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_k, dist_coeffs)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)))


def _pose_candidate(rvec, tvec, object_points, image_points, camera_k, dist_coeffs, prev_rotation=None):
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec.reshape(3)
    error = _reprojection_error(object_points, image_points, rvec, tvec, camera_k, dist_coeffs)
    interpreted = pose_from_rt(rotation, translation)
    pos = interpreted["camera_pos_mm"]
    look = interpreted["look_dir"]
    score = error
    if pos[2] >= 0:
        score += 1000.0
    if look[2] <= 0:
        score += 500.0
    if prev_rotation is not None:
        cosine = np.clip((np.trace(prev_rotation.T @ rotation) - 1.0) * 0.5, -1.0, 1.0)
        score += math.degrees(math.acos(cosine)) * 0.15
    return {
        "rvec": rvec,
        "tvec": tvec,
        "rotation": rotation,
        "translation": translation,
        "reprojection_error": error,
        "score": score,
        "angles": interpreted,
    }


def estimate_screen_pose(object_points, image_points, camera_k, dist_coeffs=None, prev_rvec=None, prev_tvec=None):
    """PnP from marker 3D mm + 2D corners. Prefers viewer-side Z<0, look +Z, low reproj."""
    if len(object_points) < 4:
        return None
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    prev_rotation = None
    if prev_rvec is not None:
        prev_rotation, _ = cv2.Rodrigues(np.asarray(prev_rvec, dtype=np.float64))

    candidates = []

    try:
        ok, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_points.reshape(-1, 1, 3),
            image_points.reshape(-1, 1, 2),
            camera_k,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if ok:
            for rvec, tvec in zip(rvecs, tvecs):
                candidates.append(
                    _pose_candidate(
                        rvec, tvec, object_points, image_points, camera_k, dist_coeffs, prev_rotation
                    )
                )
    except cv2.error:
        pass

    use_guess = prev_rvec is not None and prev_tvec is not None
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_k,
        dist_coeffs,
        None if not use_guess else np.asarray(prev_rvec, dtype=np.float64),
        None if not use_guess else np.asarray(prev_tvec, dtype=np.float64),
        useExtrinsicGuess=use_guess,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        candidates.append(
            _pose_candidate(rvec, tvec, object_points, image_points, camera_k, dist_coeffs, prev_rotation)
        )

    valid = [c for c in candidates if c["reprojection_error"] <= MAX_REPROJ_ERROR_PX]
    if not valid:
        return None
    best = min(valid, key=lambda item: item["score"])
    return best


class CornerMarkers:
    """On-screen ArUco IDs 0-3. PnP uses the inner black-square corners in mm."""

    def __init__(self, screen_width, screen_height, margin=12):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.margin = margin
        self.visible = True
        short_edge = min(screen_width, screen_height)
        self.marker_size = min(
            MARKER_SIZE_MAX,
            max(MARKER_SIZE_MIN, short_edge // MARKER_SIZE_DIVISOR),
        )
        self.border = max(10, self.marker_size // 12)
        self._markers_bgr = self._build_markers()
        self._corner_positions = self._build_corner_positions()
        self.inner_corners_px = self._build_inner_corners()

    def _build_markers(self):
        inner = self.marker_size - 2 * self.border
        markers = {}
        for marker_id in (0, 1, 2, 3):
            raw = cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, inner)
            bordered = cv2.copyMakeBorder(
                raw,
                self.border,
                self.border,
                self.border,
                self.border,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            markers[marker_id] = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        return markers

    def _build_corner_positions(self):
        size = self.marker_size
        margin = self.margin
        w, h = self.screen_width, self.screen_height
        return {
            0: (margin, margin),
            1: (w - size - margin, margin),
            2: (w - size - margin, h - size - margin),
            3: (margin, h - size - margin),
        }

    def _build_inner_corners(self):
        """Pixel corners of the black ArUco square (detector order: TL, TR, BR, BL)."""
        b = self.border
        s = self.marker_size
        inner = {}
        for marker_id, (x, y) in self._corner_positions.items():
            inner[marker_id] = np.array(
                [
                    [x + b, y + b],
                    [x + s - b, y + b],
                    [x + s - b, y + s - b],
                    [x + b, y + s - b],
                ],
                dtype=np.float64,
            )
        return inner

    def object_points_mm(self, width_mm, height_mm):
        points = {}
        for marker_id, corners_px in self.inner_corners_px.items():
            points[marker_id] = np.array(
                [
                    pixel_to_mm(px, py, self.screen_width, self.screen_height, width_mm, height_mm)
                    for px, py in corners_px
                ],
                dtype=np.float64,
            )
        return points

    def screen_corners_mm(self, width_mm, height_mm):
        half_w = 0.5 * width_mm
        half_h = 0.5 * height_mm
        return np.array(
            [
                [-half_w, half_h, 0.0],
                [half_w, half_h, 0.0],
                [half_w, -half_h, 0.0],
                [-half_w, -half_h, 0.0],
            ],
            dtype=np.float64,
        )

    def preview_rect(self, pad=16):
        size = self.marker_size
        margin = self.margin
        return (
            margin + size + pad,
            margin + size + pad,
            self.screen_width - margin - size - pad,
            self.screen_height - margin - size - pad,
        )

    def draw_center(self, frame):
        """Screen-center guide for calibration (object-frame origin)."""
        cx = self.screen_width // 2
        cy = self.screen_height // 2
        short = min(self.screen_width, self.screen_height)
        scale = short / 1080.0
        draw_center_marker(frame, cx, cy, color=(0, 255, 255), scale=scale)

    def paste_on(self, frame):
        if not self.visible:
            return frame
        for marker_id, (x, y) in self._corner_positions.items():
            marker = self._markers_bgr[marker_id]
            h, w = marker.shape[:2]
            frame[y : y + h, x : x + w] = marker
            label_y = y - 8 if y > 24 else y + h + 18
            cv2.putText(
                frame,
                f"ID{marker_id}",
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    def toggle(self):
        self.visible = not self.visible
        return self.visible


class ScreenPoseTracker:
    """Detect corner markers and estimate camera-vs-screen pose."""

    def __init__(
        self,
        screen_width,
        screen_height,
        screen_width_mm,
        screen_height_mm,
        hfov_deg=DEFAULT_HFOV_DEG,
        intrinsics_path=None,
        use_calib=False,
    ):
        from camera_intrinsics import default_calib_path, load_intrinsics

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_width_mm = screen_width_mm
        self.screen_height_mm = screen_height_mm
        self.hfov_deg = hfov_deg
        self.markers = CornerMarkers(screen_width, screen_height)
        self.object_points = self.markers.object_points_mm(screen_width_mm, screen_height_mm)

        self.intrinsics = None
        self.intrinsics_source = "hfov"
        self._dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        # Opt-in: pass use_calib=True to load camera_calib/front_camera.npz
        if use_calib:
            path = intrinsics_path or default_calib_path()
            loaded = load_intrinsics(path)
            if loaded is not None:
                self.intrinsics = loaded
                self.intrinsics_source = "calib"
                print(f"Front intrinsics: {loaded['path']}")
                if "rms" in loaded:
                    print(f"  calib RMS {loaded['rms']:.4f} px")
            else:
                print(f"Front intrinsics: none ({path}) — using HFOV {hfov_deg:.0f}°")

        self.ready = False
        self.markers_found = 0
        self.corner_ids_found = set()
        self.pose = None
        self.angles = None
        self.reprojection_error = None
        self._rvec = None
        self._tvec = None
        self._rotation = None
        self._translation = None

    def set_hfov(self, hfov_deg):
        """Manual HFOV override (ignored while chessboard calib is loaded)."""
        if self.intrinsics is not None:
            print("HFOV keys ignored — using calibrated K/dist (re-run calibrate_front_camera.py to change).")
            return
        self.hfov_deg = float(np.clip(hfov_deg, 20.0, 120.0))
        self._rvec = None
        self._tvec = None

    def camera_model(self, frame_width, frame_height):
        """Return (K, dist, source, effective_hfov) for this frame size."""
        from camera_intrinsics import resolve_camera_model

        k, dist, source, hfov = resolve_camera_model(
            frame_width,
            frame_height,
            self.hfov_deg,
            intrinsics=self.intrinsics,
        )
        self.intrinsics_source = source
        self._dist_coeffs = dist
        if source == "calib" and hfov is not None:
            self.hfov_deg = float(hfov)
        return k, dist, source, hfov

    def process(self, frame):
        """Detect ArUco → PnP → smooth → draw screen outline and axes."""
        out = frame.copy()
        corners, ids, _ = _DETECTOR.detectMarkers(out)
        self.markers_found = 0 if ids is None else len(ids)
        self.corner_ids_found = set()
        self.ready = False
        self.pose = None
        self.angles = None
        self.reprojection_error = None

        if ids is None:
            self._draw_status(out)
            return out

        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        flat_ids = [int(i[0]) for i in ids]
        self.corner_ids_found = {mid for mid in flat_ids if mid in CORNER_MARKER_IDS}

        object_pts = []
        image_pts = []
        for idx, marker_id in enumerate(flat_ids):
            if marker_id not in self.object_points:
                continue
            center = np.mean(corners[idx].reshape(-1, 2), axis=0).astype(int)
            cv2.putText(
                out,
                f"ID{marker_id} {CORNER_MARKER_IDS[marker_id]}",
                (int(center[0]) - 40, int(center[1]) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            object_pts.extend(self.object_points[marker_id])
            image_pts.extend(np.asarray(corners[idx], dtype=np.float64).reshape(-1, 2))

        if len(self.corner_ids_found) >= MIN_MARKERS_FOR_POSE and len(object_pts) >= 4:
            h, w = out.shape[:2]
            camera_k, dist_coeffs, _, _ = self.camera_model(w, h)
            pose = estimate_screen_pose(
                object_pts,
                image_pts,
                camera_k,
                dist_coeffs=dist_coeffs,
                prev_rvec=self._rvec,
                prev_tvec=self._tvec,
            )
            if pose is not None:
                rotation, translation = _smooth_pose(
                    self._rotation,
                    self._translation,
                    pose["rotation"],
                    pose["translation"],
                    POSE_SMOOTH_ALPHA,
                )
                rvec, _ = cv2.Rodrigues(rotation)
                tvec = translation.reshape(3, 1)
                self._rotation = rotation
                self._translation = translation
                self._rvec = rvec
                self._tvec = tvec
                self.pose = pose
                self.reprojection_error = pose["reprojection_error"]
                self.angles = pose_from_rt(rotation, translation)
                self.ready = True
                self._draw_pose(out, rvec, tvec, camera_k, dist_coeffs)

        self._draw_status(out)
        return out

    def _draw_pose(self, frame, rvec, tvec, camera_k, dist_coeffs=None):
        dist = (
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
            if dist_coeffs is not None
            else self._dist_coeffs
        )
        screen_corners = self.markers.screen_corners_mm(self.screen_width_mm, self.screen_height_mm)
        projected, _ = cv2.projectPoints(screen_corners, rvec, tvec, camera_k, dist)
        pts = projected.reshape(-1, 2).astype(int)
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, tuple(pts[0]), tuple(pts[2]), (0, 180, 180), 1, cv2.LINE_AA)
        cv2.line(frame, tuple(pts[1]), tuple(pts[3]), (0, 180, 180), 1, cv2.LINE_AA)

        axis_len = min(self.screen_width_mm, self.screen_height_mm) * 0.18
        cv2.drawFrameAxes(frame, camera_k, dist, rvec, tvec, axis_len)

        if self.angles is not None:
            origin_2d, _ = cv2.projectPoints(
                np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                rvec,
                tvec,
                camera_k,
                dist,
            )
            ox, oy = origin_2d.reshape(2).astype(int)
            draw_center_marker(frame, ox, oy, color=(255, 0, 255), scale=0.55)

    def _draw_status(self, frame):
        lines = []
        if self.ready and self.angles is not None:
            a = self.angles
            p = a["camera_pos_mm"]
            lines.append(
                f"Pose OK  {len(self.corner_ids_found)}/4  err={self.reprojection_error:.1f}px"
            )
            lines.append(
                f"yaw {a['yaw_deg']:+.1f}  pitch {a['pitch_deg']:+.1f}  "
                f"roll {a['roll_deg']:+.1f}  inc {a['incidence_deg']:.1f}"
            )
            lines.append(f"cam mm  X {p[0]:+.0f}  Y {p[1]:+.0f}  dist {a['distance_mm']:.0f}")
            color = (0, 255, 0)
        elif self.corner_ids_found:
            missing = sorted(set(CORNER_MARKER_IDS) - self.corner_ids_found)
            miss_txt = ",".join(str(m) for m in missing) if missing else "?"
            lines.append(
                f"ArUco: {len(self.corner_ids_found)}/4 — need all corners (missing ID {miss_txt})"
            )
            color = (0, 255, 255)
        elif self.markers_found:
            lines.append(f"ArUco: {self.markers_found} marker, ID 0-3 are needed")
            color = (0, 165, 255)
        else:
            lines.append("ArUco: no marker, point the front camera at the screen")
            color = (0, 0, 255)

        y = 24
        for text in lines:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            y += 22

    def status_line(self):
        if self.ready and self.angles is not None:
            a = self.angles
            return (
                f"yaw {a['yaw_deg']:+.1f}  pitch {a['pitch_deg']:+.1f}  "
                f"roll {a['roll_deg']:+.1f}  dist {a['distance_mm']:.0f} mm"
            )
        if self.corner_ids_found:
            n = len(self.corner_ids_found)
            if n < MIN_MARKERS_FOR_POSE:
                return f"ArUco {n}/4 — need 4/4"
            return f"ArUco {n}/4"
        return "ArUco: waiting..."


def draw_center_marker(frame, x, y, color=(0, 255, 255), scale=1.0):
    """High-contrast crosshair + ring for the screen / pose origin."""
    ix, iy = int(x), int(y)
    cross = max(20, int(44 * scale))
    ring = max(14, int(30 * scale))
    outline = max(2, int(3 * scale))
    cv2.drawMarker(frame, (ix, iy), (0, 0, 0), cv2.MARKER_CROSS, cross + 8, outline + 1, cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), ring + 3, (0, 0, 0), outline + 1, cv2.LINE_AA)
    cv2.drawMarker(frame, (ix, iy), color, cv2.MARKER_CROSS, cross, max(2, outline - 1), cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), ring, color, max(2, outline - 1), cv2.LINE_AA)
    cv2.circle(frame, (ix, iy), max(3, int(5 * scale)), color, -1, cv2.LINE_AA)


def fit_frame(frame, width, height):
    fh, fw = frame.shape[:2]
    scale = min(width / fw, height / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ox = (width - new_w) // 2
    oy = (height - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas


def draw_angle_schematic(canvas, angles, origin_x, origin_y, size=160):
    """Top-view (yaw) and side-view (pitch) diagrams of camera vs screen."""
    if angles is None:
        return canvas

    gap = 18
    panel_w = size
    panel_h = size
    yaw_x = origin_x
    pitch_x = origin_x + panel_w + gap
    y0 = origin_y

    def panel(x, title):
        cv2.rectangle(canvas, (x, y0), (x + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(canvas, (x, y0), (x + panel_w, y0 + panel_h), (90, 90, 90), 1)
        cv2.putText(
            canvas,
            title,
            (x + 8, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    panel(yaw_x, "Top (yaw)")
    panel(pitch_x, "Side (pitch)")

    pos = angles["camera_pos_mm"]
    look = angles["look_dir"]
    span = max(abs(pos[0]), abs(pos[1]), abs(pos[2]), 1.0) * 1.4

    def to_px(panel_origin, a, b):
        cx = panel_origin[0] + panel_w * 0.5
        cy = panel_origin[1] + panel_h * 0.58
        scale = (min(panel_w, panel_h) * 0.32) / span
        return int(cx + a * scale), int(cy - b * scale)

    # Top view: X right, viewer side (-Z) drawn upward.
    screen_l = to_px((yaw_x, y0), -span * 0.7, 0.0)
    screen_r = to_px((yaw_x, y0), span * 0.7, 0.0)
    cv2.line(canvas, screen_l, screen_r, (0, 255, 255), 2, cv2.LINE_AA)
    cam = to_px((yaw_x, y0), pos[0], -pos[2])
    look_end = to_px(
        (yaw_x, y0),
        pos[0] + look[0] * span * 0.5,
        -(pos[2] + look[2] * span * 0.5),
    )
    cv2.circle(canvas, cam, 5, (0, 200, 255), -1)
    cv2.arrowedLine(canvas, cam, look_end, (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(
        canvas,
        f"{angles['yaw_deg']:+.1f} deg",
        (yaw_x + 8, y0 + panel_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    def side_px(z_mm, y_mm):
        # Viewer side (-Z) to the left, up is -Y.
        return to_px((pitch_x, y0), -z_mm, -y_mm)

    cv2.line(canvas, side_px(0.0, -span * 0.55), side_px(0.0, span * 0.55), (0, 255, 255), 2, cv2.LINE_AA)
    cam_s = side_px(pos[2], pos[1])
    look_s = side_px(pos[2] + look[2] * span * 0.5, pos[1] + look[1] * span * 0.5)
    cv2.circle(canvas, cam_s, 5, (0, 200, 255), -1)
    cv2.arrowedLine(canvas, cam_s, look_s, (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(
        canvas,
        f"{angles['pitch_deg']:+.1f} deg",
        (pitch_x + 8, y0 + panel_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_hud(frame, tracker, camera_status, extra_lines=None, origin=None):
    """Top overlay: capture stats, assumed HFOV/screen mm, then pose angles."""
    lines = []
    if camera_status:
        fps = camera_status.get("fps", 0.0)
        status = camera_status.get("status", "?")
        backend = camera_status.get("backend", "?")
        lines.append(f"Front {fps:.0f} fps  [{status}]  {backend}")
    source = getattr(tracker, "intrinsics_source", "hfov")
    if source == "calib":
        lines.append(
            f"K calib  HFOV~{tracker.hfov_deg:.0f} deg  |  "
            f"schermo {tracker.screen_width_mm:.0f}x{tracker.screen_height_mm:.0f} mm"
        )
    else:
        lines.append(
            f"HFOV {tracker.hfov_deg:.0f} deg  |  "
            f"schermo {tracker.screen_width_mm:.0f}x{tracker.screen_height_mm:.0f} mm"
        )
    if tracker.ready and tracker.angles is not None:
        a = tracker.angles
        lines.append(tracker.status_line())
        lines.append(f"incidenza {a['incidence_deg']:.1f} deg   riproj {tracker.reprojection_error:.2f} px")
    else:
        lines.append(tracker.status_line())
    if source == "calib":
        lines.append("Q esci  |  M marker  |  P stampa  (FOV keys off: using calib)")
    else:
        lines.append("Q esci  |  M marker  |  -/+ FOV  |  0 reset  |  P stampa")
    if extra_lines:
        lines.extend(extra_lines)

    x, y = (16, 28) if origin is None else origin
    for text in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
        y += 26
    return y

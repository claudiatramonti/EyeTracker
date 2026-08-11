"""
ArUco screen registration for the front camera.

Marker layout (print and place on monitor bezel/corners):
  ID 0 = top-left     ID 1 = top-right
  ID 3 = bottom-left  ID 2 = bottom-right

Generate printable markers:
  python generate_aruco_markers.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# OpenCV 4.7+ API
_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, cv2.aruco.DetectorParameters())

# Marker ID → monitor pixel corner (set when screen size is known)
CORNER_MARKER_IDS = {
    0: "top-left",
    1: "top-right",
    2: "bottom-right",
    3: "bottom-left",
}

DEFAULT_MARKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aruco_markers")
CORNER_WINDOW_NAME = "ArUco Screen Corners"

# On-screen corner markers: ~20% of shorter screen edge (min 180px, max 480px).
MARKER_SIZE_DIVISOR = 5
MARKER_SIZE_MIN = 180
MARKER_SIZE_MAX = 480


def marker_center(corners_entry):
    """Center of one detected marker (4 corner points)."""
    pts = np.asarray(corners_entry, dtype=np.float32).reshape(-1, 2)
    return np.mean(pts, axis=0)


def screen_corners_for_size(width, height):
    return {
        0: (0.0, 0.0),
        1: (float(width), 0.0),
        2: (float(width), float(height)),
        3: (0.0, float(height)),
    }


def compute_homography(corners, ids, screen_width, screen_height):
    """
    Build homography from detected corner markers to monitor pixel coords.
    Needs at least 4 known marker IDs (0–3).
    """
    if ids is None or len(ids) < 4:
        return None

    id_to_screen = screen_corners_for_size(screen_width, screen_height)
    flat_ids = [int(i[0]) for i in ids]

    src = []
    dst = []
    for idx, marker_id in enumerate(flat_ids):
        if marker_id not in id_to_screen:
            continue
        src.append(marker_center(corners[idx]))
        dst.append(id_to_screen[marker_id])

    if len(src) < 4:
        return None

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return homography


def draw_screen_outline(frame, homography, screen_width, screen_height, color=(0, 255, 255), thickness=2):
    """Draw monitor rectangle projected into the front-camera image."""
    if homography is None:
        return frame

    monitor_corners = np.array(
        [
            [0, 0],
            [screen_width, 0],
            [screen_width, screen_height],
            [0, screen_height],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    try:
        projected = cv2.perspectiveTransform(monitor_corners, np.linalg.inv(homography))
    except cv2.error:
        return frame

    pts = projected.reshape(-1, 2).astype(int)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    return frame


class ArucoScreenTracker:
    """Detect ArUco markers on front-camera frames and estimate screen homography."""

    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.homography = None
        self.markers_found = 0
        self.corner_ids_found = set()
        self.ready = False

    def process(self, frame):
        """
        Detect markers, draw overlays, update homography.
        Returns annotated BGR frame (copy).
        """
        out = frame.copy()
        corners, ids, _ = _DETECTOR.detectMarkers(out)

        self.markers_found = 0 if ids is None else len(ids)
        self.corner_ids_found = set()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
            flat_ids = [int(i[0]) for i in ids]
            self.corner_ids_found = {mid for mid in flat_ids if mid in CORNER_MARKER_IDS}

            for idx, marker_id in enumerate(flat_ids):
                if marker_id not in CORNER_MARKER_IDS:
                    continue
                center = marker_center(corners[idx]).astype(int)
                label = f"ID{marker_id} {CORNER_MARKER_IDS[marker_id]}"
                cv2.putText(
                    out,
                    label,
                    (int(center[0]) - 40, int(center[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            homography = compute_homography(corners, ids, self.screen_width, self.screen_height)
            if homography is not None:
                self.homography = homography
                self.ready = len(self.corner_ids_found) >= 4
                draw_screen_outline(
                    out,
                    homography,
                    self.screen_width,
                    self.screen_height,
                )

        self._draw_status(out)
        return out

    def _draw_status(self, frame):
        if self.ready:
            text = f"ArUco: OK ({len(self.corner_ids_found)}/4 corners, homography active)"
            color = (0, 255, 0)
        elif self.corner_ids_found:
            text = f"ArUco: {len(self.corner_ids_found)}/4 corner markers visible"
            color = (0, 255, 255)
        elif self.markers_found:
            text = f"ArUco: {self.markers_found} marker(s), need IDs 0-3 at corners"
            color = (0, 165, 255)
        else:
            text = "ArUco: no markers (print IDs 0-3, place on monitor corners)"
            color = (0, 0, 255)

        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    def status_line(self):
        if self.ready:
            return "ArUco homography: OK (4/4)"
        if self.corner_ids_found:
            return f"ArUco: {len(self.corner_ids_found)}/4 corners"
        if self.markers_found:
            return f"ArUco: {self.markers_found} marker(s)"
        return "ArUco: none"


def gaze_to_front_camera_pixels(
    gaze_direction,
    R_gaze_to_cam,
    cam_width,
    cam_height,
    cam_cx=None,
    cam_cy=None,
    cam_fx=600.0,
    cam_fy=600.0,
):
    """Project fused IR gaze into front-camera pixel coordinates."""
    direction = np.asarray(gaze_direction, dtype=np.float32)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None

    rotation = np.asarray(R_gaze_to_cam, dtype=np.float32)
    g = rotation @ (direction / norm)
    if g[2] <= 1e-6:
        return None

    cx = cam_width * 0.5 if cam_cx is None else cam_cx
    cy = cam_height * 0.5 if cam_cy is None else cam_cy
    u = float(cx + cam_fx * (g[0] / g[2]))
    v = float(cy - cam_fy * (g[1] / g[2]))
    return u, v


def front_camera_to_screen(u, v, homography, screen_width, screen_height):
    """Map a front-camera pixel to monitor coordinates using ArUco homography."""
    if homography is None:
        return None

    point = np.array([[[u, v]]], dtype=np.float32)
    try:
        mapped = cv2.perspectiveTransform(point, homography)
    except cv2.error:
        return None

    x, y = mapped[0, 0]
    if not np.isfinite(x) or not np.isfinite(y):
        return None

    sx = int(np.clip(round(x), 0, screen_width - 1))
    sy = int(np.clip(round(y), 0, screen_height - 1))
    return sx, sy


def project_gaze_to_monitor_pixels(
    gaze_direction,
    R_gaze_to_cam,
    homography,
    screen_width,
    screen_height,
    cam_width,
    cam_height,
    cam_cx=None,
    cam_cy=None,
    cam_fx=600.0,
    cam_fy=600.0,
):
    """
    Full chain: IR gaze → front-camera UV → homography → monitor pixel (x, y).

    Requires R_gaze_to_cam from pressing C (links IR space to front camera).
    Requires homography from 4/4 ArUco corner markers on the front camera.
    """
    uv = gaze_to_front_camera_pixels(
        gaze_direction,
        R_gaze_to_cam,
        cam_width,
        cam_height,
        cam_cx=cam_cx,
        cam_cy=cam_cy,
        cam_fx=cam_fx,
        cam_fy=cam_fy,
    )
    if uv is None:
        return None
    return front_camera_to_screen(uv[0], uv[1], homography, screen_width, screen_height)


def generate_marker_sheet(output_dir, marker_ids=(0, 1, 2, 3), side_pixels=320, margin=24):
    """Save one PNG per marker ID for printing."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for marker_id in marker_ids:
        image = cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, side_pixels)
        bordered = cv2.copyMakeBorder(
            image,
            margin,
            margin,
            margin,
            margin,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        path = os.path.join(output_dir, f"aruco_id{marker_id}.png")
        cv2.imwrite(path, bordered)
        paths.append(path)
    return paths


def _ensure_marker_pngs(marker_dir=DEFAULT_MARKER_DIR):
    """Create marker PNGs on disk if missing."""
    expected = [os.path.join(marker_dir, f"aruco_id{marker_id}.png") for marker_id in (0, 1, 2, 3)]
    if all(os.path.isfile(path) for path in expected):
        return expected
    print(f"Generating ArUco markers in {marker_dir} ...")
    return generate_marker_sheet(marker_dir)


class ScreenCornerMarkers:
    """
    Show ArUco IDs 0–3 at monitor corners on screen (no printing needed).

    Use overlay_on() on the heatmap, or open_corner_window() for the ArUco-only test.
    """

    def __init__(self, screen_width, screen_height, marker_dir=DEFAULT_MARKER_DIR, margin=12):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.margin = margin
        self.visible = True
        short_edge = min(screen_width, screen_height)
        self.marker_size = min(
            MARKER_SIZE_MAX,
            max(MARKER_SIZE_MIN, short_edge // MARKER_SIZE_DIVISOR),
        )
        self._markers_bgr = self._load_markers(marker_dir)
        self._corner_positions = self._build_corner_positions()
        self._corner_frame = self._build_corner_frame()
        self._window_open = False

    def _load_markers(self, marker_dir):
        paths = _ensure_marker_pngs(marker_dir)
        markers = {}
        for marker_id, path in zip((0, 1, 2, 3), paths):
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            source_h = image.shape[0]
            interp = cv2.INTER_AREA if source_h > self.marker_size else cv2.INTER_CUBIC
            markers[marker_id] = cv2.resize(
                image,
                (self.marker_size, self.marker_size),
                interpolation=interp,
            )
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

    def _paste_markers(self, frame):
        for marker_id, (x, y) in self._corner_positions.items():
            marker = self._markers_bgr.get(marker_id)
            if marker is None:
                continue
            h, w = marker.shape[:2]
            y2, x2 = y + h, x + w
            if y2 > frame.shape[0] or x2 > frame.shape[1]:
                continue
            frame[y:y2, x:x2] = marker

            label = f"ID{marker_id}"
            cv2.putText(
                frame,
                label,
                (x, y - 6 if y > 20 else y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (x, y - 6 if y > 20 else y + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def _build_corner_frame(self):
        frame = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        self._paste_markers(frame)
        cv2.putText(
            frame,
            "ArUco corner markers — point front camera at this screen (Q to quit test)",
            (40, self.screen_height - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return frame

    def overlay_on(self, frame):
        """Paste corner markers onto an existing BGR frame (e.g. heatmap)."""
        if not self.visible:
            return frame
        out = frame.copy()
        self._paste_markers(out)
        return out

    def corner_frame(self):
        """Fullscreen frame for the dedicated corner window."""
        if not self.visible:
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        return self._corner_frame.copy()

    def open_corner_window(self):
        cv2.namedWindow(CORNER_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(CORNER_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self._window_open = True
        self.refresh_corner_window()

    def refresh_corner_window(self):
        if not self._window_open:
            return
        cv2.imshow(CORNER_WINDOW_NAME, self.corner_frame())

    def close_corner_window(self):
        if self._window_open:
            cv2.destroyWindow(CORNER_WINDOW_NAME)
            self._window_open = False

    def toggle(self):
        self.visible = not self.visible
        return self.visible

    def hud_safe_zone(self, pad=16):
        """
        Horizontal band cleared of corner ArUco markers (for centered top HUD).
        Returns dict with left, top, width.
        """
        size = self.marker_size
        margin = self.margin
        left = margin + size + pad
        right = self.screen_width - margin - size - pad
        width = max(280, right - left)
        return {
            "left": left,
            "top": 30,
            "width": width,
        }

    def status_line(self):
        state = "ON" if self.visible else "OFF"
        return f"ArUco corners on screen: {state} (M toggle)"

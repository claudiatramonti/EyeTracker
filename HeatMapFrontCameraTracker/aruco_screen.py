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
import time

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
MARKER_SOURCE_SIDE = 320
MARKER_SOURCE_BORDER = 24
MIN_POSE_MARKERS = 2
MAX_POSE_REPROJECTION_ERROR_PX = 12.0


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


def default_camera_matrix(width, height, fx=600.0, fy=600.0, cx=None, cy=None):
    """Approximate pinhole intrinsics; replace with calibrated values when available."""
    cx = width * 0.5 if cx is None else float(cx)
    cy = height * 0.5 if cy is None else float(cy)
    return np.array(
        [[float(fx), 0.0, cx], [0.0, float(fy), cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def screen_marker_code_corners(screen_width, screen_height, marker_size, margin):
    """
    Marker ID -> four detected ArUco-code corners in screen coordinates.

    The displayed PNG contains a white border. Detection corners surround only
    the inner generated code, so object points must use that inset.
    """
    full_source_side = MARKER_SOURCE_SIDE + 2 * MARKER_SOURCE_BORDER
    inset = marker_size * (MARKER_SOURCE_BORDER / full_source_side)
    code_size = marker_size * (MARKER_SOURCE_SIDE / full_source_side)
    top_left_positions = {
        0: (margin, margin),
        1: (screen_width - marker_size - margin, margin),
        2: (
            screen_width - marker_size - margin,
            screen_height - marker_size - margin,
        ),
        3: (margin, screen_height - marker_size - margin),
    }

    result = {}
    for marker_id, (x, y) in top_left_positions.items():
        x0 = float(x + inset)
        y0 = float(y + inset)
        x1 = x0 + float(code_size)
        y1 = y0 + float(code_size)
        result[marker_id] = np.array(
            [[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]],
            dtype=np.float32,
        )
    return result


def estimate_screen_pose(
    corners,
    ids,
    object_corners_by_id,
    camera_matrix,
    distortion_coefficients=None,
):
    """Estimate screen->camera pose from at least two visible corner markers."""
    if ids is None:
        return None

    object_points = []
    image_points = []
    visible_ids = []
    for marker_corners, marker_id_entry in zip(corners, ids):
        marker_id = int(marker_id_entry[0])
        object_corners = object_corners_by_id.get(marker_id)
        if object_corners is None:
            continue
        detected = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        if not np.all(np.isfinite(detected)):
            continue
        object_points.extend(object_corners)
        image_points.extend(detected)
        visible_ids.append(marker_id)

    if len(set(visible_ids)) < MIN_POSE_MARKERS:
        return None

    object_points = np.asarray(object_points, dtype=np.float32)
    image_points = np.asarray(image_points, dtype=np.float32)
    distortion = (
        np.zeros((5, 1), dtype=np.float64)
        if distortion_coefficients is None
        else np.asarray(distortion_coefficients, dtype=np.float64)
    )

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            np.asarray(camera_matrix, dtype=np.float64),
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not success:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                np.asarray(camera_matrix, dtype=np.float64),
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error:
        return None

    if not success or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    if tvec[2, 0] <= 1e-6 or np.linalg.det(rotation) < 0.0:
        return None

    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        distortion,
    )
    residual = projected.reshape(-1, 2) - image_points
    reprojection_error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    if (
        not np.isfinite(reprojection_error)
        or reprojection_error > MAX_POSE_REPROJECTION_ERROR_PX
    ):
        return None

    return {
        "rotation": rotation.astype(np.float64),
        "translation": tvec.reshape(3).astype(np.float64),
        "rvec": rvec.reshape(3).astype(np.float64),
        "reprojection_error": reprojection_error,
        "marker_ids": set(visible_ids),
    }


def compute_homography(
    corners,
    ids,
    screen_width,
    screen_height,
    object_corners_by_id=None,
):
    """
    Build homography from detected marker pixels to monitor pixel coordinates.
    Needs at least 4 known marker IDs (0–3).

    When marker object corners are supplied, all 16 code corners are used.
    This avoids the old systematic error from mapping marker centers to the
    extreme monitor corners.
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
        if object_corners_by_id is None:
            src.append(marker_center(corners[idx]))
            dst.append(id_to_screen[marker_id])
            continue

        object_corners = object_corners_by_id.get(marker_id)
        if object_corners is None:
            continue
        detected_corners = np.asarray(corners[idx], dtype=np.float32).reshape(4, 2)
        src.extend(detected_corners)
        dst.extend(np.asarray(object_corners, dtype=np.float32)[:, :2])

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
    """Detect markers and estimate screen homography plus screen->camera 6DoF pose."""

    def __init__(
        self,
        screen_width,
        screen_height,
        marker_size=None,
        marker_margin=12,
        cam_width=640,
        cam_height=480,
        cam_fx=600.0,
        cam_fy=600.0,
        cam_cx=None,
        cam_cy=None,
        distortion_coefficients=None,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        short_edge = min(screen_width, screen_height)
        self.marker_size = marker_size or min(
            MARKER_SIZE_MAX,
            max(MARKER_SIZE_MIN, short_edge // MARKER_SIZE_DIVISOR),
        )
        self.marker_margin = marker_margin
        self.object_corners_by_id = screen_marker_code_corners(
            screen_width,
            screen_height,
            self.marker_size,
            marker_margin,
        )
        self.camera_matrix = default_camera_matrix(
            cam_width,
            cam_height,
            fx=cam_fx,
            fy=cam_fy,
            cx=cam_cx,
            cy=cam_cy,
        )
        self.distortion_coefficients = (
            np.zeros((5, 1), dtype=np.float64)
            if distortion_coefficients is None
            else np.asarray(distortion_coefficients, dtype=np.float64)
        )
        self.homography = None
        self.markers_found = 0
        self.corner_ids_found = set()
        self.ready = False
        self.pose_ready = False
        self.pose_rotation = None
        self.pose_translation = None
        self.pose_reprojection_error = None
        self.pose_marker_ids = set()
        self.pose_timestamp = None

    def configure_camera(
        self,
        width,
        height,
        fx=600.0,
        fy=600.0,
        cx=None,
        cy=None,
    ):
        self.camera_matrix = default_camera_matrix(
            width,
            height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )

    def process(self, frame):
        """
        Detect markers, draw overlays, update homography.
        Returns annotated BGR frame (copy).
        """
        out = frame.copy()
        corners, ids, _ = _DETECTOR.detectMarkers(out)

        self.markers_found = 0 if ids is None else len(ids)
        self.corner_ids_found = set()
        self.ready = False
        self.pose_ready = False
        self.pose_marker_ids = set()

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

            homography = compute_homography(
                corners,
                ids,
                self.screen_width,
                self.screen_height,
                object_corners_by_id=self.object_corners_by_id,
            )
            if homography is not None:
                self.homography = homography
                self.ready = len(self.corner_ids_found) >= 4
                draw_screen_outline(
                    out,
                    homography,
                    self.screen_width,
                    self.screen_height,
                )

            pose = estimate_screen_pose(
                corners,
                ids,
                self.object_corners_by_id,
                self.camera_matrix,
                self.distortion_coefficients,
            )
            if pose is not None:
                self.pose_rotation = pose["rotation"]
                self.pose_translation = pose["translation"]
                self.pose_reprojection_error = pose["reprojection_error"]
                self.pose_marker_ids = pose["marker_ids"]
                self.pose_timestamp = time.perf_counter()
                self.pose_ready = True

        self._draw_status(out)
        return out

    def screen_point_in_camera(self, u, v):
        """Direction from front camera origin to one screen point."""
        point_camera = self.screen_point_position_in_camera(u, v)
        if point_camera is None:
            return None
        norm = np.linalg.norm(point_camera)
        if not np.isfinite(norm) or norm < 1e-9:
            return None
        return point_camera / norm

    def screen_point_position_in_camera(self, u, v):
        """Position of one screen point in front-camera coordinates."""
        if not self.pose_ready:
            return None
        point_screen = np.array([float(u), float(v), 0.0], dtype=np.float64)
        point_camera = self.pose_rotation @ point_screen + self.pose_translation
        if not np.all(np.isfinite(point_camera)):
            return None
        return point_camera

    def project_camera_ray_to_screen(self, direction_camera, origin_camera=None):
        """Intersect a front-camera-frame ray with the screen plane z=0."""
        if not self.pose_ready:
            return None
        direction_camera = np.asarray(direction_camera, dtype=np.float64)
        norm = np.linalg.norm(direction_camera)
        if (
            direction_camera.shape != (3,)
            or not np.all(np.isfinite(direction_camera))
            or norm < 1e-9
        ):
            return None
        direction_camera = direction_camera / norm
        if origin_camera is None:
            origin_camera = np.zeros(3, dtype=np.float64)
        origin_camera = np.asarray(origin_camera, dtype=np.float64)
        if origin_camera.shape != (3,) or not np.all(np.isfinite(origin_camera)):
            return None

        rotation_camera_to_screen = self.pose_rotation.T
        origin_screen = rotation_camera_to_screen @ (
            origin_camera - self.pose_translation
        )
        direction_screen = rotation_camera_to_screen @ direction_camera
        if abs(direction_screen[2]) < 1e-9:
            return None
        distance = -origin_screen[2] / direction_screen[2]
        if not np.isfinite(distance) or distance <= 0.0:
            return None

        point = origin_screen + distance * direction_screen
        if not np.all(np.isfinite(point)):
            return None
        return (
            int(np.clip(round(point[0]), 0, self.screen_width - 1)),
            int(np.clip(round(point[1]), 0, self.screen_height - 1)),
        )

    def _draw_status(self, frame):
        if self.pose_ready:
            text = (
                f"ArUco: pose 6DoF OK ({len(self.pose_marker_ids)}/4, "
                f"err {self.pose_reprojection_error:.1f}px)"
            )
            color = (0, 255, 0)
        elif self.ready:
            text = f"ArUco: homography OK ({len(self.corner_ids_found)}/4), pose unavailable"
            color = (0, 255, 255)
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
        if self.pose_ready:
            return (
                f"ArUco pose: OK ({len(self.pose_marker_ids)}/4, "
                f"err {self.pose_reprojection_error:.1f}px)"
            )
        if self.ready:
            return "ArUco homography: OK, pose unavailable"
        if self.corner_ids_found:
            return f"ArUco: {len(self.corner_ids_found)}/4 corners"
        if self.markers_found:
            return f"ArUco: {self.markers_found} marker(s)"
        return "ArUco: none"


class GazePoseMapper:
    """
    Calibrate the fixed rotation from IR-gaze coordinates to front-camera coordinates.

    Each five-point sample pairs an IR gaze direction with the direction from the
    front camera to that known screen target. At runtime the current ArUco pose
    converts the rotated gaze ray into an intersection with the screen plane.
    """

    REQUIRED_LABELS = frozenset(("center", "top", "bottom", "left", "right"))

    def __init__(self):
        self.samples = {}
        self.rotation_gaze_to_camera = None
        self.origin_gaze_in_camera = np.zeros(3, dtype=np.float64)
        self.angular_rms_deg = None

    @property
    def calibrated(self):
        return (
            self.rotation_gaze_to_camera is not None
            and self.REQUIRED_LABELS.issubset(self.samples)
        )

    def reset(self):
        self.samples = {}
        self.rotation_gaze_to_camera = None
        self.origin_gaze_in_camera = np.zeros(3, dtype=np.float64)
        self.angular_rms_deg = None

    def add_calibration_sample(
        self,
        label,
        gaze_direction,
        screen_point,
        screen_tracker,
    ):
        if label not in self.REQUIRED_LABELS:
            return False, f"Etichetta posa non valida: {label}"
        if screen_tracker is None or not screen_tracker.pose_ready:
            return False, "Posa 6DoF non salvata: servono almeno 2 marker ArUco visibili."

        gaze = np.asarray(gaze_direction, dtype=np.float64)
        gaze_norm = np.linalg.norm(gaze)
        if gaze.shape != (3,) or not np.all(np.isfinite(gaze)) or gaze_norm < 1e-9:
            return False, "Posa 6DoF non salvata: vettore gaze non valido."
        gaze = gaze / gaze_norm

        target_position = screen_tracker.screen_point_position_in_camera(*screen_point)
        if target_position is None:
            return False, "Posa 6DoF non salvata: target schermo non proiettabile."

        candidate_samples = dict(self.samples)
        candidate_samples[label] = (gaze, target_position)
        candidate_rotation = self.rotation_gaze_to_camera
        candidate_origin = self.origin_gaze_in_camera.copy()
        candidate_error = self.angular_rms_deg

        if self.REQUIRED_LABELS.issubset(candidate_samples):
            ordered = sorted(self.REQUIRED_LABELS)
            gaze_rows = np.asarray(
                [candidate_samples[name][0] for name in ordered],
                dtype=np.float64,
            )
            target_positions = np.asarray(
                [candidate_samples[name][1] for name in ordered],
                dtype=np.float64,
            )
            target_norms = np.linalg.norm(target_positions, axis=1)
            if np.any(target_norms < 1e-9):
                return False, "Calibrazione posa 6DoF degenerata."
            target_rows = target_positions / target_norms[:, None]
            covariance = gaze_rows.T @ target_rows
            try:
                u, _, vt = np.linalg.svd(covariance)
            except np.linalg.LinAlgError:
                return False, "Calibrazione posa 6DoF degenerata."
            candidate_rotation = vt.T @ u.T
            if np.linalg.det(candidate_rotation) < 0.0:
                vt[-1, :] *= -1.0
                candidate_rotation = vt.T @ u.T

            # Treat normalized gaze rays as ideal pinhole image coordinates.
            # solvePnP then recovers the rigid gaze-frame pose inside the fixed
            # front-camera frame, including the physical eye/camera offset.
            candidate_origin = np.zeros(3, dtype=np.float64)
            if np.all(gaze_rows[:, 2] > 1e-6):
                ideal_image_points = gaze_rows[:, :2] / gaze_rows[:, 2, None]
                try:
                    pnp_ok, rotation_camera_to_gaze_vec, translation_camera_to_gaze = (
                        cv2.solvePnP(
                            target_positions,
                            ideal_image_points,
                            np.eye(3, dtype=np.float64),
                            np.zeros((5, 1), dtype=np.float64),
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                    )
                except cv2.error:
                    pnp_ok = False

                if pnp_ok:
                    rotation_camera_to_gaze, _ = cv2.Rodrigues(
                        rotation_camera_to_gaze_vec
                    )
                    estimated_rotation = rotation_camera_to_gaze.T
                    estimated_origin = -estimated_rotation @ (
                        translation_camera_to_gaze.reshape(3)
                    )
                    if (
                        np.all(np.isfinite(estimated_origin))
                        and np.linalg.det(estimated_rotation) > 0.0
                        and np.linalg.norm(estimated_origin) <= 800.0
                        and -800.0 <= estimated_origin[2] <= 400.0
                    ):
                        candidate_rotation = estimated_rotation
                        candidate_origin = estimated_origin

            target_rows = target_positions - candidate_origin
            target_row_norms = np.linalg.norm(target_rows, axis=1)
            if np.any(target_row_norms < 1e-9):
                return False, "Calibrazione posa 6DoF degenerata."
            target_rows = target_rows / target_row_norms[:, None]

            predicted = (candidate_rotation @ gaze_rows.T).T
            dots = np.clip(np.sum(predicted * target_rows, axis=1), -1.0, 1.0)
            angular_errors = np.degrees(np.arccos(dots))
            candidate_error = float(np.sqrt(np.mean(angular_errors * angular_errors)))
            if not np.isfinite(candidate_error) or candidate_error > 15.0:
                return (
                    False,
                    f"Calibrazione posa incoerente ({candidate_error:.1f}° RMS): "
                    "ripeti C e i quattro bordi senza muovere gli occhiali.",
                )

        self.samples = candidate_samples
        self.rotation_gaze_to_camera = candidate_rotation
        self.origin_gaze_in_camera = candidate_origin
        self.angular_rms_deg = candidate_error
        count = len(self.REQUIRED_LABELS.intersection(self.samples))
        if self.calibrated:
            return True, f"Posa gaze 6DoF pronta ({self.angular_rms_deg:.1f}° RMS)."
        return True, f"Campione posa 6DoF {label} salvato ({count}/5)."

    def project(self, gaze_direction, screen_tracker):
        if not self.calibrated or screen_tracker is None or not screen_tracker.pose_ready:
            return None
        gaze = np.asarray(gaze_direction, dtype=np.float64)
        norm = np.linalg.norm(gaze)
        if gaze.shape != (3,) or not np.all(np.isfinite(gaze)) or norm < 1e-9:
            return None
        direction_camera = self.rotation_gaze_to_camera @ (gaze / norm)
        return screen_tracker.project_camera_ray_to_screen(
            direction_camera,
            origin_camera=self.origin_gaze_in_camera,
        )

    def status_line(self):
        count = len(self.REQUIRED_LABELS.intersection(self.samples))
        if self.calibrated:
            return f"Gaze 6DoF: pronto ({self.angular_rms_deg:.1f}° RMS)"
        return f"Gaze 6DoF: calibrazione {count}/5"


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

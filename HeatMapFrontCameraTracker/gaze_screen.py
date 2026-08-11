"""Gaze-to-screen mapping and heatmap rendering for HeatMapFrontCameraTracker."""

import sys

import cv2
import numpy as np

import aruco_screen
from input_poll import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_UP

MARGIN = 80
DISPLAY_SMOOTH_ALPHA = 0.55
MIN_CALIBRATION_EXCURSION = 0.02

EDGE_KEY_LABELS = {
    "top": "freccia SU",
    "bottom": "freccia GIU",
    "left": "freccia SIN",
    "right": "freccia DES",
}
EDGE_KEY_SHORT = {
    "top": "U",
    "bottom": "D",
    "left": "L",
    "right": "R",
}


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
    a = normalize_direction(a)
    b = normalize_direction(b)
    if a is None or b is None:
        return np.eye(3, dtype=np.float32)

    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if sine < 1e-6:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis -= a * np.dot(axis, a)
        axis /= np.linalg.norm(axis)
        return (2.0 * np.outer(axis, axis) - np.eye(3)).astype(np.float32)

    axis = cross / sine
    ax, ay, az = axis
    k = np.array(
        [[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32)
        + k * sine
        + (k @ k) * (1.0 - cosine)
    ).astype(np.float32)


def normalize_direction(direction):
    if direction is None:
        return None
    direction = np.asarray(direction, dtype=np.float32)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        return None
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-6:
        return None
    return direction / norm


def gaze_angles(direction, rotation, swap_axes=False):
    if direction is None or rotation is None:
        return None
    g = rotation @ direction
    if not np.all(np.isfinite(g)) or g[2] <= 1e-6:
        return None

    yaw = float(g[0] / g[2])
    pitch = float(g[1] / g[2])
    if swap_axes:
        yaw, pitch = pitch, yaw
    return yaw, pitch


def _samples_by_edge(cal_samples):
    by_edge = {}
    for sample in cal_samples:
        by_edge[sample.get("edge", "center")] = sample
    return by_edge


def _triangle_transform(source, target):
    """Precompute a barycentric transform for one gaze/screen triangle."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (
        source.shape != (3, 2)
        or target.shape != (3, 2)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
    ):
        return None

    edge_a = source[1] - source[0]
    edge_b = source[2] - source[0]
    norm_a = float(np.linalg.norm(edge_a))
    norm_b = float(np.linalg.norm(edge_b))
    if norm_a < MIN_CALIBRATION_EXCURSION or norm_b < MIN_CALIBRATION_EXCURSION:
        return None

    signed_area = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
    normalized_area = abs(float(signed_area)) / (norm_a * norm_b)
    if not np.isfinite(normalized_area) or normalized_area < 0.08:
        return None

    system = np.vstack([source.T, np.ones(3, dtype=np.float64)])
    determinant = float(np.linalg.det(system))
    if not np.isfinite(determinant) or abs(determinant) < 1e-8:
        return None
    target_system = np.vstack([target.T, np.ones(3, dtype=np.float64)])
    target_determinant = float(np.linalg.det(target_system))
    return {
        "source": source,
        "target": target,
        "inverse": np.linalg.inv(system),
        "orientation": np.sign(determinant * target_determinant),
    }


def fit_gaze_mapping(cal_samples):
    """
    Build four piecewise-affine triangles around the center sample.

    Every calibration anchor maps exactly to its screen anchor. Separate
    top/bottom and left/right triangles preserve asymmetric eye travel caused
    by an oblique camera mount without forcing artificial orthogonality.
    """
    by_edge = _samples_by_edge(cal_samples)
    required = {"center", "top", "bottom", "left", "right"}
    if not required.issubset(by_edge):
        return None

    def gaze_point(edge):
        sample = by_edge[edge]
        return [sample["yaw"], sample["pitch"]]

    def screen_point(edge):
        sample = by_edge[edge]
        return [sample["u"], sample["v"]]

    triangles = []
    for vertical, horizontal in (
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    ):
        triangle = _triangle_transform(
            [gaze_point("center"), gaze_point(vertical), gaze_point(horizontal)],
            [screen_point("center"), screen_point(vertical), screen_point(horizontal)],
        )
        if triangle is None:
            return None
        triangles.append(triangle)

    orientations = {triangle["orientation"] for triangle in triangles}
    if 0.0 in orientations or len(orientations) != 1:
        return None
    return triangles


def _project_piecewise_feature(yaw, pitch, mapping):
    """Map one gaze feature using the triangle containing it (or nearest one)."""
    query = np.array([yaw, pitch, 1.0], dtype=np.float64)
    best_weights = None
    best_triangle = None
    best_score = -np.inf

    for triangle in mapping:
        weights = triangle["inverse"] @ query
        score = float(np.min(weights))
        if score > best_score:
            best_score = score
            best_weights = weights
            best_triangle = triangle

    if best_triangle is None:
        return None
    return best_weights @ best_triangle["target"]


def project_gaze_mapping(direction, rotation, mapping, width, height, swap_axes=False):
    angles = gaze_angles(direction, rotation, swap_axes=swap_axes)
    if angles is None or mapping is None:
        return None

    yaw, pitch = angles
    point = _project_piecewise_feature(yaw, pitch, mapping)
    if point is None or not np.all(np.isfinite(point)):
        return None
    u = int(np.clip(round(point[0]), 0, width - 1))
    v = int(np.clip(round(point[1]), 0, height - 1))
    return u, v


def calibration_targets(width, height, margin=MARGIN):
    cx = width * 0.5
    cy = height * 0.5
    return {
        "top": (cx, margin),
        "bottom": (cx, height - margin),
        "left": (margin, cy),
        "right": (width - margin, cy),
    }


CALIBRATION_EDGE_ORDER = ("top", "bottom", "left", "right")


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
    """In-memory heatmap with multi-point gaze calibration (C + arrow keys)."""

    REQUIRED_EDGE_COUNT = 4

    def __init__(self, width, height, radius=12, fx=0, fy=0, swap_axes=False):
        self.width = width
        self.height = height
        self.radius = radius
        self.fx = fx
        self.fy = fy
        self.swap_axes = swap_axes

        self.cx = width * 0.5
        self.cy = height * 0.5

        self.rotation = None
        self.center_calibrated = False
        self.cal_samples = []
        self.calibrated_edges = set()
        self.calibration_mapping = None
        self.accumulator = np.zeros((height, width), dtype=np.float32)
        self.hits = 0
        self.last_yaw_pitch = None
        self.last_point = None
        self.preview_uv = None
        self.last_cal_debug = None
        self._display_u = None
        self._display_v = None

    @property
    def mapping_ready(self):
        return (
            self.center_calibrated
            and self.calibration_mapping is not None
            and len(self.calibrated_edges) >= self.REQUIRED_EDGE_COUNT
        )

    @property
    def ready(self):
        return self.mapping_ready

    @property
    def calibration_step(self):
        if not self.center_calibrated:
            return "center"
        for edge in CALIBRATION_EDGE_ORDER:
            if edge not in self.calibrated_edges:
                return edge
        return "done"

    def calibration_status_line(self):
        if not self.center_calibrated:
            return "Cal: press C at center"
        done = "".join(EDGE_KEY_SHORT[edge] for edge in CALIBRATION_EDGE_ORDER if edge in self.calibrated_edges)
        pending = "".join(EDGE_KEY_SHORT[edge] for edge in CALIBRATION_EDGE_ORDER if edge not in self.calibrated_edges)
        if self.mapping_ready:
            return f"Cal: C + {done} (mappa 5 punti OK)"
        return f"Cal: C ok, need {pending or 'frecce'} ({len(self.calibrated_edges)}/4 edges)"

    def reset_calibration(self):
        self.rotation = None
        self.center_calibrated = False
        self.cal_samples = []
        self.calibrated_edges = set()
        self.calibration_mapping = None
        self._display_u = None
        self._display_v = None
        self.preview_uv = None
        self.last_cal_debug = None

    def _record_calibration_point(self, direction, target_u, target_v, edge_name):
        direction = normalize_direction(direction)
        if direction is None or self.rotation is None:
            return False, "No valid gaze direction."

        angles = gaze_angles(direction, self.rotation, swap_axes=self.swap_axes)
        if angles is None:
            return False, f"{edge_name} calibration failed (gaze behind camera)."

        yaw, pitch = angles
        excursion = float(np.hypot(yaw, pitch))
        if excursion < MIN_CALIBRATION_EXCURSION:
            return False, (
                f"{edge_name.capitalize()} non salvato: spostamento troppo piccolo "
                f"({excursion:.3f}). Guarda il bordo fisico e ripremi."
            )

        new_sample = {
            "edge": edge_name,
            "yaw": yaw,
            "pitch": pitch,
            "u": float(target_u),
            "v": float(target_v),
        }
        candidate_samples = [
            sample for sample in self.cal_samples
            if sample["edge"] != edge_name
        ]
        candidate_samples.append(new_sample)
        candidate_edges = set(self.calibrated_edges)
        candidate_edges.add(edge_name)
        candidate_mapping = fit_gaze_mapping(candidate_samples)

        if (
            len(candidate_edges) >= self.REQUIRED_EDGE_COUNT
            and candidate_mapping is None
        ):
            return False, (
                f"{edge_name.capitalize()} non valido: i vettori di calibrazione "
                "sono quasi allineati. Mantieni la testa ferma, guarda il bordo e ripremi."
            )

        self.cal_samples = candidate_samples
        self.calibrated_edges = candidate_edges
        self.calibration_mapping = candidate_mapping

        mapped = project_gaze_mapping(
            direction,
            self.rotation,
            self.calibration_mapping,
            self.width,
            self.height,
            swap_axes=self.swap_axes,
        ) if self.calibration_mapping is not None else None

        err_px = 0.0
        if mapped is not None:
            err_px = float(np.hypot(mapped[0] - target_u, mapped[1] - target_v))

        self.last_cal_debug = {
            "edge": edge_name,
            "yaw": yaw,
            "pitch": pitch,
            "target_u": float(target_u),
            "target_v": float(target_v),
            "mapped_u": float(mapped[0]) if mapped else None,
            "mapped_v": float(mapped[1]) if mapped else None,
            "error_px": err_px,
        }

        if self.mapping_ready:
            return True, (
                f"{edge_name.capitalize()} OK: yaw={yaw:+.3f} pitch={pitch:+.3f} "
                f"-> ({int(target_u)},{int(target_v)}) err={err_px:.0f}px | Mappa pronta"
            )

        next_edge = self.calibration_step
        if next_edge == "done":
            return True, (
                f"{edge_name.capitalize()} OK: yaw={yaw:+.3f} pitch={pitch:+.3f} "
                f"-> ({int(target_u)},{int(target_v)}) err={err_px:.0f}px"
            )
        key_hint = EDGE_KEY_LABELS.get(next_edge, "?")
        return True, (
            f"{edge_name.capitalize()} OK: yaw={yaw:+.3f} pitch={pitch:+.3f} "
            f"-> ({int(target_u)},{int(target_v)}) err={err_px:.0f}px | "
            f"Next: {next_edge} ({key_hint})"
        )

    def reset_heatmap(self):
        self.accumulator.fill(0)
        self.hits = 0
        self._display_u = None
        self._display_v = None

    def set_center_calibration(self, direction, rotation=None):
        direction = normalize_direction(direction)
        if direction is None:
            return False, "No valid gaze direction."

        # Always zero yaw/pitch at center in IR space (never reuse R_gaze_to_cam).
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.rotation = rotation_from_a_to_b(direction, forward)

        self.center_calibrated = True
        self.cal_samples = [
            {
                "edge": "center",
                "yaw": 0.0,
                "pitch": 0.0,
                "u": float(self.cx),
                "v": float(self.cy),
            }
        ]
        self.calibrated_edges = set()
        self.calibration_mapping = None
        self._display_u = None
        self._display_v = None
        self.preview_uv = None
        self.last_cal_debug = None
        self.reset_heatmap()
        return True, "Center OK. Calibrate edges with arrow keys (up/down/left/right)."

    @property
    def calibrating(self):
        return self.center_calibrated and not self.mapping_ready

    def calibrate_edge(self, direction, edge_name):
        if not self.center_calibrated or self.rotation is None:
            return False, "Press C at center first."

        targets = calibration_targets(self.width, self.height)
        if edge_name not in targets:
            return False, f"Unknown edge {edge_name}."

        target_u, target_v = targets[edge_name]
        return self._record_calibration_point(direction, target_u, target_v, edge_name)

    def calibrate_top(self, direction):
        return self.calibrate_edge(direction, "top")

    def calibrate_bottom(self, direction):
        return self.calibrate_edge(direction, "bottom")

    def calibrate_left(self, direction):
        return self.calibrate_edge(direction, "left")

    def calibrate_right(self, direction):
        return self.calibrate_edge(direction, "right")

    def toggle_swap_axes(self):
        self.swap_axes = not self.swap_axes
        self.reset_calibration()
        return f"Swap axes: {'ON' if self.swap_axes else 'OFF'}; calibration reset."

    def _project_point(self, direction):
        if self.calibration_mapping is not None and self.mapping_ready:
            return project_gaze_mapping(
                direction,
                self.rotation,
                self.calibration_mapping,
                self.width,
                self.height,
                swap_axes=self.swap_axes,
            )
        return None

    def calibration_debug_lines(self):
        if not self.center_calibrated:
            return []

        lines = []
        if self.preview_uv is not None:
            u, v = self.preview_uv
            nx = (u - self.cx) / max(self.cx, 1.0)
            ny = (self.cy - v) / max(self.cy, 1.0)
            lines.append(f"Preview px=({int(u)},{int(v)})  norm=({nx:+.2f},{ny:+.2f})")

        if self.last_cal_debug is not None:
            d = self.last_cal_debug
            lines.append(
                f"Last {d['edge']}: yaw={d['yaw']:+.3f} pitch={d['pitch']:+.3f} "
                f"target=({int(d['target_u'])},{int(d['target_v'])}) err={d['error_px']:.0f}px"
            )

        if self.calibration_mapping is not None:
            lines.append("Mapping piecewise: 4 triangoli validi")

        return lines

    def draw_calibration_debug(self, frame):
        """Visual debug during C + arrow calibration."""
        cx, cy = int(self.cx), int(self.cy)
        cv2.drawMarker(frame, (cx, cy), (200, 200, 200), cv2.MARKER_CROSS, 16, 1)

        for sample in self.cal_samples:
            edge = sample.get("edge", "?")
            tu, tv = int(sample["u"]), int(sample["v"])
            if edge == "center":
                continue
            cv2.line(frame, (cx, cy), (tu, tv), (0, 180, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, (tu, tv), 10, (0, 220, 0), 2)
            label = f"{edge[0].upper()} y={sample['yaw']:+.2f} p={sample['pitch']:+.2f}"
            cv2.putText(
                frame,
                label,
                (tu + 12, tv),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )

        if self.preview_uv is not None:
            pu, pv = int(round(self.preview_uv[0])), int(round(self.preview_uv[1]))
            cv2.line(frame, (cx, cy), (pu, pv), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (pu, pv), 8, (0, 255, 255), 2)

        if self.last_cal_debug is not None and self.last_cal_debug.get("mapped_u") is not None:
            mu = int(round(self.last_cal_debug["mapped_u"]))
            mv = int(round(self.last_cal_debug["mapped_v"]))
            cv2.drawMarker(frame, (mu, mv), (255, 128, 0), cv2.MARKER_TILTED_CROSS, 12, 2)

    def _apply_gaze_point(self, point):
        if point is None:
            return False

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
        return True

    def update(self, direction):
        direction = normalize_direction(direction)
        self.last_point = None
        self.last_yaw_pitch = None

        if direction is None or self.rotation is None:
            return

        self.last_yaw_pitch = gaze_angles(direction, self.rotation, swap_axes=self.swap_axes)

        if self.mapping_ready:
            self._apply_gaze_point(self._project_point(direction))
            return

        if self.calibrating:
            self.preview_uv = None
            return

        if self.ready:
            self._apply_gaze_point(self._project_point(direction))

    def update_via_aruco(self, direction, homography, R_gaze_to_cam, cam_width, cam_height, cam_cx, cam_cy, cam_fx, cam_fy):
        """Map gaze through front camera + ArUco homography (no C/B/R scales)."""
        direction = normalize_direction(direction)
        self.last_point = None
        self.last_yaw_pitch = None

        if direction is None or homography is None:
            return False

        point = aruco_screen.project_gaze_to_monitor_pixels(
            direction,
            R_gaze_to_cam,
            homography,
            self.width,
            self.height,
            cam_width,
            cam_height,
            cam_cx=cam_cx,
            cam_cy=cam_cy,
            cam_fx=cam_fx,
            cam_fy=cam_fy,
        )
        if point is None:
            return False

        u, v = point
        rel_x = (u - self.cx) / max(self.cx, 1.0)
        rel_y = (self.cy - v) / max(self.cy, 1.0)
        self.last_yaw_pitch = (rel_x, rel_y)

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
        return True

    def draw_calibration_guides(self, frame, aruco_mapping=False):
        if aruco_mapping:
            cv2.drawMarker(frame, (int(self.cx), int(self.cy)), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
            cv2.putText(
                frame,
                "ArUco fallback: press C at center (links IR to front cam)",
                (int(self.cx) - 280, int(self.cy) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            return

        step = self.calibration_step
        if step == "done":
            return

        key_by_edge = EDGE_KEY_LABELS
        targets = calibration_targets(self.width, self.height)

        if step == "center":
            cv2.drawMarker(frame, (int(self.cx), int(self.cy)), (0, 255, 255), cv2.MARKER_CROSS, 40, 2)
            cv2.putText(
                frame,
                "Look at center, press C",
                (int(self.cx) - 130, int(self.cy) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            return

        for edge, (tx, ty) in targets.items():
            done = edge in self.calibrated_edges
            active = edge == step
            color = (0, 200, 0) if done else ((0, 255, 255) if active else (120, 120, 120))
            thickness = 3 if active else 1
            cv2.circle(frame, (int(tx), int(ty)), 18, color, thickness)
            if active:
                key_hint = key_by_edge[edge]
                cv2.putText(
                    frame,
                    f"Guarda qui, premi {key_hint}",
                    (int(tx) - 140, int(ty) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

    def draw_hud(
        self,
        frame,
        fps,
        camera_status=None,
        aruco_status=None,
        extra_lines=None,
        origin=(20, 30),
        max_width=None,
        safe_zone=None,
    ):
        if self.mapping_ready:
            status = "Recording (mappa 5 punti)"
        elif not self.center_calibrated:
            status = "Step 1: press C at screen center"
        elif self.calibration_step != "done":
            next_edge = self.calibration_step
            key_hint = EDGE_KEY_LABELS.get(next_edge, "?")
            status = f"Step: guarda {next_edge}, premi {key_hint} ({len(self.calibrated_edges)}/4 edges)"
        else:
            status = "Recording"

        lines = [
            status,
            self.calibration_status_line(),
            f"Hits: {self.hits}",
            f"Loop FPS: {fps:.1f}",
            "C=centro  frecce=su/giu/sin/des  H=reset cal  X=swap axes",
            "K=reset heatmap  S=save  V=cameras  0/1/2=gaze eye  Q=quit",
        ]

        status_lines = format_camera_status_lines(camera_status) if camera_status else []
        for index, line in enumerate(status_lines):
            lines.insert(3 + index, line)

        insert_at = 3 + len(status_lines)
        if extra_lines:
            for index, line in enumerate(extra_lines):
                lines.insert(insert_at + index, line)
            insert_at += len(extra_lines)
        elif aruco_status:
            lines.insert(insert_at, aruco_status)
            insert_at += 1

        if self.last_yaw_pitch is not None:
            yaw, pitch = self.last_yaw_pitch
            lines.insert(insert_at, f"Yaw: {yaw:+.4f}  Pitch: {pitch:+.4f}")
            insert_at += 1

        if self.calibrating:
            for line in self.calibration_debug_lines():
                lines.insert(insert_at, line)
                insert_at += 1

        line_height = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        zone_width = max_width
        if safe_zone is not None:
            zone_width = safe_zone["width"]

        wrapped_lines = []
        for text in lines:
            wrapped_lines.extend(self._wrap_hud_line(text, font, font_scale, thickness, zone_width))

        max_text_w = 0
        for text in wrapped_lines:
            size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            max_text_w = max(max_text_w, size[0])

        content_w = min(max_text_w, zone_width) if zone_width else max_text_w

        if safe_zone is not None:
            y0 = safe_zone["top"]
            x0 = safe_zone["left"] + max(0, (safe_zone["width"] - content_w) // 2)
        else:
            y0 = origin[1] if isinstance(origin, (tuple, list)) else 30
            frame_w = frame.shape[1]
            x0 = max(0, (frame_w - content_w) // 2)

        y = y0
        for text in wrapped_lines:
            cv2.putText(frame, text, (x0, y), font, font_scale, (255, 255, 255), thickness)
            y += line_height

    @staticmethod
    def _wrap_hud_line(text, font, font_scale, thickness, max_width):
        if max_width is None or not text:
            return [text]

        words = text.split(" ")
        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
            if width <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = word
        if current:
            lines.append(current)
        return lines or [text]

    def compose_display(self, fps, aruco_mapping=False):
        """Heatmap layer with calibration guides and gaze crosshair (no HUD)."""
        heatmap = render_heatmap(self.accumulator)
        display = cv2.addWeighted(heatmap, 0.85, np.zeros_like(heatmap), 0.15, 0)
        self.draw_calibration_guides(display, aruco_mapping=aruco_mapping)

        if self.calibrating:
            self.draw_calibration_debug(display)

        if self.last_point is not None and (self.center_calibrated or self.ready or aruco_mapping):
            color = (0, 255, 255) if self.calibrating else (255, 255, 255)
            cv2.drawMarker(display, self.last_point, color, cv2.MARKER_CROSS, 20, 2)

        return display

    def render(self, fps, camera_status=None, aruco_status=None, hud_origin=(20, 30)):
        display = self.compose_display(fps)
        self.draw_hud(display, fps, camera_status=camera_status, aruco_status=aruco_status, origin=hud_origin)
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
        if key == KEY_UP:
            _, message = self.calibrate_top(direction)
            return True, message
        if key == KEY_DOWN:
            _, message = self.calibrate_bottom(direction)
            return True, message
        if key == KEY_LEFT:
            _, message = self.calibrate_left(direction)
            return True, message
        if key == KEY_RIGHT:
            _, message = self.calibrate_right(direction)
            return True, message
        return False, None


def _win32_work_area():
    """Desktop work area excluding taskbar: (width, height, x, y)."""
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


def get_screen_size():
    """Window size: work area above taskbar on Windows, full screen elsewhere."""
    work_area = _win32_work_area()
    if work_area is not None:
        return work_area[0], work_area[1]

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


def get_window_placement():
    """OpenCV window (width, height, x, y) within the usable desktop area."""
    work_area = _win32_work_area()
    if work_area is not None:
        return work_area
    width, height = get_screen_size()
    return width, height, 0, 0


def create_fullscreen_window(name, width, height):
    """Large windowed mode (not exclusive fullscreen) — fits above the taskbar."""
    _, _, x, y = get_window_placement()
    create_main_window(name, width, height, x, y)


def create_main_window(name, width, height, x=0, y=0):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, x, y)
    cv2.resizeWindow(name, width, height)


def focus_window(name):
    """Route keyboard input to a specific OpenCV window when supported."""
    select_window = getattr(cv2, "selectWindow", None)
    if select_window is None:
        return
    try:
        select_window(name)
    except cv2.error:
        pass

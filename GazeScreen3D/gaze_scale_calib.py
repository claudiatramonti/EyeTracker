"""
Optional yaw/pitch scale refinement after center (C) calibration.

Pipeline (after C sets R_gaze_to_cam):
  1. User looks at an edge cross on the full monitor window, presses matching arrow key.
  2. GazeScreen3D collects ~12 raw gaze frames → record_edge_from_gaze_samples.
  3. Compare measured cam gaze vs ArUco-expected direction to that edge pixel.
  4. Store ratio → scale_x_left/right, scale_y_up/down (default 1.0).
  5. Every frame: apply_yaw_pitch_scale in ray_screen before ray ∩ ArUco plane.

Does NOT replace ArUco mapping. Head motion still comes from live R,t every frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ray_screen import gaze_dir_in_cam, normalize

# --- Tuning constants ---
MARGIN_PX = 80  # edge target inset from window border (px)
MIN_ANGLE_ABS = 0.02  # rad (~1.1°); reject samples if gaze barely off-center
SCALE_MIN = 0.45  # clamp edge scales (avoid runaway stretch)
SCALE_MAX = 2.4

EDGE_ORDER = ("top", "bottom", "left", "right")
EDGE_LABELS = {
    "top": "freccia SU",
    "bottom": "freccia GIU",
    "left": "freccia SIN",
    "right": "freccia DES",
}
EDGE_SHORT = {"top": "U", "bottom": "D", "left": "L", "right": "R"}


def calibration_targets(width_px, height_px, margin=MARGIN_PX):
    """Pixel (u,v) for center + four edge midpoints; scales with window size."""
    w = max(int(width_px), 1)
    h = max(int(height_px), 1)
    m = int(np.clip(margin, 8, min(w, h) // 4))
    cx, cy = w * 0.5, h * 0.5
    return {
        "center": (cx, cy),
        "top": (cx, float(m)),
        "bottom": (cx, float(h - 1 - m)),
        "left": (float(m), cy),
        "right": (float(w - 1 - m), cy),
    }


def pixel_to_object_mm(u, v, width_px, height_px, width_mm, height_mm):
    """OpenCV pixel (top-left origin) → point on screen plane in object mm (Y up, Z=0)."""
    x = (float(u) / max(width_px, 1) - 0.5) * width_mm
    y = (0.5 - float(v) / max(height_px, 1)) * height_mm
    return np.array([x, y, 0.0], dtype=np.float64)


def expected_cam_direction(u, v, rotation, translation, width_mm, height_mm, width_px, height_px):
    """
    Unit direction from camera origin through the 3D screen point for pixel (u, v).

    Steps: pixel → object mm → camera frame via solvePnP R,t → normalize.
    Same geometry as gaze_to_screen (inverse of the hit test).
    """
    if rotation is None or translation is None:
        return None
    p_obj = pixel_to_object_mm(u, v, width_px, height_px, width_mm, height_mm)
    p_cam = np.asarray(rotation, dtype=np.float64) @ p_obj + np.asarray(
        translation, dtype=np.float64
    ).reshape(3)
    return normalize(p_cam)


def direction_to_yaw_pitch(direction):
    """OpenCV camera frame (+Z forward, +Y down): yaw right+, pitch down+."""
    d = normalize(direction)
    if d is None:
        return None
    yaw = math.atan2(float(d[0]), float(d[2]))
    pitch = math.atan2(float(d[1]), math.hypot(float(d[0]), float(d[2])))
    return yaw, pitch


def yaw_pitch_to_direction(yaw, pitch):
    """Rebuild unit direction from yaw/pitch (inverse of direction_to_yaw_pitch)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x = sy * cp
    y = sp
    z = cy * cp
    return normalize(np.array([x, y, z], dtype=np.float64))


def apply_yaw_pitch_scale(
    direction,
    scale_x=1.0,
    scale_y=1.0,
    scale_x_left=None,
    scale_x_right=None,
    scale_y_up=None,
    scale_y_down=None,
):
    """
    Stretch gaze yaw/pitch asymmetrically before ray ∩ screen plane.

    Applied in ray_screen.gaze_dir_in_cam every frame when any scale != 1:
      yaw < 0 (left)  x scale_x_left
      yaw > 0 (right) x scale_x_right
      pitch < 0 (up)  x scale_y_up   (OpenCV Y-down: negative pitch = up)
      pitch > 0 (down)x scale_y_down
    """
    if direction is None:
        return None
    sx_left = float(scale_x if scale_x_left is None else scale_x_left)
    sx_right = float(scale_x if scale_x_right is None else scale_x_right)
    sy_up = float(scale_y if scale_y_up is None else scale_y_up)
    sy_down = float(scale_y if scale_y_down is None else scale_y_down)
    if (
        abs(sx_left - 1.0) < 1e-6
        and abs(sx_right - 1.0) < 1e-6
        and abs(sy_up - 1.0) < 1e-6
        and abs(sy_down - 1.0) < 1e-6
    ):
        return normalize(direction)

    angles = direction_to_yaw_pitch(direction)
    if angles is None:
        return normalize(direction)
    yaw, pitch = angles
    if yaw < 0.0:
        yaw *= sx_left
    elif yaw > 0.0:
        yaw *= sx_right
    if pitch < 0.0:
        pitch *= sy_up
    elif pitch > 0.0:
        pitch *= sy_down
    return yaw_pitch_to_direction(yaw, pitch)


def _clip_scale(value):
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, SCALE_MIN, SCALE_MAX))


def _angle_scale_ratio(measured, expected):
    """
    Edge scale from one angle sample: |expected / measured| when signs agree.

    > 1 means measured angle was too small (gaze fell short of edge) --> stretch.
  """
    if abs(measured) < MIN_ANGLE_ABS or abs(expected) < MIN_ANGLE_ABS:
        return None
    if measured * expected < 0.0:
        return None
    return _clip_scale(abs(expected) / abs(measured))


class GazeScaleCalib:
    """
    Holds optional edge calibration state for GazeScreen3D.

    One instance for the session; scales reset on new C (clear_edges) or E key.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = {}  # edge name → {yaw_m, pitch_m, yaw_e, pitch_e, u, v}
        self.scale_x_left = 1.0
        self.scale_x_right = 1.0
        self.scale_y_up = 1.0
        self.scale_y_down = 1.0

    def clear_edges(self):
        """Reset all four scales to 1 and drop samples (GazeScreen3D calls after C or E)."""
        self.samples.clear()
        self.scale_x_left = 1.0
        self.scale_x_right = 1.0
        self.scale_y_up = 1.0
        self.scale_y_down = 1.0

    @property
    def scale_x(self):
        """Legacy mean of left/right (e.g. old single-scale callers)."""
        return 0.5 * (self.scale_x_left + self.scale_x_right)

    @property
    def scale_y(self):
        """Legacy mean of up/down."""
        return 0.5 * (self.scale_y_up + self.scale_y_down)

    def _clip_edge_scale(self, value):
        return float(np.clip(value, SCALE_MIN, SCALE_MAX))

    def nudge_horizontal(self, delta):
        """Manual tweak: ,/. keys shift scale_x_left and scale_x_right together."""
        self.scale_x_left = self._clip_edge_scale(self.scale_x_left + delta)
        self.scale_x_right = self._clip_edge_scale(self.scale_x_right + delta)

    def nudge_vertical(self, delta):
        """Manual tweak: [/] keys shift scale_y_up and scale_y_down together."""
        self.scale_y_up = self._clip_edge_scale(self.scale_y_up + delta)
        self.scale_y_down = self._clip_edge_scale(self.scale_y_down + delta)

    @property
    def edges_done(self):
        return set(self.samples.keys())

    def scales_summary(self):
        return (
            f"sx←={self.scale_x_left:.2f} sx→={self.scale_x_right:.2f}  "
            f"sy↑={self.scale_y_up:.2f} sy↓={self.scale_y_down:.2f}"
        )

    def status_line(self):
        """HUD line: which edges sampled + current scale values."""
        done = "".join(EDGE_SHORT[e] for e in EDGE_ORDER if e in self.samples)
        pending = "".join(EDGE_SHORT[e] for e in EDGE_ORDER if e not in self.samples)
        scales = self.scales_summary()
        if len(self.samples) == 0:
            return f"Edges: ↑↓←→ after C (12 frames each)  [{scales}]"
        if len(self.samples) >= 4:
            return f"Edges: OK {done}  {scales}"
        return f"Edges: {done or '-'} need {pending}  {scales}"

    def refit(self):
        """Recompute all four scales from stored edge samples (called after each record_edge)."""
        left_ratios = []
        right_ratios = []
        up_ratios = []
        down_ratios = []
        for edge, sample in self.samples.items():
            if edge == "left":
                ratio = _angle_scale_ratio(sample["yaw_m"], sample["yaw_e"])
                if ratio is not None:
                    left_ratios.append(ratio)
            elif edge == "right":
                ratio = _angle_scale_ratio(sample["yaw_m"], sample["yaw_e"])
                if ratio is not None:
                    right_ratios.append(ratio)
            elif edge == "top":
                ratio = _angle_scale_ratio(sample["pitch_m"], sample["pitch_e"])
                if ratio is not None:
                    up_ratios.append(ratio)
            elif edge == "bottom":
                ratio = _angle_scale_ratio(sample["pitch_m"], sample["pitch_e"])
                if ratio is not None:
                    down_ratios.append(ratio)
        if left_ratios:
            self.scale_x_left = float(np.mean(left_ratios))
        if right_ratios:
            self.scale_x_right = float(np.mean(right_ratios))
        if up_ratios:
            self.scale_y_up = float(np.mean(up_ratios))
        if down_ratios:
            self.scale_y_down = float(np.mean(down_ratios))

    def record_edge(
        self,
        edge,
        gaze_dir_eye,
        R_gaze_to_cam,
        rotation,
        translation,
        width_mm,
        height_mm,
        width_px,
        height_px,
        margin=MARGIN_PX,
    ):
        """
        One edge calibration sample (usually after averaging 12 gaze frames).

        Compares IR gaze (after C, before edge scales) to ArUco-expected direction
        for the edge pixel; updates self.samples and refits scales.
        """
        if edge not in EDGE_ORDER:
            return False, f"Unknown edge {edge}"
        if gaze_dir_eye is None or R_gaze_to_cam is None:
            return False, "No gaze / press C first"
        if rotation is None or translation is None:
            return False, "ArUco pose not ready"

        targets = calibration_targets(width_px, height_px, margin=margin)
        u, v = targets[edge]

        # Measured: IR gaze → front-cam direction (no edge scales during calibration)
        measured = gaze_dir_in_cam(gaze_dir_eye, R_gaze_to_cam, opencv_y_down=True)
        # Expected: ray from camera origin through edge pixel on ArUco screen plane
        expected = expected_cam_direction(
            u, v, rotation, translation, width_mm, height_mm, width_px, height_px
        )
        if measured is None or expected is None:
            return False, f"{edge}: invalid directions"

        m_angles = direction_to_yaw_pitch(measured)
        e_angles = direction_to_yaw_pitch(expected)
        if m_angles is None or e_angles is None:
            return False, f"{edge}: angle extract failed"

        yaw_m, pitch_m = m_angles
        yaw_e, pitch_e = e_angles

        # Reject if user did not look far enough toward that edge
        if edge in ("left", "right") and abs(yaw_m) < MIN_ANGLE_ABS:
            return False, f"{edge}: look farther toward the {edge} edge"
        if edge in ("top", "bottom") and abs(pitch_m) < MIN_ANGLE_ABS:
            return False, f"{edge}: look farther toward the {edge} edge"

        self.samples[edge] = {
            "yaw_m": yaw_m,
            "pitch_m": pitch_m,
            "yaw_e": yaw_e,
            "pitch_e": pitch_e,
            "u": u,
            "v": v,
        }
        self.refit()
        return (
            True,
            f"{edge} OK ({EDGE_LABELS[edge]})  {self.scales_summary()}  "
            f"[{len(self.samples)}/4]",
        )


def average_unit_vectors(vectors):
    """Fuse multiple per-frame gaze directions: sum unit vectors, re-normalize."""
    valid = [np.asarray(v, dtype=np.float64).reshape(3) for v in vectors if v is not None]
    if not valid:
        return None
    combined = np.sum(valid, axis=0)
    return normalize(combined)


def record_edge_from_gaze_samples(
    scale_calib,
    edge,
    gaze_samples,
    R_gaze_to_cam,
    rotation,
    translation,
    width_mm,
    height_mm,
    width_px,
    height_px,
):
    """
    GazeScreen3D edge capture: average EDGE_CALIB_FRAMES raw gaze vectors, then record_edge once.
    """
    gaze_dir = average_unit_vectors(gaze_samples)
    if gaze_dir is None:
        return False, f"{edge}: no gaze samples"
    return scale_calib.record_edge(
        edge,
        gaze_dir,
        R_gaze_to_cam,
        rotation,
        translation,
        width_mm,
        height_mm,
        width_px,
        height_px,
    )


def draw_edge_targets(canvas, width_px, height_px, done_edges, margin=MARGIN_PX):
    """Edge crosses on heatmap: yellow = pending, green = already sampled."""
    targets = calibration_targets(width_px, height_px, margin=margin)
    for edge in EDGE_ORDER:
        u, v = targets[edge]
        pt = (int(round(u)), int(round(v)))
        done = edge in done_edges
        color = (0, 220, 0) if done else (0, 220, 255)
        cv2.drawMarker(canvas, pt, color, cv2.MARKER_CROSS, 36, 2)
        cv2.putText(
            canvas,
            EDGE_SHORT[edge],
            (pt[0] + 12, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_center_calib_target(canvas, width_px, height_px, margin=MARGIN_PX):
    """
    Cyan cross at monitor center — look HERE and press C.

    Full-window overlay only. NOT the Front PiP, NOT the purple ArUco cross
    in the front-camera preview (that is debug pose, not the C target).
    """
    targets = calibration_targets(width_px, height_px, margin=margin)
    cx = int(round(targets["center"][0]))
    cy = int(round(targets["center"][1]))
    color = (0, 255, 255)  # cyan BGR

    cv2.circle(canvas, (cx, cy), 56, color, 2, cv2.LINE_AA)
    cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 72, 3, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)

    label = "GUARDA QUI  e  premi C"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.85, 2)
    tx = int(np.clip(cx - tw // 2, 8, max(8, width_px - tw - 8)))
    ty = cy - 72
    cv2.putText(canvas, label, (tx + 2, ty + 2), font, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, label, (tx, ty), font, 0.85, color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "centro MONITOR",
        (tx, ty + 28),
        font,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def draw_front_preview_not_for_c(canvas, x, y, w, h):
    """Orange border on Front PiP: reminds user not to use it for C calibration."""
    if w < 40 or h < 30:
        return
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 140, 255), 2)
    cv2.putText(
        canvas,
        "Front = debug",
        (x + 6, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 140, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "NON usare per C",
        (x + 6, y + h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 140, 255),
        1,
        cv2.LINE_AA,
    )

"""
Optional yaw/pitch scale refinement after center (C) calibration.

Does NOT replace ArUco ray∩plane mapping. After R_gaze_to_cam is set, arrow-key
samples compare measured cam gaze vs the direction that should hit each screen
edge (from current ArUco R,t). Scales stretch/compress yaw/pitch before the
existing geometric hit — head motion still comes from ArUco every frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ray_screen import gaze_dir_in_cam, normalize

MARGIN_PX = 80
MIN_ANGLE_ABS = 0.02  # rad (~1.1°) — ignore near-zero samples
SCALE_MIN = 0.45
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
    """Edge midpoints from current window size (not hardcoded absolutes)."""
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
    """OpenCV pixel (top-left) → screen object mm (Y up, origin center)."""
    x = (float(u) / max(width_px, 1) - 0.5) * width_mm
    y = (0.5 - float(v) / max(height_px, 1)) * height_mm
    return np.array([x, y, 0.0], dtype=np.float64)


def expected_cam_direction(u, v, rotation, translation, width_mm, height_mm, width_px, height_px):
    """
    Unit direction from camera origin through the 3D screen point for pixel (u,v).
    Uses live ArUco R,t — same geometry as gaze_to_screen.
    """
    if rotation is None or translation is None:
        return None
    p_obj = pixel_to_object_mm(u, v, width_px, height_px, width_mm, height_mm)
    p_cam = np.asarray(rotation, dtype=np.float64) @ p_obj + np.asarray(
        translation, dtype=np.float64
    ).reshape(3)
    return normalize(p_cam)


def direction_to_yaw_pitch(direction):
    """OpenCV cam frame (+Z forward, +Y down): yaw right+, pitch down+."""
    d = normalize(direction)
    if d is None:
        return None
    yaw = math.atan2(float(d[0]), float(d[2]))
    pitch = math.atan2(float(d[1]), math.hypot(float(d[0]), float(d[2])))
    return yaw, pitch


def yaw_pitch_to_direction(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # yaw about Y, then pitch about X' — consistent with atan2 extraction above
    x = sy * cp
    y = sp
    z = cy * cp
    return normalize(np.array([x, y, z], dtype=np.float64))


def apply_yaw_pitch_scale(direction, scale_x=1.0, scale_y=1.0):
    """Stretch yaw/pitch of a cam-space unit direction. Identity if scales ~1."""
    if direction is None:
        return None
    sx = float(scale_x)
    sy = float(scale_y)
    if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6:
        return normalize(direction)
    angles = direction_to_yaw_pitch(direction)
    if angles is None:
        return normalize(direction)
    yaw, pitch = angles
    return yaw_pitch_to_direction(yaw * sx, pitch * sy)


def _clip_scale(value):
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip(value, SCALE_MIN, SCALE_MAX))


class GazeScaleCalib:
    """Collect edge samples → horizontal/vertical gaze scales (default 1,1)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = {}  # edge -> {yaw_m, pitch_m, yaw_e, pitch_e, u, v}
        self.scale_x = 1.0
        self.scale_y = 1.0

    def clear_edges(self):
        """Keep identity scales; drop edge samples (call after new C)."""
        self.samples.clear()
        self.scale_x = 1.0
        self.scale_y = 1.0

    @property
    def edges_done(self):
        return set(self.samples.keys())

    def status_line(self):
        done = "".join(EDGE_SHORT[e] for e in EDGE_ORDER if e in self.samples)
        pending = "".join(EDGE_SHORT[e] for e in EDGE_ORDER if e not in self.samples)
        if len(self.samples) == 0:
            return "Edges: frecce SU/GIU/SIN/DES after C (optional)"
        if len(self.samples) >= 4:
            return f"Edges: OK {done}  sx={self.scale_x:.2f} sy={self.scale_y:.2f}"
        return (
            f"Edges: {done or '-'} need {pending}  "
            f"sx={self.scale_x:.2f} sy={self.scale_y:.2f}"
        )

    def refit(self):
        x_ratios = []
        y_ratios = []
        for edge, sample in self.samples.items():
            if edge in ("left", "right"):
                ym, ye = sample["yaw_m"], sample["yaw_e"]
                if abs(ym) >= MIN_ANGLE_ABS and abs(ye) >= MIN_ANGLE_ABS:
                    ratio = _clip_scale(ye / ym)
                    if ratio is not None:
                        x_ratios.append(ratio)
            elif edge in ("top", "bottom"):
                pm, pe = sample["pitch_m"], sample["pitch_e"]
                if abs(pm) >= MIN_ANGLE_ABS and abs(pe) >= MIN_ANGLE_ABS:
                    ratio = _clip_scale(pe / pm)
                    if ratio is not None:
                        y_ratios.append(ratio)
        self.scale_x = float(np.mean(x_ratios)) if x_ratios else 1.0
        self.scale_y = float(np.mean(y_ratios)) if y_ratios else 1.0

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
        if edge not in EDGE_ORDER:
            return False, f"Unknown edge {edge}"
        if gaze_dir_eye is None or R_gaze_to_cam is None:
            return False, "No gaze / press C first"
        if rotation is None or translation is None:
            return False, "ArUco pose not ready"

        targets = calibration_targets(width_px, height_px, margin=margin)
        u, v = targets[edge]

        measured = gaze_dir_in_cam(gaze_dir_eye, R_gaze_to_cam, opencv_y_down=True)
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
            f"{edge} OK ({EDGE_LABELS[edge]})  "
            f"sx={self.scale_x:.2f} sy={self.scale_y:.2f}  "
            f"[{len(self.samples)}/4]",
        )


def draw_edge_targets(canvas, width_px, height_px, done_edges, margin=MARGIN_PX):
    """Small crosses at edge targets; green if sampled, yellow if pending."""
    targets = calibration_targets(width_px, height_px, margin=margin)
    for edge in EDGE_ORDER:
        u, v = targets[edge]
        pt = (int(round(u)), int(round(v)))
        done = edge in done_edges
        color = (0, 220, 0) if done else (0, 220, 255)
        cv2.drawMarker(canvas, pt, color, cv2.MARKER_CROSS, 28, 2)
        cv2.putText(
            canvas,
            EDGE_SHORT[edge],
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

"""
Gaze ray ∩ ArUco screen plane → monitor pixel (or off-screen).

Camera frame: OpenCV, optical center at origin, +Z forward.
Screen object frame (mm): origin at window center, X right, Y up, Z into screen.
OpenCV camera frame keeps Y down; convert only at pixel boundaries.
solvePnP: P_cam = R @ P_obj + t
"""

from __future__ import annotations

import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    return v / n


def rotation_from_a_to_b(a, b):
    """R such that R @ a = b (Rodrigues)."""
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return np.eye(3, dtype=np.float64)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-6:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / np.linalg.norm(v)
        # 180° about v
        return (2.0 * np.outer(v, v) - np.eye(3)).astype(np.float64)

    vx, vy, vz = v / s
    k = np.array([[0.0, -vz, vy], [vz, 0.0, -vx], [-vy, vx, 0.0]], dtype=np.float64)
    return (np.eye(3) + k * s + (k @ k) * ((1.0 - c) / (s * s))).astype(np.float64)


def gaze_dir_in_cam(gaze_dir_eye, R_gaze_to_cam, opencv_y_down=True, scale_x=1.0, scale_y=1.0):
    """
    Map eye gaze to front-camera direction.

    eye_tracker uses Y-up gaze space; R from C is ~identity when looking forward.

    Pinhole preview uses ``cy - fy * g[1]`` (projection space, Y-up).

    3D ray ∩ plane needs OpenCV camera coords (Y down) → flip g[1] when requested.

    scale_x / scale_y (default 1 = do nothing):
        Optional fine-tuning after center calib (C). Arrow keys at screen edges
        set these so left/right and up/down gaze reach the borders more accurately.
        They only stretch the gaze direction a bit; ArUco screen hit is unchanged.
    """
    if gaze_dir_eye is None or R_gaze_to_cam is None:
        return None
    g = np.asarray(R_gaze_to_cam, dtype=np.float64) @ np.asarray(gaze_dir_eye, dtype=np.float64)
    # Always flip to OpenCV Y-down for optional scale (edge calib fitted there).
    g_cv = np.asarray(g, dtype=np.float64).copy()
    g_cv[1] = -g_cv[1]
    g_cv = normalize(g_cv)
    if g_cv is None:
        return None

    sx, sy = float(scale_x), float(scale_y)
    if abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6:
        from gaze_scale_calib import apply_yaw_pitch_scale

        g_cv = apply_yaw_pitch_scale(g_cv, sx, sy)
        if g_cv is None:
            return None

    if opencv_y_down:
        return g_cv
    out = g_cv.copy()
    out[1] = -out[1]
    return out


def screen_plane_in_camera(rotation, translation):
    """
    Screen plane in front-camera coords from solvePnP R, t.
    Normal = R @ [0,0,1] (into the screen). Point = t (screen origin).
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    normal = normalize(rotation @ np.array([0.0, 0.0, 1.0]))
    return normal, translation


def intersect_ray_plane(origin, direction, plane_point, plane_normal):
    """
    Ray O + s D vs plane. Returns (point, s) or None if parallel / behind.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = normalize(direction)
    plane_normal = normalize(plane_normal)
    if direction is None or plane_normal is None:
        return None

    denom = float(np.dot(plane_normal, direction))
    if abs(denom) < 1e-8:
        return None

    s = float(np.dot(plane_normal, plane_point - origin) / denom)
    if s <= 1e-6:
        return None

    return origin + s * direction, s


def camera_point_to_screen_mm(point_cam, rotation, translation):
    """P_obj = R.T @ (P_cam - t)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    point_cam = np.asarray(point_cam, dtype=np.float64).reshape(3)
    return rotation.T @ (point_cam - translation)


def screen_mm_to_pixels(point_mm, width_mm, height_mm, width_px, height_px):
    """Object mm (Y up, origin center) → OpenCV pixel (origin top-left, v down)."""
    x_mm, y_mm = float(point_mm[0]), float(point_mm[1])
    u = (x_mm / width_mm + 0.5) * width_px
    v = (0.5 - y_mm / height_mm) * height_px
    return u, v


def gaze_to_screen(
    gaze_dir_eye,
    R_gaze_to_cam,
    rotation,
    translation,
    width_mm,
    height_mm,
    width_px,
    height_px,
    eye_origin_cam=None,
    scale_x=1.0,
    scale_y=1.0,
):
    """
    Full chain: eye gaze → front-cam direction → hit screen plane → pixel.

    eye_origin_cam defaults to front-camera optical center (0,0,0), same
    approximation as the classic pinhole gaze dot.

    scale_x / scale_y: optional yaw/pitch refinement from edge arrow calib.

    Returns dict with keys:
      on_screen (bool), u, v (float pixels), point_mm, point_cam, direction_cam
    or None if no valid hit.
    """
    if gaze_dir_eye is None or R_gaze_to_cam is None or rotation is None or translation is None:
        return None

    direction_cam = gaze_dir_in_cam(
        gaze_dir_eye,
        R_gaze_to_cam,
        opencv_y_down=True,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    if direction_cam is None:
        return None

    origin = (
        np.zeros(3, dtype=np.float64)
        if eye_origin_cam is None
        else np.asarray(eye_origin_cam, dtype=np.float64).reshape(3)
    )

    plane_n, plane_p = screen_plane_in_camera(rotation, translation)
    hit = intersect_ray_plane(origin, direction_cam, plane_p, plane_n)
    if hit is None:
        return None

    point_cam, _s = hit
    point_mm = camera_point_to_screen_mm(point_cam, rotation, translation)
    u, v = screen_mm_to_pixels(point_mm, width_mm, height_mm, width_px, height_px)

    on_screen = 0.0 <= u < width_px and 0.0 <= v < height_px
    return {
        "on_screen": on_screen,
        "u": float(u),
        "v": float(v),
        "point_mm": point_mm,
        "point_cam": point_cam,
        "direction_cam": direction_cam,
    }


def project_gaze_to_front_pixels(
    gaze_dir_eye,
    R_gaze_to_cam,
    width,
    height,
    fx,
    fy,
    cx=None,
    cy=None,
    scale_x=1.0,
    scale_y=1.0,
):
    """Debug: same pinhole as FrontCameraTracker (red dot on front feed)."""
    if gaze_dir_eye is None or R_gaze_to_cam is None:
        return None
    g = gaze_dir_in_cam(
        gaze_dir_eye,
        R_gaze_to_cam,
        opencv_y_down=False,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    if g is None or g[2] <= 1e-6:
        return None
    if cx is None:
        cx = width * 0.5
    if cy is None:
        cy = height * 0.5
    u = cx + fx * (g[0] / g[2])
    v = cy - fy * (g[1] / g[2])
    return float(u), float(v)

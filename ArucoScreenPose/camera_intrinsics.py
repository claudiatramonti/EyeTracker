"""
Front-camera intrinsics (K, dist) for ArUco PnP / GazeScreen3D.

Calibrate once with calibrate_front_camera.py, then ScreenPoseTracker loads
the .npz automatically. Falls back to assumed HFOV when missing.
"""

from __future__ import annotations

import math
import os

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIB_DIR = os.path.join(MODULE_DIR, "camera_calib")
DEFAULT_CALIB_PATH = os.path.join(DEFAULT_CALIB_DIR, "front_camera.npz")


def default_calib_path():
    return DEFAULT_CALIB_PATH


def hfov_from_k(camera_k, image_width):
    """Horizontal FOV in degrees from fx and image width."""
    fx = float(camera_k[0, 0])
    if fx <= 1e-6 or image_width <= 0:
        return None
    return float(np.degrees(2.0 * math.atan(0.5 * image_width / fx)))


def scale_intrinsics(camera_k, dist, calib_size, frame_size):
    """
    Scale K when capture resolution differs from calibration resolution.
    dist is left unchanged (OpenCV model is in normalized coords via K).
    """
    calib_w, calib_h = int(calib_size[0]), int(calib_size[1])
    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    if calib_w <= 0 or calib_h <= 0:
        return camera_k.copy(), dist.copy()
    if (calib_w, calib_h) == (frame_w, frame_h):
        return camera_k.copy(), dist.copy()

    sx = frame_w / float(calib_w)
    sy = frame_h / float(calib_h)
    k = camera_k.copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k, dist.copy()


def save_intrinsics(
    path,
    camera_k,
    dist,
    image_size,
    rms=None,
    flags=None,
    square_size_mm=None,
    board_size=None,
):
    """Write calibration .npz (creates parent dirs)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "camera_matrix": np.asarray(camera_k, dtype=np.float64),
        "dist_coeffs": np.asarray(dist, dtype=np.float64).reshape(-1),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
    }
    if rms is not None:
        payload["rms"] = float(rms)
    if flags is not None:
        payload["flags"] = int(flags)
    if square_size_mm is not None:
        payload["square_size_mm"] = float(square_size_mm)
    if board_size is not None:
        payload["board_cols"] = int(board_size[0])
        payload["board_rows"] = int(board_size[1])
    np.savez(path, **payload)
    return path


def load_intrinsics(path=None):
    """
    Load calibration file.

    Returns dict with keys:
      camera_matrix, dist_coeffs, image_size (w, h), path, rms (optional)
    or None if missing/invalid.
    """
    path = path or DEFAULT_CALIB_PATH
    if not path or not os.path.isfile(path):
        return None

    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        print(f"Intrinsics load failed ({path}): {exc}")
        return None

    try:
        camera_k = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
        width = int(data["image_width"])
        height = int(data["image_height"])
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Intrinsics file incomplete ({path}): {exc}")
        return None

    if camera_k.shape != (3, 3) or width <= 0 or height <= 0:
        print(f"Intrinsics file invalid shape/size ({path})")
        return None

    result = {
        "camera_matrix": camera_k,
        "dist_coeffs": dist,
        "image_size": (width, height),
        "path": os.path.abspath(path),
    }
    if "rms" in data.files:
        result["rms"] = float(data["rms"])
    return result


def camera_matrix_from_hfov(width, height, hfov_deg):
    """K from assumed HFOV: fx = 0.5 * width / tan(hfov/2), fy = fx, cx/cy = center."""
    fx = 0.5 * width / math.tan(math.radians(hfov_deg) * 0.5)
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def resolve_camera_model(frame_width, frame_height, hfov_deg, intrinsics=None):
    """
    Prefer calibrated K/dist (scaled to frame size); else HFOV pinhole, zero dist.
    Returns (K, dist, source_label, effective_hfov_deg).
    """
    if intrinsics is not None:
        k, dist = scale_intrinsics(
            intrinsics["camera_matrix"],
            intrinsics["dist_coeffs"],
            intrinsics["image_size"],
            (frame_width, frame_height),
        )
        hfov = hfov_from_k(k, frame_width)
        return k, dist, "calib", hfov if hfov is not None else float(hfov_deg)

    k = camera_matrix_from_hfov(frame_width, frame_height, hfov_deg)
    dist = np.zeros((5, 1), dtype=np.float64)
    return k, dist, "hfov", float(hfov_deg)

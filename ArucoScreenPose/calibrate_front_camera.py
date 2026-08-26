"""
One-shot chessboard calibration for the front camera.

Produces camera_calib/front_camera.npz (K + dist) used by ScreenPoseTracker /
GazeScreen3D. Distance to screen does not matter; recalibrate only if you change
camera or capture resolution.

Usage:
  cd ArucoScreenPose
  python calibrate_front_camera.py
  python calibrate_front_camera.py 2
  python calibrate_front_camera.py 2 --width 1280 --height 720

Print a chessboard (or display one fullscreen on another monitor):
  default inner corners 9x6, square size 25 mm (any known size is fine).

Keys:
  SPACE  capture pose when board is detected (green)
  C      run calibration (needs >= 10 poses)
  U      toggle undistort preview (after calibration or if file exists)
  S      save current result again
  R      reset captured poses
  Q      quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

from camera_intrinsics import (
    DEFAULT_CALIB_PATH,
    hfov_from_k,
    load_intrinsics,
    save_intrinsics,
)
from camera_io import open_camera_capture
from screen_pose import detect_cameras

WINDOW_NAME = "Front camera calibration"
MIN_POSES = 10
DEFAULT_COLS = 9
DEFAULT_ROWS = 6
DEFAULT_SQUARE_MM = 25.0


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate front-camera intrinsics (chessboard).")
    parser.add_argument("camera", nargs="?", type=int, default=None, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width (match GazeScreen3D)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (match GazeScreen3D)")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Inner corners across")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Inner corners down")
    parser.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM, help="Square size mm")
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_CALIB_PATH,
        help=f"Output .npz (default: {DEFAULT_CALIB_PATH})",
    )
    return parser.parse_args()


def choose_camera(explicit_index):
    if explicit_index is not None:
        return explicit_index
    print("Scanning cameras...")
    available = detect_cameras()
    if not available:
        print("No cameras found.")
        return None
    print("Cameras:", available)
    raw = input(f"Front camera index [{available[0]}]: ").strip()
    if not raw:
        return available[0]
    try:
        return int(raw)
    except ValueError:
        print("Invalid index.")
        return None


def build_object_points(cols, rows, square_mm):
    obj = np.zeros((cols * rows, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= float(square_mm)
    return obj


def draw_hud(frame, lines, color=(0, 255, 0)):
    y = 24
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        y += 22


def run_calibration(obj_points, img_points, image_size, cols, rows, square_mm):
    if len(obj_points) < MIN_POSES:
        print(f"Need at least {MIN_POSES} poses (have {len(obj_points)}).")
        return None

    flags = 0
    rms, camera_k, dist, *_ = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=flags,
    )
    hfov = hfov_from_k(camera_k, image_size[0])
    print("--- Calibration result ---")
    print(f"RMS reprojection: {rms:.4f} px  (aim < 1.0)")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"K:\n{camera_k}")
    print(f"dist: {dist.ravel()}")
    if hfov is not None:
        print(f"Implied HFOV: {hfov:.1f} deg")
    return {
        "camera_matrix": camera_k,
        "dist_coeffs": dist,
        "image_size": image_size,
        "rms": float(rms),
        "flags": int(flags),
        "square_size_mm": float(square_mm),
        "board_size": (cols, rows),
    }


def main():
    args = parse_args()
    camera_index = choose_camera(args.camera)
    if camera_index is None:
        return 1

    board_size = (args.cols, args.rows)
    objp = build_object_points(args.cols, args.rows, args.square_mm)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap, backend = open_camera_capture(camera_index, width=args.width, height=args.height)
    if cap is None:
        print(f"Could not open camera {camera_index}.")
        return 1

    print(f"Camera {camera_index} ({backend}) {args.width}x{args.height}")
    print(f"Board: {args.cols}x{args.rows} inner corners, square {args.square_mm} mm")
    print(f"Output: {args.out}")
    print("SPACE capture | C calibrate | U undistort | S save | R reset | Q quit")

    obj_points = []
    img_points = []
    last_capture_t = 0.0
    result = None
    show_undistort = False

    existing = load_intrinsics(args.out)
    if existing is not None:
        print(f"Existing calib loaded: {existing['path']}")
        if "rms" in existing:
            print(f"  previous RMS {existing['rms']:.4f} px")
        result = {
            "camera_matrix": existing["camera_matrix"],
            "dist_coeffs": existing["dist_coeffs"],
            "image_size": existing["image_size"],
            "rms": existing.get("rms"),
            "flags": 0,
            "square_size_mm": args.square_mm,
            "board_size": board_size,
        }

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, board_size, corners_refined, found)
                detect_color = (0, 255, 0)
                detect_text = "Board OK — SPACE to capture"
            else:
                corners_refined = None
                detect_color = (0, 165, 255)
                detect_text = "Show chessboard (tilt / move for variety)"

            if show_undistort and result is not None:
                h, w = frame.shape[:2]
                k = result["camera_matrix"]
                dist = result["dist_coeffs"]
                # Scale if preview size differs from calib size
                calib_w, calib_h = result["image_size"]
                if (w, h) != (calib_w, calib_h):
                    sx, sy = w / calib_w, h / calib_h
                    k = k.copy()
                    k[0, 0] *= sx
                    k[0, 2] *= sx
                    k[1, 1] *= sy
                    k[1, 2] *= sy
                display = cv2.undistort(frame, k, dist)

            lines = [
                detect_text if not show_undistort else "Undistort preview (U to toggle)",
                f"Poses: {len(obj_points)}/{MIN_POSES}+",
            ]
            if result is not None and result.get("rms") is not None:
                hfov = hfov_from_k(result["camera_matrix"], result["image_size"][0])
                hfov_txt = f"  HFOV~{hfov:.0f}" if hfov else ""
                lines.append(f"Last RMS {result['rms']:.3f}px{hfov_txt}")
            lines.append("SPACE capture | C calib | U undistort | S save | R reset | Q")
            draw_hud(display, lines, color=detect_color)

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                obj_points.clear()
                img_points.clear()
                print("Poses cleared.")
            if key == ord("u"):
                if result is None:
                    print("No calibration yet.")
                else:
                    show_undistort = not show_undistort
            if key == ord(" ") and found and corners_refined is not None:
                now = time.perf_counter()
                if now - last_capture_t < 0.4:
                    continue
                last_capture_t = now
                obj_points.append(objp.copy())
                img_points.append(corners_refined.reshape(-1, 2).astype(np.float32))
                print(f"Captured pose {len(obj_points)}")
            if key == ord("c"):
                h, w = frame.shape[:2]
                calibrated = run_calibration(
                    obj_points,
                    img_points,
                    (w, h),
                    args.cols,
                    args.rows,
                    args.square_mm,
                )
                if calibrated is not None:
                    result = calibrated
                    path = save_intrinsics(
                        args.out,
                        result["camera_matrix"],
                        result["dist_coeffs"],
                        result["image_size"],
                        rms=result["rms"],
                        flags=result["flags"],
                        square_size_mm=result["square_size_mm"],
                        board_size=result["board_size"],
                    )
                    print(f"Saved: {path}")
                    print("GazeScreen3D / ArucoScreenPose will load this file automatically.")
            if key == ord("s"):
                if result is None:
                    print("Nothing to save — press C first.")
                else:
                    path = save_intrinsics(
                        args.out,
                        result["camera_matrix"],
                        result["dist_coeffs"],
                        result["image_size"],
                        rms=result.get("rms"),
                        flags=result.get("flags"),
                        square_size_mm=result.get("square_size_mm"),
                        board_size=result.get("board_size"),
                    )
                    print(f"Saved: {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

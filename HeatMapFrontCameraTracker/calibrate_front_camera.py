"""Calibrate the front camera intrinsics with a printed chessboard."""

from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from camera_io import CameraReader


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "front_camera_calibration.npz")
WINDOW_NAME = "Front Camera Calibration"
MIN_SAMPLES = 12
CAPTURE_COOLDOWN_SEC = 0.35


def chessboard_object_points(columns, rows, square_size):
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points


def calibrate(object_points, image_points, image_size):
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    return float(rms), camera_matrix, distortion


def save_calibration(path, camera_matrix, distortion, image_size, rms):
    np.savez(
        path,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion_coefficients=np.asarray(distortion, dtype=np.float64),
        image_width=np.int32(image_size[0]),
        image_height=np.int32(image_size[1]),
        rms_error=np.float64(rms),
    )


def run(camera_index, columns, rows, square_size, output_path):
    reader = CameraReader(camera_index, width=640, height=480)
    reader.start()
    if not reader.is_opened():
        print(f"Impossibile aprire la front camera {camera_index}.")
        return 1

    pattern_size = (columns, rows)
    object_template = chessboard_object_points(columns, rows, square_size)
    object_points = []
    image_points = []
    image_size = None
    latest_corners = None
    latest_found = False
    last_capture_time = 0.0

    print(
        f"Scacchiera: {columns}x{rows} angoli interni, "
        f"quadrato {square_size:g} mm."
    )
    print("Muovi e inclina la scacchiera. SPAZIO=salva vista, C=calibra, Q=esci.")

    try:
        while True:
            ret, frame = reader.read()
            if not ret:
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    return 1
                continue

            image_size = (frame.shape[1], frame.shape[0])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            latest_found, corners = cv2.findChessboardCorners(
                gray,
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_FAST_CHECK,
            )
            latest_corners = None
            display = frame.copy()
            if latest_found:
                latest_corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )
                cv2.drawChessboardCorners(
                    display,
                    pattern_size,
                    latest_corners,
                    True,
                )

            color = (0, 255, 0) if latest_found else (0, 0, 255)
            status = (
                f"Scacchiera: {'OK' if latest_found else 'non trovata'} | "
                f"campioni {len(image_points)}/{MIN_SAMPLES}"
            )
            cv2.putText(
                display,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                "SPAZIO salva | C calibra | Q esci",
                (12, display.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord(" "):
                now = time.perf_counter()
                if not latest_found or latest_corners is None:
                    print("Scacchiera non trovata: campione ignorato.")
                elif now - last_capture_time < CAPTURE_COOLDOWN_SEC:
                    print("Aspetta e cambia posizione prima del prossimo campione.")
                else:
                    object_points.append(object_template.copy())
                    image_points.append(latest_corners.copy())
                    last_capture_time = now
                    print(f"Campione {len(image_points)} salvato.")
            if key == ord("c"):
                if len(image_points) < MIN_SAMPLES:
                    print(
                        f"Servono almeno {MIN_SAMPLES} campioni; "
                        f"attuali: {len(image_points)}."
                    )
                    continue
                rms, camera_matrix, distortion = calibrate(
                    object_points,
                    image_points,
                    image_size,
                )
                save_calibration(
                    output_path,
                    camera_matrix,
                    distortion,
                    image_size,
                    rms,
                )
                print(f"Calibrazione salvata: {output_path}")
                print(f"Errore RMS: {rms:.3f} px")
                print("Camera matrix:\n", camera_matrix)
                print("Distortion:\n", distortion.ravel())
                return 0
    finally:
        reader.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Calibra la front camera usando una scacchiera stampata."
    )
    parser.add_argument("camera_index", type=int, help="Indice della front camera")
    parser.add_argument(
        "--columns",
        type=int,
        default=9,
        help="Angoli interni orizzontali della scacchiera (default: 9)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Angoli interni verticali della scacchiera (default: 6)",
    )
    parser.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="Lato reale di un quadrato in mm (default: 25)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"File di output (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.camera_index,
            args.columns,
            args.rows,
            args.square_mm,
            os.path.abspath(args.output),
        )
    )


if __name__ == "__main__":
    main()

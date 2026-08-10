"""
Fase 1 — Test solo front camera + ArUco (senza IR / heatmap).

Usage:
  python test_aruco_front.py [camera_index]

Apre a schermo intero i 4 marker ArUco agli angoli del monitor e mostra
la webcam frontale con rilevamento + contorno schermo quando tutti e 4 gli ID sono visibili.
"""

import os
import sys

import cv2

import aruco_screen
import gaze_screen
from camera_io import CameraReader
from input_poll import poll_control_key

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    screen_w, screen_h = gaze_screen.get_screen_size()
    tracker = aruco_screen.ArucoScreenTracker(screen_w, screen_h)
    corner_markers = aruco_screen.ScreenCornerMarkers(screen_w, screen_h)

    reader = CameraReader(camera_index, width=640, height=480)
    reader.start()
    if not reader.is_opened():
        print(f"Could not open camera {camera_index}")
        return

    camera_window = "ArUco Front Camera Test"
    cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
    corner_markers.open_corner_window()

    print("ArUco test — Q to quit, M to toggle corner markers on screen")
    print("I 4 marker compaiono agli angoli del monitor (finestra 'ArUco Screen Corners').")
    print("Punta la front camera verso lo schermo finché vedi 4/4 corner markers.")

    try:
        while True:
            ret, frame = reader.read()
            if ret:
                display = tracker.process(frame)
                cv2.imshow(camera_window, display)

            corner_markers.refresh_corner_window()

            key = poll_control_key(camera_window)
            if key == ord("q"):
                break
            if key == ord("m"):
                visible = corner_markers.toggle()
                print(f"Corner markers on screen: {'ON' if visible else 'OFF'}")
                corner_markers.refresh_corner_window()
    finally:
        reader.stop()
        corner_markers.close_corner_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
Stereo IR eye tracker (2 cameras) + optional front camera + fullscreen heatmap.

Self-contained: no dependency on FrontCameraTracker or 3DTracker modules.

Usage:
  cd HeatMapFrontCameraTracker
  python HeatMapFrontCameraTracker.py

Select left/right IR cameras and optional front camera in the GUI.
Gaze on the heatmap uses the combined left+right direction (VR-style fusion).

Calibration on heatmap window: C=center, arrow keys at screen edges (up/down/left/right)
ArUco corner markers on screen when front camera is enabled (M to toggle).
"""

from app_session import run
from camera_io import (
    CAMERA_OPEN_DELAY_SEC,
    CAPTURE_TARGET_FPS,
    CameraReader,
    configure_capture,
    open_camera_capture,
    stop_all_readers,
)
from input_poll import poll_control_key
from tracker_gui import selection_gui

__all__ = [
    "CAMERA_OPEN_DELAY_SEC",
    "CAPTURE_TARGET_FPS",
    "CameraReader",
    "configure_capture",
    "open_camera_capture",
    "poll_control_key",
    "run",
    "selection_gui",
    "stop_all_readers",
]


def main():
    choice = selection_gui()
    if choice["left"] is None and choice["right"] is None:
        return
    run(
        choice["left"],
        choice["right"],
        choice["front"],
        flip_left=choice["flip_left"],
        flip_right=choice["flip_right"],
        mirror_left=choice["mirror_left"],
        mirror_right=choice["mirror_right"],
        flip_front=choice["flip_front"],
        mirror_front=choice["mirror_front"],
    )


if __name__ == "__main__":
    main()

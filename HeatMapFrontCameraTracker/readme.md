# HeatMap Stereo Tracker

Self-contained eye tracking with a **fullscreen gaze heatmap** on your monitor.

Designed for DIY glasses with:
- **2 near-eye IR cameras** (left + right), like the stereo setup in `3DTracker/Orlosky3DEyeTrackerStereo.py`
- **1 optional front camera** (scene preview with a red gaze dot)

This module does **not** import from `FrontCameraTracker/` or `3DTracker/`. All logic lives in this folder.

## What it does

1. Tracks the pupil in each IR stream and computes a 3D gaze direction per eye.
2. **Fuses** left + right into one combined gaze direction (normalized average).
3. Maps combined gaze with a five-point calibration (**C** + four arrow keys).
4. With a front camera, estimates the glasses-to-screen 6DoF pose from ArUco markers and compensates head movement.
5. Draws a live heatmap overlay on a fullscreen OpenCV window.
6. Writes `gaze_vector.txt` in **12-value stereo format** (left 6 + right 6).

## Requirements

- Python 3
- OpenCV (`opencv-python`)
- NumPy
- Tkinter (usually included with Python on Windows)

```bash
pip install opencv-python numpy
```

## Hardware

Typical setup with three USB cameras:

| Camera | Role | GUI field |
|--------|------|-----------|
| IR #0 | Left eye | Left eye IR |
| IR #1 | Right eye | Right eye IR |
| Webcam #2 | Scene / face | Front camera (optional) |

Inexpensive IR cameras used in the main project (GC0308, ~$17) work well. See the root `README.md` and `3DTracker/readme.md` for DIY glasses links.

**Mono fallback:** set **Right eye IR** to `None` to run with a single IR camera.

## Run

From this directory:

```bash
cd HeatMapFrontCameraTracker
python HeatMapFrontCameraTracker.py
```

1. Select **Left eye IR**, **Right eye IR**, and **Front camera** (`None` to disable front).
2. Enable **Flip left/right image** if a camera is mounted upside down.
3. Click **Start**.

## Windows

| Window | Description |
|--------|-------------|
| `Original Left Eye` / `Original Right Eye` | Raw IR feeds |
| `left Tracking` / `right Tracking` | Pupil ellipse + gaze vector per eye |
| `External Camera (Gaze)` | Front camera with red dot (if enabled) |
| `Gaze Heatmap` | Fullscreen heatmap (focus here for calibration) |

**Click** on a `left Tracking` or `right Tracking` window to lock that eye’s sphere center (same idea as pressing **C** for tracker lock on one eye).

## Calibration (heatmap window)

Calibrate on the **Gaze Heatmap** window while looking at the matching spot on your **physical monitor**:

| Step | Key | Action |
|------|-----|--------|
| 1 | **C** | Look at the **screen center** |
| 2 | **↑** | Look at the top target |
| 3 | **↓** | Look at the bottom target |
| 4 | **←** | Look at the left target |
| 5 | **→** | Look at the right target |

Recording starts after all five points are valid. With a front camera selected, keep at least two ArUco markers visible while saving every point. The HUD shows `Gaze 6DoF: pronto` when head compensation is calibrated.

### Other keys (heatmap window)

| Key | Action |
|-----|--------|
| **H** | Reset heatmap calibration |
| **K** | Clear heatmap accumulation |
| **X** | Swap yaw/pitch axes |
| **S** | Save heatmap PNG to `gaze_heatmap.png` |
| **Space** | Pause |
| **Q** | Quit |

## Output

### `gaze_vector.txt`

Continuously updated stereo line (12 comma-separated floats):

```text
left_origin_x,left_origin_y,left_origin_z,left_direction_x,left_direction_y,left_direction_z,right_origin_x,right_origin_y,right_origin_z,right_direction_x,right_direction_y,right_direction_z
```

An eye with no valid track is written as six zeroes. Compatible with `VREyeTracker/EyeTrackerStereo.cs` if you point `GazeFilePath` to this file.

### `gaze_heatmap.png`

Saved when you press **S** on the heatmap window.

## Files

| File | Purpose |
|------|---------|
| `HeatMapFrontCameraTracker.py` | GUI, camera loop, heatmap integration |
| `eye_tracker.py` | Per-eye pupil tracking, stereo fusion, front-camera projection |
| `gaze_screen.py` | Gaze-to-screen mapping and heatmap rendering |

## How fusion works

Each eye produces its own unit gaze direction. The heatmap and front camera use:

```text
combined = normalize(dir_left + dir_right)
```

If only one eye is valid, that eye alone is used.

This is the same *idea* as the combined gaze sphere in Unity stereo mode, but mapped to your **monitor** via **C/B/R** instead of a fixed VR distance.

## Tips

- Keep the glasses still while pressing each calibration key; after `Gaze 6DoF: pronto`, normal head movement is compensated.
- Keep all four markers visible when possible; two are the minimum for pose estimation.
- For heatmap-only use, set **Front camera** to `None`.
- Camera indices depend on USB order; use the GUI dropdowns to assign left, right, and front correctly.

## Relation to other repo modules

| Module | Use case |
|--------|----------|
| `3DTracker/Orlosky3DEyeTracker.py` | Single IR, gaze vector only |
| `3DTracker/Orlosky3DEyeTrackerStereo.py` | Stereo IR, Unity VR, no heatmap |
| `FrontCameraTracker/` | Single IR + front camera, red dot, no heatmap |
| **This folder** | Stereo IR + optional front + **monitor heatmap** |

## ArUco markers (front camera)

Optional screen registration using printed markers on the monitor corners.

### Phase 0 — Generate markers

```bash
python generate_aruco_markers.py
```

Print the PNGs in `aruco_markers/` and tape them to the monitor:

```text
[0] top-left          [1] top-right
[3] bottom-left       [2] bottom-right
```

### Phase 1 — Test front camera only

```bash
python test_aruco_front.py 2
```

Replace `2` with your front camera index. Press **Q** to quit.

### Phase 2 — Integrated in main app

Run `HeatMapFrontCameraTracker.py` with **Front camera** enabled. The window **External Camera (Gaze)** shows:

- Detected markers (green boxes + ID labels)
- Cyan monitor outline when all 4 corner IDs are visible
- Status line on the heatmap HUD (`ArUco homography: OK`)

### Phase 3 — 6DoF head compensation (integrated)

When **Front camera** is enabled:

1. `solvePnP` estimates the screen-to-front-camera rotation and translation from the marker corners.
2. The five calibration targets estimate the fixed IR-gaze-to-front-camera rotation.
3. Every gaze ray is transformed into the current front-camera frame.
4. The ray is intersected with the current screen plane, compensating glasses/head rotation and translation.

Complete **C + ↑ + ↓ + ← + →** while at least two markers are visible. Four visible markers give the most stable pose. Once calibrated, the HUD shows `Mapping: ArUco 6DoF (compensazione testa)`.

If marker pose is temporarily lost after 6DoF calibration, recording pauses instead of writing incorrect heatmap points. Without a front camera, the same five points provide a static piecewise mapping.

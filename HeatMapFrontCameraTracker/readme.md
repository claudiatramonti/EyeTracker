# HeatMap Stereo Tracker

Self-contained eye tracking with a **fullscreen gaze heatmap** on your monitor.

Designed for DIY glasses with:
- **2 near-eye IR cameras** (left + right), like the stereo setup in `3DTracker/Orlosky3DEyeTrackerStereo.py`
- **1 optional front camera** (scene preview with a red gaze dot)

This module keeps tracking/heatmap logic in this folder. Optionally it loads
`3DTracker/gl_sphere.py` and blends the wireframe sphere into the IR camera
previews on the heatmap (same composite as Orlosky; no separate GL window).

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
- Optional OpenGL sphere: `PyQt5` + `PyOpenGL` (and `3DTracker/gl_sphere.py`)

```bash
pip install opencv-python numpy
# optional sphere window:
pip install PyQt5 PyOpenGL
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

## Front camera calibration

Before using ArUco 6DoF head compensation, calibrate the front camera once.
Print a chessboard with **10×7 squares** (therefore **9×6 internal corners**) and
measure the side of one square. With the tracker closed, run:

```bash
python calibrate_front_camera.py CAMERA_INDEX --columns 9 --rows 6 --square-mm 25
```

Replace `CAMERA_INDEX` and `25` with your front camera index and measured square
size. In the calibration window:

1. Hold the whole chessboard visible.
2. Press **Space** to save a view.
3. Collect at least 12 views at different positions, distances and angles.
4. Press **C** to calibrate and save `front_camera_calibration.npz`.

Prefer 20–30 varied views. An RMS error below about 1 pixel is good; above 2
pixels, repeat with sharper and more varied images. The main tracker loads this
file automatically. The HUD reports either `Front calibrata RMS ...` or
`Front intrinseci stimati`.

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
| 1 | **C** | Look at the **screen center** (also links IR ↔ front camera) |
| 2 | **↑** | Look at the top target (optional static fallback) |
| 3 | **↓** | Look at the bottom target |
| 4 | **←** | Look at the left target |
| 5 | **→** | Look at the right target |

With a front camera and visible ArUco markers, **C alone** enables the primary
map: IR gaze → front pinhole at **FOV 60°** → ArUco homography → monitor pixels.
A red dot shows the gaze on the front preview. If the projected point leaves the
monitor, the HUD reports `FUORI SCHERMO`. The five arrow keys still build the
static piecewise map used as fallback when markers/FOV mapping are unavailable.

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
- Keep all four markers visible when possible; three are the minimum for pose estimation.
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

### Phase 3 — FOV + ArUco screen mapping (primary)

With **Front camera** enabled:

1. Front gaze projection uses a **60° FOV** pinhole (`fx = (W/2)/tan(30°)`).
2. Press **C** while looking at the monitor center → builds `R_gaze_to_cam` (IR ↔ front).
3. ArUco corner markers give a live homography front-image → monitor pixels.
4. Heatmap point = gaze projected onto the front view, then warped by the homography.
5. If the point falls outside the monitor, HUD shows `FUORI SCHERMO`.

Keep **4 markers** visible. Chessboard `front_camera_calibration.npz` (if present)
still improves ArUco pose estimates; gaze→front uses FOV 60° as specified.

The older 6DoF delta path remains in code but is **off** by default
(`ENABLE_HEAD_POSE_MAPPING = False`).

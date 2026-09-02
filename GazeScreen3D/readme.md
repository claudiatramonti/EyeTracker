# GazeScreen3D

Clean pipeline: **IR gaze ∩ ArUco screen plane → monitor pixel + heatmap**.

Does not depend on `HeatMapFrontCameraTracker`. Reuses `ArucoScreenPose` for screen pose.

## Pipeline

1. IR camera(s) → 3D gaze direction (pupil / eye sphere)
2. Press **C** looking at **screen center** → `R_gaze_to_cam` (eye space → front camera)
3. *(Optional)* Look at each screen edge and press **arrow keys** → yaw/pitch scales
   refine the eye→cam direction (does **not** replace ArUco mapping)
4. Front camera + corner ArUco → screen plane in front-camera coords (`R`, `t`)
5. Ray from camera origin along rotated gaze ∩ plane → `(u, v)` on the monitor  
   (or **off-screen** if the hit is outside the window)

## Run

```bash
cd GazeScreen3D
python GazeScreen3D.py
```

Select left IR (optional right), and front camera.

## Keys

| Key | Action |
|-----|--------|
| **C** | Calibrate: look at physical screen center |
| **↑ ↓ ← →** | Optional edge samples (refine yaw/pitch scale after C) |
| **E** | Reset edge scales (`sx=sy=1`), keep C |
| **click IR** | Lock eyeball center on that eye preview |
| **U** | Unlock eye centers (auto-track again) |
| **M** | Toggle corner ArUco markers |
| **V** | Toggle camera previews |
| **K** | Clear heatmap |
| **-** / **+** | Front-camera assumed HFOV |
| **0** | Reset HFOV to 60° |
| **Q** | Quit |

## Setup

0. *(Optional)* Chessboard calib tool lives in `ArucoScreenPose/calibrate_front_camera.py`.
   Loading of `front_camera.npz` is **off by default** for now (HFOV 60° + `-`/`+`).
1. Point the front camera at this window until HUD shows **Pose OK with 4/4** ArUco corners.
2. Look at the **cyan cross** at the **center of the monitor** (full window) and press **C**.  
   Do **not** look at the small **Front** preview — that is only for debugging ArUco.
3. *(Optional)* Look at top / bottom / left / right edges and press the matching arrow.
   Yellow crosses = pending, green = sampled. Scales help edge accuracy; ArUco still
   handles head motion.
4. Look around; green crosshair = on screen, heatmap accumulates.

IR eye model uses **80° vertical FOV** (`IR_FOV_Y_DEG` in `eye_tracker.py`, GC0308). Re-press **C** after changing it.

If the cyan screen outline on the front preview is wrong, adjust HFOV with **-** / **+**.

## Files

| File | Role |
|------|------|
| `GazeScreen3D.py` | GUI + main loop |
| `eye_tracker.py` | IR pupil / gaze (Orlosky model, `IR_FOV_Y_DEG=80` for GC0308) |
| `ray_screen.py` | Gaze→cam rotation helpers + ray∩plane + mm→pixel |
| `gaze_scale_calib.py` | Optional edge yaw/pitch scales after C |
| `heatmap.py` | Accumulation + color map |
| `camera_io.py` | USB capture thread |
| `../ArucoScreenPose/calibrate_front_camera.py` | Chessboard tool (opt-in via `use_calib=True`) |

## Math (short)

- **Coordinates:** eye gaze and screen mm use **Y up**; OpenCV camera pixels use **v down**
  (converted only in `screen_mm_to_pixels`).
- `g = R_gaze_to_cam @ gaze_eye` then **flip `g[1]`** for 3D ray (eye space is Y-up;
  OpenCV camera is Y-down; C leaves `R ≈ I` for forward gaze). Pinhole preview keeps
  `cy - fy * g[1]` without that flip.
- Screen plane: point `t`, normal `R @ [0,0,1]` from ArUco `solvePnP` (updated every frame → head motion)
- Hit: ray `O + s g` with `O = (0,0,0)` (same approx as the classic red-dot pinhole)
- `P_obj = R.T @ (P_cam - t)` → map mm to window pixels

If up/down is still inverted with the correct flip setting, re-press **C** at screen center.

See `ArucoScreenPose/math_explaination.md` for screen pose details.

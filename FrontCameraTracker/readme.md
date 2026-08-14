# Orlosky3DEyeTrackerFrontCamera

Mono IR eye tracker + **external (front) camera preview** with a red gaze dot.

This script does **one job**: estimate a 3D gaze direction from the eye camera, rotate it into the front-camera frame, and project it as a **2D point** on the front-camera image. It does **not** map gaze onto a physical monitor (see `GazeScreen3D/` for that).

Based on the Orlosky 3D eye-sphere model (same core as `3DTracker/Orlosky3DEyeTracker.py`).

---

## Run

```bash
cd FrontCameraTracker
python Orlosky3DEyeTrackerFrontCamera.py
```

Select camera index, then:

| Key | Action |
|-----|--------|
| **C** | Calibrate: current gaze → center of front camera (forward axis) |
| **Click** on “Frame with Ellipse and Rays” | Lock 2D eyeball center |
| **Q** | Quit |
| **Space** | Pause |

Eye frame is **flipped vertically** before processing (`cv2.flip(..., 0)`).

---

## Pipeline (each frame)

```
IR frame (640×480)
  → flip vertical
  → pupil (darkest blob + ellipse fit)
  → eyeball center (intersection of ellipse normals, smoothed)
  → compute_gaze_vector(pupil, center) → gaze_dir (3D, eye space)
  → [after C] R_gaze_to_cam @ gaze_dir → g (front camera space)
  → pinhole: (u, v) on front camera image → red circle
```

Two cameras:

- **Eye camera** — close-up IR; only eye rotation in the socket matters.
- **Front camera** — scene view; used only to **display** where gaze points in that image after calibration.

Both are rigidly mounted on the glasses, so they move together with the head.

---

## Coordinate systems

### Eye model space (`compute_gaze_vector`)

| Axis | Direction |
|------|-----------|
| **X** | right |
| **Y** | up |
| **Z** | forward (toward what you look at) |

Origin concept: virtual camera at `(0, 0, 3)` looks at an eye **sphere** near `z = 0`.

### Front (external) camera space (OpenCV-style)

| Axis | Direction |
|------|-----------|
| **X** | right |
| **Y** | down (image rows increase downward) |
| **Z** | forward (optical axis, +Z into the scene) |

Pinhole projection uses the **minus** on Y when drawing pixels (see below).

---

## Step 1 — Pupil and eyeball center (2D)

1. **Darkest region** → pupil seed `(x_p, y_p)`.
2. **Threshold + contours** → fit ellipse to the pupil; ellipse center ≈ pupil.
3. **Several ellipses** over time → orthogonal rays from each ellipse; **intersections** estimate the **2D eyeball center** `(c_x, c_y)` (running average, or **locked** after click / **C**).

The yellow circle = locked or averaged center. The line center → pupil is the 2D gaze hint in the IR image.

---

## Step 2 — Pupil → 3D gaze (`compute_gaze_vector`)

Inputs: pupil pixel `(x, y)`, center pixel `(c_x, c_y)`, viewport `W × H` (default 640×480).

### 2.1 Virtual camera and NDC

Fixed parameters in code:

- Vertical FOV: `fov_y_deg = 45.0` (model parameter — should match your IR module; GazeScreen3D uses 80° for GC0308)
- Camera at `C_cam = (0, 0, 3)`
- Far plane at `z = 3 - 100`

Normalized device coordinates from pupil pixel (Y up in world):

```
ndc_x = 2·x/W - 1
ndc_y = 1 - 2·y/H
```

Far-plane point (world units):

```
half_h = tan(fov_y/2) · far_clip
half_w = half_h · (W/H)

P_far = (ndc_x · half_w,  ndc_y · half_h,  z_cam - far_clip)
```

Ray from camera through pupil (then reversed for intersection math):

```
d = normalize(P_far - C_cam)
```

### 2.2 Eye sphere

Sphere center in model space (from 2D center on screen):

```
off_x = (c_x/W)·2 - 1
off_y = 1 - (c_y/H)·2
O = (off_x · 1.5,  off_y · 1.5,  0)
R = 1/1.05   (inner radius)
```

### 2.3 Ray–sphere intersection

Ray `P(t) = C_cam + t·d̂`. Solve `‖P(t) - O‖² = R²` → pick smallest positive `t`.

Intersection point `Q` on the sphere surface. Local direction from sphere center:

```
t̂ = normalize(Q - O)
```

### 2.4 Gaze direction output

The code rotates the local “rest” axis `(0, 0, R)` to align with `t̂` (axis–angle rotation matrix), then:

```
gaze_dir = normalize(R_axis @ (0, 0, R))
```

Stored as `last_gaze_dir` (unit vector, eye model space, **Y up**).

Also written to `gaze_vector.txt`: 6 floats `[O_x, O_y, O_z, gaze_x, gaze_y, gaze_z]`.

---

## Step 3 — Calibration **C** (`calibrate_gaze_to_external`)

While you look at whatever should be “straight ahead” for the front camera (typically center of its view), press **C**.

Goal: find rotation **R** such that current gaze aligns with front-camera forward:

```
R_gaze_to_cam @ gaze_dir  ≈  (0, 0, 1)
```

Computed with **Rodrigues** (`rotation_from_a_to_b`):

```
v = gaze × forward
c = gaze · forward
K = [v]×   (skew-symmetric)
R = I + K·sin(θ) + K²·(1-cos(θ))     with θ = arccos(c), axis = v/‖v‖
```

Side effects of **C**:

- Locks 2D sphere center in the IR image.
- Fixes 3D `calibrated_sphere_center` to the current `O`.

**Important:** **C** is a **rotation only**. It sets the zero direction, not how many degrees a pupil shift produces. If gaze range feels “too small” on the front image, adjust `fov_y_deg` (and/or IR resolution), not only **C**.

When gaze ≈ +Z at calibration, `R_gaze_to_cam` is often **near identity**; eye **Y-up** vs camera **Y-down** is then handled in the pinhole sign (next step).

---

## Step 4 — Gaze → red dot on front camera (`update_gaze_circle_from_current_gaze`)

After calibration, each frame:

```
g = R_gaze_to_cam @ last_gaze_dir
```

Reject if `g_z ≤ 0` (behind the camera).

Pinhole projection (focal length `f_x, f_y ≈ 600` px, principal point center of 640×480):

```
u = c_x + f_x · (g_x / g_z)
v = c_y - f_y · (g_y / g_z)    ← minus: eye Y-up ↔ image v-down
```

`(u, v)` = red circle on **External Camera (Gaze)** window.

This is **only** mapping into the front-camera **image**. No monitor geometry, no ArUco.

---

## Parameters (tunable in code)

| Symbol / name | Default | Role |
|---------------|---------|------|
| `fov_y_deg` | 45° | IR virtual camera FOV in `compute_gaze_vector` — **angular scale** of gaze |
| `EXT_FX`, `EXT_FY` | 600 | Front-camera pinhole focal length (pixels) |
| `EXT_WIDTH`, `EXT_HEIGHT` | 640×480 | Front preview size |
| `max_observed_distance` | 202 (fixed in code) | Radius of eye sphere circle drawn in IR view |

---

## What this app does **not** do

- No screen / monitor mapping.
- No head-pose compensation beyond “eye + front move together on glasses”.
- No second calibration for scale (edge vs center).

For **eye → front → physical screen**, use **`GazeScreen3D/`** (ArUco plane + ray intersection). It reuses the same eye math (`eye_tracker.py`, fork of this logic) but adds `screen_pose.py` and `ray_screen.py`.

---

## Relation to GazeScreen3D

| Piece | Orlosky FrontCamera | GazeScreen3D |
|-------|---------------------|--------------|
| Pupil + sphere + `compute_gaze_vector` | here | `GazeScreen3D/eye_tracker.py` (copy, `IR_FOV_Y_DEG=80`) |
| **C** → `R_gaze_to_cam` | yes | yes |
| Front red dot (pinhole) | yes | yes (debug preview) |
| Front → monitor | **no** | yes (3D ray ∩ ArUco plane) |

---

## Limitations

1. **Single rotation at C** — roll around forward is ambiguous when gaze ≈ forward.
2. **`fov_y_deg = 45`** may not match your IR module (e.g. GC0308 ~80°) → gaze compressed toward center of front image.
3. **No smoothing** on gaze in this script (unlike GazeScreen3D’s `GAZE_DIRECTION_SMOOTH_ALPHA`).
4. Eye camera hardcoded index `0`, front index `1` in `process_camera()` — adjust for your setup.

---

## File output

`gaze_vector.txt` (working directory): one CSV line per frame overwrite:

```
O_x, O_y, O_z, gaze_x, gaze_y, gaze_z
```

Useful for external tools; optional for the GUI itself.

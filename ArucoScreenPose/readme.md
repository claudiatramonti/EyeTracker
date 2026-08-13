# ArUco Screen Pose

Step 1 of the new gaze-on-screen pipeline: **front camera only**.

No IR cameras, no heatmap, no gaze vector. This module estimates how the
front camera is angled relative to the monitor using ArUco markers at the
four screen corners.

## Why

With a later eye-tracking step you already know where the gaze ray falls in
the front-camera image (gaze vector + FOV). If you also know where the
monitor sits in that same camera view, you know where the user is looking on
the screen — or that they are looking off-screen.

This folder is only the first piece: **register the screen in the front camera**.

## Run

```bash
cd ArucoScreenPose
python ArucoScreenPose.py
```

Or skip the GUI:

```bash
python ArucoScreenPose.py 2
python ArucoScreenPose.py 2 --flip --mirror
```

Replace `2` with the front camera index.

## What you should see

One window covering the monitor (above the taskbar):

- ArUco **ID 0–3** at the four corners (no printing needed)
- **Front camera** preview in the centre, for debugging
- Cyan monitor outline + XYZ axes when pose is valid
- HUD with yaw / pitch / roll and camera position in millimetres
- Small top-view and side-view diagrams of camera vs screen

Point the glasses (or webcam) at this window until the HUD shows `Pose OK 4/4`.

## Keys

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **M** | Toggle corner markers |
| **-** / **+** | Decrease / increase assumed horizontal FOV |
| **0** | Reset FOV to 60° |
| **P** | Print current pose to the console |

If the cyan rectangle does not line up with the real monitor in the camera
image, adjust FOV with **-** / **+** until it does. Wrong FOV warps the angles.
A wrong physical screen size mostly affects distance, not yaw/pitch/roll.

## Angles

Object frame: origin at window centre, **X right**, **Y down**, **Z into the screen**.
The front camera is on the viewer side (negative Z).

| Value | Meaning |
|-------|---------|
| **Yaw** | Camera looking toward the right (+) or left (−) of the screen |
| **Pitch** | Camera looking up (+) or down (−) |
| **Roll** | Clockwise tilt as seen from behind the camera |
| **Incidence** | Angle between the optical axis and the screen normal (0° = square-on) |
| **Distance** | Millimetres from camera to the screen plane |

## Files

| File | Purpose |
|------|---------|
| `ArucoScreenPose.py` | GUI, camera loop, debug layout |
| `screen_pose.py` | ArUco detection, `solvePnP` pose, overlays |
| `camera_io.py` | USB capture thread |

Independent of `HeatMapFrontCameraTracker/`. Same marker IDs (0–3, clockwise from top-left).

## Next steps (not in this module)

1. Combine IR gaze vector with front-camera FOV → pixel in the scene image.
2. Intersect that ray with the ArUco screen plane → monitor pixel, or off-screen.

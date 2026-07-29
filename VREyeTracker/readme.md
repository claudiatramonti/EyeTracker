# Unity VR Gaze Visualizer and Calibration Scripts

These Unity C# scripts visualize 3D gaze rays produced by the external Python eye tracker, [Orlosky3DEyeTracker.py](https://github.com/JEOresearch/EyeTracker/blob/main/3DTracker/Orlosky3DEyeTracker.py). The Python tracker continuously writes gaze origins and directions to a text file, which the Unity component reads in real time.

To help support this software and other open-source projects, please consider subscribing to my [YouTube channel](https://www.youtube.com/@jeoresearch) or [joining for $1 per month](https://www.youtube.com/@jeoresearch/join).

Demo video: https://youtu.be/h2hPR8Mx6Ho.

## Available Scripts

- `EyeTracker.cs` visualizes a single gaze origin and direction.
- `EyeTrackerStereo.cs` visualizes left- and right-eye gaze independently, calibrates both eyes, and displays a combined gaze point.

Both files currently define a component named `EyeTracker`. Use only one of them in a Unity project at a time. When using the stereo version, copy `EyeTrackerStereo.cs` into the Unity project's `Assets` folder as `EyeTracker.cs` (or rename both its file and class to `EyeTrackerStereo`) so that the MonoBehaviour filename matches the class name.

## Unity Setup

1. Choose the single-eye or stereo script and add it to the Unity project's `Assets` folder as described above.
2. Open the script and change `GazeFilePath` to the path of the gaze file written by the Python tracker. The default is:

   ```csharp
   private const string GazeFilePath = @"C:\Storage\gaze_vector.txt";
   ```

3. Add the `EyeTracker` component to the headset or center-eye camera GameObject. Gaze points and calibration targets use this transform as their local frame.
4. Enter Play mode and start the Python tracker. The script creates all gaze and calibration objects at runtime; no prefabs are required.
5. If needed, tune the public fields in the Inspector:

   - `Sphere Distance` and `Sphere Radius` control the gaze-point placement and size.
   - `Read Interval` controls how often the gaze file is read.
   - `Calibration Target Offset` controls the spacing of the five calibration targets.
   - The calibration accuracy grid fields control its transform, distance, spacing, size, and colors.

If `Grid View Transform` is not assigned, the accuracy grid is positioned relative to the GameObject containing the script.

## Single-Eye File Format

`EyeTracker.cs` expects six comma-, semicolon-, or whitespace-separated floating-point values:

```text
origin_x,origin_y,origin_z,direction_x,direction_y,direction_z
```

## Stereo File Format

`EyeTrackerStereo.cs` expects 12 values. Supply all six left-eye values first, followed by all six right-eye values:

```text
left_origin_x,left_origin_y,left_origin_z,left_direction_x,left_direction_y,left_direction_z,right_origin_x,right_origin_y,right_origin_z,right_direction_x,right_direction_y,right_direction_z
```

Commas, semicolons, spaces, tabs, and line breaks are accepted as separators. The script uses the first 12 values in the file. Direction vectors should use Unity-compatible axes; `PythonToUnityDirection` is the place to change axis order or signs if the Python coordinate system differs.

Before calibration, the left gaze point is cyan and the right gaze point is magenta. The stereo origins are displayed with a fixed horizontal offset of 0.032 Unity units on each side of the headset center. After calibration, the individual points become translucent and a white combined gaze point appears midway between them.

## Stereo Calibration

Keep the headset still and look directly at each red target before pressing `C`.

1. Press `C` to start and show the **Up** target.
2. Look at Up and press `C` to capture it.
3. Repeat for **Right**, **Down**, **Left**, and **Center**, pressing `C` once for each target.
4. After the Center sample is captured, calibration is applied independently to both eyes and the combined gaze point appears.

If either eye has no valid direction, the sample is not captured. Continue looking at the target and press `C` again. Pressing `C` after a completed calibration starts a new calibration.

## Stereo Keyboard Controls

| Key | Action |
| --- | --- |
| `C` | Start or advance five-point stereo calibration |
| `B` | Leave permanent copies of the current left- and right-eye gaze points in the scene |
| `G` | Show or hide the 5-by-5 calibration accuracy grid; hiding it resets the grid test |
| `T` | Start the grid test or record the current highlighted target |

## Calibration Accuracy Test

1. Complete the five-point calibration.
2. Press `G` to show the 5-by-5 grid.
3. Press `T` once to begin at the upper-left target.
4. Look at the highlighted green target and press `T` to record the combined gaze position.
5. Repeat for all 25 targets.

Each recorded target receives an angular-error label in degrees, and the final result includes the average angular error. Toggle the grid off and on with `G` to reset and repeat the test.

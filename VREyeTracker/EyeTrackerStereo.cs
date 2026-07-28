using System;
using System.Globalization;
using System.IO;
using UnityEngine;

public class EyeTracker : MonoBehaviour
{
    private const string GazeFilePath = @"C:\Storage\Google Drive\Software\EyeTracker3DPython\Orlosky3DEyeTracker\gaze_vector.txt";
    private const int LeftEye = 0;
    private const int RightEye = 1;
    private const int EyeCount = 2;
    private const float EyeOriginHorizontalOffset = 0.032f;
    private static readonly string[] EyeNames = { "Left", "Right" };

    [Header("Visualization")]
    public float sphereDistance = 2.0f;
    public float sphereRadius = 0.075f;
    public float readInterval = 0.01f;

    [Header("Five-Point Calibration")]
    public float calibrationTargetOffset = 0.3f;

    [Header("Calibration Accuracy Grid")]
    public Transform gridViewTransform;
    public float gridDistance = 2.0f;
    public float gridSpacing = 0.2f;
    public float gridSphereRadius = 0.0375f;
    public Color gridSphereColor = Color.yellow;
    public Color gridHighlightColor = Color.green;

    [Header("Debug")]
    public bool verboseLogging = true;

    private readonly GameObject[] gazeSpheres = new GameObject[EyeCount];
    private GameObject combinedGazeSphere;
    private GameObject calibrationSphere;
    private GameObject calibrationAccuracyGrid;
    private const int AccuracyGridSize = 5;
    private const int AccuracyGridSphereCount = AccuracyGridSize * AccuracyGridSize;
    private readonly GameObject[] accuracyGridSpheres =
        new GameObject[AccuracyGridSphereCount];
    private int activeGridTestIndex = -1;
    private float gridAngularErrorSum;
    private int gridAngularErrorCount;
    private bool calibrationComplete;

    private float nextReadTime = 0f;

    private readonly Vector3[] pythonOrigins = new Vector3[EyeCount];
    private readonly Vector3[] pythonDirections = new Vector3[EyeCount];

    private readonly Vector3[] latestRawUnityLocalDirections = {
        Vector3.forward,
        Vector3.forward
    };
    private readonly bool[] hasValidRawDirections = new bool[EyeCount];

    private readonly bool[] hasCalibrations = new bool[EyeCount];
    private readonly Vector3[] calibratedRawForward = new Vector3[EyeCount];
    private readonly Vector3[] calibratedRawUp = new Vector3[EyeCount];
    private readonly Vector3[] calibratedRawRight = new Vector3[EyeCount];
    private readonly float[] calibrationM00 = new float[EyeCount];
    private readonly float[] calibrationM01 = new float[EyeCount];
    private readonly float[] calibrationM10 = new float[EyeCount];
    private readonly float[] calibrationM11 = new float[EyeCount];

    private const int UpSample = 0;
    private const int RightSample = 1;
    private const int DownSample = 2;
    private const int LeftSample = 3;
    private const int CenterSample = 4;
    private const int CalibrationSampleCount = 5;
    private readonly Vector3[,] calibrationDirections =
        new Vector3[EyeCount, CalibrationSampleCount];
    private readonly Vector3[,] calibrationOrigins =
        new Vector3[EyeCount, CalibrationSampleCount];
    private readonly bool[,] hasCalibrationSamples =
        new bool[EyeCount, CalibrationSampleCount];

    private enum CalibrationStage
    {
        Idle,
        UpTargetVisible,
        RightTargetVisible,
        DownTargetVisible,
        LeftTargetVisible,
        CenterTargetVisible
    }

    private CalibrationStage calibrationStage = CalibrationStage.Idle;

    private void Start()
    {
        CreateGazeSphere();
    }

    private void Update()
    {
        if (Time.time >= nextReadTime)
        {
            nextReadTime = Time.time + readInterval;
            ReadAndApplyGaze();
        }

        if (Input.GetKeyDown(KeyCode.B))
        {
            CreatePermanentGazeSphereCopies();
        }

        if (Input.GetKeyDown(KeyCode.C))
        {
            HandleCalibrationKey();
        }

        if (Input.GetKeyDown(KeyCode.G))
        {
            ToggleCalibrationAccuracyGrid();
        }

        if (Input.GetKeyDown(KeyCode.T))
        {
            AdvanceGridAccuracyTest();
        }
    }

    private void OnDrawGizmos()
    {
        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            if (gazeSpheres[eyeIndex] == null)
                continue;

            Gizmos.color = eyeIndex == LeftEye ? Color.cyan : Color.magenta;
            Vector3 gazeOriginWorld = transform.TransformPoint(GetEyeGazeOriginLocal(eyeIndex));
            Gizmos.DrawLine(gazeOriginWorld, gazeSpheres[eyeIndex].transform.position);
        }
    }

    private static Vector3 GetEyeGazeOriginLocal(int eyeIndex)
    {
        return new Vector3(
            eyeIndex == LeftEye ? -EyeOriginHorizontalOffset : EyeOriginHorizontalOffset,
            0f,
            0f
        );
    }

    private void ToggleCalibrationAccuracyGrid()
    {
        if (calibrationAccuracyGrid != null)
        {
            Destroy(calibrationAccuracyGrid);
            calibrationAccuracyGrid = null;
            ResetGridTestState();

            if (verboseLogging)
                Debug.Log("Calibration accuracy grid hidden.");

            return;
        }

        Transform view = gridViewTransform != null ? gridViewTransform : transform;

        ResetGridTestState();
        calibrationAccuracyGrid = new GameObject("CalibrationAccuracyGrid");
        calibrationAccuracyGrid.transform.SetPositionAndRotation(
            view.position + view.forward * gridDistance,
            view.rotation
        );

        const int centerIndex = AccuracyGridSize / 2;
        float diameter = gridSphereRadius * 2.0f;

        for (int row = 0; row < AccuracyGridSize; row++)
        {
            for (int column = 0; column < AccuracyGridSize; column++)
            {
                int gridIndex = row * AccuracyGridSize + column;
                GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = "GridSphere_" + row + "_" + column;
                sphere.transform.SetParent(calibrationAccuracyGrid.transform, false);
                sphere.transform.localPosition = new Vector3(
                    (column - centerIndex) * gridSpacing,
                    (centerIndex - row) * gridSpacing,
                    0.0f
                );
                sphere.transform.localRotation = Quaternion.identity;
                sphere.transform.localScale = Vector3.one * diameter;

                Renderer renderer = sphere.GetComponent<Renderer>();
                if (renderer != null)
                    renderer.material.color = gridSphereColor;

                accuracyGridSpheres[gridIndex] = sphere;
            }
        }

        if (verboseLogging)
            Debug.Log("Calibration accuracy grid shown.");
    }

    private void AdvanceGridAccuracyTest()
    {
        if (calibrationAccuracyGrid == null)
        {
            Debug.LogWarning("Show the calibration accuracy grid with G before starting the T-key test.");
            return;
        }

        if (activeGridTestIndex < 0)
        {
            activeGridTestIndex = 0;
            SetGridSphereColor(activeGridTestIndex, gridHighlightColor);

            if (verboseLogging)
                Debug.Log("Grid accuracy test started at the upper-left target.");

            return;
        }

        if (activeGridTestIndex >= AccuracyGridSphereCount)
        {
            if (verboseLogging)
                Debug.Log("Grid accuracy test is already complete. Toggle the grid with G to reset it.");
            return;
        }

        if (!calibrationComplete || combinedGazeSphere == null || !combinedGazeSphere.activeInHierarchy)
        {
            Debug.LogWarning("Complete gaze calibration before recording a grid accuracy sample.");
            return;
        }

        GameObject testedGridSphere = accuracyGridSpheres[activeGridTestIndex];
        if (testedGridSphere == null)
        {
            Debug.LogWarning("The active grid target is missing. Toggle the grid with G to reset the test.");
            return;
        }

        Vector3 combinedOrigin = transform.position;
        Vector3 combinedGazePosition = combinedGazeSphere.transform.position;
        Vector3 targetPosition = testedGridSphere.transform.position;
        Vector3 gazeVector = combinedGazePosition - combinedOrigin;
        Vector3 targetVector = targetPosition - combinedOrigin;

        if (gazeVector.sqrMagnitude < 0.000001f || targetVector.sqrMagnitude < 0.000001f)
        {
            Debug.LogWarning("Could not calculate grid error because a gaze or target vector is too small.");
            return;
        }

        CreateRecordedCombinedGazeSphere(combinedGazePosition, activeGridTestIndex);
        float angularErrorDegrees = Vector3.Angle(gazeVector, targetVector);
        CreateGridErrorLabel(testedGridSphere, activeGridTestIndex, angularErrorDegrees);
        gridAngularErrorSum += angularErrorDegrees;
        gridAngularErrorCount++;

        if (verboseLogging)
        {
            Debug.Log(
                "Grid target " + (activeGridTestIndex + 1) + "/" + AccuracyGridSphereCount
                + " angular error: "
                + angularErrorDegrees.ToString("F2", CultureInfo.InvariantCulture) + " degrees."
            );
        }

        SetGridSphereColor(activeGridTestIndex, gridSphereColor);
        activeGridTestIndex++;

        if (activeGridTestIndex < AccuracyGridSphereCount)
        {
            SetGridSphereColor(activeGridTestIndex, gridHighlightColor);
        }
        else
        {
            CreateAverageAccuracyLabel(gridAngularErrorSum / gridAngularErrorCount);

            if (verboseLogging)
                Debug.Log("Grid accuracy test complete.");
        }
    }

    private void CreateRecordedCombinedGazeSphere(Vector3 worldPosition, int gridIndex)
    {
        GameObject recordedSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        recordedSphere.name = "GridTestCombinedGaze_" + gridIndex;
        recordedSphere.transform.position = worldPosition;
        recordedSphere.transform.rotation = Quaternion.identity;
        recordedSphere.transform.localScale = Vector3.one * (gridSphereRadius * 2.0f);
        recordedSphere.transform.SetParent(null, true);

        Renderer renderer = recordedSphere.GetComponent<Renderer>();
        if (renderer != null)
        {
            Color recordedColor = new Color(0f, 0f, 0f, 0.5f);
            SetMaterialTransparency(renderer.material, recordedColor, true);
        }
    }

    private void CreateGridErrorLabel(
        GameObject testedGridSphere,
        int gridIndex,
        float angularErrorDegrees)
    {
        GameObject labelRoot = new GameObject("GridAngularError_" + gridIndex);
        labelRoot.transform.SetParent(calibrationAccuracyGrid.transform, false);
        labelRoot.transform.localPosition =
            testedGridSphere.transform.localPosition
            + new Vector3(0f, gridSphereRadius * 2.2f, -gridSphereRadius * 0.5f);
        labelRoot.transform.localRotation = Quaternion.identity;

        GameObject background = GameObject.CreatePrimitive(PrimitiveType.Quad);
        background.name = "Background";
        background.transform.SetParent(labelRoot.transform, false);
        background.transform.localPosition = Vector3.zero;
        background.transform.localRotation = Quaternion.identity;
        background.transform.localScale = new Vector3(0.14f, 0.05f, 1f);

        Collider backgroundCollider = background.GetComponent<Collider>();
        if (backgroundCollider != null)
            Destroy(backgroundCollider);

        Renderer backgroundRenderer = background.GetComponent<Renderer>();
        if (backgroundRenderer != null)
            SetMaterialTransparency(
                backgroundRenderer.material,
                new Color(1f, 1f, 1f, 0.85f),
                true
            );

        GameObject textObject = new GameObject("Text");
        textObject.transform.SetParent(labelRoot.transform, false);
        textObject.transform.localPosition = new Vector3(0f, 0f, -0.002f);
        textObject.transform.localRotation = Quaternion.identity;

        TextMesh textMesh = textObject.AddComponent<TextMesh>();
        textMesh.text = angularErrorDegrees.ToString("F2", CultureInfo.InvariantCulture) + "\u00B0";
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.fontSize = 64;
        textMesh.characterSize = 0.006f;
        textMesh.color = Color.black;
    }

    private void CreateAverageAccuracyLabel(float averageAngularErrorDegrees)
    {
        GameObject labelRoot = new GameObject("GridAverageAccuracy");
        labelRoot.transform.SetParent(calibrationAccuracyGrid.transform, false);
        labelRoot.transform.localPosition = new Vector3(
            0f,
            (AccuracyGridSize / 2) * gridSpacing + gridSphereRadius * 4.5f,
            -gridSphereRadius * 0.5f
        );
        labelRoot.transform.localRotation = Quaternion.identity;

        GameObject background = GameObject.CreatePrimitive(PrimitiveType.Quad);
        background.name = "Background";
        background.transform.SetParent(labelRoot.transform, false);
        background.transform.localPosition = Vector3.zero;
        background.transform.localRotation = Quaternion.identity;
        background.transform.localScale = new Vector3(0.64f, 0.06f, 1f);

        Collider backgroundCollider = background.GetComponent<Collider>();
        if (backgroundCollider != null)
            Destroy(backgroundCollider);

        Renderer backgroundRenderer = background.GetComponent<Renderer>();
        if (backgroundRenderer != null)
            SetMaterialTransparency(
                backgroundRenderer.material,
                new Color(1f, 1f, 1f, 0.85f),
                true
            );

        GameObject textObject = new GameObject("Text");
        textObject.transform.SetParent(labelRoot.transform, false);
        textObject.transform.localPosition = new Vector3(0f, 0f, -0.002f);
        textObject.transform.localRotation = Quaternion.identity;

        TextMesh textMesh = textObject.AddComponent<TextMesh>();
        textMesh.text =
            "Average Accuracy: "
            + averageAngularErrorDegrees.ToString("F2", CultureInfo.InvariantCulture)
            + "\u00B0";
        textMesh.anchor = TextAnchor.MiddleCenter;
        textMesh.alignment = TextAlignment.Center;
        textMesh.fontSize = 64;
        textMesh.characterSize = 0.006f;
        textMesh.color = Color.black;
    }

    private void SetGridSphereColor(int gridIndex, Color color)
    {
        if (gridIndex < 0 || gridIndex >= AccuracyGridSphereCount)
            return;

        GameObject sphere = accuracyGridSpheres[gridIndex];
        if (sphere == null)
            return;

        Renderer renderer = sphere.GetComponent<Renderer>();
        if (renderer != null)
            renderer.material.color = color;
    }

    private void ResetGridTestState()
    {
        activeGridTestIndex = -1;
        gridAngularErrorSum = 0f;
        gridAngularErrorCount = 0;

        for (int i = 0; i < accuracyGridSpheres.Length; i++)
            accuracyGridSpheres[i] = null;
    }

    private void HandleCalibrationKey()
    {
        switch (calibrationStage)
        {
            case CalibrationStage.Idle:
                ResetCalibrationSamples();
                SetPostCalibrationVisualization(false);
                ShowCalibrationTarget(UpSample);
                calibrationStage = CalibrationStage.UpTargetVisible;

                if (verboseLogging)
                    Debug.Log("Five-point calibration started. Look at UP and press C.");
                break;

            case CalibrationStage.UpTargetVisible:
                if (CaptureCalibrationSample(UpSample))
                {
                    ShowCalibrationTarget(RightSample);
                    calibrationStage = CalibrationStage.RightTargetVisible;
                }
                break;

            case CalibrationStage.RightTargetVisible:
                if (CaptureCalibrationSample(RightSample))
                {
                    ShowCalibrationTarget(DownSample);
                    calibrationStage = CalibrationStage.DownTargetVisible;
                }
                break;

            case CalibrationStage.DownTargetVisible:
                if (CaptureCalibrationSample(DownSample))
                {
                    ShowCalibrationTarget(LeftSample);
                    calibrationStage = CalibrationStage.LeftTargetVisible;
                }
                break;

            case CalibrationStage.LeftTargetVisible:
                if (CaptureCalibrationSample(LeftSample))
                {
                    ShowCalibrationTarget(CenterSample);
                    calibrationStage = CalibrationStage.CenterTargetVisible;
                }
                break;

            case CalibrationStage.CenterTargetVisible:
                if (CaptureCalibrationSample(CenterSample))
                {
                    if (BuildFivePointCalibration())
                    {
                        ClearCalibrationSphere();
                        calibrationStage = CalibrationStage.Idle;
                        SetPostCalibrationVisualization(true);

                        if (verboseLogging)
                            Debug.Log("Five-point calibration complete.");
                    }
                    else
                    {
                        ClearCalibrationSphere();
                        calibrationStage = CalibrationStage.Idle;
                        Debug.LogWarning("Five-point calibration failed. Press C to restart it.");
                    }
                }
                break;
        }
    }

    private void ResetCalibrationSamples()
    {
        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            hasCalibrations[eyeIndex] = false;

            for (int sampleIndex = 0; sampleIndex < CalibrationSampleCount; sampleIndex++)
            {
                calibrationDirections[eyeIndex, sampleIndex] = Vector3.forward;
                calibrationOrigins[eyeIndex, sampleIndex] = Vector3.zero;
                hasCalibrationSamples[eyeIndex, sampleIndex] = false;
            }
        }
    }

    private bool CaptureCalibrationSample(int sampleIndex)
    {
        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            if (!hasValidRawDirections[eyeIndex])
            {
                Debug.LogWarning(
                    EyeNames[eyeIndex] + " " + GetCalibrationSampleName(sampleIndex)
                    + " sample not captured: no valid gaze direction. Keep looking at the target and press C again."
                );
                return false;
            }
        }

        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            calibrationDirections[eyeIndex, sampleIndex] =
                latestRawUnityLocalDirections[eyeIndex].normalized;
            calibrationOrigins[eyeIndex, sampleIndex] = pythonOrigins[eyeIndex];
            hasCalibrationSamples[eyeIndex, sampleIndex] = true;

            if (verboseLogging)
            {
                Debug.Log(
                    EyeNames[eyeIndex] + " " + GetCalibrationSampleName(sampleIndex)
                    + " captured: origin=" + calibrationOrigins[eyeIndex, sampleIndex].ToString("F4")
                    + ", direction=" + calibrationDirections[eyeIndex, sampleIndex].ToString("F4")
                );
            }
        }

        return true;
    }

    private bool BuildFivePointCalibration()
    {
        bool calibratedAll = true;

        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            if (!TryBuildEyeCalibration(eyeIndex))
            {
                hasCalibrations[eyeIndex] = false;
                calibratedAll = false;
            }
        }

        if (!calibratedAll)
        {
            for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
                hasCalibrations[eyeIndex] = false;
        }

        return calibratedAll;
    }

    private bool TryBuildEyeCalibration(int eyeIndex)
    {
        if (sphereDistance <= 0.0001f || calibrationTargetOffset <= 0.0001f)
        {
            Debug.LogWarning("Calibration requires positive sphere distance and target offset values.");
            return false;
        }

        for (int sampleIndex = 0; sampleIndex < CalibrationSampleCount; sampleIndex++)
        {
            if (!hasCalibrationSamples[eyeIndex, sampleIndex])
            {
                Debug.LogWarning(
                    EyeNames[eyeIndex] + " calibration is missing the "
                    + GetCalibrationSampleName(sampleIndex) + " sample."
                );
                return false;
            }
        }

        // Construct the up plane from a common eye origin, the center gaze point,
        // and the top gaze point. Using a common origin makes these point
        // calculations equivalent to using the two captured gaze rays.
        Vector3 eyeOrigin = calibrationOrigins[eyeIndex, CenterSample];
        Vector3 centerGazePoint =
            eyeOrigin + calibrationDirections[eyeIndex, CenterSample] * sphereDistance;
        Vector3 topGazePoint =
            eyeOrigin + calibrationDirections[eyeIndex, UpSample] * sphereDistance;
        Vector3 rawForward = (centerGazePoint - eyeOrigin).normalized;
        Vector3 topRay = (topGazePoint - eyeOrigin).normalized;

        Vector3 planeNormal = Vector3.Cross(rawForward, topRay);
        if (planeNormal.sqrMagnitude < 0.000001f)
        {
            Debug.LogWarning(EyeNames[eyeIndex] + " calibration failed: center and up rays are too similar.");
            return false;
        }

        planeNormal.Normalize();
        Vector3 rawUp = Vector3.Cross(planeNormal, rawForward).normalized;
        if (Vector3.Dot(rawUp, topRay - rawForward) < 0f)
            rawUp = -rawUp;

        Vector3 rawRight = Vector3.Cross(rawUp, rawForward).normalized;

        float targetOffsetRatio = calibrationTargetOffset / sphereDistance;
        Vector2[] targetCoordinates = {
            new Vector2(0f, targetOffsetRatio),
            new Vector2(targetOffsetRatio, 0f),
            new Vector2(0f, -targetOffsetRatio),
            new Vector2(-targetOffsetRatio, 0f)
        };

        float sxx = 0f;
        float sxy = 0f;
        float syy = 0f;
        float txX = 0f;
        float txY = 0f;
        float tyX = 0f;
        float tyY = 0f;

        for (int sampleIndex = 0; sampleIndex < CenterSample; sampleIndex++)
        {
            if (!TryProjectOntoTangentPlane(
                calibrationDirections[eyeIndex, sampleIndex],
                rawForward,
                rawRight,
                rawUp,
                out Vector2 measured))
            {
                Debug.LogWarning(
                    EyeNames[eyeIndex] + " calibration failed: "
                    + GetCalibrationSampleName(sampleIndex) + " ray cannot be projected."
                );
                return false;
            }

            Vector2 target = targetCoordinates[sampleIndex];
            sxx += measured.x * measured.x;
            sxy += measured.x * measured.y;
            syy += measured.y * measured.y;
            txX += target.x * measured.x;
            txY += target.x * measured.y;
            tyX += target.y * measured.x;
            tyY += target.y * measured.y;
        }

        float normalDeterminant = sxx * syy - sxy * sxy;
        if (Mathf.Abs(normalDeterminant) < 0.000001f)
        {
            Debug.LogWarning(EyeNames[eyeIndex] + " calibration failed: outer samples do not span two axes.");
            return false;
        }

        float inverseDeterminant = 1f / normalDeterminant;
        float m00 = (txX * syy - txY * sxy) * inverseDeterminant;
        float m01 = (-txX * sxy + txY * sxx) * inverseDeterminant;
        float m10 = (tyX * syy - tyY * sxy) * inverseDeterminant;
        float m11 = (-tyX * sxy + tyY * sxx) * inverseDeterminant;

        calibratedRawForward[eyeIndex] = rawForward;
        calibratedRawUp[eyeIndex] = rawUp;
        calibratedRawRight[eyeIndex] = rawRight;
        calibrationM00[eyeIndex] = m00;
        calibrationM01[eyeIndex] = m01;
        calibrationM10[eyeIndex] = m10;
        calibrationM11[eyeIndex] = m11;
        hasCalibrations[eyeIndex] = true;

        if (verboseLogging)
        {
            float mappingDeterminant = m00 * m11 - m01 * m10;
            Debug.Log(
                EyeNames[eyeIndex] + " calibration matrix=["
                + m00.ToString("F4", CultureInfo.InvariantCulture) + ", "
                + m01.ToString("F4", CultureInfo.InvariantCulture) + "; "
                + m10.ToString("F4", CultureInfo.InvariantCulture) + ", "
                + m11.ToString("F4", CultureInfo.InvariantCulture) + "]"
                + ", determinant=" + mappingDeterminant.ToString("F4", CultureInfo.InvariantCulture)
                + (mappingDeterminant < 0f ? " (reflection corrected)" : "")
            );
        }

        return true;
    }

    private static bool TryProjectOntoTangentPlane(
        Vector3 direction,
        Vector3 forward,
        Vector3 right,
        Vector3 up,
        out Vector2 coordinates)
    {
        Vector3 normalizedDirection = direction.normalized;
        float forwardComponent = Vector3.Dot(normalizedDirection, forward);
        if (forwardComponent <= 0.0001f)
        {
            coordinates = Vector2.zero;
            return false;
        }

        coordinates = new Vector2(
            Vector3.Dot(normalizedDirection, right) / forwardComponent,
            Vector3.Dot(normalizedDirection, up) / forwardComponent
        );
        return true;
    }

    private Vector3 ApplyFivePointCalibration(int eyeIndex, Vector3 rawDirection)
    {
        if (!TryProjectOntoTangentPlane(
            rawDirection,
            calibratedRawForward[eyeIndex],
            calibratedRawRight[eyeIndex],
            calibratedRawUp[eyeIndex],
            out Vector2 measured))
        {
            return Vector3.forward;
        }

        float correctedX =
            calibrationM00[eyeIndex] * measured.x
            + calibrationM01[eyeIndex] * measured.y;
        float correctedY =
            calibrationM10[eyeIndex] * measured.x
            + calibrationM11[eyeIndex] * measured.y;

        return new Vector3(correctedX, correctedY, 1f).normalized;
    }

    private void ShowCalibrationTarget(int sampleIndex)
    {
        Vector2 offset = GetCalibrationTargetOffset(sampleIndex);
        Vector3 localPosition = new Vector3(offset.x, offset.y, sphereDistance);
        ShowCalibrationSphere(localPosition, GetCalibrationSampleName(sampleIndex) + "CalibrationSphere");

        if (verboseLogging)
            Debug.Log(GetCalibrationSampleName(sampleIndex) + " target shown. Look at it and press C.");
    }

    private Vector2 GetCalibrationTargetOffset(int sampleIndex)
    {
        switch (sampleIndex)
        {
            case UpSample:
                return new Vector2(0f, calibrationTargetOffset);
            case RightSample:
                return new Vector2(calibrationTargetOffset, 0f);
            case DownSample:
                return new Vector2(0f, -calibrationTargetOffset);
            case LeftSample:
                return new Vector2(-calibrationTargetOffset, 0f);
            default:
                return Vector2.zero;
        }
    }

    private static string GetCalibrationSampleName(int sampleIndex)
    {
        switch (sampleIndex)
        {
            case UpSample:
                return "Up";
            case RightSample:
                return "Right";
            case DownSample:
                return "Down";
            case LeftSample:
                return "Left";
            default:
                return "Center";
        }
    }

    private void ShowCalibrationSphere(Vector3 localPosition, string sphereName)
    {
        ClearCalibrationSphere();

        calibrationSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        calibrationSphere.name = sphereName;
        calibrationSphere.transform.SetParent(transform, false);

        float diameter = sphereRadius * 2.0f;
        calibrationSphere.transform.localScale = new Vector3(diameter, diameter, diameter);
        calibrationSphere.transform.localPosition = localPosition;
        calibrationSphere.transform.localRotation = Quaternion.identity;

        Renderer renderer = calibrationSphere.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.color = Color.red;
        }
    }

    private void ClearCalibrationSphere()
    {
        if (calibrationSphere != null)
        {
            Destroy(calibrationSphere);
            calibrationSphere = null;
        }
    }

    private void CreatePermanentGazeSphereCopies()
    {
        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            if (gazeSpheres[eyeIndex] == null)
                continue;

            GameObject permanentSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            permanentSphere.name = "Python" + EyeNames[eyeIndex] + "GazeSphere_Permanent";
            permanentSphere.transform.position = gazeSpheres[eyeIndex].transform.position;
            permanentSphere.transform.rotation = gazeSpheres[eyeIndex].transform.rotation;
            permanentSphere.transform.localScale = gazeSpheres[eyeIndex].transform.lossyScale;
            permanentSphere.transform.SetParent(null, true);

            Renderer renderer = permanentSphere.GetComponent<Renderer>();
            if (renderer != null)
            {
                Color eyeColor = eyeIndex == LeftEye ? Color.cyan : Color.magenta;
                eyeColor.a = 1.0f;
                SetMaterialTransparency(renderer.material, eyeColor, false);
            }
        }
    }

    private void CreateGazeSphere()
    {
        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            gazeSpheres[eyeIndex] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            gazeSpheres[eyeIndex].name = "Python" + EyeNames[eyeIndex] + "GazeSphere";

            gazeSpheres[eyeIndex].transform.SetParent(transform, false);

            float diameter = sphereRadius * 2.0f;
            gazeSpheres[eyeIndex].transform.localScale = new Vector3(diameter, diameter, diameter);
            gazeSpheres[eyeIndex].transform.localPosition =
                GetEyeGazeOriginLocal(eyeIndex) + Vector3.forward * sphereDistance;
            gazeSpheres[eyeIndex].transform.localRotation = Quaternion.identity;

            Renderer renderer = gazeSpheres[eyeIndex].GetComponent<Renderer>();
            if (renderer != null)
                renderer.material.color = eyeIndex == LeftEye
                    ? Color.cyan
                    : eyeIndex == RightEye
                        ? Color.magenta
                        : Color.white;
        }

        combinedGazeSphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        combinedGazeSphere.name = "PythonCombinedGazeSphere";
        combinedGazeSphere.transform.SetParent(transform, false);

        float combinedDiameter = sphereRadius * 2.0f;
        combinedGazeSphere.transform.localScale =
            new Vector3(combinedDiameter, combinedDiameter, combinedDiameter);
        combinedGazeSphere.transform.localPosition = Vector3.forward * sphereDistance;
        combinedGazeSphere.transform.localRotation = Quaternion.identity;

        Renderer combinedRenderer = combinedGazeSphere.GetComponent<Renderer>();
        if (combinedRenderer != null)
            combinedRenderer.material.color = Color.white;

        combinedGazeSphere.SetActive(false);
    }

    private void SetPostCalibrationVisualization(bool isComplete)
    {
        calibrationComplete = isComplete;

        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            if (gazeSpheres[eyeIndex] == null)
                continue;

            Renderer renderer = gazeSpheres[eyeIndex].GetComponent<Renderer>();
            if (renderer == null)
                continue;

            Color eyeColor = eyeIndex == LeftEye ? Color.cyan : Color.magenta;
            eyeColor.a = isComplete ? 0.2f : 1.0f;
            SetMaterialTransparency(renderer.material, eyeColor, isComplete);
        }

        if (combinedGazeSphere != null)
            combinedGazeSphere.SetActive(isComplete);
    }

    private static void SetMaterialTransparency(Material material, Color color, bool transparent)
    {
        material.color = color;

        if (!material.HasProperty("_Mode"))
            return;

        material.SetFloat("_Mode", transparent ? 3.0f : 0.0f);
        material.SetInt("_SrcBlend", transparent
            ? (int)UnityEngine.Rendering.BlendMode.SrcAlpha
            : (int)UnityEngine.Rendering.BlendMode.One);
        material.SetInt("_DstBlend", transparent
            ? (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha
            : (int)UnityEngine.Rendering.BlendMode.Zero);
        material.SetInt("_ZWrite", transparent ? 0 : 1);

        if (transparent)
        {
            material.DisableKeyword("_ALPHATEST_ON");
            material.EnableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }
        else
        {
            material.DisableKeyword("_ALPHATEST_ON");
            material.DisableKeyword("_ALPHABLEND_ON");
            material.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            material.renderQueue = -1;
        }
    }

    private void ReadAndApplyGaze()
    {
        string rawText = ReadTextFileSafe(GazeFilePath);
        if (string.IsNullOrWhiteSpace(rawText))
            return;

        if (!TryParseTwelveFloats(rawText, out float[] values))
            return;

        for (int eyeIndex = 0; eyeIndex < EyeCount; eyeIndex++)
        {
            int offset = eyeIndex * 6;
            pythonOrigins[eyeIndex] = new Vector3(values[offset], values[offset + 1], values[offset + 2]);
            pythonDirections[eyeIndex] = new Vector3(values[offset + 3], values[offset + 4], values[offset + 5]);

            Vector3 unityLocalDirection = PythonToUnityDirection(pythonDirections[eyeIndex]);

            if (unityLocalDirection.sqrMagnitude < 0.000001f)
            {
                hasValidRawDirections[eyeIndex] = false;
                continue;
            }

            unityLocalDirection.Normalize();

            latestRawUnityLocalDirections[eyeIndex] = unityLocalDirection;
            hasValidRawDirections[eyeIndex] = true;

            Vector3 correctedDirection = hasCalibrations[eyeIndex]
                ? ApplyFivePointCalibration(eyeIndex, unityLocalDirection)
                : unityLocalDirection;

            if (gazeSpheres[eyeIndex] != null)
                gazeSpheres[eyeIndex].transform.localPosition =
                    GetEyeGazeOriginLocal(eyeIndex) + correctedDirection * sphereDistance;
        }

        if (calibrationComplete
            && combinedGazeSphere != null
            && gazeSpheres[LeftEye] != null
            && gazeSpheres[RightEye] != null)
        {
            combinedGazeSphere.transform.localPosition =
                (gazeSpheres[LeftEye].transform.localPosition
                + gazeSpheres[RightEye].transform.localPosition) * 0.5f;
        }
    }

    private Vector3 PythonToUnityDirection(Vector3 pythonVec)
    {
        return new Vector3(
            pythonVec.x,
            pythonVec.y,
            pythonVec.z
        );
    }

    private string ReadTextFileSafe(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                Debug.LogWarning("Gaze file not found: " + path);
                return null;
            }

            using (FileStream fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (StreamReader reader = new StreamReader(fs))
            {
                return reader.ReadToEnd();
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning("Failed to read gaze file: " + e.Message);
            return null;
        }
    }

    private bool TryParseTwelveFloats(string text, out float[] values)
    {
        values = null;

        string[] tokens = text.Split(
            new char[] { ',', ' ', '\t', '\r', '\n', ';' },
            StringSplitOptions.RemoveEmptyEntries
        );

        int expectedValueCount = EyeCount * 6;

        if (tokens.Length < expectedValueCount)
            return false;

        values = new float[expectedValueCount];

        for (int i = 0; i < expectedValueCount; i++)
        {
            if (!float.TryParse(tokens[i], NumberStyles.Float, CultureInfo.InvariantCulture, out values[i]))
            {
                values = null;
                return false;
            }
        }

        return true;
    }
}

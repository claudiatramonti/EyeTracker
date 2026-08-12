import unittest
import os
import tempfile

import cv2
import numpy as np

import aruco_screen


def rotation_error_degrees(actual, expected):
    delta = actual @ expected.T
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class ArucoPoseMathTests(unittest.TestCase):
    def setUp(self):
        self.width = 1920
        self.height = 1080
        self.marker_size = 216
        self.margin = 12
        self.camera_matrix = aruco_screen.default_camera_matrix(640, 480)
        self.objects = aruco_screen.screen_marker_code_corners(
            self.width,
            self.height,
            self.marker_size,
            self.margin,
        )

    def make_pose(self, rvec, distance):
        rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
        center = np.array(
            [self.width * 0.5, self.height * 0.5, 0.0],
            dtype=np.float64,
        )
        translation = np.array([0.0, 0.0, distance]) - rotation @ center
        return rotation, translation

    def projected_detections(self, rotation, translation):
        rvec, _ = cv2.Rodrigues(rotation)
        corners = []
        ids = []
        for marker_id in (0, 1, 2, 3):
            projected, _ = cv2.projectPoints(
                self.objects[marker_id],
                rvec,
                translation,
                self.camera_matrix,
                np.zeros((5, 1), dtype=np.float64),
            )
            corners.append(projected.reshape(1, 4, 2).astype(np.float32))
            ids.append([marker_id])
        return corners, np.asarray(ids, dtype=np.int32)

    def test_solve_pnp_recovers_screen_pose(self):
        expected_rotation, expected_translation = self.make_pose(
            [0.08, -0.16, 0.03],
            2200.0,
        )
        corners, ids = self.projected_detections(
            expected_rotation,
            expected_translation,
        )

        pose = aruco_screen.estimate_screen_pose(
            corners,
            ids,
            self.objects,
            self.camera_matrix,
        )

        self.assertIsNotNone(pose)
        self.assertLess(
            rotation_error_degrees(pose["rotation"], expected_rotation),
            0.02,
        )
        self.assertLess(
            np.linalg.norm(pose["translation"] - expected_translation),
            0.5,
        )
        self.assertLess(pose["reprojection_error"], 0.05)

    def test_homography_uses_true_marker_code_coordinates(self):
        rotation, translation = self.make_pose([0.06, -0.14, 0.02], 2200.0)
        corners, ids = self.projected_detections(rotation, translation)
        homography = aruco_screen.compute_homography(
            corners,
            ids,
            self.width,
            self.height,
            object_corners_by_id=self.objects,
        )
        target_screen = np.array([[[1380.0, 310.0]]], dtype=np.float32)
        target_3d = np.array([[1380.0, 310.0, 0.0]], dtype=np.float32)
        rvec, _ = cv2.Rodrigues(rotation)
        target_image, _ = cv2.projectPoints(
            target_3d,
            rvec,
            translation,
            self.camera_matrix,
            np.zeros((5, 1), dtype=np.float64),
        )

        recovered = cv2.perspectiveTransform(
            target_image.reshape(1, 1, 2).astype(np.float32),
            homography,
        )

        self.assertLess(np.linalg.norm(recovered - target_screen), 0.1)

    def test_camera_calibration_scales_intrinsics(self):
        camera_matrix = np.array(
            [[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        distortion = np.array([[0.1, -0.03, 0.001, 0.002, 0.0]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "calibration.npz")
            np.savez(
                path,
                camera_matrix=camera_matrix,
                distortion_coefficients=distortion,
                image_width=np.int32(640),
                image_height=np.int32(480),
                rms_error=np.float64(0.42),
            )
            loaded = aruco_screen.load_camera_calibration(
                path,
                target_width=1280,
                target_height=960,
            )

        self.assertIsNotNone(loaded)
        np.testing.assert_allclose(
            loaded["camera_matrix"],
            np.array(
                [[1400.0, 0.0, 640.0], [0.0, 1420.0, 480.0], [0.0, 0.0, 1.0]]
            ),
        )
        np.testing.assert_allclose(loaded["distortion_coefficients"], distortion)
        self.assertAlmostEqual(loaded["rms_error"], 0.42)

    def test_calibrated_gaze_ray_survives_head_pose_change(self):
        tracker = aruco_screen.ArucoScreenTracker(
            self.width,
            self.height,
            marker_size=self.marker_size,
            marker_margin=self.margin,
        )
        calibration_rotation, calibration_translation = self.make_pose(
            [0.04, -0.10, 0.02],
            2200.0,
        )
        tracker.pose_rotation = calibration_rotation
        tracker.pose_translation = calibration_translation
        tracker.pose_ready = True

        gaze_to_camera, _ = cv2.Rodrigues(
            np.array([0.12, 0.06, -0.08], dtype=np.float64)
        )
        eye_origin_camera = np.array([35.0, -20.0, -120.0], dtype=np.float64)
        targets = {
            "center": (self.width * 0.5, self.height * 0.5),
            "top": (self.width * 0.5, 100.0),
            "bottom": (self.width * 0.5, self.height - 100.0),
            "left": (100.0, self.height * 0.5),
            "right": (self.width - 100.0, self.height * 0.5),
        }

        mapper = aruco_screen.GazePoseMapper()
        for label, point in targets.items():
            target_camera = tracker.screen_point_position_in_camera(*point)
            direction_camera = target_camera - eye_origin_camera
            direction_camera /= np.linalg.norm(direction_camera)
            direction_gaze = gaze_to_camera.T @ direction_camera
            saved, _ = mapper.add_calibration_sample(
                label,
                direction_gaze,
                point,
                tracker,
            )
            self.assertTrue(saved)

        self.assertTrue(mapper.calibrated)
        self.assertLess(
            rotation_error_degrees(mapper.rotation_gaze_to_camera, gaze_to_camera),
            0.02,
        )
        self.assertLess(
            np.linalg.norm(mapper.origin_gaze_in_camera - eye_origin_camera),
            0.5,
        )

        runtime_rotation, runtime_translation = self.make_pose(
            [-0.09, 0.18, -0.04],
            2050.0,
        )
        tracker.pose_rotation = runtime_rotation
        tracker.pose_translation = runtime_translation
        target = np.array([1420.0, 340.0, 0.0], dtype=np.float64)
        direction_camera = (
            runtime_rotation @ target
            + runtime_translation
            - eye_origin_camera
        )
        direction_camera /= np.linalg.norm(direction_camera)
        direction_gaze = gaze_to_camera.T @ direction_camera

        projected = mapper.project(direction_gaze, tracker)

        self.assertIsNotNone(projected)
        self.assertLessEqual(abs(projected[0] - target[0]), 1.0)
        self.assertLessEqual(abs(projected[1] - target[1]), 1.0)

    def test_per_eye_projection_averages_pixels(self):
        tracker = aruco_screen.ArucoScreenTracker(
            self.width,
            self.height,
            marker_size=self.marker_size,
            marker_margin=self.margin,
        )
        calibration_rotation, calibration_translation = self.make_pose(
            [0.03, -0.08, 0.01],
            2200.0,
        )
        tracker.pose_rotation = calibration_rotation
        tracker.pose_translation = calibration_translation
        tracker.pose_ready = True

        left_to_camera, _ = cv2.Rodrigues(
            np.array([0.10, 0.05, -0.06], dtype=np.float64)
        )
        right_to_camera, _ = cv2.Rodrigues(
            np.array([0.11, 0.04, -0.07], dtype=np.float64)
        )
        left_origin = np.array([40.0, -15.0, -110.0], dtype=np.float64)
        right_origin = np.array([-40.0, -15.0, -110.0], dtype=np.float64)
        targets = {
            "center": (self.width * 0.5, self.height * 0.5),
            "top": (self.width * 0.5, 100.0),
            "bottom": (self.width * 0.5, self.height - 100.0),
            "left": (100.0, self.height * 0.5),
            "right": (self.width - 100.0, self.height * 0.5),
        }

        mapper = aruco_screen.GazePoseMapper(eye_ids=("left", "right"))
        for label, point in targets.items():
            target_camera = tracker.screen_point_position_in_camera(*point)
            left_dir_cam = target_camera - left_origin
            left_dir_cam /= np.linalg.norm(left_dir_cam)
            right_dir_cam = target_camera - right_origin
            right_dir_cam /= np.linalg.norm(right_dir_cam)
            saved, _ = mapper.add_calibration_sample(
                label,
                {
                    "left": left_to_camera.T @ left_dir_cam,
                    "right": right_to_camera.T @ right_dir_cam,
                },
                point,
                tracker,
            )
            self.assertTrue(saved)

        self.assertTrue(mapper.calibrated)
        self.assertEqual(set(mapper.calibrated_eyes()), {"left", "right"})

        runtime_rotation, runtime_translation = self.make_pose(
            [-0.07, 0.15, -0.03],
            2100.0,
        )
        tracker.pose_rotation = runtime_rotation
        tracker.pose_translation = runtime_translation
        target = np.array([1500.0, 420.0, 0.0], dtype=np.float64)
        left_dir_cam = (
            runtime_rotation @ target + runtime_translation - left_origin
        )
        left_dir_cam /= np.linalg.norm(left_dir_cam)
        right_dir_cam = (
            runtime_rotation @ target + runtime_translation - right_origin
        )
        right_dir_cam /= np.linalg.norm(right_dir_cam)

        projected = mapper.project(
            {
                "left": left_to_camera.T @ left_dir_cam,
                "right": right_to_camera.T @ right_dir_cam,
            },
            tracker,
        )
        self.assertIsNotNone(projected)
        self.assertLessEqual(abs(projected[0] - target[0]), 1.0)
        self.assertLessEqual(abs(projected[1] - target[1]), 1.0)

    def test_average_unit_directions(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        averaged = aruco_screen.average_unit_directions([a, b])
        self.assertIsNotNone(averaged)
        np.testing.assert_allclose(
            averaged,
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
